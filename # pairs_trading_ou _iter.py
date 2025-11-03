# pairs_trading_ou.py
# ------------------------------------------------------------
# A complete, theoretically-consistent pairs trading backtester
# ------------------------------------------------------------

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ========== Helpers: hedge -> AR(1) -> OU -> thresholds ==========

def hedge_ols(logA: pd.Series, logB: pd.Series):
    """
    OLS regression in log space: logA = alpha + beta*logB + u
    Returns alpha, beta, fitted model
    """
    y = logA.values
    X = sm.add_constant(logB.values)
    res = sm.OLS(y, X).fit()
    alpha, beta = res.params
    return float(alpha), float(beta), res


def fit_ar1(S: pd.Series):
    """
    Fit AR(1): S_{t+1} = a + b S_t + eps
    Returns dict with a, b, var_eps, res; or None if too little data.
    """
    s = S.dropna()
    St  = s.shift(1).dropna()      # predictor S_t
    St1 = s.dropna()               # response  S_{t+1}
    St, St1 = St.align(St1, join="inner")
    if len(St) < 20:
        return None
    X = sm.add_constant(St.values)
    res = sm.OLS(St1.values, X).fit()
    a, b = res.params
    var_eps = res.mse_resid
    return dict(a=float(a), b=float(b), var_eps=float(var_eps), res=res)


def ou_from_ar1(a: float, b: float, var_eps: float):
    """
    Map AR(1) params to OU params (Δ=1 day).
    Returns dict with mu, sigma_stat, theta, half_life, b; or None if not mean-reverting.
    """
    if not (0 < b < 1):
        return None
    mu = a / (1 - b)
    sigma_stat = np.sqrt(var_eps / (1 - b**2))  # == sqrt(sigma^2 / (2*theta))
    theta = -np.log(b)
    half_life = np.log(2) / theta
    return dict(mu=float(mu), sigma_stat=float(sigma_stat),
                theta=float(theta), half_life=float(half_life), b=float(b))


def compute_thresholds(b: float, sigma_stat: float, beta: float,
                       z_out=0.25, k=0.5, lam=1.2, tc_per_leg=0.0001):
    """
    Entry threshold z_in must beat both:
      - statistical floor from one-day SNR >= k
      - economic floor from round-trip costs (with buffer λ)
    """
    # statistical floor (SNR)
    z_snr  = k * np.sqrt(1 - b*b) / (1 - b)
    # economic floor
    c_rt   = 2.0 * (1.0 + abs(beta)) * tc_per_leg
    z_cost = z_out + lam * (c_rt / max(1e-12, sigma_stat))
    # final entry
    z_in   = max(1.5, z_snr, z_cost)
    z_in   = min(z_in, 2.5)
    z_stop = max(3.5, 1.5*z_in)  # optional stop band
    return dict(z_in=float(z_in), z_out=float(z_out), z_stop=float(z_stop))


# ========== Rolling z variants ==========

def rolling_z_plain(spreads: pd.Series, W: int = 63) -> pd.Series:
    """Plain rolling mean/std z-score (fast, not OU)."""
    roll_mu = spreads.rolling(W, min_periods=max(20, W//3)).mean().shift(1)
    roll_sd = spreads.rolling(W, min_periods=max(20, W//3)).std(ddof=0).clip(lower=1e-12).shift(1)
    return (spreads - roll_mu) / roll_sd


def rolling_ou_z(spreads: pd.Series, W: int = 63) -> pd.Series:
    """
    Rolling OU-consistent z: re-fit AR(1) on the last W points at each t,
    map to OU to get mu_t and sigma_stat_t, then z_t = (S_t - mu_t)/sigma_stat_t.
    """
    z = pd.Series(index=spreads.index, dtype=float)
    s = spreads.dropna()
    for i in range(W-1, len(s)):
        window = s.iloc[i-W+1:i]
        St  = window.shift(1).dropna()
        St1 = window.dropna()
        St, St1 = St.align(St1, join="inner")
        if len(St) < max(20, W//3):
            continue
        X = sm.add_constant(St.values)
        res = sm.OLS(St1.values, X).fit()
        a, b = res.params
        if not (0 < b < 1):
            continue
        var_eps = res.mse_resid
        mu_t = a / (1 - b)
        sigma_stat_t = np.sqrt(var_eps / (1 - b*b))
        if not np.isfinite(sigma_stat_t) or sigma_stat_t <= 0:
            continue
        z.iloc[i] = (s.iloc[i] - mu_t) / sigma_stat_t
    return z.reindex(spreads.index)


def ewma_ou_z(spreads: pd.Series, half_life_days: float, min_warmup: int = 30) -> pd.Series:
    """
    OU-inspired EWMA z-score with decay tied to half-life: λ = 2^{-1/HL}.
    """
    lam = np.exp(-np.log(2) / max(1e-9, half_life_days))
    mu, var = np.nan, np.nan
    out = []
    for i, s_t in enumerate(spreads):
        if not np.isfinite(s_t):
            out.append(np.nan); continue
        if i == 0 or not np.isfinite(mu):
            mu = s_t; var = 0.0
        else:
            mu = lam*mu + (1-lam)*s_t
            var = lam*var + (1-lam)*(s_t - mu)**2
        sd = np.sqrt(max(var, 1e-12))
        out.append((s_t - mu)/sd if i >= min_warmup else np.nan)
    return pd.Series(out, index=spreads.index)


# ========== Build pair info on a formation window ==========

def build_pair_info(y_prices: pd.Series, x_prices: pd.Series,
                    use_ou_z: bool = True,
                    tc_per_leg: float = 0.0001,
                    z_out: float = 0.25,
                    k: float = 0.5,
                    lam: float = 1.2,
                    adf_alpha: float = 0.10,
                    max_half_life: float = 60.0):
    """
    Estimate α, β, check stationarity, fit AR(1), map to OU, and compute thresholds.
    Returns info dict or None if filters fail.
    """
    y_prices, x_prices = y_prices.dropna().align(x_prices.dropna(), join="inner")
    if len(y_prices) < 60:
        return None

    logA, logB = np.log(y_prices), np.log(x_prices)

    # Hedge ratio
    alpha, beta, _ = hedge_ols(logA, logB)

    # Spread on formation
    S = logA - (alpha + beta*logB)
    # Stationarity (ADF)
    try:
        pval = adfuller(S.dropna())[1]
    except Exception:
        pval = 1.0
    if pval >= adf_alpha:
        return None

    # Correlation (for info)
    corr = float(logA.diff().corr(logB.diff()))
    if corr < 0.6:
        return None
     #AR(1) -> OU
    ar = fit_ar1(S)
    if ar is None:
        return None
    ou = ou_from_ar1(ar["a"], ar["b"], ar["var_eps"])
    if ou is None:
        return None
    if ou["half_life"] > max_half_life:
        return None

    # Thresholds (OU & cost-aware)
    th = compute_thresholds(ou["b"], ou["sigma_stat"], beta,
                            z_out=z_out, k=k, lam=lam, tc_per_leg=tc_per_leg)

    info = {
        "alpha": alpha, "beta": beta,
        "correlation": corr, "pvalue": pval,
        "theta": ou["theta"], "half_life": ou["half_life"],
        "b": ou["b"], "z_in": th["z_in"], "z_out": th["z_out"],
        "z_stop": th["z_stop"],
        # Z-source (filled below if using OU z)
        "mean": None, "stddev": None
    }
    if use_ou_z:
        info["mean"] = ou["mu"]
        info["stddev"] = ou["sigma_stat"]
    return info


def select_pairs(px: pd.DataFrame,
                 tickers: list[str] | None = None,
                 single_date : int = 0,
                 formation: int = 252,
                 use_ou_z: bool = True,
                 tc_per_leg: float = 0.0001,
                 z_out: float = 0.25,
                 k: float = 0.5,
                 lam: float = 1.2,
                 adf_alpha: float = 0.10,
                 max_half_life: float = 60.0):
    """
    Return a list of (x, y, info) estimated on the formation slice.
    """
    if tickers is None:
        tickers = list(px.columns)
    form_px = px.iloc[single_date:formation].dropna(how='all')

    pairs = []
    for x, y in itertools.combinations(tickers, 2):
        if x not in form_px or y not in form_px:
            continue
        info = build_pair_info(
            y_prices=form_px[y], x_prices=form_px[x],
            use_ou_z=use_ou_z, tc_per_leg=tc_per_leg,
            z_out=z_out, k=k, lam=lam,
            adf_alpha=adf_alpha, max_half_life=max_half_life
        )
        if info is not None:
            pairs.append((x, y, info))
    return pairs


# ========== Backtest ==========

def backtest(px: pd.DataFrame,
             pairs: list[tuple],
             formation: int,
             trading: int,
             tc_per_leg: float = 0.0005,
             z_mode: str = "ou_fixed",
             # z_mode options: "ou_fixed", "rolling_plain", "rolling_ou", "ewma_ou"
             W: int = 63):
    """
    Walk-forward backtest on a single (formation, trading) block.
    - px: price DataFrame
    - pairs: [(x, y, info), ...] from select_pairs (estimated on the same formation slice)
    - z_mode:
        "ou_fixed"    -> z_t = (S_t - mu) / sigma_stat  (from formation)
        "rolling_plain" -> rolling mean/std on spreads with window W
        "rolling_ou"    -> rolling AR(1)->OU re-fit window W
        "ewma_ou"       -> EWMA decay matched to OU half-life (from formation)
    """
    ann_factor = np.sqrt(252.0)
    results = []

    # Define the trading slice
    trade_px = px.iloc[formation:formation+trading].dropna(how='all')

    for x, y, info in pairs:
        # Align trading series
        if (x not in trade_px.columns) or (y not in trade_px.columns):
            continue
        x_series = trade_px[x].dropna()
        y_series = trade_px[y].dropna()
        y_series, x_series = y_series.align(x_series, join='inner')
        if len(y_series) < 2:
            continue

        # Spread (consistent with formation)
        spreads = np.log(y_series) - (info["beta"]*np.log(x_series) + info["alpha"])

        # z-series according to mode
        if z_mode == "ou_fixed":
            mu = info.get("mean", None)
            sd = info.get("stddev", None)
            if mu is None or sd is None or not np.isfinite(sd) or sd <= 0:
                # cannot compute OU z for this pair
                continue
            z_series = (spreads - mu) / (sd + 1e-12)

        elif z_mode == "rolling_plain":
            z_series = rolling_z_plain(spreads, W=W)

        elif z_mode == "rolling_ou":
            z_series = rolling_ou_z(spreads, W=W)

        elif z_mode == "ewma_ou":
            hl = float(info.get("half_life", 10.0))
            z_series = ewma_ou_z(spreads, half_life_days=hl, min_warmup=max(30, int(hl)))

        else:
            raise ValueError("Unknown z_mode")

        # Precompute leg log returns (spread returns in return units)
        retA = np.log(y_series).diff()
        retB = np.log(x_series).diff()

        beta = float(info["beta"])
        rt_cost = 2.0 * (1.0 + abs(beta)) * tc_per_leg

        pos = 0  # +1 long spread, -1 short spread
        returns = np.zeros(len(spreads))

        for i in range(1, len(spreads)):
            z = z_series.iloc[i]
            prev_pos = pos

            # Entries
            if pos == 0 and np.isfinite(z):
                if z >= info["z_in"]:
                    pos = -1
                    returns[i] -= rt_cost / 2.0
                elif z <= -info["z_in"]:
                    pos = +1
                    returns[i] -= rt_cost / 2.0

            # PnL while in position
            if prev_pos != 0 and np.isfinite(retA.iloc[i]) and np.isfinite(retB.iloc[i]):
                spread_ret = retA.iloc[i] - beta * retB.iloc[i]
                returns[i] += prev_pos * spread_ret

            # Exit
            if pos != 0 and np.isfinite(z) and abs(z) <= info["z_out"]:
                pos = 0
                returns[i] -= rt_cost / 2.0

        # Performance metrics
        equity = (1.0 + returns).cumprod()
        n = len(returns)
        years = max(1e-9, n/252.0)
        cagr = equity[-1]**(1/years) - 1.0
        vol = returns.std(ddof=1) * ann_factor
        sharpe = (returns.mean() / (returns.std(ddof=1) + 1e-12)) * ann_factor
        equity = (1.0 + returns).cumprod()

# Max drawdown
        running_max = np.maximum.accumulate(equity, axis=0)
        drawdowns = equity / running_max - 1.0
        max_dd = drawdowns.min()
        results.append({
            "stock_y": y,
            "stock_x": x,
            "beta": info["beta"],
            "corr": info["correlation"],
            "pvalue": info["pvalue"],
            "z_in": info["z_in"],
            "z_out": info["z_out"],
            "z_mode": z_mode,
            "W": W if "rolling" in z_mode else None,
            "cagr": cagr,
            "vol": vol,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        })

    return pd.DataFrame(results)


# ========== Example usage (optional) ==========

if __name__ == "__main__":
    # Demo: download a few tickers, run one formation/trading block.
    # If you don't have internet / yfinance, skip this and pass your own px DataFrame.
    if True:

        import yfinance as yf
        tickers =  ["JPM", "GS", "KO", "PEP", "XOM", "CVX","AAPL","MSFT","GOOGL","IBM","AMD","NVDA"]
        
        #tickers = ["XOM","OXY"]
        px = yf.download(tickers, start="2015-01-01")["Close"].dropna(how="all")

        formation_len = 252  # 1Y formation
        trading_len   = 126  # 6M trading
        from datetime import date, timedelta

        def daterange(start_date: date, end_date: date):
            days = int((end_date - start_date).days)
            for n in range(days):
                yield start_date + timedelta(n)

        start_date = date(2015, 1, 1)
        end_date = date(2024, 6, 2)
        info = {"stock 1":[],"stock 2":[], "sharpe":[],"max_drawdown":[]}
        for single_date in range(0,2400,30):
            try:
                print(type(single_date))
                pairs_ou = select_pairs(px, single_date=single_date, formation=single_date+formation_len, use_ou_z=True,
                            tc_per_leg=0.0005, z_out=0.25, k=0.5, lam=1.2,
                            adf_alpha=0.10, max_half_life=60.0)
                res_roll = backtest(px, pairs_ou, formation=single_date+formation_len, trading=single_date+trading_len,
                            tc_per_leg=0.0005, z_mode="rolling_plain", W=65)
                res_roll.sort_values("sharpe", ascending=False)
                info["stock 1"].append(res_roll["stock_y"].iloc[1])
                info["stock 2"].append(res_roll["stock_x"].iloc[1])
                info["sharpe"].append(res_roll["sharpe"].iloc[1])
                info["max_drawdown"].append(res_roll["max_drawdown"].iloc[1])
            except Exception:
                continue

print (info)
    