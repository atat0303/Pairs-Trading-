📈 OU-Based Pairs Trading Backtester

This repository implements a Pairs Trading strategy leveraging a statistically consistent Ornstein-Uhlenbeck (OU) mean-reversion model. It includes spread construction, stationarity filtering, mean-reversion modeling, cost-aware threshold generation, and walk-forward backtesting with standard industry performance metrics.

🔍 Strategy Overview

Pairs trading profits from temporary divergence between two historically related securities. This framework:

Uses OLS on log-prices to estimate hedge ratios

log(Y) = α + β log(X)


Validates spread mean-reversion via the ADF stationarity test

Fits an AR(1) process to the spread and maps it to an OU process to extract:

Equilibrium mean (μ)

Mean-reversion speed (θ)

Half-life of reversion

Equilibrium volatility (σ)

Pairs with weak correlation, high ADF p-values, or slow mean-reversion are filtered out.

⚙️ Trade Signal Generation

Z-scores of the spread guide entry and exits:

Condition	Action
z > z_in	Short spread (short Y, long X)
z < −z_in	Long spread
	
	

Entry thresholds adapt to both:

Statistical significance (based on OU parameters)

Transaction costs (round-trip cost modeled using β)

Supported z-score modes:

ou_fixed — OU params fixed from formation window

rolling_plain — Rolling mean/std

rolling_ou — Rolling AR(1) → OU re-fit

ewma_ou — EWMA with decay tied to OU half-life

🧪 Walk-Forward Backtesting

Trades are evaluated on future data after the formation period. For each pair, the system computes:

CAGR (annualized return)

Annualized volatility

Sharpe ratio

Max drawdown

Trade-by-trade returns with cost deductions

🧠 Why OU?

OU modeling allows:

Statistically grounded view of mispricing

Speed-aware signal timing (via θ & half-life)

Realistic volatility scaling for risk control

This yields more robust trades than simple correlation-based pairs.

▶️ Example Usage
pairs = select_pairs(px, formation=252)
results = backtest(px, pairs, formation=252, trading=126)
print(results.sort_values("sharpe", ascending=False))


Prices can be sourced from yfinance or any custom DataFrame of prices.

📂 Project Structure
│
├── pairs_trading_ou.py   ← All model & backtesting logic
└── README.md

✅ Requirements
numpy
pandas
statsmodels
yfinance  # optional for data loading


Install via:

pip install -r requirements.txt

✅ Status

✅ Complete implementation of OU-based signal generation
✅ Cost-aware trade filters
✅ Robust walk-forward testing
📌 Future improvements: portfolio allocation, visualization tools, slippage modeling

📬 Contact

Contributions and feedback are welcome — feel free to open an issue or PR!
