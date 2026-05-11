import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.regression import compute_loadings, compute_premium, compute_signals, compute_market_portfolio_weights
from src.backtest import rebalancing
from src.metrics import metrics
from src.sanity_check import sum_weights, positivity_weights, max_weights, turnover_weights

BASE_DIR = Path(__file__).resolve().parent
prices_path = BASE_DIR / "results" / "prices"
data_path = BASE_DIR / "results" / "data"
portfolio_path = BASE_DIR / "results" / "portfolio"
metrics_path = BASE_DIR / "results" / "metrics"
portfolio_path.mkdir(parents = True, exist_ok = True)
metrics_path.mkdir(parents = True, exist_ok = True)

excess_returns = pd.read_parquet(prices_path / "excess_returns.parquet")
sp500 = pd.read_parquet(prices_path / "sp500_returns.parquet").squeeze()
log_returns = pd.read_parquet(prices_path / "log_returns.parquet")
factors = pd.read_parquet(data_path / "factors.parquet")
capitalization = pd.read_parquet(prices_path / "capitalization.parquet").squeeze()

betas = compute_loadings(excess_returns, factors)
for asset in excess_returns.columns:
    betas[asset].dropna(inplace = True)
factor_premia = compute_premium(factors)
factor_premia.dropna(inplace = True)
signal = compute_signals(excess_returns, factor_premia, betas)

log_returns = log_returns.loc[signal.index]
market_weights = compute_market_portfolio_weights(capitalization, log_returns.columns)

portfolio_weights, portfolio_returns = rebalancing(signal, market_weights, log_returns)
sp500 = sp500.loc[portfolio_returns.index]

sum_weights(portfolio_weights)
positivity_weights(portfolio_weights)
max_weights(portfolio_weights)
turnover_weights(portfolio_weights)

net_return, turnover, sharpe, drawdown, informatio_ratio = metrics(portfolio_returns, portfolio_weights, sp500)

net_return.to_frame().to_parquet(portfolio_path / "portfolio_returns.parquet")
portfolio_weights.to_parquet(portfolio_path / "portfolio_weights.parquet")
turnover.to_frame().to_parquet(metrics_path / "turnover.parquet")
sharpe.to_frame().to_parquet(metrics_path / "sharpe_ratio.parquet")
drawdown.to_frame().to_parquet(metrics_path / "drawdown.parquet")
informatio_ratio.to_frame().to_parquet(metrics_path / "information_ratio.parquet")