import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src.data import get_sp500_tickers, get_data, get_capitalization, get_market_returns, compute_returns, excess_returns

BASE_DIR = Path(__file__).resolve().parent.parent
price_path = BASE_DIR / "results" / "prices"
data_path = BASE_DIR / "results" / "fama_french"
factors_path = BASE_DIR / "results" / "data"
price_path.mkdir(parents = True, exist_ok = True)
factors_path.mkdir(parents = True, exist_ok = True)

tickers = get_sp500_tickers()
prices = get_data(tickers)
prices.to_parquet(price_path / f"prices.parquet")
log_returns = compute_returns(prices)
log_returns.to_parquet(price_path / f"log_returns.parquet")
capitalization = get_capitalization(prices)
capitalization.to_frame().to_parquet(price_path / f"capitalization.parquet")
market_portoflio = get_market_returns()
market_portoflio.to_frame().to_parquet(price_path / f"sp500_returns.parquet")

ff = pd.read_csv(data_path / "F-F_Research_Data_Factors_daily.csv", skiprows = 4)
ff.rename(columns = {"Unnamed: 0": "Date"}, inplace = True)
ff = ff[pd.to_numeric(ff["Date"], errors = "coerce").notna()]
ff["Date"] = pd.to_datetime(ff["Date"], format = "%Y%m%d")
ff.set_index("Date", inplace = True)
ff = ff.astype(float) / 100
common_index = log_returns.index.intersection(ff.index)
log_returns = log_returns.loc[common_index]
ff = ff.loc[common_index]

momentum = pd.read_csv(data_path / "F-F_Momentum_Factor_daily.csv", skiprows = 13)
momentum.rename(columns = {"Unnamed: 0": "Date"}, inplace = True)
momentum = momentum[pd.to_numeric(momentum["Date"], errors = "coerce").notna()]
momentum["Date"] = pd.to_datetime(momentum["Date"], format = "%Y%m%d")
momentum.set_index("Date", inplace = True)
momentum = momentum.astype(float) / 100
momentum = momentum.loc[common_index]

excess = excess_returns(log_returns, ff)
excess.to_parquet(price_path / f"excess_returns.parquet")

factors = pd.concat([ff, momentum], axis = 1)
factors.drop("RF", axis = 1, inplace = True)
factors.to_parquet(factors_path / f"factors.parquet")