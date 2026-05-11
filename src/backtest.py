import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from src.portfolio import portfolio
from config import window

def rebalancing(signal: pd.DataFrame, market_weights: np.ndarray, returns: pd.DataFrame):
    T = len(returns)
    returns_np = returns.values
    signal_np = signal.values
    asset = returns.columns
    index = returns.index

    portfolio_weights = []
    portfolio_returns = []
    prev_weights = market_weights.copy()
    rebalance_date = []

    for t in range(window, T):
        train = returns_np[t-window:t]
        test = returns_np[t]
        current_signal = signal_np[t-1]
        current_date = index[t]
        if current_date.month != index[t-1].month or t == window:
            w_new = portfolio(current_signal, train, market_weights, prev_weights)
            prev_weights = w_new
        portfolio_weights.append(prev_weights)
        portfolio_returns.append(test @ prev_weights)
        rebalance_date.append(current_date)
    portfolio_weights = pd.DataFrame(portfolio_weights, columns = asset, index = rebalance_date)
    portfolio_returns = pd.Series(portfolio_returns, index = rebalance_date)
    portfolio_returns.rename("Returns", inplace = True)
    return portfolio_weights, portfolio_returns