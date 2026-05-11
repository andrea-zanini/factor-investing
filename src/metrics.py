import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from config import window, rf, cost_rate

def metrics(return_portfolio: pd.Series, weights_portfolio: pd.DataFrame, market_portfolio_returns: pd.Series):
    turnover: pd.Series = 0.5 * weights_portfolio.diff().abs().sum(axis = 1)
    net_return: pd.Series = return_portfolio - (turnover * cost_rate)
    sharpe: pd.Series = (net_return - rf).rolling(window).mean() / net_return.rolling(window).std() * np.sqrt(252)
    wealth = np.exp(net_return.cumsum())
    drawdown: pd.Series = (wealth - wealth.cummax()) / wealth.cummax()
    informatio_ratio: pd.Series = (return_portfolio - market_portfolio_returns).rolling(window).mean() / (return_portfolio - market_portfolio_returns).rolling(window).std() * np.sqrt(252)
    return net_return, turnover, sharpe, drawdown, informatio_ratio