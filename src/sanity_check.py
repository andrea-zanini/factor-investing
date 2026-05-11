import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np

def sum_weights(weights_portfolio: pd.DataFrame):
    assert (weights_portfolio.sum(axis = 1) - 1).abs().max() <= 1e-6

def positivity_weights(weights_portfolio: pd.DataFrame):
    assert (weights_portfolio >= -1e-8).all().all()

def max_weights(weights_portfolio: pd.DataFrame):
    assert (weights_portfolio.max().max() - 0.05) <= 1e-6

def turnover_weights(weights_portfolio: pd.DataFrame):
    weights_portfolio_t1 = weights_portfolio.shift(1).dropna()
    weights_portfolio = weights_portfolio.loc[weights_portfolio_t1.index]
    turnover = 0.5 * np.abs(weights_portfolio - weights_portfolio_t1).sum(axis = 1)
    assert (turnover <= 0.20 + 1e-6).all()