import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
from config import window

def compute_loadings(y: pd.DataFrame, x: pd.DataFrame):
    loadings = {}
    for asset in y.columns:
        rolling_beta = RollingOLS(y[asset], x, window = window).fit()
        loadings[asset] = rolling_beta.params
    return loadings

def compute_premium(x: pd.DataFrame):
    return x.rolling(window).mean()

def compute_signals(y: pd.DataFrame, premium: pd.DataFrame, loadings: dict):
    date = premium.index
    premium_np = premium.values
    signal = []
    for asset in y.columns:
        factors = loadings[asset].values
        asset_signal = (factors * premium_np).sum(axis = 1)
        signal.append(asset_signal)
    signal = pd.DataFrame(signal, index = y.columns, columns = date).T
    return signal

def compute_market_portfolio_weights(capitalization: pd.Series, assets: pd.Index):
    capitalization = capitalization.reindex(assets).fillna(0)
    total = capitalization.sum()
    return (capitalization / total).values