import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import numpy as np
from config import SP500_URL, market_ticker, horizon, time, threshold, method

def get_sp500_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(SP500_URL, headers = headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    tickers = [row.find("td").get_text(strip=True) for row in rows if row.find("td")]
    return tickers

def get_data(tickers: list):
    tickers = [asset.replace(".", "-") for asset in tickers]
    n = len(tickers)
    step = 50
    chunks = []
    for i in range(0, n, step):
        data: pd.DataFrame = yf.download(tickers = tickers[i:i + step], period = horizon, interval = time, auto_adjust = False)
        if "Adj Close" in data:
            group: pd.DataFrame = data["Adj Close"]
        else:
            group: pd.DataFrame = data["Close"]
        group.index = pd.to_datetime(group.index)
        if group.isna().sum().sum() > 0:
            mask = group.isna().mean(axis = 0) < (1 - threshold)
            group = group.loc[:, mask]
            group.ffill(inplace=True)
            group.bfill(inplace=True)
        chunks.append(group)
    prices: pd.DataFrame = pd.concat(chunks, axis = 1) 
    return prices

def get_capitalization(prices: pd.DataFrame):
    market_caps = []
    for asset in prices.columns:
        try:
            t = yf.Ticker(asset)
            cap = t.fast_info['market_cap']
        except Exception:
            cap = None
        market_caps.append(cap)
    market_caps = pd.Series(market_caps, index = prices.columns)
    market_caps.rename("Capitalization", inplace = True)
    market_caps.dropna(inplace = True)
    return market_caps

def get_market_returns():
    data: pd.DataFrame = yf.download(tickers = market_ticker, period = horizon, interval = time, auto_adjust = False)
    if "Adj Close" in data:
        price: pd.Series = data["Adj Close"].squeeze()
    else:
        price: pd.Series = data["Close"].squeeze()
    price.index = pd.to_datetime(price.index)
    if price.isna().sum().sum() > 0:
        price.ffill(inplace=True)
        price.bfill(inplace=True)
    if method == "log":
        returns: pd.Series = np.log(price).diff().dropna()
    elif method == "simple":
        returns: pd.Series = price.pct_change().dropna()
    else:
        raise ValueError("method must be log or simple")
    returns.rename("SP500", inplace = True)
    return returns

def compute_returns(prices: pd.DataFrame):
    if method == "log":
        returns: pd.DataFrame = np.log(prices).diff().dropna()
    elif method == "simple":
        returns: pd.DataFrame = prices.pct_change().dropna()
    else:
        raise ValueError("method must be log or simple")
    return returns

def excess_returns(returns: pd.DataFrame, ff: pd.DataFrame):
    excess: pd.DataFrame = returns.sub(ff["RF"], axis = 0)
    return excess