import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
plot_path = BASE_DIR / "results" / "plots"
plot_path.mkdir(parents=True, exist_ok=True)

STYLE = {
    "style": "ticks",
    "palette": "muted",
    "font": "serif",
    "rc": {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (12, 5),
        "axes.grid": True,
        "grid.alpha": 0.4
    }
}

def cumulative_returns(portfolio_returns: pd.Series, sp500_returns: pd.Series):
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.despine()

    cum_portfolio = np.exp(portfolio_returns.cumsum())
    cum_sp500 = np.exp(sp500_returns.cumsum())

    sns.lineplot(x=portfolio_returns.index, y=cum_portfolio,
                 color="#457b9d", linewidth=1.5, label="Factor Portfolio", ax=ax)
    sns.lineplot(x=sp500_returns.index, y=cum_sp500,
                 color="#2a2a2a", linewidth=1, linestyle="--", label="S&P 500", ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index (base = 1)")
    ax.set_title("Cumulative returns — Factor Portfolio vs S&P 500")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()

def drawdown_plot(drawdown: pd.Series, sp500_returns: pd.Series):
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.despine()

    sp500_wealth = np.exp(sp500_returns.cumsum())
    sp500_drawdown = (sp500_wealth - sp500_wealth.cummax()) / sp500_wealth.cummax()
    sp500_drawdown = sp500_drawdown.loc[drawdown.index]

    sns.lineplot(x=drawdown.index, y=drawdown,
                 color="#e63946", linewidth=1.5, label="Factor Portfolio", ax=ax)
    sns.lineplot(x=sp500_drawdown.index, y=sp500_drawdown,
                 color="#2a2a2a", linewidth=1, linestyle="--", label="S&P 500", ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown — Factor Portfolio vs S&P 500")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path / "drawdown.png", dpi=150, bbox_inches="tight")
    plt.close()


def rolling_sharpe_plot(sharpe: pd.Series):
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.despine()

    sharpe_clean = sharpe.dropna()

    sns.lineplot(x=sharpe_clean.index, y=sharpe_clean,
                 color="#2a9d8f", linewidth=1.5, label="Rolling Sharpe (252d)", ax=ax)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Rolling Sharpe ratio — Factor Portfolio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_path / "rolling_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close()


def factor_loadings_plot(loadings: pd.Series):
    sns.set_theme(**STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.despine()

    colors = ["#457b9d" if v >= 0 else "#e63946" for v in loadings.values]
    ax.bar(loadings.index, loadings.values, color=colors, width=0.5)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Factor")
    ax.set_ylabel("Loading")
    ax.set_title("Factor loadings — Factor Portfolio (OLS attribution)")
    plt.tight_layout()
    plt.savefig(plot_path / "factor_loadings.png", dpi=150, bbox_inches="tight")
    plt.close()