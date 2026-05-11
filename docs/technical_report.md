# Factor Investing & Systematic Portfolio Construction

## Technical Report

---

## 1. Objective

The objective of this project is to evaluate whether rolling factor exposures estimated from the Fama-French four-factor model can serve as reliable signals for systematic portfolio construction.

The analysis addresses three questions:

- Do factor loadings estimated on a rolling window provide stable and informative signals?
- Can a constrained mean-variance optimiser translate these signals into a disciplined portfolio?
- Is the resulting performance attributable to systematic factor exposures or genuine alpha?

---

## 2. Data

**Equity universe**

- Source: S&P 500 current composition, scraped from Wikipedia
- Download: daily adjusted closing prices via yfinance (10-year horizon)
- Filtering: stocks with more than 10% missing observations are excluded; remaining gaps filled via forward and backward fill
- Final universe: 475 stocks

Note: the universe is based on current composition, introducing survivorship bias. Firms delisted or bankrupt during the sample period are not represented.

**Factor data**

- Source: Kenneth French Data Library
- Factors: MKT-RF, SMB, HML (from F-F Research Data Factors daily), WML (from F-F Momentum Factor daily)
- All factors expressed as daily excess returns (divided by 100)
- Risk-free rate: RF series from the same source, used to compute equity excess returns

**Benchmark**

- S&P 500 index (^GSPC) downloaded via yfinance
- Used for performance comparison and tracking error constraint

---

## 3. Methodology

### 3.1 Excess returns

Daily log returns are computed for all assets. The risk-free rate is subtracted to obtain excess returns used in factor regression:

$$r_{i,t}^e = r_{i,t} - r_{f,t}$$

### 3.2 Rolling factor loadings

For each asset i, factor loadings are estimated via rolling OLS regression on a 252-day window:

$$r_{i,t}^e = \alpha_i + \beta_i^{MKT} \cdot MKT_t + \beta_i^{SMB} \cdot SMB_t + \beta_i^{HML} \cdot HML_t + \beta_i^{WML} \cdot WML_t + \varepsilon_{i,t}$$

Implemented via `statsmodels.regression.rolling.RollingOLS` for computational efficiency.

### 3.3 Signal construction

The expected return signal for asset i at time t is:

$$\mu_{i,t} = \hat{\beta}_{i,t} \cdot \bar{f}_t$$

where $\hat{\beta}_{i,t}$ is the vector of rolling loadings and $\bar{f}_t$ is the rolling mean of factor returns over the past 252 days.

This assumes that historical factor premia are informative about future expected returns — a standard assumption in factor investing.

### 3.4 Portfolio optimisation

At each monthly rebalancing date, the portfolio solves:

$$\max_w \quad \mu^T w - \gamma \cdot w^T \Sigma w$$

subject to:

$$\sum_i w_i = 1$$
$$w_i \geq 0 \quad \forall i$$
$$w_i \leq 0.05 \quad \forall i$$
$$(w - w_b)^T \Sigma (w - w_b) \leq TE^2$$
$$\frac{1}{2} \sum_i |w_i - w_{i,t-1}| \leq TO$$

where:
- $\gamma = 1$ is the risk aversion parameter
- $\Sigma$ is the Ledoit-Wolf shrinkage covariance matrix estimated on the rolling window
- $w_b$ is the market-cap benchmark weight vector
- $TE = 4\%$ annualised tracking error limit
- $TO = 20\%$ monthly turnover limit

Implemented via cvxpy with the default solver. If the optimiser fails or returns a non-optimal solution, weights fall back to market-cap weights.

Note: the tracking error constraint proved binding in most periods given the concentration of market-cap weights in large-cap stocks. This limits the portfolio's ability to deviate significantly from the benchmark.

### 3.5 Transaction costs

Net returns are computed as:

$$r_t^{net} = r_t - c \cdot \text{turnover}_t$$

with:

$$\text{turnover}_t = \frac{1}{2} \sum_i |w_{i,t} - w_{i,t-1}|$$

and cost rate $c = 0.1\%$.

---

## 4. Validation

Sanity checks applied after each backtest run:

- Weights sum to 1 (tolerance 1e-6)
- No negative weights (tolerance 1e-8)
- Maximum weight ≤ 5% (tolerance 1e-6)
- Monthly turnover ≤ 20% (first period excluded — initialisation artefact)

---

## 5. Results

### Performance metrics

| Metric | Value |
|---|---|
| Annualised net return | 15.89% |
| Annualised volatility | 23.00% |
| Sharpe ratio | 0.670 |
| Maximum drawdown | -31.38% |
| Average Information Ratio | 0.731 |
| Average monthly turnover | 0.88% |

### Performance attribution

OLS regression of portfolio excess returns on four factors:

| Factor | Loading | Std. Error | t-stat | p-value |
|---|---|---|---|---|
| MKT-RF | 1.116 | 0.006 | 180.4 | < 0.001 |
| SMB | -0.053 | 0.011 | -4.7 | < 0.001 |
| HML | -0.110 | 0.008 | -13.8 | < 0.001 |
| Momentum | 0.123 | 0.007 | 18.2 | < 0.001 |
| Alpha | -6.8e-05 | 7.5e-05 | -0.91 | 0.361 |

R² = 0.948, Adj. R² = 0.947

---

## 6. Economic Interpretation

**Market beta > 1**: the portfolio carries slightly more market risk than the benchmark. This is consistent with the signal tilting toward assets with higher factor exposures, which tend to be more volatile.

**Negative SMB**: mild large-cap bias, expected given the S&P 500 universe which excludes small-cap stocks by construction.

**Negative HML**: growth tilt. The optimiser systematically underweights value stocks in favour of growth, which has been rewarded over the 2015–2026 sample period dominated by technology outperformance.

**Positive momentum**: the most significant active tilt. The portfolio systematically overweights recent winners, consistent with the momentum signal driving portfolio construction.

**Alpha not significant**: the portfolio does not generate returns beyond what is explained by factor exposures. This is the expected result for a strategy built on publicly available data applied to a highly efficient large-cap universe. The absence of alpha is an honest result, not a failure — it confirms that returns are fair compensation for systematic risk.

**R² = 0.948**: nearly all portfolio variance is explained by the four factors. The strategy is essentially a systematic, rules-based vehicle for factor exposure, not an alpha-generating strategy.

---

## 7. Limitations

- **Survivorship bias**: current S&P 500 composition used as universe
- **Look-ahead bias in benchmark**: market-cap weights based on current capitalisation
- **Static risk aversion**: γ = 1 not calibrated; sensitivity analysis not performed
- **Tight tracking error constraint**: limits active tilts; a less constrained version would show stronger factor exposures
- **Linear transaction costs**: no bid-ask spread, market impact or regime-dependent cost modelling
- **Static signal construction**: rolling mean of factor premia assumes stationarity of risk premia over time

---

## 8. Conclusion

The factor portfolio demonstrates that rolling OLS loadings can serve as effective signals for systematic portfolio construction. The strategy delivers a positive Sharpe ratio (0.670) with low average turnover (0.88% monthly) and no statistically significant alpha.

The result is consistent with two well-established findings in empirical asset pricing:

1. Factor exposures explain the vast majority of equity portfolio returns
2. Large-cap markets are sufficiently efficient that publicly available signals do not generate persistent alpha after transaction costs

The connection to the other two projects in this series is direct: as shown in the portfolio optimisation project, estimation error dominates model complexity. Here, Ledoit-Wolf shrinkage and bounded weights serve the same purpose — stabilising the optimisation under noisy inputs. And as shown in the VaR backtesting project, model complexity does not guarantee better outcomes; robust and simple frameworks consistently perform well.
