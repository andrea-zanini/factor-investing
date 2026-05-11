import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.covariance import ledoit_wolf
import cvxpy as cp
from config import max_weight, tracking_error_limit, turnover_limit

def portfolio(signal: np.ndarray, returns: np.ndarray, market_portoflio_weights: np.ndarray, w_prev: np.ndarray):
    n = len(signal)
    w = cp.Variable(n)

    sigma, shrink = ledoit_wolf(returns)
    s = sigma.shape[0]
    sigma += 1e-8 * np.eye(s)
    gamma = 1
    mu = w @ signal
    risk = cp.quad_form(w, cp.psd_wrap(sigma))
    objective = cp.Maximize(mu - gamma * risk)
    
    gap = w - market_portoflio_weights
    tracking_error = cp.quad_form(gap, cp.psd_wrap(sigma))
    daily_te_limit = (tracking_error_limit / np.sqrt(252))**2
    constraints = [tracking_error <= daily_te_limit, cp.sum(w) == 1, w >= 0, 
                   w <= max_weight, 0.5 * cp.norm1(w - w_prev) <= turnover_limit]

    problem = cp.Problem(objective, constraints = constraints)
    problem.solve()
    if problem.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        return market_portoflio_weights
    weights = w.value
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return weights