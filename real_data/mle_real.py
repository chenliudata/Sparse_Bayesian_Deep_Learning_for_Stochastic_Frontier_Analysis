
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def log_density(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> np.ndarray:
    
    d = X.shape[1]
    beta0 = theta[0]
    beta = theta[1:1 + d]
    sigma2u = theta[1 + d]
    sigma2v = theta[2 + d]

    lam = np.sqrt(sigma2u / sigma2v)
    sigma = np.sqrt(sigma2u + sigma2v)
    eps = y - beta0 - X @ beta

    return (
        np.log(2.0)
        - np.log(sigma)
        + norm.logpdf(eps / sigma)
        + norm.logcdf(-lam * eps / sigma)
    )


def neg_loglikelihood(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    logDen = log_density(theta, y, X)
    if not np.all(np.isfinite(logDen)):
        return 1e10
    return -np.sum(logDen)


def make_theta0_ols(y: np.ndarray, X: np.ndarray, var_split: float = 0.5) -> np.ndarray:
  
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    Xd = np.column_stack([np.ones(n), X])
    beta_ols = np.linalg.lstsq(Xd, y, rcond=None)[0]
    resid = y - Xd @ beta_ols
    s2 = max(np.var(resid, ddof=Xd.shape[1]), 1e-6)

    sigma2u_0 = max(var_split * s2, 1e-6)
    sigma2v_0 = max((1.0 - var_split) * s2, 1e-6)
    return np.concatenate([beta_ols, [sigma2u_0, sigma2v_0]])


def estimate_mle(y: np.ndarray, X: np.ndarray, theta0: np.ndarray = None):
    
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if theta0 is None:
        theta0 = make_theta0_ols(y, X)

    bounds = (
        [(None, None)] * (d + 1)        
        + [(1e-6, None), (1e-6, None)]  
    )

    res = minimize(
        fun=neg_loglikelihood,
        x0=theta0,
        args=(y, X),
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-6, "maxiter": 1000, "maxfun": 6000},
    )
    theta = res.x
    neg_ll = res.fun

    base_delta = 1e-6
    grad = np.zeros((n, len(theta)))
    for i in range(len(theta)):
        delta_i = base_delta * max(abs(theta[i]), 1.0)
        lo, hi = bounds[i]
        plus = delta_i if (hi is None or theta[i] + delta_i <= hi) else 0.0
        minus = delta_i if (lo is None or theta[i] - delta_i >= lo) else 0.0
        if plus > 0.0 and minus > 0.0:
            tp = theta.copy(); tp[i] += plus
            tm = theta.copy(); tm[i] -= minus
            grad[:, i] = (
                log_density(tp, y, X) - log_density(tm, y, X)
            ) / (plus + minus)
        else:
            step = plus if plus > 0.0 else -minus
            ts = theta.copy(); ts[i] += step
            grad[:, i] = (log_density(ts, y, X) - log_density(theta, y, X)) / step

    OPG = grad.T @ grad
    try:
        cov = np.linalg.inv(OPG)
        diag = np.diag(cov)
        if np.any(diag < 0):
            raise np.linalg.LinAlgError("Negative diagonal in OPG inverse")
        sterr = np.sqrt(diag)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(OPG)
        diag = np.diag(cov)
        sterr = np.where(diag >= 0, np.sqrt(np.maximum(diag, 0.0)), np.nan)

    return theta, sterr, neg_ll, res


def mle_summary(theta: np.ndarray, sterr: np.ndarray):
    zscores = theta / sterr
    pvalues = 2 * (1 - norm.cdf(np.abs(zscores)))
    lower_95 = norm.ppf(0.025, loc=theta, scale=sterr)
    upper_95 = norm.ppf(0.975, loc=theta, scale=sterr)
    return zscores, pvalues, lower_95, upper_95


def mu_cobb_multivar(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cobb-Douglas mean: mu = beta[0] + X @ beta[1:]."""
    beta = np.asarray(beta, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    return beta[0] + X @ beta[1:]
