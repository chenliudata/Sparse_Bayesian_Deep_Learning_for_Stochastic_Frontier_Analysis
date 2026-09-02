import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def mu_cobb(beta, x1, x2):
    """Cobb-Douglas frontier on the log scale."""
    beta0, beta1, beta2 = beta
    return beta0 + beta1 * x1 + beta2 * x2


def log_density(coefs, y, x1, x2):
    """Log-density for normal-half-normal SFM errors."""
    beta = coefs[:3]
    sigma2u = coefs[3]
    sigma2v = coefs[4]

    lam = np.sqrt(sigma2u / sigma2v)
    sigma = np.sqrt(sigma2u + sigma2v)

    mu = mu_cobb(beta, x1, x2)
    eps = y - mu

    den = (2.0 / sigma) * norm.pdf(eps / sigma) * norm.cdf(-lam * eps / sigma)
    den = np.clip(den, 1e-300, np.inf)
    return np.log(den)


def loglikelihood(coefs, y, x1, x2):
    """Negative log-likelihood."""
    return -np.sum(log_density(coefs, y, x1, x2))


def estimate_mle(y, x1, x2):
    """Estimate two-input Cobb-Douglas SFM by MLE."""

    X = np.column_stack([
        np.ones(len(y)),
        x1,
        x2,
    ])

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    resid_var = max(np.var(resid_ols, ddof=X.shape[1]), 1e-6)

    theta0 = np.array([
        beta_ols[0], beta_ols[1], beta_ols[2],
        resid_var / 2, resid_var / 2,
    ], dtype=float)

    bounds = [(None, None)] * 3 + [(1e-6, np.inf), (1e-6, np.inf)]

    mle_results = minimize(
        fun=loglikelihood,
        x0=theta0,
        method="L-BFGS-B",
        tol=1e-6,
        options={"ftol": 1e-6, "maxiter": 1000, "maxfun": 6000},
        args=(y, x1, x2),
        bounds=bounds,
    )

    theta = mle_results.x
    neg_loglikelihood = mle_results.fun

    base_delta = 1e-6
    grad = np.zeros((len(y), len(theta)))
    bounds_arr = bounds  
    for i in range(len(theta)):
        delta_i = base_delta * max(abs(theta[i]), 1.0)
        lo, hi = bounds_arr[i]
        plus_step = delta_i if (hi is None or theta[i] + delta_i <= hi) else 0.0
        minus_step = delta_i if (lo is None or theta[i] - delta_i >= lo) else 0.0
        if plus_step > 0.0 and minus_step > 0.0:
            theta_plus = np.copy(theta); theta_plus[i] += plus_step
            theta_minus = np.copy(theta); theta_minus[i] -= minus_step
            grad[:, i] = (
                log_density(theta_plus, y, x1, x2)
                - log_density(theta_minus, y, x1, x2)
            ) / (plus_step + minus_step)
        else:
            step = plus_step if plus_step > 0.0 else -minus_step
            theta_step = np.copy(theta); theta_step[i] += step
            grad[:, i] = (
                log_density(theta_step, y, x1, x2)
                - log_density(theta, y, x1, x2)
            ) / step

    opg = grad.T @ grad
    try:
        cov = np.linalg.inv(opg)
        diag = np.diag(cov)
        if np.any(diag < 0):
            raise np.linalg.LinAlgError("Negative diagonal in OPG inverse")
        ster = np.sqrt(diag)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(opg)
        diag = np.diag(cov)
        ster = np.where(diag >= 0, np.sqrt(np.maximum(diag, 0.0)), np.nan)

    return theta, ster, neg_loglikelihood, mle_results


def mle_summary(theta, sterr):
    """Compute summary statistics."""
    zscores = theta / sterr
    pvalues = 2 * (1 - norm.cdf(np.abs(zscores)))
    lower_95 = theta - 1.96 * sterr
    upper_95 = theta + 1.96 * sterr
    return zscores, pvalues, lower_95, upper_95
