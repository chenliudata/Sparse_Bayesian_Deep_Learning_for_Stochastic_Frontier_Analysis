import numpy as np
import pandas as pd


def validate_ces_params(params):
    """Validate sufficient CES conditions for monotonicity and concavity."""
    log_A = float(params["log_A"])
    delta = float(params["delta"])
    rho = float(params["rho"])
    nu = float(params.get("nu", 1.0))

    if not np.isfinite(log_A):
        raise ValueError("CES log_A must be finite.")
    if not 0.0 < delta < 1.0:
        raise ValueError("CES delta must be between 0 and 1.")
    if not 0.0 < nu <= 1.0:
        raise ValueError("CES nu must be in (0, 1] for monotonicity and concavity.")
    if rho > 1.0:
        raise ValueError("CES rho must be <= 1 for concavity.")

    return log_A, delta, rho, nu


def mu_ces(params, logx1, logx2):
    """Two-input CES frontier on the log-output scale.

    With positive inputs, 0 < delta < 1, 0 < nu <= 1, and rho <= 1, the
    production function is monotone nondecreasing and concave.
    """
    log_A, delta, rho, nu = validate_ces_params(params)

    logx1 = np.asarray(logx1, dtype=float)
    logx2 = np.asarray(logx2, dtype=float)

    if abs(rho) < 1e-8:
        return log_A + nu * (delta * logx1 + (1.0 - delta) * logx2)

    a = np.log(delta) + rho * logx1
    b = np.log1p(-delta) + rho * logx2
    log_inner = np.logaddexp(a, b)
    return log_A + (nu / rho) * log_inner


def simulate_ces_2in(
    N: int,
    params,
    sigma_u: float,
    sigma_v: float,
    seed: int = 0,
    x_low: float = 0.05,
    x_high: float = 10.0,
):
    """Simulate two-input CES SFM data."""
    rng = np.random.default_rng(seed)

    X1 = rng.uniform(x_low, x_high, size=N)
    X2 = rng.uniform(x_low, x_high, size=N)
    logx1 = np.log(X1)
    logx2 = np.log(X2)

    u = np.abs(rng.normal(0.0, sigma_u, size=N))
    v = rng.normal(0.0, sigma_v, size=N)

    logy = mu_ces(params, logx1, logx2) + v - u
    Y = np.exp(logy)
    TE = np.exp(-u)

    return pd.DataFrame({
        "X1": X1, "X2": X2,
        "logx1": logx1, "logx2": logx2,
        "logy": logy, "Y": Y,
        "u": u, "v": v, "TE": TE
    })

mu_translog = mu_ces
simulate_translog_2in = simulate_ces_2in
