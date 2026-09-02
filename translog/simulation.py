import numpy as np
import pandas as pd

def mu_translog(beta, x1, x2):
    """Two-input translog frontier."""
    beta0, beta1, beta2, beta11, beta22, beta12 = beta
    return (
        beta0
        + beta1 * x1
        + beta2 * x2
        + 0.5 * beta11 * x1**2
        + 0.5 * beta22 * x2**2
        + beta12 * x1 * x2
    )

def simulate_translog_2in(
    N: int,
    beta,
    sigma_u: float,
    sigma_v: float,
    seed: int = 0,
    x_low: float = 0.05,
    x_high: float = 10.0,
):
    """Simulate two-input translog SFM data."""
    rng = np.random.default_rng(seed)

    X1 = rng.uniform(x_low, x_high, size=N)
    X2 = rng.uniform(x_low, x_high, size=N)
    logx1 = np.log(X1)
    logx2 = np.log(X2)

    u = np.abs(rng.normal(0.0, sigma_u, size=N))
    v = rng.normal(0.0, sigma_v, size=N)

    logy = mu_translog(beta, logx1, logx2) + v - u
    Y = np.exp(logy)
    TE = np.exp(-u)

    return pd.DataFrame({
        "X1": X1, "X2": X2,
        "logx1": logx1, "logx2": logx2,
        "logy": logy, "Y": Y,
        "u": u, "v": v, "TE": TE
    })
