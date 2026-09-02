from __future__ import annotations

from typing import Dict
import numpy as np
import torch


def build_std_stats(logx1_obs: np.ndarray, logx2_obs: np.ndarray, logy_obs: np.ndarray) -> Dict[str, float]:
    """Compute standardization statistics."""
    logx1_obs = np.asarray(logx1_obs, dtype=np.float64)
    logx2_obs = np.asarray(logx2_obs, dtype=np.float64)
    logy_obs = np.asarray(logy_obs, dtype=np.float64)
    x1_mean = float(logx1_obs.mean())
    x1_std = float(max(logx1_obs.std(ddof=0), 1e-8))
    x2_mean = float(logx2_obs.mean())
    x2_std = float(max(logx2_obs.std(ddof=0), 1e-8))
    y_mean = float(logy_obs.mean())
    y_std = float(max(logy_obs.std(ddof=0), 1e-8))
    return {"x1_mean": x1_mean, "x1_std": x1_std, "x2_mean": x2_mean, "x2_std": x2_std, "y_mean": y_mean, "y_std": y_std}


def standardize_inputs_2in(logx1_obs: np.ndarray, logx2_obs: np.ndarray, stats: Dict[str, float]) -> torch.Tensor:
    """Standardize two input variables."""
    z1 = (np.asarray(logx1_obs) - stats["x1_mean"]) / stats["x1_std"]
    z2 = (np.asarray(logx2_obs) - stats["x2_mean"]) / stats["x2_std"]
    return torch.tensor(np.column_stack([z1, z2]), dtype=torch.float32)


def standardize_output(logy_obs: np.ndarray, stats: Dict[str, float]) -> torch.Tensor:
    """Standardize output variable."""
    return torch.tensor((np.asarray(logy_obs) - stats["y_mean"]) / stats["y_std"], dtype=torch.float32)


def unscale_sigma2(sigma2_std: torch.Tensor, y_std: float) -> torch.Tensor:
    """Convert variance to logY scale."""
    return sigma2_std * (y_std ** 2)


def unscale_u(u_std: torch.Tensor, y_std: float) -> torch.Tensor:
    """Convert inefficiency to logY scale."""
    return u_std * y_std


def unscale_mu(mu_std: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    """Convert frontier mean to logY scale."""
    return mu_std * stats["y_std"] + stats["y_mean"]


def translog_log_to_std_betas(beta_log: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Convert translog betas to standardized scale."""
    b0, b1, b2, b11, b22, b12 = [float(v) for v in beta_log]
    m1, s1 = stats["x1_mean"], stats["x1_std"]
    m2, s2 = stats["x2_mean"], stats["x2_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    c0 = b0 + b1*m1 + b2*m2 + 0.5*b11*(m1**2) + 0.5*b22*(m2**2) + b12*m1*m2
    c1 = s1 * (b1 + b11*m1 + b12*m2)
    c2 = s2 * (b2 + b22*m2 + b12*m1)
    c11 = b11 * (s1**2)
    c22 = b22 * (s2**2)
    c12 = b12 * s1 * s2

    return np.array([(c0 - y_mean)/y_std, c1/y_std, c2/y_std, c11/y_std, c22/y_std, c12/y_std], dtype=float)


def unscale_blm_betas_translog(beta0_std, beta1_std, beta2_std, beta11_std, beta22_std, beta12_std, stats: Dict[str, float]):
    """Convert translog betas to logY scale."""
    y_mean, y_std = stats["y_mean"], stats["y_std"]
    m1, s1 = stats["x1_mean"], stats["x1_std"]
    m2, s2 = stats["x2_mean"], stats["x2_std"]

    c0 = beta0_std * y_std + y_mean
    c1 = beta1_std * y_std
    c2 = beta2_std * y_std
    c11 = beta11_std * y_std
    c22 = beta22_std * y_std
    c12 = beta12_std * y_std

    b11 = c11 / (s1 ** 2)
    b22 = c22 / (s2 ** 2)
    b12 = c12 / (s1 * s2)
    b1 = c1 / s1 - b11 * m1 - b12 * m2
    b2 = c2 / s2 - b22 * m2 - b12 * m1
    b0 = c0 - b1*m1 - b2*m2 - 0.5*b11*(m1**2) - 0.5*b22*(m2**2) - b12*m1*m2

    return b0, b1, b2, b11, b22, b12


def cobb_log_to_std_betas(beta_log: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Convert Cobb-Douglas betas to standardized scale."""
    beta_log = np.asarray(beta_log, dtype=float).ravel()
    if beta_log.size != 3:
        raise ValueError(f"cobb_log_to_std_betas expects 3 coefficients, got {beta_log.size}")
    b0, b1, b2 = [float(v) for v in beta_log]
    m1, s1 = stats["x1_mean"], stats["x1_std"]
    m2, s2 = stats["x2_mean"], stats["x2_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    c0 = (b0 + b1 * m1 + b2 * m2 - y_mean) / y_std
    c1 = b1 * s1 / y_std
    c2 = b2 * s2 / y_std
    return np.array([c0, c1, c2], dtype=float)


def unscale_blm_betas_cobb(beta0_std, beta1_std, beta2_std, stats: Dict[str, float]):
    """Convert Cobb-Douglas betas to logY scale."""
    y_mean, y_std = stats["y_mean"], stats["y_std"]
    m1, s1 = stats["x1_mean"], stats["x1_std"]
    m2, s2 = stats["x2_mean"], stats["x2_std"]

    b1 = beta1_std * y_std / s1
    b2 = beta2_std * y_std / s2
    b0 = beta0_std * y_std + y_mean - b1 * m1 - b2 * m2
    return b0, b1, b2
