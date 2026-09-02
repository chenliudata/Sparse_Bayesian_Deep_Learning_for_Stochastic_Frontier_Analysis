
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import torch


def build_std_stats(logX: np.ndarray, logY: np.ndarray,
                    input_names: Sequence[str]) -> Dict[str, float]:
    
    logX = np.asarray(logX, dtype=np.float64)
    logY = np.asarray(logY, dtype=np.float64)
    if logX.ndim != 2:
        raise ValueError(f"logX must be 2D, got shape {logX.shape}")
    n, d = logX.shape
    if len(input_names) != d:
        raise ValueError(
            f"input_names has length {len(input_names)} but logX has {d} columns."
        )

    stats: Dict[str, float] = {}
    for j, name in enumerate(input_names):
        col = logX[:, j]
        stats[f"x_mean_{name}"] = float(col.mean())
        stats[f"x_std_{name}"] = float(max(col.std(ddof=0), 1e-8))
    stats["y_mean"] = float(logY.mean())
    stats["y_std"] = float(max(logY.std(ddof=0), 1e-8))
    stats["input_names"] = list(input_names)
    return stats


def standardize_inputs(logX: np.ndarray, stats: Dict[str, float]) -> torch.Tensor:
   
    logX = np.asarray(logX, dtype=np.float64)
    names: List[str] = stats["input_names"]
    cols = []
    for j, name in enumerate(names):
        m = stats[f"x_mean_{name}"]
        s = stats[f"x_std_{name}"]
        cols.append((logX[:, j] - m) / s)
    Z = np.stack(cols, axis=1)
    return torch.tensor(Z, dtype=torch.float32)


def standardize_output(logY: np.ndarray, stats: Dict[str, float]) -> torch.Tensor:
   
    return torch.tensor(
        (np.asarray(logY) - stats["y_mean"]) / stats["y_std"],
        dtype=torch.float32,
    )


def unscale_sigma2(sigma2_std: torch.Tensor, y_std: float) -> torch.Tensor:
    return sigma2_std * (y_std ** 2)


def unscale_u(u_std: torch.Tensor, y_std: float) -> torch.Tensor:
    return u_std * y_std


def unscale_mu(mu_std: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    return mu_std * stats["y_std"] + stats["y_mean"]


def unscale_blm_betas(beta_std_dict: Dict[str, torch.Tensor],
                      stats: Dict[str, float]) -> Dict[str, torch.Tensor]:
    
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    names: List[str] = stats["input_names"]
    d = len(names)

    out: Dict[str, torch.Tensor] = {}
    sum_bj_xmean = None
    for j, name in enumerate(names, start=1):
        m = stats[f"x_mean_{name}"]
        s = stats[f"x_std_{name}"]
        b_log = beta_std_dict[f"beta{j}_blm"] * (y_std / s)
        out[f"beta{j}_blm"] = b_log
        contribution = b_log * m
        sum_bj_xmean = contribution if sum_bj_xmean is None else sum_bj_xmean + contribution

    out["beta0_blm"] = (
        beta_std_dict["beta0_blm"] * y_std + y_mean - sum_bj_xmean
    )
    return out
