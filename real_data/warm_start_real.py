
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch


def build_blm_init_real(
    beta_log: np.ndarray,
    sigma2u_log: float,
    sigma2v_log: float,
    stats: Dict[str, float],
    n: int,
) -> dict:
   
    beta_log = np.asarray(beta_log, dtype=float).ravel()
    names: List[str] = stats["input_names"]
    d = len(names)
    if beta_log.size != d + 1:
        raise ValueError(
            f"beta_log has length {beta_log.size}, expected {d + 1} for "
            f"{d} inputs."
        )

    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])

    beta_std = np.zeros(d + 1, dtype=float)
    sum_bj_xmean = 0.0
    for j, name in enumerate(names, start=1):
        m = float(stats[f"x_mean_{name}"])
        s = float(stats[f"x_std_{name}"])
        beta_std[j] = beta_log[j] * (s / y_std)
        sum_bj_xmean += beta_log[j] * m
    beta_std[0] = (beta_log[0] - y_mean + sum_bj_xmean) / y_std

    sigma2u_std = sigma2u_log / (y_std ** 2)
    sigma2v_std = sigma2v_log / (y_std ** 2)
    u_init = max(math.sqrt(max(sigma2u_std, 1e-8)) * math.sqrt(2.0 / math.pi), 1e-3)

    init: dict = {
        "beta0_blm": torch.tensor(float(beta_std[0]), dtype=torch.float32),
        "sigma2_u_blm": torch.tensor(max(sigma2u_std, 1e-4), dtype=torch.float32),
        "sigma2_v_blm": torch.tensor(max(sigma2v_std, 1e-4), dtype=torch.float32),
        "u_blm": torch.full((n,), u_init, dtype=torch.float32),
    }
    for j in range(1, d + 1):
        init[f"beta{j}_blm"] = torch.tensor(float(beta_std[j]), dtype=torch.float32)
    return init


def build_fc_nn_init(
    model,
    sigma2_u_site: str,
    sigma2_v_site: str,
    u_site: str,
    sigma2_u_init: float,
    sigma2_v_init: float,
    n: int,
) -> dict:
    
    init = model.mle_initial_params() if hasattr(model, "mle_initial_params") else {}
    init[sigma2_u_site] = torch.tensor(max(float(sigma2_u_init), 1e-4), dtype=torch.float32)
    init[sigma2_v_site] = torch.tensor(max(float(sigma2_v_init), 1e-4), dtype=torch.float32)
    u_init = max(math.sqrt(max(float(sigma2_u_init), 1e-8)) * math.sqrt(2.0 / math.pi), 1e-3)
    init[u_site] = torch.full((n,), u_init, dtype=torch.float32)
    return init


def build_hs_init(
    model,
    sigma2_u_site: str,
    sigma2_v_site: str,
    u_site: str,
    sigma2_u_init: float,
    sigma2_v_init: float,
    n: int,
) -> dict:
   
    return build_fc_nn_init(
        model=model,
        sigma2_u_site=sigma2_u_site,
        sigma2_v_site=sigma2_v_site,
        u_site=u_site,
        sigma2_u_init=sigma2_u_init,
        sigma2_v_init=sigma2_v_init,
        n=n,
    )
