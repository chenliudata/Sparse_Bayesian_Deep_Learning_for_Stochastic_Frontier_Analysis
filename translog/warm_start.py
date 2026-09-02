
from __future__ import annotations

import math
from typing import Dict
import numpy as np
import torch

from standardize import cobb_log_to_std_betas, translog_log_to_std_betas


def build_blm_init(beta_log, sigma2u_log: float, sigma2v_log: float, stats: Dict[str, float], n: int) -> dict:
    """Build Cobb-Douglas BLM initial values."""
    beta_log = np.asarray(beta_log, dtype=float).ravel()
    if beta_log.size != 3:
        raise ValueError(
            f"build_blm_init expects a length-3 beta vector, got {beta_log.size}. "
        )
    beta_std = cobb_log_to_std_betas(beta_log, stats)
    sigma2u_std = float(sigma2u_log) / (stats["y_std"] ** 2)
    sigma2v_std = float(sigma2v_log) / (stats["y_std"] ** 2)
    u_init = max(math.sqrt(max(sigma2u_std, 1e-8)) * math.sqrt(2.0 / math.pi), 1e-3)

    return {
        "beta0": torch.tensor(beta_std[0], dtype=torch.float32),
        "beta1": torch.tensor(beta_std[1], dtype=torch.float32),
        "beta2": torch.tensor(beta_std[2], dtype=torch.float32),
        "sigma2_u": torch.tensor(max(sigma2u_std, 1e-4), dtype=torch.float32),
        "sigma2_v": torch.tensor(max(sigma2v_std, 1e-4), dtype=torch.float32),
        "u": torch.full((n,), u_init, dtype=torch.float32),
    }


def build_blm_init_translog(beta_log, sigma2u_log: float, sigma2v_log: float, stats: Dict[str, float], n: int) -> dict:
    """Build translog BLM initial values."""
    beta_std = translog_log_to_std_betas(np.asarray(beta_log, dtype=float), stats)
    sigma2u_std = float(sigma2u_log) / (stats["y_std"] ** 2)
    sigma2v_std = float(sigma2v_log) / (stats["y_std"] ** 2)
    u_init = max(math.sqrt(max(sigma2u_std, 1e-8)) * math.sqrt(2.0 / math.pi), 1e-3)

    return {
        "beta0": torch.tensor(beta_std[0], dtype=torch.float32),
        "beta1": torch.tensor(beta_std[1], dtype=torch.float32),
        "beta2": torch.tensor(beta_std[2], dtype=torch.float32),
        "beta11": torch.tensor(beta_std[3], dtype=torch.float32),
        "beta22": torch.tensor(beta_std[4], dtype=torch.float32),
        "beta12": torch.tensor(beta_std[5], dtype=torch.float32),
        "sigma2_u": torch.tensor(max(sigma2u_std, 1e-4), dtype=torch.float32),
        "sigma2_v": torch.tensor(max(sigma2v_std, 1e-4), dtype=torch.float32),
        "u": torch.full((n,), u_init, dtype=torch.float32),
    }


def build_fc_nn_init(model, sigma2_u_init: float, sigma2_v_init: float, n: int) -> dict:
    init = model.mle_initial_params() if hasattr(model, "mle_initial_params") else {}
    init["sigma2_u"] = torch.tensor(max(float(sigma2_u_init), 1e-4), dtype=torch.float32)
    init["sigma2_v"] = torch.tensor(max(float(sigma2_v_init), 1e-4), dtype=torch.float32)
    u_init = max(math.sqrt(max(float(sigma2_u_init), 1e-8)) * math.sqrt(2.0 / math.pi), 1e-3)
    init["u"] = torch.full((n,), u_init, dtype=torch.float32)
    return init


def build_hs_init(model, sigma2_u_init: float, sigma2_v_init: float, n: int) -> dict:
    return build_fc_nn_init(model, sigma2_u_init, sigma2_v_init, n)
