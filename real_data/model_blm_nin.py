
from __future__ import annotations

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from config_real import SCALE_NORMAL, ALPHA_U, ALPHA_V, N_INPUTS


class BLM_SFM_NIn(PyroModule):
    def __init__(self, n_inputs: int = N_INPUTS,
                 alpha_u: float = ALPHA_U, alpha_v: float = ALPHA_V,
                 beta_u: float = None, beta_v: float = None):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError("beta_u and beta_v must be calibrated first.")
        self.n_inputs = n_inputs
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v

        self.beta0_blm = PyroSample(dist.Normal(0, SCALE_NORMAL))
        for j in range(1, n_inputs + 1):
            setattr(self, f"beta{j}_blm", PyroSample(dist.Normal(0, SCALE_NORMAL)))

    def forward(self, x, y=None):
        x = x.reshape(-1, self.n_inputs)
        n = x.shape[0]

        mu_blm = self.beta0_blm
        for j in range(1, self.n_inputs + 1):
            mu_blm = mu_blm + getattr(self, f"beta{j}_blm") * x[:, j - 1]
        pyro.deterministic("mu", mu_blm)

        sigma2_u_blm = pyro.sample("sigma2_u_blm", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v_blm = pyro.sample("sigma2_v_blm", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u_blm = torch.sqrt(sigma2_u_blm)
        sigma_v_blm = torch.sqrt(sigma2_v_blm)

        with pyro.plate("data_blm", n):
            u_blm = pyro.sample("u_blm", dist.HalfNormal(sigma_u_blm))
            pyro.sample("y_obs_blm", dist.Normal(mu_blm - u_blm, sigma_v_blm), obs=y)

        return mu_blm
