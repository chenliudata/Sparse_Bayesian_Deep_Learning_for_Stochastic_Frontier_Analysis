
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from activations import MyCReLU
from config_real import (
    N_HIDDEN1, N_HIDDEN2, ALPHA_U, ALPHA_V,
    SCALE_HALFNORMAL, SCALE_NORMAL, N_INPUTS,
)


class BNN_SFM_CRELU_HS_NIn(PyroModule):
    def __init__(
        self,
        n_inputs: int = N_INPUTS,
        n_hidden1: int = N_HIDDEN1,
        n_hidden2: int = N_HIDDEN2,
        a: float = 0.8,
        b0: float = 0.5,
        bg: float = 0.5,
        b_kappa: float = 0.5,
        beta_scale: float = SCALE_HALFNORMAL,
        bias_scale: float = SCALE_NORMAL,
        alpha_u: float = ALPHA_U,
        alpha_v: float = ALPHA_V,
        beta_u: float = None,
        beta_v: float = None,
        mle_layers: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError("beta_u and beta_v must be calibrated first.")
        self.n_inputs = n_inputs
        self.h1 = n_hidden1
        self.h2 = n_hidden2
        self.act = MyCReLU(a=a)
        self.b0 = b0
        self.bg = bg
        self.b_kappa = b_kappa
        self.beta_scale = beta_scale
        self.bias_scale = bias_scale
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v

        self._using_mle = mle_layers is not None
        if self._using_mle:
            if len(mle_layers) != 3:
                raise ValueError("mle_layers must contain exactly three (W, b) pairs.")
            (W1, b1), (W2, b2), (W3, b3) = mle_layers
            expected = [
                ("W1", W1, (n_hidden1, n_inputs)),
                ("b1", b1, (n_hidden1,)),
                ("W2", W2, (n_hidden2, n_hidden1)),
                ("b2", b2, (n_hidden2,)),
                ("W3", W3, (1, n_hidden2)),
                ("b3", b3, (1,)),
            ]
            for name, tensor, shape in expected:
                if tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"mle_layers shape mismatch: {name} has shape "
                        f"{tuple(tensor.shape)}, expected {shape}."
                    )
            self._mle_layers = mle_layers

    @staticmethod
    def _decompose_hs_layer(W, b, tau_floor: float = 1e-3):
        W = W.detach().clone()
        b = b.detach().clone()
        row_scale = W.abs().amax(dim=1)
        row_scale = torch.clamp(row_scale, min=tau_floor)
        nu = torch.tensor(1.0, dtype=W.dtype, device=W.device)
        tau = row_scale
        beta_w = (W / (tau[:, None] * nu)).clamp(min=0.0)
        beta_b = b / (tau * nu)
        return nu, tau, beta_w, beta_b

    def mle_initial_params(self) -> dict:
        if not self._using_mle:
            return {}

        (W1, b1), (W2, b2), (W3, b3) = self._mle_layers

        nu1, tau1, beta_w1, beta_b1 = self._decompose_hs_layer(W1, b1)
        nu2, tau2, beta_w2, beta_b2 = self._decompose_hs_layer(W2, b2)

        kappa_val = float(W3.abs().amax().clamp(min=1e-3))
        kappa = torch.tensor(kappa_val, dtype=W3.dtype, device=W3.device)
        z_out = (W3 / kappa).clamp(min=0.0)

        tau_aux_init1 = torch.full_like(tau1, (1.0 / (self.b0 ** 2)) / 1.5)
        tau_aux_init2 = torch.full_like(tau2, (1.0 / (self.b0 ** 2)) / 1.5)
        kappa_aux_init = torch.tensor((1.0 / (self.b_kappa ** 2)) / 1.5, dtype=W3.dtype)

        return {
            # Layer 1
            "layer1_tau_aux": tau_aux_init1,
            "layer1_tau_sq":  tau1 ** 2,
            "layer1_z_w":     beta_w1,
            "layer1_beta_b":  b1.detach().clone(),
            # Layer 2
            "layer2_tau_aux": tau_aux_init2,
            "layer2_tau_sq":  tau2 ** 2,
            "layer2_z_w":     beta_w2,
            "layer2_beta_b":  b2.detach().clone(),
            # Output layer
            "out_kappa_aux": kappa_aux_init,
            "out_kappa_sq":  kappa ** 2,
            "out_z_w":       z_out,
            "out_bias":      b3.detach().clone().reshape(()),
        }

    def _hs_layer(self, name, x, out_dim):
        _, in_dim = x.shape
        b0_sq_inv = torch.full((out_dim,), 1.0 / (self.b0 ** 2))
        tau_aux = pyro.sample(
            f"{name}_tau_aux",
            dist.InverseGamma(torch.full((out_dim,), 0.5), b0_sq_inv).to_event(1),
        )
        tau_sq = pyro.sample(
            f"{name}_tau_sq",
            dist.InverseGamma(torch.full((out_dim,), 0.5), 1.0 / tau_aux).to_event(1),
        )
        tau = pyro.deterministic(f"{name}_tau", torch.sqrt(tau_sq))

        z_w = pyro.sample(
            f"{name}_z_w",
            dist.HalfNormal(torch.tensor(1.0)).expand([out_dim, in_dim]).to_event(2),
        )

        beta_b = pyro.sample(
            f"{name}_beta_b",
            dist.Normal(0.0, self.bias_scale).expand([out_dim]).to_event(1),
        )

        w = pyro.deterministic(f"{name}_w", tau[:, None] * z_w)
        b = pyro.deterministic(f"{name}_b", beta_b)
        return x @ w.T + b

    def _gaussian_output(self, name, h):
        _, in_dim = h.shape
        kappa_aux = pyro.sample(
            f"{name}_kappa_aux",
            dist.InverseGamma(torch.tensor(0.5), torch.tensor(1.0 / (self.b_kappa ** 2))),
        )
        kappa_sq = pyro.sample(
            f"{name}_kappa_sq",
            dist.InverseGamma(torch.tensor(0.5), 1.0 / kappa_aux),
        )
        kappa = pyro.deterministic(f"{name}_kappa", torch.sqrt(kappa_sq))

        z_out = pyro.sample(
            f"{name}_z_w",
            dist.HalfNormal(torch.tensor(1.0)).expand([1, in_dim]).to_event(2),
        )

        bias_out = pyro.sample(f"{name}_bias", dist.Normal(0.0, self.bias_scale))

        w_out = pyro.deterministic(f"{name}_w", kappa * z_out)
        return (h @ w_out.T).squeeze(-1) + bias_out

    def forward(self, x, y=None):
        x = x.reshape(-1, self.n_inputs)
        n = x.shape[0]

        h1 = self.act(self._hs_layer("layer1", x, out_dim=self.h1))
        h2 = self.act(self._hs_layer("layer2", h1, out_dim=self.h2))
        mu = self._gaussian_output("out", h2)
        pyro.deterministic("mu", mu)

        sigma2_u = pyro.sample("sigma2_u_hs", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v = pyro.sample("sigma2_v_hs", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u = torch.sqrt(sigma2_u)
        sigma_v = torch.sqrt(sigma2_v)

        with pyro.plate("data_hs", n):
            u = pyro.sample("u_hs", dist.HalfNormal(sigma_u))
            pyro.sample("y_obs_hs", dist.Normal(mu - u, sigma_v), obs=y)

        return mu
