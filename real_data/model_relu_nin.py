
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from config_real import (
    N_HIDDEN1, N_HIDDEN2, ALPHA_U, ALPHA_V, SCALE_NORMAL, N_INPUTS,
)


class BNN_SFM_RELU_N_NIn(PyroModule):
    def __init__(
        self,
        n_inputs: int = N_INPUTS,
        n_hidden1: int = N_HIDDEN1,
        n_hidden2: int = N_HIDDEN2,
        prior_scale_weight: float = SCALE_NORMAL,
        prior_scale_bias: float = SCALE_NORMAL,
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
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v
        self.activation = nn.ReLU()
        self._using_mle = mle_layers is not None

        self.fc1 = PyroModule[nn.Linear](n_inputs, n_hidden1)
        self.fc2 = PyroModule[nn.Linear](n_hidden1, n_hidden2)
        self.fc3 = PyroModule[nn.Linear](n_hidden2, 1)

        self.fc1.weight = PyroSample(
            dist.Normal(0.0, prior_scale_weight).expand([n_hidden1, n_inputs]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0.0, prior_scale_bias).expand([n_hidden1]).to_event(1)
        )
        self.fc2.weight = PyroSample(
            dist.Normal(0.0, prior_scale_weight).expand([n_hidden2, n_hidden1]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0.0, prior_scale_bias).expand([n_hidden2]).to_event(1)
        )
        self.fc3.weight = PyroSample(
            dist.Normal(0.0, prior_scale_weight).expand([1, n_hidden2]).to_event(2)
        )
        self.fc3.bias = PyroSample(
            dist.Normal(0.0, prior_scale_bias).expand([1]).to_event(1)
        )

        if mle_layers is not None:
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

    def mle_initial_params(self) -> dict:
        if not self._using_mle:
            return {}
        (W1, b1), (W2, b2), (W3, b3) = self._mle_layers
        return {
            "fc1.weight": W1.detach().clone(),
            "fc1.bias": b1.detach().clone(),
            "fc2.weight": W2.detach().clone(),
            "fc2.bias": b2.detach().clone(),
            "fc3.weight": W3.detach().clone(),
            "fc3.bias": b3.detach().clone(),
        }

    def forward(self, x, y=None):
        x = x.reshape(-1, self.n_inputs)
        n = x.shape[0]

        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        mu = self.fc3(h2).squeeze(-1)
        pyro.deterministic("mu", mu)

        sigma2_u = pyro.sample("sigma2_u_bnn_relu", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v = pyro.sample("sigma2_v_bnn_relu", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u = torch.sqrt(sigma2_u)
        sigma_v = torch.sqrt(sigma2_v)

        with pyro.plate("data_bnn_relu", n):
            u = pyro.sample("u_bnn_relu", dist.HalfNormal(sigma_u))
            pyro.sample("y_obs_bnn_relu", dist.Normal(mu - u, sigma_v), obs=y)

        return mu
