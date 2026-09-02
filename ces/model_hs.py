import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule

from activations import MyCReLU
from config_ces import (
    N_HIDDEN1,
    N_HIDDEN2,
    SCALE_HALFNORMAL,
    SCALE_NORMAL,
    ALPHA_U,
    ALPHA_V,
    BETA_U,
    BETA_V,
)


class BNN_SFM_CRELU_HS_Node_2In(PyroModule):
    """Two-input CReLU BNN-SFM with row-wise horseshoe shrinkage."""

    def __init__(
        self,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        a=0.8,
        b0=0.5,
        bg=0.5,
        b_kappa=0.5,
        weight_scale=SCALE_HALFNORMAL,
        bias_scale=SCALE_NORMAL,
        alpha_u=ALPHA_U,
        alpha_v=ALPHA_V,
        beta_u=BETA_U,
        beta_v=BETA_V,
        mle_layers=None,
    ):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError(
                "beta_u and beta_v must be calibrated via calibrate_ig_prior."
            )
        self.h1 = n_hidden1
        self.h2 = n_hidden2
        self.act = MyCReLU(a=a)
        self.b0 = b0

        self.bg = bg
        self.b_kappa = b_kappa

        self.weight_scale = weight_scale
        self.bias_scale = bias_scale
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v
        self._using_mle = mle_layers is not None
        if self._using_mle:
            self._mle_layers = mle_layers

    @staticmethod
    def _decompose_hs_layer(W, b, tau_floor: float = 1e-3):
        """Convert MLE weights to horseshoe initial values."""
        W = W.detach().clone()
        b = b.detach().clone()

        row_scale = W.abs().amax(dim=1)
        row_scale = torch.clamp(row_scale, min=tau_floor)

        nu = torch.tensor(1.0, dtype=W.dtype, device=W.device)
        tau = row_scale
        beta_w = (W / (tau[:, None] * nu)).clamp(min=0.0)
        beta_b = b / (tau * nu)
        return nu, tau, beta_w, beta_b

    def mle_initial_params(self):
        """Initial values from NN-MLE."""
        if not self._using_mle:
            return {}

        (W1, b1), (W2, b2), (W3, b3) = self._mle_layers

        nu1, tau1, z_w1, beta_b1 = self._decompose_hs_layer(W1, b1)
        nu2, tau2, z_w2, beta_b2 = self._decompose_hs_layer(W2, b2)

        kappa_val = float(W3.abs().amax().clamp(min=1e-3))
        kappa = torch.tensor(kappa_val, dtype=W3.dtype, device=W3.device)
        z_out = (W3 / kappa).clamp(min=0.0)

        tau_aux_init1 = torch.full_like(tau1, (1.0 / (self.b0**2)) / 1.5)
        tau_aux_init2 = torch.full_like(tau2, (1.0 / (self.b0**2)) / 1.5)
        kappa_aux_init = torch.tensor((1.0 / (self.b_kappa**2)) / 1.5, dtype=W3.dtype)

        return {
            "layer1_tau_aux": tau_aux_init1,
            "layer1_tau_sq": tau1**2,
            "layer1_z_w": z_w1,
            "layer1_beta_b": b1.clamp(min=1e-6),
            "layer2_tau_aux": tau_aux_init2,
            "layer2_tau_sq": tau2**2,
            "layer2_z_w": z_w2,
            "layer2_beta_b": b2.clamp(min=1e-6),
            "out_kappa_aux": kappa_aux_init,
            "out_kappa_sq": kappa**2,
            "out_z_w": z_out,
            "out_bias": b3.detach().clone().reshape(()),
        }

    def _hs_layer(self, name, x, out_dim):
        """Hidden layer with row-wise horseshoe weights."""
        _, in_dim = x.shape

        b0_sq_inv = torch.full((out_dim,), 1.0 / (self.b0**2))
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
            dist.HalfNormal(self.bias_scale).expand([out_dim]).to_event(1),
        )

        w = pyro.deterministic(f"{name}_w", tau[:, None] * z_w)
        b = pyro.deterministic(f"{name}_b", beta_b)

        return x @ w.T + b

    def _gaussian_output(self, name, h):
        """Output layer with scalar horseshoe scale."""
        _, in_dim = h.shape

        kappa_aux = pyro.sample(
            f"{name}_kappa_aux",
            dist.InverseGamma(torch.tensor(0.5), torch.tensor(1.0 / (self.b_kappa**2))),
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

        b_out = pyro.sample(f"{name}_bias", dist.Normal(0.0, self.bias_scale))

        w_out = pyro.deterministic(f"{name}_w", kappa * z_out)
        return (h @ w_out.T).squeeze(-1) + b_out

    def forward(self, x, y=None):
        x = x.reshape(-1, 2)
        n = x.shape[0]

        h1 = self.act(self._hs_layer("layer1", x, self.h1))
        h2 = self.act(self._hs_layer("layer2", h1, self.h2))
        mu = self._gaussian_output("out", h2)
        pyro.deterministic("mu", mu)

        sigma2_u = pyro.sample("sigma2_u", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v = pyro.sample("sigma2_v", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u = torch.sqrt(sigma2_u)
        sigma_v = torch.sqrt(sigma2_v)

        with pyro.plate("data", n):
            u = pyro.sample("u", dist.HalfNormal(sigma_u))
            pyro.sample("y_obs", dist.Normal(mu - u, sigma_v), obs=y)

        return mu
