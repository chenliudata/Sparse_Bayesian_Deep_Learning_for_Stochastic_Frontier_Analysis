import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from activations import MyCReLU
from config_translog import (
    N_HIDDEN1,
    N_HIDDEN2,
    SCALE_HALFLAPLACE,
    SCALE_LAPLACE,
    ALPHA_U,
    ALPHA_V,
    BETA_U,
    BETA_V,
)


def half_laplace(scale, shape, event_dim):
    """Folded Laplace prior."""
    base = dist.FoldedDistribution(
        dist.Laplace(torch.tensor(0.0), torch.tensor(scale)), validate_args=False
    )
    return base.expand(shape).to_event(event_dim)


class BNN_SFM_CRELU_LAPLACE_2In(PyroModule):
    """Two-input CReLU BNN-SFM with nonnegative hidden weights and biases."""

    def __init__(
        self,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        prior_scale_weight=SCALE_HALFLAPLACE,
        prior_scale_bias=SCALE_LAPLACE,
        a=0.8,
        alpha_u=ALPHA_U,
        alpha_v=ALPHA_V,
        beta_u=BETA_U,
        beta_v=BETA_V,
        mle_layers=None,
    ):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError(
                "beta_u and beta_v must be calibrated via calibrate_ig_prior. "
            )
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v
        self.act = MyCReLU(a=a)
        self._using_mle = mle_layers is not None

        self.fc1 = PyroModule[nn.Linear](2, n_hidden1)
        self.fc2 = PyroModule[nn.Linear](n_hidden1, n_hidden2)
        self.fc3 = PyroModule[nn.Linear](n_hidden2, 1)

        self.fc1.weight = PyroSample(
            half_laplace(prior_scale_weight, [n_hidden1, 2], 2)
        )
        self.fc1.bias = PyroSample(half_laplace(prior_scale_bias, [n_hidden1], 1))
        self.fc2.weight = PyroSample(
            half_laplace(prior_scale_weight, [n_hidden2, n_hidden1], 2)
        )
        self.fc2.bias = PyroSample(half_laplace(prior_scale_bias, [n_hidden2], 1))
        self.fc3.weight = PyroSample(
            half_laplace(prior_scale_weight, [1, n_hidden2], 2)
        )
        self.fc3.bias = PyroSample(
            dist.Laplace(0.0, prior_scale_bias).expand([1]).to_event(1)
        )

        if mle_layers is not None:
            if len(mle_layers) != 3:
                raise ValueError("mle_layers must contain exactly three (W, b) pairs.")
            (W1, b1), (W2, b2), (W3, b3) = mle_layers

            expected = [
                ("W1", W1, (n_hidden1, 2)),
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

    def mle_initial_params(self):
        """Initial values from NN-MLE."""
        if not self._using_mle:
            return {}
        (W1, b1), (W2, b2), (W3, b3) = self._mle_layers
        return {
            "fc1.weight": W1.clamp(min=1e-6),
            "fc1.bias": b1.clamp(min=1e-6),
            "fc2.weight": W2.clamp(min=1e-6),
            "fc2.bias": b2.clamp(min=1e-6),
            "fc3.weight": W3.clamp(min=1e-6),
            "fc3.bias": b3.detach().clone(),
        }

    def forward(self, x, y=None):
        x = x.reshape(-1, 2)
        n = x.shape[0]

        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        mu = self.fc3(h2).squeeze(-1)
        pyro.deterministic("mu", mu)

        sigma2_u = pyro.sample("sigma2_u", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v = pyro.sample("sigma2_v", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u = torch.sqrt(sigma2_u)
        sigma_v = torch.sqrt(sigma2_v)

        with pyro.plate("data", n):
            u = pyro.sample("u", dist.HalfNormal(sigma_u))
            pyro.sample("y_obs", dist.Normal(mu - u, sigma_v), obs=y)

        return mu
