import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from config_ces import N_HIDDEN1, N_HIDDEN2, SCALE_NORMAL, ALPHA_U, ALPHA_V, BETA_U, BETA_V

class BNN_SFM_RELU_2In(PyroModule):
    """Two-input ReLU BNN-SFM with normal weights."""
    def __init__(self, n_hidden1=N_HIDDEN1, n_hidden2=N_HIDDEN2, prior_scale_weight=SCALE_NORMAL, prior_scale_bias=SCALE_NORMAL,
                 alpha_u=ALPHA_U, alpha_v=ALPHA_V, beta_u=BETA_U, beta_v=BETA_V, mle_layers=None):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError(
                "beta_u and beta_v must be calibrated via calibrate_ig_prior "
            )
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v
        self._using_mle = mle_layers is not None

        self.fc1 = PyroModule[nn.Linear](2, n_hidden1)
        self.fc2 = PyroModule[nn.Linear](n_hidden1, n_hidden2)
        self.fc3 = PyroModule[nn.Linear](n_hidden2, 1)

        self.fc1.weight = PyroSample(dist.Normal(0.0, prior_scale_weight).expand([n_hidden1, 2]).to_event(2))
        self.fc1.bias   = PyroSample(dist.Normal(0.0, prior_scale_bias).expand([n_hidden1]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0.0, prior_scale_weight).expand([n_hidden2, n_hidden1]).to_event(2))
        self.fc2.bias   = PyroSample(dist.Normal(0.0, prior_scale_bias).expand([n_hidden2]).to_event(1))
        self.fc3.weight = PyroSample(dist.Normal(0.0, prior_scale_weight).expand([1, n_hidden2]).to_event(2))
        self.fc3.bias   = PyroSample(dist.Normal(0.0, prior_scale_bias).expand([1]).to_event(1))

        self.act = nn.ReLU()
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
            "fc1.weight": W1.detach().clone(),
            "fc1.bias": b1.detach().clone(),
            "fc2.weight": W2.detach().clone(),
            "fc2.bias": b2.detach().clone(),
            "fc3.weight": W3.detach().clone(),
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
