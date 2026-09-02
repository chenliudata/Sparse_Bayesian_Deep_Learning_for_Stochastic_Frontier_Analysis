import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from config_translog import SCALE_NORMAL, ALPHA_U, ALPHA_V, BETA_U, BETA_V


class BLM_Cobb_2In(PyroModule):
    """Two-input Cobb-Douglas Bayesian SFM."""
    def __init__(self, alpha_u=ALPHA_U, alpha_v=ALPHA_V, beta_u=BETA_U, beta_v=BETA_V):
        super().__init__()
        if beta_u is None or beta_v is None:
            raise ValueError("beta_u and beta_v must be calibrated first.")
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v

        self.beta0 = PyroSample(dist.Normal(0.0, SCALE_NORMAL))
        self.beta1 = PyroSample(dist.Normal(0.0, SCALE_NORMAL))
        self.beta2 = PyroSample(dist.Normal(0.0, SCALE_NORMAL))

    def forward(self, x, y=None):
        x = x.reshape(-1, 2)
        x1, x2 = x[:, 0], x[:, 1]
        n = x.shape[0]

        mu = self.beta0 + self.beta1 * x1 + self.beta2 * x2
        pyro.deterministic("mu", mu)

        sigma2_u = pyro.sample("sigma2_u", dist.InverseGamma(self.alpha_u, self.beta_u))
        sigma2_v = pyro.sample("sigma2_v", dist.InverseGamma(self.alpha_v, self.beta_v))
        sigma_u = torch.sqrt(sigma2_u)
        sigma_v = torch.sqrt(sigma2_v)

        with pyro.plate("data", n):
            u = pyro.sample("u", dist.HalfNormal(sigma_u))
            pyro.sample("y_obs", dist.Normal(mu - u, sigma_v), obs=y)

        return mu

