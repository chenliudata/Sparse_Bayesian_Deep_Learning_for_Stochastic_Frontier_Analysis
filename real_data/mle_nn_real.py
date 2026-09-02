
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from activations import MyCReLU
from config_real import N_HIDDEN1, N_HIDDEN2, MLE_NN_EPOCHS, MLE_NN_LR, N_INPUTS


def _stable_log_ndtr(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.special, "log_ndtr"):
        return torch.special.log_ndtr(x)
    return torch.log(torch.clamp(torch.distributions.Normal(0.0, 1.0).cdf(x), min=1e-12))


class NN_MLP_NIn(nn.Module):
    """Two-hidden-layer MLP with configurable input dimension."""

    def __init__(self, n_inputs: int = N_INPUTS,
                 n_hidden1: int = N_HIDDEN1, n_hidden2: int = N_HIDDEN2,
                 activation: str = "crelu", a: float = 0.8):
        super().__init__()
        self.n_inputs = n_inputs
        self.fc1 = nn.Linear(n_inputs, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 1)
        self.activation_name = activation.lower()
        if self.activation_name == "relu":
            self.act = nn.ReLU()
        elif self.activation_name == "crelu":
            self.act = MyCReLU(a=a)
        else:
            raise ValueError(f"Unknown activation '{activation}'.")

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)


class SFMNegLogLik(nn.Module):
    def __init__(self, init_log_sigma_u: float = -0.69, init_log_sigma_v: float = -0.69):
        super().__init__()
        self.log_sigma_u = nn.Parameter(torch.tensor(float(init_log_sigma_u), dtype=torch.float64))
        self.log_sigma_v = nn.Parameter(torch.tensor(float(init_log_sigma_v), dtype=torch.float64))

    def forward(self, mu, y):
        sigma_u = torch.exp(self.log_sigma_u).clamp(min=1e-6)
        sigma_v = torch.exp(self.log_sigma_v).clamp(min=1e-6)
        sigma = torch.sqrt(sigma_u ** 2 + sigma_v ** 2).clamp(min=1e-6)
        lam = sigma_u / sigma_v
        eps = y - mu
        z = eps / sigma
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=z.dtype, device=z.device),
            torch.tensor(1.0, dtype=z.dtype, device=z.device),
        )
        log_phi = normal.log_prob(z)
        log_cdf = _stable_log_ndtr(-lam * z)
        return -(math.log(2.0) - torch.log(sigma) + log_phi + log_cdf).sum()


@dataclass
class MLEResult:
    model: NN_MLP_NIn
    mle_layers: List[Tuple[torch.Tensor, torch.Tensor]]
    sigma_u_std: float
    sigma_v_std: float
    loss_history: List[float]
    activation: str


def fit_nn_sfm(
    logX: np.ndarray,
    logY: np.ndarray,
    activation: str = "crelu",
    n_inputs: int = N_INPUTS,
    n_hidden1: int = N_HIDDEN1,
    n_hidden2: int = N_HIDDEN2,
    epochs: int = MLE_NN_EPOCHS,
    lr: float = MLE_NN_LR,
    a: float = 0.8,
    seed: int = 0,
    sigma_u_init: float = 0.5,
    sigma_v_init: float = 0.5,
    weight_decay: Optional[float] = None,
) -> Tuple[MLEResult, dict]:
    """Fit NN-SFM by MLE on standardized data.
    """
    torch.manual_seed(seed)
    activation = activation.lower()
    if activation not in ("relu", "crelu"):
        raise ValueError(f"activation must be 'relu' or 'crelu', got '{activation}'.")

    logX = np.asarray(logX, dtype=np.float64)
    logY = np.asarray(logY, dtype=np.float64).ravel()
    if logX.ndim != 2 or logX.shape[1] != n_inputs:
        raise ValueError(
            f"logX must have shape (n, {n_inputs}), got {logX.shape}"
        )

    x_mean = logX.mean(axis=0)
    x_std = np.maximum(logX.std(axis=0, ddof=0), 1e-8)
    y_mean = float(logY.mean())
    y_std = float(max(logY.std(ddof=0), 1e-8))

    Xs = torch.tensor((logX - x_mean) / x_std, dtype=torch.float64)
    Ys = torch.tensor((logY - y_mean) / y_std, dtype=torch.float64)

    model = NN_MLP_NIn(n_inputs=n_inputs, n_hidden1=n_hidden1, n_hidden2=n_hidden2,
                       activation=activation, a=a).double()
    if activation == "crelu":
        with torch.no_grad():
            for lyr in (model.fc1, model.fc2, model.fc3):
                lyr.weight.data.clamp_(min=0.0)

    nll = SFMNegLogLik(
        init_log_sigma_u=math.log(max(sigma_u_init, 1e-4)),
        init_log_sigma_v=math.log(max(sigma_v_init, 1e-4)),
    )

    weight_params = [model.fc1.weight, model.fc2.weight, model.fc3.weight]
    no_decay_params = [model.fc1.bias, model.fc2.bias, model.fc3.bias,
                       nll.log_sigma_u, nll.log_sigma_v]
    if weight_decay is not None and float(weight_decay) > 0.0:
        optimizer = optim.Adam(
            [
                {"params": weight_params, "weight_decay": float(weight_decay)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )
    else:
        optimizer = optim.Adam(weight_params + no_decay_params, lr=lr)

    clamp_nonneg = (activation == "crelu")
    best_loss = float("inf")
    best_state = None
    best_nll_state = None
    history: List[float] = []

    for _ in range(epochs):
        model.train()
        mu = model(Xs)
        loss = nll(mu, Ys)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(nll.parameters()), max_norm=10.0
        )
        optimizer.step()
        with torch.no_grad():
            if clamp_nonneg:
                for lyr in (model.fc1, model.fc2, model.fc3):
                    lyr.weight.data.clamp_(min=0.0)
            nll.log_sigma_u.data.clamp_(min=math.log(1e-6), max=math.log(100.0))
            nll.log_sigma_v.data.clamp_(min=math.log(1e-6), max=math.log(100.0))

        loss_item = float(loss.detach().cpu())
        history.append(loss_item)
        if math.isfinite(loss_item) and loss_item < best_loss:
            best_loss = loss_item
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_nll_state = {
                "log_sigma_u": nll.log_sigma_u.detach().clone(),
                "log_sigma_v": nll.log_sigma_v.detach().clone(),
            }

    if best_state is None or best_nll_state is None:
        raise ValueError(f"NN-SFM MLE fit failed for activation={activation}.")

    model.load_state_dict(best_state)
    nll.log_sigma_u.data.copy_(best_nll_state["log_sigma_u"])
    nll.log_sigma_v.data.copy_(best_nll_state["log_sigma_v"])

    sigma_u_est = float(torch.exp(best_nll_state["log_sigma_u"]).cpu())
    sigma_v_est = float(torch.exp(best_nll_state["log_sigma_v"]).cpu())

    mle_layers = [
        (model.fc1.weight.detach().float().clone(), model.fc1.bias.detach().float().clone()),
        (model.fc2.weight.detach().float().clone(), model.fc2.bias.detach().float().clone()),
        (model.fc3.weight.detach().float().clone(), model.fc3.bias.detach().float().clone()),
    ]
    result = MLEResult(
        model=model,
        mle_layers=mle_layers,
        sigma_u_std=sigma_u_est,
        sigma_v_std=sigma_v_est,
        loss_history=history,
        activation=activation,
    )
    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return result, stats


def fit_crelu_nn_sfm(logX, logY, **kwargs):
    """CReLU NN-SFM MLE warm start."""
    return fit_nn_sfm(logX, logY, activation="crelu", **kwargs)


def fit_relu_nn_sfm(logX, logY, **kwargs):
    """ReLU NN-SFM MLE warm start. """
    kwargs.setdefault("weight_decay", 1e-3)
    return fit_nn_sfm(logX, logY, activation="relu", **kwargs)
