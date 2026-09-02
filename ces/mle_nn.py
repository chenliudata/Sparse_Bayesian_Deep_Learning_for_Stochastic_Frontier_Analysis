from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from activations import MyCReLU
from config_ces import (
    N_HIDDEN1,
    N_HIDDEN2,
    MLE_RELU_EPOCHS,
    MLE_CRELU_EPOCHS,
    MLE_NN_LR,
)


def _stable_log_ndtr(x: torch.Tensor) -> torch.Tensor:
    """Stable log CDF of standard normal."""
    if hasattr(torch.special, "log_ndtr"):
        return torch.special.log_ndtr(x)
    return torch.log(
        torch.clamp(torch.distributions.Normal(0.0, 1.0).cdf(x), min=1e-12)
    )


class NN_MLP_2In(nn.Module):
    """Two-input MLP with ReLU or CReLU."""

    def __init__(
        self,
        n_hidden1=N_HIDDEN1,
        n_hidden2=N_HIDDEN2,
        activation: str = "crelu",
        a: float = 0.9,
    ):
        super().__init__()
        self.fc1 = nn.Linear(2, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 1)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "crelu":
            self.act = MyCReLU(a=a)
        else:
            raise ValueError(f"Unknown activation {activation}")
        self.activation_name = activation

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2).squeeze(-1)


class SFMNegLogLik(nn.Module):
    """Negative log-likelihood for SFM errors."""

    def __init__(self, init_log_sigma_u=-0.69, init_log_sigma_v=-0.69):  # Placeholder
        super().__init__()
        self.log_sigma_u = nn.Parameter(
            torch.tensor(float(init_log_sigma_u), dtype=torch.float64)
        )
        self.log_sigma_v = nn.Parameter(
            torch.tensor(float(init_log_sigma_v), dtype=torch.float64)
        )

    def forward(self, mu, y):
        sigma_u = torch.exp(self.log_sigma_u).clamp(min=1e-6)
        sigma_v = torch.exp(self.log_sigma_v).clamp(min=1e-6)
        sigma = torch.sqrt(sigma_u**2 + sigma_v**2).clamp(min=1e-6)
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
    """NN-MLE warm-start results."""

    model: NN_MLP_2In
    mle_layers: List[Tuple[torch.Tensor, torch.Tensor]]
    sigma_u_std: float
    sigma_v_std: float
    loss_history: List[float]
    activation: str
    best_epoch: int
    epochs_run: int


def fit_nn_sfm_2in(
    logx1,
    logx2,
    logy,
    activation="crelu",
    seed=0,
    sigma_u_init=0.5,
    sigma_v_init=0.5,
    a=0.8,
    weight_decay: Optional[float] = None,
    max_epochs: int = MLE_RELU_EPOCHS,
):
    """Fit two-input NN-SFM by MLE on standardized data."""

    torch.manual_seed(seed)

    logx1 = np.asarray(logx1, dtype=np.float64).ravel()
    logx2 = np.asarray(logx2, dtype=np.float64).ravel()
    logy = np.asarray(logy, dtype=np.float64).ravel()

    x1_mean, x1_std = logx1.mean(), max(logx1.std(ddof=0), 1e-8)
    x2_mean, x2_std = logx2.mean(), max(logx2.std(ddof=0), 1e-8)
    y_mean, y_std = logy.mean(), max(logy.std(ddof=0), 1e-8)

    Xs = torch.tensor(
        np.column_stack([(logx1 - x1_mean) / x1_std, (logx2 - x2_mean) / x2_std]),
        dtype=torch.float64,
    )
    Ys = torch.tensor((logy - y_mean) / y_std, dtype=torch.float64)

    model = NN_MLP_2In(activation=activation, a=a).double()
    if activation == "crelu":
        with torch.no_grad():
            for lyr in (model.fc1, model.fc2, model.fc3):
                lyr.weight.data.clamp_(min=0.0)
            model.fc1.bias.data.clamp_(min=0.0)
            model.fc2.bias.data.clamp_(min=0.0)
    nll = SFMNegLogLik(
        init_log_sigma_u=math.log(max(sigma_u_init, 1e-4)),
        init_log_sigma_v=math.log(max(sigma_v_init, 1e-4)),
    )

    weight_params = [model.fc1.weight, model.fc2.weight, model.fc3.weight]
    no_decay_params = [
        model.fc1.bias,
        model.fc2.bias,
        model.fc3.bias,
        nll.log_sigma_u,
        nll.log_sigma_v,
    ]
    if weight_decay is not None and float(weight_decay) > 0.0:
        optimizer = optim.Adam(
            [
                {"params": weight_params, "weight_decay": float(weight_decay)},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=MLE_NN_LR,
        )
    else:
        optimizer = optim.Adam(weight_params + no_decay_params, lr=MLE_NN_LR)

    best_loss = float("inf")
    best_state = None
    best_nll_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        mu = model(Xs)
        loss = nll(mu, Ys)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(nll.parameters()), max_norm=10.0
        )
        optimizer.step()

        with torch.no_grad():
            if activation == "crelu":
                for lyr in (model.fc1, model.fc2, model.fc3):
                    lyr.weight.data.clamp_(min=0.0)
                model.fc1.bias.data.clamp_(min=0.0)
                model.fc2.bias.data.clamp_(min=0.0)
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
            best_epoch = epoch

    if best_state is None or best_nll_state is None:
        raise ValueError(f"NN-SFM MLE fit ({activation}) failed")

    model.load_state_dict(best_state)
    nll.log_sigma_u.data.copy_(best_nll_state["log_sigma_u"])
    nll.log_sigma_v.data.copy_(best_nll_state["log_sigma_v"])

    result = MLEResult(
        model=model,
        mle_layers=[
            (
                model.fc1.weight.detach().float().clone(),
                model.fc1.bias.detach().float().clone(),
            ),
            (
                model.fc2.weight.detach().float().clone(),
                model.fc2.bias.detach().float().clone(),
            ),
            (
                model.fc3.weight.detach().float().clone(),
                model.fc3.bias.detach().float().clone(),
            ),
        ],
        sigma_u_std=float(torch.exp(best_nll_state["log_sigma_u"]).cpu()),
        sigma_v_std=float(torch.exp(best_nll_state["log_sigma_v"]).cpu()),
        loss_history=history,
        activation=activation,
        best_epoch=best_epoch,
        epochs_run=len(history),
    )
    stats = {
        "x1_mean": float(x1_mean),
        "x1_std": float(x1_std),
        "x2_mean": float(x2_mean),
        "x2_std": float(x2_std),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
    return result, stats


def fit_relu_nn_sfm(
    logx1,
    logx2,
    logy,
    seed=0,
    sigma_u_init=0.5,
    sigma_v_init=0.5,
    weight_decay: Optional[float] = None,
    max_epochs: int = MLE_RELU_EPOCHS,
    **kwargs,
):
    """Fit ReLU NN-SFM."""

    return fit_nn_sfm_2in(
        logx1,
        logx2,
        logy,
        activation="relu",
        seed=seed,
        sigma_u_init=sigma_u_init,
        sigma_v_init=sigma_v_init,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        **kwargs,
    )


def fit_crelu_nn_sfm(
    logx1,
    logx2,
    logy,
    seed=0,
    sigma_u_init=0.5,
    sigma_v_init=0.5,
    weight_decay: Optional[float] = None,
    max_epochs: int = MLE_CRELU_EPOCHS,
    **kwargs,
):
    """Fit CReLU NN-SFM."""

    return fit_nn_sfm_2in(
        logx1,
        logx2,
        logy,
        activation="crelu",
        seed=seed,
        sigma_u_init=sigma_u_init,
        sigma_v_init=sigma_v_init,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        **kwargs,
    )
