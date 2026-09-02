import torch
import torch.nn as nn


class MyCReLU(nn.Module):
    """Constrained ReLU activation: f(x)=a*x for x>=0, else x."""

    def __init__(self, a: float = 0.8):
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, self.a * x, x)
