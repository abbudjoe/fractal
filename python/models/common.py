from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gated_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(tensor)


def one_minus(tensor: torch.Tensor) -> torch.Tensor:
    return 1.0 - tensor


def rotate_state_pairs(state: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    batch_size, width = state.shape
    pair_count = width // 2
    state_pairs = state.reshape(batch_size, pair_count, 2)
    first = state_pairs[..., 0]
    second = state_pairs[..., 1]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack([rotated_first, rotated_second], dim=-1).reshape(batch_size, width)


def leading_state_slice(state: torch.Tensor, width: int) -> torch.Tensor:
    return state[..., :width]


class SimpleRmsNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        denom = torch.sqrt(torch.mean(tensor * tensor, dim=-1, keepdim=True) + self.eps)
        return tensor / denom * self.weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden)))


def build_linear(d_in: int, d_out: int, *, bias: bool = True) -> nn.Linear:
    return nn.Linear(d_in, d_out, bias=bias)
