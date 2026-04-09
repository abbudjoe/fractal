from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from python.specs.runtime import PrimitiveStateTransformMode


@dataclass(frozen=True)
class SequencePrimitiveStepResult:
    next_state: torch.Tensor
    emitted_output: torch.Tensor


@dataclass(frozen=True)
class SequencePrimitiveScanResult:
    emitted_outputs: torch.Tensor
    final_state: torch.Tensor


def allocate_emitted_outputs(
    *,
    batch_size: int,
    seq_len: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        batch_size,
        seq_len,
        width,
        device=device,
        dtype=dtype,
    )


def packed_linear_chunks(inputs: torch.Tensor, *layers: nn.Linear) -> tuple[torch.Tensor, ...]:
    with record_function("path1.primitive.runtime.packed_input_projection"):
        packed_weight = torch.cat([layer.weight for layer in layers], dim=0)
        if all(layer.bias is not None for layer in layers):
            packed_bias = torch.cat([layer.bias for layer in layers if layer.bias is not None], dim=0)
        else:
            packed_bias = None
        packed_outputs = F.linear(inputs, packed_weight, packed_bias)
    split_sizes = [layer.out_features for layer in layers]
    return tuple(packed_outputs.split(split_sizes, dim=-1))


def rotary_runtime_components(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with record_function("path1.primitive.runtime.rotary_trig"):
        return torch.cos(angles), torch.sin(angles)


def rotate_state_pairs_with_trig(
    state: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    batch_size, width = state.shape
    pair_count = width // 2
    state_pairs = state.reshape(batch_size, pair_count, 2)
    first = state_pairs[..., 0]
    second = state_pairs[..., 1]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return torch.stack([rotated_first, rotated_second], dim=-1).reshape(batch_size, width)


class BlockDiagonalLinear(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        if width % blocks != 0:
            raise ValueError(f"block-diagonal transform width {width} must be divisible by blocks {blocks}")
        self.width = width
        self.blocks = blocks
        self.block_width = width // blocks
        self.weight = nn.Parameter(torch.empty(blocks, self.block_width, self.block_width))
        self.bias = nn.Parameter(torch.empty(width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        bound = 1 / self.block_width**0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        leading_shape = inputs.shape[:-1]
        reshaped = inputs.reshape(-1, self.blocks, self.block_width)
        projected = torch.einsum("bgi,goi->bgo", reshaped, self.weight)
        projected = projected.reshape(*leading_shape, self.width)
        return projected + self.bias


def build_state_transform_projection(
    width: int,
    mode: PrimitiveStateTransformMode,
) -> nn.Module:
    if mode is PrimitiveStateTransformMode.DENSE:
        return nn.Linear(width, width)
    if mode is PrimitiveStateTransformMode.BLOCK_DIAGONAL_2:
        return BlockDiagonalLinear(width, blocks=2)
    if mode is PrimitiveStateTransformMode.BLOCK_DIAGONAL_4:
        return BlockDiagonalLinear(width, blocks=4)
    raise ValueError(f"unsupported primitive state transform mode: {mode}")
