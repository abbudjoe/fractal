from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.runtime.cuda_timing import timed_region
from python.specs.runtime import PrimitiveStateTransformMode


@dataclass(frozen=True)
class SequencePrimitiveStepResult:
    next_state: torch.Tensor
    emitted_output: torch.Tensor


@dataclass(frozen=True)
class SequencePrimitiveScanResult:
    emitted_outputs: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class EggrollPerturbationSpec:
    """Low-rank ES perturbation contract for virtual LoRA-style linear probes."""

    rank: int = 1
    sigma: float = 1.0e-3
    population_size: int = 1
    antithetic: bool = True
    seed: int = 0

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"Eggroll rank must be positive, got {self.rank}")
        if self.sigma < 0.0:
            raise ValueError(f"Eggroll sigma must be non-negative, got {self.sigma}")
        if self.population_size <= 0:
            raise ValueError(f"Eggroll population_size must be positive, got {self.population_size}")


def make_eggroll_factors(
    *,
    population_size: int,
    d_out: int,
    d_in: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic A/B factors for virtual low-rank perturbations."""

    if population_size <= 0 or d_out <= 0 or d_in <= 0 or rank <= 0:
        raise ValueError(
            "Eggroll factors require positive population_size, d_out, d_in, and rank: "
            f"{population_size=}, {d_out=}, {d_in=}, {rank=}"
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    a = torch.randn(population_size, d_out, rank, device=device, dtype=dtype, generator=generator)
    b = torch.randn(population_size, d_in, rank, device=device, dtype=dtype, generator=generator)
    return a, b


def virtual_eggroll_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    perturbation_a: torch.Tensor,
    perturbation_b: torch.Tensor,
    sigma: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply W + sigma / sqrt(rank) * A B^T without materializing W per population.

    Expected shapes:
    - inputs: [population, ... , d_in]
    - weight: [d_out, d_in]
    - perturbation_a: [population, d_out, rank]
    - perturbation_b: [population, d_in, rank]
    """

    if inputs.ndim < 2:
        raise ValueError(f"Eggroll inputs must have shape [population, ..., d_in], got {tuple(inputs.shape)}")
    population_size = inputs.shape[0]
    if perturbation_a.shape[0] != population_size or perturbation_b.shape[0] != population_size:
        raise ValueError(
            "Eggroll population dimension mismatch: "
            f"inputs={population_size}, A={perturbation_a.shape[0]}, B={perturbation_b.shape[0]}"
        )
    if perturbation_a.ndim != 3 or perturbation_b.ndim != 3:
        raise ValueError("Eggroll perturbation factors must have rank-3 shapes [population, width, rank]")
    if perturbation_a.shape[2] != perturbation_b.shape[2]:
        raise ValueError(f"Eggroll factor ranks must match, got A={perturbation_a.shape} B={perturbation_b.shape}")
    if weight.shape != (perturbation_a.shape[1], perturbation_b.shape[1]):
        raise ValueError(
            "Eggroll weight/factor shape mismatch: "
            f"weight={tuple(weight.shape)}, A={tuple(perturbation_a.shape)}, B={tuple(perturbation_b.shape)}"
        )
    if inputs.shape[-1] != weight.shape[1]:
        raise ValueError(f"Eggroll input dim {inputs.shape[-1]} does not match weight input dim {weight.shape[1]}")

    with timed_region("path1.eggroll.virtual_linear.base"):
        base = F.linear(inputs, weight, bias)
    with timed_region("path1.eggroll.virtual_linear.low_rank"):
        inner = torch.einsum("p...i,pir->p...r", inputs, perturbation_b)
        delta = torch.einsum("p...r,por->p...o", inner, perturbation_a)
    return base + (sigma / math.sqrt(float(perturbation_a.shape[2]))) * delta


def materialized_eggroll_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    perturbation_a: torch.Tensor,
    perturbation_b: torch.Tensor,
    sigma: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference path that explicitly materializes W + low-rank noise per population."""

    rank = perturbation_a.shape[2]
    perturbation = torch.einsum("por,pir->poi", perturbation_a, perturbation_b)
    population_weights = weight.unsqueeze(0) + (sigma / math.sqrt(float(rank))) * perturbation
    output = torch.einsum("p...i,poi->p...o", inputs, population_weights)
    if bias is not None:
        output = output + bias.view(*([1] * (output.ndim - 1)), -1)
    return output


def eggroll_update_matrix(
    perturbation_a: torch.Tensor,
    perturbation_b: torch.Tensor,
    fitness: torch.Tensor,
    *,
    learning_rate: float,
) -> torch.Tensor:
    """Score-weighted low-rank ES update for a matrix with shape [d_out, d_in]."""

    if perturbation_a.ndim != 3 or perturbation_b.ndim != 3:
        raise ValueError("Eggroll update factors must have shapes [population, width, rank]")
    if fitness.ndim != 1 or fitness.shape[0] != perturbation_a.shape[0]:
        raise ValueError(
            f"Eggroll fitness must be [population], got fitness={tuple(fitness.shape)} A={tuple(perturbation_a.shape)}"
        )
    if perturbation_a.shape[0] != perturbation_b.shape[0] or perturbation_a.shape[2] != perturbation_b.shape[2]:
        raise ValueError(f"Eggroll A/B shape mismatch: A={tuple(perturbation_a.shape)} B={tuple(perturbation_b.shape)}")
    weighted_a = perturbation_a * fitness.to(dtype=perturbation_a.dtype, device=perturbation_a.device).view(-1, 1, 1)
    update = torch.einsum("por,pir->oi", weighted_a, perturbation_b)
    normalizer = float(perturbation_a.shape[0] * perturbation_a.shape[2])
    return (learning_rate / normalizer) * update


class PackedLinearProjection(nn.Module):
    def __init__(self, d_in: int, split_sizes: tuple[int, ...], *, bias: bool = True) -> None:
        super().__init__()
        if not split_sizes:
            raise ValueError("packed projection requires at least one split")
        if any(size <= 0 for size in split_sizes):
            raise ValueError(f"packed projection split sizes must be positive, got {split_sizes}")
        self.split_sizes = split_sizes
        self.projection = nn.Linear(d_in, sum(split_sizes), bias=bias)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.projection.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.projection.bias

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with timed_region("path1.primitive.runtime.packed_input_projection"):
            packed_outputs = self.projection(inputs)
        return tuple(packed_outputs.split(self.split_sizes, dim=-1))


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
    with timed_region("path1.primitive.runtime.packed_input_projection"):
        packed_weight = torch.cat([layer.weight for layer in layers], dim=0)
        if all(layer.bias is not None for layer in layers):
            packed_bias = torch.cat([layer.bias for layer in layers if layer.bias is not None], dim=0)
        else:
            packed_bias = None
        packed_outputs = F.linear(inputs, packed_weight, packed_bias)
    split_sizes = [layer.out_features for layer in layers]
    return tuple(packed_outputs.split(split_sizes, dim=-1))


def rotary_runtime_components(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with timed_region("path1.primitive.runtime.rotary_trig"):
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


def _sum_to_input_shape(gradient: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if gradient.shape == shape:
        return gradient
    return gradient.sum_to_size(shape)


class _ParcaeStateMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state: torch.Tensor, decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(state, decay)
        ctx.injection_shape = injection.shape
        return decay * state + injection

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        state, decay = ctx.saved_tensors
        grad_state = grad_output * decay
        grad_decay = _sum_to_input_shape(grad_output * state, decay.shape)
        grad_injection = _sum_to_input_shape(grad_output, ctx.injection_shape)
        return grad_state, grad_decay, grad_injection


class _ParcaeResidualMix(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mixed: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(mixed, block_out, nonlinear)
        return mixed + nonlinear * (block_out - mixed)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        mixed, block_out, nonlinear = ctx.saved_tensors
        grad_mixed = grad_output * (1.0 - nonlinear)
        grad_block_out = grad_output * nonlinear
        grad_nonlinear = _sum_to_input_shape(grad_output * (block_out - mixed), nonlinear.shape)
        return grad_mixed, grad_block_out, grad_nonlinear


def manual_autograd_parcae_state_mix(
    state: torch.Tensor,
    decay: torch.Tensor,
    injection: torch.Tensor,
) -> torch.Tensor:
    return _ParcaeStateMix.apply(state, decay, injection)


def manual_autograd_parcae_residual_mix(
    mixed: torch.Tensor,
    block_out: torch.Tensor,
    nonlinear: torch.Tensor,
) -> torch.Tensor:
    return _ParcaeResidualMix.apply(mixed, block_out, nonlinear)


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
    if mode is PrimitiveStateTransformMode.BLOCK_DIAGONAL_8:
        return BlockDiagonalLinear(width, blocks=8)
    raise ValueError(f"unsupported primitive state transform mode: {mode}")
