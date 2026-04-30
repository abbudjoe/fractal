from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from python.runtime.cuda_timing import timed_region


@dataclass(frozen=True)
class ParcaeLoopRegionTimingNames:
    """Stable timing slots owned by the Parcae loop-region contract."""

    loop_step: str = "loop_step"
    loop_iteration_compiled: str = "loop_iteration_compiled"
    first_state_mix_fused: str = "first_state_mix_fused"
    state_mix: str = "state_mix"
    recurrent_blocks: str = "recurrent_blocks"
    recurrent_block_forward: str = "recurrent_block_forward"
    recurrent_residual_mix: str = "recurrent_residual_mix"
    detach_truncated_history: str = "detach_truncated_history"
    state_diagnostic_reduction: str = "state_diagnostic_reduction"

    def scoped(self, prefix: str, name: str) -> str:
        if not prefix:
            raise ValueError("timing prefix must be non-empty")
        if name not in self.__dict__.values():
            raise ValueError(f"unknown Parcae loop timing name: {name!r}")
        return f"{prefix}.{name}"


PARCAE_LOOP_REGION_TIMING_NAMES = ParcaeLoopRegionTimingNames()


@dataclass(frozen=True)
class ParcaeLoopRegionTensorLayout:
    """Shape/device/dtype contract consumed by native loop-region kernels."""

    batch_size: int
    seq_len: int
    width: int
    dtype: torch.dtype
    device: torch.device
    state_is_contiguous: bool

    @classmethod
    def from_state(cls, state: torch.Tensor) -> "ParcaeLoopRegionTensorLayout":
        if state.ndim != 3:
            raise ValueError(
                "Parcae loop state must be a rank-3 [batch, sequence, width] tensor, "
                f"got shape={tuple(state.shape)}"
            )
        batch_size, seq_len, width = state.shape
        if batch_size <= 0 or seq_len <= 0 or width <= 0:
            raise ValueError(
                "Parcae loop state dimensions must be positive, "
                f"got shape={tuple(state.shape)}"
            )
        return cls(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            width=int(width),
            dtype=state.dtype,
            device=state.device,
            state_is_contiguous=state.is_contiguous(),
        )

    @property
    def state_shape(self) -> torch.Size:
        return torch.Size((self.batch_size, self.seq_len, self.width))


def _validate_loop_control_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
    layout: ParcaeLoopRegionTensorLayout,
) -> None:
    if tensor.device != layout.device:
        raise ValueError(
            f"Parcae loop control {name} must be on {layout.device}, "
            f"got {tensor.device}"
        )
    if tensor.dtype != layout.dtype:
        raise ValueError(
            f"Parcae loop control {name} must use dtype {layout.dtype}, "
            f"got {tensor.dtype}"
        )
    if tensor.ndim != 3:
        raise ValueError(
            f"Parcae loop control {name} must be rank-3 and broadcastable to "
            f"[batch, sequence, width], got shape={tuple(tensor.shape)}"
        )
    if tensor.shape[-1] != layout.width:
        raise ValueError(
            f"Parcae loop control {name} width must match state width "
            f"{layout.width}, got shape={tuple(tensor.shape)}"
        )
    try:
        broadcast_shape = torch.broadcast_shapes(tensor.shape, layout.state_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Parcae loop control {name} shape {tuple(tensor.shape)} cannot "
            f"broadcast to state shape {tuple(layout.state_shape)}"
        ) from exc
    if broadcast_shape != layout.state_shape:
        raise ValueError(
            f"Parcae loop control {name} must broadcast exactly to state shape "
            f"{tuple(layout.state_shape)}, got {tuple(broadcast_shape)}"
        )


@dataclass(frozen=True)
class ParcaeLoopRegionControls:
    """Tensors owned by one Parcae recurrent loop region."""

    decay: torch.Tensor
    injection: torch.Tensor
    nonlinear: torch.Tensor

    def validate_against(self, state: torch.Tensor) -> ParcaeLoopRegionTensorLayout:
        layout = ParcaeLoopRegionTensorLayout.from_state(state)
        _validate_loop_control_tensor(name="decay", tensor=self.decay, layout=layout)
        _validate_loop_control_tensor(name="injection", tensor=self.injection, layout=layout)
        _validate_loop_control_tensor(name="nonlinear", tensor=self.nonlinear, layout=layout)
        return layout

    def contiguous(self) -> "ParcaeLoopRegionControls":
        return ParcaeLoopRegionControls(
            decay=self.decay.contiguous(),
            injection=self.injection.contiguous(),
            nonlinear=self.nonlinear.contiguous(),
        )


@dataclass(frozen=True)
class ParcaeLoopRegionConfig:
    """Explicit runtime contract for one Parcae recurrent loop region."""

    loop_count: int
    gradient_start_step: int
    recurrent_block_count: int
    timing_prefix: str
    fuse_first_state_mix: bool = False
    diagnostics_enabled: bool = False

    def validate(self) -> None:
        if self.loop_count <= 0:
            raise ValueError(f"loop_count must be positive, got {self.loop_count}")
        if not 0 <= self.gradient_start_step <= self.loop_count:
            raise ValueError(
                "gradient_start_step must be in [0, loop_count], "
                f"got {self.gradient_start_step} for loop_count={self.loop_count}"
            )
        if self.recurrent_block_count <= 0:
            raise ValueError(
                "recurrent_block_count must be positive, "
                f"got {self.recurrent_block_count}"
            )
        if not self.timing_prefix:
            raise ValueError("timing_prefix must be non-empty")


@dataclass(frozen=True)
class ParcaeLoopRegionKernels:
    """Callable ownership boundary for the loop-region runtime.

    This is intentionally larger than a tiny state/residual update op. Future
    native kernels should replace this full boundary, not just the reductions
    inside one scalar-looking update.
    """

    state_mix: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    forward_recurrent_block: Callable[[int, torch.Tensor], torch.Tensor]
    apply_recurrent_residual: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor, ParcaeLoopRegionControls],
        torch.Tensor,
    ]
    compiled_iteration: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclass(frozen=True)
class ParcaeLoopRegionResult:
    final_state: torch.Tensor
    norm_history: list[torch.Tensor]


def run_parcae_loop_region(
    *,
    initial_state: torch.Tensor,
    controls: ParcaeLoopRegionControls,
    config: ParcaeLoopRegionConfig,
    kernels: ParcaeLoopRegionKernels,
) -> ParcaeLoopRegionResult:
    """Run one Parcae recurrent loop region with an explicit ownership boundary."""

    config.validate()
    controls.validate_against(initial_state)
    timing_names = PARCAE_LOOP_REGION_TIMING_NAMES
    state = initial_state
    norm_history: list[torch.Tensor] = []

    def run_loop_iteration(current_state: torch.Tensor, *, first_state_mix_fused: bool) -> torch.Tensor:
        with timed_region(timing_names.scoped(config.timing_prefix, timing_names.loop_step)):
            if kernels.compiled_iteration is not None:
                with timed_region(timing_names.scoped(config.timing_prefix, timing_names.loop_iteration_compiled)):
                    return kernels.compiled_iteration(current_state)
            if first_state_mix_fused:
                with timed_region(timing_names.scoped(config.timing_prefix, timing_names.first_state_mix_fused)):
                    mixed = controls.injection.expand_as(current_state)
            else:
                with timed_region(timing_names.scoped(config.timing_prefix, timing_names.state_mix)):
                    mixed = kernels.state_mix(current_state, controls.decay, controls.injection)
            with timed_region(timing_names.scoped(config.timing_prefix, timing_names.recurrent_blocks)):
                for block_index in range(config.recurrent_block_count):
                    with timed_region(timing_names.scoped(config.timing_prefix, timing_names.recurrent_block_forward)):
                        block_out = kernels.forward_recurrent_block(block_index, mixed)
                    with timed_region(timing_names.scoped(config.timing_prefix, timing_names.recurrent_residual_mix)):
                        mixed = kernels.apply_recurrent_residual(
                            block_index,
                            current_state,
                            mixed,
                            block_out,
                            controls,
                        )
            return mixed

    for loop_index in range(config.loop_count):
        first_state_mix_fused = config.fuse_first_state_mix and loop_index == 0
        if loop_index < config.gradient_start_step:
            # The BPTT boundary is part of the loop-region contract. Truncated
            # iterations provide value context without building an autograd tape
            # that is immediately detached.
            with torch.no_grad():
                mixed = run_loop_iteration(state, first_state_mix_fused=first_state_mix_fused)
        else:
            mixed = run_loop_iteration(state, first_state_mix_fused=first_state_mix_fused)
        if loop_index < config.gradient_start_step:
            with timed_region(timing_names.scoped(config.timing_prefix, timing_names.detach_truncated_history)):
                mixed = mixed.detach()
        state = mixed
        if config.diagnostics_enabled:
            with timed_region(timing_names.scoped(config.timing_prefix, timing_names.state_diagnostic_reduction)):
                norm_history.append(state.detach().float().norm(dim=-1).mean())

    return ParcaeLoopRegionResult(final_state=state, norm_history=norm_history)
