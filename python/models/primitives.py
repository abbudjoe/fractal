from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from python.models.common import (
    PositionWiseFeedForward,
    SimpleRmsNorm,
    build_linear,
    gated_sigmoid,
    leading_state_slice,
    one_minus,
    rotate_state_pairs,
)
from python.runtime.recurrent import (
    BlockDiagonalLinear,
    PackedLinearProjection,
    SequencePrimitiveScanResult,
    SequencePrimitiveStepResult,
    allocate_emitted_outputs as _allocate_emitted_outputs,
    build_state_transform_projection as _build_state_transform_projection,
    rotate_state_pairs_with_trig as _rotate_state_pairs_with_trig,
    rotary_runtime_components as _rotary_runtime_components,
)
from python.runtime.triton_primitives import (
    TritonPrimitiveBackend,
    build_triton_primitive_backend,
    ensure_triton_runtime_available,
)
from python.specs.path1 import (
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveWrapperMode,
)
from python.specs.runtime import PrimitiveStateTransformMode


@dataclass(frozen=True)
class ContractiveRuntimePlan:
    gates: torch.Tensor
    projected_inputs: torch.Tensor


@dataclass(frozen=True)
class P1FractalHybridRuntimePlan:
    gates: torch.Tensor
    projected_inputs: torch.Tensor


@dataclass(frozen=True)
class PackedChunksRuntimePlan:
    chunks: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class P20RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
    output_gates: torch.Tensor


@dataclass(frozen=True)
class P20GdnRoleRuntimePlan:
    queries: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    alpha_gates: torch.Tensor
    beta_gates: torch.Tensor
    output_gates: torch.Tensor


@dataclass(frozen=True)
class P2RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
    output_gates: torch.Tensor


@dataclass(frozen=True)
class P23RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    dynamics_mix_gates: torch.Tensor
    dynamics_keep_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
    output_gates: torch.Tensor


@dataclass(frozen=True)
class P21RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
    output_gates: torch.Tensor


@dataclass(frozen=True)
class P22RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
    output_gates: torch.Tensor


class SequencePrimitive(nn.Module):
    state_width: int
    d_model: int

    def __init__(self) -> None:
        super().__init__()
        self._primitive_runtime_backend = "torch"
        self._triton_backend: TritonPrimitiveBackend | None = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_width, device=device, dtype=dtype)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> Any:  # pragma: no cover - contract boundary
        raise RuntimeError(f"{self.__class__.__name__} does not expose a runtime execution plan")

    def scan_with_runtime_plan(
        self,
        runtime_plan: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:  # pragma: no cover - contract boundary
        raise RuntimeError(f"{self.__class__.__name__} does not implement runtime-plan scanning")

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        batch_size, seq_len, _width = inputs.shape
        state = (
            initial_state
            if initial_state is not None
            else self.init_state(batch_size, inputs.device, inputs.dtype)
        )
        outputs = []
        for position in range(seq_len):
            step = self.step(state, inputs[:, position, :])
            state = step.next_state
            outputs.append(step.emitted_output.unsqueeze(1))
        return SequencePrimitiveScanResult(emitted_outputs=torch.cat(outputs, dim=1), final_state=state)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        if primitive_runtime_backend == "triton":
            ensure_triton_runtime_available()
            self._triton_backend = build_triton_primitive_backend()
        else:
            self._triton_backend = None
        self._primitive_runtime_backend = primitive_runtime_backend or "torch"
        del compile_mode


def _resolve_initial_state(
    primitive: SequencePrimitive,
    inputs: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    if initial_state is not None:
        return initial_state
    return primitive.init_state(inputs.shape[0], inputs.device, inputs.dtype)


def _row_l2_norm(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor * tensor).sum(dim=-1, keepdim=True).add(1.0e-12).sqrt()


def _clamp_symmetric_by_row(tensor: torch.Tensor, clamp: torch.Tensor) -> torch.Tensor:
    return torch.maximum(torch.minimum(tensor, clamp), -clamp)


def _clamp_max_by_row(tensor: torch.Tensor, clamp: torch.Tensor) -> torch.Tensor:
    return torch.minimum(tensor, clamp)


def _repeat_row_value(value: torch.Tensor, width: int) -> torch.Tensor:
    return value.repeat(1, width)


def _complex_square(state: torch.Tensor) -> torch.Tensor:
    real, imag = state.chunk(2, dim=-1)
    next_real = real * real - imag * imag
    next_imag = 2.0 * real * imag
    return torch.cat((next_real, next_imag), dim=-1)


def _hierarchical_top_level(state: torch.Tensor) -> torch.Tensor:
    return state[:, -1, :]


LEGACY_HIERARCHICAL_LEVELS = 3
LEGACY_ROOT_DEPTH_FRACTION = 0.0
LEGACY_ROOT_ITERATIONS = 1
NUM_IFS_MAPS = 4


class PackedRuntimeSequencePrimitive(SequencePrimitive):
    in_projection: PackedLinearProjection

    def step_projected(
        self,
        state: torch.Tensor,
        *projected_chunks: torch.Tensor,
    ) -> SequencePrimitiveStepResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        return self.step_projected(state, *self.in_projection(inputs))

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> PackedChunksRuntimePlan:
        return PackedChunksRuntimePlan(chunks=self.in_projection(inputs))

    def scan_with_runtime_plan(
        self,
        runtime_plan: PackedChunksRuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            step = self.step_projected(
                state,
                *(chunk[:, position, ...] for chunk in runtime_plan.chunks),
            )
            state = step.next_state
            outputs[:, position, :] = step.emitted_output
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        return SequencePrimitive.scan(self, inputs, initial_state=initial_state)


class ComplexStatePackedRuntimeSequencePrimitive(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model * 2
        self.state_readout_projection = build_linear(self.state_width, d_model)

    def _emit_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_readout_projection(state)


class HierarchicalStatePackedRuntimeSequencePrimitive(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, per_level_width: int, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__()
        self.d_model = d_model
        self.levels = levels
        self.state_width = per_level_width

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.levels, self.state_width, device=device, dtype=dtype)

    def _emit_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return _hierarchical_top_level(state)


class HierarchicalComplexStatePackedRuntimeSequencePrimitive(HierarchicalStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__(d_model, per_level_width=d_model * 2, levels=levels)
        self.state_readout_projection = build_linear(self.state_width, d_model)

    def _emit_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_readout_projection(_hierarchical_top_level(state))


class ContractiveSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(
            d_model,
            (d_model, d_model),
        )
        self.state_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        gate_inputs, projected_inputs = self.in_projection(inputs)
        gate = gated_sigmoid(gate_inputs)
        mix = self.state_projection(state) + projected_inputs
        next_state = gate * mix + one_minus(gate) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class ContractiveRuntimeSequenceMixer(ContractiveSequenceMixer):
    def prepare_runtime_plan(self, inputs: torch.Tensor) -> ContractiveRuntimePlan:
        gate_inputs, projected_inputs = self.in_projection(inputs)
        return ContractiveRuntimePlan(
            gates=gated_sigmoid(gate_inputs),
            projected_inputs=projected_inputs,
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: ContractiveRuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            gate = runtime_plan.gates[:, position, :]
            mix = self.state_projection(state) + runtime_plan.projected_inputs[:, position, :]
            state = gate * mix + one_minus(gate) * state
            outputs[:, position, :] = state
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P1FractalHybridSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(
            d_model,
            (d_model, d_model),
        )
        self.state_projection = build_linear(d_model, d_model)

    def _clamp_value(self, state: torch.Tensor) -> torch.Tensor:
        return gated_sigmoid(_row_l2_norm(state)).mul(-0.225).add(0.75)

    def _squared_update(self, state: torch.Tensor, projected_inputs: torch.Tensor) -> torch.Tensor:
        squared = _clamp_symmetric_by_row(state * state, self._clamp_value(state))
        return self.state_projection(state) + projected_inputs + squared

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        gate_inputs, projected_inputs = self.in_projection(inputs)
        gate = gated_sigmoid(gate_inputs)
        main_update = self._squared_update(state, projected_inputs)
        next_state = gate * main_update + one_minus(gate) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class P1FractalHybridRuntimeSequenceMixer(P1FractalHybridSequenceMixer):
    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P1FractalHybridRuntimePlan:
        gate_inputs, projected_inputs = self.in_projection(inputs)
        return P1FractalHybridRuntimePlan(
            gates=gated_sigmoid(gate_inputs),
            projected_inputs=projected_inputs,
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: P1FractalHybridRuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            gate = runtime_plan.gates[:, position, :]
            main_update = self._squared_update(
                state,
                runtime_plan.projected_inputs[:, position, :],
            )
            state = gate * main_update + one_minus(gate) * state
            outputs[:, position, :] = state
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P1FractalHybridCompositeSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model))
        self.state_projection = build_linear(d_model, d_model)

    def _clamp_value(self, state: torch.Tensor) -> torch.Tensor:
        return gated_sigmoid(_row_l2_norm(state)).mul(-0.225).add(0.75)

    def _squared_update(self, state: torch.Tensor, projected_inputs: torch.Tensor) -> torch.Tensor:
        squared = _clamp_symmetric_by_row(state * state, self._clamp_value(state))
        return self.state_projection(state) + projected_inputs + squared

    def _contractive_inner(self, state: torch.Tensor, gate: torch.Tensor, projected_inputs: torch.Tensor) -> torch.Tensor:
        mix = self.state_projection(state) + projected_inputs
        return gate * mix + one_minus(gate) * state

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        projected_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        outer_gate = gated_sigmoid(gate_inputs)
        base_state = self._contractive_inner(state, outer_gate, projected_inputs)
        main_update = self._squared_update(base_state, projected_inputs)
        next_state = outer_gate * main_update + one_minus(outer_gate) * base_state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class P1FractalHybridDynGateSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model))
        self.state_projection = build_linear(d_model, d_model)

    def _clamp_value(self, state: torch.Tensor) -> torch.Tensor:
        return gated_sigmoid(_row_l2_norm(state)).mul(-0.225).add(0.75)

    def _squared_update(self, state: torch.Tensor, projected_inputs: torch.Tensor) -> torch.Tensor:
        squared = _clamp_symmetric_by_row(state * state, self._clamp_value(state))
        return self.state_projection(state) + projected_inputs + squared

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        projected_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        gate_cap = torch.full_like(gate_inputs, 0.95 - 0.25 * LEGACY_ROOT_DEPTH_FRACTION)
        tuned_gate = _clamp_max_by_row(gated_sigmoid(gate_inputs), gate_cap)
        main_update = self._squared_update(state, projected_inputs)
        next_state = tuned_gate * main_update + one_minus(tuned_gate) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class P2MandelbrotSequenceMixer(ComplexStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.in_projection = PackedLinearProjection(d_model, (self.state_width, self.state_width))

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        c_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        clamp_value = _repeat_row_value(gated_sigmoid(_row_l2_norm(state)).mul(-0.225).add(0.9), self.state_width)
        gate = _clamp_max_by_row(gated_sigmoid(gate_inputs), clamp_value)
        next_state = gate * _complex_square(state) + c_inputs
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class B1FractalGatedSequenceMixer(ComplexStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.in_projection = PackedLinearProjection(d_model, (self.state_width, self.state_width))

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        c_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        gate = gated_sigmoid(gate_inputs)
        main_update = gate * (_complex_square(state) + c_inputs) + one_minus(gate) * state
        alpha = _repeat_row_value(gated_sigmoid(_row_l2_norm(state)).mul(0.08).add(0.02), self.state_width)
        next_state = alpha * main_update + one_minus(alpha) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class JuliaRecursiveEscapeSequenceMixer(ComplexStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model)
        self.in_projection = PackedLinearProjection(d_model, (self.state_width,))

    def step_projected(
        self,
        state: torch.Tensor,
        c_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        next_state = _complex_square(state) + c_inputs
        escape_radius = _repeat_row_value(
            gated_sigmoid(_row_l2_norm(next_state))
            .mul(1.0 - 0.2 * LEGACY_ROOT_DEPTH_FRACTION)
            .add(2.0),
            self.state_width,
        )
        bounded_state = _clamp_symmetric_by_row(next_state, escape_radius)
        return SequencePrimitiveStepResult(
            next_state=bounded_state,
            emitted_output=self._emit_from_state(bounded_state),
        )


class LogisticChaoticMapSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model))

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        r_inputs, gate_inputs = self.in_projection(inputs)
        return self.step_projected(state, r_inputs, gate_inputs, inputs)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> PackedChunksRuntimePlan:
        return PackedChunksRuntimePlan(chunks=(*self.in_projection(inputs), inputs))

    def step_projected(
        self,
        state: torch.Tensor,
        r_inputs: torch.Tensor,
        gate_inputs: torch.Tensor,
        raw_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        bounded_state = gated_sigmoid(state)
        r_t = (
            gated_sigmoid(r_inputs)
            .mul(3.95 - 3.6)
            .add(3.6)
            .clamp(3.6, 3.95)
        )
        gate = gated_sigmoid(gate_inputs)
        next_state = r_t * bounded_state * one_minus(bounded_state) + gate * raw_inputs
        alpha = _repeat_row_value(gated_sigmoid(_row_l2_norm(state)).mul(0.08).add(0.02), self.d_model)
        next_state = alpha * next_state + one_minus(alpha) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class GeneralizedMobiusSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model, d_model, d_model))

    def step_projected(
        self,
        state: torch.Tensor,
        a_inputs: torch.Tensor,
        b_inputs: torch.Tensor,
        c_inputs: torch.Tensor,
        d_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        a = a_inputs.tanh()
        b = b_inputs.tanh().mul(0.5)
        c = c_inputs.tanh()
        d = d_inputs.tanh().mul(0.5).add(1.0)
        norm = _row_l2_norm(state)
        epsilon = (norm.mul(1.0e-5).add(1.0e-6) * norm.mul(0.5).add(1.0)).repeat(1, self.d_model)
        numerator = a * state + b
        denominator = c * state + d + epsilon
        next_state = numerator * denominator.reciprocal()
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class IfsSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (NUM_IFS_MAPS,))
        self.a_diag = nn.Parameter(torch.empty(NUM_IFS_MAPS, d_model))
        self.b_bias = nn.Parameter(torch.empty(NUM_IFS_MAPS, d_model))
        nn.init.uniform_(self.a_diag, -0.1, 0.1)
        nn.init.uniform_(self.b_bias, -0.1, 0.1)

    def step_projected(
        self,
        state: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        probs = torch.softmax(router_logits, dim=-1)
        next_state = torch.zeros_like(state)
        radius = 0.98 - 0.03 * LEGACY_ROOT_DEPTH_FRACTION
        for map_index in range(NUM_IFS_MAPS):
            a = self.a_diag[map_index].unsqueeze(0).expand_as(state) * radius
            b = self.b_bias[map_index].unsqueeze(0).expand_as(state)
            weight = probs[:, map_index : map_index + 1].expand_as(state)
            candidate = a * state + b
            next_state = next_state + weight * candidate
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class MandelboxRecursiveSequenceMixer(PackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.in_projection = PackedLinearProjection(d_model, (d_model,))

    def _mandelbox_step(
        self,
        state: torch.Tensor,
        *,
        escape_radius: torch.Tensor,
        drive: torch.Tensor,
    ) -> torch.Tensor:
        folded = _clamp_symmetric_by_row(state, escape_radius) * 2.0 - state
        return folded * 2.0 + drive

    def step_projected(
        self,
        state: torch.Tensor,
        drive_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        drive = torch.tanh(drive_inputs).mul(0.1)
        escape_radius = torch.full_like(state, 1.8 - 0.45 * LEGACY_ROOT_DEPTH_FRACTION)
        current = state + drive
        for _ in range(LEGACY_ROOT_ITERATIONS):
            current = self._mandelbox_step(current, escape_radius=escape_radius, drive=drive)
        next_state = state.mul(0.7) + current.mul(0.3)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class P3HierarchicalSequenceMixer(HierarchicalStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__(d_model, per_level_width=d_model, levels=levels)
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model, d_model, d_model))
        self.compressor = build_linear(d_model, d_model)

    def step_projected(
        self,
        state: torch.Tensor,
        u_inputs: torch.Tensor,
        alpha_inputs: torch.Tensor,
        beta_inputs: torch.Tensor,
        gamma_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        alpha = gated_sigmoid(alpha_inputs)
        beta = gated_sigmoid(beta_inputs)
        gamma = gated_sigmoid(gamma_inputs)
        next_levels: list[torch.Tensor] = []
        for level in range(self.levels):
            prev = state[:, level, :]
            next_level = alpha * prev + beta * u_inputs
            if level > 0:
                next_level = next_level + gamma * self.compressor(next_levels[level - 1])
            next_levels.append(next_level)
        next_state = torch.stack(next_levels, dim=1)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class B2StableHierarchicalSequenceMixer(HierarchicalStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__(d_model, per_level_width=d_model, levels=levels)
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model, d_model))
        self.state_projection = build_linear(d_model, d_model)
        self.compressor = build_linear(d_model, d_model)

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        u_inputs: torch.Tensor,
        gamma_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        gate = gated_sigmoid(gate_inputs)
        gamma = gated_sigmoid(gamma_inputs)
        next_levels: list[torch.Tensor] = []
        for level in range(self.levels):
            prev = state[:, level, :]
            base = gate * (self.state_projection(prev) + u_inputs) + one_minus(gate) * prev
            if level > 0:
                base = base + gamma * self.compressor(next_levels[level - 1])
            next_levels.append(base)
        next_state = torch.stack(next_levels, dim=1)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class B3FractalHierarchicalSequenceMixer(HierarchicalComplexStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__(d_model, levels=levels)
        self.in_projection = PackedLinearProjection(d_model, (self.state_width, self.state_width))
        self.compressor = build_linear(self.state_width, self.state_width)

    def step_projected(
        self,
        state: torch.Tensor,
        c_inputs: torch.Tensor,
        gamma_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        gamma = gated_sigmoid(gamma_inputs)
        radius = 0.98 - 0.03 * LEGACY_ROOT_DEPTH_FRACTION
        next_levels: list[torch.Tensor] = []
        for level in range(self.levels):
            prev = state[:, level, :]
            next_level = _complex_square(prev).mul(radius) + c_inputs
            if level > 0:
                next_level = next_level + gamma * self.compressor(next_levels[level - 1])
            next_levels.append(next_level)
        next_state = torch.stack(next_levels, dim=1)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class B4UniversalSequenceMixer(HierarchicalComplexStatePackedRuntimeSequencePrimitive):
    def __init__(self, d_model: int, *, levels: int = LEGACY_HIERARCHICAL_LEVELS) -> None:
        super().__init__(d_model, levels=levels)
        self.in_projection = PackedLinearProjection(d_model, (self.state_width, self.state_width, self.state_width))
        self.compressor = build_linear(self.state_width, self.state_width)

    def step_projected(
        self,
        state: torch.Tensor,
        gate_inputs: torch.Tensor,
        c_inputs: torch.Tensor,
        gamma_inputs: torch.Tensor,
    ) -> SequencePrimitiveStepResult:
        gate = gated_sigmoid(gate_inputs)
        gamma = gated_sigmoid(gamma_inputs)
        next_levels: list[torch.Tensor] = []
        for level in range(self.levels):
            prev = state[:, level, :]
            base = gate * (_complex_square(prev) + c_inputs) + one_minus(gate) * prev
            if level > 0:
                base = base + gamma * self.compressor(next_levels[level - 1])
            alpha = _repeat_row_value(gated_sigmoid(_row_l2_norm(prev)).mul(0.08).add(0.02), self.state_width)
            next_level = alpha * base + one_minus(alpha) * prev
            next_levels.append(next_level)
        next_state = torch.stack(next_levels, dim=1)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=self._emit_from_state(next_state))


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, width: int, *, kernel_size: int = 4, activation: bool = True) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"causal depthwise conv kernel_size must be positive, got {kernel_size}")
        self.width = width
        self.kernel_size = kernel_size
        self.activation = activation
        self.weight = nn.Parameter(torch.zeros(width, 1, kernel_size))
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with record_function("path1.primitive.runtime.short_causal_conv"):
            conv_inputs = F.pad(inputs.transpose(1, 2), (self.kernel_size - 1, 0))
            mixed = F.conv1d(conv_inputs, self.weight, groups=self.width).transpose(1, 2)
        return F.silu(mixed) if self.activation else mixed


class P20RotaryStateOutputSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2_0 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.state_transform_mode = state_transform_mode
        self.in_projection = PackedLinearProjection(
            d_model,
            (d_model, d_model // 2, d_model, d_model),
        )
        self.state_transform_projection = _build_state_transform_projection(d_model, state_transform_mode)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gate = gated_sigmoid(update_gate_inputs)
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            angle_inputs,
        )
        candidate = torch.tanh(candidate_inputs)
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(output_gate_inputs) * next_state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P20RotaryStateOutputRuntimeSequenceMixer(P20RotaryStateOutputSequenceMixer):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__(d_model, state_transform_mode=state_transform_mode)
        self._compiled_scan_impl = None

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P20RuntimePlan:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gates = gated_sigmoid(update_gate_inputs)
        angle_cos, angle_sin = _rotary_runtime_components(angle_inputs)
        return P20RuntimePlan(
            update_gates=update_gates,
            retain_gates=one_minus(update_gates),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=torch.tanh(candidate_inputs),
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        super().configure_runtime_policy(
            compile_mode=compile_mode,
            primitive_runtime_backend=primitive_runtime_backend,
        )
        if compile_mode is None:
            self._compiled_scan_impl = None
            return
        self._compiled_scan_impl = torch.compile(
            self._scan_runtime_impl,
            mode=compile_mode,
            fullgraph=False,
        )

    def _scan_runtime_impl(
        self,
        state: torch.Tensor,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        output_gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emitted_outputs: list[torch.Tensor] = []
        seq_len = update_gates.shape[1]
        for position in range(seq_len):
            update_gate = update_gates[:, position, :]
            retain_gate = retain_gates[:, position, :]
            cos = angle_cos[:, position, :]
            sin = angle_sin[:, position, :]
            candidate = candidates[:, position, :]
            output_gate = output_gates[:, position, :]
            transformed_state = _rotate_state_pairs_with_trig(
                self.state_transform_projection(state),
                cos=cos,
                sin=sin,
            )
            state = update_gate * transformed_state + retain_gate * candidate
            emitted_outputs.append(output_gate * state)
        return torch.stack(emitted_outputs, dim=1), state

    def _scan_runtime_impl_profiled(
        self,
        state: torch.Tensor,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        output_gates: torch.Tensor,
        outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = update_gates.shape[1]
        for position in range(seq_len):
            update_gate = update_gates[:, position, :]
            retain_gate = retain_gates[:, position, :]
            cos = angle_cos[:, position, :]
            sin = angle_sin[:, position, :]
            candidate = candidates[:, position, :]
            output_gate = output_gates[:, position, :]
            with record_function("path1.primitive.runtime.state_transform_projection"):
                projected_state = self.state_transform_projection(state)
            with record_function("path1.primitive.runtime.rotary_apply"):
                transformed_state = _rotate_state_pairs_with_trig(
                    projected_state,
                    cos=cos,
                    sin=sin,
                )
            with record_function("path1.primitive.runtime.state_update"):
                state = update_gate * transformed_state + retain_gate * candidate
            with record_function("path1.primitive.runtime.output_readout"):
                outputs[:, position, :] = output_gate * state
        return outputs, state

    def _scan_runtime_impl_triton_profiled(
        self,
        state: torch.Tensor,
        update_gates: torch.Tensor,
        retain_gates: torch.Tensor,
        angle_cos: torch.Tensor,
        angle_sin: torch.Tensor,
        candidates: torch.Tensor,
        output_gates: torch.Tensor,
        outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._triton_backend is None:
            raise RuntimeError("P20 Triton runtime requested without an initialized Triton backend")
        if (
            self.state_transform_mode is PrimitiveStateTransformMode.DENSE
            and isinstance(self.state_transform_projection, nn.Linear)
        ):
            with record_function("path1.primitive.runtime.triton_sequence_scan"):
                return self._triton_backend.scan_p20_dense_sequence(
                    update_gate=update_gates,
                    retain_gate=retain_gates,
                    angle_cos=angle_cos,
                    angle_sin=angle_sin,
                    candidate=candidates,
                    output_gate=output_gates,
                    initial_state=state,
                    transform_weight=self.state_transform_projection.weight,
                    transform_bias=self.state_transform_projection.bias,
                )
        if (
            self.state_transform_mode
            in {
                PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
            }
            and isinstance(self.state_transform_projection, BlockDiagonalLinear)
        ):
            with record_function("path1.primitive.runtime.triton_sequence_scan"):
                return self._triton_backend.scan_p20_block_diagonal_sequence(
                    update_gate=update_gates,
                    retain_gate=retain_gates,
                    angle_cos=angle_cos,
                    angle_sin=angle_sin,
                    candidate=candidates,
                    output_gate=output_gates,
                    initial_state=state,
                    transform_weight=self.state_transform_projection.weight,
                    transform_bias=self.state_transform_projection.bias,
                )
        seq_len = update_gates.shape[1]
        for position in range(seq_len):
            update_gate = update_gates[:, position, :]
            retain_gate = retain_gates[:, position, :]
            cos = angle_cos[:, position, :]
            sin = angle_sin[:, position, :]
            candidate = candidates[:, position, :]
            output_gate = output_gates[:, position, :]
            with record_function("path1.primitive.runtime.state_transform_projection"):
                projected_state = self.state_transform_projection(state)
            with record_function("path1.primitive.runtime.rotary_apply"):
                transformed_state = _rotate_state_pairs_with_trig(
                    projected_state,
                    cos=cos,
                    sin=sin,
                )
            with record_function("path1.primitive.runtime.triton_state_update"):
                state, emitted = self._triton_backend.fused_p20_update_readout(
                    update_gate=update_gate,
                    retain_gate=retain_gate,
                    transformed_state=transformed_state,
                    candidate=candidate,
                    output_gate=output_gate,
                )
            with record_function("path1.primitive.runtime.output_readout"):
                outputs[:, position, :] = emitted
        return outputs, state

    def scan_with_runtime_plan(
        self,
        runtime_plan: P20RuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        if self._primitive_runtime_backend == "triton":
            outputs, state = self._scan_runtime_impl_triton_profiled(
                state,
                runtime_plan.update_gates,
                runtime_plan.retain_gates,
                runtime_plan.angle_cos,
                runtime_plan.angle_sin,
                runtime_plan.candidates,
                runtime_plan.output_gates,
                outputs,
            )
        else:
            scan_impl = self._compiled_scan_impl or self._scan_runtime_impl_profiled
            if self._compiled_scan_impl is not None:
                outputs, state = scan_impl(
                    state,
                    runtime_plan.update_gates,
                    runtime_plan.retain_gates,
                    runtime_plan.angle_cos,
                    runtime_plan.angle_sin,
                    runtime_plan.candidates,
                    runtime_plan.output_gates,
                )
            else:
                outputs, state = scan_impl(
                    state,
                    runtime_plan.update_gates,
                    runtime_plan.retain_gates,
                    runtime_plan.angle_cos,
                    runtime_plan.angle_sin,
                    runtime_plan.candidates,
                    runtime_plan.output_gates,
                    outputs,
                )
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P20GatedDeltaRoleSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        local_kernel_size: int = 4,
        state_heads: int | None = None,
    ) -> None:
        super().__init__()
        resolved_heads = state_heads or (4 if d_model % 4 == 0 else 1)
        if d_model % resolved_heads != 0:
            raise ValueError(
                f"p2_0_gdn_role requires d_model divisible by state_heads, got {d_model} and {resolved_heads}"
            )
        self.d_model = d_model
        self.state_heads = resolved_heads
        self.head_dim = d_model // resolved_heads
        self.state_width = self.state_heads * self.head_dim * self.head_dim
        self.in_projection = PackedLinearProjection(d_model, (d_model, d_model, d_model), bias=False)
        self.control_projection = PackedLinearProjection(
            d_model,
            (self.state_heads, self.state_heads, d_model),
            bias=True,
        )
        self.q_local = CausalDepthwiseConv1d(d_model, kernel_size=local_kernel_size)
        self.k_local = CausalDepthwiseConv1d(d_model, kernel_size=local_kernel_size)
        self.v_local = CausalDepthwiseConv1d(d_model, kernel_size=local_kernel_size)
        self.output_norm = SimpleRmsNorm(self.head_dim)
        self.output_projection = build_linear(d_model, d_model)
        self.readout_ramp_logit = nn.Parameter(torch.full((d_model,), -2.1972246))
        self._compiled_scan_impl = None
        self._init_gdn_role_parameters()

    def _init_gdn_role_parameters(self) -> None:
        if self.control_projection.bias is not None:
            with torch.no_grad():
                alpha_bias, beta_bias, output_gate_bias = self.control_projection.bias.split(
                    self.control_projection.split_sizes,
                    dim=0,
                )
                alpha_bias.fill_(1.5)
                beta_bias.fill_(-2.0)
                output_gate_bias.fill_(-1.5)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.state_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(*tensor.shape[:-1], self.state_heads, self.head_dim)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P20GdnRoleRuntimePlan:
        q_inputs, k_inputs, v_inputs = self.in_projection(inputs)
        alpha_inputs, beta_inputs, output_gate_inputs = self.control_projection(inputs)
        queries = F.normalize(self._reshape_heads(self.q_local(q_inputs)), p=2.0, dim=-1, eps=1.0e-6)
        keys = F.normalize(self._reshape_heads(self.k_local(k_inputs)), p=2.0, dim=-1, eps=1.0e-6)
        values = self._reshape_heads(self.v_local(v_inputs))
        alpha_gates = gated_sigmoid(alpha_inputs).mul(0.98).add(0.01)
        beta_gates = gated_sigmoid(beta_inputs)
        return P20GdnRoleRuntimePlan(
            queries=queries,
            keys=keys,
            values=values,
            alpha_gates=alpha_gates,
            beta_gates=beta_gates,
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def _scan_runtime_impl(
        self,
        state: torch.Tensor,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        alpha_gates: torch.Tensor,
        beta_gates: torch.Tensor,
        output_gates: torch.Tensor,
        outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = queries.shape[1]
        readout_ramp = gated_sigmoid(self.readout_ramp_logit).view(1, 1, self.d_model)
        for position in range(seq_len):
            query = queries[:, position, :, :]
            key = keys[:, position, :, :]
            value = values[:, position, :, :]
            alpha = alpha_gates[:, position, :].view(-1, self.state_heads, 1, 1)
            beta = beta_gates[:, position, :].view(-1, self.state_heads, 1, 1)
            old_value = torch.einsum("bhvk,bhk->bhv", state, key)
            erase = torch.einsum("bhv,bhk->bhvk", old_value, key)
            write = torch.einsum("bhv,bhk->bhvk", value, key)
            state = alpha * (state - beta * erase) + beta * write
            read = torch.einsum("bhvk,bhk->bhv", state, query)
            read = self.output_norm(read).reshape(read.shape[0], self.d_model)
            projected = self.output_projection(read).unsqueeze(1)
            gate = output_gates[:, position : position + 1, :]
            outputs[:, position : position + 1, :] = gate * projected * readout_ramp
        return outputs, state

    def scan_with_runtime_plan(
        self,
        runtime_plan: P20GdnRoleRuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        if self._primitive_runtime_backend == "triton":
            raise NotImplementedError("p2-0-gdn-role currently has a torch runtime only")
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        scan_impl = self._compiled_scan_impl or self._scan_runtime_impl
        outputs, state = scan_impl(
            state,
            runtime_plan.queries,
            runtime_plan.keys,
            runtime_plan.values,
            runtime_plan.alpha_gates,
            runtime_plan.beta_gates,
            runtime_plan.output_gates,
            outputs,
        )
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        super().configure_runtime_policy(
            compile_mode=None,
            primitive_runtime_backend=primitive_runtime_backend,
        )
        if compile_mode is None:
            self._compiled_scan_impl = None
            return
        self._compiled_scan_impl = torch.compile(
            self._scan_runtime_impl,
            mode=compile_mode,
            fullgraph=False,
        )

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        runtime_plan = self.prepare_runtime_plan(inputs.unsqueeze(1))
        result = self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=1,
            initial_state=state,
        )
        return SequencePrimitiveStepResult(
            next_state=result.final_state,
            emitted_output=result.emitted_outputs[:, 0, :],
        )


class P2RotaryReadoutSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.state_transform_mode = state_transform_mode
        self.in_projection = PackedLinearProjection(
            d_model,
            (d_model, d_model // 2, d_model, d_model),
        )
        self.state_transform_projection = _build_state_transform_projection(d_model, state_transform_mode)
        self.output_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gate = gated_sigmoid(update_gate_inputs)
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            angle_inputs,
        )
        candidate = torch.tanh(candidate_inputs)
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(output_gate_inputs) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P2RotaryReadoutRuntimeSequenceMixer(P2RotaryReadoutSequenceMixer):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__(d_model, state_transform_mode=state_transform_mode)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P2RuntimePlan:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gates = gated_sigmoid(update_gate_inputs)
        angle_cos, angle_sin = _rotary_runtime_components(angle_inputs)
        return P2RuntimePlan(
            update_gates=update_gates,
            retain_gates=one_minus(update_gates),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=torch.tanh(candidate_inputs),
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: P2RuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        if self._primitive_runtime_backend == "triton":
            if self._triton_backend is None:
                raise RuntimeError("P2 Triton runtime requested without an initialized Triton backend")
            if (
                self.state_transform_mode
                in {
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                }
                and isinstance(self.state_transform_projection, BlockDiagonalLinear)
            ):
                with record_function("path1.primitive.runtime.triton_sequence_scan"):
                    state_outputs, state = self._triton_backend.scan_rotary_state_block_diagonal_sequence(
                        update_gate=runtime_plan.update_gates,
                        retain_gate=runtime_plan.retain_gates,
                        angle_cos=runtime_plan.angle_cos,
                        angle_sin=runtime_plan.angle_sin,
                        candidate=runtime_plan.candidates,
                        initial_state=state,
                        transform_weight=self.state_transform_projection.weight,
                        transform_bias=self.state_transform_projection.bias,
                    )
                with record_function("path1.primitive.runtime.output_projection"):
                    projected_output = self.output_projection(state_outputs)
                with record_function("path1.primitive.runtime.output_readout"):
                    outputs = runtime_plan.output_gates * projected_output
                return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)
            if (
                self.state_transform_mode is PrimitiveStateTransformMode.DENSE
                and isinstance(self.state_transform_projection, nn.Linear)
            ):
                raise NotImplementedError(
                    "primitive_runtime_backend=triton for primitive profile p2 currently requires a block-diagonal state transform"
                )
            raise NotImplementedError(
                f"primitive_runtime_backend=triton is not implemented for P2 state transform mode {self.state_transform_mode.value}"
            )

        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        update_gates = runtime_plan.update_gates.unbind(dim=1)
        retain_gates = runtime_plan.retain_gates.unbind(dim=1)
        angle_cos = runtime_plan.angle_cos.unbind(dim=1)
        angle_sin = runtime_plan.angle_sin.unbind(dim=1)
        candidates = runtime_plan.candidates.unbind(dim=1)
        output_gates = runtime_plan.output_gates.unbind(dim=1)
        for position, (
            update_gate,
            retain_gate,
            cos,
            sin,
            candidate,
            output_gate,
        ) in enumerate(zip(update_gates, retain_gates, angle_cos, angle_sin, candidates, output_gates)):
            with record_function("path1.primitive.runtime.state_transform_projection"):
                projected_state = self.state_transform_projection(state)
            with record_function("path1.primitive.runtime.rotary_apply"):
                transformed_state = _rotate_state_pairs_with_trig(
                    projected_state,
                    cos=cos,
                    sin=sin,
                )
            with record_function("path1.primitive.runtime.state_update"):
                state = update_gate * transformed_state + retain_gate * candidate
            with record_function("path1.primitive.runtime.output_projection"):
                projected_output = self.output_projection(state)
            with record_function("path1.primitive.runtime.output_readout"):
                outputs[:, position, :] = output_gate * projected_output
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P23RotaryCarryBlendReadoutSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2_3 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.state_transform_mode = state_transform_mode
        self.in_projection = PackedLinearProjection(
            d_model,
            (d_model, d_model, d_model // 2, d_model, d_model),
        )
        self.state_transform_projection = _build_state_transform_projection(d_model, state_transform_mode)
        self.carry_state_projection = build_linear(d_model, d_model)
        self.output_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        (
            update_gate_inputs,
            dynamics_mix_gate_inputs,
            angle_inputs,
            candidate_inputs,
            output_gate_inputs,
        ) = self.in_projection(inputs)
        update_gate = gated_sigmoid(update_gate_inputs)
        dynamics_mix_gate = gated_sigmoid(dynamics_mix_gate_inputs)
        rotated_state = rotate_state_pairs(
            self.state_transform_projection(state),
            angle_inputs,
        )
        carried_state = torch.tanh(self.carry_state_projection(state))
        transformed_state = dynamics_mix_gate * rotated_state + one_minus(dynamics_mix_gate) * carried_state
        candidate = torch.tanh(candidate_inputs)
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(output_gate_inputs) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P23RotaryCarryBlendReadoutRuntimeSequenceMixer(P23RotaryCarryBlendReadoutSequenceMixer):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__(d_model, state_transform_mode=state_transform_mode)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P23RuntimePlan:
        (
            update_gate_inputs,
            dynamics_mix_gate_inputs,
            angle_inputs,
            candidate_inputs,
            output_gate_inputs,
        ) = self.in_projection(inputs)
        update_gates = gated_sigmoid(update_gate_inputs)
        dynamics_mix_gates = gated_sigmoid(dynamics_mix_gate_inputs)
        angle_cos, angle_sin = _rotary_runtime_components(angle_inputs)
        return P23RuntimePlan(
            update_gates=update_gates,
            retain_gates=one_minus(update_gates),
            dynamics_mix_gates=dynamics_mix_gates,
            dynamics_keep_gates=one_minus(dynamics_mix_gates),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=torch.tanh(candidate_inputs),
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: P23RuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            rotated_state = _rotate_state_pairs_with_trig(
                self.state_transform_projection(state),
                cos=runtime_plan.angle_cos[:, position, :],
                sin=runtime_plan.angle_sin[:, position, :],
            )
            carried_state = torch.tanh(self.carry_state_projection(state))
            transformed_state = (
                runtime_plan.dynamics_mix_gates[:, position, :] * rotated_state
                + runtime_plan.dynamics_keep_gates[:, position, :] * carried_state
            )
            state = (
                runtime_plan.update_gates[:, position, :] * transformed_state
                + runtime_plan.retain_gates[:, position, :] * runtime_plan.candidates[:, position, :]
            )
            outputs[:, position, :] = runtime_plan.output_gates[:, position, :] * self.output_projection(state)
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P21WideLatentSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model * 2
        self.state_transform_mode = state_transform_mode
        self.in_projection = PackedLinearProjection(
            d_model,
            (self.state_width, self.state_width // 2, self.state_width, d_model),
        )
        self.state_transform_projection = _build_state_transform_projection(self.state_width, state_transform_mode)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gate = gated_sigmoid(update_gate_inputs)
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            angle_inputs,
        )
        candidate = torch.tanh(candidate_inputs)
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        readout = leading_state_slice(next_state, self.d_model)
        emitted_output = gated_sigmoid(output_gate_inputs) * readout
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P21WideLatentRuntimeSequenceMixer(P21WideLatentSequenceMixer):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__(d_model, state_transform_mode=state_transform_mode)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P21RuntimePlan:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gates = gated_sigmoid(update_gate_inputs)
        angle_cos, angle_sin = _rotary_runtime_components(angle_inputs)
        return P21RuntimePlan(
            update_gates=update_gates,
            retain_gates=one_minus(update_gates),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=torch.tanh(candidate_inputs),
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: P21RuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        if self._primitive_runtime_backend == "triton":
            if self._triton_backend is None:
                raise RuntimeError("P2.1 Triton runtime requested without an initialized Triton backend")
            if (
                self.state_transform_mode
                in {
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                }
                and isinstance(self.state_transform_projection, BlockDiagonalLinear)
            ):
                with record_function("path1.primitive.runtime.triton_sequence_scan"):
                    state_outputs, state = self._triton_backend.scan_rotary_state_block_diagonal_sequence(
                        update_gate=runtime_plan.update_gates,
                        retain_gate=runtime_plan.retain_gates,
                        angle_cos=runtime_plan.angle_cos,
                        angle_sin=runtime_plan.angle_sin,
                        candidate=runtime_plan.candidates,
                        initial_state=state,
                        transform_weight=self.state_transform_projection.weight,
                        transform_bias=self.state_transform_projection.bias,
                    )
                with record_function("path1.primitive.runtime.output_readout"):
                    outputs = runtime_plan.output_gates * leading_state_slice(state_outputs, self.d_model)
                return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)
            if (
                self.state_transform_mode is PrimitiveStateTransformMode.DENSE
                and isinstance(self.state_transform_projection, nn.Linear)
            ):
                raise NotImplementedError(
                    "primitive_runtime_backend=triton for primitive profile p2-1 currently requires a block-diagonal state transform"
                )
            raise NotImplementedError(
                f"primitive_runtime_backend=triton is not implemented for P2.1 state transform mode {self.state_transform_mode.value}"
            )
        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            transformed_state = _rotate_state_pairs_with_trig(
                self.state_transform_projection(state),
                cos=runtime_plan.angle_cos[:, position, :],
                sin=runtime_plan.angle_sin[:, position, :],
            )
            state = (
                runtime_plan.update_gates[:, position, :] * transformed_state
                + runtime_plan.retain_gates[:, position, :] * runtime_plan.candidates[:, position, :]
            )
            outputs[:, position, :] = runtime_plan.output_gates[:, position, :] * leading_state_slice(
                state, self.d_model
            )
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


class P22WideLatentReadoutSequenceMixer(SequencePrimitive):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model * 2
        self.state_transform_mode = state_transform_mode
        self.in_projection = PackedLinearProjection(
            d_model,
            (self.state_width, self.state_width // 2, self.state_width, d_model),
        )
        self.state_transform_projection = _build_state_transform_projection(self.state_width, state_transform_mode)
        self.output_projection = build_linear(self.state_width, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gate = gated_sigmoid(update_gate_inputs)
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            angle_inputs,
        )
        candidate = torch.tanh(candidate_inputs)
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(output_gate_inputs) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P22WideLatentReadoutRuntimeSequenceMixer(P22WideLatentReadoutSequenceMixer):
    def __init__(
        self,
        d_model: int,
        *,
        state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
    ) -> None:
        super().__init__(d_model, state_transform_mode=state_transform_mode)

    def prepare_runtime_plan(self, inputs: torch.Tensor) -> P22RuntimePlan:
        update_gate_inputs, angle_inputs, candidate_inputs, output_gate_inputs = self.in_projection(inputs)
        update_gates = gated_sigmoid(update_gate_inputs)
        angle_cos, angle_sin = _rotary_runtime_components(angle_inputs)
        return P22RuntimePlan(
            update_gates=update_gates,
            retain_gates=one_minus(update_gates),
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            candidates=torch.tanh(candidate_inputs),
            output_gates=gated_sigmoid(output_gate_inputs),
        )

    def scan_with_runtime_plan(
        self,
        runtime_plan: P22RuntimePlan,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        seq_len: int,
        initial_state: torch.Tensor | None = None,
    ) -> SequencePrimitiveScanResult:
        state = initial_state if initial_state is not None else self.init_state(batch_size, device, dtype)
        if self._primitive_runtime_backend == "triton":
            if self._triton_backend is None:
                raise RuntimeError("P2.2 Triton runtime requested without an initialized Triton backend")
            if (
                self.state_transform_mode
                in {
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                    PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                }
                and isinstance(self.state_transform_projection, BlockDiagonalLinear)
            ):
                with record_function("path1.primitive.runtime.triton_sequence_scan"):
                    state_outputs, state = self._triton_backend.scan_rotary_state_block_diagonal_sequence(
                        update_gate=runtime_plan.update_gates,
                        retain_gate=runtime_plan.retain_gates,
                        angle_cos=runtime_plan.angle_cos,
                        angle_sin=runtime_plan.angle_sin,
                        candidate=runtime_plan.candidates,
                        initial_state=state,
                        transform_weight=self.state_transform_projection.weight,
                        transform_bias=self.state_transform_projection.bias,
                    )
                with record_function("path1.primitive.runtime.output_projection"):
                    projected_output = self.output_projection(state_outputs)
                with record_function("path1.primitive.runtime.output_readout"):
                    outputs = runtime_plan.output_gates * projected_output
                return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)
            if (
                self.state_transform_mode is PrimitiveStateTransformMode.DENSE
                and isinstance(self.state_transform_projection, nn.Linear)
            ):
                raise NotImplementedError(
                    "primitive_runtime_backend=triton for primitive profile p2-2 currently requires a block-diagonal state transform"
                )
            raise NotImplementedError(
                f"primitive_runtime_backend=triton is not implemented for P2.2 state transform mode {self.state_transform_mode.value}"
            )

        outputs = _allocate_emitted_outputs(
            batch_size=batch_size,
            seq_len=seq_len,
            width=self.d_model,
            device=device,
            dtype=dtype,
        )
        for position in range(seq_len):
            transformed_state = _rotate_state_pairs_with_trig(
                self.state_transform_projection(state),
                cos=runtime_plan.angle_cos[:, position, :],
                sin=runtime_plan.angle_sin[:, position, :],
            )
            state = (
                runtime_plan.update_gates[:, position, :] * transformed_state
                + runtime_plan.retain_gates[:, position, :] * runtime_plan.candidates[:, position, :]
            )
            outputs[:, position, :] = runtime_plan.output_gates[:, position, :] * self.output_projection(state)
        return SequencePrimitiveScanResult(emitted_outputs=outputs, final_state=state)

    def scan(self, inputs: torch.Tensor, initial_state: torch.Tensor | None = None) -> SequencePrimitiveScanResult:
        runtime_plan = self.prepare_runtime_plan(inputs)
        return self.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
            initial_state=initial_state,
        )


def build_sequence_primitive(
    profile: PrimitiveProfile,
    d_model: int,
    execution_profile: PrimitiveExecutionProfile,
    state_transform_mode: PrimitiveStateTransformMode = PrimitiveStateTransformMode.DENSE,
) -> SequencePrimitive:
    if state_transform_mode is not PrimitiveStateTransformMode.DENSE and profile in {
        PrimitiveProfile.P1,
        PrimitiveProfile.P1_FRACTAL_HYBRID,
        PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE,
        PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE,
        PrimitiveProfile.P20_GDN_ROLE,
        PrimitiveProfile.P2_MANDELBROT,
        PrimitiveProfile.P3_HIERARCHICAL,
        PrimitiveProfile.B1_FRACTAL_GATED,
        PrimitiveProfile.B2_STABLE_HIERARCHICAL,
        PrimitiveProfile.B3_FRACTAL_HIERARCHICAL,
        PrimitiveProfile.B4_UNIVERSAL,
        PrimitiveProfile.IFS,
        PrimitiveProfile.GENERALIZED_MOBIUS,
        PrimitiveProfile.LOGISTIC_CHAOTIC_MAP,
        PrimitiveProfile.JULIA_RECURSIVE_ESCAPE,
        PrimitiveProfile.MANDELBOX_RECURSIVE,
    }:
        raise ValueError(
            f"primitive profile {profile.value} does not support non-dense state transforms"
        )
    if execution_profile is PrimitiveExecutionProfile.RUNTIME:
        if profile is PrimitiveProfile.P1:
            return ContractiveRuntimeSequenceMixer(d_model)
        if profile is PrimitiveProfile.P1_FRACTAL_HYBRID:
            return P1FractalHybridRuntimeSequenceMixer(d_model)
        if profile is PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE:
            return P1FractalHybridCompositeSequenceMixer(d_model)
        if profile is PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE:
            return P1FractalHybridDynGateSequenceMixer(d_model)
        if profile is PrimitiveProfile.P20:
            return P20RotaryStateOutputRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P20_GDN_ROLE:
            return P20GatedDeltaRoleSequenceMixer(d_model)
        if profile is PrimitiveProfile.P2:
            return P2RotaryReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P23:
            return P23RotaryCarryBlendReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P21:
            return P21WideLatentRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P22:
            return P22WideLatentReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P2_MANDELBROT:
            return P2MandelbrotSequenceMixer(d_model)
        if profile is PrimitiveProfile.P3_HIERARCHICAL:
            return P3HierarchicalSequenceMixer(d_model)
        if profile is PrimitiveProfile.B1_FRACTAL_GATED:
            return B1FractalGatedSequenceMixer(d_model)
        if profile is PrimitiveProfile.B2_STABLE_HIERARCHICAL:
            return B2StableHierarchicalSequenceMixer(d_model)
        if profile is PrimitiveProfile.B3_FRACTAL_HIERARCHICAL:
            return B3FractalHierarchicalSequenceMixer(d_model)
        if profile is PrimitiveProfile.B4_UNIVERSAL:
            return B4UniversalSequenceMixer(d_model)
        if profile is PrimitiveProfile.IFS:
            return IfsSequenceMixer(d_model)
        if profile is PrimitiveProfile.GENERALIZED_MOBIUS:
            return GeneralizedMobiusSequenceMixer(d_model)
        if profile is PrimitiveProfile.LOGISTIC_CHAOTIC_MAP:
            return LogisticChaoticMapSequenceMixer(d_model)
        if profile is PrimitiveProfile.JULIA_RECURSIVE_ESCAPE:
            return JuliaRecursiveEscapeSequenceMixer(d_model)
        if profile is PrimitiveProfile.MANDELBOX_RECURSIVE:
            return MandelboxRecursiveSequenceMixer(d_model)
        raise ValueError(f"unsupported primitive runtime profile: {profile}")

    if profile is PrimitiveProfile.P1:
        return ContractiveSequenceMixer(d_model)
    if profile is PrimitiveProfile.P1_FRACTAL_HYBRID:
        return P1FractalHybridSequenceMixer(d_model)
    if profile is PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE:
        return P1FractalHybridCompositeSequenceMixer(d_model)
    if profile is PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE:
        return P1FractalHybridDynGateSequenceMixer(d_model)
    if profile is PrimitiveProfile.P20:
        return P20RotaryStateOutputSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P20_GDN_ROLE:
        return P20GatedDeltaRoleSequenceMixer(d_model)
    if profile is PrimitiveProfile.P2:
        return P2RotaryReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P23:
        return P23RotaryCarryBlendReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P21:
        return P21WideLatentSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P22:
        return P22WideLatentReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P2_MANDELBROT:
        return P2MandelbrotSequenceMixer(d_model)
    if profile is PrimitiveProfile.P3_HIERARCHICAL:
        return P3HierarchicalSequenceMixer(d_model)
    if profile is PrimitiveProfile.B1_FRACTAL_GATED:
        return B1FractalGatedSequenceMixer(d_model)
    if profile is PrimitiveProfile.B2_STABLE_HIERARCHICAL:
        return B2StableHierarchicalSequenceMixer(d_model)
    if profile is PrimitiveProfile.B3_FRACTAL_HIERARCHICAL:
        return B3FractalHierarchicalSequenceMixer(d_model)
    if profile is PrimitiveProfile.B4_UNIVERSAL:
        return B4UniversalSequenceMixer(d_model)
    if profile is PrimitiveProfile.IFS:
        return IfsSequenceMixer(d_model)
    if profile is PrimitiveProfile.GENERALIZED_MOBIUS:
        return GeneralizedMobiusSequenceMixer(d_model)
    if profile is PrimitiveProfile.LOGISTIC_CHAOTIC_MAP:
        return LogisticChaoticMapSequenceMixer(d_model)
    if profile is PrimitiveProfile.JULIA_RECURSIVE_ESCAPE:
        return JuliaRecursiveEscapeSequenceMixer(d_model)
    if profile is PrimitiveProfile.MANDELBOX_RECURSIVE:
        return MandelboxRecursiveSequenceMixer(d_model)
    raise ValueError(f"unsupported primitive profile: {profile}")


class PrimitiveMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        primitive_profile: PrimitiveProfile,
        execution_profile: PrimitiveExecutionProfile,
        residual_mode: PrimitiveResidualMode,
        readout_mode: PrimitiveReadoutMode,
        norm_mode: PrimitiveNormMode,
        wrapper_mode: PrimitiveWrapperMode,
        state_transform_mode: PrimitiveStateTransformMode,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.primitive_profile = primitive_profile
        self.execution_profile = execution_profile
        self.residual_mode = residual_mode
        self.readout_mode = readout_mode
        self.norm_mode = norm_mode
        self.wrapper_mode = wrapper_mode
        self.state_transform_mode = state_transform_mode
        self.primitive = build_sequence_primitive(
            primitive_profile,
            d_model,
            execution_profile,
            state_transform_mode=state_transform_mode,
        )
        self.input_norm = nn.LayerNorm(d_model) if wrapper_mode is PrimitiveWrapperMode.STANDARD else None
        self.output_norm = nn.LayerNorm(d_model) if wrapper_mode is PrimitiveWrapperMode.STANDARD else None
        self.input_rms_norm = SimpleRmsNorm(d_model) if wrapper_mode is PrimitiveWrapperMode.MAMBA_RMS else None
        self.output_rms_norm = SimpleRmsNorm(d_model) if wrapper_mode is PrimitiveWrapperMode.MAMBA_RMS else None
        residual_scale_init = 0.1 if primitive_profile is PrimitiveProfile.P20_GDN_ROLE else 0.5
        self.residual_scale = (
            nn.Parameter(torch.full((d_model,), residual_scale_init))
            if residual_mode is PrimitiveResidualMode.SCALED
            else None
        )
        self.residual_gate_projection = build_linear(d_model, d_model) if residual_mode is PrimitiveResidualMode.GATED else None
        self.wrapper_readout_projection = build_linear(d_model, d_model) if readout_mode in {PrimitiveReadoutMode.PROJECTED, PrimitiveReadoutMode.PROJECTED_NORM} else None
        self.wrapper_readout_norm = nn.LayerNorm(d_model) if readout_mode is PrimitiveReadoutMode.PROJECTED_NORM else None
        self.wrapper_post_readout_norm = nn.LayerNorm(d_model) if norm_mode is PrimitiveNormMode.POST_READOUT_NORM else None
        self.wrapper_residual_renorm = nn.LayerNorm(d_model) if norm_mode is PrimitiveNormMode.RESIDUAL_RENORM else None
        self.feedforward = PositionWiseFeedForward(d_model, d_ff)

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        if self.execution_profile is PrimitiveExecutionProfile.RUNTIME:
            if primitive_runtime_backend == "triton":
                if self.primitive_profile is PrimitiveProfile.P20:
                    pass
                elif (
                    self.primitive_profile in {PrimitiveProfile.P2, PrimitiveProfile.P21, PrimitiveProfile.P22}
                    and self.state_transform_mode
                    in {
                        PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                        PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                    }
                ):
                    pass
                else:
                    raise NotImplementedError(
                        "primitive_runtime_backend=triton is currently implemented for primitive profile p2-0, "
                        "and for primitive profiles p2 / p2-1 / p2-2 with block-diagonal state transforms"
                    )
            self.primitive.configure_runtime_policy(
                compile_mode=compile_mode,
                primitive_runtime_backend=primitive_runtime_backend,
            )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.primitive.input_norm"):
            if self.wrapper_mode is PrimitiveWrapperMode.STANDARD:
                normed = self.input_norm(hidden)
            else:
                normed = self.input_rms_norm(hidden)

        if self.execution_profile is PrimitiveExecutionProfile.RUNTIME:
            with record_function("path1.primitive.prepare_runtime_plan"):
                runtime_plan = self.primitive.prepare_runtime_plan(normed)
            with record_function("path1.primitive.scan_runtime"):
                mixed = self.primitive.scan_with_runtime_plan(
                    runtime_plan,
                    batch_size=normed.shape[0],
                    device=normed.device,
                    dtype=normed.dtype,
                    seq_len=normed.shape[1],
                ).emitted_outputs
        else:
            with record_function("path1.primitive.scan_reference"):
                mixed = self.primitive.scan(normed).emitted_outputs
        with record_function("path1.primitive.readout"):
            if self.readout_mode is PrimitiveReadoutMode.DIRECT:
                readout = mixed
            elif self.readout_mode is PrimitiveReadoutMode.PROJECTED:
                readout = self.wrapper_readout_projection(mixed)
            else:
                readout = self.wrapper_readout_norm(self.wrapper_readout_projection(mixed))

        if self.norm_mode is PrimitiveNormMode.POST_READOUT_NORM:
            readout = self.wrapper_post_readout_norm(readout)

        if self.residual_mode is PrimitiveResidualMode.PLAIN:
            residual = hidden + readout
        elif self.residual_mode is PrimitiveResidualMode.SCALED:
            residual = hidden + readout * self.residual_scale.view(1, 1, self.d_model)
        else:
            gate = gated_sigmoid(self.residual_gate_projection(normed))
            residual = hidden + gate * readout

        if self.norm_mode is PrimitiveNormMode.RESIDUAL_RENORM:
            residual = self.wrapper_residual_renorm(residual)

        with record_function("path1.primitive.feedforward"):
            if self.wrapper_mode is PrimitiveWrapperMode.STANDARD:
                ff_input = self.output_norm(residual)
            else:
                ff_input = self.output_rms_norm(residual)
            return residual + self.feedforward(ff_input)
