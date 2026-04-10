from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
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
class P20RuntimePlan:
    update_gates: torch.Tensor
    retain_gates: torch.Tensor
    angle_cos: torch.Tensor
    angle_sin: torch.Tensor
    candidates: torch.Tensor
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
    if state_transform_mode is not PrimitiveStateTransformMode.DENSE and profile is PrimitiveProfile.P1:
        raise ValueError("primitive profile p1 does not support non-dense state transforms")
    if execution_profile is PrimitiveExecutionProfile.RUNTIME:
        if profile is PrimitiveProfile.P1:
            return ContractiveRuntimeSequenceMixer(d_model)
        if profile is PrimitiveProfile.P20:
            return P20RotaryStateOutputRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P2:
            return P2RotaryReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P23:
            return P23RotaryCarryBlendReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P21:
            return P21WideLatentRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        if profile is PrimitiveProfile.P22:
            return P22WideLatentReadoutRuntimeSequenceMixer(d_model, state_transform_mode=state_transform_mode)
        raise ValueError(f"unsupported primitive runtime profile: {profile}")

    if profile is PrimitiveProfile.P1:
        return ContractiveSequenceMixer(d_model)
    if profile is PrimitiveProfile.P20:
        return P20RotaryStateOutputSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P2:
        return P2RotaryReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P23:
        return P23RotaryCarryBlendReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P21:
        return P21WideLatentSequenceMixer(d_model, state_transform_mode=state_transform_mode)
    if profile is PrimitiveProfile.P22:
        return P22WideLatentReadoutSequenceMixer(d_model, state_transform_mode=state_transform_mode)
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
        self.residual_scale = nn.Parameter(torch.full((d_model,), 0.5)) if residual_mode is PrimitiveResidualMode.SCALED else None
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
                    self.primitive_profile in {PrimitiveProfile.P2, PrimitiveProfile.P22}
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
                        "and for primitive profiles p2 / p2-2 with block-diagonal state transforms"
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
