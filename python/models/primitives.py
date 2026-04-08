from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from python.models.common import (
    PositionWiseFeedForward,
    SimpleRmsNorm,
    build_linear,
    gated_sigmoid,
    leading_state_slice,
    one_minus,
    rotate_state_pairs,
)
from python.specs.path1 import (
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveWrapperMode,
)


@dataclass(frozen=True)
class SequencePrimitiveStepResult:
    next_state: torch.Tensor
    emitted_output: torch.Tensor


@dataclass(frozen=True)
class SequencePrimitiveScanResult:
    emitted_outputs: torch.Tensor
    final_state: torch.Tensor


class SequencePrimitive(nn.Module):
    state_width: int
    d_model: int

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.state_width, device=device, dtype=dtype)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:  # pragma: no cover - abstract
        raise NotImplementedError

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


class ContractiveSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model
        self.gate_projection = build_linear(d_model, d_model)
        self.state_projection = build_linear(d_model, d_model)
        self.input_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        gate = gated_sigmoid(self.gate_projection(inputs))
        mix = self.state_projection(state) + self.input_projection(inputs)
        next_state = gate * mix + one_minus(gate) * state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=next_state)


class P20RotaryStateOutputSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2_0 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.update_gate_projection = build_linear(d_model, d_model)
        self.state_transform_projection = build_linear(d_model, d_model)
        self.angle_projection = build_linear(d_model, d_model // 2)
        self.candidate_projection = build_linear(d_model, d_model)
        self.output_gate_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate = gated_sigmoid(self.update_gate_projection(inputs))
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            self.angle_projection(inputs),
        )
        candidate = torch.tanh(self.candidate_projection(inputs))
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(self.output_gate_projection(inputs)) * next_state
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P2RotaryReadoutSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.update_gate_projection = build_linear(d_model, d_model)
        self.state_transform_projection = build_linear(d_model, d_model)
        self.angle_projection = build_linear(d_model, d_model // 2)
        self.candidate_projection = build_linear(d_model, d_model)
        self.output_gate_projection = build_linear(d_model, d_model)
        self.output_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate = gated_sigmoid(self.update_gate_projection(inputs))
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            self.angle_projection(inputs),
        )
        candidate = torch.tanh(self.candidate_projection(inputs))
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(self.output_gate_projection(inputs)) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P23RotaryCarryBlendReadoutSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"p2_3 requires even d_model, got {d_model}")
        self.d_model = d_model
        self.state_width = d_model
        self.update_gate_projection = build_linear(d_model, d_model)
        self.state_transform_projection = build_linear(d_model, d_model)
        self.carry_state_projection = build_linear(d_model, d_model)
        self.dynamics_mix_gate_projection = build_linear(d_model, d_model)
        self.angle_projection = build_linear(d_model, d_model // 2)
        self.candidate_projection = build_linear(d_model, d_model)
        self.output_gate_projection = build_linear(d_model, d_model)
        self.output_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate = gated_sigmoid(self.update_gate_projection(inputs))
        dynamics_mix_gate = gated_sigmoid(self.dynamics_mix_gate_projection(inputs))
        rotated_state = rotate_state_pairs(
            self.state_transform_projection(state),
            self.angle_projection(inputs),
        )
        carried_state = torch.tanh(self.carry_state_projection(state))
        transformed_state = dynamics_mix_gate * rotated_state + one_minus(dynamics_mix_gate) * carried_state
        candidate = torch.tanh(self.candidate_projection(inputs))
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(self.output_gate_projection(inputs)) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P21WideLatentSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model * 2
        self.update_gate_projection = build_linear(d_model, self.state_width)
        self.state_transform_projection = build_linear(self.state_width, self.state_width)
        self.angle_projection = build_linear(d_model, self.state_width // 2)
        self.candidate_projection = build_linear(d_model, self.state_width)
        self.output_gate_projection = build_linear(d_model, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate = gated_sigmoid(self.update_gate_projection(inputs))
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            self.angle_projection(inputs),
        )
        candidate = torch.tanh(self.candidate_projection(inputs))
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        readout = leading_state_slice(next_state, self.d_model)
        emitted_output = gated_sigmoid(self.output_gate_projection(inputs)) * readout
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


class P22WideLatentReadoutSequenceMixer(SequencePrimitive):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_width = d_model * 2
        self.update_gate_projection = build_linear(d_model, self.state_width)
        self.state_transform_projection = build_linear(self.state_width, self.state_width)
        self.angle_projection = build_linear(d_model, self.state_width // 2)
        self.candidate_projection = build_linear(d_model, self.state_width)
        self.output_gate_projection = build_linear(d_model, d_model)
        self.output_projection = build_linear(self.state_width, d_model)

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> SequencePrimitiveStepResult:
        update_gate = gated_sigmoid(self.update_gate_projection(inputs))
        transformed_state = rotate_state_pairs(
            self.state_transform_projection(state),
            self.angle_projection(inputs),
        )
        candidate = torch.tanh(self.candidate_projection(inputs))
        next_state = update_gate * transformed_state + one_minus(update_gate) * candidate
        emitted_output = gated_sigmoid(self.output_gate_projection(inputs)) * self.output_projection(next_state)
        return SequencePrimitiveStepResult(next_state=next_state, emitted_output=emitted_output)


def build_sequence_primitive(profile: PrimitiveProfile, d_model: int) -> SequencePrimitive:
    if profile is PrimitiveProfile.P1:
        return ContractiveSequenceMixer(d_model)
    if profile is PrimitiveProfile.P20:
        return P20RotaryStateOutputSequenceMixer(d_model)
    if profile is PrimitiveProfile.P2:
        return P2RotaryReadoutSequenceMixer(d_model)
    if profile is PrimitiveProfile.P23:
        return P23RotaryCarryBlendReadoutSequenceMixer(d_model)
    if profile is PrimitiveProfile.P21:
        return P21WideLatentSequenceMixer(d_model)
    if profile is PrimitiveProfile.P22:
        return P22WideLatentReadoutSequenceMixer(d_model)
    raise ValueError(f"unsupported primitive profile: {profile}")


class PrimitiveMixerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        primitive_profile: PrimitiveProfile,
        residual_mode: PrimitiveResidualMode,
        readout_mode: PrimitiveReadoutMode,
        norm_mode: PrimitiveNormMode,
        wrapper_mode: PrimitiveWrapperMode,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.primitive_profile = primitive_profile
        self.residual_mode = residual_mode
        self.readout_mode = readout_mode
        self.norm_mode = norm_mode
        self.wrapper_mode = wrapper_mode
        self.primitive = build_sequence_primitive(primitive_profile, d_model)
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

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.wrapper_mode is PrimitiveWrapperMode.STANDARD:
            normed = self.input_norm(hidden)
        else:
            normed = self.input_rms_norm(hidden)

        mixed = self.primitive.scan(normed).emitted_outputs
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

        if self.wrapper_mode is PrimitiveWrapperMode.STANDARD:
            ff_input = self.output_norm(residual)
        else:
            ff_input = self.output_rms_norm(residual)
        return residual + self.feedforward(ff_input)

