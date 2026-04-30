from __future__ import annotations

import inspect
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from python.models.common import (
    EmlInspiredTreeFeedForward,
    GatedEmlFeedForward,
    GenericTreeExpertFeedForward,
    RoutedEmlFeedForward,
    SimpleRmsNorm,
    TinyGluExpertFeedForward,
    TinyMlpExpertFeedForward,
)
from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer, PrimitiveMixerBlock
from python.models.reference_ssm import ReferenceSsmHybridBlock
from python.models.transformer import (
    LocalCausalSelfAttention,
    LocalCausalTransformerBlock,
    Pr5LocalCausalTransformerBlock,
    local_causal_attention_bias,
)
from python.runtime.cuda_timing import timed_region
from python.runtime.parcae_loop_region import (
    ParcaeLoopRegionConfig,
    ParcaeLoopRegionControls,
    ParcaeLoopRegionKernels,
    run_parcae_loop_region,
)
from python.runtime.recurrent import (
    manual_autograd_parcae_residual_mix,
    manual_autograd_parcae_state_mix,
)
from python.runtime.triton_primitives import (
    TritonPrimitiveBackend,
    build_triton_primitive_backend,
    ensure_triton_runtime_available,
)
from python.specs.common import FFN_BACKENDS, HEAD_LOSS_BACKENDS
from python.specs.path1 import (
    FeedForwardProfile,
    HybridAttentionLayerRole,
    Path1ScaffoldProfile,
    Path1VariantSpec,
    PrimitiveProfile,
)
from python.specs.runtime import PrimitiveStateTransformMode


def _diagnostic_float(value: torch.Tensor | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    return float(value.detach().float().cpu().item())


def _diagnostic_float_list(values: list[torch.Tensor | float]) -> list[float]:
    return [float_value for value in values if (float_value := _diagnostic_float(value)) is not None]


def _build_attention_feed_forward(variant: Path1VariantSpec, *, layer_index: int) -> nn.Module | None:
    if variant.feed_forward_profile is FeedForwardProfile.STANDARD:
        return None
    if (
        variant.feed_forward_layer_indices is not None
        and layer_index not in variant.feed_forward_layer_indices
    ):
        return None
    if variant.feed_forward_profile is FeedForwardProfile.EML_TREE:
        return EmlInspiredTreeFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
            slot_count=variant.eml_slot_count,
            tree_depth=variant.eml_tree_depth,
        )
    if variant.feed_forward_profile is FeedForwardProfile.MLP_EML_GATED:
        return GatedEmlFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
            slot_count=variant.eml_slot_count,
            tree_depth=variant.eml_tree_depth,
        )
    if variant.feed_forward_profile is FeedForwardProfile.MLP_EML_ROUTED:
        return RoutedEmlFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
            slot_count=variant.eml_slot_count,
            tree_depth=variant.eml_tree_depth,
            route_fraction=variant.eml_route_fraction,
        )
    if variant.feed_forward_profile is FeedForwardProfile.TINY_MLP_GATED:
        return TinyMlpExpertFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
        )
    if variant.feed_forward_profile is FeedForwardProfile.TINY_GLU_GATED:
        return TinyGluExpertFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
        )
    if variant.feed_forward_profile is FeedForwardProfile.GENERIC_TREE_GATED:
        return GenericTreeExpertFeedForward(
            variant.shape.d_model,
            variant.shape.d_ff,
            slot_count=variant.eml_slot_count,
            tree_depth=variant.eml_tree_depth,
        )
    raise ValueError(f"unsupported feed-forward profile: {variant.feed_forward_profile}")


class Pr5HashContextEmbedding(nn.Module):
    """Small n-gram context side channel from the PR5 HybridGDN contract."""

    def __init__(self, vocab_size: int, d_model: int, *, hash_vocab_size: int = 3072, hash_dim: int = 112) -> None:
        super().__init__()
        if hash_vocab_size <= 1:
            raise ValueError(f"hash_vocab_size must be greater than 1, got {hash_vocab_size}")
        resolved_hash_dim = min(hash_dim, d_model)
        self.hash_vocab_size = hash_vocab_size
        self.embedding = nn.Embedding(hash_vocab_size, resolved_hash_dim)
        self.projection = (
            nn.Identity()
            if resolved_hash_dim == d_model
            else nn.Linear(resolved_hash_dim, d_model, bias=False)
        )
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        nn.init.zeros_(self.embedding.weight)
        if isinstance(self.projection, nn.Linear):
            nn.init.zeros_(self.projection.weight)
        del vocab_size

    def _bigram_hash(self, tokens: torch.Tensor) -> torch.Tensor:
        typed = tokens.to(torch.int32)
        modulus = self.hash_vocab_size - 1
        hashed = torch.empty_like(typed)
        hashed[..., 0] = modulus
        hashed[..., 1:] = torch.bitwise_xor(36313 * typed[..., 1:], 27191 * typed[..., :-1]) % modulus
        return hashed.long()

    def _trigram_hash(self, tokens: torch.Tensor) -> torch.Tensor:
        typed = tokens.to(torch.int32)
        modulus = self.hash_vocab_size - 1
        hashed = torch.empty_like(typed)
        hashed[..., :2] = modulus
        hashed[..., 2:] = (
            36313 * typed[..., 2:] ^ 27191 * typed[..., 1:-1] ^ 51497 * typed[..., :-2]
        ) % modulus
        return hashed.long()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        context = self.embedding(self._bigram_hash(input_ids)) + self.embedding(self._trigram_hash(input_ids))
        projected = self.projection(context)
        return projected * self.scale.to(dtype=projected.dtype)


class SmearGate(nn.Module):
    """Learned mix between current and previous hidden token."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate.to(dtype=hidden.dtype)).view(1, 1, -1)
        previous = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)
        return (1.0 - gate) * hidden + gate * previous


def _middle_loop_bounds(total_layers: int, loop_layer_count: int | None = None) -> tuple[int, int]:
    loop_width = loop_layer_count if loop_layer_count is not None else max(1, total_layers // 3)
    if loop_width <= 0 or loop_width > total_layers:
        raise ValueError(
            f"loop_layer_count must be in [1, {total_layers}], got {loop_width}"
        )
    start = max(0, (total_layers - loop_width) // 2)
    end = min(total_layers, start + loop_width)
    return start, end


def _hourglass_loop_ranges(
    total_layers: int,
    *,
    loop_layer_count: int | None,
    band_schedule: tuple[int, ...] | None,
) -> tuple[tuple[int, int], ...]:
    if band_schedule is None:
        return (_middle_loop_bounds(total_layers, loop_layer_count),)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for segment_index, segment_length in enumerate(band_schedule):
        start = cursor
        cursor += segment_length
        if segment_index % 2 == 1:
            ranges.append((start, cursor))
    return tuple(ranges)


def _layer_in_ranges(layer_index: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= layer_index < end for start, end in ranges)


def _configure_p20_control_state_transform(controller: P20RotaryStateOutputRuntimeSequenceMixer, profile: str) -> None:
    if profile in {"trainable", "trainable-block-diagonal-8"}:
        controller._triton_identity_state_transform = False
        return
    projection = controller.state_transform_projection
    if profile != "frozen-identity":
        raise ValueError(f"unsupported Parcae P20 control state transform profile: {profile}")
    with torch.no_grad():
        weight = getattr(projection, "weight", None)
        bias = getattr(projection, "bias", None)
        if weight is None:
            raise RuntimeError("frozen-identity Parcae control transform requires a weight parameter")
        weight.zero_()
        if weight.ndim == 2:
            if weight.shape[0] != weight.shape[1]:
                raise RuntimeError("frozen-identity Parcae control transform requires a square dense weight")
            weight.copy_(torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype))
        elif weight.ndim == 3:
            block_width = weight.shape[-1]
            eye = torch.eye(block_width, device=weight.device, dtype=weight.dtype)
            weight.copy_(eye.view(1, block_width, block_width).expand_as(weight))
        else:
            raise RuntimeError("frozen-identity Parcae control transform only supports dense or block-diagonal weights")
        if bias is not None:
            bias.zero_()
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    controller._triton_identity_state_transform = True


def _p20_control_state_transform_mode(profile: str) -> PrimitiveStateTransformMode:
    if profile in {"trainable", "frozen-identity"}:
        return PrimitiveStateTransformMode.BLOCK_DIAGONAL_4
    if profile == "trainable-block-diagonal-8":
        return PrimitiveStateTransformMode.BLOCK_DIAGONAL_8
    raise ValueError(f"unsupported Parcae P20 control state transform profile: {profile}")


class Path1HybridLanguageModel(nn.Module):
    def __init__(self, variant: Path1VariantSpec, *, dtype_mode: str) -> None:
        super().__init__()
        variant.validate()
        self.variant = variant
        self.uses_pr5_scaffold = variant.scaffold_profile is Path1ScaffoldProfile.PR5_HYBRID_GDN
        self.uses_parcae_scaffold = variant.scaffold_profile in {
            Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
        }
        self.uses_parcae_hourglass_scaffold = variant.scaffold_profile in {
            Path1ScaffoldProfile.PARCAE_HOURGLASS_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
        }
        self.uses_parcae_bx = variant.scaffold_profile in {
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_BX_LOOPED_ATTENTION,
        }
        self.uses_parcae_p20_control = variant.scaffold_profile in {
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
        }
        self.embedding = nn.Embedding(
            variant.shape.vocab_size,
            variant.shape.d_model,
        )
        self.register_buffer(
            "_position_ids",
            torch.arange(variant.max_position_embeddings, dtype=torch.long),
            persistent=False,
        )
        self.attention_position_embeddings = nn.ModuleDict()
        self.position_embedding = (
            nn.Embedding(variant.max_position_embeddings, variant.shape.d_model)
            if variant.position_encoding_kind == "learned"
            and variant.attention_position_contract == "shared-input"
            else None
        )
        self.context_embedding = (
            Pr5HashContextEmbedding(variant.shape.vocab_size, variant.shape.d_model)
            if self.uses_pr5_scaffold
            else None
        )
        self.smear_gate = SmearGate(variant.shape.d_model) if self.uses_pr5_scaffold else None
        if self.uses_pr5_scaffold:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.005)
        self.blocks = nn.ModuleList()
        shared_attention_by_width: dict[int, LocalCausalSelfAttention] = {}
        reference_ordinal = 0
        planned_parcae_loop_ranges = (
            _hourglass_loop_ranges(
                len(variant.layer_schedule),
                loop_layer_count=variant.parcae_loop_layer_count,
                band_schedule=variant.parcae_hourglass_band_schedule,
            )
            if self.uses_parcae_hourglass_scaffold
            else ()
        )
        loop_d_model = (
            variant.parcae_loop_d_model
            if self.uses_parcae_hourglass_scaffold and variant.parcae_loop_d_model is not None
            else variant.shape.d_model
        )
        loop_head_count = (
            variant.parcae_loop_head_count
            if self.uses_parcae_hourglass_scaffold and variant.parcae_loop_head_count is not None
            else variant.shape.head_count
        )
        loop_ffn_multiplier = (
            variant.parcae_loop_ffn_multiplier
            if self.uses_parcae_hourglass_scaffold and variant.parcae_loop_ffn_multiplier is not None
            else variant.shape.ffn_multiplier
        )
        if variant.position_encoding_kind == "learned" and variant.attention_position_contract == "attention-only":
            attention_widths = {variant.shape.d_model}
            if self.uses_parcae_hourglass_scaffold:
                attention_widths.add(loop_d_model)
            for width in sorted(attention_widths):
                self.attention_position_embeddings[str(width)] = nn.Embedding(
                    variant.max_position_embeddings,
                    width,
                )
        for layer_index, role in enumerate(variant.layer_schedule):
            layer_d_model = (
                loop_d_model
                if self.uses_parcae_hourglass_scaffold
                and _layer_in_ranges(layer_index, planned_parcae_loop_ranges)
                else variant.shape.d_model
            )
            layer_head_count = loop_head_count if layer_d_model == loop_d_model else variant.shape.head_count
            layer_d_ff = layer_d_model * (loop_ffn_multiplier if layer_d_model == loop_d_model else variant.shape.ffn_multiplier)
            if role is HybridAttentionLayerRole.EXACT_ATTENTION:
                attention = LocalCausalSelfAttention(
                    layer_d_model,
                    layer_head_count,
                    local_window=variant.shape.local_window,
                    attention_kernel=variant.shape.attention_kernel,
                )
                shared_attention_by_width.setdefault(layer_d_model, attention)
                attention_block = (
                    Pr5LocalCausalTransformerBlock(
                        layer_d_model,
                        layer_head_count,
                        layer_d_ff,
                        attention_module=attention,
                    )
                    if self.uses_pr5_scaffold
                    else LocalCausalTransformerBlock(
                        layer_d_model,
                        layer_head_count,
                        layer_d_ff,
                        attention_module=attention,
                        ffn_module=(
                            None
                            if layer_d_model != variant.shape.d_model
                            else _build_attention_feed_forward(variant, layer_index=layer_index)
                        ),
                    )
                )
                self.blocks.append(attention_block)
            elif role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION:
                if layer_d_model not in shared_attention_by_width:
                    shared_attention_by_width[layer_d_model] = LocalCausalSelfAttention(
                        layer_d_model,
                        layer_head_count,
                        local_window=variant.shape.local_window,
                        attention_kernel=variant.shape.attention_kernel,
                    )
                attention_block = (
                    Pr5LocalCausalTransformerBlock(
                        layer_d_model,
                        layer_head_count,
                        layer_d_ff,
                        attention_module=shared_attention_by_width[layer_d_model],
                    )
                    if self.uses_pr5_scaffold
                    else LocalCausalTransformerBlock(
                        layer_d_model,
                        layer_head_count,
                        layer_d_ff,
                        attention_module=shared_attention_by_width[layer_d_model],
                        ffn_module=(
                            None
                            if layer_d_model != variant.shape.d_model
                            else _build_attention_feed_forward(variant, layer_index=layer_index)
                        ),
                    )
                )
                self.blocks.append(attention_block)
            elif role is HybridAttentionLayerRole.REFERENCE_SSM:
                profile = variant.reference_profile_for_ordinal(reference_ordinal)
                reference_ordinal += 1
                self.blocks.append(
                    ReferenceSsmHybridBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        profile=profile,
                        dtype_mode=dtype_mode,
                        layer_index=layer_index,
                        use_pr5_scaffold=self.uses_pr5_scaffold,
                        p20_ramp_init=variant.reference_p20_ramp_init,
                    )
                )
            elif role is HybridAttentionLayerRole.PRIMITIVE:
                self.blocks.append(
                    PrimitiveMixerBlock(
                        variant.shape.d_model,
                        variant.shape.d_ff,
                        primitive_profile=variant.primitive_profile,
                        execution_profile=variant.primitive_execution_profile,
                        residual_mode=variant.primitive_residual_mode,
                        readout_mode=variant.primitive_readout_mode,
                        norm_mode=variant.primitive_norm_mode,
                        wrapper_mode=variant.primitive_wrapper_mode,
                        state_transform_mode=variant.primitive_state_transform_mode,
                    )
                )
            else:
                raise ValueError(f"unsupported path1 layer role: {role}")
        if self.uses_parcae_scaffold:
            self.parcae_loop_ranges = _hourglass_loop_ranges(
                len(self.blocks),
                loop_layer_count=variant.parcae_loop_layer_count,
                band_schedule=variant.parcae_hourglass_band_schedule,
            )
            self.parcae_loop_start, self.parcae_loop_end = self.parcae_loop_ranges[0]
            self.parcae_loop_d_model = loop_d_model
            self.parcae_wide_d_model = variant.shape.d_model
            self.parcae_prelude_norm = (
                SimpleRmsNorm(variant.shape.d_model)
                if variant.parcae_prelude_norm_kind == "rmsnorm"
                else nn.LayerNorm(variant.shape.d_model)
            )
            self.parcae_hourglass_down_projection = (
                nn.Linear(variant.shape.d_model, loop_d_model, bias=False)
                if self.uses_parcae_hourglass_scaffold
                else None
            )
            self.parcae_hourglass_up_projection = (
                nn.Linear(loop_d_model, variant.shape.d_model, bias=False)
                if self.uses_parcae_hourglass_scaffold
                else None
            )
            self.parcae_hourglass_residual_logit = (
                nn.Parameter(torch.full((variant.shape.d_model,), -2.1972246, dtype=torch.float32))
                if self.uses_parcae_hourglass_scaffold
                else None
            )
            self.parcae_decay_raw = nn.Parameter(torch.full((loop_d_model,), -2.0, dtype=torch.float32))
            self.parcae_dt_raw = (
                nn.Parameter(torch.full((loop_d_model,), variant.parcae_dt_raw_init, dtype=torch.float32))
                if variant.parcae_discretization == "zoh"
                else None
            )
            self.parcae_injection_logit = nn.Parameter(torch.full((loop_d_model,), -2.1972246, dtype=torch.float32))
            self.parcae_nonlinear_logit = nn.Parameter(torch.zeros(loop_d_model, dtype=torch.float32))
            self.parcae_b_value_projection = (
                nn.Linear(loop_d_model, loop_d_model, bias=False)
                if self.uses_parcae_bx
                else None
            )
            self.parcae_b_gate_projection = (
                nn.Linear(loop_d_model, loop_d_model)
                if self.uses_parcae_bx
                else None
            )
            if self.parcae_b_value_projection is not None:
                nn.init.eye_(self.parcae_b_value_projection.weight)
            if self.parcae_b_gate_projection is not None:
                nn.init.zeros_(self.parcae_b_gate_projection.weight)
                nn.init.constant_(self.parcae_b_gate_projection.bias, -2.1972246)
            self.parcae_p20_controller = (
                P20RotaryStateOutputRuntimeSequenceMixer(
                    loop_d_model,
                    state_transform_mode=_p20_control_state_transform_mode(
                        variant.parcae_control_state_transform
                    ),
                )
                if self.uses_parcae_p20_control
                else None
            )
            self.parcae_p20_control_norm = SimpleRmsNorm(loop_d_model) if self.uses_parcae_p20_control else None
            self.parcae_p20_position_embedding = (
                nn.Embedding(variant.max_position_embeddings, loop_d_model)
                if self.uses_parcae_p20_control and variant.parcae_control_position_kind == "learned"
                else None
            )
            self.parcae_p20_position_scale = (
                nn.Parameter(torch.tensor(variant.parcae_control_position_scale_init, dtype=torch.float32))
                if self.parcae_p20_position_embedding is not None
                else None
            )
            self.parcae_p20_control_projection = (
                nn.Linear(loop_d_model, loop_d_model * 2)
                if self.uses_parcae_p20_control
                else None
            )
            if self.parcae_p20_control_projection is not None:
                nn.init.normal_(self.parcae_p20_control_projection.weight, mean=0.0, std=1.0e-3)
                nn.init.zeros_(self.parcae_p20_control_projection.bias)
                with torch.no_grad():
                    self.parcae_p20_control_projection.bias[loop_d_model:].fill_(-2.1972246)
            if self.parcae_p20_position_embedding is not None:
                nn.init.normal_(self.parcae_p20_position_embedding.weight, mean=0.0, std=0.02)
            if self.parcae_p20_controller is not None:
                _configure_p20_control_state_transform(
                    self.parcae_p20_controller,
                    variant.parcae_control_state_transform,
                )
            self._last_parcae_norms: list[torch.Tensor | float] = []
            self._last_parcae_injection_gate_mean: torch.Tensor | float | None = None
            self._last_parcae_injection_norm: torch.Tensor | float | None = None
            self._last_parcae_p20_control_norm: torch.Tensor | float | None = None
            self._last_parcae_p20_control_steps: int | None = None
        else:
            self.parcae_loop_start = 0
            self.parcae_loop_end = 0
            self.parcae_loop_ranges = ()
            self.parcae_loop_d_model = variant.shape.d_model
            self.parcae_wide_d_model = variant.shape.d_model
            self.parcae_prelude_norm = None
            self.parcae_hourglass_down_projection = None
            self.parcae_hourglass_up_projection = None
            self.parcae_hourglass_residual_logit = None
            self.parcae_decay_raw = None
            self.parcae_dt_raw = None
            self.parcae_injection_logit = None
            self.parcae_nonlinear_logit = None
            self.parcae_b_value_projection = None
            self.parcae_b_gate_projection = None
            self.parcae_p20_controller = None
            self.parcae_p20_control_norm = None
            self.parcae_p20_position_embedding = None
            self.parcae_p20_position_scale = None
            self.parcae_p20_control_projection = None
            self._last_parcae_norms = []
            self._last_parcae_injection_gate_mean = None
            self._last_parcae_injection_norm = None
            self._last_parcae_p20_control_norm = None
            self._last_parcae_p20_control_steps = None
        if variant.final_norm_kind == "rmsnorm":
            self.final_norm = SimpleRmsNorm(variant.shape.d_model)
        elif variant.final_norm_kind == "layernorm":
            self.final_norm = nn.LayerNorm(variant.shape.d_model)
        else:
            self.final_norm = nn.Identity()
        self.output = nn.Linear(variant.shape.d_model, variant.shape.vocab_size, bias=False)
        self._head_loss_backend = "dense"
        self.variant_runtime_ffn_backend = "dense"
        self.parcae_runtime_diagnostics = True
        self._compiled_head_loss_impl = None
        self._compiled_parcae_p20_post_scan_injection_impl = None
        self._compiled_parcae_loop_input_projection_impl = None
        self._compiled_parcae_loop_output_projection_impl = None
        self._compiled_parcae_band_prepare_impl = None
        self._compiled_parcae_state_mix_impl = None
        self._compiled_parcae_residual_mix_impl = None
        self._compiled_parcae_loop_iteration_impl = None
        self._compiled_parcae_loop_bands_impl = None
        self._compiled_parcae_loop_impl = None
        self._parcae_triton_backend = None
        self._parcae_triton_backend: TritonPrimitiveBackend | None = None

    @property
    def model_label(self) -> str:
        return f"path1_{self.variant.label.replace('-', '_')}"

    def _position_indices(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.variant.max_position_embeddings:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_position_embeddings "
                f"{self.variant.max_position_embeddings}"
            )
        position_ids = self._position_ids
        if position_ids.device != device:
            position_ids = position_ids.to(device=device)
        return position_ids[:seq_len]

    def _resolve_attention_position_features(
        self,
        *,
        positions: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        if len(self.attention_position_embeddings) == 0:
            return {}
        resolved_positions = positions if positions is not None else self._position_indices(seq_len, device)
        return {
            int(width_key): embedding(resolved_positions).view(1, seq_len, -1).to(dtype=dtype)
            for width_key, embedding in self.attention_position_embeddings.items()
        }

    def _forward_block(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        residual_anchor: torch.Tensor | None = None,
        position_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(block, Pr5LocalCausalTransformerBlock):
            return block(hidden, mask, residual_anchor, position_features=position_features)
        if isinstance(block, LocalCausalTransformerBlock):
            return block(hidden, mask, position_features=position_features)
        return block(hidden, mask)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        seq_len = input_ids.shape[1]
        positions = None
        if self.position_embedding is not None or len(self.attention_position_embeddings) > 0:
            positions = self._position_indices(seq_len, input_ids.device)
        if self.position_embedding is not None:
            hidden = hidden + self.position_embedding(positions).view(1, seq_len, -1)
        if self.context_embedding is not None:
            hidden = hidden + self.context_embedding(input_ids)
            hidden = F.rms_norm(hidden, (hidden.size(-1),))
        if self.smear_gate is not None:
            hidden = self.smear_gate(hidden)
        residual_anchor = hidden
        mask = (
            None
            if self.variant.shape.local_window >= seq_len
            else local_causal_attention_bias(
                seq_len,
                self.variant.shape.local_window,
                input_ids.device,
                hidden.dtype,
            )
        )
        attention_position_features = self._resolve_attention_position_features(
            positions=positions,
            seq_len=seq_len,
            device=input_ids.device,
            dtype=hidden.dtype,
        )
        if self.uses_parcae_scaffold:
            if self.variant.parcae_hourglass_band_schedule is not None:
                parcae_loop_bands_impl = (
                    self._compiled_parcae_loop_bands_impl
                    or self._forward_parcae_loop_bands
                )
                hidden = parcae_loop_bands_impl(
                    hidden,
                    mask,
                    attention_position_features=attention_position_features,
                )
            else:
                pass_count = self.variant.parcae_hourglass_pass_count if self.uses_parcae_hourglass_scaffold else 1
                parcae_loop_impl = self._compiled_parcae_loop_impl or self._forward_parcae_loop
                for _ in range(pass_count):
                    hidden = parcae_loop_impl(
                        hidden,
                        mask,
                        attention_position_features=attention_position_features,
                    )
        else:
            for block in self.blocks:
                hidden = self._forward_block(
                    block,
                    hidden,
                    mask,
                    residual_anchor=residual_anchor,
                    position_features=attention_position_features.get(hidden.shape[-1]),
                )
        return hidden

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        with timed_region("path1.lm_head.total"):
            with timed_region("path1.lm_head.final_norm"):
                hidden = self.final_norm(hidden)
            with timed_region("path1.lm_head.output_projection"):
                return self.output(hidden)

    def _head_loss_impl(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        pad_token: int,
    ) -> torch.Tensor:
        hidden = self.final_norm(hidden)
        logits = self.output(hidden)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=pad_token,
        )

    def loss_from_hidden(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        pad_token: int,
    ) -> torch.Tensor:
        if self._head_loss_backend == "compiled":
            compiled_loss = self._compiled_head_loss_impl
            if compiled_loss is None:
                if not hasattr(torch, "compile"):
                    raise RuntimeError("head_loss_backend=compiled requires torch.compile")
                compiled_loss = torch.compile(self._head_loss_impl, mode="reduce-overhead")
                self._compiled_head_loss_impl = compiled_loss
            with timed_region("path1.lm_head.loss_total"):
                with timed_region("path1.lm_head.compiled_loss"):
                    return compiled_loss(hidden, target_ids, pad_token)
        if self._head_loss_backend == "streaming-kernel":
            raise RuntimeError(
                "head_loss_backend=streaming-kernel requested, but no streaming LM-head "
                "cross-entropy kernel is registered for Path1HybridLanguageModel"
            )

        with timed_region("path1.lm_head.loss_total"):
            with timed_region("path1.lm_head.final_norm"):
                hidden = self.final_norm(hidden)
            with timed_region("path1.lm_head.output_projection"):
                logits = self.output(hidden)
            with timed_region("path1.lm_head.cross_entropy"):
                return F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_ids.reshape(-1),
                    ignore_index=pad_token,
                )

    def forward_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        pad_token: int,
    ) -> torch.Tensor:
        return self.loss_from_hidden(
            self.forward_hidden(input_ids),
            target_ids,
            pad_token=pad_token,
        )

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden(self.forward_hidden(input_ids))

    def _forward_parcae_loop(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        attention_position_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.parcae_prelude_norm is None
            or self.parcae_decay_raw is None
            or self.parcae_injection_logit is None
            or self.parcae_nonlinear_logit is None
        ):
            raise RuntimeError("parcae scaffold is not initialized")

        with timed_region("path1.parcae.prelude_blocks"):
            for block in self.blocks[: self.parcae_loop_start]:
                hidden = self._forward_block(
                    block,
                    hidden,
                    mask,
                    position_features=attention_position_features.get(hidden.shape[-1]),
                )

        loop_anchor = hidden
        loop_input_projection_impl = (
            self._compiled_parcae_loop_input_projection_impl
            or self._parcae_loop_input_projection_impl
        )
        with timed_region("path1.parcae.loop_input_projection"):
            loop_input = loop_input_projection_impl(hidden)
        state = torch.zeros_like(loop_input)
        decay_rate = F.softplus(self.parcae_decay_raw.to(dtype=hidden.dtype)).view(1, 1, -1)
        if self.variant.parcae_discretization == "stable-exp":
            decay = torch.exp(-decay_rate)
            input_scale = torch.ones_like(decay)
        elif self.variant.parcae_discretization == "zoh":
            if self.parcae_dt_raw is None:
                raise RuntimeError("Parcae ZOH discretization requested without parcae_dt_raw")
            step_size = F.softplus(self.parcae_dt_raw.to(dtype=hidden.dtype)).view(1, 1, -1)
            decay = torch.exp(-step_size * decay_rate)
            input_scale = (1.0 - decay) / decay_rate.clamp_min(1.0e-6)
        else:
            raise RuntimeError(f"unsupported Parcae discretization: {self.variant.parcae_discretization}")
        nonlinear = torch.sigmoid(self.parcae_nonlinear_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        with timed_region("path1.parcae.injection"):
            injection_value, injection_gate = self._parcae_injection(loop_input)
            injection = input_scale * injection_gate * injection_value
        recurrent_blocks = self.blocks[self.parcae_loop_start : self.parcae_loop_end]
        backward_steps = self.variant.parcae_backward_steps or self.variant.parcae_loop_count
        gradient_start_step = self.variant.parcae_loop_count - min(backward_steps, self.variant.parcae_loop_count)

        state_mix_impl = self._compiled_parcae_state_mix_impl or self._parcae_state_mix_impl
        residual_mix_impl = self._compiled_parcae_residual_mix_impl or self._parcae_residual_mix_impl
        if self.variant.parcae_loop_update_backend == "manual-autograd":
            state_mix_impl = manual_autograd_parcae_state_mix
            residual_mix_impl = manual_autograd_parcae_residual_mix
        elif self.variant.parcae_loop_update_backend == "triton-glue":
            if self._parcae_triton_backend is None:
                raise RuntimeError("parcae_loop_update_backend=triton-glue requires a configured Triton backend")
            state_mix_impl = self._parcae_triton_backend.parcae_state_mix
            residual_mix_impl = self._parcae_triton_backend.parcae_residual_mix
        elif self.variant.parcae_loop_update_backend == "triton-loop-forward":
            if self._parcae_triton_backend is None:
                raise RuntimeError("parcae_loop_update_backend=triton-loop-forward requires a configured Triton backend")
            state_mix_impl = self._parcae_triton_backend.parcae_state_mix
            residual_mix_impl = self._parcae_triton_backend.parcae_residual_mix

        compiled_loop_iteration_impl = self._compiled_parcae_loop_iteration_impl
        compiled_position_features_by_block: tuple[torch.Tensor | None, ...] = ()
        compiled_flex_block_masks: tuple[object | None, ...] = ()
        precompute_loop_context = (
            compiled_loop_iteration_impl is not None
            or self.variant.parcae_loop_update_backend == "lean-eager"
        )
        if precompute_loop_context:
            with timed_region("path1.parcae.loop_context"):
                compiled_position_features_by_block = tuple(
                    attention_position_features.get(loop_input.shape[-1])
                    for _ in recurrent_blocks
                )
                compiled_flex_block_masks = tuple(
                    block._full_block_flex_block_mask(state, mask)
                    if isinstance(block, LocalCausalTransformerBlock)
                    else None
                    for block in recurrent_blocks
                )

        controls = ParcaeLoopRegionControls(
            decay=decay,
            injection=injection,
            nonlinear=nonlinear,
        )

        def forward_recurrent_block(block_index: int, mixed: torch.Tensor) -> torch.Tensor:
            block = recurrent_blocks[block_index]
            if precompute_loop_context:
                position_features = compiled_position_features_by_block[block_index]
                flex_block_mask = compiled_flex_block_masks[block_index]
                compiled_full_block = getattr(block, "_compiled_full_block_impl", None)
                if (
                    isinstance(block, LocalCausalTransformerBlock)
                    and compiled_full_block is not None
                ):
                    return compiled_full_block(mixed, mask, position_features, flex_block_mask)
                return self._forward_block(
                    block,
                    mixed,
                    mask,
                    position_features=position_features,
                )
            return self._forward_block(
                block,
                mixed,
                mask,
                position_features=attention_position_features.get(mixed.shape[-1]),
            )

        def apply_recurrent_residual(
            block_index: int,
            current_state: torch.Tensor,
            mixed: torch.Tensor,
            block_out: torch.Tensor,
            loop_controls: ParcaeLoopRegionControls,
        ) -> torch.Tensor:
            if (
                self.variant.parcae_loop_update_backend == "triton-loop-forward"
                and block_index == 0
            ):
                with timed_region("path1.parcae.triton_loop_update_forward"):
                    return self._parcae_triton_backend.parcae_loop_update(
                        current_state,
                        loop_controls.decay,
                        loop_controls.injection,
                        block_out,
                        loop_controls.nonlinear,
                    )
            return residual_mix_impl(mixed, block_out, loop_controls.nonlinear)

        compiled_iteration = None
        if compiled_loop_iteration_impl is not None:
            compiled_iteration = lambda current_state: compiled_loop_iteration_impl(
                current_state,
                decay,
                injection,
                nonlinear,
                mask,
                compiled_position_features_by_block,
                compiled_flex_block_masks,
            )

        loop_result = run_parcae_loop_region(
            initial_state=state,
            controls=controls,
            config=ParcaeLoopRegionConfig(
                loop_count=self.variant.parcae_loop_count,
                gradient_start_step=gradient_start_step,
                recurrent_block_count=len(recurrent_blocks),
                timing_prefix="path1.parcae",
                diagnostics_enabled=self.parcae_runtime_diagnostics,
            ),
            kernels=ParcaeLoopRegionKernels(
                state_mix=state_mix_impl,
                forward_recurrent_block=forward_recurrent_block,
                apply_recurrent_residual=apply_recurrent_residual,
                compiled_iteration=compiled_iteration,
            ),
        )
        state = loop_result.final_state
        norm_history = loop_result.norm_history

        loop_output_projection_impl = (
            self._compiled_parcae_loop_output_projection_impl
            or self._parcae_loop_output_projection_impl
        )
        with timed_region("path1.parcae.loop_output_projection"):
            hidden = loop_output_projection_impl(state, loop_anchor)
        with timed_region("path1.parcae.coda_blocks"):
            for block in self.blocks[self.parcae_loop_end :]:
                hidden = self._forward_block(
                    block,
                    hidden,
                    mask,
                    position_features=attention_position_features.get(hidden.shape[-1]),
                )
        self._last_parcae_norms = norm_history
        return hidden

    def _forward_parcae_loop_bands(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        attention_position_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        cursor = 0
        all_norms: list[torch.Tensor] = []
        for band_index, (loop_start, loop_end) in enumerate(self.parcae_loop_ranges):
            with timed_region(f"path1.parcae.band{band_index}.wide_blocks_before"):
                for block in self.blocks[cursor:loop_start]:
                    hidden = self._forward_block(
                        block,
                        hidden,
                        mask,
                        position_features=attention_position_features.get(hidden.shape[-1]),
                    )
            hidden, band_norms = self._forward_parcae_band_core(
                hidden,
                self.blocks[loop_start:loop_end],
                mask,
                attention_position_features=attention_position_features,
                band_index=band_index,
            )
            all_norms.extend(band_norms)
            cursor = loop_end
        with timed_region("path1.parcae.final_wide_blocks"):
            for block in self.blocks[cursor:]:
                hidden = self._forward_block(
                    block,
                    hidden,
                    mask,
                    position_features=attention_position_features.get(hidden.shape[-1]),
                )
        self._last_parcae_norms = all_norms
        return hidden

    def _forward_parcae_band_core(
        self,
        hidden: torch.Tensor,
        recurrent_blocks: Sequence[nn.Module],
        mask: torch.Tensor | None,
        *,
        attention_position_features: dict[int, torch.Tensor],
        band_index: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if (
            self.parcae_prelude_norm is None
            or self.parcae_decay_raw is None
            or self.parcae_injection_logit is None
            or self.parcae_nonlinear_logit is None
        ):
            raise RuntimeError("parcae scaffold is not initialized")

        loop_anchor = hidden
        prepare_impl = self._compiled_parcae_band_prepare_impl or self._parcae_band_prepare_impl
        with timed_region(f"path1.parcae.band{band_index}.prepare"):
            loop_input, decay, injection, nonlinear = prepare_impl(hidden)
            decay, injection, nonlinear = self._normalize_parcae_loop_controls(
                loop_input,
                decay,
                injection,
                nonlinear,
            )
        state = torch.zeros_like(loop_input)
        backward_steps = self.variant.parcae_backward_steps or self.variant.parcae_loop_count
        gradient_start_step = self.variant.parcae_loop_count - min(backward_steps, self.variant.parcae_loop_count)

        state_mix_impl = self._compiled_parcae_state_mix_impl or self._parcae_state_mix_impl
        residual_mix_impl = self._compiled_parcae_residual_mix_impl or self._parcae_residual_mix_impl
        if self.variant.parcae_loop_update_backend == "manual-autograd":
            state_mix_impl = manual_autograd_parcae_state_mix
            residual_mix_impl = manual_autograd_parcae_residual_mix
        elif self.variant.parcae_loop_update_backend == "triton-glue":
            if self._parcae_triton_backend is None:
                raise RuntimeError("parcae_loop_update_backend=triton-glue requires a configured Triton backend")
            state_mix_impl = self._parcae_triton_backend.parcae_state_mix
            residual_mix_impl = self._parcae_triton_backend.parcae_residual_mix
        elif self.variant.parcae_loop_update_backend == "triton-loop-forward":
            if self._parcae_triton_backend is None:
                raise RuntimeError("parcae_loop_update_backend=triton-loop-forward requires a configured Triton backend")
            state_mix_impl = self._parcae_triton_backend.parcae_state_mix
            residual_mix_impl = self._parcae_triton_backend.parcae_residual_mix

        if self.variant.parcae_band_block_contract == "compiled-direct":
            position_features_by_block = tuple(
                attention_position_features.get(loop_input.shape[-1])
                for _ in recurrent_blocks
            )
            flex_block_masks = tuple(
                block._full_block_flex_block_mask(state, mask)
                if isinstance(block, LocalCausalTransformerBlock)
                else None
                for block in recurrent_blocks
            )
        else:
            position_features_by_block = ()
            flex_block_masks = ()

        def forward_recurrent_block(block_index: int, mixed: torch.Tensor) -> torch.Tensor:
            block = recurrent_blocks[block_index]
            if self.variant.parcae_band_block_contract == "compiled-direct":
                position_features = position_features_by_block[block_index]
                flex_block_mask = flex_block_masks[block_index]
                compiled_full_block = getattr(block, "_compiled_full_block_impl", None)
                if (
                    isinstance(block, LocalCausalTransformerBlock)
                    and compiled_full_block is not None
                ):
                    return compiled_full_block(
                        mixed,
                        mask,
                        position_features,
                        flex_block_mask,
                    )
                return self._forward_block(
                    block,
                    mixed,
                    mask,
                    position_features=position_features,
                )
            return self._forward_block(
                block,
                mixed,
                mask,
                position_features=attention_position_features.get(mixed.shape[-1]),
            )

        controls = ParcaeLoopRegionControls(
            decay=decay,
            injection=injection,
            nonlinear=nonlinear,
        )

        def apply_recurrent_residual(
            block_index: int,
            current_state: torch.Tensor,
            mixed: torch.Tensor,
            block_out: torch.Tensor,
            loop_controls: ParcaeLoopRegionControls,
        ) -> torch.Tensor:
            if (
                self.variant.parcae_loop_update_backend == "triton-loop-forward"
                and block_index == 0
            ):
                with timed_region(f"path1.parcae.band{band_index}.triton_loop_update_forward"):
                    return self._parcae_triton_backend.parcae_loop_update(
                        current_state,
                        loop_controls.decay,
                        loop_controls.injection,
                        block_out,
                        loop_controls.nonlinear,
                    )
            return residual_mix_impl(mixed, block_out, loop_controls.nonlinear)

        loop_result = run_parcae_loop_region(
            initial_state=state,
            controls=controls,
            config=ParcaeLoopRegionConfig(
                loop_count=self.variant.parcae_loop_count,
                gradient_start_step=gradient_start_step,
                recurrent_block_count=len(recurrent_blocks),
                timing_prefix=f"path1.parcae.band{band_index}",
                fuse_first_state_mix=self.variant.parcae_fuse_first_state_mix,
                diagnostics_enabled=self.parcae_runtime_diagnostics,
            ),
            kernels=ParcaeLoopRegionKernels(
                state_mix=state_mix_impl,
                forward_recurrent_block=forward_recurrent_block,
                apply_recurrent_residual=apply_recurrent_residual,
            ),
        )
        state = loop_result.final_state
        norm_history = loop_result.norm_history

        loop_output_projection_impl = (
            self._compiled_parcae_loop_output_projection_impl
            or self._parcae_loop_output_projection_impl
        )
        with timed_region(f"path1.parcae.band{band_index}.loop_output_projection"):
            hidden = loop_output_projection_impl(state, loop_anchor)
        return hidden, norm_history

    def _parcae_band_prepare_impl(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parcae_decay_raw is None or self.parcae_nonlinear_logit is None:
            raise RuntimeError("parcae scaffold is not initialized")
        loop_input = self._parcae_loop_input_projection_impl(hidden)
        decay_rate = F.softplus(self.parcae_decay_raw.to(dtype=hidden.dtype)).view(1, 1, -1)
        if self.variant.parcae_discretization == "stable-exp":
            decay = torch.exp(-decay_rate)
            input_scale = torch.ones_like(decay)
        elif self.variant.parcae_discretization == "zoh":
            if self.parcae_dt_raw is None:
                raise RuntimeError("Parcae ZOH discretization requested without parcae_dt_raw")
            step_size = F.softplus(self.parcae_dt_raw.to(dtype=hidden.dtype)).view(1, 1, -1)
            decay = torch.exp(-step_size * decay_rate)
            input_scale = (1.0 - decay) / decay_rate.clamp_min(1.0e-6)
        else:
            raise RuntimeError(f"unsupported Parcae discretization: {self.variant.parcae_discretization}")
        nonlinear = torch.sigmoid(self.parcae_nonlinear_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        injection_value, injection_gate = self._parcae_injection(loop_input)
        injection = input_scale * injection_gate * injection_value
        return loop_input, decay, injection, nonlinear

    def _normalize_parcae_loop_controls(
        self,
        loop_input: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make prepared loop controls match the recurrent state contract.

        Scalar control parameters are intentionally stored in fp32, but the
        loop-region runtime owns bf16/fp16/fp32 state tensors. Compilation and
        autocast may promote the control math back to fp32, so normalize once at
        the explicit loop boundary before validation/native kernels consume it.
        """

        target_device = loop_input.device
        target_dtype = loop_input.dtype
        return (
            decay.to(device=target_device, dtype=target_dtype).contiguous(),
            injection.to(device=target_device, dtype=target_dtype).contiguous(),
            nonlinear.to(device=target_device, dtype=target_dtype).contiguous(),
        )

    def _parcae_loop_input_projection_impl(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.parcae_prelude_norm is None:
            raise RuntimeError("parcae scaffold is not initialized")
        loop_input = self.parcae_prelude_norm(hidden)
        if self.uses_parcae_hourglass_scaffold:
            if self.parcae_hourglass_down_projection is None:
                raise RuntimeError("parcae hourglass scaffold is not initialized")
            loop_input = self.parcae_hourglass_down_projection(loop_input)
        return loop_input

    def _parcae_loop_output_projection_impl(
        self,
        state: torch.Tensor,
        loop_anchor: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_parcae_hourglass_scaffold:
            return state
        if self.parcae_hourglass_up_projection is None or self.parcae_hourglass_residual_logit is None:
            raise RuntimeError("parcae hourglass scaffold is not initialized")
        loop_delta = self.parcae_hourglass_up_projection(state)
        residual_gate = torch.sigmoid(
            self.parcae_hourglass_residual_logit.to(dtype=state.dtype)
        ).view(1, 1, -1)
        if self._parcae_triton_backend is not None and self.variant.parcae_output_mix_backend == "triton":
            return self._parcae_triton_backend.parcae_output_mix(
                loop_anchor,
                loop_delta,
                residual_gate,
            )
        return loop_anchor + residual_gate * loop_delta

    def _parcae_injection(self, loop_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parcae_injection_logit is None:
            raise RuntimeError("parcae scaffold is not initialized")
        if self.uses_parcae_p20_control:
            if (
                self.parcae_p20_controller is None
                or self.parcae_p20_control_norm is None
                or self.parcae_p20_control_projection is None
            ):
                raise RuntimeError("parcae P20-control scaffold is not initialized")
            seq_len = loop_input.shape[1]
            control_stride = self.variant.parcae_control_stride
            with timed_region("path1.parcae.injection_control_input"):
                control_input = loop_input[:, ::control_stride, :].contiguous() if control_stride != 1 else loop_input
            if self.parcae_p20_position_embedding is not None:
                if self.parcae_p20_position_scale is None:
                    raise RuntimeError("parcae P20 position scale is not initialized")
                if seq_len > self.variant.max_position_embeddings:
                    raise ValueError(
                        f"sequence length {seq_len} exceeds max_position_embeddings "
                        f"{self.variant.max_position_embeddings}"
                    )
                with timed_region("path1.parcae.injection_control_position"):
                    positions = self._position_indices(seq_len, loop_input.device)[::control_stride]
                    position_features = self.parcae_p20_position_embedding(positions).view(1, control_input.shape[1], -1)
                    control_input = control_input + self.parcae_p20_position_scale.to(dtype=loop_input.dtype) * position_features
            with timed_region("path1.parcae.injection_p20_scan"):
                control = self.parcae_p20_controller.scan(control_input).emitted_outputs
            compiled_post_scan = self._compiled_parcae_p20_post_scan_injection_impl
            if compiled_post_scan is not None and control_stride == 1:
                with timed_region("path1.parcae.injection_p20_post_compiled"):
                    injection_value, injection_gate, control = compiled_post_scan(control, loop_input)
            else:
                with timed_region("path1.parcae.injection_p20_norm"):
                    control = self.parcae_p20_control_norm(control)
                with timed_region("path1.parcae.injection_p20_projection"):
                    control_value, control_gate_logits = self.parcae_p20_control_projection(control).chunk(2, dim=-1)
                with timed_region("path1.parcae.injection_p20_gate"):
                    control_gate = torch.sigmoid(control_gate_logits)
                if control_stride != 1:
                    with timed_region("path1.parcae.injection_p20_stride_expand"):
                        control_gate = torch.repeat_interleave(control_gate, repeats=control_stride, dim=1)[:, :seq_len, :]
                        control_value = torch.repeat_interleave(control_value, repeats=control_stride, dim=1)[:, :seq_len, :]
                with timed_region("path1.parcae.injection_p20_value_add"):
                    injection_gate = control_gate
                    injection_value = loop_input + control_value
            self._last_parcae_p20_control_norm = (
                control.detach().float().norm(dim=-1).mean()
                if self.parcae_runtime_diagnostics
                else None
            )
            self._last_parcae_p20_control_steps = int(control.shape[1])
        elif self.uses_parcae_bx:
            if self.parcae_b_value_projection is None or self.parcae_b_gate_projection is None:
                raise RuntimeError("parcae B(x) scaffold is not initialized")
            injection_gate = torch.sigmoid(self.parcae_b_gate_projection(loop_input))
            injection_value = self.parcae_b_value_projection(loop_input)
            self._last_parcae_p20_control_norm = None
            self._last_parcae_p20_control_steps = None
        else:
            injection_gate = torch.sigmoid(self.parcae_injection_logit.to(dtype=loop_input.dtype)).view(1, 1, -1)
            injection_value = loop_input
            self._last_parcae_p20_control_norm = None
            self._last_parcae_p20_control_steps = None
        if self.parcae_runtime_diagnostics:
            with timed_region("path1.parcae.injection_diagnostic_reduction"):
                self._last_parcae_injection_gate_mean = injection_gate.detach().float().mean()
                self._last_parcae_injection_norm = (injection_gate * injection_value).detach().float().norm(dim=-1).mean()
        else:
            self._last_parcae_injection_gate_mean = None
            self._last_parcae_injection_norm = None
        return injection_value, injection_gate

    def _parcae_loop_iteration_impl(
        self,
        current_state: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
        nonlinear: torch.Tensor,
        mask: torch.Tensor | None,
        position_features_by_block: tuple[torch.Tensor | None, ...],
        flex_block_masks: tuple[object | None, ...],
        ) -> torch.Tensor:
        mixed = self._parcae_state_mix_impl(current_state, decay, injection)
        recurrent_blocks = self.blocks[self.parcae_loop_start : self.parcae_loop_end]
        for block_index, block in enumerate(recurrent_blocks):
            position_features = position_features_by_block[block_index]
            flex_block_mask = flex_block_masks[block_index]
            if isinstance(block, LocalCausalTransformerBlock):
                block_out = block._full_block_impl_no_timing(
                    mixed,
                    mask,
                    position_features,
                    flex_block_mask,
                )
            else:
                block_out = self._forward_block(
                    block,
                    mixed,
                    mask,
                    position_features=position_features,
                )
            mixed = self._parcae_residual_mix_impl(mixed, block_out, nonlinear)
        return mixed

    def _parcae_state_mix_impl(
        self,
        state: torch.Tensor,
        decay: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return decay * state + injection

    def _parcae_residual_mix_impl(
        self,
        mixed: torch.Tensor,
        block_out: torch.Tensor,
        nonlinear: torch.Tensor,
    ) -> torch.Tensor:
        return mixed + nonlinear * (block_out - mixed)

    def _parcae_p20_post_scan_injection_impl(
        self,
        control: torch.Tensor,
        loop_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parcae_p20_control_norm is None or self.parcae_p20_control_projection is None:
            raise RuntimeError("parcae P20-control scaffold is not initialized")
        control = self.parcae_p20_control_norm(control)
        control_value, control_gate_logits = self.parcae_p20_control_projection(control).chunk(2, dim=-1)
        return loop_input + control_value, torch.sigmoid(control_gate_logits), control

    def diagnostic_payload(self) -> dict[str, Any]:
        reference_ssm_blocks: list[dict[str, Any]] = []
        composite_branch_means: dict[str, list[float]] = {}
        for block in self.blocks:
            block_diagnostic = getattr(block, "diagnostic_payload", None)
            if not callable(block_diagnostic):
                continue
            payload = block_diagnostic()
            if not payload:
                continue
            reference_ssm_blocks.append(payload)
            mixer_payload = payload.get("mixer", {})
            if not isinstance(mixer_payload, dict):
                continue
            for branch in mixer_payload.get("branches", []):
                if not isinstance(branch, dict):
                    continue
                branch_name = branch.get("branch")
                mean_weight = branch.get("mean_weight")
                if isinstance(branch_name, str) and isinstance(mean_weight, float):
                    composite_branch_means.setdefault(branch_name, []).append(mean_weight)

        diagnostics: dict[str, Any] = {}
        diagnostics["parameter_count"] = sum(parameter.numel() for parameter in self.parameters())
        diagnostics["trainable_parameter_count"] = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        diagnostics["scaffold_profile"] = self.variant.scaffold_profile.value
        diagnostics["feed_forward_profile"] = self.variant.feed_forward_profile.value
        diagnostics["attention_kernel"] = self.variant.shape.attention_kernel.value
        diagnostics["position_encoding_kind"] = self.variant.position_encoding_kind
        diagnostics["attention_position_contract"] = self.variant.attention_position_contract
        diagnostics["max_position_embeddings"] = self.variant.max_position_embeddings
        diagnostics["head_loss_backend"] = self._head_loss_backend
        diagnostics["ffn_backend"] = self.variant_runtime_ffn_backend
        diagnostics["parcae_control_position_kind"] = self.variant.parcae_control_position_kind
        diagnostics["parcae_control_position_scale_init"] = self.variant.parcae_control_position_scale_init
        diagnostics["parcae_control_stride"] = self.variant.parcae_control_stride
        diagnostics["attention_position_embedding_widths"] = sorted(
            int(width_key) for width_key in self.attention_position_embeddings.keys()
        )
        if self.parcae_p20_position_scale is not None:
            diagnostics["parcae_control_position_scale"] = float(
                self.parcae_p20_position_scale.detach().float().item()
            )
        if self.variant.feed_forward_profile is not FeedForwardProfile.STANDARD:
            selected_layers = self.variant.feed_forward_layer_indices
            diagnostics["eml_inspired_feed_forward"] = {
                "slot_count": self.variant.eml_slot_count,
                "tree_depth": self.variant.eml_tree_depth,
                "leaf_count": 1 << self.variant.eml_tree_depth,
                "layer_indices": selected_layers if selected_layers is not None else "all",
                "route_fraction": self.variant.eml_route_fraction,
                "label": "EML-inspired real-valued tree, not exact complex EML",
            }
            feed_forward_experts: list[dict[str, Any]] = []
            for layer_index, block in enumerate(self.blocks):
                ffn = getattr(block, "ffn", None)
                ffn_diagnostic = getattr(ffn, "diagnostic_payload", None)
                if callable(ffn_diagnostic):
                    feed_forward_experts.append(
                        {
                            "layer_index": layer_index,
                            "payload": ffn_diagnostic(),
                        }
                    )
            diagnostics["feed_forward_experts"] = feed_forward_experts
        if reference_ssm_blocks:
            diagnostics["reference_ssm_blocks"] = reference_ssm_blocks
        if composite_branch_means:
            diagnostics["composite_branch_weight_summary"] = {
                branch_name: {
                    "layer_count": len(values),
                    "mean_weight_across_layers": sum(values) / len(values),
                    "min_layer_mean_weight": min(values),
                    "max_layer_mean_weight": max(values),
                }
                for branch_name, values in sorted(composite_branch_means.items())
            }
        if self.uses_pr5_scaffold:
            diagnostics["scaffold"] = {
                "profile": self.variant.scaffold_profile.value,
                "hash_context_embedding": self.context_embedding is not None,
                "smear_gate": self.smear_gate is not None,
            }
        if self.uses_parcae_scaffold:
            if (
                self.parcae_decay_raw is None
                or self.parcae_injection_logit is None
                or self.parcae_nonlinear_logit is None
            ):
                raise RuntimeError("parcae scaffold is not initialized")
            decay = torch.exp(-F.softplus(self.parcae_decay_raw.detach().float()))
            injection = torch.sigmoid(self.parcae_injection_logit.detach().float())
            nonlinear = torch.sigmoid(self.parcae_nonlinear_logit.detach().float())
            last_loop_end = self.parcae_loop_ranges[-1][1] if self.parcae_loop_ranges else self.parcae_loop_end
            recurrent_blocks = [
                block
                for start, end in self.parcae_loop_ranges
                for block in self.blocks[start:end]
            ]
            compiled_recurrent_blocks = sum(
                1
                for block in recurrent_blocks
                if getattr(block, "_compiled_full_block_impl", None) is not None
            )
            if compiled_recurrent_blocks == 0:
                recurrent_block_backend = "standard"
            elif compiled_recurrent_blocks == len(recurrent_blocks):
                recurrent_block_backend = "compiled-full-block"
            else:
                recurrent_block_backend = "mixed"
            diagnostics["parcae_looped_attention"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "hourglass_pass_count": self.variant.parcae_hourglass_pass_count,
                "hourglass_band_schedule": self.variant.parcae_hourglass_band_schedule,
                "backward_steps": self.variant.parcae_backward_steps or self.variant.parcae_loop_count,
                "prelude_norm_kind": self.variant.parcae_prelude_norm_kind,
                "discretization": self.variant.parcae_discretization,
                "prelude_layers": self.parcae_loop_start,
                "recurrent_layers": self.parcae_loop_end - self.parcae_loop_start,
                "recurrent_layers_total": sum(end - start for start, end in self.parcae_loop_ranges),
                "loop_ranges": self.parcae_loop_ranges,
                "configured_loop_layer_count": self.variant.parcae_loop_layer_count,
                "coda_layers": len(self.blocks) - last_loop_end,
                "recurrent_block_backend": recurrent_block_backend,
                "runtime_diagnostics": self.parcae_runtime_diagnostics,
                "loop_projection_backend": (
                    "compiled"
                    if (
                        self._compiled_parcae_loop_input_projection_impl is not None
                        and self._compiled_parcae_loop_output_projection_impl is not None
                    )
                    else "standard"
                ),
                "wide_d_model": self.parcae_wide_d_model,
                "loop_d_model": self.parcae_loop_d_model,
                "hourglass": self.uses_parcae_hourglass_scaffold,
                "attention_position_contract": self.variant.attention_position_contract,
                "control_position_kind": self.variant.parcae_control_position_kind,
                "control_stride": self.variant.parcae_control_stride,
                "control_state_transform": self.variant.parcae_control_state_transform,
                "recurrent_compile_mode": self.variant.parcae_recurrent_compile_mode,
                "scaffold_backend": (
                    "compiled"
                    if (
                        self._compiled_parcae_loop_bands_impl is not None
                        or self._compiled_parcae_loop_impl is not None
                    )
                    else self.variant.parcae_scaffold_backend
                ),
                "band_block_contract": self.variant.parcae_band_block_contract,
                "band_prepare_backend": (
                    "compiled"
                    if self._compiled_parcae_band_prepare_impl is not None
                    else self.variant.parcae_band_prepare_backend
                ),
                "output_mix_backend": self.variant.parcae_output_mix_backend,
                "first_state_mix_fused": self.variant.parcae_fuse_first_state_mix,
                "loop_update_backend": (
                    "compiled"
                    if self._compiled_parcae_loop_iteration_impl is not None
                    else self.variant.parcae_loop_update_backend
                ),
                "loop_update_triton": self._parcae_triton_backend is not None,
                "p20_control_projection": "packed-value-gate" if self.uses_parcae_p20_control else None,
                "control_position_scale": (
                    float(self.parcae_p20_position_scale.detach().float().item())
                    if self.parcae_p20_position_scale is not None
                    else None
                ),
                "decay_mean": float(decay.mean().item()),
                "decay_min": float(decay.min().item()),
                "decay_max": float(decay.max().item()),
                "injection_mean": float(injection.mean().item()),
                "last_injection_gate_mean": _diagnostic_float(self._last_parcae_injection_gate_mean),
                "last_injection_norm": _diagnostic_float(self._last_parcae_injection_norm),
                "last_p20_control_norm": _diagnostic_float(self._last_parcae_p20_control_norm),
                "last_p20_control_steps": self._last_parcae_p20_control_steps,
                "nonlinear_delta_scale_mean": float(nonlinear.mean().item()),
                "last_recurrent_state_norms": _diagnostic_float_list(self._last_parcae_norms),
                "label": "Parcae-inspired stable middle-loop scaffold, not an exact reproduction",
            }
        return diagnostics

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
        head_loss_backend: str = "dense",
        ffn_backend: str = "dense",
        parcae_runtime_diagnostics: bool | None = None,
    ) -> None:
        if head_loss_backend not in HEAD_LOSS_BACKENDS:
            raise ValueError(f"unsupported head_loss_backend: {head_loss_backend}")
        if ffn_backend not in FFN_BACKENDS:
            raise ValueError(f"unsupported ffn_backend: {ffn_backend}")
        self.variant_runtime_ffn_backend = ffn_backend
        self.parcae_runtime_diagnostics = (
            ffn_backend != "compiled"
            if parcae_runtime_diagnostics is None
            else parcae_runtime_diagnostics
        )
        self._head_loss_backend = head_loss_backend
        self._compiled_head_loss_impl = None
        self._compiled_parcae_p20_post_scan_injection_impl = None
        self._compiled_parcae_loop_input_projection_impl = None
        self._compiled_parcae_loop_output_projection_impl = None
        self._compiled_parcae_band_prepare_impl = None
        self._compiled_parcae_state_mix_impl = None
        self._compiled_parcae_residual_mix_impl = None
        self._compiled_parcae_loop_iteration_impl = None
        if head_loss_backend == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("head_loss_backend=compiled requires torch.compile")
            self._compiled_head_loss_impl = torch.compile(
                self._head_loss_impl,
                mode="reduce-overhead",
            )
        for block in self.blocks:
            configure = getattr(block, "configure_runtime_policy", None)
            if callable(configure):
                available = inspect.signature(configure).parameters
                kwargs: dict[str, object] = {"compile_mode": compile_mode}
                if "primitive_runtime_backend" in available:
                    kwargs["primitive_runtime_backend"] = primitive_runtime_backend
                if "ffn_backend" in available:
                    kwargs["ffn_backend"] = ffn_backend
                configure(**kwargs)
        if self.parcae_p20_controller is not None:
            self.parcae_p20_controller.configure_runtime_policy(
                compile_mode=compile_mode,
                primitive_runtime_backend=primitive_runtime_backend,
            )
            post_scan_compile_mode = compile_mode or ("reduce-overhead" if ffn_backend == "compiled" else None)
            if post_scan_compile_mode is not None:
                if not hasattr(torch, "compile"):
                    raise RuntimeError("Parcae P20 post-scan compile requires torch.compile")
                self._compiled_parcae_p20_post_scan_injection_impl = torch.compile(
                    self._parcae_p20_post_scan_injection_impl,
                    mode=post_scan_compile_mode,
                )
        if self.uses_parcae_scaffold and self.variant.parcae_loop_update_backend in {
            "triton-glue",
            "triton-loop-forward",
        }:
            ensure_triton_runtime_available()
            self._parcae_triton_backend = build_triton_primitive_backend()
        if self.uses_parcae_scaffold and ffn_backend == "compiled":
            if not hasattr(torch, "compile"):
                raise RuntimeError("Parcae loop-region compile requires torch.compile")
            self._compiled_parcae_loop_input_projection_impl = torch.compile(
                self._parcae_loop_input_projection_impl,
                mode="reduce-overhead",
            )
            self._compiled_parcae_loop_output_projection_impl = torch.compile(
                self._parcae_loop_output_projection_impl,
                mode="reduce-overhead",
            )
            if self.variant.parcae_band_prepare_backend == "compiled":
                self._compiled_parcae_band_prepare_impl = torch.compile(
                    self._parcae_band_prepare_impl,
                    mode="reduce-overhead",
                )
            if self.variant.parcae_scaffold_backend == "compiled":
                self._compiled_parcae_loop_bands_impl = torch.compile(
                    self._forward_parcae_loop_bands,
                    mode=self.variant.parcae_recurrent_compile_mode,
                )
                self._compiled_parcae_loop_impl = torch.compile(
                    self._forward_parcae_loop,
                    mode=self.variant.parcae_recurrent_compile_mode,
                )
            if self.variant.parcae_loop_update_backend == "compiled":
                self._compiled_parcae_loop_iteration_impl = torch.compile(
                    self._parcae_loop_iteration_impl,
                    mode=self.variant.parcae_recurrent_compile_mode,
                )
            else:
                for start, end in self.parcae_loop_ranges:
                    for block in self.blocks[start:end]:
                        configure_full_block_compile = getattr(block, "configure_full_block_compile", None)
                        if callable(configure_full_block_compile):
                            configure_full_block_compile(
                                enabled=True,
                                compile_mode=self.variant.parcae_recurrent_compile_mode,
                            )

    def optimizer_parameter_groups(self, base_lr: float) -> list[dict[str, object]]:
        if self.uses_pr5_scaffold:
            groups: dict[str, dict[str, object]] = {
                "default": {"name": "default", "params": [], "lr": base_lr},
                "pr5_context": {"name": "pr5_context", "params": [], "lr": base_lr},
                "pr5_recurrent": {"name": "pr5_recurrent", "params": [], "lr": base_lr * 0.5},
                "pr5_gates_controls": {
                    "name": "pr5_gates_controls",
                    "params": [],
                    "lr": base_lr * 0.5,
                },
                "pr5_readout": {"name": "pr5_readout", "params": [], "lr": base_lr},
                "pr5_scalars": {"name": "pr5_scalars", "params": [], "lr": base_lr},
            }
            scalar_markers = (
                "attention_scale",
                "ffn_scale",
                "mixer_scale",
                "feedforward_scale",
                "residual_mix",
                "readout_ramp_logit",
                "p20_to_gdn_ramp_logit",
                "qkv_ramp_logit",
                "beta_ramp_logit",
                "context_embedding.scale",
                "smear_gate.gate",
            )
            gate_markers = (
                ".mixer.control_projection",
                ".mixer.p20_condition_projection",
                ".mixer.conditioner.delta_projection",
                ".mixer.beta_state_weight",
                ".mixer.query_state_scale",
                ".mixer.key_state_scale",
                ".mixer.value_state_scale",
                ".mixer.aux_query_state_scale",
            )
            recurrent_markers = (
                ".mixer.qkv_projection",
                ".mixer.q_local",
                ".mixer.k_local",
                ".mixer.v_local",
                ".mixer.state_transform_projection",
                ".mixer.conditioner.input_projection",
                ".mixer.conditioner.primitive",
                ".mixer.conditioner.norm",
                ".mixer.p20.",
            )
            readout_markers = (
                ".mixer.output_norm",
                ".mixer.output_projection",
                ".mixer.matrix_read_norm",
                ".mixer.vector_read_norm",
                ".mixer.matrix_output_projection",
                ".mixer.aux_matrix_output_projection",
                ".mixer.vector_output_projection",
                ".feedforward",
                ".ffn",
                ".output_norm",
                "final_norm",
                "output.",
            )
            context_markers = (
                "embedding.",
                "context_embedding.embedding",
                "context_embedding.projection",
            )
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if any(marker in name for marker in scalar_markers):
                    groups["pr5_scalars"]["params"].append(param)
                elif any(marker in name for marker in gate_markers):
                    groups["pr5_gates_controls"]["params"].append(param)
                elif any(marker in name for marker in recurrent_markers):
                    groups["pr5_recurrent"]["params"].append(param)
                elif any(marker in name for marker in readout_markers):
                    groups["pr5_readout"]["params"].append(param)
                elif any(marker in name for marker in context_markers):
                    groups["pr5_context"]["params"].append(param)
                else:
                    groups["default"]["params"].append(param)
            return [group for group in groups.values() if group["params"]]

        if self.variant.primitive_profile is not PrimitiveProfile.P20_GDN_ROLE:
            return [{"name": "default", "params": list(self.parameters()), "lr": base_lr}]

        groups: dict[str, dict[str, object]] = {
            "default": {"name": "default", "params": [], "lr": base_lr},
            "p20_gdn_recurrent": {
                "name": "p20_gdn_recurrent",
                "params": [],
                "lr": base_lr * 0.5,
            },
            "p20_gdn_gates": {
                "name": "p20_gdn_gates",
                "params": [],
                "lr": base_lr * 0.5,
            },
            "p20_gdn_readout": {
                "name": "p20_gdn_readout",
                "params": [],
                "lr": base_lr,
            },
            "p20_gdn_scalars": {
                "name": "p20_gdn_scalars",
                "params": [],
                "lr": base_lr,
            },
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".primitive.in_projection" in name or any(
                marker in name for marker in (".primitive.q_local", ".primitive.k_local", ".primitive.v_local")
            ):
                groups["p20_gdn_recurrent"]["params"].append(param)
            elif ".primitive.control_projection" in name:
                groups["p20_gdn_gates"]["params"].append(param)
            elif any(marker in name for marker in (".primitive.output_norm", ".primitive.output_projection")):
                groups["p20_gdn_readout"]["params"].append(param)
            elif ".primitive.readout_ramp_logit" in name or name.endswith(".residual_scale"):
                groups["p20_gdn_scalars"]["params"].append(param)
            else:
                groups["default"]["params"].append(param)
        return [group for group in groups.values() if group["params"]]


def build_path1_model(variant: Path1VariantSpec, *, dtype_mode: str) -> Path1HybridLanguageModel:
    return Path1HybridLanguageModel(variant, dtype_mode=dtype_mode)
