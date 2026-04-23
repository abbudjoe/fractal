from __future__ import annotations

import math
from typing import Any

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
from python.models.primitives import (
    P20RotaryStateOutputRuntimeSequenceMixer,
    PrimitiveMixerBlock,
)
from python.models.reference_ssm import ReferenceSsmHybridBlock
from python.models.transformer import (
    DepthAugmentedCausalSelfAttention,
    LocalCausalSelfAttention,
    LocalCausalTransformerBlock,
    PaperMoDACausalSelfAttention,
    PaperMoDTrainTopCTransformerBlock,
    Pr5LocalCausalTransformerBlock,
    RotarySoftGatedLocalCausalTransformerBlock,
    SoftGatedLocalCausalTransformerBlock,
    TokenRoutedLocalCausalTransformerBlock,
    causal_topk_token_positions,
    local_causal_attention_bias,
    selected_positions_to_mask,
)
from python.specs.path1 import (
    AttentionProfile,
    FeedForwardProfile,
    HybridAttentionLayerRole,
    Path1ScaffoldProfile,
    Path1VariantSpec,
    PrimitiveProfile,
    RecurrentHaltingProfile,
    RecurrentTokenRoutingProfile,
    TokenRoutingProfile,
)
from python.specs.runtime import PrimitiveStateTransformMode


def _build_attention_feed_forward(
    variant: Path1VariantSpec, *, layer_index: int
) -> nn.Module | None:
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
    raise ValueError(
        f"unsupported feed-forward profile: {variant.feed_forward_profile}"
    )


def _build_attention_module(variant: Path1VariantSpec) -> LocalCausalSelfAttention:
    if variant.attention_profile is AttentionProfile.STANDARD:
        return LocalCausalSelfAttention(
            variant.shape.d_model,
            variant.shape.head_count,
        )
    if variant.attention_profile is AttentionProfile.MODA_DEPTH_KV:
        return DepthAugmentedCausalSelfAttention(
            variant.shape.d_model,
            variant.shape.head_count,
        )
    if variant.attention_profile is AttentionProfile.PAPER_MODA_DEPTH_KV:
        return PaperMoDACausalSelfAttention(
            variant.shape.d_model,
            variant.shape.head_count,
        )
    raise ValueError(f"unsupported attention profile: {variant.attention_profile}")


def _uses_token_routing(variant: Path1VariantSpec, *, layer_index: int) -> bool:
    if variant.token_routing_profile is TokenRoutingProfile.NONE:
        return False
    return (
        variant.token_routing_layer_indices is None
        or layer_index in variant.token_routing_layer_indices
    )


def _build_token_routed_block(
    variant: Path1VariantSpec,
    *,
    layer_index: int,
    attention: LocalCausalSelfAttention,
) -> LocalCausalTransformerBlock:
    kwargs = {
        "route_fraction": variant.token_route_fraction,
        "attention_module": attention,
        "ffn_module": _build_attention_feed_forward(variant, layer_index=layer_index),
    }
    if variant.token_routing_profile is TokenRoutingProfile.CAUSAL_TOPK_BLOCK:
        return TokenRoutedLocalCausalTransformerBlock(
            variant.shape.d_model,
            variant.shape.head_count,
            variant.shape.d_ff,
            **kwargs,
        )
    if variant.token_routing_profile is TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK:
        return PaperMoDTrainTopCTransformerBlock(
            variant.shape.d_model,
            variant.shape.head_count,
            variant.shape.d_ff,
            **kwargs,
        )
    if variant.token_routing_profile is TokenRoutingProfile.SOFT_GATE_BLOCK:
        return SoftGatedLocalCausalTransformerBlock(
            variant.shape.d_model,
            variant.shape.head_count,
            variant.shape.d_ff,
            gate_floor=variant.token_route_fraction,
            attention_module=attention,
            ffn_module=_build_attention_feed_forward(variant, layer_index=layer_index),
        )
    if variant.token_routing_profile is TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK:
        return RotarySoftGatedLocalCausalTransformerBlock(
            variant.shape.d_model,
            variant.shape.head_count,
            variant.shape.d_ff,
            gate_floor=variant.token_route_fraction,
            attention_module=attention,
            ffn_module=_build_attention_feed_forward(variant, layer_index=layer_index),
        )
    raise ValueError(
        f"unsupported token routing profile: {variant.token_routing_profile}"
    )


class Pr5HashContextEmbedding(nn.Module):
    """Small n-gram context side channel from the PR5 HybridGDN contract."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        hash_vocab_size: int = 3072,
        hash_dim: int = 112,
    ) -> None:
        super().__init__()
        if hash_vocab_size <= 1:
            raise ValueError(
                f"hash_vocab_size must be greater than 1, got {hash_vocab_size}"
            )
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
        hashed[..., 1:] = (
            torch.bitwise_xor(36313 * typed[..., 1:], 27191 * typed[..., :-1]) % modulus
        )
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
        context = self.embedding(self._bigram_hash(input_ids)) + self.embedding(
            self._trigram_hash(input_ids)
        )
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


def _middle_loop_bounds(total_layers: int) -> tuple[int, int]:
    loop_width = max(1, total_layers // 3)
    start = max(0, (total_layers - loop_width) // 2)
    end = min(total_layers, start + loop_width)
    return start, end


def _parcae_fractional_control_width(
    d_model: int, *, divisor: int, label: str
) -> int:
    width = d_model // divisor
    width -= width % 4
    if width < 4:
        raise ValueError(
            f"{label} Parcae P20 control requires a projected width >= 4 and "
            f"divisible by 4, got d_model={d_model}"
        )
    return width


def _parcae_thin_control_width(d_model: int) -> int:
    return _parcae_fractional_control_width(d_model, divisor=2, label="thin")


def _parcae_quarter_control_width(d_model: int) -> int:
    return _parcae_fractional_control_width(d_model, divisor=4, label="quarter")


def _mean_l2_norm(value: torch.Tensor) -> float:
    return float(value.detach().float().norm(dim=-1).mean().item())


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def _global_l2_norm(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().norm()


def _safe_float_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((numerator / denominator.clamp_min(1.0e-6)).item())


def _nan_or_inf_seen(*values: torch.Tensor) -> float:
    seen = False
    for value in values:
        seen = seen or bool((~torch.isfinite(value.detach().float())).any().item())
    return float(seen)


class Path1HybridLanguageModel(nn.Module):
    def __init__(self, variant: Path1VariantSpec, *, dtype_mode: str) -> None:
        super().__init__()
        variant.validate()
        self.variant = variant
        self.uses_pr5_scaffold = (
            variant.scaffold_profile is Path1ScaffoldProfile.PR5_HYBRID_GDN
        )
        self.uses_parcae_scaffold = variant.scaffold_profile in {
            Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
        }
        self.uses_parcae_bx = (
            variant.scaffold_profile is Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION
        )
        self.uses_parcae_full_p20_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_thin_p20_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_thin_p20_gate_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_quarter_p20_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_thin_p20_value_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_thin_p20_gate_baseblend = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION
        )
        self.uses_parcae_thin_p20_baseblend_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION
        )
        self.uses_parcae_p20_mod_gate_bias_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION
        )
        self.uses_parcae_p20_mod_value_scale_control = (
            variant.scaffold_profile
            is Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION
        )
        self.uses_parcae_p20_mod_control = (
            self.uses_parcae_p20_mod_gate_bias_control
            or self.uses_parcae_p20_mod_value_scale_control
        )
        self.uses_parcae_p20_control = (
            self.uses_parcae_full_p20_control
            or self.uses_parcae_thin_p20_control
            or self.uses_parcae_thin_p20_gate_control
            or self.uses_parcae_quarter_p20_control
            or self.uses_parcae_thin_p20_value_control
            or self.uses_parcae_thin_p20_gate_baseblend
            or self.uses_parcae_thin_p20_baseblend_control
            or self.uses_parcae_p20_mod_control
        )
        self.uses_parcae_p20_gate_only_control = (
            self.uses_parcae_thin_p20_gate_control
            or self.uses_parcae_thin_p20_gate_baseblend
        )
        self.uses_parcae_p20_value_only_control = (
            self.uses_parcae_thin_p20_value_control
        )
        self.uses_parcae_p20_base_gate_blend = (
            self.uses_parcae_thin_p20_gate_baseblend
            or self.uses_parcae_thin_p20_baseblend_control
        )
        self.parcae_p20_control_mode = (
            "full"
            if self.uses_parcae_full_p20_control
            else "thin"
            if self.uses_parcae_thin_p20_control
            else "thin-gate-only"
            if self.uses_parcae_thin_p20_gate_control
            else "quarter"
            if self.uses_parcae_quarter_p20_control
            else "thin-value-only"
            if self.uses_parcae_thin_p20_value_control
            else "thin-gate-baseblend"
            if self.uses_parcae_thin_p20_gate_baseblend
            else "thin-baseblend"
            if self.uses_parcae_thin_p20_baseblend_control
            else "mod-gate-bias"
            if self.uses_parcae_p20_mod_gate_bias_control
            else "mod-value-scale"
            if self.uses_parcae_p20_mod_value_scale_control
            else "none"
        )
        self.uses_looped_transformer_scaffold = variant.scaffold_profile in {
            Path1ScaffoldProfile.FIXED_LOOPED_LM,
            Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
            Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
        }
        self.uses_universal_transformer_scaffold = variant.scaffold_profile in {
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
            Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
        }
        self.uses_universal_transformer_act = (
            variant.scaffold_profile is Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT
        )
        self.uses_ouro_learned_exit = (
            variant.scaffold_profile is Path1ScaffoldProfile.OURO_LEARNED_EXIT
        )
        self.uses_rrt_cycle_scaffold = (
            variant.scaffold_profile is Path1ScaffoldProfile.RRT_CYCLE
        )
        self.uses_mor_expert_choice_scaffold = (
            variant.scaffold_profile is Path1ScaffoldProfile.MOR_EXPERT_CHOICE
        )
        self.uses_looped_additive_input = (
            variant.scaffold_profile is Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT
        )
        self.uses_huginn_adapter_recurrence = (
            variant.scaffold_profile is Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE
        )
        self.uses_depth_attention = variant.attention_profile in {
            AttentionProfile.MODA_DEPTH_KV,
            AttentionProfile.PAPER_MODA_DEPTH_KV,
        }
        self.uses_paper_moda_attention = (
            variant.attention_profile is AttentionProfile.PAPER_MODA_DEPTH_KV
        )
        self.uses_recurrent_token_routing = (
            variant.recurrent_token_routing_profile
            is RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE
        )
        self.embedding = nn.Embedding(
            variant.shape.vocab_size,
            variant.shape.d_model,
        )
        self.context_embedding = (
            Pr5HashContextEmbedding(variant.shape.vocab_size, variant.shape.d_model)
            if self.uses_pr5_scaffold
            else None
        )
        self.smear_gate = (
            SmearGate(variant.shape.d_model) if self.uses_pr5_scaffold else None
        )
        if self.uses_pr5_scaffold:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.005)
        self.blocks = nn.ModuleList()
        self.rrt_effective_layer_count = variant.shape.total_layers
        self.rrt_stored_layer_count = (
            variant.shape.total_layers // variant.parcae_loop_count
            if self.uses_rrt_cycle_scaffold
            else variant.shape.total_layers
        )
        build_schedule = (
            variant.layer_schedule[: self.rrt_stored_layer_count]
            if self.uses_rrt_cycle_scaffold
            else variant.layer_schedule
        )
        shared_attention: LocalCausalSelfAttention | None = None
        reference_ordinal = 0
        for layer_index, role in enumerate(build_schedule):
            if role is HybridAttentionLayerRole.EXACT_ATTENTION:
                attention = _build_attention_module(variant)
                if shared_attention is None:
                    shared_attention = attention
                attention_block = (
                    Pr5LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=attention,
                    )
                    if self.uses_pr5_scaffold
                    else (
                        _build_token_routed_block(
                            variant,
                            layer_index=layer_index,
                            attention=attention,
                        )
                        if _uses_token_routing(variant, layer_index=layer_index)
                        else LocalCausalTransformerBlock(
                            variant.shape.d_model,
                            variant.shape.head_count,
                            variant.shape.d_ff,
                            attention_module=attention,
                            ffn_module=_build_attention_feed_forward(
                                variant, layer_index=layer_index
                            ),
                        )
                    )
                )
                self.blocks.append(attention_block)
            elif role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION:
                if shared_attention is None:
                    shared_attention = _build_attention_module(variant)
                attention_block = (
                    Pr5LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=shared_attention,
                    )
                    if self.uses_pr5_scaffold
                    else (
                        _build_token_routed_block(
                            variant,
                            layer_index=layer_index,
                            attention=shared_attention,
                        )
                        if _uses_token_routing(variant, layer_index=layer_index)
                        else LocalCausalTransformerBlock(
                            variant.shape.d_model,
                            variant.shape.head_count,
                            variant.shape.d_ff,
                            attention_module=shared_attention,
                            ffn_module=_build_attention_feed_forward(
                                variant, layer_index=layer_index
                            ),
                        )
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
            self.parcae_loop_start, self.parcae_loop_end = _middle_loop_bounds(
                len(self.blocks)
            )
            self.parcae_prelude_norm = nn.LayerNorm(variant.shape.d_model)
            self.parcae_decay_raw = nn.Parameter(
                torch.full((variant.shape.d_model,), -2.0, dtype=torch.float32)
            )
            self.parcae_injection_logit = nn.Parameter(
                torch.full((variant.shape.d_model,), -2.1972246, dtype=torch.float32)
            )
            self.parcae_nonlinear_logit = nn.Parameter(
                torch.zeros(variant.shape.d_model, dtype=torch.float32)
            )
            self.parcae_b_value_projection = (
                nn.Linear(variant.shape.d_model, variant.shape.d_model, bias=False)
                if self.uses_parcae_bx
                else None
            )
            self.parcae_b_gate_projection = (
                nn.Linear(variant.shape.d_model, variant.shape.d_model)
                if self.uses_parcae_bx
                else None
            )
            if self.parcae_b_value_projection is not None:
                nn.init.eye_(self.parcae_b_value_projection.weight)
            if self.parcae_b_gate_projection is not None:
                nn.init.zeros_(self.parcae_b_gate_projection.weight)
                nn.init.constant_(self.parcae_b_gate_projection.bias, -2.1972246)
            if self.uses_parcae_quarter_p20_control:
                self.parcae_p20_control_width = _parcae_quarter_control_width(
                    variant.shape.d_model
                )
            elif self.uses_parcae_p20_control and (
                self.uses_parcae_thin_p20_control
                or self.uses_parcae_thin_p20_gate_control
                or self.uses_parcae_thin_p20_value_control
                or self.uses_parcae_thin_p20_gate_baseblend
                or self.uses_parcae_thin_p20_baseblend_control
            ):
                self.parcae_p20_control_width = _parcae_thin_control_width(
                    variant.shape.d_model
                )
            elif self.uses_parcae_p20_control:
                self.parcae_p20_control_width = variant.shape.d_model
            else:
                self.parcae_p20_control_width = 0
            self.parcae_p20_input_projection = (
                nn.Linear(
                    variant.shape.d_model,
                    self.parcae_p20_control_width,
                    bias=False,
                )
                if self.uses_parcae_p20_control
                and self.parcae_p20_control_width != variant.shape.d_model
                else None
            )
            self.parcae_p20_controller = (
                P20RotaryStateOutputRuntimeSequenceMixer(
                    self.parcae_p20_control_width,
                    state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                )
                if self.uses_parcae_p20_control
                else None
            )
            self.parcae_p20_control_norm = (
                SimpleRmsNorm(self.parcae_p20_control_width)
                if self.uses_parcae_p20_control
                else None
            )
            self.parcae_p20_value_projection = (
                nn.Linear(
                    self.parcae_p20_control_width,
                    variant.shape.d_model,
                    bias=False,
                )
                if self.uses_parcae_p20_control
                and not self.uses_parcae_p20_gate_only_control
                else None
            )
            self.parcae_p20_gate_projection = (
                nn.Linear(self.parcae_p20_control_width, variant.shape.d_model)
                if self.uses_parcae_p20_control
                and not self.uses_parcae_p20_value_only_control
                else None
            )
            if self.parcae_p20_input_projection is not None:
                nn.init.orthogonal_(self.parcae_p20_input_projection.weight)
            if self.parcae_p20_value_projection is not None:
                nn.init.normal_(
                    self.parcae_p20_value_projection.weight, mean=0.0, std=1.0e-3
                )
            if self.parcae_p20_gate_projection is not None:
                nn.init.normal_(
                    self.parcae_p20_gate_projection.weight, mean=0.0, std=1.0e-3
                )
                nn.init.constant_(self.parcae_p20_gate_projection.bias, -2.1972246)
            self._last_parcae_norms: list[float] = []
            self._last_parcae_injection_gate_mean: float | None = None
            self._last_parcae_injection_norm: float | None = None
            self._last_parcae_p20_control_norm: float | None = None
            self._last_parcae_control_diagnostics: dict[str, float | None] = {}
            self.parcae_mod_router_norm = (
                nn.LayerNorm(variant.shape.d_model)
                if self.uses_parcae_p20_mod_control
                else None
            )
            self.parcae_mod_router = (
                nn.Linear(variant.shape.d_model, 1)
                if self.uses_parcae_p20_mod_control
                else None
            )
            self.parcae_mod_gate_bias_scale = (
                nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
                if self.uses_parcae_p20_mod_gate_bias_control
                else None
            )
            self.parcae_mod_value_scale_strength = (
                nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                if self.uses_parcae_p20_mod_value_scale_control
                else None
            )
            if self.parcae_mod_router is not None:
                nn.init.normal_(self.parcae_mod_router.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.parcae_mod_router.bias)
            self._last_parcae_mod_control_diagnostics: dict[str, float] = {}
        else:
            self.parcae_loop_start = 0
            self.parcae_loop_end = 0
            self.parcae_prelude_norm = None
            self.parcae_decay_raw = None
            self.parcae_injection_logit = None
            self.parcae_nonlinear_logit = None
            self.parcae_b_value_projection = None
            self.parcae_b_gate_projection = None
            self.parcae_p20_control_width = 0
            self.parcae_p20_control_mode = "none"
            self.parcae_p20_input_projection = None
            self.parcae_p20_controller = None
            self.parcae_p20_control_norm = None
            self.parcae_p20_value_projection = None
            self.parcae_p20_gate_projection = None
            self._last_parcae_norms = []
            self._last_parcae_injection_gate_mean = None
            self._last_parcae_injection_norm = None
            self._last_parcae_p20_control_norm = None
            self._last_parcae_control_diagnostics = {}
            self.parcae_mod_router_norm = None
            self.parcae_mod_router = None
            self.parcae_mod_gate_bias_scale = None
            self.parcae_mod_value_scale_strength = None
            self._last_parcae_mod_control_diagnostics = {}
        self._last_parcae_steps_used = 0
        self._last_parcae_step_norms: list[float] = []
        self._last_parcae_step_cosines: list[float | None] = []
        self._last_parcae_step_accelerations: list[float | None] = []
        self._last_parcae_raw_step_delta_norms: list[float] = []
        self._last_parcae_raw_acceleration_norms: list[float] = []
        self._last_parcae_drift_norms: list[float] = []
        self._last_parcae_halting_metric: float | None = None
        self._last_parcae_halted_early = False
        self._parcae_forward_count = 0
        self._parcae_total_steps = 0
        self._parcae_exit_count = 0
        self.parcae_token_router_norm = (
            nn.LayerNorm(variant.shape.d_model)
            if self.uses_recurrent_token_routing
            else None
        )
        self.parcae_token_router = (
            nn.Linear(variant.shape.d_model, 1)
            if self.uses_recurrent_token_routing
            else None
        )
        if self.parcae_token_router is not None:
            nn.init.normal_(self.parcae_token_router.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.parcae_token_router.bias)
        self._last_parcae_selected_token_fractions: list[float] = []
        self._last_parcae_selected_gate_means: list[float] = []
        self._parcae_selected_tokens = 0
        self._parcae_possible_tokens = 0
        self.huginn_adapter_norm = (
            nn.LayerNorm(variant.shape.d_model * 2)
            if self.uses_huginn_adapter_recurrence
            else None
        )
        self.huginn_adapter = (
            nn.Linear(variant.shape.d_model * 2, variant.shape.d_model)
            if self.uses_huginn_adapter_recurrence
            else None
        )
        self.act_halt_projection = (
            nn.Linear(variant.shape.d_model, 1)
            if self.uses_universal_transformer_act
            else None
        )
        self.ouro_exit_gate = (
            nn.Linear(variant.shape.d_model, 1) if self.uses_ouro_learned_exit else None
        )
        self.mor_routers = (
            nn.ModuleList(
                [
                    nn.Linear(variant.shape.d_model, 1)
                    for _ in range(variant.parcae_loop_count)
                ]
            )
            if self.uses_mor_expert_choice_scaffold
            else None
        )
        self._last_looped_hidden_norms: list[float] = []
        self._last_looped_step_norms: list[float] = []
        self._last_looped_input_injection_norms: list[float] = []
        self._last_looped_adapter_input_norms: list[float] = []
        self._last_looped_adapter_output_norms: list[float] = []
        self._last_looped_initial_state_norm: float | None = None
        self._last_looped_final_state_norm: float | None = None
        self._looped_forward_count = 0
        self._looped_total_steps = 0
        self._last_ut_hidden_norms: list[float] = []
        self._last_ut_step_norms: list[float] = []
        self._last_ut_step_coordinate_norms: list[float] = []
        self._last_ut_position_coordinate_norm: float | None = None
        self._last_ut_initial_state_norm: float | None = None
        self._last_ut_final_state_norm: float | None = None
        self._ut_forward_count = 0
        self._ut_total_steps = 0
        self._last_act_halt_probability_mean: float | None = None
        self._last_act_remainder_mean: float | None = None
        self._last_act_update_count_mean: float | None = None
        self._last_act_update_count_min: float | None = None
        self._last_act_update_count_max: float | None = None
        self._last_act_weight_sum_min: float | None = None
        self._last_act_weight_sum_max: float | None = None
        self._last_act_forced_final_halt_fraction: float | None = None
        self._last_act_ponder_loss: float | None = None
        self._last_ouro_step_logits: tuple[torch.Tensor, ...] = ()
        self._last_ouro_exit_pdf: torch.Tensor | None = None
        self._last_ouro_hidden_norms: list[float] = []
        self._last_ouro_step_norms: list[float] = []
        self._last_ouro_gate_probability_means: list[float] = []
        self._last_ouro_exit_pdf_mean_by_step: list[float] = []
        self._last_ouro_per_step_ce_mean: list[float] = []
        self._last_ouro_initial_state_norm: float | None = None
        self._last_ouro_final_state_norm: float | None = None
        self._last_ouro_exit_pdf_sum_min: float | None = None
        self._last_ouro_exit_pdf_sum_max: float | None = None
        self._last_ouro_final_step_mass_mean: float | None = None
        self._last_ouro_expected_exit_step_mean: float | None = None
        self._last_ouro_q_exit_step_mean: float | None = None
        self._last_ouro_q_exit_step_min: float | None = None
        self._last_ouro_q_exit_step_max: float | None = None
        self._last_ouro_expected_ce: float | None = None
        self._last_ouro_entropy_mean: float | None = None
        self._last_ouro_entropy_regularization: float | None = None
        self._ouro_forward_count = 0
        self._ouro_total_steps = 0
        self._last_rrt_hidden_norms: list[float] = []
        self._last_rrt_step_norms: list[float] = []
        self._last_rrt_shared_layer_indices: list[int] = []
        self._last_rrt_initial_state_norm: float | None = None
        self._last_rrt_final_state_norm: float | None = None
        self._rrt_forward_count = 0
        self._rrt_total_effective_layers = 0
        self._last_mor_active_token_counts: list[float] = []
        self._last_mor_selected_token_counts: list[float] = []
        self._last_mor_selected_token_fractions: list[float] = []
        self._last_mor_selected_gate_means: list[float] = []
        self._last_mor_selected_positions: list[list[list[int]]] = []
        self._last_mor_subset_flags: list[bool] = []
        self._last_mor_router_aux_loss: float | None = None
        self._last_mor_hidden_norms: list[float] = []
        self._last_mor_step_norms: list[float] = []
        self._last_mor_initial_state_norm: float | None = None
        self._last_mor_final_state_norm: float | None = None
        self._mor_forward_count = 0
        self._mor_total_selected_tokens = 0
        self._mor_total_possible_tokens = 0
        self._last_auxiliary_loss: torch.Tensor | None = None
        self.final_norm = (
            SimpleRmsNorm(variant.shape.d_model)
            if variant.final_norm_kind == "rmsnorm"
            else nn.Identity()
        )
        self.output = nn.Linear(
            variant.shape.d_model, variant.shape.vocab_size, bias=False
        )

    @property
    def model_label(self) -> str:
        return f"path1_{self.variant.label.replace('-', '_')}"

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        self._last_auxiliary_loss = hidden.new_zeros(())
        if self.context_embedding is not None:
            hidden = hidden + self.context_embedding(input_ids)
            hidden = F.rms_norm(hidden, (hidden.size(-1),))
        if self.smear_gate is not None:
            hidden = self.smear_gate(hidden)
        residual_anchor = hidden
        seq_len = input_ids.shape[1]
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
        if self.uses_ouro_learned_exit:
            return self._forward_ouro_logits(hidden, mask)
        if self.uses_mor_expert_choice_scaffold:
            hidden = self._forward_mor_expert_choice(hidden, mask)
        elif self.uses_rrt_cycle_scaffold:
            hidden = self._forward_rrt_cycle(hidden, mask)
        elif self.uses_universal_transformer_scaffold:
            hidden = self._forward_universal_transformer(hidden, mask)
        elif self.uses_looped_transformer_scaffold:
            hidden = self._forward_looped_transformer(hidden, mask)
        elif self.uses_parcae_scaffold:
            hidden = self._forward_parcae_loop(hidden, mask)
        else:
            depth_memory_states: list[Any] = []
            for block in self.blocks:
                if self.uses_pr5_scaffold:
                    hidden = block(hidden, mask, residual_anchor)
                else:
                    hidden = self._forward_block_with_depth(
                        block, hidden, mask, depth_memory_states
                    )
                if self.uses_depth_attention:
                    self._append_depth_memory_state(block, hidden, depth_memory_states)
        hidden = self.final_norm(hidden)
        return self.output(hidden)

    def auxiliary_loss(self) -> torch.Tensor | None:
        return self._last_auxiliary_loss

    def supervised_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        pad_token: int,
        training: bool,
    ) -> torch.Tensor:
        if not self.uses_ouro_learned_exit:
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=pad_token,
            )
        if self._last_ouro_exit_pdf is None or not self._last_ouro_step_logits:
            raise RuntimeError("Ouro supervised loss requires a preceding forward pass")
        exit_pdf = self._last_ouro_exit_pdf
        if exit_pdf.shape[:2] != target_ids.shape:
            raise RuntimeError(
                "Ouro exit distribution shape must match target token shape"
            )
        step_losses: list[torch.Tensor] = []
        for step_logits in self._last_ouro_step_logits:
            step_loss = F.cross_entropy(
                step_logits.reshape(-1, step_logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=pad_token,
                reduction="none",
            ).view_as(target_ids)
            step_losses.append(step_loss)
        per_step_loss = torch.stack(step_losses, dim=-1)
        valid_tokens = target_ids.ne(pad_token)
        expected_token_loss = (
            exit_pdf.to(dtype=per_step_loss.dtype) * per_step_loss
        ).sum(dim=-1)
        if bool(valid_tokens.detach().any().item()):
            expected_loss = expected_token_loss[valid_tokens].mean()
            entropy_tokens = -(exit_pdf * exit_pdf.clamp_min(1.0e-8).log()).sum(dim=-1)
            entropy = entropy_tokens[valid_tokens].mean()
            self._last_ouro_per_step_ce_mean = [
                float(
                    per_step_loss[..., step_index][valid_tokens].detach().mean().item()
                )
                for step_index in range(per_step_loss.shape[-1])
            ]
        else:
            expected_loss = expected_token_loss.mean() * 0.0
            entropy = exit_pdf.sum() * 0.0
            self._last_ouro_per_step_ce_mean = [
                0.0 for _ in range(per_step_loss.shape[-1])
            ]
        entropy_regularization = -self.variant.ouro_entropy_weight * entropy
        self._last_auxiliary_loss = (
            entropy_regularization if training else expected_loss.new_zeros(())
        )
        self._last_ouro_expected_ce = float(expected_loss.detach().float().item())
        self._last_ouro_entropy_mean = float(entropy.detach().float().item())
        self._last_ouro_entropy_regularization = float(
            entropy_regularization.detach().float().item()
        )
        return expected_loss

    def _depth_memory_tensor(self, depth_memory_states: list[Any]) -> object | None:
        if not self.uses_depth_attention or not depth_memory_states:
            return None
        if self.uses_paper_moda_attention:
            return depth_memory_states[-self.variant.depth_memory_layers :]
        selected = depth_memory_states[-self.variant.depth_memory_layers :]
        return torch.stack(selected, dim=1)

    def _append_depth_memory_state(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        depth_memory_states: list[Any],
    ) -> None:
        if self.uses_paper_moda_attention:
            attention = getattr(block, "attention", None)
            sequence_kv = getattr(attention, "_last_sequence_kv", None)
            if sequence_kv is None:
                raise RuntimeError(
                    "paper MoDA attention did not expose current sequence KV"
                )
            depth_memory_states.append(sequence_kv)
            attention._last_sequence_kv = None
            return
        depth_memory_states.append(hidden)

    def _forward_block_with_depth(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        depth_memory_states: list[Any],
    ) -> torch.Tensor:
        depth_memory = self._depth_memory_tensor(depth_memory_states)
        if isinstance(block, LocalCausalTransformerBlock):
            return block(hidden, mask, depth_memory)
        return block(hidden, mask)

    def _forward_block_selected(
        self,
        block: nn.Module,
        hidden: torch.Tensor,
        mask: torch.Tensor | None,
        selected_positions: list[torch.Tensor],
    ) -> torch.Tensor:
        if not isinstance(block, LocalCausalTransformerBlock):
            raise RuntimeError(
                "token-selective recurrence currently requires local attention blocks"
            )
        return block.forward_selected(hidden, selected_positions, mask)

    def _parcae_selected_positions(
        self, hidden: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], float, float]:
        if self.parcae_token_router_norm is None or self.parcae_token_router is None:
            raise RuntimeError("Parcae token router is not initialized")
        router_scores = self.parcae_token_router(
            self.parcae_token_router_norm(hidden)
        ).squeeze(-1)
        selected_positions = causal_topk_token_positions(
            router_scores, self.variant.recurrent_token_route_fraction
        )
        selected_scores = [
            router_scores[batch_index].index_select(0, positions)
            for batch_index, positions in enumerate(selected_positions)
        ]
        selected_gates = [
            torch.sigmoid(scores).to(dtype=hidden.dtype) for scores in selected_scores
        ]
        selected_count = sum(int(positions.numel()) for positions in selected_positions)
        token_count = int(router_scores.numel())
        selected_fraction = selected_count / token_count if token_count else 0.0
        gate_mean = (
            float(torch.cat(selected_gates).detach().float().mean().item())
            if selected_gates
            else 0.0
        )
        self._parcae_selected_tokens += selected_count
        self._parcae_possible_tokens += token_count
        return selected_positions, selected_gates, selected_fraction, gate_mean

    def _apply_selected_token_gates(
        self,
        hidden: torch.Tensor,
        updated: torch.Tensor,
        selected_positions: list[torch.Tensor],
        selected_gates: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            hidden_selected = hidden[batch_index].index_select(0, positions)
            updated_selected = updated[batch_index].index_select(0, positions)
            gated = hidden_selected + selected_gates[batch_index].unsqueeze(-1) * (
                updated_selected - hidden_selected
            )
            batch_outputs.append(
                hidden[batch_index].clone().index_copy(0, positions, gated)
            )
        return torch.stack(batch_outputs, dim=0)

    def _forward_looped_block_group(
        self, hidden: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        for block in self.blocks:
            if not isinstance(block, LocalCausalTransformerBlock):
                raise RuntimeError(
                    "looped-transformer scaffold currently requires local attention blocks"
                )
            hidden = block(hidden, mask)
        return hidden

    def _looped_recurrence_input(
        self, state: torch.Tensor, prompt: torch.Tensor
    ) -> torch.Tensor:
        if self.uses_looped_additive_input:
            injected = state + prompt
            self._last_looped_input_injection_norms.append(
                float(prompt.detach().float().norm(dim=-1).mean().item())
            )
            return injected
        if self.uses_huginn_adapter_recurrence:
            if self.huginn_adapter_norm is None or self.huginn_adapter is None:
                raise RuntimeError("Huginn adapter recurrence is not initialized")
            adapter_input = torch.cat((state, prompt), dim=-1)
            self._last_looped_adapter_input_norms.append(
                float(adapter_input.detach().float().norm(dim=-1).mean().item())
            )
            adapted = self.huginn_adapter(self.huginn_adapter_norm(adapter_input))
            self._last_looped_adapter_output_norms.append(
                float(adapted.detach().float().norm(dim=-1).mean().item())
            )
            self._last_looped_input_injection_norms.append(
                float(prompt.detach().float().norm(dim=-1).mean().item())
            )
            return adapted
        return state

    def _forward_looped_transformer(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.uses_looped_additive_input or self.uses_huginn_adapter_recurrence:
            state = torch.zeros_like(prompt)
        else:
            state = prompt
        self._last_looped_initial_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._last_looped_hidden_norms = []
        self._last_looped_step_norms = []
        self._last_looped_input_injection_norms = []
        self._last_looped_adapter_input_norms = []
        self._last_looped_adapter_output_norms = []
        for _ in range(self.variant.parcae_loop_count):
            previous_state = state
            loop_input = self._looped_recurrence_input(state, prompt)
            state = self._forward_looped_block_group(loop_input, mask)
            self._last_looped_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_looped_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
        self._last_looped_final_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._looped_forward_count += 1
        self._looped_total_steps += self.variant.parcae_loop_count
        return state

    def _ouro_exit_pdf(
        self, gate_logits_by_step: tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> torch.Tensor:
        if not gate_logits_by_step:
            raise RuntimeError("Ouro exit distribution requires at least one step")
        masses: list[torch.Tensor] = []
        first = gate_logits_by_step[0]
        if first.dim() == 3 and first.shape[-1] == 1:
            first = first.squeeze(-1)
        survival = torch.ones_like(first)
        for step_index, gate_logits in enumerate(gate_logits_by_step):
            gate = gate_logits
            if gate.dim() == 3 and gate.shape[-1] == 1:
                gate = gate.squeeze(-1)
            if gate.shape != survival.shape:
                raise RuntimeError("Ouro gate logits must share batch/token shape")
            is_final_step = step_index == len(gate_logits_by_step) - 1
            if is_final_step:
                mass = survival
            else:
                halt_probability = torch.sigmoid(gate)
                mass = survival * halt_probability
                survival = survival * (1.0 - halt_probability)
            masses.append(mass)
        return torch.stack(masses, dim=-1)

    def _ouro_q_exit_steps(self, exit_pdf: torch.Tensor) -> torch.Tensor:
        cdf = exit_pdf.detach().float().cumsum(dim=-1)
        crossed = cdf >= self.variant.ouro_q_exit_threshold
        has_crossed = crossed.any(dim=-1)
        first_crossing = crossed.to(dtype=torch.int64).argmax(dim=-1) + 1
        final_step = torch.full_like(first_crossing, exit_pdf.shape[-1])
        return torch.where(has_crossed, first_crossing, final_step)

    def _forward_ouro_logits(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.ouro_exit_gate is None:
            raise RuntimeError("Ouro learned-exit gate is not initialized")
        state = prompt
        self._last_ouro_step_logits = ()
        self._last_ouro_exit_pdf = None
        self._last_ouro_hidden_norms = []
        self._last_ouro_step_norms = []
        self._last_ouro_gate_probability_means = []
        self._last_ouro_exit_pdf_mean_by_step = []
        self._last_ouro_per_step_ce_mean = []
        self._last_ouro_initial_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        step_logits: list[torch.Tensor] = []
        gate_logits_by_step: list[torch.Tensor] = []
        for _ in range(self.variant.parcae_loop_count):
            previous_state = state
            state = self._forward_looped_block_group(state, mask)
            gate_logits = self.ouro_exit_gate(state).squeeze(-1)
            gate_logits_by_step.append(gate_logits)
            step_logits.append(self.output(self.final_norm(state)))
            self._last_ouro_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_ouro_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
            self._last_ouro_gate_probability_means.append(
                float(torch.sigmoid(gate_logits).detach().float().mean().item())
            )
        exit_pdf = self._ouro_exit_pdf(gate_logits_by_step)
        stacked_logits = torch.stack(step_logits, dim=2)
        expected_logits = (exit_pdf.unsqueeze(-1) * stacked_logits).sum(dim=2)
        detached_pdf = exit_pdf.detach().float()
        pdf_sums = detached_pdf.sum(dim=-1)
        step_numbers = torch.arange(
            1,
            exit_pdf.shape[-1] + 1,
            device=exit_pdf.device,
            dtype=detached_pdf.dtype,
        )
        q_exit_steps = self._ouro_q_exit_steps(exit_pdf)
        self._last_ouro_step_logits = tuple(step_logits)
        self._last_ouro_exit_pdf = exit_pdf
        self._last_ouro_exit_pdf_mean_by_step = [
            float(value.item()) for value in detached_pdf.mean(dim=(0, 1))
        ]
        self._last_ouro_exit_pdf_sum_min = float(pdf_sums.min().item())
        self._last_ouro_exit_pdf_sum_max = float(pdf_sums.max().item())
        self._last_ouro_final_step_mass_mean = float(
            detached_pdf[..., -1].mean().item()
        )
        self._last_ouro_expected_exit_step_mean = float(
            (detached_pdf * step_numbers.view(1, 1, -1)).sum(dim=-1).mean().item()
        )
        self._last_ouro_q_exit_step_mean = float(
            q_exit_steps.detach().float().mean().item()
        )
        self._last_ouro_q_exit_step_min = float(
            q_exit_steps.detach().float().min().item()
        )
        self._last_ouro_q_exit_step_max = float(
            q_exit_steps.detach().float().max().item()
        )
        self._last_ouro_final_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._ouro_forward_count += 1
        self._ouro_total_steps += self.variant.parcae_loop_count
        return expected_logits

    def _rrt_shared_layer_index(self, absolute_depth: int) -> int:
        if not 0 <= absolute_depth < self.rrt_effective_layer_count:
            raise IndexError(
                f"absolute_depth must be in [0, {self.rrt_effective_layer_count}), "
                f"got {absolute_depth}"
            )
        return absolute_depth % self.rrt_stored_layer_count

    def _rrt_block_for_absolute_depth(self, absolute_depth: int) -> nn.Module:
        return self.blocks[self._rrt_shared_layer_index(absolute_depth)]

    def _forward_rrt_cycle(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        state = prompt
        self._last_rrt_initial_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._last_rrt_hidden_norms = []
        self._last_rrt_step_norms = []
        self._last_rrt_shared_layer_indices = []
        for absolute_depth in range(self.rrt_effective_layer_count):
            previous_state = state
            shared_layer_index = self._rrt_shared_layer_index(absolute_depth)
            block = self.blocks[shared_layer_index]
            if not isinstance(block, LocalCausalTransformerBlock):
                raise RuntimeError(
                    "rrt-cycle currently requires local attention blocks"
                )
            state = block(state, mask)
            self._last_rrt_shared_layer_indices.append(shared_layer_index)
            self._last_rrt_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_rrt_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
        self._last_rrt_final_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._rrt_forward_count += 1
        self._rrt_total_effective_layers += self.rrt_effective_layer_count
        return state

    def _mor_apply_selected_middle_stack(
        self,
        state: torch.Tensor,
        selected_positions: list[torch.Tensor],
        selected_gates: list[torch.Tensor],
    ) -> torch.Tensor:
        middle_blocks = self.blocks[1:-1]
        batch_outputs: list[torch.Tensor] = []
        for batch_index, positions in enumerate(selected_positions):
            selected_state = state[batch_index : batch_index + 1].index_select(
                1, positions
            )
            selected_mask = (
                None
                if self.variant.shape.local_window >= positions.numel()
                else local_causal_attention_bias(
                    positions.numel(),
                    self.variant.shape.local_window,
                    state.device,
                    state.dtype,
                )
            )
            updated = selected_state
            for block in middle_blocks:
                if not isinstance(block, LocalCausalTransformerBlock):
                    raise RuntimeError(
                        "mor-expert-choice currently requires local attention blocks"
                    )
                updated = block(updated, selected_mask)
            current = state[batch_index].index_select(0, positions)
            gated = current + selected_gates[batch_index].unsqueeze(-1) * (
                updated.squeeze(0) - current
            )
            batch_outputs.append(
                state[batch_index].clone().index_copy(0, positions, gated)
            )
        return torch.stack(batch_outputs, dim=0)

    def _forward_mor_expert_choice(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.mor_routers is None:
            raise RuntimeError("MoR expert-choice routers are not initialized")
        if len(self.blocks) < 3:
            raise RuntimeError("MoR expert-choice requires at least three blocks")
        prelude = self.blocks[0]
        coda = self.blocks[-1]
        if not isinstance(prelude, LocalCausalTransformerBlock) or not isinstance(
            coda, LocalCausalTransformerBlock
        ):
            raise RuntimeError("MoR expert-choice requires local attention blocks")

        state = prompt
        batch_size, seq_len, _ = state.shape
        self._last_mor_initial_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._last_mor_active_token_counts = []
        self._last_mor_selected_token_counts = []
        self._last_mor_selected_token_fractions = []
        self._last_mor_selected_gate_means = []
        self._last_mor_selected_positions = []
        self._last_mor_subset_flags = []
        self._last_mor_hidden_norms = []
        self._last_mor_step_norms = []

        state = prelude(state, mask)
        active_positions = [
            torch.arange(seq_len, device=state.device, dtype=torch.long)
            for _ in range(batch_size)
        ]
        router_auxiliary = state.new_zeros(())
        for router in self.mor_routers:
            previous_state = state
            selected_positions: list[torch.Tensor] = []
            selected_gates: list[torch.Tensor] = []
            router_losses: list[torch.Tensor] = []
            subset_flags: list[bool] = []
            active_counts: list[int] = []
            selected_counts: list[int] = []
            selected_positions_payload: list[list[int]] = []
            gate_means: list[float] = []

            for batch_index, active in enumerate(active_positions):
                active_counts.append(int(active.numel()))
                active_hidden = state[batch_index].index_select(0, active)
                logits = router(active_hidden).squeeze(-1)
                capacity = max(
                    1,
                    min(
                        int(active.numel()),
                        math.ceil(
                            int(active.numel())
                            * self.variant.recurrent_token_route_fraction
                        ),
                    ),
                )
                local_indices = logits.topk(capacity, dim=0).indices.sort().values
                positions = active.index_select(0, local_indices)
                gates = (
                    torch.sigmoid(logits.index_select(0, local_indices))
                    * self.variant.mor_update_scale
                )
                targets = torch.zeros_like(logits)
                targets.index_fill_(0, local_indices, 1.0)
                router_losses.append(
                    F.binary_cross_entropy_with_logits(logits, targets)
                )
                selected_positions.append(positions)
                selected_gates.append(gates)
                selected_counts.append(int(positions.numel()))
                selected_positions_payload.append(
                    [int(value) for value in positions.detach().cpu().tolist()]
                )
                gate_means.append(float(gates.detach().float().mean().item()))
                active_set = set(int(value) for value in active.detach().cpu().tolist())
                subset_flags.append(
                    all(value in active_set for value in selected_positions_payload[-1])
                )

            router_auxiliary = router_auxiliary + torch.stack(router_losses).mean()
            state = self._mor_apply_selected_middle_stack(
                state, selected_positions, selected_gates
            )
            active_positions = selected_positions
            mean_active_count = sum(active_counts) / max(len(active_counts), 1)
            mean_selected_count = sum(selected_counts) / max(len(selected_counts), 1)
            self._last_mor_active_token_counts.append(mean_active_count)
            self._last_mor_selected_token_counts.append(mean_selected_count)
            self._last_mor_selected_token_fractions.append(
                mean_selected_count / max(mean_active_count, 1.0)
            )
            self._last_mor_selected_gate_means.append(
                sum(gate_means) / max(len(gate_means), 1)
            )
            self._last_mor_selected_positions.append(selected_positions_payload)
            self._last_mor_subset_flags.append(all(subset_flags))
            self._last_mor_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_mor_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
            self._mor_total_selected_tokens += sum(selected_counts)
            self._mor_total_possible_tokens += sum(active_counts)

        auxiliary_loss = self.variant.mor_router_aux_loss_weight * router_auxiliary
        if self._last_auxiliary_loss is None:
            self._last_auxiliary_loss = auxiliary_loss
        else:
            self._last_auxiliary_loss = self._last_auxiliary_loss + auxiliary_loss
        self._last_mor_router_aux_loss = float(auxiliary_loss.detach().float().item())
        state = coda(state, mask)
        self._last_mor_final_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._mor_forward_count += 1
        return state

    def _sinusoidal_coordinate_signal(
        self,
        indices: torch.Tensor,
        *,
        width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        signal = torch.zeros(indices.numel(), width, device=indices.device, dtype=dtype)
        if width == 0:
            return signal
        half_width = width // 2
        if half_width == 0:
            signal[:, 0] = torch.sin(indices.to(dtype=dtype))
            return signal
        frequencies = torch.exp(
            torch.arange(half_width, device=indices.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half_width - 1, 1))
        ).to(dtype=dtype)
        angles = indices.to(dtype=dtype).unsqueeze(-1) * frequencies.unsqueeze(0)
        signal[:, 0 : 2 * half_width : 2] = torch.sin(angles)
        signal[:, 1 : 2 * half_width : 2] = torch.cos(angles)
        if width % 2:
            signal[:, -1] = torch.sin(indices.to(dtype=dtype))
        return signal

    def _universal_transformer_recurrence_input(
        self, state: torch.Tensor, *, step_index: int
    ) -> torch.Tensor:
        seq_len = state.shape[1]
        positions = torch.arange(seq_len, device=state.device)
        position_signal = self._sinusoidal_coordinate_signal(
            positions,
            width=self.variant.shape.d_model,
            dtype=state.dtype,
        ).view(1, seq_len, self.variant.shape.d_model)
        step_signal = self._sinusoidal_coordinate_signal(
            torch.tensor([step_index + 1], device=state.device),
            width=self.variant.shape.d_model,
            dtype=state.dtype,
        ).view(1, 1, self.variant.shape.d_model)
        if self._last_ut_position_coordinate_norm is None:
            self._last_ut_position_coordinate_norm = float(
                position_signal.detach().float().norm(dim=-1).mean().item()
            )
        self._last_ut_step_coordinate_norms.append(
            float(step_signal.detach().float().norm(dim=-1).mean().item())
        )
        return state + position_signal + step_signal

    def _forward_universal_transformer_fixed(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        state = prompt
        for step_index in range(self.variant.parcae_loop_count):
            previous_state = state
            step_input = self._universal_transformer_recurrence_input(
                state, step_index=step_index
            )
            state = self._forward_looped_block_group(step_input, mask)
            self._last_ut_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_ut_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
        return state

    def _forward_universal_transformer_act(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.act_halt_projection is None:
            raise RuntimeError("Universal Transformer ACT halt projection is missing")
        batch_size, seq_len, width = prompt.shape
        state = prompt
        halting = prompt.new_zeros(batch_size, seq_len)
        remainders = prompt.new_zeros(batch_size, seq_len)
        update_counts = prompt.new_zeros(batch_size, seq_len)
        update_weight_sums = prompt.new_zeros(batch_size, seq_len)
        weighted_state = prompt.new_zeros(batch_size, seq_len, width)
        halt_probability_means: list[float] = []
        forced_final_halt_fraction = 0.0
        threshold = self.variant.act_halting_threshold
        for step_index in range(self.variant.parcae_loop_count):
            previous_state = state
            step_input = self._universal_transformer_recurrence_input(
                state, step_index=step_index
            )
            next_state = self._forward_looped_block_group(step_input, mask)
            halt_probability = torch.sigmoid(
                self.act_halt_projection(next_state)
            ).squeeze(-1)
            halt_probability_means.append(
                float(halt_probability.detach().float().mean().item())
            )
            still_running = halting < threshold
            is_final_step = step_index == self.variant.parcae_loop_count - 1
            proposed_halt = halting + halt_probability
            if is_final_step:
                newly_halted = still_running
                still_running_next = torch.zeros_like(still_running)
                forced_final_halt_fraction = float(
                    newly_halted.detach().float().mean().item()
                )
            else:
                newly_halted = (proposed_halt >= threshold) & still_running
                still_running_next = (proposed_halt < threshold) & still_running
            remainder = (1.0 - halting).clamp_min(0.0)
            update_weight = torch.where(
                newly_halted,
                remainder,
                torch.where(
                    still_running_next,
                    halt_probability,
                    torch.zeros_like(halt_probability),
                ),
            )
            halting = halting + update_weight
            remainders = torch.where(newly_halted, remainder, remainders)
            update_counts = update_counts + still_running.to(dtype=prompt.dtype)
            update_weight_sums = update_weight_sums + update_weight
            weighted_state = weighted_state + update_weight.unsqueeze(-1) * next_state
            state = next_state
            self._last_ut_step_norms.append(
                self._normalized_step_norm(previous_state, state)
            )
            self._last_ut_hidden_norms.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
        ponder = (update_counts + remainders).mean()
        auxiliary_loss = self.variant.act_ponder_loss_weight * ponder
        self._last_auxiliary_loss = auxiliary_loss
        self._last_act_halt_probability_mean = (
            sum(halt_probability_means) / len(halt_probability_means)
            if halt_probability_means
            else 0.0
        )
        self._last_act_remainder_mean = float(remainders.detach().float().mean().item())
        self._last_act_update_count_mean = float(
            update_counts.detach().float().mean().item()
        )
        self._last_act_update_count_min = float(
            update_counts.detach().float().min().item()
        )
        self._last_act_update_count_max = float(
            update_counts.detach().float().max().item()
        )
        self._last_act_weight_sum_min = float(
            update_weight_sums.detach().float().min().item()
        )
        self._last_act_weight_sum_max = float(
            update_weight_sums.detach().float().max().item()
        )
        self._last_act_forced_final_halt_fraction = forced_final_halt_fraction
        self._last_act_ponder_loss = float(auxiliary_loss.detach().float().item())
        return weighted_state

    def _forward_universal_transformer(
        self, prompt: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        self._last_ut_initial_state_norm = float(
            prompt.detach().float().norm(dim=-1).mean().item()
        )
        self._last_ut_hidden_norms = []
        self._last_ut_step_norms = []
        self._last_ut_step_coordinate_norms = []
        self._last_ut_position_coordinate_norm = None
        if self.uses_universal_transformer_act:
            state = self._forward_universal_transformer_act(prompt, mask)
        else:
            state = self._forward_universal_transformer_fixed(prompt, mask)
        self._last_ut_final_state_norm = float(
            state.detach().float().norm(dim=-1).mean().item()
        )
        self._ut_forward_count += 1
        self._ut_total_steps += self.variant.parcae_loop_count
        return state

    def _forward_parcae_loop(
        self, hidden: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if (
            self.parcae_prelude_norm is None
            or self.parcae_decay_raw is None
            or self.parcae_injection_logit is None
            or self.parcae_nonlinear_logit is None
        ):
            raise RuntimeError("parcae scaffold is not initialized")

        depth_memory_states: list[Any] = []
        for block in self.blocks[: self.parcae_loop_start]:
            hidden = self._forward_block_with_depth(
                block, hidden, mask, depth_memory_states
            )
            if self.uses_depth_attention:
                self._append_depth_memory_state(block, hidden, depth_memory_states)

        loop_input = self.parcae_prelude_norm(hidden)
        state = torch.zeros_like(loop_input)
        decay = torch.exp(
            -F.softplus(self.parcae_decay_raw.to(dtype=hidden.dtype))
        ).view(1, 1, -1)
        nonlinear = torch.sigmoid(
            self.parcae_nonlinear_logit.to(dtype=hidden.dtype)
        ).view(1, 1, -1)
        injection_value, injection_gate = self._parcae_injection(loop_input)
        norm_history: list[float] = []
        step_norm_history: list[float] = []
        step_cosine_history: list[float | None] = []
        step_acceleration_history: list[float | None] = []
        raw_step_delta_norm_history: list[float] = []
        raw_acceleration_norm_history: list[float] = []
        drift_norm_history: list[float] = []
        last_step_norm: float | None = None
        last_delta: torch.Tensor | None = None
        halting_metric: float | None = None
        halted_early = False
        steps_used = 0
        selected_token_fractions: list[float] = []
        selected_gate_means: list[float] = []
        recurrent_blocks = self.blocks[self.parcae_loop_start : self.parcae_loop_end]
        for step_index in range(self.variant.parcae_loop_count):
            previous_state = state
            mixed = decay * state + injection_gate * injection_value
            selected_positions: list[torch.Tensor] | None = None
            selected_gates: list[torch.Tensor] | None = None
            if self.uses_recurrent_token_routing:
                (
                    selected_positions,
                    selected_gates,
                    selected_fraction,
                    selected_gate_mean,
                ) = self._parcae_selected_positions(mixed)
                selected_token_fractions.append(selected_fraction)
                selected_gate_means.append(selected_gate_mean)
            for block in recurrent_blocks:
                if selected_positions is None or selected_gates is None:
                    block_out = self._forward_block_with_depth(
                        block, mixed, mask, depth_memory_states
                    )
                else:
                    block_out = self._forward_block_selected(
                        block, mixed, mask, selected_positions
                    )
                    block_out = self._apply_selected_token_gates(
                        mixed, block_out, selected_positions, selected_gates
                    )
                mixed = mixed + nonlinear * (block_out - mixed)
            state = mixed
            steps_used = step_index + 1
            delta = state - previous_state
            step_norm = self._normalized_step_norm(previous_state, state)
            step_norm_history.append(step_norm)
            raw_step_delta_norm_history.append(_mean_l2_norm(delta))
            if last_delta is None:
                step_cosine_history.append(None)
                step_acceleration_history.append(None)
                raw_acceleration_norm_history.append(_mean_l2_norm(delta))
            else:
                cosine = F.cosine_similarity(
                    delta.detach().float().reshape(1, -1),
                    last_delta.detach().float().reshape(1, -1),
                    dim=-1,
                )
                step_cosine_history.append(float(cosine.item()))
                step_acceleration_history.append(
                    float(
                        (delta - last_delta).detach().float().norm(dim=-1).mean().item()
                    )
                )
                raw_acceleration_norm_history.append(_mean_l2_norm(delta - last_delta))
            drift_norm_history.append(self._normalized_step_norm(loop_input, state))
            norm_history.append(
                float(state.detach().float().norm(dim=-1).mean().item())
            )
            if self.uses_depth_attention:
                depth_memory_states.append(state)
            if self._should_halt_parcae(
                step_norm, last_step_norm, steps_used, delta, last_delta
            ):
                halting_metric = self._parcae_halting_metric(
                    step_norm, last_step_norm, delta, last_delta
                )
                halted_early = True
                break
            halting_metric = self._parcae_halting_metric(
                step_norm, last_step_norm, delta, last_delta
            )
            last_step_norm = step_norm
            last_delta = delta.detach()

        hidden = state
        for block in self.blocks[self.parcae_loop_end :]:
            hidden = self._forward_block_with_depth(
                block, hidden, mask, depth_memory_states
            )
            if self.uses_depth_attention:
                self._append_depth_memory_state(block, hidden, depth_memory_states)
        self._last_parcae_norms = norm_history
        self._last_parcae_selected_token_fractions = selected_token_fractions
        self._last_parcae_selected_gate_means = selected_gate_means
        self._record_parcae_halting_stats(
            steps_used=steps_used,
            step_norm_history=step_norm_history,
            step_cosine_history=step_cosine_history,
            step_acceleration_history=step_acceleration_history,
            raw_step_delta_norm_history=raw_step_delta_norm_history,
            raw_acceleration_norm_history=raw_acceleration_norm_history,
            drift_norm_history=drift_norm_history,
            halting_metric=halting_metric,
            halted_early=halted_early,
        )
        return hidden

    def _normalized_step_norm(
        self, previous_state: torch.Tensor, state: torch.Tensor
    ) -> float:
        delta_norm = (state - previous_state).detach().float().norm(dim=-1).mean()
        previous_norm = previous_state.detach().float().norm(dim=-1).mean()
        current_norm = state.detach().float().norm(dim=-1).mean()
        reference_norm = torch.maximum(previous_norm, current_norm).clamp_min(1.0e-6)
        return float((delta_norm / reference_norm).item())

    def _parcae_halting_metric(
        self,
        step_norm: float,
        last_step_norm: float | None,
        delta: torch.Tensor | None = None,
        last_delta: torch.Tensor | None = None,
    ) -> float:
        if (
            self.variant.recurrent_halting_profile
            is RecurrentHaltingProfile.ACCELERATION
        ):
            return (
                step_norm if last_step_norm is None else abs(step_norm - last_step_norm)
            )
        if (
            self.variant.recurrent_halting_profile
            is RecurrentHaltingProfile.VECTOR_ACCELERATION
        ):
            if delta is None or last_delta is None:
                return float("inf")
            current_norm = _mean_l2_norm(delta)
            previous_norm = _mean_l2_norm(last_delta)
            acceleration_norm = _mean_l2_norm(delta - last_delta)
            return acceleration_norm / max(current_norm, previous_norm, 1.0e-6)
        return step_norm

    def _should_halt_parcae(
        self,
        step_norm: float,
        last_step_norm: float | None,
        steps_used: int,
        delta: torch.Tensor | None = None,
        last_delta: torch.Tensor | None = None,
    ) -> bool:
        if self.variant.recurrent_halting_profile is RecurrentHaltingProfile.FIXED:
            return False
        if steps_used < self.variant.recurrent_min_steps:
            return False
        if steps_used >= self.variant.parcae_loop_count:
            return False
        return (
            self._parcae_halting_metric(step_norm, last_step_norm, delta, last_delta)
            <= self.variant.recurrent_halting_threshold
        )

    def _record_parcae_halting_stats(
        self,
        *,
        steps_used: int,
        step_norm_history: list[float],
        step_cosine_history: list[float | None],
        step_acceleration_history: list[float | None],
        raw_step_delta_norm_history: list[float],
        raw_acceleration_norm_history: list[float],
        drift_norm_history: list[float],
        halting_metric: float | None,
        halted_early: bool,
    ) -> None:
        self._last_parcae_steps_used = steps_used
        self._last_parcae_step_norms = step_norm_history
        self._last_parcae_step_cosines = step_cosine_history
        self._last_parcae_step_accelerations = step_acceleration_history
        self._last_parcae_raw_step_delta_norms = raw_step_delta_norm_history
        self._last_parcae_raw_acceleration_norms = raw_acceleration_norm_history
        self._last_parcae_drift_norms = drift_norm_history
        self._last_parcae_halting_metric = halting_metric
        self._last_parcae_halted_early = halted_early
        self._parcae_forward_count += 1
        self._parcae_total_steps += steps_used
        if halted_early:
            self._parcae_exit_count += 1

    def _record_parcae_control_diagnostics(
        self,
        *,
        loop_input: torch.Tensor,
        control: torch.Tensor,
        injection_gate: torch.Tensor,
        injection_value: torch.Tensor,
        control_value: torch.Tensor | None,
    ) -> None:
        injection = injection_gate * injection_value
        injection_delta = injection - loop_input
        loop_input_norm = _global_l2_norm(loop_input)
        control_value_norm = (
            _mean_l2_norm(control_value) if control_value is not None else None
        )
        control_value_global_norm = (
            _global_l2_norm(control_value) if control_value is not None else None
        )
        diagnostics: dict[str, float | None] = {
            "controller/control_norm_mean": _mean_l2_norm(control),
            "controller/control_norm_rms": _rms(control),
            "controller/gate_mean": float(
                injection_gate.detach().float().mean().item()
            ),
            "controller/gate_std": float(injection_gate.detach().float().std().item()),
            "controller/gate_min": float(injection_gate.detach().float().min().item()),
            "controller/gate_max": float(injection_gate.detach().float().max().item()),
            "controller/gate_saturation_low_frac": float(
                (injection_gate.detach().float() < 0.05).float().mean().item()
            ),
            "controller/gate_saturation_high_frac": float(
                (injection_gate.detach().float() > 0.95).float().mean().item()
            ),
            "controller/value_norm_mean": control_value_norm,
            "controller/value_to_loop_input_norm_ratio": (
                _safe_float_ratio(control_value_global_norm, loop_input_norm)
                if control_value_global_norm is not None
                else None
            ),
            "controller/injection_delta_norm_mean": _mean_l2_norm(injection_delta),
            "controller/injection_delta_to_loop_input_ratio": _safe_float_ratio(
                _global_l2_norm(injection_delta), loop_input_norm
            ),
            "stability/nan_or_inf_seen": _nan_or_inf_seen(
                loop_input,
                control,
                injection_gate,
                injection_value,
                injection,
                injection_delta,
            )
            or (
                _nan_or_inf_seen(control_value)
                if control_value is not None
                else 0.0
            ),
        }
        diagnostics.update(self._last_parcae_mod_control_diagnostics)
        self._last_parcae_control_diagnostics = diagnostics

    def _parcae_mod_control_tensors(
        self, loop_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parcae_mod_router_norm is None or self.parcae_mod_router is None:
            raise RuntimeError("parcae MoD-control router is not initialized")
        router_scores = self.parcae_mod_router(
            self.parcae_mod_router_norm(loop_input)
        ).squeeze(-1)
        selected_positions = causal_topk_token_positions(
            router_scores, self.variant.token_route_fraction
        )
        selected_mask = selected_positions_to_mask(selected_positions, router_scores)
        selected_scores = torch.cat(
            [
                router_scores[batch_index].index_select(0, positions)
                for batch_index, positions in enumerate(selected_positions)
            ]
        )
        salience = torch.sigmoid(router_scores)
        self._last_parcae_mod_control_diagnostics = {
            "mod_router/route_fraction": float(self.variant.token_route_fraction),
            "mod_router/selected_fraction": float(
                selected_mask.detach().float().mean().item()
            ),
            "mod_router/score_mean": float(
                router_scores.detach().float().mean().item()
            ),
            "mod_router/score_std": float(
                router_scores.detach().float().std().item()
            ),
            "mod_router/selected_score_mean": float(
                selected_scores.detach().float().mean().item()
            ),
            "mod_router/salience_mean": float(salience.detach().float().mean().item()),
        }
        return router_scores, selected_mask, salience

    def _parcae_control_diagnostic_payload(self) -> dict[str, float | None]:
        if not self._last_parcae_control_diagnostics:
            return {}
        payload = dict(self._last_parcae_control_diagnostics)
        payload["loop/steps_used_mean"] = (
            self._parcae_total_steps / self._parcae_forward_count
            if self._parcae_forward_count
            else 0.0
        )
        payload["loop/early_exit_fraction"] = (
            self._parcae_exit_count / self._parcae_forward_count
            if self._parcae_forward_count
            else 0.0
        )
        for step_index, state_norm in enumerate(self._last_parcae_norms):
            payload[f"loop/state_norm_step_{step_index}"] = state_norm
        for step_index, step_delta_norm in enumerate(
            self._last_parcae_raw_step_delta_norms
        ):
            payload[f"loop/step_delta_norm_step_{step_index}"] = step_delta_norm
        for step_index, acceleration_norm in enumerate(
            self._last_parcae_raw_acceleration_norms
        ):
            payload[f"loop/acceleration_norm_step_{step_index}"] = acceleration_norm
        finite_values = [
            value
            for value in payload.values()
            if isinstance(value, (float, int))
        ]
        if any(not math.isfinite(float(value)) for value in finite_values):
            payload["stability/nan_or_inf_seen"] = 1.0
        return payload

    def _parcae_injection(
        self, loop_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parcae_injection_logit is None:
            raise RuntimeError("parcae scaffold is not initialized")
        if self.uses_parcae_p20_control:
            if (
                self.parcae_p20_controller is None
                or self.parcae_p20_control_norm is None
            ):
                raise RuntimeError("parcae P20-control scaffold is not initialized")
            control_input = (
                self.parcae_p20_input_projection(loop_input)
                if self.parcae_p20_input_projection is not None
                else loop_input
            )
            control = self.parcae_p20_controller.scan(control_input).emitted_outputs
            control = self.parcae_p20_control_norm(control)
            mod_router_scores: torch.Tensor | None = None
            mod_salience: torch.Tensor | None = None
            if self.uses_parcae_p20_mod_control:
                mod_router_scores, _, mod_salience = self._parcae_mod_control_tensors(
                    loop_input
                )
            else:
                self._last_parcae_mod_control_diagnostics = {}
            base_gate = torch.sigmoid(
                self.parcae_injection_logit.to(dtype=loop_input.dtype)
            ).view(1, 1, -1)
            control_value: torch.Tensor | None = None
            if self.uses_parcae_p20_value_only_control:
                injection_gate = base_gate
                if self.parcae_p20_value_projection is None:
                    raise RuntimeError(
                        "parcae P20 value-control projection is not initialized"
                    )
                control_value = (
                    self.variant.parcae_p20_value_scale
                    * self.parcae_p20_value_projection(control)
                )
                injection_value = loop_input + control_value
            else:
                if self.parcae_p20_gate_projection is None:
                    raise RuntimeError(
                        "parcae P20 gate-control projection is not initialized"
                    )
                control_gate_logits = self.parcae_p20_gate_projection(control)
                if self.uses_parcae_p20_mod_gate_bias_control:
                    if (
                        mod_router_scores is None
                        or self.parcae_mod_gate_bias_scale is None
                    ):
                        raise RuntimeError("parcae MoD gate-bias control is not initialized")
                    scale = self.parcae_mod_gate_bias_scale.to(
                        dtype=control_gate_logits.dtype
                    )
                    control_gate_logits = control_gate_logits + scale * mod_router_scores.unsqueeze(-1)
                    self._last_parcae_mod_control_diagnostics[
                        "mod_router/gate_bias_scale"
                    ] = float(scale.detach().float().item())
                control_gate = torch.sigmoid(control_gate_logits)
                injection_gate = (
                    0.5 * (base_gate + control_gate)
                    if self.uses_parcae_p20_base_gate_blend
                    else control_gate
                )
                if self.uses_parcae_p20_gate_only_control:
                    injection_value = loop_input
                else:
                    if self.parcae_p20_value_projection is None:
                        raise RuntimeError(
                            "parcae P20 value-control projection is not initialized"
                        )
                    control_value = (
                        self.variant.parcae_p20_value_scale
                        * self.parcae_p20_value_projection(control)
                    )
                    if self.uses_parcae_p20_mod_value_scale_control:
                        if (
                            mod_salience is None
                            or self.parcae_mod_value_scale_strength is None
                        ):
                            raise RuntimeError(
                                "parcae MoD value-scale control is not initialized"
                            )
                        strength = self.parcae_mod_value_scale_strength.to(
                            dtype=control_value.dtype
                        )
                        value_multiplier = 1.0 + strength * (
                            mod_salience.unsqueeze(-1).to(dtype=control_value.dtype)
                            - 0.5
                        )
                        control_value = control_value * value_multiplier
                        self._last_parcae_mod_control_diagnostics[
                            "mod_router/value_scale_strength"
                        ] = float(strength.detach().float().item())
                        self._last_parcae_mod_control_diagnostics[
                            "mod_router/value_multiplier_mean"
                        ] = float(
                            value_multiplier.detach().float().mean().item()
                        )
                        self._last_parcae_mod_control_diagnostics[
                            "mod_router/value_multiplier_min"
                        ] = float(value_multiplier.detach().float().min().item())
                        self._last_parcae_mod_control_diagnostics[
                            "mod_router/value_multiplier_max"
                        ] = float(value_multiplier.detach().float().max().item())
                    injection_value = loop_input + control_value
            self._last_parcae_p20_control_norm = float(
                control.detach().float().norm(dim=-1).mean().item()
            )
            self._record_parcae_control_diagnostics(
                loop_input=loop_input,
                control=control,
                injection_gate=injection_gate,
                injection_value=injection_value,
                control_value=control_value,
            )
        elif self.uses_parcae_bx:
            if (
                self.parcae_b_value_projection is None
                or self.parcae_b_gate_projection is None
            ):
                raise RuntimeError("parcae B(x) scaffold is not initialized")
            injection_gate = torch.sigmoid(self.parcae_b_gate_projection(loop_input))
            injection_value = self.parcae_b_value_projection(loop_input)
            self._last_parcae_p20_control_norm = None
            self._last_parcae_control_diagnostics = {}
        else:
            injection_gate = torch.sigmoid(
                self.parcae_injection_logit.to(dtype=loop_input.dtype)
            ).view(1, 1, -1)
            injection_value = loop_input
            self._last_parcae_p20_control_norm = None
            self._last_parcae_control_diagnostics = {}
        self._last_parcae_injection_gate_mean = float(
            injection_gate.detach().float().mean().item()
        )
        self._last_parcae_injection_norm = float(
            (injection_gate * injection_value)
            .detach()
            .float()
            .norm(dim=-1)
            .mean()
            .item()
        )
        return injection_value, injection_gate

    def diagnostic_payload(self) -> dict[str, Any]:
        reference_ssm_blocks: list[dict[str, Any]] = []
        attention_blocks: list[dict[str, Any]] = []
        token_routed_blocks: list[dict[str, Any]] = []
        composite_branch_means: dict[str, list[float]] = {}
        for layer_index, block in enumerate(self.blocks):
            attention = getattr(block, "attention", None)
            attention_diagnostic = getattr(attention, "diagnostic_payload", None)
            if callable(attention_diagnostic):
                attention_blocks.append(
                    {
                        "layer_index": layer_index,
                        "payload": attention_diagnostic(),
                    }
                )
            token_routing_payload = getattr(block, "token_routing_payload", None)
            if callable(token_routing_payload):
                token_routed_blocks.append(
                    {
                        "layer_index": layer_index,
                        "payload": token_routing_payload(),
                    }
                )
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
                    composite_branch_means.setdefault(branch_name, []).append(
                        mean_weight
                    )

        diagnostics: dict[str, Any] = {}
        diagnostics["parameter_count"] = sum(
            parameter.numel() for parameter in self.parameters()
        )
        diagnostics["trainable_parameter_count"] = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        diagnostics["scaffold_profile"] = self.variant.scaffold_profile.value
        diagnostics["feed_forward_profile"] = self.variant.feed_forward_profile.value
        diagnostics["attention_profile"] = self.variant.attention_profile.value
        diagnostics["token_routing_profile"] = self.variant.token_routing_profile.value
        diagnostics["recurrent_token_routing_profile"] = (
            self.variant.recurrent_token_routing_profile.value
        )
        if self.variant.token_routing_profile is not TokenRoutingProfile.NONE:
            token_routing_labels = {
                TokenRoutingProfile.CAUSAL_TOPK_BLOCK: "Causal prefix-top-k token block routing; practical MoD approximation",
                TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK: "Paper MoD training-time full-sequence top-C block routing",
                TokenRoutingProfile.SOFT_GATE_BLOCK: "Decode-safe soft partial-update block routing",
                TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK: "Decode-safe P20/RGRP-style rotary soft partial-update block routing",
            }
            diagnostics["token_routing"] = {
                "profile": self.variant.token_routing_profile.value,
                "route_fraction": self.variant.token_route_fraction,
                "layer_indices": (
                    self.variant.token_routing_layer_indices
                    if self.variant.token_routing_layer_indices is not None
                    else "all"
                ),
                "blocks": token_routed_blocks,
                "label": token_routing_labels[self.variant.token_routing_profile],
            }
        if self.variant.attention_profile is AttentionProfile.MODA_DEPTH_KV:
            diagnostics["depth_augmented_attention"] = {
                "depth_memory_layers": self.variant.depth_memory_layers,
                "attention_blocks": attention_blocks,
                "label": "Approximate MoDA-style prior-depth hidden-state KV retrieval",
            }
        if self.variant.attention_profile is AttentionProfile.PAPER_MODA_DEPTH_KV:
            diagnostics["depth_augmented_attention"] = {
                "depth_memory_layers": self.variant.depth_memory_layers,
                "attention_blocks": attention_blocks,
                "label": "Paper-faithful MoDA same-token prior-depth sequence KV reference",
            }
        if self.variant.feed_forward_profile is not FeedForwardProfile.STANDARD:
            selected_layers = self.variant.feed_forward_layer_indices
            diagnostics["eml_inspired_feed_forward"] = {
                "slot_count": self.variant.eml_slot_count,
                "tree_depth": self.variant.eml_tree_depth,
                "leaf_count": 1 << self.variant.eml_tree_depth,
                "layer_indices": (
                    selected_layers if selected_layers is not None else "all"
                ),
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
        if self.uses_universal_transformer_scaffold:
            diagnostics["universal_transformer"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "stored_block_count": len(self.blocks),
                "effective_layer_count": len(self.blocks)
                * self.variant.parcae_loop_count,
                "parameters_shared_across_steps": True,
                "coordinate_embeddings_each_step": True,
                "position_coordinate_mode": "sinusoidal",
                "time_coordinate_mode": "sinusoidal",
                "act_enabled": self.uses_universal_transformer_act,
                "act_halting_threshold": self.variant.act_halting_threshold,
                "act_ponder_loss_weight": self.variant.act_ponder_loss_weight,
                "last_initial_state_norm": self._last_ut_initial_state_norm,
                "last_final_state_norm": self._last_ut_final_state_norm,
                "last_hidden_norms": list(self._last_ut_hidden_norms),
                "last_step_norms": list(self._last_ut_step_norms),
                "last_position_coordinate_norm": (
                    self._last_ut_position_coordinate_norm
                ),
                "last_step_coordinate_norms": list(self._last_ut_step_coordinate_norms),
                "average_steps_used": (
                    self._ut_total_steps / self._ut_forward_count
                    if self._ut_forward_count
                    else 0.0
                ),
                "forward_count": self._ut_forward_count,
                "label": (
                    "Universal Transformer fixed recurrence with tied transition and per-step coordinate signals"
                    if not self.uses_universal_transformer_act
                    else "Universal Transformer with ACT weighted-state halting reference"
                ),
            }
            if self.uses_universal_transformer_act:
                diagnostics["universal_transformer"]["act"] = {
                    "halt_probability_mean": self._last_act_halt_probability_mean,
                    "remainder_mean": self._last_act_remainder_mean,
                    "update_count_mean": self._last_act_update_count_mean,
                    "update_count_min": self._last_act_update_count_min,
                    "update_count_max": self._last_act_update_count_max,
                    "update_weight_sum_min": self._last_act_weight_sum_min,
                    "update_weight_sum_max": self._last_act_weight_sum_max,
                    "forced_final_halt_fraction": (
                        self._last_act_forced_final_halt_fraction
                    ),
                    "ponder_loss": self._last_act_ponder_loss,
                    "loss_contract": "LM loss plus separate ponder loss during training",
                }
        if self.uses_ouro_learned_exit:
            diagnostics["ouro_learned_exit"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "stored_block_count": len(self.blocks),
                "effective_layer_count": len(self.blocks)
                * self.variant.parcae_loop_count,
                "parameters_shared_across_steps": True,
                "embedding_applied_once": True,
                "output_head_applied_each_step": True,
                "gate": "per-token sigmoid exit gate with final-step survival mass",
                "entropy_weight": self.variant.ouro_entropy_weight,
                "q_exit_threshold": self.variant.ouro_q_exit_threshold,
                "exit_pdf_mean_by_step": list(self._last_ouro_exit_pdf_mean_by_step),
                "exit_pdf_sum_min": self._last_ouro_exit_pdf_sum_min,
                "exit_pdf_sum_max": self._last_ouro_exit_pdf_sum_max,
                "final_step_receives_remaining_survival": True,
                "final_step_mass_mean": self._last_ouro_final_step_mass_mean,
                "expected_exit_step_mean": self._last_ouro_expected_exit_step_mean,
                "q_exit_step_mean": self._last_ouro_q_exit_step_mean,
                "q_exit_step_min": self._last_ouro_q_exit_step_min,
                "q_exit_step_max": self._last_ouro_q_exit_step_max,
                "last_gate_probability_means": list(
                    self._last_ouro_gate_probability_means
                ),
                "last_hidden_norms": list(self._last_ouro_hidden_norms),
                "last_step_norms": list(self._last_ouro_step_norms),
                "last_initial_state_norm": self._last_ouro_initial_state_norm,
                "last_final_state_norm": self._last_ouro_final_state_norm,
                "last_expected_ce": self._last_ouro_expected_ce,
                "last_entropy_mean": self._last_ouro_entropy_mean,
                "last_entropy_regularization": (self._last_ouro_entropy_regularization),
                "last_per_step_ce_mean": list(self._last_ouro_per_step_ce_mean),
                "average_compute_steps": (
                    self._ouro_total_steps / self._ouro_forward_count
                    if self._ouro_forward_count
                    else 0.0
                ),
                "forward_count": self._ouro_forward_count,
                "loss_contract": (
                    "Stage-1 expected CE over the exit distribution plus a separate "
                    "negative entropy auxiliary term during training"
                ),
                "deferred_contract": (
                    "Stage-2 frozen-LM continuation-benefit gate training is not "
                    "implemented in this primitive yet"
                ),
                "label": "Ouro-style learned exit distribution on a tied looped decoder stack",
            }
        if self.uses_rrt_cycle_scaffold:
            cycle_map = [
                {
                    "absolute_depth": absolute_depth,
                    "shared_layer_index": self._rrt_shared_layer_index(absolute_depth),
                }
                for absolute_depth in range(self.rrt_effective_layer_count)
            ]
            diagnostics["rrt_cycle"] = {
                "profile": self.variant.scaffold_profile.value,
                "recursion_count": self.variant.parcae_loop_count,
                "stored_layer_count": self.rrt_stored_layer_count,
                "effective_layer_count": self.rrt_effective_layer_count,
                "parameters_shared_by_cycle": True,
                "cycle_rule": "shared_layer_index = absolute_depth % stored_layer_count",
                "cycle_map": cycle_map,
                "last_shared_layer_indices": list(self._last_rrt_shared_layer_indices),
                "absolute_depth_cache_keys": list(
                    range(self.rrt_effective_layer_count)
                ),
                "cache_key_mode": "absolute_depth",
                "incremental_kv_cache_implemented": False,
                "relaxed_lora_enabled": False,
                "lora_rank": 0,
                "strict_recursion_shares_norms": True,
                "last_initial_state_norm": self._last_rrt_initial_state_norm,
                "last_final_state_norm": self._last_rrt_final_state_norm,
                "last_hidden_norms": list(self._last_rrt_hidden_norms),
                "last_step_norms": list(self._last_rrt_step_norms),
                "average_effective_layers": (
                    self._rrt_total_effective_layers / self._rrt_forward_count
                    if self._rrt_forward_count
                    else 0.0
                ),
                "forward_count": self._rrt_forward_count,
                "label": (
                    "Strict Relaxed Recursive Transformer CYCLE sharing; "
                    "LoRA/SVD relaxation deferred"
                ),
            }
        if self.uses_mor_expert_choice_scaffold:
            diagnostics["mor_expert_choice"] = {
                "profile": self.variant.scaffold_profile.value,
                "max_recursions": self.variant.parcae_loop_count,
                "stored_layer_count": len(self.blocks),
                "unique_prelude_layer_count": 1,
                "shared_middle_layer_count": max(len(self.blocks) - 2, 0),
                "unique_coda_layer_count": 1,
                "effective_max_layer_count": 2
                + max(len(self.blocks) - 2, 0) * self.variant.parcae_loop_count,
                "middle_cycle_sharing": True,
                "routing": "expert-choice full-sequence top-k over currently active tokens",
                "capacity_fraction": self.variant.recurrent_token_route_fraction,
                "router_count": (
                    len(self.mor_routers) if self.mor_routers is not None else 0
                ),
                "router_aux_loss_weight": self.variant.mor_router_aux_loss_weight,
                "router_update_scale": self.variant.mor_update_scale,
                "router_aux_loss": self._last_mor_router_aux_loss,
                "kv_policy": "recursion-wise-selected-subsequence-reference",
                "decode_safe": False,
                "selected_indices_sorted": all(
                    positions == sorted(positions)
                    for recursion_positions in self._last_mor_selected_positions
                    for positions in recursion_positions
                ),
                "unselected_tokens_stop_recursing": all(self._last_mor_subset_flags),
                "last_active_token_counts": list(self._last_mor_active_token_counts),
                "last_selected_token_counts": list(
                    self._last_mor_selected_token_counts
                ),
                "last_selected_token_fractions": list(
                    self._last_mor_selected_token_fractions
                ),
                "last_selected_gate_means": list(self._last_mor_selected_gate_means),
                "last_selected_positions": list(self._last_mor_selected_positions),
                "last_subset_flags": list(self._last_mor_subset_flags),
                "average_selected_token_fraction": (
                    self._mor_total_selected_tokens / self._mor_total_possible_tokens
                    if self._mor_total_possible_tokens
                    else 0.0
                ),
                "last_initial_state_norm": self._last_mor_initial_state_norm,
                "last_final_state_norm": self._last_mor_final_state_norm,
                "last_hidden_norms": list(self._last_mor_hidden_norms),
                "last_step_norms": list(self._last_mor_step_norms),
                "forward_count": self._mor_forward_count,
                "label": (
                    "MoR expert-choice reference with real active-token shrinkage; "
                    "full-sequence router is a training/eval primitive, not a decode path"
                ),
            }
        if self.uses_looped_transformer_scaffold:
            if self.uses_huginn_adapter_recurrence:
                injection_mode = "concat-adapter"
                initial_state = "zero"
            elif self.uses_looped_additive_input:
                injection_mode = "additive"
                initial_state = "zero"
            else:
                injection_mode = "none"
                initial_state = "input-embedding"
            diagnostics["looped_transformer"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "stored_block_count": len(self.blocks),
                "effective_layer_count": len(self.blocks)
                * self.variant.parcae_loop_count,
                "parameters_shared_across_loops": True,
                "embedding_applied_once": True,
                "output_head_applied_once": True,
                "initial_state": initial_state,
                "input_injection_mode": injection_mode,
                "prompt_injected_each_loop": injection_mode
                in {"additive", "concat-adapter"},
                "adapter": (
                    {
                        "kind": "concat-linear",
                        "input_width": self.variant.shape.d_model * 2,
                        "output_width": self.variant.shape.d_model,
                    }
                    if self.uses_huginn_adapter_recurrence
                    else None
                ),
                "last_initial_state_norm": self._last_looped_initial_state_norm,
                "last_final_state_norm": self._last_looped_final_state_norm,
                "last_hidden_norms": list(self._last_looped_hidden_norms),
                "last_step_norms": list(self._last_looped_step_norms),
                "last_input_injection_norms": list(
                    self._last_looped_input_injection_norms
                ),
                "last_adapter_input_norms": list(self._last_looped_adapter_input_norms),
                "last_adapter_output_norms": list(
                    self._last_looped_adapter_output_norms
                ),
                "average_steps_used": (
                    self._looped_total_steps / self._looped_forward_count
                    if self._looped_forward_count
                    else 0.0
                ),
                "forward_count": self._looped_forward_count,
                "label": (
                    "Strict fixed looped LM with a shared k-layer block group"
                    if injection_mode == "none"
                    else (
                        "Looped Transformer input-injected recurrence Y <- M(Y + P)"
                        if injection_mode == "additive"
                        else "Huginn-style recurrent adapter R(e, s) with deterministic zero initial state"
                    )
                ),
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
            diagnostics["parcae_looped_attention"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "halting_profile": self.variant.recurrent_halting_profile.value,
                "halting_threshold": self.variant.recurrent_halting_threshold,
                "min_steps": self.variant.recurrent_min_steps,
                "token_routing_profile": self.variant.recurrent_token_routing_profile.value,
                "token_route_fraction": self.variant.recurrent_token_route_fraction,
                "last_selected_token_fractions": list(
                    self._last_parcae_selected_token_fractions
                ),
                "last_selected_gate_means": list(self._last_parcae_selected_gate_means),
                "average_selected_token_fraction": (
                    self._parcae_selected_tokens / self._parcae_possible_tokens
                    if self._parcae_possible_tokens
                    else 0.0
                ),
                "last_steps_used": self._last_parcae_steps_used,
                "average_steps_used": (
                    self._parcae_total_steps / self._parcae_forward_count
                    if self._parcae_forward_count
                    else 0.0
                ),
                "early_exit_count": self._parcae_exit_count,
                "forward_count": self._parcae_forward_count,
                "last_halted_early": self._last_parcae_halted_early,
                "last_step_norms": list(self._last_parcae_step_norms),
                "last_step_cosines": list(self._last_parcae_step_cosines),
                "last_step_accelerations": list(self._last_parcae_step_accelerations),
                "last_drift_norms": list(self._last_parcae_drift_norms),
                "last_halting_metric": self._last_parcae_halting_metric,
                "prelude_layers": self.parcae_loop_start,
                "recurrent_layers": self.parcae_loop_end - self.parcae_loop_start,
                "coda_layers": len(self.blocks) - self.parcae_loop_end,
                "decay_mean": float(decay.mean().item()),
                "decay_min": float(decay.min().item()),
                "decay_max": float(decay.max().item()),
                "injection_mean": float(injection.mean().item()),
                "last_injection_gate_mean": self._last_parcae_injection_gate_mean,
                "last_injection_norm": self._last_parcae_injection_norm,
                "p20_control_width": self.parcae_p20_control_width,
                "p20_control_mode": self.parcae_p20_control_mode,
                "p20_value_scale": self.variant.parcae_p20_value_scale,
                "p20_gate_only_control": self.uses_parcae_p20_gate_only_control,
                "p20_value_only_control": self.uses_parcae_p20_value_only_control,
                "p20_base_gate_blend": self.uses_parcae_p20_base_gate_blend,
                "last_p20_control_norm": self._last_parcae_p20_control_norm,
                "nonlinear_delta_scale_mean": float(nonlinear.mean().item()),
                "last_recurrent_state_norms": list(self._last_parcae_norms),
                "label": "Parcae-inspired stable middle-loop scaffold, not an exact reproduction",
            }
            control_diagnostics = self._parcae_control_diagnostic_payload()
            if control_diagnostics:
                diagnostics["parcae_looped_attention"][
                    "control_diagnostics"
                ] = control_diagnostics
        return diagnostics

    def configure_runtime_policy(
        self,
        *,
        compile_mode: str | None,
        primitive_runtime_backend: str | None = "torch",
    ) -> None:
        for block in self.blocks:
            configure = getattr(block, "configure_runtime_policy", None)
            if callable(configure):
                configure(
                    compile_mode=compile_mode,
                    primitive_runtime_backend=primitive_runtime_backend,
                )
        if self.parcae_p20_controller is not None:
            self.parcae_p20_controller.configure_runtime_policy(
                compile_mode=compile_mode,
                primitive_runtime_backend=primitive_runtime_backend,
            )

    def optimizer_parameter_groups(self, base_lr: float) -> list[dict[str, object]]:
        if self.uses_pr5_scaffold:
            groups: dict[str, dict[str, object]] = {
                "default": {"name": "default", "params": [], "lr": base_lr},
                "pr5_context": {"name": "pr5_context", "params": [], "lr": base_lr},
                "pr5_recurrent": {
                    "name": "pr5_recurrent",
                    "params": [],
                    "lr": base_lr * 0.5,
                },
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
            return [
                {"name": "default", "params": list(self.parameters()), "lr": base_lr}
            ]

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
                marker in name
                for marker in (
                    ".primitive.q_local",
                    ".primitive.k_local",
                    ".primitive.v_local",
                )
            ):
                groups["p20_gdn_recurrent"]["params"].append(param)
            elif ".primitive.control_projection" in name:
                groups["p20_gdn_gates"]["params"].append(param)
            elif any(
                marker in name
                for marker in (".primitive.output_norm", ".primitive.output_projection")
            ):
                groups["p20_gdn_readout"]["params"].append(param)
            elif ".primitive.readout_ramp_logit" in name or name.endswith(
                ".residual_scale"
            ):
                groups["p20_gdn_scalars"]["params"].append(param)
            else:
                groups["default"]["params"].append(param)
        return [group for group in groups.values() if group["params"]]


def build_path1_model(
    variant: Path1VariantSpec, *, dtype_mode: str
) -> Path1HybridLanguageModel:
    return Path1HybridLanguageModel(variant, dtype_mode=dtype_mode)
