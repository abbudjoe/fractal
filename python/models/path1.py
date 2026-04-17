from __future__ import annotations

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
from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer, PrimitiveMixerBlock
from python.models.reference_ssm import ReferenceSsmHybridBlock
from python.models.transformer import (
    LocalCausalSelfAttention,
    LocalCausalTransformerBlock,
    Pr5LocalCausalTransformerBlock,
    local_causal_attention_bias,
)
from python.specs.path1 import (
    FeedForwardProfile,
    HybridAttentionLayerRole,
    Path1ScaffoldProfile,
    Path1VariantSpec,
    PrimitiveProfile,
)
from python.specs.runtime import PrimitiveStateTransformMode


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


def _middle_loop_bounds(total_layers: int) -> tuple[int, int]:
    loop_width = max(1, total_layers // 3)
    start = max(0, (total_layers - loop_width) // 2)
    end = min(total_layers, start + loop_width)
    return start, end


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
        }
        self.uses_parcae_bx = variant.scaffold_profile is Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION
        self.uses_parcae_p20_control = (
            variant.scaffold_profile is Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
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
        self.smear_gate = SmearGate(variant.shape.d_model) if self.uses_pr5_scaffold else None
        if self.uses_pr5_scaffold:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.005)
        self.blocks = nn.ModuleList()
        shared_attention: LocalCausalSelfAttention | None = None
        reference_ordinal = 0
        for layer_index, role in enumerate(variant.layer_schedule):
            if role is HybridAttentionLayerRole.EXACT_ATTENTION:
                attention = LocalCausalSelfAttention(
                    variant.shape.d_model,
                    variant.shape.head_count,
                )
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
                    else LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=attention,
                        ffn_module=_build_attention_feed_forward(variant, layer_index=layer_index),
                    )
                )
                self.blocks.append(attention_block)
            elif role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION:
                if shared_attention is None:
                    shared_attention = LocalCausalSelfAttention(
                        variant.shape.d_model,
                        variant.shape.head_count,
                    )
                attention_block = (
                    Pr5LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=shared_attention,
                    )
                    if self.uses_pr5_scaffold
                    else LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=shared_attention,
                        ffn_module=_build_attention_feed_forward(variant, layer_index=layer_index),
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
            self.parcae_loop_start, self.parcae_loop_end = _middle_loop_bounds(len(self.blocks))
            self.parcae_prelude_norm = nn.LayerNorm(variant.shape.d_model)
            self.parcae_decay_raw = nn.Parameter(torch.full((variant.shape.d_model,), -2.0, dtype=torch.float32))
            self.parcae_injection_logit = nn.Parameter(torch.full((variant.shape.d_model,), -2.1972246, dtype=torch.float32))
            self.parcae_nonlinear_logit = nn.Parameter(torch.zeros(variant.shape.d_model, dtype=torch.float32))
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
            self.parcae_p20_controller = (
                P20RotaryStateOutputRuntimeSequenceMixer(
                    variant.shape.d_model,
                    state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
                )
                if self.uses_parcae_p20_control
                else None
            )
            self.parcae_p20_control_norm = SimpleRmsNorm(variant.shape.d_model) if self.uses_parcae_p20_control else None
            self.parcae_p20_value_projection = (
                nn.Linear(variant.shape.d_model, variant.shape.d_model, bias=False)
                if self.uses_parcae_p20_control
                else None
            )
            self.parcae_p20_gate_projection = (
                nn.Linear(variant.shape.d_model, variant.shape.d_model)
                if self.uses_parcae_p20_control
                else None
            )
            if self.parcae_p20_value_projection is not None:
                nn.init.normal_(self.parcae_p20_value_projection.weight, mean=0.0, std=1.0e-3)
            if self.parcae_p20_gate_projection is not None:
                nn.init.normal_(self.parcae_p20_gate_projection.weight, mean=0.0, std=1.0e-3)
                nn.init.constant_(self.parcae_p20_gate_projection.bias, -2.1972246)
            self._last_parcae_norms: list[float] = []
            self._last_parcae_injection_gate_mean: float | None = None
            self._last_parcae_injection_norm: float | None = None
            self._last_parcae_p20_control_norm: float | None = None
        else:
            self.parcae_loop_start = 0
            self.parcae_loop_end = 0
            self.parcae_prelude_norm = None
            self.parcae_decay_raw = None
            self.parcae_injection_logit = None
            self.parcae_nonlinear_logit = None
            self.parcae_b_value_projection = None
            self.parcae_b_gate_projection = None
            self.parcae_p20_controller = None
            self.parcae_p20_control_norm = None
            self.parcae_p20_value_projection = None
            self.parcae_p20_gate_projection = None
            self._last_parcae_norms = []
            self._last_parcae_injection_gate_mean = None
            self._last_parcae_injection_norm = None
            self._last_parcae_p20_control_norm = None
        self.final_norm = SimpleRmsNorm(variant.shape.d_model) if variant.final_norm_kind == "rmsnorm" else nn.Identity()
        self.output = nn.Linear(variant.shape.d_model, variant.shape.vocab_size, bias=False)

    @property
    def model_label(self) -> str:
        return f"path1_{self.variant.label.replace('-', '_')}"

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
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
        if self.uses_parcae_scaffold:
            hidden = self._forward_parcae_loop(hidden, mask)
        else:
            for block in self.blocks:
                if self.uses_pr5_scaffold:
                    hidden = block(hidden, mask, residual_anchor)
                else:
                    hidden = block(hidden, mask)
        hidden = self.final_norm(hidden)
        return self.output(hidden)

    def _forward_parcae_loop(self, hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if (
            self.parcae_prelude_norm is None
            or self.parcae_decay_raw is None
            or self.parcae_injection_logit is None
            or self.parcae_nonlinear_logit is None
        ):
            raise RuntimeError("parcae scaffold is not initialized")

        for block in self.blocks[: self.parcae_loop_start]:
            hidden = block(hidden, mask)

        loop_input = self.parcae_prelude_norm(hidden)
        state = torch.zeros_like(loop_input)
        decay = torch.exp(-F.softplus(self.parcae_decay_raw.to(dtype=hidden.dtype))).view(1, 1, -1)
        nonlinear = torch.sigmoid(self.parcae_nonlinear_logit.to(dtype=hidden.dtype)).view(1, 1, -1)
        injection_value, injection_gate = self._parcae_injection(loop_input)
        norm_history: list[float] = []
        recurrent_blocks = self.blocks[self.parcae_loop_start : self.parcae_loop_end]
        for _ in range(self.variant.parcae_loop_count):
            mixed = decay * state + injection_gate * injection_value
            for block in recurrent_blocks:
                block_out = block(mixed, mask)
                mixed = mixed + nonlinear * (block_out - mixed)
            state = mixed
            norm_history.append(float(state.detach().float().norm(dim=-1).mean().item()))

        hidden = state
        for block in self.blocks[self.parcae_loop_end :]:
            hidden = block(hidden, mask)
        self._last_parcae_norms = norm_history
        return hidden

    def _parcae_injection(self, loop_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.parcae_injection_logit is None:
            raise RuntimeError("parcae scaffold is not initialized")
        if self.uses_parcae_p20_control:
            if (
                self.parcae_p20_controller is None
                or self.parcae_p20_control_norm is None
                or self.parcae_p20_value_projection is None
                or self.parcae_p20_gate_projection is None
            ):
                raise RuntimeError("parcae P20-control scaffold is not initialized")
            control = self.parcae_p20_controller.scan(loop_input).emitted_outputs
            control = self.parcae_p20_control_norm(control)
            base_gate = torch.sigmoid(self.parcae_injection_logit.to(dtype=loop_input.dtype)).view(1, 1, -1)
            control_gate = torch.sigmoid(self.parcae_p20_gate_projection(control))
            control_value = self.parcae_p20_value_projection(control)
            injection_gate = control_gate
            injection_value = loop_input + control_value
            del base_gate
            self._last_parcae_p20_control_norm = float(control.detach().float().norm(dim=-1).mean().item())
        elif self.uses_parcae_bx:
            if self.parcae_b_value_projection is None or self.parcae_b_gate_projection is None:
                raise RuntimeError("parcae B(x) scaffold is not initialized")
            injection_gate = torch.sigmoid(self.parcae_b_gate_projection(loop_input))
            injection_value = self.parcae_b_value_projection(loop_input)
            self._last_parcae_p20_control_norm = None
        else:
            injection_gate = torch.sigmoid(self.parcae_injection_logit.to(dtype=loop_input.dtype)).view(1, 1, -1)
            injection_value = loop_input
            self._last_parcae_p20_control_norm = None
        self._last_parcae_injection_gate_mean = float(injection_gate.detach().float().mean().item())
        self._last_parcae_injection_norm = float(
            (injection_gate * injection_value).detach().float().norm(dim=-1).mean().item()
        )
        return injection_value, injection_gate

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
            diagnostics["parcae_looped_attention"] = {
                "profile": self.variant.scaffold_profile.value,
                "loop_count": self.variant.parcae_loop_count,
                "prelude_layers": self.parcae_loop_start,
                "recurrent_layers": self.parcae_loop_end - self.parcae_loop_start,
                "coda_layers": len(self.blocks) - self.parcae_loop_end,
                "decay_mean": float(decay.mean().item()),
                "decay_min": float(decay.min().item()),
                "decay_max": float(decay.max().item()),
                "injection_mean": float(injection.mean().item()),
                "last_injection_gate_mean": self._last_parcae_injection_gate_mean,
                "last_injection_norm": self._last_parcae_injection_norm,
                "last_p20_control_norm": self._last_parcae_p20_control_norm,
                "nonlinear_delta_scale_mean": float(nonlinear.mean().item()),
                "last_recurrent_state_norms": list(self._last_parcae_norms),
                "label": "Parcae-inspired stable middle-loop scaffold, not an exact reproduction",
            }
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
