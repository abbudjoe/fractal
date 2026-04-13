from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from python.models.common import SimpleRmsNorm
from python.models.primitives import PrimitiveMixerBlock
from python.models.reference_ssm import ReferenceSsmHybridBlock
from python.models.transformer import LocalCausalSelfAttention, LocalCausalTransformerBlock, local_causal_attention_bias
from python.specs.path1 import (
    BYTE_LEVEL_PAD_TOKEN,
    HybridAttentionLayerRole,
    Path1VariantSpec,
    PrimitiveProfile,
)


class Path1HybridLanguageModel(nn.Module):
    def __init__(self, variant: Path1VariantSpec, *, dtype_mode: str) -> None:
        super().__init__()
        variant.validate()
        self.variant = variant
        self.embedding = nn.Embedding(
            variant.shape.vocab_size,
            variant.shape.d_model,
            padding_idx=BYTE_LEVEL_PAD_TOKEN,
        )
        self.blocks = nn.ModuleList()
        shared_attention: LocalCausalSelfAttention | None = None
        for layer_index, role in enumerate(variant.layer_schedule):
            if role is HybridAttentionLayerRole.EXACT_ATTENTION:
                attention = LocalCausalSelfAttention(
                    variant.shape.d_model,
                    variant.shape.head_count,
                )
                if shared_attention is None:
                    shared_attention = attention
                self.blocks.append(
                    LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=attention,
                    )
                )
            elif role is HybridAttentionLayerRole.SHARED_EXACT_ATTENTION:
                if shared_attention is None:
                    shared_attention = LocalCausalSelfAttention(
                        variant.shape.d_model,
                        variant.shape.head_count,
                    )
                self.blocks.append(
                    LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        attention_module=shared_attention,
                    )
                )
            elif role is HybridAttentionLayerRole.REFERENCE_SSM:
                self.blocks.append(
                    ReferenceSsmHybridBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
                        profile=variant.reference_ssm_profile,
                        dtype_mode=dtype_mode,
                        layer_index=layer_index,
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
        self.final_norm = SimpleRmsNorm(variant.shape.d_model) if variant.final_norm_kind == "rmsnorm" else nn.Identity()
        self.output = nn.Linear(variant.shape.d_model, variant.shape.vocab_size, bias=False)

    @property
    def model_label(self) -> str:
        return f"path1_{self.variant.label.replace('-', '_')}"

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
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
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_norm(hidden)
        return self.output(hidden)

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

    def optimizer_parameter_groups(self, base_lr: float) -> list[dict[str, object]]:
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
