from __future__ import annotations

import torch
import torch.nn as nn

from python.models.common import SimpleRmsNorm
from python.models.primitives import PrimitiveMixerBlock
from python.models.reference_ssm import ReferenceSsmHybridBlock
from python.models.transformer import LocalCausalTransformerBlock, local_causal_attention_bias
from python.specs.path1 import (
    BYTE_LEVEL_PAD_TOKEN,
    HybridAttentionLayerRole,
    Path1VariantSpec,
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
        for role in variant.layer_schedule:
            if role is HybridAttentionLayerRole.EXACT_ATTENTION:
                self.blocks.append(
                    LocalCausalTransformerBlock(
                        variant.shape.d_model,
                        variant.shape.head_count,
                        variant.shape.d_ff,
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
        mask = local_causal_attention_bias(
            input_ids.shape[1],
            self.variant.shape.local_window,
            input_ids.device,
            hidden.dtype,
        )
        for block in self.blocks:
            hidden = block(hidden, mask)
        hidden = self.final_norm(hidden)
        return self.output(hidden)


def build_path1_model(variant: Path1VariantSpec, *, dtype_mode: str) -> Path1HybridLanguageModel:
    return Path1HybridLanguageModel(variant, dtype_mode=dtype_mode)
