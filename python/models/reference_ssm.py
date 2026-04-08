from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.profiler import record_function

from python.models.common import PositionWiseFeedForward, SimpleRmsNorm
from python.specs.path1 import ReferenceSsmProfile


@dataclass(frozen=True)
class ResolvedReferenceSsmConfig:
    d_model: int
    head_count: int
    d_state: int = 128
    expand: int = 2
    is_mimo: bool = False
    mimo_rank: int = 1
    chunk_size: int = 16
    is_outproj_norm: bool = False
    profile: ReferenceSsmProfile = ReferenceSsmProfile.MAMBA3_SISO_RUNTIME
    runtime_oriented: bool = False


def resolve_reference_ssm_config(
    d_model: int,
    head_count: int,
    profile: ReferenceSsmProfile,
    dtype_mode: str,
) -> ResolvedReferenceSsmConfig:
    if profile is ReferenceSsmProfile.MAMBA3_SISO_REFERENCE:
        chunk_size = 1
    else:
        chunk_size = 16 if dtype_mode == "bf16" else 8
    return ResolvedReferenceSsmConfig(
        d_model=d_model,
        head_count=head_count,
        is_mimo=profile.is_mimo,
        mimo_rank=profile.mimo_rank,
        chunk_size=chunk_size,
        profile=profile,
        runtime_oriented=profile.runtime_oriented,
    )


class OfficialMamba3SequenceMixer(nn.Module):
    def __init__(self, config: ResolvedReferenceSsmConfig) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba3
        except Exception as exc:  # pragma: no cover - exercised in runtime environments with mamba_ssm
            raise RuntimeError(
                "official PyTorch Mamba3 import failed. Install requirements from scripts/requirements-v3a-python-mamba3.txt"
            ) from exc

        self.mixer = Mamba3(
            d_model=config.d_model,
            d_state=config.d_state,
            headdim=config.d_model // config.head_count,
            is_mimo=config.is_mimo,
            mimo_rank=config.mimo_rank,
            chunk_size=config.chunk_size,
            is_outproj_norm=config.is_outproj_norm,
        )

    def forward(self, hidden: torch.Tensor, _attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.reference_ssm.native_mamba3"):
            return self.mixer(hidden)


class ReferenceSsmHybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_count: int,
        d_ff: int,
        *,
        profile: ReferenceSsmProfile,
        dtype_mode: str,
    ) -> None:
        super().__init__()
        self.input_norm = SimpleRmsNorm(d_model)
        self.output_norm = SimpleRmsNorm(d_model)
        self.mixer = OfficialMamba3SequenceMixer(
            resolve_reference_ssm_config(d_model=d_model, head_count=head_count, profile=profile, dtype_mode=dtype_mode)
        )
        self.feedforward = PositionWiseFeedForward(d_model, d_ff)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("path1.reference_ssm.input_norm"):
            normed = self.input_norm(hidden)
        mixed = self.mixer(normed, attn_mask)
        residual = hidden + mixed
        with record_function("path1.reference_ssm.feedforward"):
            return residual + self.feedforward(self.output_norm(residual))
