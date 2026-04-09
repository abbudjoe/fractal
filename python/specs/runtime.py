from __future__ import annotations

from dataclasses import dataclass

from .common import StringEnum


class PrimitiveStateTransformMode(StringEnum):
    DENSE = "dense"
    BLOCK_DIAGONAL_2 = "block-diagonal-2"
    BLOCK_DIAGONAL_4 = "block-diagonal-4"


class RuntimeOptimizationScope(StringEnum):
    UNIVERSAL_SEQUENCE = "universal-sequence"
    RECURRENT_SCAN = "recurrent-scan"
    PURE_TRANSFORMER = "pure-transformer"


class RuntimeOptimizationTarget(StringEnum):
    PACKED_PROJECTIONS = "packed-projections"
    LAYOUT_STRIDE_CONTRACT = "layout-stride-contract"
    AUTOTUNED_KERNEL_LAUNCH = "autotuned-kernel-launch"
    CHUNKED_EXECUTION = "chunked-execution"
    BACKWARD_WORKSPACE = "backward-workspace"
    CHUNKED_STATE_PASSING = "chunked-state-passing"
    SEQUENCE_SCAN_KERNEL = "sequence-scan-kernel"
    RECURRENCE_FUSION_BOUNDARY = "recurrence-fusion-boundary"
    STRUCTURED_STATE_TRANSFORM = "structured-state-transform"
    LATENT_STATE_UPDATE_KERNEL = "latent-state-update-kernel"
    ATTENTION_KERNEL = "attention-kernel"
    KV_CACHE_LAYOUT = "kv-cache-layout"
    MLP_FUSION = "mlp-fusion"
    SEQUENCE_PARALLELISM = "sequence-parallelism"
    MEMORY_EFFICIENT_ATTENTION_BACKWARD = "memory-efficient-attention-backward"


@dataclass(frozen=True)
class RuntimeOptimizationProfile:
    architecture_family: str
    scopes: tuple[RuntimeOptimizationScope, ...]
    targets: tuple[RuntimeOptimizationTarget, ...]


def runtime_optimization_profile(architecture_family: str) -> RuntimeOptimizationProfile:
    normalized = architecture_family.strip().lower().replace("_", "-")
    if normalized in {"recurrent-scan-hybrid", "transformer-ssm-hybrid", "scan-hybrid"}:
        return RuntimeOptimizationProfile(
            architecture_family="recurrent-scan-hybrid",
            scopes=(
                RuntimeOptimizationScope.UNIVERSAL_SEQUENCE,
                RuntimeOptimizationScope.RECURRENT_SCAN,
            ),
            targets=(
                RuntimeOptimizationTarget.PACKED_PROJECTIONS,
                RuntimeOptimizationTarget.LAYOUT_STRIDE_CONTRACT,
                RuntimeOptimizationTarget.AUTOTUNED_KERNEL_LAUNCH,
                RuntimeOptimizationTarget.CHUNKED_EXECUTION,
                RuntimeOptimizationTarget.BACKWARD_WORKSPACE,
                RuntimeOptimizationTarget.CHUNKED_STATE_PASSING,
                RuntimeOptimizationTarget.SEQUENCE_SCAN_KERNEL,
                RuntimeOptimizationTarget.RECURRENCE_FUSION_BOUNDARY,
                RuntimeOptimizationTarget.STRUCTURED_STATE_TRANSFORM,
                RuntimeOptimizationTarget.LATENT_STATE_UPDATE_KERNEL,
            ),
        )
    if normalized in {"pure-transformer", "transformer"}:
        return RuntimeOptimizationProfile(
            architecture_family="pure-transformer",
            scopes=(
                RuntimeOptimizationScope.UNIVERSAL_SEQUENCE,
                RuntimeOptimizationScope.PURE_TRANSFORMER,
            ),
            targets=(
                RuntimeOptimizationTarget.PACKED_PROJECTIONS,
                RuntimeOptimizationTarget.LAYOUT_STRIDE_CONTRACT,
                RuntimeOptimizationTarget.AUTOTUNED_KERNEL_LAUNCH,
                RuntimeOptimizationTarget.CHUNKED_EXECUTION,
                RuntimeOptimizationTarget.BACKWARD_WORKSPACE,
                RuntimeOptimizationTarget.ATTENTION_KERNEL,
                RuntimeOptimizationTarget.KV_CACHE_LAYOUT,
                RuntimeOptimizationTarget.MLP_FUSION,
                RuntimeOptimizationTarget.SEQUENCE_PARALLELISM,
                RuntimeOptimizationTarget.MEMORY_EFFICIENT_ATTENTION_BACKWARD,
            ),
        )
    raise ValueError(f"unsupported runtime optimization architecture family: {architecture_family}")
