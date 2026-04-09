from __future__ import annotations

from .common import StringEnum


class RuntimeOptimizationFamily(StringEnum):
    PURE_TRANSFORMER = "pure-transformer"
    RECURRENT_SCAN_HYBRID = "recurrent-scan-hybrid"
    TRANSFORMER_MOE_ROUTING = "transformer-moe-routing"


class PrimitiveStateTransformMode(StringEnum):
    DENSE = "dense"
    BLOCK_DIAGONAL_2 = "block-diagonal-2"
    BLOCK_DIAGONAL_4 = "block-diagonal-4"
