from __future__ import annotations

from .common import StringEnum


class PrimitiveStateTransformMode(StringEnum):
    DENSE = "dense"
    BLOCK_DIAGONAL_2 = "block-diagonal-2"
    BLOCK_DIAGONAL_4 = "block-diagonal-4"
