"""Runtime helpers for research runs."""

from .compilation import apply_runtime_policy
from .recurrent import (
    BlockDiagonalLinear,
    SequencePrimitiveScanResult,
    SequencePrimitiveStepResult,
    allocate_emitted_outputs,
    build_state_transform_projection,
    packed_linear_chunks,
    rotate_state_pairs_with_trig,
    rotary_runtime_components,
)
from .seeding import configure_reproducibility, resolve_autocast_dtype, resolve_torch_device
from .train_eval import evaluate_model, materialize_batch, run_training_benchmark, warmup_model
