"""Runtime helpers for research runs."""

from .compilation import apply_runtime_policy
from .cuda_timing import (
    CudaEventTimingCollector,
    cuda_timing_summary_to_dict,
    timed_region,
    use_cuda_timing,
    use_named_timing_regions,
)
from .optimizers import (
    CompositeOptimizer,
    MuonParameterSplit,
    NativeAdamParameterSplit,
    ReferenceMuon,
    TritonAdam2D,
    build_optimizer,
    split_muon_parameters,
    split_triton_adam_2d_parameters,
)
from .parcae_loop_region import (
    PARCAE_LOOP_REGION_TIMING_NAMES,
    ParcaeLoopRegionConfig,
    ParcaeLoopRegionControls,
    ParcaeLoopRegionKernels,
    ParcaeLoopRegionResult,
    ParcaeLoopRegionTensorLayout,
    ParcaeLoopRegionTimingNames,
    run_parcae_loop_region,
)
from .recurrent import (
    BlockDiagonalLinear,
    PackedLinearProjection,
    SequencePrimitiveScanResult,
    SequencePrimitiveStepResult,
    allocate_emitted_outputs,
    build_state_transform_projection,
    packed_linear_chunks,
    rotate_state_pairs_with_trig,
    rotary_runtime_components,
)
from .seeding import configure_reproducibility, resolve_autocast_dtype, resolve_torch_device
from .train_eval import (
    evaluate_model,
    language_model_cross_entropy,
    materialize_batch,
    run_training_benchmark,
    warmup_model,
)
