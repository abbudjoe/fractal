pub mod data_generator;
pub mod diagnostics;
pub mod error;
pub mod fitness;
pub mod hybrid;
pub mod hybrid_attention;
pub mod language_model_head;
pub mod lifecycle;
pub mod model;
pub mod primitives;
pub mod projection;
pub mod registry;
pub mod router;
pub mod rule_trait;
pub mod state;
pub mod v2;

pub use data_generator::{
    GeneratorDepthConfig, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
};
pub use diagnostics::{
    BoundaryMemoryDelta, CudaMemorySnapshot, DiagnosticBoundary, DiagnosticEvent,
    DiagnosticEventKind, DiagnosticEventSummary, DiagnosticIdentity, DiagnosticProbeKind,
    DiagnosticProbeRequest, DiagnosticsPolicy, DiagnosticsRecorder, DiagnosticsRuntimeArtifact,
    DiagnosticsRuntimeFailure, DiagnosticsRuntimeFailureKind, ForwardGraphBurden,
    LinearProjectionLayoutMetadata, OutputProjectionDiagnosticContext,
    OutputProjectionDiagnosticEventSummary, OutputProjectionDiagnosticSpec,
    OutputProjectionIdentity, ProbeCadence, ProjectionDiagnosticsSink,
    RuleProjectionDiagnosticContext, RuleProjectionDiagnosticEventSummary,
    RuleProjectionDiagnosticSpec, RuleProjectionDiagnosticsSink, RuleProjectionIdentity,
    RuleProjectionKind, StructuredDiagnosticsOutput, TensorLayoutMetadata, TensorLayoutOrigin,
    TensorLayoutTransform, TrainStepDiagnosticContext,
};
pub use fitness::{RankedSpeciesResult, SpeciesRawMetrics};
pub use hybrid::{
    BaselineRescueAttentionBlock, BaselineRescueAttentionConfig,
    FractalHybridRescuePrevalidationModel, GatheredCandidateRecall, GatheredRetrievalContext,
    GatheredRetrievalContextShape, GatheredRetrievalLayout, GatheredRetrievalProvenance,
    HybridModelShape, HybridRescueForwardOutput, HybridRescuePrevalidationMode,
    HybridRescueStepOutput, PreparedHybridRescueStep, RescueAttentionBlock,
    RescueAttentionDiagnostics, RescueAttentionInput, RescueAttentionOutput, RescueAttentionShape,
    SealedTokenStateStore, PHASE1_LEAF_SIZE, PHASE1_LOCAL_WINDOW_SIZE, PHASE1_REMOTE_TOKEN_BUDGET,
    PHASE1_ROUTED_SPAN_COUNT, PHASE1_TOTAL_TOKEN_BUDGET,
};
pub use hybrid_attention::mini_moe::*;
pub use hybrid_attention::{
    build_attention_only_graph_of_experts_model, build_attention_only_hybrid_attention_model,
    build_attention_only_recurrent_graph_of_experts_model,
    build_attention_only_recurrent_scale_proxy_model, build_attention_only_scale_proxy_model,
    build_primitive_hybrid_attention_model, build_reference_ssm_graph_of_experts_model,
    build_rust_mamba3_reference_hybrid_attention_model, goe_over_attention_only_variant,
    goe_over_attention_only_variant_with_controller, goe_over_reference_ssm_variant,
    goe_over_reference_ssm_variant_with_controller, minimal_recurrent_router_experiment_matrix,
    phase1_hybrid_attention_baseline_matrix, phase1_p20_candidate_variant,
    phase1_p21_candidate_variant, phase1_p22_candidate_variant, phase1_p23_candidate_variant,
    phase1_p2_candidate_variant, phase1_p2_factor_candidate_variant,
    phase1_p2_interface_candidate_variant, primitive_from_p2_factors,
    recurrent_goe_over_attention_only_variant,
    recurrent_goe_over_attention_only_variant_with_router,
    scale_proxy_one_shot_over_attention_only_variant,
    scale_proxy_recurrent_over_attention_only_variant,
    scale_proxy_recurrent_over_attention_only_variant_with_router,
    AttentionOnlyGraphOfExpertsModel, AttentionOnlyHybridAttentionModel,
    AttentionOnlyRecurrentGraphOfExpertsModel, AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    AttentionOnlyRecurrentScaleProxyModel, AttentionOnlyRecurrentScaleProxyVariantSpec,
    AttentionOnlyScaleProxyModel, AttentionOnlyScaleProxyVariantSpec, GraphOfExpertsBackboneKind,
    GraphOfExpertsControllerSpec, GraphOfExpertsRoutingMode, GraphOfExpertsRoutingProbe,
    GraphOfExpertsTopology, GraphOfExpertsVariantSpec, HybridAttentionBaselineMatrix,
    HybridAttentionComparisonContract, HybridAttentionEfficiencyTarget, HybridAttentionLayerRole,
    HybridAttentionModelShape, HybridAttentionVariantKind, HybridAttentionVariantSpec,
    HybridSequenceKernelContract, HybridSequenceScanMode, HybridSequenceStateLayout,
    P2InternalReadoutFactor, P2LatentWidthFactor, PrimitiveHybridAttentionModel,
    PrimitiveHybridNormMode, PrimitiveHybridPrimitive, PrimitiveHybridReadoutMode,
    PrimitiveHybridResidualMode, PrimitiveHybridWrapperSymmetryMode,
    RecurrentGraphOfExpertsRoutingProbe, RecurrentRouterExperimentVariantKind,
    RecurrentRouterExperimentVariantSpec, RecurrentRouterFeedbackMode,
    RecurrentRouterPrimitiveKind, RecurrentRouterSelectionMode, RecurrentRouterSpec,
    ReferenceSsmFamily, ReferenceSsmGraphOfExpertsModel, RustMamba3BaselineConfig,
    RustMamba3DerivedShape, RustMamba3Mixer, RustMamba3MixerBlock,
    RustMamba3ReferenceHybridAttentionModel, RustMamba3RopeFraction, ScaleProxyRoutingProbe,
    VirtualNodeRecurrentRouter, DEFAULT_RECURRENT_ROUTER_ROUND_COUNT,
    DEFAULT_RECURRENT_ROUTER_STATE_WIDTH, DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX,
    GOE_CHANNEL_COUNT, MAX_RECURRENT_ROUTER_ROUND_COUNT, PATH1_PHASE1_LOCAL_WINDOW_SIZE,
    RECURRENT_DREEGMOR_CHANNEL_COUNT, SCALE_PROXY_CHANNEL_COUNT,
};
pub use language_model_head::{LanguageModelHead, LanguageModelHeadConfig};
pub use lifecycle::{
    ArcSourceMode, ArcSourceSpec, ArtifactCompleteness, ArtifactPolicy, BatchingPolicy,
    BridgePackagingSpec, BridgeSplitPolicy, BridgeSubstrateMode, BudgetSpec, BufferReusePolicy,
    CheckpointPolicy, ComparisonAuthority, ComparisonContract, DecisionIntent, EvalBackendPolicy,
    EvalCadencePolicy, ExecutionBackend, ExecutionTarget, ExecutionTargetKind, ExperimentId,
    ExperimentQuestion, ExperimentSpec, ExperimentSpecTemplate, FailureDiagnosticBoundary,
    FailureDiagnosticEvent, FailureSnapshotArtifact, FailureSnapshotArtifactFormat,
    FailureSnapshotArtifactKind, FailureSnapshotAttempt, FailureSnapshotAttemptOutcome,
    FailureSnapshotCaptureTiming, FailureSnapshotContract, FailureSnapshotErrorClass,
    FailureSnapshotPolicy, FailureSnapshotRuntimeState, ForwardExecutionPolicy, LaneIntent,
    LaunchPolicySpec, LearningRateScheduleKind, LearningRateScheduleSpec, ModelArchitectureKind,
    ModelContractSpec, NumericPrecisionKind, OptimizerKind, OptimizerSpec, PhaseTiming,
    PrecisionPolicy, QuantizationPolicy, QuantizedPrecisionKind, ResumePolicy, RunExecutionOutcome,
    RunManifest, RunOutcomeClass, RunPhase, RunQualityOutcome, RuntimeBackendPolicy,
    RuntimeSurfaceSpec, SnapshotCompleteness, SpeciesCompletion, SpeciesRunArtifact,
    SpeciesRunStage, TextCorpusFormat, TextCorpusSourceSpec, TextCorpusSplitSpec,
    TokenizerArtifactSpec, TokenizerBridgeSpec, Tournament, TournamentConfig, TournamentPreset,
    TournamentProgressEvent, TournamentReporter, TournamentRunArtifact, TournamentSequence,
    TrainingInputMode, TrainingInputSpec, VariantSpec, WeightExportArtifact, WeightExportAttempt,
    WeightExportAttemptOutcome, WeightExportContract, WeightExportFormat, WeightExportPhase,
    WeightExportPolicy, WeightExportRuntimeState,
};
pub use model::FractalModel;
pub use projection::{
    ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig,
    StructuredProjectionRecord,
};
#[cfg(feature = "cuda")]
pub use registry::cuda_device;
pub use registry::{
    initialize_metal_runtime, is_valid_primitive_variant_name, load_weight_export_artifact,
    load_weight_export_metadata, read_cuda_memory_snapshot_for_device, read_weight_export_metadata,
    resolve_precision_profile, run_species_with_batches, CandleBf16Backend, CandleBf16TrainBackend,
    CandleF32Backend, CandleF32TrainBackend, ComputeBackend, CpuBackend, CpuTrainBackend,
    ExecutionMode, MetalBackend, MetalBf16Backend, MetalBf16TrainBackend, MetalF32Backend,
    MetalF32TrainBackend, MetalTrainBackend, PrimitiveVariantName,
    ResolvedExecutablePrecisionProfile, SpeciesDefinition, SpeciesId, TrainingBatchSet,
};
pub use router::EarlyExitRouter;
pub use state::{FractalState, StateLayout};
pub use v2::{
    summarize_root_readout_sequence, BaselineExactLeafRead, BaselineExactLeafReadConfig,
    BaselineFractalRouterHead, BaselineFractalRouterHeadConfig, BaselineLeafSummarizer,
    BaselineLeafSummarizerConfig, BaselineLocalTrunk, BaselineLocalTrunkConfig,
    BaselineTreeMergeCell, BaselineTreeMergeCellConfig, BatchHeadRoute, BatchRouteStep,
    BatchTimelineMode, ExactLeafRead, ExactLeafReadDiagnostics, ExactLeafReadOutput,
    ExactLeafReadShape, ExactReadHistogramBin, FractalRouteOutput, FractalRouterHead,
    FractalRouterHeadShape, FractalRoutingDiagnostics, FractalV2Components,
    FractalV2LocalBaselineModel, FractalV2LocalBaselineOutput, FractalV2LocalBaselineShape,
    FractalV2MemoryMode, FractalV2Model, FractalV2ModelShape, FractalV2ProjectionBreakdown,
    FractalV2RetrievalStepOutput, FractalV2RetrievalTrace, FractalV2State, FractalV2StateLayout,
    FractalV2StateRecord, FractalV2StateShape, HeadRouteTrace, LeafSummarizer,
    LeafSummarizerOutput, LeafSummarizerShape, LeafSummaryStore, LeafSummaryStoreRecord,
    LeafSummaryStoreShape, LeafTokenCache, LeafTokenCacheRecord, LeafTokenCacheShape,
    LiveLeafState, LiveLeafStateRecord, LiveLeafStateShape, LocalTrunk, LocalTrunkDiagnostics,
    LocalTrunkShape, LocalTrunkStepOutput, MergeCheckpointPolicy, MultiRootState,
    MultiRootStateRecord, MultiRootStateShape, ReadFusion, ReadFusionShape, RetrievalPolicy,
    RootActivationStats, RoutingHistogramBin, SealedLeafMaterialization, TokenSpan, TreeLevelStore,
    TreeLevelStoreRecord, TreeLevelStoreShape, TreeMergeCell, TreeMergeCellShape, TreeMergeOutput,
    TreeNodeAddress, TreeNodeBatch, TreeSummaryDiagnostics, TreeSummaryState,
    TreeSummaryStateRecord, TreeSummaryStateShape,
};

#[cfg(test)]
mod tests;
