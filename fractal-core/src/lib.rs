pub mod data_generator;
pub mod diagnostics;
pub mod error;
pub mod fitness;
pub mod hybrid;
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
    BaselineRescueAttentionBlock, BaselineRescueAttentionConfig, GatheredCandidateRecall,
    GatheredRetrievalContext, GatheredRetrievalContextShape, GatheredRetrievalLayout,
    GatheredRetrievalProvenance, FractalHybridRescuePrevalidationModel, HybridModelShape,
    HybridRescueForwardOutput, HybridRescuePrevalidationMode, HybridRescueStepOutput,
    RescueAttentionBlock, RescueAttentionDiagnostics, RescueAttentionInput,
    RescueAttentionOutput, RescueAttentionShape, SealedTokenStateStore, PHASE1_LEAF_SIZE,
    PHASE1_LOCAL_WINDOW_SIZE, PHASE1_REMOTE_TOKEN_BUDGET, PHASE1_ROUTED_SPAN_COUNT,
    PHASE1_TOTAL_TOKEN_BUDGET,
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
pub use registry::{
    is_valid_primitive_variant_name, load_weight_export_artifact, load_weight_export_metadata,
    read_weight_export_metadata, resolve_precision_profile, run_species_with_batches,
    CandleBf16Backend, CandleBf16TrainBackend, CandleF32Backend, CandleF32TrainBackend,
    ComputeBackend, CpuBackend, CpuTrainBackend, ExecutionMode, MetalBackend, MetalBf16Backend,
    MetalBf16TrainBackend, MetalF32Backend, MetalF32TrainBackend, MetalTrainBackend,
    PrimitiveVariantName, ResolvedExecutablePrecisionProfile, SpeciesDefinition, SpeciesId,
    TrainingBatchSet,
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
