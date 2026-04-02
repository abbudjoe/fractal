pub mod data_generator;
pub mod diagnostics;
pub mod error;
pub mod fitness;
pub mod lifecycle;
pub mod model;
pub mod projection;
pub mod primitives;
pub mod registry;
pub mod router;
pub mod rule_trait;
pub mod state;

pub use data_generator::{
    GeneratorDepthConfig, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
};
pub use diagnostics::{
    BoundaryMemoryDelta, CudaMemorySnapshot, DiagnosticBoundary, DiagnosticEvent,
    DiagnosticEventKind, DiagnosticEventSummary, DiagnosticIdentity, DiagnosticProbeKind,
    DiagnosticProbeRequest, DiagnosticsPolicy, DiagnosticsRecorder, DiagnosticsRuntimeArtifact,
    DiagnosticsRuntimeFailure, DiagnosticsRuntimeFailureKind, ForwardGraphBurden, ProbeCadence,
    LinearProjectionLayoutMetadata, OutputProjectionDiagnosticContext,
    OutputProjectionDiagnosticEventSummary, OutputProjectionDiagnosticSpec,
    OutputProjectionIdentity, ProjectionDiagnosticsSink, RuleProjectionDiagnosticContext,
    RuleProjectionDiagnosticEventSummary, RuleProjectionDiagnosticSpec,
    RuleProjectionDiagnosticsSink, RuleProjectionIdentity, RuleProjectionKind,
    StructuredDiagnosticsOutput, TensorLayoutMetadata, TensorLayoutOrigin,
    TensorLayoutTransform, TrainStepDiagnosticContext,
};
pub use fitness::{RankedSpeciesResult, SpeciesRawMetrics};
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

#[cfg(test)]
mod tests;
