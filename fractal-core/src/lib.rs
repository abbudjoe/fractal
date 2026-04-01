pub mod data_generator;
pub mod error;
pub mod fitness;
pub mod lifecycle;
pub mod model;
pub mod primitives;
pub mod registry;
pub mod router;
pub mod rule_trait;
pub mod state;

pub use data_generator::{
    GeneratorDepthConfig, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
};
pub use fitness::{RankedSpeciesResult, SpeciesRawMetrics};
pub use lifecycle::{
    ArcSourceMode, ArcSourceSpec, ArtifactPolicy, BatchingPolicy, BudgetSpec, BufferReusePolicy,
    CheckpointPolicy, ComparisonAuthority, ComparisonContract, DecisionIntent, EvalBackendPolicy,
    EvalCadencePolicy, ExecutionBackend, ExecutionTarget, ExecutionTargetKind, ExperimentId,
    ExperimentQuestion, ExperimentSpec, ExperimentSpecTemplate, ForwardExecutionPolicy, LaneIntent,
    LaunchPolicySpec, LearningRateScheduleKind, LearningRateScheduleSpec, NumericPrecisionKind,
    OptimizerKind, OptimizerSpec, PhaseTiming, PrecisionPolicy, ResumePolicy, RunExecutionOutcome,
    RunManifest, RunOutcomeClass, RunPhase, RunQualityOutcome, RuntimeBackendPolicy,
    RuntimeSurfaceSpec, SpeciesCompletion, SpeciesRunArtifact, SpeciesRunStage,
    TokenizerArtifactSpec, TokenizerBridgeSpec, Tournament, TournamentConfig, TournamentPreset,
    TournamentProgressEvent, TournamentReporter, TournamentRunArtifact, TournamentSequence,
    TrainingInputMode, TrainingInputSpec, VariantSpec,
};
pub use model::FractalModel;
pub use registry::{
    is_valid_primitive_variant_name, run_species_with_batches, ComputeBackend, CpuBackend,
    CpuTrainBackend, ExecutionMode, MetalBackend, MetalTrainBackend, PrimitiveVariantName,
    SpeciesDefinition, SpeciesId, TrainingBatchSet,
};
pub use router::EarlyExitRouter;
pub use state::{FractalState, StateLayout};

#[cfg(test)]
mod tests;
