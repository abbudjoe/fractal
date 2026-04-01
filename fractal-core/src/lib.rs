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
    ArtifactPolicy, BatchingPolicy, BudgetSpec, BufferReusePolicy, ComparisonAuthority,
    ComparisonContract, DecisionIntent, EvalBackendPolicy, ExecutionBackend, ExecutionTarget,
    ExecutionTargetKind, ExperimentId, ExperimentQuestion, ExperimentSpec, ExperimentSpecTemplate,
    ForwardExecutionPolicy, LaneIntent, LearningRateScheduleKind, LearningRateScheduleSpec,
    OptimizerKind, OptimizerSpec, PhaseTiming, RunExecutionOutcome, RunManifest, RunOutcomeClass,
    RunPhase, RunQualityOutcome, RuntimeBackendPolicy, RuntimeSurfaceSpec, SpeciesCompletion,
    SpeciesRunArtifact, SpeciesRunStage, Tournament, TournamentConfig, TournamentPreset,
    TournamentProgressEvent, TournamentReporter, TournamentRunArtifact, TournamentSequence,
    VariantSpec,
};
pub use model::FractalModel;
pub use registry::{
    is_valid_primitive_variant_name, ComputeBackend, CpuBackend, CpuTrainBackend, ExecutionMode,
    MetalBackend, MetalTrainBackend, PrimitiveVariantName, SpeciesDefinition, SpeciesId,
};
pub use router::EarlyExitRouter;
pub use state::{FractalState, StateLayout};

#[cfg(test)]
mod tests;
