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

pub use data_generator::{SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN};
pub use fitness::{RankedSpeciesResult, SpeciesRawMetrics};
pub use lifecycle::{Tournament, TournamentConfig, TournamentPreset, TournamentSequence};
pub use model::FractalModel;
pub use registry::{
    ComputeBackend, CpuBackend, CpuTrainBackend, ExecutionMode, MetalBackend, MetalTrainBackend,
    SpeciesDefinition, SpeciesId,
};
pub use router::EarlyExitRouter;
pub use state::{FractalState, StateLayout};

#[cfg(test)]
mod tests;
