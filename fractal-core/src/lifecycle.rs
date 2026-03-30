use std::{sync::Arc, thread};

use crate::{
    data_generator::{
        GeneratorConfig, SimpleHierarchicalGenerator, MIN_SEQUENCE_LEN, MIN_VOCAB_SIZE,
    },
    error::FractalError,
    fitness::{aggregate_results, RankedSpeciesResult, SpeciesRawMetrics},
    registry::{
        species_registry, ComputeBackend, ExecutionMode, SpeciesDefinition, SpeciesRunContext,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentPreset {
    Default,
    FastTest,
    ResearchMedium,
    PressureTest,
}

impl TournamentPreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::FastTest => "fast-test",
            Self::ResearchMedium => "research-medium",
            Self::PressureTest => "pressure-test",
        }
    }

    pub fn config(self) -> TournamentConfig {
        match self {
            Self::Default => TournamentConfig::default(),
            Self::FastTest => TournamentConfig::fast_test(),
            Self::ResearchMedium => TournamentConfig::research_medium(),
            Self::PressureTest => TournamentConfig::pressure_test(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentSequence {
    FirstRun,
}

const FIRST_RUN_SEQUENCE: [TournamentPreset; 3] = [
    TournamentPreset::FastTest,
    TournamentPreset::ResearchMedium,
    TournamentPreset::PressureTest,
];

impl TournamentSequence {
    pub fn name(self) -> &'static str {
        match self {
            Self::FirstRun => "first-run",
        }
    }

    pub fn stages(self) -> &'static [TournamentPreset] {
        match self {
            Self::FirstRun => &FIRST_RUN_SEQUENCE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TournamentConfig {
    pub dim: usize,
    pub levels: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub max_recursion_depth: usize,
    pub router_threshold: f32,
    pub batch_size: usize,
    pub train_steps_per_species: usize,
    pub eval_batches_per_family: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub execution_backend: ComputeBackend,
    pub execution_mode: ExecutionMode,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 16,
            max_recursion_depth: 1,
            router_threshold: 1.1,
            batch_size: 1,
            train_steps_per_species: 1,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
        }
    }
}

impl TournamentConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.dim == 0 {
            return Err(FractalError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.levels < 2 {
            return Err(FractalError::InvalidConfig(
                "levels must be at least 2 for hierarchical species".into(),
            ));
        }
        if self.vocab_size < MIN_VOCAB_SIZE {
            return Err(FractalError::InvalidConfig(format!(
                "vocab_size must be at least {MIN_VOC_SIZE}",
                MIN_VOC_SIZE = MIN_VOCAB_SIZE
            )));
        }
        if self.max_seq_len < MIN_SEQUENCE_LEN {
            return Err(FractalError::InvalidConfig(
                format!(
                    "max_seq_len must be at least {MIN_SEQUENCE_LEN} to encode the smallest recursive task"
                ),
            ));
        }
        if self.max_recursion_depth == 0 {
            return Err(FractalError::InvalidConfig(
                "max_recursion_depth must be greater than zero".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "batch_size must be greater than zero".into(),
            ));
        }
        if self.eval_batches_per_family == 0 {
            return Err(FractalError::InvalidConfig(
                "eval_batches_per_family must be greater than zero".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(FractalError::InvalidConfig(
                "learning_rate must be greater than zero".into(),
            ));
        }
        if !self.execution_backend.is_supported_on_current_platform() {
            return Err(FractalError::InvalidConfig(
                "Metal execution is only supported on macOS".into(),
            ));
        }

        Ok(())
    }

    pub fn pressure_test() -> Self {
        Self {
            dim: 128,
            levels: 4,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 20,
            router_threshold: 0.90,
            batch_size: 16,
            train_steps_per_species: 50,
            eval_batches_per_family: 8,
            learning_rate: 1e-3,
            seed: 42,
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
        }
    }

    pub fn research_medium() -> Self {
        Self {
            dim: 16,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 32,
            max_recursion_depth: 4,
            router_threshold: 0.95,
            batch_size: 2,
            train_steps_per_species: 5,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
        }
    }

    pub fn fast_test() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 16,
            max_recursion_depth: 1,
            router_threshold: 1.1,
            batch_size: 1,
            train_steps_per_species: 0,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
            execution_backend: ComputeBackend::CpuCandle,
            execution_mode: ExecutionMode::Sequential,
        }
    }

    pub fn with_execution_mode(mut self, execution_mode: ExecutionMode) -> Self {
        self.execution_mode = execution_mode;
        self
    }

    pub fn with_execution_backend(mut self, execution_backend: ComputeBackend) -> Self {
        self.execution_backend = execution_backend;
        self
    }
}

#[derive(Clone, Debug)]
pub struct Tournament {
    pub config: TournamentConfig,
    generator: Arc<SimpleHierarchicalGenerator>,
}

impl Tournament {
    pub fn new(config: TournamentConfig) -> Result<Self, FractalError> {
        config.validate()?;
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family: 96,
            eval_examples_per_family: 32,
            seed: config.seed,
        })?;

        Ok(Self {
            config,
            generator: Arc::new(generator),
        })
    }

    pub fn run_generation(&self) -> Result<Vec<RankedSpeciesResult>, FractalError> {
        let species = species_registry();
        let metrics = match self.config.execution_mode {
            ExecutionMode::Sequential => self.run_sequential(species)?,
            ExecutionMode::Parallel => self.run_parallel(species)?,
        };

        Ok(aggregate_results(metrics))
    }

    fn run_sequential(
        &self,
        species: &[SpeciesDefinition],
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let mut metrics = Vec::with_capacity(species.len());
        for (index, definition) in species.iter().enumerate() {
            metrics.push(definition.run(self.run_context(index), &self.config.execution_backend)?);
        }

        Ok(metrics)
    }

    fn run_parallel(
        &self,
        species: &[SpeciesDefinition],
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let mut handles = Vec::with_capacity(species.len());
        for (index, definition) in species.iter().copied().enumerate() {
            let context = self.run_context(index);
            let backend = self.config.execution_backend.clone();
            handles.push(thread::spawn(move || definition.run(context, &backend)));
        }

        let mut metrics = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| FractalError::InvalidState("species worker panicked".into()))??;
            metrics.push(result);
        }

        Ok(metrics)
    }

    fn run_context(&self, index: usize) -> SpeciesRunContext {
        SpeciesRunContext {
            index,
            config: self.config.clone(),
            generator: Arc::clone(&self.generator),
        }
    }
}
