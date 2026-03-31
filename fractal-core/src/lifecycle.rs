use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use crate::{
    data_generator::{
        GeneratorConfig, GeneratorDepthConfig, SimpleHierarchicalGenerator, MIN_SEQUENCE_LEN,
        MIN_VOCAB_SIZE,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    registry::{
        build_failure_artifact, build_success_artifact, classify_quality_outcome, phase_timing,
        take_last_species_run_artifact, ComputeBackend, ExecutionMode, PrimitiveVariantName,
        SpeciesDefinition, SpeciesId, SpeciesRunContext,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentPreset {
    Default,
    FastTest,
    ResearchMedium,
    ChallengerLane,
    MinimalBaseline,
    MinimalStressLane,
    MinimalProvingGround,
    ProvingGroundBaseline,
    BullpenPolish,
    LighterIntermediateStress,
    IntermediateStress,
    FullMediumStress,
    MediumStress,
    PressureTest,
    CandidateStress,
    GenerationFour,
}

impl TournamentPreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::FastTest => "fast-test",
            Self::ResearchMedium => "research-medium",
            Self::ChallengerLane => "challenger-lane",
            Self::MinimalBaseline => "minimal-baseline",
            Self::MinimalStressLane => "minimal-stress-lane",
            Self::MinimalProvingGround => "minimal-proving-ground",
            Self::ProvingGroundBaseline => "proving-ground-baseline",
            Self::BullpenPolish => "bullpen-polish",
            Self::LighterIntermediateStress => "lighter-intermediate-stress",
            Self::IntermediateStress => "intermediate-stress",
            Self::FullMediumStress => "full-medium-stress",
            Self::MediumStress => "medium-stress",
            Self::PressureTest => "pressure-test",
            Self::CandidateStress => "candidate-stress",
            Self::GenerationFour => "generation-four",
        }
    }

    pub fn config(self) -> TournamentConfig {
        match self {
            Self::Default => TournamentConfig::default(),
            Self::FastTest => TournamentConfig::fast_test(),
            Self::ResearchMedium => TournamentConfig::research_medium(),
            Self::ChallengerLane => TournamentConfig::challenger_lane(),
            Self::MinimalBaseline => TournamentConfig::minimal_baseline(),
            Self::MinimalStressLane => TournamentConfig::minimal_stress_lane(),
            Self::MinimalProvingGround => TournamentConfig::minimal_proving_ground(),
            Self::ProvingGroundBaseline => TournamentConfig::proving_ground_baseline(),
            Self::BullpenPolish => TournamentConfig::bullpen_polish(),
            Self::LighterIntermediateStress => TournamentConfig::lighter_intermediate_stress(),
            Self::IntermediateStress => TournamentConfig::intermediate_stress(),
            Self::FullMediumStress => TournamentConfig::full_medium_stress(),
            Self::MediumStress => TournamentConfig::medium_stress(),
            Self::PressureTest => TournamentConfig::pressure_test(),
            Self::CandidateStress => TournamentConfig::candidate_stress(),
            Self::GenerationFour => TournamentConfig::generation_four(),
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
pub struct SpeciesPresetOverride {
    pub species: SpeciesId,
    pub train_batch_size: Option<usize>,
    pub eval_batch_size: Option<usize>,
    pub train_steps_per_species: Option<usize>,
    pub max_recursion_depth: Option<usize>,
    pub stability_depth: Option<usize>,
}

impl SpeciesPresetOverride {
    pub const fn for_species(species: SpeciesId) -> Self {
        Self {
            species,
            train_batch_size: None,
            eval_batch_size: None,
            train_steps_per_species: None,
            max_recursion_depth: None,
            stability_depth: None,
        }
    }

    fn apply(&self, config: &mut TournamentConfig) {
        if let Some(train_batch_size) = self.train_batch_size {
            config.train_batch_size = train_batch_size;
        }
        if let Some(eval_batch_size) = self.eval_batch_size {
            config.eval_batch_size = eval_batch_size;
        }
        if let Some(train_steps_per_species) = self.train_steps_per_species {
            config.train_steps_per_species = train_steps_per_species;
        }
        if let Some(max_recursion_depth) = self.max_recursion_depth {
            config.max_recursion_depth = max_recursion_depth;
        }
        if let Some(stability_depth) = self.stability_depth {
            config.stability_depth = stability_depth;
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
    pub stability_depth: usize,
    pub router_threshold: f32,
    pub train_batch_size: usize,
    pub eval_batch_size: usize,
    pub train_steps_per_species: usize,
    pub eval_batches_per_family: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub run_timeout: Option<Duration>,
    pub generator_depth_config: GeneratorDepthConfig,
    pub execution_backend: ComputeBackend,
    pub execution_mode: ExecutionMode,
    pub parallelism: usize,
    pub species_overrides: Vec<SpeciesPresetOverride>,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 16,
            max_recursion_depth: 1,
            stability_depth: 1,
            router_threshold: 1.1,
            train_batch_size: 1,
            eval_batch_size: 1,
            train_steps_per_species: 1,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
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
        if self.stability_depth == 0 {
            return Err(FractalError::InvalidConfig(
                "stability_depth must be greater than zero".into(),
            ));
        }
        if self.train_batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "train_batch_size must be greater than zero".into(),
            ));
        }
        if self.eval_batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "eval_batch_size must be greater than zero".into(),
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
        if let Some(run_timeout) = self.run_timeout {
            if run_timeout.is_zero() {
                return Err(FractalError::InvalidConfig(
                    "run_timeout must be greater than zero when configured".into(),
                ));
            }
        }
        if !self.execution_backend.is_supported_on_current_platform() {
            return Err(FractalError::InvalidConfig(
                "selected execution backend is not supported on this platform".into(),
            ));
        }
        if self.parallelism == 0 {
            return Err(FractalError::InvalidConfig(
                "parallelism must be greater than zero".into(),
            ));
        }
        let mut override_species = HashSet::new();
        for override_config in &self.species_overrides {
            if !override_species.insert(override_config.species) {
                return Err(FractalError::InvalidConfig(format!(
                    "duplicate species override for {}",
                    override_config.species
                )));
            }
            if override_config.train_batch_size == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override train_batch_size must be greater than zero".into(),
                ));
            }
            if override_config.eval_batch_size == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override eval_batch_size must be greater than zero".into(),
                ));
            }
            if override_config.train_steps_per_species == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override train_steps_per_species must be greater than zero".into(),
                ));
            }
            if override_config.max_recursion_depth == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override max_recursion_depth must be greater than zero".into(),
                ));
            }
            if override_config.stability_depth == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override stability_depth must be greater than zero".into(),
                ));
            }
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
            stability_depth: 20,
            router_threshold: 0.90,
            train_batch_size: 16,
            eval_batch_size: 8,
            train_steps_per_species: 50,
            eval_batches_per_family: 8,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn challenger_lane() -> Self {
        Self {
            dim: 96,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 64,
            max_recursion_depth: 6,
            stability_depth: 6,
            router_threshold: 0.92,
            train_batch_size: 8,
            eval_batch_size: 4,
            train_steps_per_species: 12,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn bullpen_polish() -> Self {
        Self {
            dim: 128,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 96,
            max_recursion_depth: 8,
            stability_depth: 8,
            router_threshold: 0.92,
            train_batch_size: 16,
            eval_batch_size: 8,
            train_steps_per_species: 24,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: vec![SpeciesPresetOverride {
                // Temporary Generation 4 polish override until IFS training cost is better bounded.
                train_batch_size: Some(8),
                eval_batch_size: Some(4),
                train_steps_per_species: Some(16),
                ..SpeciesPresetOverride::for_species(SpeciesId::Ifs)
            }],
        }
    }

    pub fn minimal_proving_ground() -> Self {
        Self {
            dim: 128,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 96,
            max_recursion_depth: 8,
            stability_depth: 8,
            router_threshold: 0.92,
            train_batch_size: 8,
            eval_batch_size: 8,
            train_steps_per_species: 30,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn proving_ground_baseline() -> Self {
        Self {
            dim: 128,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 64,
            max_recursion_depth: 6,
            stability_depth: 6,
            router_threshold: 0.92,
            train_batch_size: 16,
            eval_batch_size: 16,
            train_steps_per_species: 5,
            eval_batches_per_family: 2,
            learning_rate: 5e-4,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn minimal_baseline() -> Self {
        Self::minimal_proving_ground()
    }

    pub fn minimal_stress_lane() -> Self {
        Self::minimal_proving_ground()
    }

    pub fn medium_stress() -> Self {
        Self {
            dim: 192,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 12,
            stability_depth: 12,
            router_threshold: 0.92,
            train_batch_size: 8,
            eval_batch_size: 4,
            train_steps_per_species: 80,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn full_medium_stress() -> Self {
        Self::medium_stress()
    }

    pub fn intermediate_stress() -> Self {
        Self {
            dim: 160,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 96,
            max_recursion_depth: 10,
            stability_depth: 10,
            router_threshold: 0.92,
            train_batch_size: 8,
            eval_batch_size: 4,
            train_steps_per_species: 48,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn lighter_intermediate_stress() -> Self {
        Self {
            dim: 160,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 96,
            max_recursion_depth: 10,
            stability_depth: 10,
            router_threshold: 0.92,
            train_batch_size: 4,
            eval_batch_size: 2,
            train_steps_per_species: 48,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn generation_four() -> Self {
        #[cfg(feature = "cuda")]
        {
            let mut config = Self::pressure_test();
            config.execution_backend = ComputeBackend::cuda_default();
            config
        }
        #[cfg(not(feature = "cuda"))]
        {
            Self::pressure_test()
        }
    }

    pub fn research_medium() -> Self {
        Self {
            dim: 16,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 32,
            max_recursion_depth: 4,
            stability_depth: 4,
            router_threshold: 0.95,
            train_batch_size: 2,
            eval_batch_size: 2,
            train_steps_per_species: 5,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn fast_test() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 16,
            max_recursion_depth: 1,
            stability_depth: 1,
            router_threshold: 1.1,
            train_batch_size: 1,
            eval_batch_size: 1,
            train_steps_per_species: 0,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::CpuCandle,
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
        }
    }

    pub fn candidate_stress() -> Self {
        Self {
            dim: 192,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 16,
            stability_depth: 20,
            router_threshold: 0.92,
            train_batch_size: 8,
            eval_batch_size: 4,
            train_steps_per_species: 120,
            eval_batches_per_family: 4,
            learning_rate: 1e-3,
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::stress_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
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

    pub fn with_parallelism(mut self, parallelism: usize) -> Self {
        self.parallelism = parallelism;
        self
    }

    pub fn effective_for_species(&self, species: SpeciesId) -> Self {
        let mut config = self.clone();
        for override_config in self
            .species_overrides
            .iter()
            .filter(|override_config| override_config.species == species)
        {
            override_config.apply(&mut config);
        }
        config.species_overrides.clear();
        config
    }

    fn max_train_batch_size(&self) -> usize {
        self.species_overrides
            .iter()
            .filter_map(|override_config| override_config.train_batch_size)
            .fold(self.train_batch_size, usize::max)
    }

    fn max_eval_batch_size(&self) -> usize {
        self.species_overrides
            .iter()
            .filter_map(|override_config| override_config.eval_batch_size)
            .fold(self.eval_batch_size, usize::max)
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesRunStage {
    pub species: SpeciesId,
    pub ordinal: usize,
    pub total: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunPhase {
    Train,
    Stability,
    Perplexity,
    ArcSpeed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunExecutionOutcome {
    Success,
    TrainTimeout,
    EvalConstrained,
    InfraFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunQualityOutcome {
    Clean,
    NumericFailure,
    LowSignal,
    RuntimeCost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcomeClass {
    Success,
    TrainTimeout,
    EvalConstrained,
    NumericFailure,
    LowSignal,
    RuntimeCost,
    InfraFailure,
}

impl RunOutcomeClass {
    pub fn from_components(
        execution_outcome: RunExecutionOutcome,
        quality_outcome: RunQualityOutcome,
    ) -> Self {
        match execution_outcome {
            RunExecutionOutcome::Success => match quality_outcome {
                RunQualityOutcome::Clean => Self::Success,
                RunQualityOutcome::NumericFailure => Self::NumericFailure,
                RunQualityOutcome::LowSignal => Self::LowSignal,
                RunQualityOutcome::RuntimeCost => Self::RuntimeCost,
            },
            RunExecutionOutcome::TrainTimeout => Self::TrainTimeout,
            RunExecutionOutcome::EvalConstrained => Self::EvalConstrained,
            RunExecutionOutcome::InfraFailure => Self::InfraFailure,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PhaseTiming {
    pub phase: RunPhase,
    pub elapsed: Duration,
    pub completed: usize,
    pub total: usize,
}

#[derive(Clone, Debug)]
pub struct RunManifest {
    pub variant_name: PrimitiveVariantName,
    pub timeout_budget: Option<Duration>,
    pub config: TournamentConfig,
}

#[derive(Clone, Debug)]
pub struct SpeciesRunArtifact {
    pub stage: SpeciesRunStage,
    pub manifest: RunManifest,
    pub phase_timings: Vec<PhaseTiming>,
    pub execution_outcome: RunExecutionOutcome,
    pub quality_outcome: RunQualityOutcome,
    pub error: Option<String>,
    pub metrics: Option<SpeciesRawMetrics>,
}

impl SpeciesRunArtifact {
    pub fn with_stage(mut self, stage: SpeciesRunStage) -> Self {
        self.stage = stage;
        self
    }

    pub fn outcome_class(&self) -> RunOutcomeClass {
        RunOutcomeClass::from_components(self.execution_outcome, self.quality_outcome)
    }

    pub fn is_success(&self) -> bool {
        self.outcome_class() == RunOutcomeClass::Success
    }
}

#[derive(Clone, Debug)]
pub struct TournamentRunArtifact {
    pub config: TournamentConfig,
    pub species: Vec<SpeciesRunArtifact>,
}

impl TournamentRunArtifact {
    pub fn outcome_class(&self) -> RunOutcomeClass {
        self.species
            .iter()
            .map(SpeciesRunArtifact::outcome_class)
            .find(|outcome| *outcome != RunOutcomeClass::Success)
            .unwrap_or(RunOutcomeClass::Success)
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesCompletion {
    pub stage: SpeciesRunStage,
    pub elapsed: Duration,
    pub metrics: SpeciesRawMetrics,
}

#[derive(Clone, Debug)]
pub enum TournamentProgressEvent {
    SpeciesStarted(SpeciesRunStage),
    SpeciesCompleted(SpeciesCompletion),
}

#[derive(Debug)]
struct SpeciesWorkerMessage {
    index: usize,
    elapsed: Duration,
    result: Result<SpeciesRawMetrics, FractalError>,
    artifact: SpeciesRunArtifact,
}

pub trait TournamentReporter: Send + Sync {
    fn on_event(&self, event: TournamentProgressEvent);
}

impl<F> TournamentReporter for F
where
    F: Fn(TournamentProgressEvent) + Send + Sync,
{
    fn on_event(&self, event: TournamentProgressEvent) {
        self(event);
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
        let train_examples_per_family = (config.max_train_batch_size() * 8).max(96);
        let eval_examples_per_family =
            (config.max_eval_batch_size() * config.eval_batches_per_family).max(32);
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family,
            eval_examples_per_family,
            seed: config.seed,
            depth_config: config.generator_depth_config,
        })?;

        Ok(Self {
            config,
            generator: Arc::new(generator),
        })
    }

    pub fn run_generation(
        &self,
        species: &[SpeciesDefinition],
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let artifact = self.run_generation_artifacts(species, None)?;
        if artifact.species.iter().any(|record| {
            matches!(
                record.execution_outcome,
                RunExecutionOutcome::InfraFailure
                    | RunExecutionOutcome::TrainTimeout
                    | RunExecutionOutcome::EvalConstrained
            )
        }) {
            let failure = first_execution_failure(&artifact.species).ok_or_else(|| {
                FractalError::InvalidState(
                    "tournament run reported execution failure without a failure artifact".into(),
                )
            })?;
            return Err(FractalError::InvalidState(format!(
                "species {} failed with {:?}: {}",
                failure.stage.species,
                failure.outcome_class(),
                failure.error.as_deref().unwrap_or("unknown error")
            )));
        }

        Ok(artifact
            .species
            .into_iter()
            .filter_map(|record| record.metrics)
            .collect())
    }

    pub fn run_generation_with_reporter(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let artifact = self.run_generation_artifacts(species, reporter)?;
        if artifact.species.iter().any(|record| {
            matches!(
                record.execution_outcome,
                RunExecutionOutcome::InfraFailure
                    | RunExecutionOutcome::TrainTimeout
                    | RunExecutionOutcome::EvalConstrained
            )
        }) {
            let failure = first_execution_failure(&artifact.species).ok_or_else(|| {
                FractalError::InvalidState(
                    "tournament run reported execution failure without a failure artifact".into(),
                )
            })?;
            return Err(FractalError::InvalidState(format!(
                "species {} failed with {:?}: {}",
                failure.stage.species,
                failure.outcome_class(),
                failure.error.as_deref().unwrap_or("unknown error")
            )));
        }

        Ok(artifact
            .species
            .into_iter()
            .filter_map(|record| record.metrics)
            .collect())
    }

    pub fn run_generation_artifacts(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<TournamentRunArtifact, FractalError> {
        Self::validate_species_definitions(species)?;
        let species_artifacts = match self.config.execution_mode {
            ExecutionMode::Sequential => self.run_sequential(species, reporter),
            ExecutionMode::Parallel => self.run_parallel(species, reporter),
        }?;

        Ok(TournamentRunArtifact {
            config: self.config.clone(),
            species: species_artifacts,
        })
    }

    fn run_sequential(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRunArtifact>, FractalError> {
        let mut artifacts = Vec::with_capacity(species.len());
        for (index, definition) in species.iter().enumerate() {
            let stage = Self::run_stage(definition.id, index, species.len());
            Self::emit_event(
                reporter.as_ref(),
                TournamentProgressEvent::SpeciesStarted(stage.clone()),
            );
            let started = Instant::now();
            let context = self.run_context(index, definition);
            let result = definition.run(context.clone(), &self.config.execution_backend);
            let artifact = Self::capture_species_artifact(
                definition,
                stage.clone(),
                &context,
                started,
                &result,
            );
            if let Ok(metrics) = &result {
                Self::emit_event(
                    reporter.as_ref(),
                    TournamentProgressEvent::SpeciesCompleted(SpeciesCompletion {
                        stage: stage.clone(),
                        elapsed: started.elapsed(),
                        metrics: metrics.clone(),
                    }),
                );
            }
            artifacts.push(artifact);
            if result.is_err() {
                break;
            }
        }

        Ok(artifacts)
    }

    fn run_parallel(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRunArtifact>, FractalError> {
        let total = species.len();
        let concurrency = self.config.parallelism.min(total).max(1);
        let (tx, rx) = mpsc::channel();
        let mut launched = 0usize;
        let mut completed = 0usize;
        let mut failure_encountered = false;
        let mut artifacts = vec![None; total];

        while launched < concurrency {
            self.spawn_species_worker(species[launched], launched, total, reporter.as_ref(), &tx);
            launched += 1;
        }

        while completed < total && (!failure_encountered || completed < launched) {
            let message = rx.recv().map_err(|_| {
                FractalError::InvalidState("species worker channel closed unexpectedly".into())
            })?;
            let stage = Self::run_stage(species[message.index].id, message.index, total);
            match message.result {
                Ok(result) => {
                    Self::emit_event(
                        reporter.as_ref(),
                        TournamentProgressEvent::SpeciesCompleted(SpeciesCompletion {
                            stage,
                            elapsed: message.elapsed,
                            metrics: result.clone(),
                        }),
                    );
                    artifacts[message.index] = Some(message.artifact);
                    completed += 1;
                    if launched < total {
                        self.spawn_species_worker(
                            species[launched],
                            launched,
                            total,
                            reporter.as_ref(),
                            &tx,
                        );
                        launched += 1;
                    }
                }
                Err(_) => {
                    failure_encountered = true;
                    artifacts[message.index] = Some(message.artifact);
                    completed += 1;
                }
            }
        }

        Ok(artifacts.into_iter().flatten().collect())
    }

    fn run_context(&self, index: usize, definition: &SpeciesDefinition) -> SpeciesRunContext {
        SpeciesRunContext {
            index,
            config: self.config.effective_for_species(definition.id),
            generator: Arc::clone(&self.generator),
            variant_name: definition.variant_name,
        }
    }

    fn capture_species_artifact(
        _definition: &SpeciesDefinition,
        stage: SpeciesRunStage,
        context: &SpeciesRunContext,
        started: Instant,
        result: &Result<SpeciesRawMetrics, FractalError>,
    ) -> SpeciesRunArtifact {
        let mut artifact = take_last_species_run_artifact().unwrap_or_else(|| {
            let manifest = RunManifest {
                variant_name: context.variant_name,
                timeout_budget: context.config.run_timeout,
                config: context.config.clone(),
            };
            match result {
                Ok(metrics) => build_success_artifact(
                    stage.clone(),
                    manifest,
                    vec![phase_timing(RunPhase::Train, started.elapsed(), 0, 0)],
                    metrics.clone(),
                ),
                Err(error) => build_failure_artifact(
                    stage.clone(),
                    manifest,
                    vec![phase_timing(RunPhase::Train, started.elapsed(), 0, 0)],
                    RunExecutionOutcome::InfraFailure,
                    error.to_string(),
                ),
            }
        });

        artifact.stage = stage;
        if artifact.phase_timings.is_empty() {
            artifact
                .phase_timings
                .push(phase_timing(RunPhase::Train, started.elapsed(), 0, 0));
        }
        if artifact.metrics.is_none() {
            if let Ok(metrics) = result {
                artifact.quality_outcome = classify_quality_outcome(metrics);
                artifact.metrics = Some(metrics.clone());
                artifact.execution_outcome = RunExecutionOutcome::Success;
            }
        }
        if artifact.error.is_none() {
            if let Err(error) = result {
                artifact.error = Some(error.to_string());
                artifact.execution_outcome = RunExecutionOutcome::InfraFailure;
            }
        }

        artifact
    }

    fn spawn_species_worker(
        &self,
        definition: SpeciesDefinition,
        index: usize,
        total: usize,
        reporter: Option<&Arc<dyn TournamentReporter>>,
        tx: &mpsc::Sender<SpeciesWorkerMessage>,
    ) {
        let stage = Self::run_stage(definition.id, index, total);
        Self::emit_event(reporter, TournamentProgressEvent::SpeciesStarted(stage));

        let context = self.run_context(index, &definition);
        let backend = self.config.execution_backend.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let started = Instant::now();
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                definition.run(context.clone(), &backend)
            }))
            .map_err(|_| FractalError::InvalidState("species worker panicked".into()))
            .and_then(|result| result);
            let stage = Self::run_stage(definition.id, index, total);
            let artifact =
                Self::capture_species_artifact(&definition, stage, &context, started, &result);
            let _ = tx.send(SpeciesWorkerMessage {
                index,
                elapsed: started.elapsed(),
                result,
                artifact,
            });
        });
    }

    fn run_stage(species: SpeciesId, index: usize, total: usize) -> SpeciesRunStage {
        SpeciesRunStage {
            species,
            ordinal: index + 1,
            total,
        }
    }

    fn emit_event(reporter: Option<&Arc<dyn TournamentReporter>>, event: TournamentProgressEvent) {
        if let Some(reporter) = reporter {
            reporter.on_event(event);
        }
    }

    fn validate_species_definitions(species: &[SpeciesDefinition]) -> Result<(), FractalError> {
        let mut ids = HashSet::with_capacity(species.len());
        let mut variant_names = HashSet::with_capacity(species.len());
        for definition in species {
            if !ids.insert(definition.id) {
                return Err(FractalError::InvalidConfig(format!(
                    "duplicate species id in tournament registry: {}",
                    definition.id
                )));
            }
            definition.variant_name.validate()?;
            if !variant_names.insert(definition.variant_name) {
                return Err(FractalError::InvalidConfig(format!(
                    "duplicate primitive variant name in tournament registry: {}",
                    definition.variant_name
                )));
            }
        }
        Ok(())
    }
}

fn first_execution_failure(records: &[SpeciesRunArtifact]) -> Option<&SpeciesRunArtifact> {
    records.iter().find(|record| {
        matches!(
            record.execution_outcome,
            RunExecutionOutcome::InfraFailure
                | RunExecutionOutcome::TrainTimeout
                | RunExecutionOutcome::EvalConstrained
        )
    })
}
