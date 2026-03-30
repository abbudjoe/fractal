use std::{
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
    registry::{ComputeBackend, ExecutionMode, SpeciesDefinition, SpeciesId, SpeciesRunContext},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentPreset {
    Default,
    FastTest,
    ResearchMedium,
    ChallengerLane,
    BullpenPolish,
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
            Self::BullpenPolish => "bullpen-polish",
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
            Self::BullpenPolish => TournamentConfig::bullpen_polish(),
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
    pub generator_depth_config: GeneratorDepthConfig,
    pub execution_backend: ComputeBackend,
    pub execution_mode: ExecutionMode,
    pub parallelism: usize,
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
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
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
                "selected execution backend is not supported on this platform".into(),
            ));
        }
        if self.parallelism == 0 {
            return Err(FractalError::InvalidConfig(
                "parallelism must be greater than zero".into(),
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
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
        }
    }

    pub fn challenger_lane() -> Self {
        Self {
            dim: 96,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 64,
            max_recursion_depth: 8,
            router_threshold: 0.92,
            batch_size: 8,
            train_steps_per_species: 20,
            eval_batches_per_family: 4,
            learning_rate: 1e-3,
            seed: 42,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
        }
    }

    pub fn bullpen_polish() -> Self {
        Self {
            dim: 192,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 12,
            router_threshold: 0.92,
            batch_size: 8,
            train_steps_per_species: 50,
            eval_batches_per_family: 4,
            learning_rate: 1e-3,
            seed: 42,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
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
            router_threshold: 0.95,
            batch_size: 2,
            train_steps_per_species: 5,
            eval_batches_per_family: 2,
            learning_rate: 1e-3,
            seed: 42,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
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
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::CpuCandle,
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
        }
    }

    pub fn candidate_stress() -> Self {
        Self {
            dim: 192,
            levels: 3,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 20,
            router_threshold: 0.92,
            batch_size: 8,
            train_steps_per_species: 200,
            eval_batches_per_family: 8,
            learning_rate: 1e-3,
            seed: 42,
            generator_depth_config: GeneratorDepthConfig::stress_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
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
}

#[derive(Clone, Debug)]
pub struct SpeciesRunStage {
    pub species: SpeciesId,
    pub ordinal: usize,
    pub total: usize,
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
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family: 96,
            eval_examples_per_family: 32,
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
        self.run_generation_with_reporter(species, None)
    }

    pub fn run_generation_with_reporter(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        match self.config.execution_mode {
            ExecutionMode::Sequential => self.run_sequential(species, reporter),
            ExecutionMode::Parallel => self.run_parallel(species, reporter),
        }
    }

    fn run_sequential(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let mut metrics = Vec::with_capacity(species.len());
        for (index, definition) in species.iter().enumerate() {
            let stage = Self::run_stage(definition.id, index, species.len());
            Self::emit_event(
                reporter.as_ref(),
                TournamentProgressEvent::SpeciesStarted(stage.clone()),
            );
            let started = Instant::now();
            let result = definition.run(self.run_context(index), &self.config.execution_backend)?;
            Self::emit_event(
                reporter.as_ref(),
                TournamentProgressEvent::SpeciesCompleted(SpeciesCompletion {
                    stage,
                    elapsed: started.elapsed(),
                    metrics: result.clone(),
                }),
            );
            metrics.push(result);
        }

        Ok(metrics)
    }

    fn run_parallel(
        &self,
        species: &[SpeciesDefinition],
        reporter: Option<Arc<dyn TournamentReporter>>,
    ) -> Result<Vec<SpeciesRawMetrics>, FractalError> {
        let total = species.len();
        let concurrency = self.config.parallelism.min(total).max(1);
        let (tx, rx) = mpsc::channel();
        let mut launched = 0usize;
        let mut completed = 0usize;
        let mut metrics = vec![None; total];

        while launched < concurrency {
            self.spawn_species_worker(species[launched], launched, total, reporter.as_ref(), &tx);
            launched += 1;
        }

        while completed < total {
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
                    metrics[message.index] = Some(result);
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
                Err(error) => return Err(error),
            }
        }

        metrics
            .into_iter()
            .map(|metric| {
                metric.ok_or_else(|| {
                    FractalError::InvalidState("parallel species result missing".into())
                })
            })
            .collect()
    }

    fn run_context(&self, index: usize) -> SpeciesRunContext {
        SpeciesRunContext {
            index,
            config: self.config.clone(),
            generator: Arc::clone(&self.generator),
        }
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

        let context = self.run_context(index);
        let backend = self.config.execution_backend.clone();
        let tx = tx.clone();
        thread::spawn(move || {
            let started = Instant::now();
            let result =
                std::panic::catch_unwind(AssertUnwindSafe(|| definition.run(context, &backend)))
                    .map_err(|_| FractalError::InvalidState("species worker panicked".into()))
                    .and_then(|result| result);
            let _ = tx.send(SpeciesWorkerMessage {
                index,
                elapsed: started.elapsed(),
                result,
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
}
