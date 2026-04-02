use std::{
    collections::{HashSet, VecDeque},
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use burn::{
    backend::{
        candle::CandleDevice,
        wgpu::{self, graphics::Metal, WgpuDevice},
        Autodiff, Candle, Wgpu as BurnWgpu,
    },
    module::{AutodiffModule, Module, ModuleDisplay, ModuleVisitor, Param},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{
        adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig, AdamW, AdamWConfig,
        GradientsParams, Optimizer,
    },
    prelude::Backend,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::AutodiffBackend, bf16, Bool, ElementConversion, Int, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::{
    data_generator::{SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN},
    error::FractalError,
    fitness::SpeciesRawMetrics,
    lifecycle::{
        CheckpointArtifact, CheckpointArtifactKind, ExperimentSpec, FailureDiagnosticBoundary,
        FailureDiagnosticEvent, FailureSnapshotArtifact, FailureSnapshotArtifactKind,
        FailureSnapshotCaptureTiming, FailureSnapshotContract, FailureSnapshotErrorClass,
        FailureSnapshotPolicy, FailureSnapshotRuntimeState, InterimEvalSnapshot, OptimizerKind,
        OptimizerSpec, PhaseTiming, RunExecutionOutcome, RunManifest, RunPhase, RunQualityOutcome,
        SpeciesRunArtifact, SpeciesRunStage, TournamentConfig, TrainingRuntimeArtifact,
        WeightExportArtifact, WeightExportContract, WeightExportFormat, WeightExportPhase,
        WeightExportPolicy, WeightExportRuntimeState,
    },
    model::{ForwardDebugProbe, FractalModel},
    rule_trait::FractalRule,
};

pub type CandleF32Backend = Candle<f32, i64>;
pub type CandleF32TrainBackend = Autodiff<CandleF32Backend>;
pub type CandleBf16Backend = Candle<bf16, i64>;
pub type CandleBf16TrainBackend = Autodiff<CandleBf16Backend>;
pub type CpuBackend = CandleF32Backend;
pub type CpuTrainBackend = CandleF32TrainBackend;
pub type MetalF32Backend = BurnWgpu<f32, i32>;
pub type MetalF32TrainBackend = Autodiff<MetalF32Backend>;
pub type MetalBf16Backend = BurnWgpu<bf16, i32>;
pub type MetalBf16TrainBackend = Autodiff<MetalBf16Backend>;
pub type MetalBackend = MetalF32Backend;
pub type MetalTrainBackend = MetalF32TrainBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedExecutablePrecisionProfile {
    CandleF32,
    CandleBf16,
    MetalF32,
}

impl ResolvedExecutablePrecisionProfile {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CandleF32 => "candle-f32",
            Self::CandleBf16 => "candle-bf16",
            Self::MetalF32 => "metal-f32",
        }
    }
}

pub fn resolve_precision_profile(
    backend: &ComputeBackend,
    policy: &crate::PrecisionPolicy,
) -> Result<ResolvedExecutablePrecisionProfile, FractalError> {
    use crate::NumericPrecisionKind;

    if policy.quantization.is_enabled() {
        return Err(FractalError::InvalidConfig(format!(
            "quantization profile {} does not yet have an executable runtime implementation",
            policy.quantization.label()
        )));
    }

    match backend {
        ComputeBackend::CpuCandle => match policy.compute {
            NumericPrecisionKind::BackendDefault | NumericPrecisionKind::Fp32 => {
                Ok(ResolvedExecutablePrecisionProfile::CandleF32)
            }
            NumericPrecisionKind::Bf16 => Err(FractalError::InvalidConfig(
                "cpu candle backend does not yet have an executable bf16 precision profile".into(),
            )),
        },
        #[cfg(feature = "cuda")]
        ComputeBackend::CudaCandle { .. } => match policy.compute {
            NumericPrecisionKind::BackendDefault | NumericPrecisionKind::Fp32 => {
                Ok(ResolvedExecutablePrecisionProfile::CandleF32)
            }
            NumericPrecisionKind::Bf16 => Ok(ResolvedExecutablePrecisionProfile::CandleBf16),
        },
        ComputeBackend::MetalWgpu { .. } => match policy.compute {
            NumericPrecisionKind::BackendDefault | NumericPrecisionKind::Fp32 => {
                Ok(ResolvedExecutablePrecisionProfile::MetalF32)
            }
            NumericPrecisionKind::Bf16 => Err(FractalError::InvalidConfig(
                "metal backend does not yet have an executable bf16 precision profile".into(),
            )),
        },
    }
}

enum ConfiguredOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    Adam(OptimizerAdaptor<Adam, M, B>),
    AdamW(OptimizerAdaptor<AdamW, M, B>),
}

impl<M, B> ConfiguredOptimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn from_spec(spec: &OptimizerSpec) -> Self {
        match spec.kind {
            OptimizerKind::Adam => {
                let optimizer = AdamConfig::new()
                    .with_beta_1(spec.beta_1)
                    .with_beta_2(spec.beta_2)
                    .with_epsilon(spec.epsilon)
                    .with_weight_decay(Some(WeightDecayConfig::new(spec.weight_decay as f32)))
                    .init::<B, M>();
                Self::Adam(optimizer)
            }
            OptimizerKind::AdamW => {
                let optimizer = AdamWConfig::new()
                    .with_beta_1(spec.beta_1)
                    .with_beta_2(spec.beta_2)
                    .with_epsilon(spec.epsilon)
                    .with_weight_decay(spec.weight_decay as f32)
                    .init::<B, M>();
                Self::AdamW(optimizer)
            }
        }
    }

    fn load_checkpoint_record(self, path: &Path, device: &B::Device) -> Result<Self, FractalError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        match self {
            Self::Adam(optimizer) => {
                let record = recorder
                    .load(path.to_path_buf(), device)
                    .map_err(recorder_error)?;
                Ok(Self::Adam(optimizer.load_record(record)))
            }
            Self::AdamW(optimizer) => {
                let record = recorder
                    .load(path.to_path_buf(), device)
                    .map_err(recorder_error)?;
                Ok(Self::AdamW(optimizer.load_record(record)))
            }
        }
    }

    fn save_checkpoint_record(&self, path: &Path) -> Result<(), FractalError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        match self {
            Self::Adam(optimizer) => recorder
                .record(optimizer.to_record(), path.to_path_buf())
                .map(|_| ())
                .map_err(recorder_error),
            Self::AdamW(optimizer) => recorder
                .record(optimizer.to_record(), path.to_path_buf())
                .map(|_| ())
                .map_err(recorder_error),
        }
    }

    fn step(&mut self, lr: f64, module: M, grads: GradientsParams) -> M {
        match self {
            Self::Adam(optimizer) => optimizer.step(lr, module, grads),
            Self::AdamW(optimizer) => optimizer.step(lr, module, grads),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CheckpointContractSnapshot {
    species: String,
    variant_name: String,
    dim: usize,
    levels: usize,
    vocab_size: usize,
    max_seq_len: usize,
    max_recursion_depth: usize,
    stability_depth: usize,
    train_batch_size: usize,
    eval_batch_size: usize,
    train_steps_per_species: usize,
    train_token_budget: Option<usize>,
    eval_batches_per_family: usize,
    perplexity_eval_batches: usize,
    arc_eval_batches: usize,
    seed: u64,
    optimizer: OptimizerSpec,
    launch_policy: crate::LaunchPolicySpec,
    training_input: Option<crate::TrainingInputSpec>,
    execution_backend: String,
    branch: Option<String>,
    commit_sha: Option<String>,
}

impl CheckpointContractSnapshot {
    fn from_manifest(stage: SpeciesRunStage, manifest: &RunManifest) -> Self {
        let experiment = manifest.experiment.as_ref();
        let config = &manifest.config;
        Self {
            species: stage.species.as_str().to_owned(),
            variant_name: manifest.variant_name.as_str().to_owned(),
            dim: config.dim,
            levels: config.levels,
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            max_recursion_depth: config.max_recursion_depth,
            stability_depth: config.stability_depth,
            train_batch_size: config.train_batch_size,
            eval_batch_size: config.eval_batch_size,
            train_steps_per_species: config.train_steps_per_species,
            train_token_budget: config.train_token_budget,
            eval_batches_per_family: config.eval_batches_per_family,
            perplexity_eval_batches: config.effective_perplexity_eval_batches(),
            arc_eval_batches: config.effective_arc_eval_batches(),
            seed: config.seed,
            optimizer: config.optimizer.clone(),
            launch_policy: config.launch_policy.clone(),
            training_input: experiment.map(|spec| spec.training_input.clone()),
            execution_backend: backend_name(&config.execution_backend).to_owned(),
            branch: experiment.and_then(|spec| spec.experiment_id.branch.clone()),
            commit_sha: experiment.and_then(|spec| spec.experiment_id.commit_sha.clone()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RuntimeCheckpointState {
    contract: CheckpointContractSnapshot,
    completed_steps: usize,
    planned_steps: usize,
    train_tokens_seen: usize,
    target_train_tokens: usize,
    best_perplexity: Option<f64>,
    next_checkpoint_token: Option<usize>,
    next_perplexity_token: Option<usize>,
    next_stability_token: Option<usize>,
    next_arc_token: Option<usize>,
    next_systems_speed_token: Option<usize>,
    checkpoints: Vec<CheckpointArtifactState>,
    weight_exports: PersistedWeightExportState,
    interim_evaluations: Vec<InterimEvalSnapshotState>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum PersistedWeightExportState {
    Legacy(Vec<WeightExportArtifact>),
    Current(WeightExportRuntimeState),
}

impl PersistedWeightExportState {
    fn into_runtime_state(self, policy: &WeightExportPolicy) -> WeightExportRuntimeState {
        match self {
            Self::Legacy(artifacts) => {
                let mut state = WeightExportRuntimeState::from_policy(policy.clone());
                for artifact in artifacts {
                    state.record_success(artifact);
                }
                state
            }
            Self::Current(state) => state,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointArtifactState {
    kind: String,
    tokens_seen: usize,
    completed_steps: usize,
    directory: String,
    long_context_perplexity: Option<f64>,
}

impl CheckpointArtifactState {
    fn into_runtime_artifact(self) -> CheckpointArtifact {
        CheckpointArtifact {
            kind: match self.kind.as_str() {
                "latest" => CheckpointArtifactKind::Latest,
                "previous" => CheckpointArtifactKind::Previous,
                "best" => CheckpointArtifactKind::Best,
                "final" => CheckpointArtifactKind::Final,
                other => panic!("unknown checkpoint kind in saved state: {other}"),
            },
            tokens_seen: self.tokens_seen,
            completed_steps: self.completed_steps,
            directory: self.directory,
            long_context_perplexity: self.long_context_perplexity,
        }
    }
}

impl From<&CheckpointArtifact> for CheckpointArtifactState {
    fn from(value: &CheckpointArtifact) -> Self {
        Self {
            kind: value.kind.as_str().to_owned(),
            tokens_seen: value.tokens_seen,
            completed_steps: value.completed_steps,
            directory: value.directory.clone(),
            long_context_perplexity: value.long_context_perplexity,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InterimEvalSnapshotState {
    tokens_seen: usize,
    completed_steps: usize,
    stability_score: Option<f64>,
    long_context_perplexity: Option<f64>,
    arc_accuracy: Option<f64>,
    tokens_per_sec: Option<f64>,
}

impl InterimEvalSnapshotState {
    fn into_runtime_artifact(self) -> InterimEvalSnapshot {
        InterimEvalSnapshot {
            tokens_seen: self.tokens_seen,
            completed_steps: self.completed_steps,
            stability_score: self.stability_score,
            long_context_perplexity: self.long_context_perplexity,
            arc_accuracy: self.arc_accuracy,
            tokens_per_sec: self.tokens_per_sec,
        }
    }
}

impl From<&InterimEvalSnapshot> for InterimEvalSnapshotState {
    fn from(value: &InterimEvalSnapshot) -> Self {
        Self {
            tokens_seen: value.tokens_seen,
            completed_steps: value.completed_steps,
            stability_score: value.stability_score,
            long_context_perplexity: value.long_context_perplexity,
            arc_accuracy: value.arc_accuracy,
            tokens_per_sec: value.tokens_per_sec,
        }
    }
}

#[derive(Clone, Debug)]
struct TrainingExecutionPlan {
    planned_steps: usize,
    target_train_tokens: usize,
}

#[derive(Clone, Debug)]
struct CheckpointRuntimePaths {
    root: PathBuf,
    latest: PathBuf,
    previous: PathBuf,
    best: PathBuf,
    final_slot: PathBuf,
}

#[derive(Clone, Debug)]
pub(crate) struct WeightExportRuntimePaths {
    root: PathBuf,
}

#[derive(Clone, Debug)]
struct FailureSnapshotRuntimePaths {
    root: PathBuf,
}

#[derive(Clone, Debug)]
struct LaunchRuntimeState {
    completed_steps: usize,
    planned_steps: usize,
    train_tokens_seen: usize,
    target_train_tokens: usize,
    resumed_from_checkpoint: bool,
    best_perplexity: Option<f64>,
    next_checkpoint_token: Option<usize>,
    next_perplexity_token: Option<usize>,
    next_stability_token: Option<usize>,
    next_arc_token: Option<usize>,
    next_systems_speed_token: Option<usize>,
    checkpoints: Vec<CheckpointArtifact>,
    weight_export: WeightExportRuntimeState,
    interim_evaluations: Vec<InterimEvalSnapshot>,
}

impl LaunchRuntimeState {
    fn fresh(plan: &TrainingExecutionPlan, launch_policy: &crate::LaunchPolicySpec) -> Self {
        Self {
            completed_steps: 0,
            planned_steps: plan.planned_steps,
            train_tokens_seen: 0,
            target_train_tokens: plan.target_train_tokens,
            resumed_from_checkpoint: false,
            best_perplexity: None,
            next_checkpoint_token: launch_policy.checkpoint.interval_tokens,
            next_perplexity_token: launch_policy.eval_cadence.perplexity_interval_tokens,
            next_stability_token: launch_policy.eval_cadence.stability_interval_tokens,
            next_arc_token: launch_policy.eval_cadence.arc_interval_tokens,
            next_systems_speed_token: launch_policy.eval_cadence.systems_speed_interval_tokens,
            checkpoints: Vec::new(),
            weight_export: WeightExportRuntimeState::from_policy(
                launch_policy.weight_export.clone(),
            ),
            interim_evaluations: Vec::new(),
        }
    }

    fn from_checkpoint(
        state: RuntimeCheckpointState,
        launch_policy: &crate::LaunchPolicySpec,
    ) -> Self {
        Self {
            completed_steps: state.completed_steps,
            planned_steps: state.planned_steps,
            train_tokens_seen: state.train_tokens_seen,
            target_train_tokens: state.target_train_tokens,
            resumed_from_checkpoint: true,
            best_perplexity: state.best_perplexity,
            next_checkpoint_token: state.next_checkpoint_token,
            next_perplexity_token: state.next_perplexity_token,
            next_stability_token: state.next_stability_token,
            next_arc_token: state.next_arc_token,
            next_systems_speed_token: state.next_systems_speed_token,
            checkpoints: state
                .checkpoints
                .into_iter()
                .map(CheckpointArtifactState::into_runtime_artifact)
                .collect(),
            weight_export: state
                .weight_exports
                .into_runtime_state(&launch_policy.weight_export),
            interim_evaluations: state
                .interim_evaluations
                .into_iter()
                .map(InterimEvalSnapshotState::into_runtime_artifact)
                .collect(),
        }
    }

    fn checkpoint_state(&self, contract: CheckpointContractSnapshot) -> RuntimeCheckpointState {
        RuntimeCheckpointState {
            contract,
            completed_steps: self.completed_steps,
            planned_steps: self.planned_steps,
            train_tokens_seen: self.train_tokens_seen,
            target_train_tokens: self.target_train_tokens,
            best_perplexity: self.best_perplexity,
            next_checkpoint_token: self.next_checkpoint_token,
            next_perplexity_token: self.next_perplexity_token,
            next_stability_token: self.next_stability_token,
            next_arc_token: self.next_arc_token,
            next_systems_speed_token: self.next_systems_speed_token,
            checkpoints: self
                .checkpoints
                .iter()
                .map(CheckpointArtifactState::from)
                .collect(),
            weight_exports: PersistedWeightExportState::Current(self.weight_export.clone()),
            interim_evaluations: self
                .interim_evaluations
                .iter()
                .map(InterimEvalSnapshotState::from)
                .collect(),
        }
    }

    fn artifact(&self, launch_policy: &crate::LaunchPolicySpec) -> TrainingRuntimeArtifact {
        TrainingRuntimeArtifact {
            completed_steps: self.completed_steps,
            planned_steps: self.planned_steps,
            train_tokens_seen: self.train_tokens_seen,
            target_train_tokens: self.target_train_tokens,
            resumed_from_checkpoint: self.resumed_from_checkpoint,
            checkpoints: self.checkpoints.clone(),
            weight_export: self.weight_export.clone(),
            failure_snapshot: FailureSnapshotRuntimeState::from_policy(
                launch_policy.failure_snapshot.clone(),
            ),
            interim_evaluations: self.interim_evaluations.clone(),
        }
    }
}

const FAILURE_DIAGNOSTIC_TAIL_LIMIT: usize = 32;

#[derive(Clone, Debug)]
struct FailureDiagnosticsRecorder {
    enabled: bool,
    events: VecDeque<FailureDiagnosticEvent>,
}

impl FailureDiagnosticsRecorder {
    fn from_policy(policy: &FailureSnapshotPolicy) -> Self {
        Self {
            enabled: policy.enabled && policy.capture_diagnostics_tail,
            events: VecDeque::new(),
        }
    }

    fn record(
        &mut self,
        boundary: FailureDiagnosticBoundary,
        phase: RunPhase,
        runtime: &TrainingRuntimeArtifact,
        step: Option<usize>,
    ) {
        if !self.enabled {
            return;
        }
        if self.events.len() == FAILURE_DIAGNOSTIC_TAIL_LIMIT {
            self.events.pop_front();
        }
        self.events.push_back(FailureDiagnosticEvent {
            boundary,
            phase,
            completed_steps: runtime.completed_steps,
            planned_steps: runtime.planned_steps,
            train_tokens_seen: runtime.train_tokens_seen,
            step,
        });
    }

    fn snapshot(&self) -> Vec<FailureDiagnosticEvent> {
        self.events.iter().cloned().collect()
    }

    fn last_boundary(&self) -> Option<FailureDiagnosticBoundary> {
        self.events.back().map(|event| event.boundary)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeBackend {
    CpuCandle,
    #[cfg(feature = "cuda")]
    CudaCandle {
        device_index: usize,
    },
    MetalWgpu {
        device: WgpuDevice,
    },
}

impl ComputeBackend {
    pub fn default_for_current_platform() -> Self {
        if cfg!(target_os = "macos") {
            Self::metal_default()
        } else {
            Self::CpuCandle
        }
    }

    pub fn metal_default() -> Self {
        Self::MetalWgpu {
            device: WgpuDevice::DefaultDevice,
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn cuda_default() -> Self {
        Self::CudaCandle { device_index: 0 }
    }

    pub fn is_supported_on_current_platform(&self) -> bool {
        match self {
            Self::CpuCandle => true,
            #[cfg(feature = "cuda")]
            Self::CudaCandle { .. } => cfg!(not(target_os = "macos")),
            Self::MetalWgpu { .. } => cfg!(target_os = "macos"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpeciesId {
    P1Contractive,
    P3Hierarchical,
    B2StableHierarchical,
    B1FractalGated,
    P1FractalHybrid,
    P1FractalHybridComposite,
    P1FractalHybridDynGate,
    P2Mandelbrot,
    B3FractalHierarchical,
    B4Universal,
    Ifs,
    GeneralizedMobius,
    LogisticChaoticMap,
    JuliaRecursiveEscape,
    MandelboxRecursive,
}

impl SpeciesId {
    pub const ALL: [Self; 15] = [
        Self::P1Contractive,
        Self::P3Hierarchical,
        Self::B2StableHierarchical,
        Self::B1FractalGated,
        Self::P1FractalHybrid,
        Self::P1FractalHybridComposite,
        Self::P1FractalHybridDynGate,
        Self::P2Mandelbrot,
        Self::B3FractalHierarchical,
        Self::B4Universal,
        Self::Ifs,
        Self::GeneralizedMobius,
        Self::LogisticChaoticMap,
        Self::JuliaRecursiveEscape,
        Self::MandelboxRecursive,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1Contractive => "p1_contractive",
            Self::P3Hierarchical => "p3_hierarchical",
            Self::B2StableHierarchical => "b2_stable_hierarchical",
            Self::B1FractalGated => "b1_fractal_gated",
            Self::P1FractalHybrid => "p1_fractal_hybrid",
            Self::P1FractalHybridComposite => "p1_fractal_hybrid_composite",
            Self::P1FractalHybridDynGate => "p1_fractal_hybrid_dyn_gate",
            Self::P2Mandelbrot => "p2_mandelbrot",
            Self::B3FractalHierarchical => "b3_fractal_hierarchical",
            Self::B4Universal => "b4_universal",
            Self::Ifs => "ifs",
            Self::GeneralizedMobius => "generalized_mobius",
            Self::LogisticChaoticMap => "logistic_chaotic_map",
            Self::JuliaRecursiveEscape => "julia_recursive_escape",
            Self::MandelboxRecursive => "mandelbox_recursive",
        }
    }
}

impl Display for SpeciesId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).as_str())
    }
}

impl FromStr for SpeciesId {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "p1_contractive" => Ok(Self::P1Contractive),
            "p3_hierarchical" => Ok(Self::P3Hierarchical),
            "b2_stable_hierarchical" => Ok(Self::B2StableHierarchical),
            "b1_fractal_gated" => Ok(Self::B1FractalGated),
            "p1_fractal_hybrid" => Ok(Self::P1FractalHybrid),
            "p1_fractal_hybrid_composite" => Ok(Self::P1FractalHybridComposite),
            "p1_fractal_hybrid_dyn_gate" => Ok(Self::P1FractalHybridDynGate),
            "p2_mandelbrot" => Ok(Self::P2Mandelbrot),
            "b3_fractal_hierarchical" => Ok(Self::B3FractalHierarchical),
            "b4_universal" => Ok(Self::B4Universal),
            "ifs" => Ok(Self::Ifs),
            "generalized_mobius" => Ok(Self::GeneralizedMobius),
            "logistic_chaotic_map" => Ok(Self::LogisticChaoticMap),
            "julia_recursive_escape" => Ok(Self::JuliaRecursiveEscape),
            "mandelbox_recursive" => Ok(Self::MandelboxRecursive),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrimitiveVariantName(&'static str);

impl PrimitiveVariantName {
    pub const fn new_unchecked(name: &'static str) -> Self {
        Self(name)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }

    pub fn validate(self) -> Result<(), FractalError> {
        if is_valid_primitive_variant_name(self.0) {
            Ok(())
        } else {
            Err(FractalError::InvalidConfig(format!(
                "primitive variant name must match [base]_[lever-description]_v[version]: {}",
                self.0
            )))
        }
    }
}

impl Display for PrimitiveVariantName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

pub fn is_valid_primitive_variant_name(name: &str) -> bool {
    let mut parts = name.split('_').peekable();
    let mut count = 0usize;

    while let Some(part) = parts.next() {
        count += 1;
        if part.is_empty() {
            return false;
        }
        if parts.peek().is_none() {
            return is_valid_variant_version(part) && count >= 3;
        }
        if !part
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
        {
            return false;
        }
    }

    false
}

fn is_valid_variant_version(part: &str) -> bool {
    let Some(version) = part.strip_prefix('v') else {
        return false;
    };
    !version.is_empty()
        && !version.starts_with('0')
        && version.chars().all(|ch| ch.is_ascii_digit())
}

#[derive(Clone, Debug)]
pub struct SpeciesRunContext {
    pub index: usize,
    pub config: TournamentConfig,
    pub generator: Arc<SimpleHierarchicalGenerator>,
    pub variant_name: PrimitiveVariantName,
    pub experiment: Option<ExperimentSpec>,
}

type CpuRunner = fn(SpeciesRunContext) -> Result<SpeciesRawMetrics, FractalError>;
#[cfg(feature = "cuda")]
type CudaRunner = fn(SpeciesRunContext, CandleDevice) -> Result<SpeciesRawMetrics, FractalError>;
type MetalRunner = fn(SpeciesRunContext, WgpuDevice) -> Result<SpeciesRawMetrics, FractalError>;

#[derive(Clone, Copy)]
pub struct SpeciesDefinition {
    pub id: SpeciesId,
    pub variant_name: PrimitiveVariantName,
    cpu_runner: CpuRunner,
    #[cfg(feature = "cuda")]
    cuda_runner: CudaRunner,
    metal_runner: MetalRunner,
}

impl SpeciesDefinition {
    #[cfg(not(feature = "cuda"))]
    pub const fn new(
        id: SpeciesId,
        variant_name: PrimitiveVariantName,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
    ) -> Self {
        Self {
            id,
            variant_name,
            cpu_runner,
            metal_runner,
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn new(
        id: SpeciesId,
        variant_name: PrimitiveVariantName,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
        cuda_runner: CudaRunner,
    ) -> Self {
        Self {
            id,
            variant_name,
            cpu_runner,
            cuda_runner,
            metal_runner,
        }
    }

    pub fn run(
        &self,
        context: SpeciesRunContext,
        backend: &ComputeBackend,
    ) -> Result<SpeciesRawMetrics, FractalError> {
        match backend {
            ComputeBackend::CpuCandle => (self.cpu_runner)(context),
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaCandle { device_index } => {
                (self.cuda_runner)(context, cuda_device(*device_index))
            }
            ComputeBackend::MetalWgpu { device } => (self.metal_runner)(context, device.clone()),
        }
    }
}

pub fn cpu_device() -> CandleDevice {
    CandleDevice::Cpu
}

#[cfg(feature = "cuda")]
pub fn cuda_device(index: usize) -> CandleDevice {
    CandleDevice::cuda(index)
}

pub fn run_species_with_factory<B, R, F>(
    species: SpeciesId,
    context: SpeciesRunContext,
    device: B::Device,
    factory: F,
) -> Result<SpeciesRawMetrics, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B>
        + Module<B>
        + AutodiffModule<B>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
    F: FnOnce(&TournamentConfig, &B::Device) -> R,
{
    let variant_name = context.variant_name;
    let experiment = context.experiment.clone();
    B::seed(
        &device,
        context.config.seed.wrapping_add(context.index as u64 * 101),
    );

    let batches = prepare_batches_for_run::<B>(&context.generator, &context.config, &device)?;
    let rule = factory(&context.config, &device);
    run_species_with_batches(
        species,
        variant_name,
        context.config,
        experiment,
        device,
        rule,
        batches,
    )
}

pub fn run_species_with_factory_candle<R, F>(
    species: SpeciesId,
    context: SpeciesRunContext,
    device: CandleDevice,
    factory: F,
) -> Result<SpeciesRawMetrics, FractalError>
where
    R: FractalRule<CpuTrainBackend>
        + Module<CpuTrainBackend>
        + AutodiffModule<CpuTrainBackend>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<CpuTrainBackend>>::InnerModule:
        Module<<CpuTrainBackend as AutodiffBackend>::InnerBackend> + ModuleDisplay,
    F: FnOnce(&TournamentConfig, &CandleDevice) -> R,
{
    let variant_name = context.variant_name;
    let experiment = context.experiment.clone();
    CpuTrainBackend::seed(
        &device,
        context.config.seed.wrapping_add(context.index as u64 * 101),
    );

    let batches = prepare_candle_batches_for_run(&context.generator, &context.config, &device)?;
    let rule = factory(&context.config, &device);
    run_species_with_batches(
        species,
        variant_name,
        context.config,
        experiment,
        device,
        rule,
        batches,
    )
}

pub fn initialize_metal_runtime(device: &WgpuDevice) {
    static INITIALIZED_DEVICES: OnceLock<Mutex<HashSet<WgpuDevice>>> = OnceLock::new();

    let devices = INITIALIZED_DEVICES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut initialized = match devices.lock() {
        Ok(initialized) => initialized,
        Err(poisoned) => poisoned.into_inner(),
    };
    if initialized.contains(device) {
        return;
    }

    wgpu::init_setup::<Metal>(device, Default::default());
    initialized.insert(device.clone());
}

thread_local! {
    static LAST_SPECIES_RUN_ARTIFACT: std::cell::RefCell<Option<SpeciesRunArtifact>> = const {
        std::cell::RefCell::new(None)
    };
}

pub fn take_last_species_run_artifact() -> Option<SpeciesRunArtifact> {
    LAST_SPECIES_RUN_ARTIFACT.with(|slot| slot.borrow_mut().take())
}

#[derive(Clone, Debug)]
pub struct TrainingBatchSet<B: AutodiffBackend> {
    pub train_sentence: Vec<TokenBatch<B>>,
    pub train_arc: Option<Vec<TokenBatch<B>>>,
    pub eval_sentence: Vec<TokenBatch<B>>,
    pub eval_arc: Vec<TokenBatch<B>>,
}

impl<B: AutodiffBackend> TrainingBatchSet<B> {
    pub fn train_batches_for_step(&self, step: usize) -> &[TokenBatch<B>] {
        match (&self.train_arc, step % 2) {
            (Some(train_arc), 1) => train_arc,
            _ => &self.train_sentence,
        }
    }
}

fn record_species_run_artifact(artifact: SpeciesRunArtifact) {
    LAST_SPECIES_RUN_ARTIFACT.with(|slot| {
        *slot.borrow_mut() = Some(artifact);
    });
}

fn build_run_manifest(
    species: SpeciesId,
    config: &TournamentConfig,
    timeout_budget: Option<Duration>,
    variant_name: PrimitiveVariantName,
    experiment: Option<ExperimentSpec>,
) -> RunManifest {
    RunManifest {
        variant_name,
        timeout_budget,
        config: config.clone(),
        experiment: experiment.or_else(|| config.resolved_experiment(species, variant_name)),
    }
}

pub(crate) fn classify_quality_outcome(metrics: &SpeciesRawMetrics) -> RunQualityOutcome {
    if !metrics.grad_norm_depth_20.is_finite()
        || !metrics.long_context_perplexity.is_finite()
        || !metrics.arc_accuracy.is_finite()
        || !metrics.tokens_per_sec.is_finite()
    {
        RunQualityOutcome::NumericFailure
    } else if metrics.arc_accuracy <= 0.05 && metrics.long_context_perplexity > 8.0 {
        RunQualityOutcome::LowSignal
    } else if metrics.tokens_per_sec <= 1.0 {
        RunQualityOutcome::RuntimeCost
    } else {
        RunQualityOutcome::Clean
    }
}

pub(crate) fn build_success_artifact(
    stage: SpeciesRunStage,
    manifest: RunManifest,
    phase_timings: Vec<PhaseTiming>,
    metrics: SpeciesRawMetrics,
) -> SpeciesRunArtifact {
    let quality_outcome = classify_quality_outcome(&metrics);
    let training_runtime = TrainingRuntimeArtifact::empty(&manifest.config.launch_policy);
    SpeciesRunArtifact {
        stage,
        manifest,
        phase_timings,
        training_runtime,
        execution_outcome: RunExecutionOutcome::Success,
        quality_outcome,
        error: None,
        metrics: Some(metrics),
    }
}

pub(crate) fn build_failure_artifact(
    stage: SpeciesRunStage,
    manifest: RunManifest,
    phase_timings: Vec<PhaseTiming>,
    execution_outcome: RunExecutionOutcome,
    error: FractalError,
) -> SpeciesRunArtifact {
    let mut training_runtime = TrainingRuntimeArtifact::empty(&manifest.config.launch_policy);
    let diagnostics =
        FailureDiagnosticsRecorder::from_policy(&manifest.config.launch_policy.failure_snapshot);
    let snapshot = capture_failure_snapshot_without_model(
        stage.clone(),
        &manifest,
        &training_runtime,
        &diagnostics,
        execution_outcome,
        &error,
        FailureSnapshotCaptureTiming::AfterPanicPropagation,
    );
    attach_failure_snapshot(&mut training_runtime, snapshot);
    SpeciesRunArtifact {
        stage,
        manifest,
        phase_timings,
        training_runtime,
        execution_outcome,
        quality_outcome: RunQualityOutcome::Clean,
        error: Some(error.to_string()),
        metrics: None,
    }
}

pub(crate) fn phase_timing(
    phase: RunPhase,
    elapsed: Duration,
    completed: usize,
    total: usize,
) -> PhaseTiming {
    PhaseTiming {
        phase,
        elapsed,
        completed,
        total,
    }
}

fn timeout_outcome_for_phase(phase: RunPhase) -> RunExecutionOutcome {
    match phase {
        RunPhase::Train => RunExecutionOutcome::TrainTimeout,
        RunPhase::Stability | RunPhase::Perplexity | RunPhase::ArcSpeed => {
            RunExecutionOutcome::EvalConstrained
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_failure_species_artifact<B, R>(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    phase_timings: Vec<PhaseTiming>,
    training_runtime: &TrainingRuntimeArtifact,
    diagnostics: &FailureDiagnosticsRecorder,
    execution_outcome: RunExecutionOutcome,
    error: &FractalError,
    capture_timing: FailureSnapshotCaptureTiming,
    model: Option<&FractalModel<B, R>>,
) -> SpeciesRunArtifact
where
    B: Backend,
    R: FractalRule<B> + Module<B> + ModuleDisplay + Clone + std::fmt::Debug,
{
    let mut runtime = training_runtime.clone();
    let snapshot = capture_failure_snapshot(
        stage.clone(),
        manifest,
        training_runtime,
        diagnostics,
        execution_outcome,
        error,
        capture_timing,
        model,
    );
    attach_failure_snapshot(&mut runtime, snapshot);
    SpeciesRunArtifact {
        stage,
        manifest: manifest.clone(),
        phase_timings,
        training_runtime: runtime,
        execution_outcome,
        quality_outcome: RunQualityOutcome::Clean,
        error: Some(error.to_string()),
        metrics: None,
    }
}

pub fn run_species_with_batches<B, R>(
    species: SpeciesId,
    variant_name: PrimitiveVariantName,
    config: TournamentConfig,
    experiment: Option<ExperimentSpec>,
    device: B::Device,
    rule: R,
    batches: TrainingBatchSet<B>,
) -> Result<SpeciesRawMetrics, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B>
        + Module<B>
        + AutodiffModule<B>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let manifest = build_run_manifest(
        species,
        &config,
        config.run_timeout,
        variant_name,
        experiment,
    );
    let mut diagnostics =
        FailureDiagnosticsRecorder::from_policy(&config.launch_policy.failure_snapshot);
    let run_started = Instant::now();
    let deadline = config.run_timeout.map(|timeout| run_started + timeout);
    let stage = SpeciesRunStage {
        species,
        ordinal: config.parallelism.max(1),
        total: config.parallelism.max(1),
    };
    let model = FractalModel::new(
        config.vocab_size,
        config.dim,
        config.max_recursion_depth,
        config.router_threshold,
        PAD_TOKEN,
        rule,
        &device,
    );
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN]))
        .init(&device);
    let optimizer = ConfiguredOptimizer::<FractalModel<B, R>, B>::from_spec(&config.optimizer);
    let execution_plan = match build_training_execution_plan(&batches, &config) {
        Ok(plan) => plan,
        Err(error) => {
            let runtime = TrainingRuntimeArtifact::empty(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                Vec::new(),
                &runtime,
                &diagnostics,
                RunExecutionOutcome::InfraFailure,
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
    };
    let checkpoint_contract = CheckpointContractSnapshot::from_manifest(stage.clone(), &manifest);
    let weight_export_paths = resolve_weight_export_paths(&stage, &manifest);
    let (mut model, mut optimizer, mut launch_runtime, checkpoint_paths) =
        match maybe_restore_checkpoint(
            stage.clone(),
            &manifest,
            &execution_plan,
            model,
            optimizer,
            &device,
        ) {
            Ok(restored) => restored,
            Err(error) => {
                let runtime = TrainingRuntimeArtifact::empty(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    Vec::new(),
                    &runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    None::<&FractalModel<B, R>>,
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
    let mut phase_timings = Vec::with_capacity(4);
    diagnostics.record(
        FailureDiagnosticBoundary::ExecutionPlanBuilt,
        RunPhase::Train,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );
    diagnostics.record(
        FailureDiagnosticBoundary::CheckpointRestoreComplete,
        RunPhase::Train,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );
    diagnostics.record(
        FailureDiagnosticBoundary::TrainPhaseStarted,
        RunPhase::Train,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );

    log_species_phase_start(
        species,
        "train",
        &[
            format!("steps={}", launch_runtime.planned_steps),
            format!("target_tokens={}", launch_runtime.target_train_tokens),
            format!("train_batch={}", config.train_batch_size),
        ],
    );
    let train_started = Instant::now();
    for step in launch_runtime.completed_steps..launch_runtime.planned_steps {
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                let elapsed = train_started.elapsed();
                phase_timings.push(phase_timing(
                    RunPhase::Train,
                    elapsed,
                    launch_runtime.completed_steps,
                    launch_runtime.planned_steps,
                ));
                let error =
                    FractalError::InvalidState("run timeout exceeded during training".into());
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    timeout_outcome_for_phase(RunPhase::Train),
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        }
        let train_batches = batches.train_batches_for_step(step);
        if train_batches.is_empty() {
            phase_timings.push(phase_timing(
                RunPhase::Train,
                train_started.elapsed(),
                launch_runtime.completed_steps,
                launch_runtime.planned_steps,
            ));
            let error = FractalError::InvalidState("training batch cache was empty".into());
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                RunExecutionOutcome::InfraFailure,
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
        diagnostics.record(
            FailureDiagnosticBoundary::TrainStepStarted,
            RunPhase::Train,
            &launch_runtime.artifact(&config.launch_policy),
            Some(step),
        );
        let batch = &train_batches[step % train_batches.len()];
        if should_fire_debug_probe(
            step,
            config.launch_policy.debug.train_step_log_interval_steps,
        ) {
            let mut details = vec![
                format!("train_step={step}"),
                format!("planned_steps={}", launch_runtime.planned_steps),
                format!("tokens_seen={}", launch_runtime.train_tokens_seen),
                format!("batch_token_count={}", batch.token_count),
                format!("input_shape={:?}", batch.input_ids.dims()),
                format!("target_shape={:?}", batch.target_ids.dims()),
            ];
            if should_fire_debug_probe(
                step,
                config.launch_policy.debug.cuda_memory_log_interval_steps,
            ) {
                if let Some(snapshot) = cuda_memory_snapshot() {
                    details.push(snapshot);
                }
            }
            log_species_debug_probe(species, &details);
        }
        let debug_probe = if config
            .launch_policy
            .debug
            .forward_trace_train_steps
            .is_some_and(|limit| step < limit)
        {
            Some(ForwardDebugProbe {
                train_step: Some(step),
                position_log_interval: config.launch_policy.debug.forward_position_log_interval,
            })
        } else {
            None
        };
        let loss = match model.loss(batch, &criterion, None, true, debug_probe) {
            Ok(loss) => loss,
            Err(error) => {
                phase_timings.push(phase_timing(
                    RunPhase::Train,
                    train_started.elapsed(),
                    launch_runtime.completed_steps,
                    launch_runtime.planned_steps,
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        let scheduled_learning_rate = config.optimizer.learning_rate_at_tokens(
            launch_runtime.train_tokens_seen,
            launch_runtime.target_train_tokens,
        );
        let mut grads = grads;
        if let Some(max_norm) = config.optimizer.gradient_clip_norm {
            clip_gradients_global_norm(&model, &mut grads, max_norm);
        }
        model = optimizer.step(scheduled_learning_rate, model, grads);
        launch_runtime.train_tokens_seen += batch.token_count;
        launch_runtime.completed_steps = step + 1;
        let completed_step = launch_runtime.completed_steps;
        if should_fire_debug_probe(
            step,
            config.launch_policy.debug.train_step_log_interval_steps,
        ) {
            let mut details = vec![
                format!("train_step_done={completed_step}"),
                format!("tokens_seen={}", launch_runtime.train_tokens_seen),
                format!("scheduled_lr={scheduled_learning_rate:.8}"),
            ];
            if should_fire_debug_probe(
                step,
                config.launch_policy.debug.cuda_memory_log_interval_steps,
            ) {
                if let Some(snapshot) = cuda_memory_snapshot() {
                    details.push(snapshot);
                }
            }
            log_species_debug_probe(species, &details);
        }
        diagnostics.record(
            FailureDiagnosticBoundary::TrainStepCompleted,
            RunPhase::Train,
            &launch_runtime.artifact(&config.launch_policy),
            Some(completed_step),
        );

        let latest_snapshot = match maybe_capture_interim_eval(
            species,
            &model,
            &criterion,
            &batches,
            &config,
            &mut launch_runtime,
        ) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                phase_timings.push(phase_timing(
                    RunPhase::Train,
                    train_started.elapsed(),
                    launch_runtime.completed_steps,
                    launch_runtime.planned_steps,
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
        if latest_snapshot.is_some() {
            diagnostics.record(
                FailureDiagnosticBoundary::InterimEvaluationComplete,
                RunPhase::Train,
                &launch_runtime.artifact(&config.launch_policy),
                Some(completed_step),
            );
        }
        let latest_perplexity = latest_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.long_context_perplexity);
        let best_improved = latest_perplexity
            .map(|perplexity| match launch_runtime.best_perplexity {
                Some(best) => perplexity < best,
                None => true,
            })
            .unwrap_or(false);
        if let Some(perplexity) = latest_perplexity {
            if best_improved {
                launch_runtime.best_perplexity = Some(perplexity);
            }
        }
        let checkpoint_due = advance_token_milestone(
            &mut launch_runtime.next_checkpoint_token,
            config.launch_policy.checkpoint.interval_tokens,
            launch_runtime.train_tokens_seen,
        );
        if checkpoint_due {
            if let Err(error) = maybe_persist_latest_checkpoint(
                &checkpoint_paths,
                &mut launch_runtime,
                &checkpoint_contract,
                &config.launch_policy,
                &model,
                &optimizer,
                latest_perplexity,
            ) {
                phase_timings.push(phase_timing(
                    RunPhase::Train,
                    train_started.elapsed(),
                    launch_runtime.completed_steps,
                    launch_runtime.planned_steps,
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
            diagnostics.record(
                FailureDiagnosticBoundary::LatestCheckpointPersisted,
                RunPhase::Train,
                &launch_runtime.artifact(&config.launch_policy),
                Some(completed_step),
            );
        }
        if best_improved {
            if let Err(error) = maybe_persist_best_checkpoint(
                &checkpoint_paths,
                &mut launch_runtime,
                &checkpoint_contract,
                &config.launch_policy,
                &model,
                &optimizer,
                latest_perplexity,
            ) {
                phase_timings.push(phase_timing(
                    RunPhase::Train,
                    train_started.elapsed(),
                    launch_runtime.completed_steps,
                    launch_runtime.planned_steps,
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
            diagnostics.record(
                FailureDiagnosticBoundary::BestCheckpointPersisted,
                RunPhase::Train,
                &launch_runtime.artifact(&config.launch_policy),
                Some(completed_step),
            );
            if config
                .launch_policy
                .weight_export
                .phases
                .contains(&WeightExportPhase::Best)
            {
                if let Err(error) = apply_weight_export_attempt(
                    &mut launch_runtime.weight_export,
                    WeightExportPhase::Best,
                    export_weight_phase(
                        stage.clone(),
                        &manifest,
                        &config.launch_policy.weight_export,
                        WeightExportPhase::Best,
                        &model,
                        &weight_export_paths,
                        config.launch_policy.weight_export.required,
                    ),
                ) {
                    phase_timings.push(phase_timing(
                        RunPhase::Train,
                        train_started.elapsed(),
                        launch_runtime.completed_steps,
                        launch_runtime.planned_steps,
                    ));
                    let training_runtime = launch_runtime.artifact(&config.launch_policy);
                    let artifact = build_failure_species_artifact(
                        stage,
                        &manifest,
                        phase_timings,
                        &training_runtime,
                        &diagnostics,
                        RunExecutionOutcome::InfraFailure,
                        &error,
                        FailureSnapshotCaptureTiming::NoPanic,
                        Some(&model),
                    );
                    record_species_run_artifact(artifact);
                    return Err(error);
                }
                diagnostics.record(
                    FailureDiagnosticBoundary::BestWeightExportComplete,
                    RunPhase::Train,
                    &launch_runtime.artifact(&config.launch_policy),
                    Some(completed_step),
                );
            }
        }

        if should_log_training_checkpoint(completed_step, launch_runtime.planned_steps) {
            log_species_phase_progress(
                species,
                "train",
                completed_step,
                launch_runtime.planned_steps,
                train_started.elapsed(),
            );
        }
    }
    let train_elapsed = train_started.elapsed();
    phase_timings.push(phase_timing(
        RunPhase::Train,
        train_elapsed,
        launch_runtime.completed_steps,
        launch_runtime.planned_steps,
    ));
    log_species_phase_done(species, "train", train_elapsed);
    diagnostics.record(
        FailureDiagnosticBoundary::StabilityPhaseStarted,
        RunPhase::Stability,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );

    log_species_phase_start(
        species,
        "stability",
        &[format!("depth={}", config.stability_depth)],
    );
    let stability_started = Instant::now();
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            let elapsed = stability_started.elapsed();
            phase_timings.push(phase_timing(RunPhase::Stability, elapsed, 0, 1));
            let error = FractalError::InvalidState("run timeout exceeded during stability".into());
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                timeout_outcome_for_phase(RunPhase::Stability),
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
    }
    let stability_batch = match batches.eval_sentence.first() {
        Some(batch) => batch,
        None => {
            phase_timings.push(phase_timing(
                RunPhase::Stability,
                stability_started.elapsed(),
                0,
                1,
            ));
            let error = FractalError::InvalidState("stability batch cache was empty".into());
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                RunExecutionOutcome::InfraFailure,
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
    };
    let grad_norm_depth_20 =
        match evaluate_stability_score(&model, &criterion, stability_batch, config.stability_depth)
        {
            Ok(score) => score,
            Err(error) => {
                phase_timings.push(phase_timing(
                    RunPhase::Stability,
                    stability_started.elapsed(),
                    0,
                    1,
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
    let stability_elapsed = stability_started.elapsed();
    phase_timings.push(phase_timing(RunPhase::Stability, stability_elapsed, 1, 1));
    log_species_phase_done(species, "stability", stability_elapsed);
    diagnostics.record(
        FailureDiagnosticBoundary::StabilityPhaseComplete,
        RunPhase::Stability,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );
    diagnostics.record(
        FailureDiagnosticBoundary::PerplexityPhaseStarted,
        RunPhase::Perplexity,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );

    log_species_phase_start(
        species,
        "perplexity",
        &[format!(
            "batches={}",
            config.effective_perplexity_eval_batches()
        )],
    );
    let perplexity_started = Instant::now();
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            let elapsed = perplexity_started.elapsed();
            phase_timings.push(phase_timing(
                RunPhase::Perplexity,
                elapsed,
                0,
                batches.eval_sentence.len(),
            ));
            let error = FractalError::InvalidState("run timeout exceeded during perplexity".into());
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                timeout_outcome_for_phase(RunPhase::Perplexity),
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
    }
    let long_context_perplexity =
        match evaluate_perplexity(&model, &criterion, &batches.eval_sentence) {
            Ok(perplexity) => perplexity,
            Err(error) => {
                phase_timings.push(phase_timing(
                    RunPhase::Perplexity,
                    perplexity_started.elapsed(),
                    0,
                    batches.eval_sentence.len(),
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
    let perplexity_elapsed = perplexity_started.elapsed();
    phase_timings.push(phase_timing(
        RunPhase::Perplexity,
        perplexity_elapsed,
        batches.eval_sentence.len(),
        batches.eval_sentence.len(),
    ));
    log_species_phase_done(species, "perplexity", perplexity_elapsed);
    diagnostics.record(
        FailureDiagnosticBoundary::PerplexityPhaseComplete,
        RunPhase::Perplexity,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );
    diagnostics.record(
        FailureDiagnosticBoundary::ArcSpeedPhaseStarted,
        RunPhase::ArcSpeed,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );

    log_species_phase_start(
        species,
        "arc_speed",
        &[format!("batches={}", config.effective_arc_eval_batches())],
    );
    let accuracy_started = Instant::now();
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            let elapsed = accuracy_started.elapsed();
            phase_timings.push(phase_timing(
                RunPhase::ArcSpeed,
                elapsed,
                0,
                batches.eval_arc.len(),
            ));
            let error = FractalError::InvalidState(
                "run timeout exceeded during ARC/speed evaluation".into(),
            );
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                timeout_outcome_for_phase(RunPhase::ArcSpeed),
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
    }
    let (arc_accuracy, tokens_per_sec) =
        match evaluate_accuracy_and_speed(&model, &batches.eval_arc) {
            Ok(result) => result,
            Err(error) => {
                phase_timings.push(phase_timing(
                    RunPhase::ArcSpeed,
                    accuracy_started.elapsed(),
                    0,
                    batches.eval_arc.len(),
                ));
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
        };
    let accuracy_elapsed = accuracy_started.elapsed();
    phase_timings.push(phase_timing(
        RunPhase::ArcSpeed,
        accuracy_elapsed,
        batches.eval_arc.len(),
        batches.eval_arc.len(),
    ));
    log_species_phase_done(species, "arc_speed", accuracy_elapsed);
    diagnostics.record(
        FailureDiagnosticBoundary::ArcSpeedPhaseComplete,
        RunPhase::ArcSpeed,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );

    let metrics = SpeciesRawMetrics {
        species,
        grad_norm_depth_20,
        long_context_perplexity,
        arc_accuracy,
        tokens_per_sec,
    };
    let final_best_improved = match launch_runtime.best_perplexity {
        Some(best) => long_context_perplexity < best,
        None => true,
    };
    if final_best_improved {
        launch_runtime.best_perplexity = Some(long_context_perplexity);
        if let Err(error) = maybe_persist_best_checkpoint(
            &checkpoint_paths,
            &mut launch_runtime,
            &checkpoint_contract,
            &config.launch_policy,
            &model,
            &optimizer,
            Some(long_context_perplexity),
        ) {
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                RunExecutionOutcome::InfraFailure,
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
        diagnostics.record(
            FailureDiagnosticBoundary::BestCheckpointPersisted,
            RunPhase::Train,
            &launch_runtime.artifact(&config.launch_policy),
            Some(launch_runtime.completed_steps),
        );
        if config
            .launch_policy
            .weight_export
            .phases
            .contains(&WeightExportPhase::Best)
        {
            if let Err(error) = apply_weight_export_attempt(
                &mut launch_runtime.weight_export,
                WeightExportPhase::Best,
                export_weight_phase(
                    stage.clone(),
                    &manifest,
                    &config.launch_policy.weight_export,
                    WeightExportPhase::Best,
                    &model,
                    &weight_export_paths,
                    config.launch_policy.weight_export.required,
                ),
            ) {
                let training_runtime = launch_runtime.artifact(&config.launch_policy);
                let artifact = build_failure_species_artifact(
                    stage,
                    &manifest,
                    phase_timings,
                    &training_runtime,
                    &diagnostics,
                    RunExecutionOutcome::InfraFailure,
                    &error,
                    FailureSnapshotCaptureTiming::NoPanic,
                    Some(&model),
                );
                record_species_run_artifact(artifact);
                return Err(error);
            }
            diagnostics.record(
                FailureDiagnosticBoundary::BestWeightExportComplete,
                RunPhase::Train,
                &launch_runtime.artifact(&config.launch_policy),
                Some(launch_runtime.completed_steps),
            );
        }
    }
    if let Err(error) = maybe_persist_final_checkpoint(
        &checkpoint_paths,
        &mut launch_runtime,
        &checkpoint_contract,
        &config.launch_policy,
        &model,
        &optimizer,
        Some(long_context_perplexity),
    ) {
        let training_runtime = launch_runtime.artifact(&config.launch_policy);
        let artifact = build_failure_species_artifact(
            stage,
            &manifest,
            phase_timings,
            &training_runtime,
            &diagnostics,
            RunExecutionOutcome::InfraFailure,
            &error,
            FailureSnapshotCaptureTiming::NoPanic,
            Some(&model),
        );
        record_species_run_artifact(artifact);
        return Err(error);
    }
    diagnostics.record(
        FailureDiagnosticBoundary::FinalCheckpointPersisted,
        RunPhase::Train,
        &launch_runtime.artifact(&config.launch_policy),
        Some(launch_runtime.completed_steps),
    );
    if config
        .launch_policy
        .weight_export
        .phases
        .contains(&WeightExportPhase::Final)
    {
        if let Err(error) = apply_weight_export_attempt(
            &mut launch_runtime.weight_export,
            WeightExportPhase::Final,
            export_weight_phase(
                stage.clone(),
                &manifest,
                &config.launch_policy.weight_export,
                WeightExportPhase::Final,
                &model,
                &weight_export_paths,
                config.launch_policy.weight_export.required,
            ),
        ) {
            let training_runtime = launch_runtime.artifact(&config.launch_policy);
            let artifact = build_failure_species_artifact(
                stage,
                &manifest,
                phase_timings,
                &training_runtime,
                &diagnostics,
                RunExecutionOutcome::InfraFailure,
                &error,
                FailureSnapshotCaptureTiming::NoPanic,
                Some(&model),
            );
            record_species_run_artifact(artifact);
            return Err(error);
        }
        diagnostics.record(
            FailureDiagnosticBoundary::FinalWeightExportComplete,
            RunPhase::Train,
            &launch_runtime.artifact(&config.launch_policy),
            Some(launch_runtime.completed_steps),
        );
    }
    let quality_outcome = classify_quality_outcome(&metrics);
    diagnostics.record(
        FailureDiagnosticBoundary::RunComplete,
        RunPhase::ArcSpeed,
        &launch_runtime.artifact(&config.launch_policy),
        None,
    );
    let artifact = SpeciesRunArtifact {
        stage,
        manifest,
        phase_timings,
        training_runtime: launch_runtime.artifact(&config.launch_policy),
        execution_outcome: RunExecutionOutcome::Success,
        quality_outcome,
        error: None,
        metrics: Some(metrics.clone()),
    };
    record_species_run_artifact(artifact);

    Ok(metrics)
}

fn prepare_batches_for_run<B: AutodiffBackend>(
    generator: &SimpleHierarchicalGenerator,
    config: &TournamentConfig,
    device: &B::Device,
) -> Result<TrainingBatchSet<B>, FractalError> {
    Ok(TrainingBatchSet {
        train_sentence: generator.train_batches_for::<B>(
            TaskFamily::RecursiveSentence,
            config.train_batch_size,
            device,
        )?,
        train_arc: Some(generator.train_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.train_batch_size,
            device,
        )?),
        eval_sentence: generator.eval_batches_for::<B>(
            TaskFamily::RecursiveSentence,
            config.eval_batch_size,
            config.effective_perplexity_eval_batches(),
            device,
        )?,
        eval_arc: generator.eval_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.eval_batch_size,
            config.effective_arc_eval_batches(),
            device,
        )?,
    })
}

fn prepare_candle_batches_for_run(
    generator: &SimpleHierarchicalGenerator,
    config: &TournamentConfig,
    device: &CandleDevice,
) -> Result<TrainingBatchSet<CpuTrainBackend>, FractalError> {
    let staging_device = CandleDevice::Cpu;
    let move_batches = |batches: Vec<TokenBatch<CpuTrainBackend>>| {
        batches
            .into_iter()
            .map(|batch| batch.to_device(device))
            .collect::<Vec<_>>()
    };

    Ok(TrainingBatchSet {
        train_sentence: move_batches(generator.train_batches_for::<CpuTrainBackend>(
            TaskFamily::RecursiveSentence,
            config.train_batch_size,
            &staging_device,
        )?),
        train_arc: Some(move_batches(
            generator.train_batches_for::<CpuTrainBackend>(
                TaskFamily::ArcGrid,
                config.train_batch_size,
                &staging_device,
            )?,
        )),
        eval_sentence: move_batches(generator.eval_batches_for::<CpuTrainBackend>(
            TaskFamily::RecursiveSentence,
            config.eval_batch_size,
            config.effective_perplexity_eval_batches(),
            &staging_device,
        )?),
        eval_arc: move_batches(generator.eval_batches_for::<CpuTrainBackend>(
            TaskFamily::ArcGrid,
            config.eval_batch_size,
            config.effective_arc_eval_batches(),
            &staging_device,
        )?),
    })
}

fn evaluate_perplexity<B, R>(
    model: &FractalModel<B, R>,
    criterion: &CrossEntropyLoss<B>,
    batches: &[TokenBatch<B>],
) -> Result<f64, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + Clone + std::fmt::Debug,
{
    let mut total_loss = 0.0f64;
    for batch in batches {
        let loss = model.loss(batch, criterion, None, true, None)?;
        total_loss += loss.into_scalar().elem::<f64>();
    }
    let mean_loss = total_loss / batches.len() as f64;
    Ok(mean_loss.exp())
}

fn evaluate_accuracy_and_speed<B, R>(
    model: &FractalModel<B, R>,
    batches: &[TokenBatch<B>],
) -> Result<(f64, f64), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + Clone + std::fmt::Debug,
{
    let mut correct = 0usize;
    let mut total = 0usize;
    let start = Instant::now();

    for batch in batches {
        let logits = model.forward_tokens(batch.input_ids.clone())?;
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let flat_logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let flat_targets = batch.target_ids.clone().reshape([batch_size * seq_len]);
        let predictions = flat_logits.argmax(1).reshape([batch_size * seq_len]);
        let valid_mask = flat_targets
            .clone()
            .equal_elem((PAD_TOKEN as i64).elem::<B::IntElem>())
            .bool_not();
        let correct_mask = predictions.equal(flat_targets).bool_and(valid_mask.clone());

        correct += correct_mask.int().sum().into_scalar().elem::<i64>() as usize;
        total += valid_mask.int().sum().into_scalar().elem::<i64>() as usize;
    }

    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };
    let tokens_per_sec = total as f64 / elapsed;

    Ok((accuracy, tokens_per_sec))
}

const TRAIN_PROGRESS_TARGET_EVENTS: usize = 4;

pub(crate) fn should_log_training_checkpoint(completed_step: usize, total_steps: usize) -> bool {
    if total_steps == 0 || completed_step == 0 {
        return false;
    }

    if completed_step == total_steps {
        return total_steps > 0 && completed_step == total_steps;
    }

    completed_step.is_multiple_of(training_progress_interval(total_steps))
}

pub(crate) fn training_progress_interval(total_steps: usize) -> usize {
    total_steps.max(1).div_ceil(TRAIN_PROGRESS_TARGET_EVENTS)
}

fn should_fire_debug_probe(step: usize, interval: Option<usize>) -> bool {
    interval.is_some_and(|interval| step.is_multiple_of(interval))
}

fn cuda_memory_snapshot() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_owned();
    let mut values = line.split(',').map(|value| value.trim());
    let used_mib = values.next()?;
    let free_mib = values.next()?;
    let total_mib = values.next()?;
    Some(format!(
        "cuda_mem_mib(used={used_mib},free={free_mib},total={total_mib})"
    ))
}

fn log_species_phase_start(species: SpeciesId, phase: &str, details: &[String]) {
    let suffix = if details.is_empty() {
        String::new()
    } else {
        format!(" {}", details.join(" "))
    };
    println!("[phase:start] {species} {phase}{suffix}");
}

fn log_species_phase_progress(
    species: SpeciesId,
    phase: &str,
    completed: usize,
    total: usize,
    elapsed: Duration,
) {
    println!(
        "[phase:progress] {species} {phase} {completed}/{total} elapsed={:.1}s",
        elapsed.as_secs_f64()
    );
}

fn log_species_phase_done(species: SpeciesId, phase: &str, elapsed: Duration) {
    println!(
        "[phase:done] {species} {phase} elapsed={:.1}s",
        elapsed.as_secs_f64()
    );
}

fn log_species_debug_probe(species: SpeciesId, details: &[String]) {
    let suffix = if details.is_empty() {
        String::new()
    } else {
        format!(" {}", details.join(" "))
    };
    println!("[phase:debug] {species}{suffix}");
}

pub(crate) fn gradient_l2_norm<M, B>(module: &M, grads: &GradientsParams) -> f64
where
    M: Module<B>,
    B: AutodiffBackend,
{
    struct Collector<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f64,
        _marker: std::marker::PhantomData<B>,
    }

    impl<'a, B: AutodiffBackend> ModuleVisitor<B> for Collector<'a, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
                let value = (grad.clone() * grad).sum().into_scalar().elem::<f64>();
                self.sum_sq += value;
            }
        }

        fn visit_int<const D: usize>(&mut self, _param: &burn::module::Param<Tensor<B, D, Int>>) {}

        fn visit_bool<const D: usize>(&mut self, _param: &burn::module::Param<Tensor<B, D, Bool>>) {
        }
    }

    let mut collector = Collector::<B> {
        grads,
        sum_sq: 0.0,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut collector);
    collector.sum_sq.sqrt()
}

pub(crate) fn clip_gradients_global_norm<M, B>(
    module: &M,
    grads: &mut GradientsParams,
    max_norm: f64,
) where
    M: Module<B>,
    B: AutodiffBackend,
{
    let norm = gradient_l2_norm(module, grads);
    if norm == 0.0 || norm <= max_norm {
        return;
    }
    let scale = (max_norm / norm) as f32;

    struct Scaler<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _marker: std::marker::PhantomData<B>,
    }

    impl<'a, B: AutodiffBackend> ModuleVisitor<B> for Scaler<'a, B> {
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
            let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) else {
                return;
            };
            self.grads
                .register::<B::InnerBackend, D>(param.id, grad.mul_scalar(self.scale));
        }

        fn visit_int<const D: usize>(&mut self, _param: &Param<Tensor<B, D, Int>>) {}

        fn visit_bool<const D: usize>(&mut self, _param: &Param<Tensor<B, D, Bool>>) {}
    }

    let mut scaler = Scaler::<B> {
        grads,
        scale,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut scaler);
}

fn training_token_budget<B: AutodiffBackend>(
    batches: &TrainingBatchSet<B>,
    train_steps: usize,
) -> usize {
    if train_steps == 0 {
        return 0;
    }

    (0..train_steps)
        .map(|step| {
            let train_batches = batches.train_batches_for_step(step);
            let batch = &train_batches[step % train_batches.len()];
            batch.token_count
        })
        .sum()
}

fn build_training_execution_plan<B: AutodiffBackend>(
    batches: &TrainingBatchSet<B>,
    config: &TournamentConfig,
) -> Result<TrainingExecutionPlan, FractalError> {
    if let Some(target_train_tokens) = config.train_token_budget {
        let mut planned_steps = 0usize;
        let mut tokens = 0usize;
        while tokens < target_train_tokens {
            let train_batches = batches.train_batches_for_step(planned_steps);
            if train_batches.is_empty() {
                return Err(FractalError::InvalidState(
                    "training batch cache was empty while computing token budget".into(),
                ));
            }
            let batch = &train_batches[planned_steps % train_batches.len()];
            if batch.token_count == 0 {
                return Err(FractalError::InvalidState(
                    "training batch token count must be greater than zero".into(),
                ));
            }
            tokens += batch.token_count;
            planned_steps += 1;
        }
        return Ok(TrainingExecutionPlan {
            planned_steps,
            target_train_tokens,
        });
    }

    Ok(TrainingExecutionPlan {
        planned_steps: config.train_steps_per_species,
        target_train_tokens: training_token_budget(batches, config.train_steps_per_species),
    })
}

fn advance_token_milestone(
    next_token: &mut Option<usize>,
    interval_tokens: Option<usize>,
    train_tokens_seen: usize,
) -> bool {
    let Some(interval_tokens) = interval_tokens else {
        return false;
    };
    let Some(mut current_next) = *next_token else {
        return false;
    };
    let mut triggered = false;
    while train_tokens_seen >= current_next {
        triggered = true;
        current_next = current_next.saturating_add(interval_tokens);
    }
    *next_token = Some(current_next);
    triggered
}

fn evaluate_stability_score<B, R>(
    model: &FractalModel<B, R>,
    criterion: &CrossEntropyLoss<B>,
    batch: &TokenBatch<B>,
    stability_depth: usize,
) -> Result<f64, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let stability_loss = model.loss(batch, criterion, Some(stability_depth), false, None)?;
    let stability_grads = GradientsParams::from_grads(stability_loss.backward(), model);
    Ok(gradient_l2_norm(model, &stability_grads))
}

fn recorder_error(error: burn::record::RecorderError) -> FractalError {
    FractalError::InvalidState(error.to_string())
}

fn backend_name(backend: &ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::CpuCandle => "cpu-candle",
        #[cfg(feature = "cuda")]
        ComputeBackend::CudaCandle { .. } => "cuda-candle",
        ComputeBackend::MetalWgpu { .. } => "metal-wgpu",
    }
}

fn resolve_checkpoint_paths(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
) -> CheckpointRuntimePaths {
    let root = if let Some(root) = std::env::var_os("FRACTAL_RUN_CHECKPOINT_DIR") {
        PathBuf::from(root).join(stage.species.as_str())
    } else if let Some(root) = std::env::var_os("FRACTAL_RUN_ARTIFACT_DIR") {
        PathBuf::from(root)
            .join("checkpoints")
            .join(stage.species.as_str())
    } else {
        let run_id = manifest
            .experiment
            .as_ref()
            .map(|experiment| experiment.experiment_id.run_id.clone())
            .or_else(|| std::env::var("FRACTAL_RUN_ID").ok())
            .unwrap_or_else(|| format!("run-{}", stage.species.as_str()));
        PathBuf::from(".fractal-run-results")
            .join(run_id)
            .join("checkpoints")
            .join(stage.species.as_str())
    };

    CheckpointRuntimePaths {
        latest: root.join("latest"),
        previous: root.join("previous"),
        best: root.join("best"),
        final_slot: root.join("final"),
        root,
    }
}

fn checkpoint_state_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join("state.json")
}

fn checkpoint_model_stem(slot_dir: &Path) -> PathBuf {
    slot_dir.join("model")
}

fn checkpoint_optimizer_stem(slot_dir: &Path) -> PathBuf {
    slot_dir.join("optimizer")
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' => ch,
            _ => '_',
        })
        .collect()
}

fn resolve_weight_identity_prefix(manifest: &RunManifest, stage: &SpeciesRunStage) -> PathBuf {
    let Some(experiment) = manifest.experiment.as_ref() else {
        return PathBuf::from(sanitize_path_component(
            &std::env::var("FRACTAL_RUN_ID")
                .unwrap_or_else(|_| format!("run-{}", stage.species.as_str())),
        ));
    };

    let logical_name = experiment.experiment_id.logical_name.trim();
    let run_id = experiment.experiment_id.run_id.trim();
    if logical_name.is_empty() || run_id.is_empty() {
        return PathBuf::from(format!("run-{}", stage.species.as_str()));
    }

    PathBuf::from(sanitize_path_component(logical_name)).join(sanitize_path_component(run_id))
}

pub(crate) fn resolve_weight_export_paths(
    stage: &SpeciesRunStage,
    manifest: &RunManifest,
) -> WeightExportRuntimePaths {
    let export_prefix = resolve_weight_identity_prefix(manifest, stage);
    let root = if let Some(root) = std::env::var_os("FRACTAL_RUN_EXPORT_DIR") {
        PathBuf::from(root)
            .join(export_prefix)
            .join(stage.species.as_str())
    } else if let Some(root) = std::env::var_os("FRACTAL_RUN_ARTIFACT_DIR") {
        PathBuf::from(root)
            .join("exports")
            .join(export_prefix)
            .join(stage.species.as_str())
    } else {
        PathBuf::from(".fractal-run-results")
            .join(export_prefix)
            .join("exports")
            .join(stage.species.as_str())
    };

    WeightExportRuntimePaths { root }
}

fn resolve_failure_snapshot_paths(
    stage: &SpeciesRunStage,
    manifest: &RunManifest,
) -> FailureSnapshotRuntimePaths {
    let snapshot_prefix = resolve_weight_identity_prefix(manifest, stage);
    let root = if let Some(root) = std::env::var_os("FRACTAL_RUN_FAILURE_SNAPSHOT_DIR") {
        PathBuf::from(root)
            .join(snapshot_prefix)
            .join(stage.species.as_str())
    } else if let Some(root) = std::env::var_os("FRACTAL_RUN_ARTIFACT_DIR") {
        PathBuf::from(root)
            .join("failure-snapshots")
            .join(snapshot_prefix)
            .join(stage.species.as_str())
    } else {
        PathBuf::from(".fractal-run-results")
            .join(snapshot_prefix)
            .join("failure-snapshots")
            .join(stage.species.as_str())
    };

    FailureSnapshotRuntimePaths { root }
}

fn weight_export_slot_dir(
    paths: &WeightExportRuntimePaths,
    format: &WeightExportFormat,
    phase: WeightExportPhase,
) -> PathBuf {
    paths.root.join(format.as_str()).join(phase.as_str())
}

fn weight_export_weights_stem(slot_dir: &Path) -> PathBuf {
    slot_dir.join("weights")
}

fn weight_export_metadata_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join("metadata.json")
}

fn failure_snapshot_metadata_path(paths: &FailureSnapshotRuntimePaths) -> PathBuf {
    paths.root.join("metadata.json")
}

fn failure_snapshot_runtime_state_path(paths: &FailureSnapshotRuntimePaths) -> PathBuf {
    paths.root.join("runtime-state.json")
}

fn failure_snapshot_diagnostics_tail_path(paths: &FailureSnapshotRuntimePaths) -> PathBuf {
    paths.root.join("diagnostics-tail.json")
}

fn failure_snapshot_model_weights_stem(paths: &FailureSnapshotRuntimePaths) -> PathBuf {
    paths.root.join("model-weights").join("weights")
}

fn clear_checkpoint_root(root: &Path) -> Result<(), FractalError> {
    if root.exists() {
        fs::remove_dir_all(root).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to clear checkpoint root {}: {error}",
                root.display()
            ))
        })?;
    }
    Ok(())
}

fn write_checkpoint_state(path: &Path, state: &RuntimeCheckpointState) -> Result<(), FractalError> {
    fs::write(
        path,
        serde_json::to_vec_pretty(state)
            .map_err(|error| FractalError::InvalidState(error.to_string()))?,
    )
    .map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write checkpoint state {}: {error}",
            path.display()
        ))
    })
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> Result<(), FractalError> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value)
            .map_err(|error| FractalError::InvalidState(error.to_string()))?,
    )
    .map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write json artifact {}: {error}",
            path.display()
        ))
    })
}

fn read_checkpoint_state(path: &Path) -> Result<RuntimeCheckpointState, FractalError> {
    let bytes = fs::read(path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to read checkpoint state {}: {error}",
            path.display()
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to parse checkpoint state {}: {error}",
            path.display()
        ))
    })
}

fn upsert_checkpoint_artifact(
    checkpoints: &mut Vec<CheckpointArtifact>,
    artifact: CheckpointArtifact,
) {
    if let Some(slot) = checkpoints
        .iter_mut()
        .find(|existing| existing.kind == artifact.kind)
    {
        *slot = artifact;
    } else {
        checkpoints.push(artifact);
    }
}

fn promote_latest_to_previous(
    runtime: &mut LaunchRuntimeState,
    paths: &CheckpointRuntimePaths,
) -> Result<(), FractalError> {
    if !paths.latest.exists() {
        return Ok(());
    }
    if paths.previous.exists() {
        fs::remove_dir_all(&paths.previous).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to remove previous checkpoint {}: {error}",
                paths.previous.display()
            ))
        })?;
    }
    fs::rename(&paths.latest, &paths.previous).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to rotate latest checkpoint {} -> {}: {error}",
            paths.latest.display(),
            paths.previous.display()
        ))
    })?;

    if let Some(latest) = runtime
        .checkpoints
        .iter()
        .find(|artifact| artifact.kind == CheckpointArtifactKind::Latest)
        .cloned()
    {
        upsert_checkpoint_artifact(
            &mut runtime.checkpoints,
            CheckpointArtifact {
                kind: CheckpointArtifactKind::Previous,
                directory: paths.previous.display().to_string(),
                ..latest
            },
        );
    }
    Ok(())
}

fn write_checkpoint_slot<B, R>(
    slot_dir: &Path,
    state: &RuntimeCheckpointState,
    model: &FractalModel<B, R>,
    optimizer: &ConfiguredOptimizer<FractalModel<B, R>, B>,
) -> Result<(), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    if slot_dir.exists() {
        fs::remove_dir_all(slot_dir).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to clear checkpoint slot {}: {error}",
                slot_dir.display()
            ))
        })?;
    }
    fs::create_dir_all(slot_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create checkpoint slot {}: {error}",
            slot_dir.display()
        ))
    })?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(checkpoint_model_stem(slot_dir), &recorder)
        .map_err(recorder_error)?;
    optimizer.save_checkpoint_record(&checkpoint_optimizer_stem(slot_dir))?;
    write_checkpoint_state(&checkpoint_state_path(slot_dir), state)?;
    Ok(())
}

type RestoredCheckpoint<B, R> = (
    FractalModel<B, R>,
    ConfiguredOptimizer<FractalModel<B, R>, B>,
    LaunchRuntimeState,
    CheckpointRuntimePaths,
);

fn maybe_restore_checkpoint<B, R>(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    plan: &TrainingExecutionPlan,
    model: FractalModel<B, R>,
    optimizer: ConfiguredOptimizer<FractalModel<B, R>, B>,
    device: &B::Device,
) -> Result<RestoredCheckpoint<B, R>, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let paths = resolve_checkpoint_paths(stage.clone(), manifest);
    let resume_policy = &manifest.config.launch_policy.resume;
    let contract = CheckpointContractSnapshot::from_manifest(stage.clone(), manifest);

    if !resume_policy.resume_on_interrupt {
        return Ok((
            model,
            optimizer,
            LaunchRuntimeState::fresh(plan, &manifest.config.launch_policy),
            paths,
        ));
    }

    let latest_state_path = checkpoint_state_path(&paths.latest);
    if !latest_state_path.exists() {
        return Ok((
            model,
            optimizer,
            LaunchRuntimeState::fresh(plan, &manifest.config.launch_policy),
            paths,
        ));
    }

    let checkpoint_state = match read_checkpoint_state(&latest_state_path) {
        Ok(state) => state,
        Err(error) if resume_policy.restart_on_corruption => {
            clear_checkpoint_root(&paths.root)?;
            return Ok((
                model,
                optimizer,
                LaunchRuntimeState::fresh(plan, &manifest.config.launch_policy),
                paths,
            ));
        }
        Err(error) => return Err(error),
    };

    if checkpoint_state.contract != contract {
        if resume_policy.restart_on_contract_ambiguity {
            clear_checkpoint_root(&paths.root)?;
            return Ok((
                model,
                optimizer,
                LaunchRuntimeState::fresh(plan, &manifest.config.launch_policy),
                paths,
            ));
        }
        return Err(FractalError::InvalidConfig(format!(
            "checkpoint contract mismatch for {}",
            latest_state_path.display()
        )));
    }

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = model
        .load_file(checkpoint_model_stem(&paths.latest), &recorder, device)
        .inspect_err(|_| {
            if resume_policy.restart_on_corruption {
                let _ = clear_checkpoint_root(&paths.root);
            }
        })
        .map_err(recorder_error)?;
    let optimizer = optimizer
        .load_checkpoint_record(&checkpoint_optimizer_stem(&paths.latest), device)
        .inspect_err(|_| {
            if resume_policy.restart_on_corruption {
                let _ = clear_checkpoint_root(&paths.root);
            }
        })?;
    Ok((
        model,
        optimizer,
        LaunchRuntimeState::from_checkpoint(checkpoint_state, &manifest.config.launch_policy),
        paths,
    ))
}

fn persist_checkpoint_alias<B, R>(
    kind: CheckpointArtifactKind,
    slot_dir: &Path,
    runtime: &mut LaunchRuntimeState,
    contract: &CheckpointContractSnapshot,
    model: &FractalModel<B, R>,
    optimizer: &ConfiguredOptimizer<FractalModel<B, R>, B>,
    long_context_perplexity: Option<f64>,
) -> Result<(), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let state = runtime.checkpoint_state(contract.clone());
    write_checkpoint_slot(slot_dir, &state, model, optimizer)?;
    upsert_checkpoint_artifact(
        &mut runtime.checkpoints,
        CheckpointArtifact {
            kind,
            tokens_seen: runtime.train_tokens_seen,
            completed_steps: runtime.completed_steps,
            directory: slot_dir.display().to_string(),
            long_context_perplexity,
        },
    );
    Ok(())
}

fn build_weight_export_contract(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    format: &WeightExportFormat,
) -> Result<WeightExportContract, FractalError> {
    let experiment = manifest.experiment.as_ref().ok_or_else(|| {
        FractalError::InvalidConfig(
            "weight export requires a resolved experiment spec in the run manifest".into(),
        )
    })?;
    let commit_sha = experiment.experiment_id.commit_sha.clone().ok_or_else(|| {
        FractalError::InvalidConfig(
            "weight export requires the producing experiment commit_sha".into(),
        )
    })?;
    let contract = WeightExportContract {
        experiment_logical_name: experiment.experiment_id.logical_name.clone(),
        experiment_run_id: experiment.experiment_id.run_id.clone(),
        experiment_branch: experiment.experiment_id.branch.clone(),
        experiment_commit_sha: commit_sha,
        species: stage.species.as_str().to_owned(),
        variant_name: manifest.variant_name.as_str().to_owned(),
        model: experiment.model.clone(),
        vocab_size: manifest.config.vocab_size,
        precision: manifest.config.launch_policy.precision.clone(),
        format: format.clone(),
    };
    contract.validate_against_config(&manifest.config)?;
    Ok(contract)
}

fn build_failure_snapshot_contract(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    error_class: FailureSnapshotErrorClass,
    capture_timing: FailureSnapshotCaptureTiming,
    last_successful_boundary: Option<FailureDiagnosticBoundary>,
) -> Result<FailureSnapshotContract, FractalError> {
    let experiment = manifest.experiment.as_ref().ok_or_else(|| {
        FractalError::InvalidConfig(
            "failure snapshot requires a resolved experiment spec in the run manifest".into(),
        )
    })?;
    let commit_sha = experiment.experiment_id.commit_sha.clone().ok_or_else(|| {
        FractalError::InvalidConfig(
            "failure snapshot requires the producing experiment commit_sha".into(),
        )
    })?;
    let contract = FailureSnapshotContract {
        experiment_logical_name: experiment.experiment_id.logical_name.clone(),
        experiment_run_id: experiment.experiment_id.run_id.clone(),
        experiment_branch: experiment.experiment_id.branch.clone(),
        experiment_commit_sha: commit_sha,
        species: stage.species.as_str().to_owned(),
        variant_name: manifest.variant_name.as_str().to_owned(),
        model: experiment.model.clone(),
        vocab_size: manifest.config.vocab_size,
        precision: manifest.config.launch_policy.precision.clone(),
        error_class,
        capture_timing,
        last_successful_boundary,
    };
    contract.validate_against_config(&manifest.config)?;
    Ok(contract)
}

fn apply_weight_export_attempt(
    runtime: &mut WeightExportRuntimeState,
    phase: WeightExportPhase,
    result: Result<WeightExportArtifact, FractalError>,
) -> Result<(), FractalError> {
    match result {
        Ok(artifact) => {
            runtime.record_success(artifact);
            Ok(())
        }
        Err(error) => {
            runtime.record_failure(phase, error.to_string());
            if runtime.policy.required {
                Err(error)
            } else {
                Ok(())
            }
        }
    }
}

fn classify_failure_snapshot_error(
    execution_outcome: RunExecutionOutcome,
    error: &FractalError,
) -> FailureSnapshotErrorClass {
    match execution_outcome {
        RunExecutionOutcome::TrainTimeout => FailureSnapshotErrorClass::TrainTimeout,
        RunExecutionOutcome::EvalConstrained => FailureSnapshotErrorClass::EvalConstrained,
        RunExecutionOutcome::InfraFailure | RunExecutionOutcome::Success => match error {
            FractalError::InvalidConfig(_) => FailureSnapshotErrorClass::InvalidConfig,
            FractalError::InvalidState(_) => FailureSnapshotErrorClass::InvalidState,
            FractalError::Shape(_) => FailureSnapshotErrorClass::Shape,
        },
    }
}

fn mark_failure_snapshot_root_error(state: &mut FailureSnapshotRuntimeState, error: &FractalError) {
    for kind in state.policy.requested_artifact_kinds() {
        state.record_failure(kind, error.to_string());
    }
}

fn persist_failure_snapshot_payload<T: Serialize>(
    state: &mut FailureSnapshotRuntimeState,
    kind: FailureSnapshotArtifactKind,
    path: &Path,
    value: &T,
) {
    let result = write_pretty_json(path, value).map(|_| FailureSnapshotArtifact {
        kind,
        path: path.display().to_string(),
    });
    match result {
        Ok(artifact) => state.record_success(artifact),
        Err(error) => state.record_failure(kind, error.to_string()),
    }
}

fn finalize_failure_snapshot_metadata(
    state: &mut FailureSnapshotRuntimeState,
    metadata_path: &Path,
) {
    state.record_success(FailureSnapshotArtifact {
        kind: FailureSnapshotArtifactKind::Metadata,
        path: metadata_path.display().to_string(),
    });
    if let Err(error) = write_pretty_json(metadata_path, state) {
        state.record_failure(FailureSnapshotArtifactKind::Metadata, error.to_string());
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_failure_snapshot<B, R>(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    training_runtime: &TrainingRuntimeArtifact,
    diagnostics: &FailureDiagnosticsRecorder,
    execution_outcome: RunExecutionOutcome,
    error: &FractalError,
    capture_timing: FailureSnapshotCaptureTiming,
    model: Option<&FractalModel<B, R>>,
) -> FailureSnapshotRuntimeState
where
    B: Backend,
    R: FractalRule<B> + Module<B> + ModuleDisplay + Clone + std::fmt::Debug,
{
    let policy = manifest.config.launch_policy.failure_snapshot.clone();
    let mut state = FailureSnapshotRuntimeState::from_policy(policy.clone());
    if !policy.enabled {
        return state;
    }

    let error_class = classify_failure_snapshot_error(execution_outcome, error);
    let contract = match build_failure_snapshot_contract(
        stage.clone(),
        manifest,
        error_class,
        capture_timing,
        diagnostics.last_boundary(),
    ) {
        Ok(contract) => contract,
        Err(contract_error) => {
            state.mark_attempted();
            state.record_failure(
                FailureSnapshotArtifactKind::Metadata,
                contract_error.to_string(),
            );
            return state;
        }
    };
    state.begin_capture(contract);

    let paths = resolve_failure_snapshot_paths(&stage, manifest);
    if let Err(error) = fs::create_dir_all(&paths.root).map_err(|io_error| {
        FractalError::InvalidState(format!(
            "failed to create failure snapshot root {}: {io_error}",
            paths.root.display()
        ))
    }) {
        mark_failure_snapshot_root_error(&mut state, &error);
        return state;
    }

    if policy.capture_runtime_state {
        persist_failure_snapshot_payload(
            &mut state,
            FailureSnapshotArtifactKind::RuntimeState,
            &failure_snapshot_runtime_state_path(&paths),
            training_runtime,
        );
    }
    if policy.capture_diagnostics_tail {
        let diagnostics_tail = diagnostics.snapshot();
        persist_failure_snapshot_payload(
            &mut state,
            FailureSnapshotArtifactKind::DiagnosticsTail,
            &failure_snapshot_diagnostics_tail_path(&paths),
            &diagnostics_tail,
        );
    }
    if policy.capture_model_weights {
        match model {
            Some(model) => {
                let model_weights_path = failure_snapshot_model_weights_stem(&paths);
                if let Some(parent) = model_weights_path.parent() {
                    if let Err(error) = fs::create_dir_all(parent).map_err(|io_error| {
                        FractalError::InvalidState(format!(
                            "failed to create failure snapshot model-weight root {}: {io_error}",
                            parent.display()
                        ))
                    }) {
                        state.record_failure(
                            FailureSnapshotArtifactKind::ModelWeights,
                            error.to_string(),
                        );
                    } else {
                        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
                        match model
                            .clone()
                            .save_file(model_weights_path.clone(), &recorder)
                            .map_err(recorder_error)
                        {
                            Ok(_) => state.record_success(FailureSnapshotArtifact {
                                kind: FailureSnapshotArtifactKind::ModelWeights,
                                path: model_weights_path.display().to_string(),
                            }),
                            Err(error) => state.record_failure(
                                FailureSnapshotArtifactKind::ModelWeights,
                                error.to_string(),
                            ),
                        }
                    }
                }
            }
            None => state.record_failure(
                FailureSnapshotArtifactKind::ModelWeights,
                "model weights were unavailable for failure snapshot capture".to_owned(),
            ),
        }
    }

    finalize_failure_snapshot_metadata(&mut state, &failure_snapshot_metadata_path(&paths));
    state
}

fn capture_failure_snapshot_without_model(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    training_runtime: &TrainingRuntimeArtifact,
    diagnostics: &FailureDiagnosticsRecorder,
    execution_outcome: RunExecutionOutcome,
    error: &FractalError,
    capture_timing: FailureSnapshotCaptureTiming,
) -> FailureSnapshotRuntimeState {
    let policy = manifest.config.launch_policy.failure_snapshot.clone();
    let mut state = FailureSnapshotRuntimeState::from_policy(policy.clone());
    if !policy.enabled {
        return state;
    }

    let error_class = classify_failure_snapshot_error(execution_outcome, error);
    let contract = match build_failure_snapshot_contract(
        stage.clone(),
        manifest,
        error_class,
        capture_timing,
        diagnostics.last_boundary(),
    ) {
        Ok(contract) => contract,
        Err(contract_error) => {
            state.mark_attempted();
            state.record_failure(
                FailureSnapshotArtifactKind::Metadata,
                contract_error.to_string(),
            );
            return state;
        }
    };
    state.begin_capture(contract);

    let paths = resolve_failure_snapshot_paths(&stage, manifest);
    if let Err(error) = fs::create_dir_all(&paths.root).map_err(|io_error| {
        FractalError::InvalidState(format!(
            "failed to create failure snapshot root {}: {io_error}",
            paths.root.display()
        ))
    }) {
        mark_failure_snapshot_root_error(&mut state, &error);
        return state;
    }

    if policy.capture_runtime_state {
        persist_failure_snapshot_payload(
            &mut state,
            FailureSnapshotArtifactKind::RuntimeState,
            &failure_snapshot_runtime_state_path(&paths),
            training_runtime,
        );
    }
    if policy.capture_diagnostics_tail {
        let diagnostics_tail = diagnostics.snapshot();
        persist_failure_snapshot_payload(
            &mut state,
            FailureSnapshotArtifactKind::DiagnosticsTail,
            &failure_snapshot_diagnostics_tail_path(&paths),
            &diagnostics_tail,
        );
    }
    if policy.capture_model_weights {
        state.record_failure(
            FailureSnapshotArtifactKind::ModelWeights,
            "model weights were unavailable for failure snapshot capture".to_owned(),
        );
    }

    finalize_failure_snapshot_metadata(&mut state, &failure_snapshot_metadata_path(&paths));
    state
}

fn attach_failure_snapshot(
    training_runtime: &mut TrainingRuntimeArtifact,
    snapshot: FailureSnapshotRuntimeState,
) {
    training_runtime.failure_snapshot = snapshot;
}

pub(crate) fn export_weight_phase<B, R>(
    stage: SpeciesRunStage,
    manifest: &RunManifest,
    policy: &WeightExportPolicy,
    phase: WeightExportPhase,
    model: &FractalModel<B, R>,
    paths: &WeightExportRuntimePaths,
    required: bool,
) -> Result<WeightExportArtifact, FractalError>
where
    B: Backend,
    R: FractalRule<B> + Module<B> + ModuleDisplay + Clone + std::fmt::Debug,
{
    if !policy.phases.contains(&phase) {
        return Err(FractalError::InvalidConfig(format!(
            "weight export policy does not request phase {}",
            phase.as_str()
        )));
    }
    policy.validate()?;
    policy.format.validate_supported()?;
    phase.validate_supported()?;
    let contract = build_weight_export_contract(stage, manifest, &policy.format)?;
    let slot_dir = weight_export_slot_dir(paths, &policy.format, phase);
    if slot_dir.exists() {
        fs::remove_dir_all(&slot_dir).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to clear weight export slot {}: {error}",
                slot_dir.display()
            ))
        })?;
    }
    fs::create_dir_all(&slot_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create weight export slot {}: {error}",
            slot_dir.display()
        ))
    })?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let weights_path = weight_export_weights_stem(&slot_dir);
    model
        .clone()
        .save_file(weights_path.clone(), &recorder)
        .map_err(recorder_error)?;
    let metadata_path = weight_export_metadata_path(&slot_dir);
    let artifact = WeightExportArtifact {
        format: policy.format.clone(),
        phase,
        path: weights_path.display().to_string(),
        metadata_path: metadata_path.display().to_string(),
        required,
        contract,
    };
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&artifact)
            .map_err(|error| FractalError::InvalidState(error.to_string()))?,
    )
    .map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write weight export metadata {}: {error}",
            metadata_path.display()
        ))
    })?;
    artifact.validate()?;
    Ok(artifact)
}

pub fn read_weight_export_metadata(path: &Path) -> Result<WeightExportArtifact, FractalError> {
    let bytes = fs::read(path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to read weight export metadata {}: {error}",
            path.display()
        ))
    })?;
    let artifact: WeightExportArtifact = serde_json::from_slice(&bytes).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to parse weight export metadata {}: {error}",
            path.display()
        ))
    })?;
    artifact.validate()?;
    Ok(artifact)
}

pub fn load_weight_export_artifact<B, R>(
    artifact: &WeightExportArtifact,
    model: FractalModel<B, R>,
    config: &TournamentConfig,
    device: &B::Device,
) -> Result<FractalModel<B, R>, FractalError>
where
    B: Backend,
    R: FractalRule<B> + Module<B> + ModuleDisplay + Clone + std::fmt::Debug,
{
    artifact.validate_against_config(config)?;
    match &artifact.format {
        WeightExportFormat::BurnBin => {
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            model
                .load_file(PathBuf::from(&artifact.path), &recorder, device)
                .map_err(recorder_error)
        }
        WeightExportFormat::SafeTensors => Err(FractalError::InvalidConfig(
            "safe-tensors weight export loading is not yet executable in the runtime".into(),
        )),
        WeightExportFormat::Quantized { precision } => Err(FractalError::InvalidConfig(format!(
            "quantized weight export loading for {} is not yet executable in the runtime",
            precision.as_str()
        ))),
    }
}

pub fn load_weight_export_metadata<B, R>(
    metadata_path: &Path,
    model: FractalModel<B, R>,
    config: &TournamentConfig,
    device: &B::Device,
) -> Result<FractalModel<B, R>, FractalError>
where
    B: Backend,
    R: FractalRule<B> + Module<B> + ModuleDisplay + Clone + std::fmt::Debug,
{
    let artifact = read_weight_export_metadata(metadata_path)?;
    load_weight_export_artifact(&artifact, model, config, device)
}

fn maybe_persist_latest_checkpoint<B, R>(
    paths: &CheckpointRuntimePaths,
    runtime: &mut LaunchRuntimeState,
    contract: &CheckpointContractSnapshot,
    launch_policy: &crate::LaunchPolicySpec,
    model: &FractalModel<B, R>,
    optimizer: &ConfiguredOptimizer<FractalModel<B, R>, B>,
    long_context_perplexity: Option<f64>,
) -> Result<(), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    if launch_policy.checkpoint.keep_previous {
        promote_latest_to_previous(runtime, paths)?;
    } else if paths.latest.exists() {
        fs::remove_dir_all(&paths.latest).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to clear latest checkpoint {}: {error}",
                paths.latest.display()
            ))
        })?;
    }

    if launch_policy.checkpoint.keep_latest {
        persist_checkpoint_alias(
            CheckpointArtifactKind::Latest,
            &paths.latest,
            runtime,
            contract,
            model,
            optimizer,
            long_context_perplexity,
        )?;
    }
    Ok(())
}

fn maybe_persist_best_checkpoint<B, R>(
    paths: &CheckpointRuntimePaths,
    runtime: &mut LaunchRuntimeState,
    contract: &CheckpointContractSnapshot,
    launch_policy: &crate::LaunchPolicySpec,
    model: &FractalModel<B, R>,
    optimizer: &ConfiguredOptimizer<FractalModel<B, R>, B>,
    long_context_perplexity: Option<f64>,
) -> Result<(), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    if !launch_policy.checkpoint.keep_best {
        return Ok(());
    }
    persist_checkpoint_alias(
        CheckpointArtifactKind::Best,
        &paths.best,
        runtime,
        contract,
        model,
        optimizer,
        long_context_perplexity,
    )
}

fn maybe_persist_final_checkpoint<B, R>(
    paths: &CheckpointRuntimePaths,
    runtime: &mut LaunchRuntimeState,
    contract: &CheckpointContractSnapshot,
    launch_policy: &crate::LaunchPolicySpec,
    model: &FractalModel<B, R>,
    optimizer: &ConfiguredOptimizer<FractalModel<B, R>, B>,
    long_context_perplexity: Option<f64>,
) -> Result<(), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    if !launch_policy.checkpoint.keep_final {
        return Ok(());
    }
    persist_checkpoint_alias(
        CheckpointArtifactKind::Final,
        &paths.final_slot,
        runtime,
        contract,
        model,
        optimizer,
        long_context_perplexity,
    )
}

fn maybe_capture_interim_eval<B, R>(
    species: SpeciesId,
    model: &FractalModel<B, R>,
    criterion: &CrossEntropyLoss<B>,
    batches: &TrainingBatchSet<B>,
    config: &TournamentConfig,
    runtime: &mut LaunchRuntimeState,
) -> Result<Option<InterimEvalSnapshot>, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + AutodiffModule<B> + ModuleDisplay + Clone + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let launch_policy = &config.launch_policy;
    let stability_due = advance_token_milestone(
        &mut runtime.next_stability_token,
        launch_policy.eval_cadence.stability_interval_tokens,
        runtime.train_tokens_seen,
    );
    let perplexity_due = advance_token_milestone(
        &mut runtime.next_perplexity_token,
        launch_policy.eval_cadence.perplexity_interval_tokens,
        runtime.train_tokens_seen,
    );
    let arc_due = advance_token_milestone(
        &mut runtime.next_arc_token,
        launch_policy.eval_cadence.arc_interval_tokens,
        runtime.train_tokens_seen,
    );
    let systems_speed_due = advance_token_milestone(
        &mut runtime.next_systems_speed_token,
        launch_policy.eval_cadence.systems_speed_interval_tokens,
        runtime.train_tokens_seen,
    );

    if !(stability_due || perplexity_due || arc_due || systems_speed_due) {
        return Ok(None);
    }

    let mut snapshot = InterimEvalSnapshot {
        tokens_seen: runtime.train_tokens_seen,
        completed_steps: runtime.completed_steps,
        stability_score: None,
        long_context_perplexity: None,
        arc_accuracy: None,
        tokens_per_sec: None,
    };

    if stability_due {
        let batch = batches
            .eval_sentence
            .first()
            .ok_or_else(|| FractalError::InvalidState("stability batch cache was empty".into()))?;
        snapshot.stability_score = Some(evaluate_stability_score(
            model,
            criterion,
            batch,
            config.stability_depth,
        )?);
        println!(
            "[phase:checkpoint] {species} stability tokens={} steps={} value={:.4}",
            snapshot.tokens_seen,
            snapshot.completed_steps,
            snapshot.stability_score.unwrap_or(0.0)
        );
    }

    if perplexity_due {
        snapshot.long_context_perplexity = Some(evaluate_perplexity(
            model,
            criterion,
            &batches.eval_sentence,
        )?);
        println!(
            "[phase:checkpoint] {species} perplexity tokens={} steps={} value={:.4}",
            snapshot.tokens_seen,
            snapshot.completed_steps,
            snapshot.long_context_perplexity.unwrap_or(0.0)
        );
    }

    if arc_due || systems_speed_due {
        let (arc_accuracy, tokens_per_sec) = evaluate_accuracy_and_speed(model, &batches.eval_arc)?;
        snapshot.arc_accuracy = Some(arc_accuracy);
        snapshot.tokens_per_sec = Some(tokens_per_sec);
        println!(
            "[phase:checkpoint] {species} arc_speed tokens={} steps={} arc={:.4} tok/s={:.2}",
            snapshot.tokens_seen, snapshot.completed_steps, arc_accuracy, tokens_per_sec
        );
    }

    runtime.interim_evaluations.push(snapshot.clone());
    Ok(Some(snapshot))
}
