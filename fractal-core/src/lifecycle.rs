use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

use crate::{
    data_generator::{
        GeneratorConfig, GeneratorDepthConfig, SimpleHierarchicalGenerator, MIN_SEQUENCE_LEN,
        MIN_VOCAB_SIZE, PAD_TOKEN,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    registry::{
        build_failure_artifact, build_success_artifact, classify_quality_outcome, phase_timing,
        resolve_precision_profile, take_last_species_run_artifact, ComputeBackend, ExecutionMode,
        PrimitiveVariantName, SpeciesDefinition, SpeciesId, SpeciesRunContext,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComparisonAuthority {
    Authoritative,
    Advisory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonContract {
    pub authority: ComparisonAuthority,
    pub requires_same_preset: bool,
    pub requires_same_runtime_surfaces: bool,
    pub requires_frozen_commit: bool,
    pub requires_same_backend: bool,
}

impl ComparisonContract {
    pub const fn authoritative_same_preset() -> Self {
        Self {
            authority: ComparisonAuthority::Authoritative,
            requires_same_preset: true,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: true,
            requires_same_backend: true,
        }
    }

    pub const fn advisory_mixed_preset() -> Self {
        Self {
            authority: ComparisonAuthority::Advisory,
            requires_same_preset: false,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: false,
            requires_same_backend: true,
        }
    }

    pub const fn label(&self) -> &'static str {
        match (self.authority, self.requires_same_preset) {
            (ComparisonAuthority::Authoritative, true) => "authoritative same-preset",
            (ComparisonAuthority::Authoritative, false) => "authoritative mixed-preset",
            (ComparisonAuthority::Advisory, true) => "advisory same-preset",
            (ComparisonAuthority::Advisory, false) => "advisory mixed-preset",
        }
    }

    pub const fn is_authoritative_same_preset(&self) -> bool {
        matches!(self.authority, ComparisonAuthority::Authoritative) && self.requires_same_preset
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LaneIntent {
    Benchmark,
    Bullpen,
    Validation,
    Winner,
}

impl LaneIntent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Benchmark => "benchmark",
            Self::Bullpen => "bullpen",
            Self::Validation => "validation",
            Self::Winner => "winner",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecisionIntent {
    Promote,
    Hold,
    Retire,
    Benchmark,
    Optimize,
}

impl DecisionIntent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Promote => "promote",
            Self::Hold => "hold",
            Self::Retire => "retire",
            Self::Benchmark => "benchmark",
            Self::Optimize => "optimize",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentId {
    pub logical_name: String,
    pub run_id: String,
    pub branch: Option<String>,
    pub commit_sha: Option<String>,
    pub created_at_unix_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentQuestion {
    pub summary: String,
    pub lane_intent: LaneIntent,
    pub decision_intent: DecisionIntent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantSpec {
    pub species: SpeciesId,
    pub variant_name: PrimitiveVariantName,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerKind {
    #[serde(rename = "adam")]
    Adam,
    #[serde(rename = "adamw", alias = "adam-w")]
    AdamW,
}

impl OptimizerKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::AdamW => "adamw",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TrainingInputMode {
    Synthetic,
    TokenizerBackedText,
}

impl TrainingInputMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::TokenizerBackedText => "tokenizer-backed-text",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LearningRateScheduleKind {
    Constant,
    WarmupCosine,
}

impl LearningRateScheduleKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Constant => "constant",
            Self::WarmupCosine => "warmup-cosine",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArcSourceMode {
    SyntheticCanonical,
    TokenizerBackedCanonical,
    Unavailable,
}

impl ArcSourceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SyntheticCanonical => "synthetic-canonical",
            Self::TokenizerBackedCanonical => "tokenizer-backed-canonical",
            Self::Unavailable => "unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelArchitectureKind {
    RecursiveKernelV1,
    SharedOuterScaffoldV1,
}

impl ModelArchitectureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RecursiveKernelV1 => "recursive-kernel-v1",
            Self::SharedOuterScaffoldV1 => "shared-outer-scaffold-v1",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelContractSpec {
    pub architecture: ModelArchitectureKind,
    pub hidden_dim: usize,
    pub max_recursion_depth: usize,
    pub router_enabled: bool,
}

impl ModelContractSpec {
    pub const fn recursive_kernel_v1(hidden_dim: usize, max_recursion_depth: usize) -> Self {
        Self {
            architecture: ModelArchitectureKind::RecursiveKernelV1,
            hidden_dim,
            max_recursion_depth,
            router_enabled: true,
        }
    }

    pub fn label(&self) -> String {
        format!(
            "{} hidden_dim={} max_recursion_depth={} router_enabled={}",
            self.architecture.as_str(),
            self.hidden_dim,
            self.max_recursion_depth,
            self.router_enabled
        )
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.hidden_dim == 0 {
            return Err(FractalError::InvalidConfig(
                "model hidden_dim must be greater than zero".into(),
            ));
        }
        if self.max_recursion_depth == 0 {
            return Err(FractalError::InvalidConfig(
                "model max_recursion_depth must be greater than zero".into(),
            ));
        }
        match self.architecture {
            ModelArchitectureKind::RecursiveKernelV1 => {
                if !self.router_enabled {
                    return Err(FractalError::InvalidConfig(
                        "recursive-kernel-v1 currently requires router_enabled=true".into(),
                    ));
                }
                Ok(())
            }
            ModelArchitectureKind::SharedOuterScaffoldV1 => Err(FractalError::InvalidConfig(
                "shared-outer-scaffold-v1 is not yet executable in the runtime".into(),
            )),
        }
    }

    pub fn validate_against_config(&self, config: &TournamentConfig) -> Result<(), FractalError> {
        self.validate()?;
        if self.hidden_dim != config.dim {
            return Err(FractalError::InvalidConfig(format!(
                "model hidden_dim {} must match config dim {}",
                self.hidden_dim, config.dim
            )));
        }
        if self.max_recursion_depth != config.max_recursion_depth {
            return Err(FractalError::InvalidConfig(format!(
                "model max_recursion_depth {} must match config max_recursion_depth {}",
                self.max_recursion_depth, config.max_recursion_depth
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "format", rename_all = "kebab-case")]
pub enum TextCorpusFormat {
    JsonlText { text_field: String },
    PlainTextLines,
}

impl TextCorpusFormat {
    pub fn validate(&self) -> Result<(), FractalError> {
        match self {
            Self::JsonlText { text_field } if text_field.trim().is_empty() => {
                Err(FractalError::InvalidConfig(
                    "text corpus jsonl text_field must be non-empty".into(),
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextCorpusSplitSpec {
    pub path: String,
    pub format: TextCorpusFormat,
    pub max_documents: Option<usize>,
}

impl TextCorpusSplitSpec {
    pub fn validate(&self, split_name: &str) -> Result<(), FractalError> {
        if self.path.trim().is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "text corpus {split_name} path must be non-empty"
            )));
        }
        if self.max_documents == Some(0) {
            return Err(FractalError::InvalidConfig(format!(
                "text corpus {split_name} max_documents must be greater than zero when configured"
            )));
        }
        self.format.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextCorpusSourceSpec {
    pub train: TextCorpusSplitSpec,
    pub eval: TextCorpusSplitSpec,
}

impl TextCorpusSourceSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.train.validate("train")?;
        self.eval.validate("eval")?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningRateScheduleSpec {
    pub kind: LearningRateScheduleKind,
    pub warmup_fraction: f64,
    pub decay_floor_fraction: f64,
}

impl Default for LearningRateScheduleSpec {
    fn default() -> Self {
        Self::constant()
    }
}

impl LearningRateScheduleSpec {
    pub const fn constant() -> Self {
        Self {
            kind: LearningRateScheduleKind::Constant,
            warmup_fraction: 0.0,
            decay_floor_fraction: 1.0,
        }
    }

    pub fn warmup_cosine(warmup_fraction: f64, decay_floor_fraction: f64) -> Self {
        Self {
            kind: LearningRateScheduleKind::WarmupCosine,
            warmup_fraction,
            decay_floor_fraction,
        }
    }

    pub fn label(&self) -> String {
        match self.kind {
            LearningRateScheduleKind::Constant => "constant".to_owned(),
            LearningRateScheduleKind::WarmupCosine => format!(
                "warmup_cosine(warmup_fraction={:.4}, decay_floor_fraction={:.4})",
                self.warmup_fraction, self.decay_floor_fraction
            ),
        }
    }

    pub fn learning_rate_at_tokens(
        &self,
        peak_learning_rate: f64,
        seen_tokens: usize,
        total_tokens: usize,
    ) -> f64 {
        match self.kind {
            LearningRateScheduleKind::Constant => peak_learning_rate,
            LearningRateScheduleKind::WarmupCosine => {
                let total_tokens = total_tokens.max(1) as f64;
                let progress = (seen_tokens as f64 / total_tokens).clamp(0.0, 1.0);
                let warmup_fraction = self.warmup_fraction.clamp(0.0, 1.0);
                let min_lr = peak_learning_rate * self.decay_floor_fraction.clamp(0.0, 1.0);

                if warmup_fraction > 0.0 && progress < warmup_fraction {
                    peak_learning_rate * (progress / warmup_fraction)
                } else if progress >= 1.0 {
                    min_lr
                } else {
                    let decay_progress = if warmup_fraction >= 1.0 {
                        1.0
                    } else {
                        ((progress - warmup_fraction) / (1.0 - warmup_fraction)).clamp(0.0, 1.0)
                    };
                    min_lr
                        + 0.5
                            * (peak_learning_rate - min_lr)
                            * (1.0 + (decay_progress * std::f64::consts::PI).cos())
                }
            }
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.warmup_fraction < 0.0 || self.warmup_fraction > 1.0 {
            return Err(FractalError::InvalidConfig(
                "optimizer warmup_fraction must be within [0, 1]".into(),
            ));
        }
        if self.decay_floor_fraction < 0.0 || self.decay_floor_fraction > 1.0 {
            return Err(FractalError::InvalidConfig(
                "optimizer decay_floor_fraction must be within [0, 1]".into(),
            ));
        }
        if matches!(self.kind, LearningRateScheduleKind::Constant)
            && (self.warmup_fraction != 0.0 || self.decay_floor_fraction != 1.0)
        {
            return Err(FractalError::InvalidConfig(
                "constant optimizer schedule must use warmup_fraction=0 and decay_floor_fraction=1"
                    .into(),
            ));
        }
        if matches!(self.kind, LearningRateScheduleKind::WarmupCosine)
            && self.warmup_fraction >= 1.0
        {
            return Err(FractalError::InvalidConfig(
                "warmup_cosine optimizer schedule requires warmup_fraction < 1".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerSpec {
    pub kind: OptimizerKind,
    pub peak_learning_rate: f64,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub weight_decay: f64,
    pub gradient_clip_norm: Option<f64>,
    pub schedule: LearningRateScheduleSpec,
}

impl Default for OptimizerSpec {
    fn default() -> Self {
        Self::legacy_adam(1e-3)
    }
}

impl OptimizerSpec {
    pub fn legacy_adam(learning_rate: f64) -> Self {
        Self {
            kind: OptimizerKind::Adam,
            peak_learning_rate: learning_rate,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            gradient_clip_norm: None,
            schedule: LearningRateScheduleSpec::constant(),
        }
    }

    pub fn stage0_adamw() -> Self {
        Self {
            kind: OptimizerKind::AdamW,
            peak_learning_rate: 2e-4,
            beta_1: 0.9,
            beta_2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.05,
            gradient_clip_norm: Some(1.0),
            schedule: LearningRateScheduleSpec::warmup_cosine(0.02, 0.1),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "kind={} peak_lr={} beta_1={} beta_2={} epsilon={} weight_decay={} grad_clip_norm={} schedule={}",
            self.kind.as_str(),
            self.peak_learning_rate,
            self.beta_1,
            self.beta_2,
            self.epsilon,
            self.weight_decay,
            self.gradient_clip_norm
                .map(|value| value.to_string())
                .unwrap_or_else(|| "disabled".to_owned()),
            self.schedule.label(),
        )
    }

    pub fn learning_rate_at_tokens(&self, seen_tokens: usize, total_tokens: usize) -> f64 {
        self.schedule
            .learning_rate_at_tokens(self.peak_learning_rate, seen_tokens, total_tokens)
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.peak_learning_rate <= 0.0 {
            return Err(FractalError::InvalidConfig(
                "optimizer peak_learning_rate must be greater than zero".into(),
            ));
        }
        if self.weight_decay < 0.0 {
            return Err(FractalError::InvalidConfig(
                "optimizer weight_decay must be non-negative".into(),
            ));
        }
        if let Some(clip_norm) = self.gradient_clip_norm {
            if clip_norm <= 0.0 {
                return Err(FractalError::InvalidConfig(
                    "optimizer gradient_clip_norm must be greater than zero when configured".into(),
                ));
            }
        }
        self.schedule.validate()?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArcSourceSpec {
    pub mode: ArcSourceMode,
}

impl ArcSourceSpec {
    pub const fn synthetic_canonical() -> Self {
        Self {
            mode: ArcSourceMode::SyntheticCanonical,
        }
    }

    pub const fn tokenizer_backed_canonical() -> Self {
        Self {
            mode: ArcSourceMode::TokenizerBackedCanonical,
        }
    }

    pub const fn unavailable() -> Self {
        Self {
            mode: ArcSourceMode::Unavailable,
        }
    }

    pub fn validate(&self, training_input_mode: TrainingInputMode) -> Result<(), FractalError> {
        match training_input_mode {
            TrainingInputMode::Synthetic => Ok(()),
            TrainingInputMode::TokenizerBackedText => match self.mode {
                ArcSourceMode::SyntheticCanonical => Ok(()),
                ArcSourceMode::TokenizerBackedCanonical => Err(FractalError::InvalidConfig(
                    "tokenizer-backed Stage 0 cannot yet map canonical ARC from text batches"
                        .into(),
                )),
                ArcSourceMode::Unavailable => Err(FractalError::InvalidConfig(
                    "tokenizer-backed Stage 0 must declare an explicit canonical ARC source".into(),
                )),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BridgeSplitPolicy {
    Balanced,
    BoundaryAware,
    SyntaxAware,
}

impl BridgeSplitPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::BoundaryAware => "boundary-aware",
            Self::SyntaxAware => "syntax-aware",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BridgeSubstrateMode {
    RawBytes,
    LexicalAtoms,
}

impl BridgeSubstrateMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RawBytes => "raw-bytes",
            Self::LexicalAtoms => "lexical-atoms",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BridgePackagingSpec {
    pub vocab_artifact_path: String,
    pub dim: usize,
    pub levels: usize,
    pub max_depth: usize,
    pub seed: u64,
    pub split_policy: BridgeSplitPolicy,
    pub substrate_mode: BridgeSubstrateMode,
    pub chunk_max_tokens: usize,
    pub chunk_max_bytes: usize,
}

impl BridgePackagingSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.vocab_artifact_path.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "bridge packaging vocab_artifact_path must be non-empty".into(),
            ));
        }
        for (name, value) in [
            ("dim", self.dim),
            ("levels", self.levels),
            ("max_depth", self.max_depth),
            ("chunk_max_tokens", self.chunk_max_tokens),
            ("chunk_max_bytes", self.chunk_max_bytes),
        ] {
            if value == 0 {
                return Err(FractalError::InvalidConfig(format!(
                    "bridge packaging {name} must be greater than zero"
                )));
            }
        }
        Ok(())
    }

    pub fn validate_against_config(&self, config: &TournamentConfig) -> Result<(), FractalError> {
        self.validate()?;
        for (name, bridge_value, config_value) in [
            ("dim", self.dim, config.dim),
            ("levels", self.levels, config.levels),
            ("max_depth", self.max_depth, config.max_recursion_depth),
        ] {
            if bridge_value != config_value {
                return Err(FractalError::InvalidConfig(format!(
                    "bridge packaging {name} {} must match config {} {}",
                    bridge_value, name, config_value
                )));
            }
        }
        if self.seed != config.seed {
            return Err(FractalError::InvalidConfig(format!(
                "bridge packaging seed {} must match config seed {}",
                self.seed, config.seed
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerArtifactSpec {
    pub artifact_id: String,
    pub artifact_path: Option<String>,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl TokenizerArtifactSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.artifact_id.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "tokenizer artifact_id must be non-empty".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(FractalError::InvalidConfig(
                "tokenizer vocab_size must be greater than zero".into(),
            ));
        }
        if self.pad_token_id >= self.vocab_size {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer pad_token_id {} must be less than vocab_size {}",
                self.pad_token_id, self.vocab_size
            )));
        }
        Ok(())
    }

    pub fn validate_against_model(
        &self,
        model_vocab_size: usize,
        model_pad_token_id: usize,
    ) -> Result<(), FractalError> {
        self.validate()?;
        if self.vocab_size != model_vocab_size {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer vocab_size {} must match model vocab_size {}",
                self.vocab_size, model_vocab_size
            )));
        }
        if self.pad_token_id != model_pad_token_id {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer pad_token_id {} must match model pad_token_id {}",
                self.pad_token_id, model_pad_token_id
            )));
        }
        if self.pad_token_id != PAD_TOKEN {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer pad_token_id {} must match canonical PAD_TOKEN {}",
                self.pad_token_id, PAD_TOKEN
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerBridgeSpec {
    pub enabled: bool,
    pub observational_only: bool,
}

impl TokenizerBridgeSpec {
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            observational_only: false,
        }
    }

    pub const fn observational_only() -> Self {
        Self {
            enabled: true,
            observational_only: true,
        }
    }

    pub fn validate(&self, mode: TrainingInputMode) -> Result<(), FractalError> {
        match mode {
            TrainingInputMode::Synthetic => Ok(()),
            TrainingInputMode::TokenizerBackedText => {
                if !self.enabled {
                    return Err(FractalError::InvalidConfig(
                        "tokenizer-backed text training requires the tokenizer bridge to be enabled"
                            .into(),
                    ));
                }
                if !self.observational_only {
                    return Err(FractalError::InvalidConfig(
                        "Stage 0 tokenizer bridge must remain observational only".into(),
                    ));
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingInputSpec {
    pub mode: TrainingInputMode,
    pub corpus_name: Option<String>,
    pub corpus_source: Option<TextCorpusSourceSpec>,
    pub tokenizer: Option<TokenizerArtifactSpec>,
    pub bridge: TokenizerBridgeSpec,
    pub bridge_packaging: Option<BridgePackagingSpec>,
    pub arc_source: ArcSourceSpec,
}

impl TrainingInputSpec {
    pub fn synthetic() -> Self {
        Self {
            mode: TrainingInputMode::Synthetic,
            corpus_name: None,
            corpus_source: None,
            tokenizer: None,
            bridge: TokenizerBridgeSpec::disabled(),
            bridge_packaging: None,
            arc_source: ArcSourceSpec::synthetic_canonical(),
        }
    }

    pub fn tokenizer_backed_text(
        corpus_name: impl Into<String>,
        tokenizer: TokenizerArtifactSpec,
        corpus_source: TextCorpusSourceSpec,
    ) -> Self {
        Self {
            mode: TrainingInputMode::TokenizerBackedText,
            corpus_name: Some(corpus_name.into()),
            corpus_source: Some(corpus_source),
            tokenizer: Some(tokenizer),
            bridge: TokenizerBridgeSpec::observational_only(),
            bridge_packaging: None,
            arc_source: ArcSourceSpec::synthetic_canonical(),
        }
    }

    pub fn with_bridge_packaging(mut self, bridge_packaging: BridgePackagingSpec) -> Self {
        self.bridge_packaging = Some(bridge_packaging);
        self
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.bridge.validate(self.mode)?;
        self.arc_source.validate(self.mode)?;
        match self.mode {
            TrainingInputMode::Synthetic => {
                if self.corpus_name.is_some()
                    || self.corpus_source.is_some()
                    || self.tokenizer.is_some()
                    || self.bridge_packaging.is_some()
                {
                    return Err(FractalError::InvalidConfig(
                        "synthetic training input must not carry tokenizer corpus metadata".into(),
                    ));
                }
                Ok(())
            }
            TrainingInputMode::TokenizerBackedText => {
                let corpus_name = self.corpus_name.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires a corpus name".into(),
                    )
                })?;
                if corpus_name.trim().is_empty() {
                    return Err(FractalError::InvalidConfig(
                        "tokenizer-backed text corpus name must be non-empty".into(),
                    ));
                }
                let corpus_source = self.corpus_source.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires corpus split sources".into(),
                    )
                })?;
                corpus_source.validate()?;
                let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires tokenizer artifact metadata"
                            .into(),
                    )
                })?;
                tokenizer.validate()?;
                let bridge_packaging = self.bridge_packaging.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires bridge packaging metadata".into(),
                    )
                })?;
                bridge_packaging.validate()
            }
        }
    }

    pub fn validate_against_config(&self, config: &TournamentConfig) -> Result<(), FractalError> {
        self.validate()?;
        match self.mode {
            TrainingInputMode::Synthetic => Ok(()),
            TrainingInputMode::TokenizerBackedText => {
                let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires tokenizer artifact metadata"
                            .into(),
                    )
                })?;
                tokenizer.validate_against_model(config.vocab_size, PAD_TOKEN)?;
                let bridge_packaging = self.bridge_packaging.as_ref().ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "tokenizer-backed text training requires bridge packaging metadata".into(),
                    )
                })?;
                bridge_packaging.validate_against_config(config)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BudgetSpec {
    pub preset: TournamentPreset,
    pub seed: u64,
    pub train_batch_size: usize,
    pub eval_batch_size: usize,
    pub train_steps_per_species: usize,
    pub train_token_budget: Option<usize>,
    pub eval_batches_per_family: usize,
    pub perplexity_eval_batches: usize,
    pub arc_eval_batches: usize,
    pub max_recursion_depth: usize,
    pub stability_depth: usize,
    pub learning_rate: f64,
    pub timeout_seconds: Option<f64>,
}

impl BudgetSpec {
    pub fn from_config(preset: TournamentPreset, config: &TournamentConfig) -> Self {
        Self {
            preset,
            seed: config.seed,
            train_batch_size: config.train_batch_size,
            eval_batch_size: config.eval_batch_size,
            train_steps_per_species: config.train_steps_per_species,
            train_token_budget: config.train_token_budget,
            eval_batches_per_family: config.eval_batches_per_family,
            perplexity_eval_batches: config.effective_perplexity_eval_batches(),
            arc_eval_batches: config.effective_arc_eval_batches(),
            max_recursion_depth: config.max_recursion_depth,
            stability_depth: config.stability_depth,
            learning_rate: config.learning_rate,
            timeout_seconds: config.run_timeout.map(|timeout| timeout.as_secs_f64()),
        }
    }

    fn matches_config(&self, config: &TournamentConfig) -> bool {
        self.seed == config.seed
            && self.train_batch_size == config.train_batch_size
            && self.eval_batch_size == config.eval_batch_size
            && self.train_steps_per_species == config.train_steps_per_species
            && self.train_token_budget == config.train_token_budget
            && self.eval_batches_per_family == config.eval_batches_per_family
            && self.perplexity_eval_batches == config.effective_perplexity_eval_batches()
            && self.arc_eval_batches == config.effective_arc_eval_batches()
            && self.max_recursion_depth == config.max_recursion_depth
            && self.stability_depth == config.stability_depth
            && (self.learning_rate - config.learning_rate).abs() < f64::EPSILON
            && self.timeout_seconds == config.run_timeout.map(|timeout| timeout.as_secs_f64())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvalBackendPolicy {
    SharedBackend,
}

impl EvalBackendPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SharedBackend => "shared-backend",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BatchingPolicy {
    Padded,
}

impl BatchingPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Padded => "padded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForwardExecutionPolicy {
    SimpleLoop,
}

impl ForwardExecutionPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SimpleLoop => "simple-loop",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BufferReusePolicy {
    Disabled,
}

impl BufferReusePolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkMode {
    Leaderboard,
    SystemsSpeed,
}

impl BenchmarkMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Leaderboard => "leaderboard",
            Self::SystemsSpeed => "systems-speed",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeBackendPolicy {
    ActiveExecutionBackend,
}

impl RuntimeBackendPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ActiveExecutionBackend => "active-execution-backend",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NumericPrecisionKind {
    BackendDefault,
    Bf16,
    Fp32,
}

impl NumericPrecisionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BackendDefault => "backend-default",
            Self::Bf16 => "bf16",
            Self::Fp32 => "fp32",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantizedPrecisionKind {
    Int8,
    Int4,
    Bit1,
}

impl QuantizedPrecisionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Bit1 => "bit1",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationPolicy {
    pub weights: Option<QuantizedPrecisionKind>,
    pub activations: Option<QuantizedPrecisionKind>,
}

impl QuantizationPolicy {
    pub const fn disabled() -> Self {
        Self {
            weights: None,
            activations: None,
        }
    }

    pub const fn is_enabled(&self) -> bool {
        self.weights.is_some() || self.activations.is_some()
    }

    pub fn label(&self) -> String {
        if !self.is_enabled() {
            return "disabled".to_owned();
        }

        format!(
            "weights={} activations={}",
            self.weights.map(|kind| kind.as_str()).unwrap_or("none"),
            self.activations.map(|kind| kind.as_str()).unwrap_or("none")
        )
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionPolicy {
    pub compute: NumericPrecisionKind,
    pub optimizer_state: NumericPrecisionKind,
    pub reduction: NumericPrecisionKind,
    pub tf32_enabled: bool,
    #[serde(default)]
    pub quantization: QuantizationPolicy,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::legacy_default()
    }
}

impl PrecisionPolicy {
    pub const fn legacy_default() -> Self {
        Self {
            compute: NumericPrecisionKind::BackendDefault,
            optimizer_state: NumericPrecisionKind::BackendDefault,
            reduction: NumericPrecisionKind::BackendDefault,
            tf32_enabled: false,
            quantization: QuantizationPolicy::disabled(),
        }
    }

    pub const fn stage0_default() -> Self {
        Self {
            compute: NumericPrecisionKind::Bf16,
            optimizer_state: NumericPrecisionKind::Fp32,
            reduction: NumericPrecisionKind::Fp32,
            tf32_enabled: true,
            quantization: QuantizationPolicy::disabled(),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "compute={} optimizer_state={} reduction={} tf32={} quantization={}",
            self.compute.as_str(),
            self.optimizer_state.as_str(),
            self.reduction.as_str(),
            self.tf32_enabled,
            self.quantization.label()
        )
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.quantization.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPolicy {
    pub interval_tokens: Option<usize>,
    pub keep_latest: bool,
    pub keep_best: bool,
    pub keep_final: bool,
    pub keep_previous: bool,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self::legacy_default()
    }
}

impl CheckpointPolicy {
    pub const fn legacy_default() -> Self {
        Self {
            interval_tokens: None,
            keep_latest: false,
            keep_best: false,
            keep_final: false,
            keep_previous: false,
        }
    }

    pub const fn stage0_default() -> Self {
        Self {
            interval_tokens: Some(10_000_000),
            keep_latest: true,
            keep_best: true,
            keep_final: true,
            keep_previous: true,
        }
    }

    pub fn label(&self) -> String {
        match self.interval_tokens {
            None => "disabled".to_owned(),
            Some(interval) => format!(
                "interval_tokens={} retain(latest={},best={},final={},previous={})",
                interval, self.keep_latest, self.keep_best, self.keep_final, self.keep_previous
            ),
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.interval_tokens == Some(0) {
            return Err(FractalError::InvalidConfig(
                "checkpoint interval_tokens must be greater than zero when configured".into(),
            ));
        }
        if self.interval_tokens.is_none()
            && (self.keep_latest || self.keep_best || self.keep_final || self.keep_previous)
        {
            return Err(FractalError::InvalidConfig(
                "checkpoint retention cannot be enabled when interval_tokens is unset".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCadencePolicy {
    pub perplexity_interval_tokens: Option<usize>,
    pub stability_interval_tokens: Option<usize>,
    pub arc_interval_tokens: Option<usize>,
    pub systems_speed_interval_tokens: Option<usize>,
    pub final_full_eval: bool,
}

impl Default for EvalCadencePolicy {
    fn default() -> Self {
        Self::legacy_default()
    }
}

impl EvalCadencePolicy {
    pub const fn legacy_default() -> Self {
        Self {
            perplexity_interval_tokens: None,
            stability_interval_tokens: None,
            arc_interval_tokens: None,
            systems_speed_interval_tokens: None,
            final_full_eval: true,
        }
    }

    pub const fn stage0_default() -> Self {
        Self {
            perplexity_interval_tokens: Some(10_000_000),
            stability_interval_tokens: Some(10_000_000),
            arc_interval_tokens: Some(20_000_000),
            systems_speed_interval_tokens: Some(20_000_000),
            final_full_eval: true,
        }
    }

    pub fn label(&self) -> String {
        format!(
            "perplexity={:?} stability={:?} arc={:?} systems_speed={:?} final_full_eval={}",
            self.perplexity_interval_tokens,
            self.stability_interval_tokens,
            self.arc_interval_tokens,
            self.systems_speed_interval_tokens,
            self.final_full_eval
        )
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        for (name, interval) in [
            (
                "perplexity_interval_tokens",
                self.perplexity_interval_tokens,
            ),
            ("stability_interval_tokens", self.stability_interval_tokens),
            ("arc_interval_tokens", self.arc_interval_tokens),
            (
                "systems_speed_interval_tokens",
                self.systems_speed_interval_tokens,
            ),
        ] {
            if interval == Some(0) {
                return Err(FractalError::InvalidConfig(format!(
                    "{name} must be greater than zero when configured"
                )));
            }
        }
        if !self.final_full_eval {
            return Err(FractalError::InvalidConfig(
                "final_full_eval=false is not yet supported by the runtime".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResumePolicy {
    pub resume_on_interrupt: bool,
    pub restart_on_corruption: bool,
    pub restart_on_contract_ambiguity: bool,
}

impl Default for ResumePolicy {
    fn default() -> Self {
        Self::legacy_default()
    }
}

impl ResumePolicy {
    pub const fn legacy_default() -> Self {
        Self {
            resume_on_interrupt: false,
            restart_on_corruption: false,
            restart_on_contract_ambiguity: false,
        }
    }

    pub const fn stage0_default() -> Self {
        Self {
            resume_on_interrupt: true,
            restart_on_corruption: true,
            restart_on_contract_ambiguity: true,
        }
    }

    pub fn label(&self) -> String {
        format!(
            "resume_on_interrupt={} restart_on_corruption={} restart_on_contract_ambiguity={}",
            self.resume_on_interrupt,
            self.restart_on_corruption,
            self.restart_on_contract_ambiguity
        )
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if !self.resume_on_interrupt
            && (self.restart_on_corruption || self.restart_on_contract_ambiguity)
        {
            return Err(FractalError::InvalidConfig(
                "resume restart semantics require resume_on_interrupt to be enabled".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPolicySpec {
    pub precision: PrecisionPolicy,
    pub checkpoint: CheckpointPolicy,
    pub eval_cadence: EvalCadencePolicy,
    pub resume: ResumePolicy,
}

impl Default for LaunchPolicySpec {
    fn default() -> Self {
        Self::legacy_default()
    }
}

impl LaunchPolicySpec {
    pub const fn legacy_default() -> Self {
        Self {
            precision: PrecisionPolicy::legacy_default(),
            checkpoint: CheckpointPolicy::legacy_default(),
            eval_cadence: EvalCadencePolicy::legacy_default(),
            resume: ResumePolicy::legacy_default(),
        }
    }

    pub const fn stage0_default() -> Self {
        Self {
            precision: PrecisionPolicy::stage0_default(),
            checkpoint: CheckpointPolicy::stage0_default(),
            eval_cadence: EvalCadencePolicy::stage0_default(),
            resume: ResumePolicy::stage0_default(),
        }
    }

    pub fn label(&self) -> String {
        if *self == Self::legacy_default() {
            "legacy-default".to_owned()
        } else {
            format!(
                "precision=[{}] checkpoint=[{}] eval=[{}] resume=[{}]",
                self.precision.label(),
                self.checkpoint.label(),
                self.eval_cadence.label(),
                self.resume.label()
            )
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.precision.validate()?;
        self.checkpoint.validate()?;
        self.eval_cadence.validate()?;
        self.resume.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSurfaceSpec {
    pub eval_backend_policy: EvalBackendPolicy,
    pub batching_policy: BatchingPolicy,
    pub execution_policy: ForwardExecutionPolicy,
    pub buffer_reuse_policy: BufferReusePolicy,
    pub benchmark_mode: BenchmarkMode,
    pub backend_policy: RuntimeBackendPolicy,
    pub launch_policy: LaunchPolicySpec,
}

impl Default for RuntimeSurfaceSpec {
    fn default() -> Self {
        Self {
            eval_backend_policy: EvalBackendPolicy::SharedBackend,
            batching_policy: BatchingPolicy::Padded,
            execution_policy: ForwardExecutionPolicy::SimpleLoop,
            buffer_reuse_policy: BufferReusePolicy::Disabled,
            benchmark_mode: BenchmarkMode::Leaderboard,
            backend_policy: RuntimeBackendPolicy::ActiveExecutionBackend,
            launch_policy: LaunchPolicySpec::legacy_default(),
        }
    }
}

impl RuntimeSurfaceSpec {
    pub fn label(&self) -> String {
        if *self == Self::default() {
            "conservative-defaults".to_owned()
        } else {
            format!(
                "eval_backend={} batching={} execution={} buffer_reuse={} benchmark_mode={} backend_policy={} launch_policy={}",
                self.eval_backend_policy.as_str(),
                self.batching_policy.as_str(),
                self.execution_policy.as_str(),
                self.buffer_reuse_policy.as_str(),
                self.benchmark_mode.as_str(),
                self.backend_policy.as_str(),
                self.launch_policy.label(),
            )
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTargetKind {
    Local,
    RunPod,
}

impl ExecutionTargetKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::RunPod => "runpod",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionBackend {
    Cpu,
    Metal,
    Cuda,
}

impl ExecutionBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    pub fn from_compute_backend(backend: &ComputeBackend) -> Self {
        match backend {
            ComputeBackend::CpuCandle => Self::Cpu,
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaCandle { .. } => Self::Cuda,
            ComputeBackend::MetalWgpu { .. } => Self::Metal,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionTarget {
    pub kind: ExecutionTargetKind,
    pub backend: ExecutionBackend,
    pub execution_mode: ExecutionMode,
    pub pod_id: Option<String>,
    pub wrapper_timeout_seconds: Option<u64>,
}

impl ExecutionTarget {
    fn matches_config(&self, config: &TournamentConfig) -> bool {
        self.backend == ExecutionBackend::from_compute_backend(&config.execution_backend)
            && self.execution_mode == config.execution_mode
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactPolicy {
    pub manifest_required: bool,
    pub structured_artifact_required: bool,
    pub final_log_required: bool,
    pub tracker_ready_output_required: bool,
}

impl Default for ArtifactPolicy {
    fn default() -> Self {
        Self {
            manifest_required: true,
            structured_artifact_required: true,
            final_log_required: true,
            tracker_ready_output_required: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentSpec {
    pub experiment_id: ExperimentId,
    pub question: ExperimentQuestion,
    pub variant: VariantSpec,
    pub model: ModelContractSpec,
    pub training_input: TrainingInputSpec,
    pub budget: BudgetSpec,
    pub optimizer: OptimizerSpec,
    pub runtime: RuntimeSurfaceSpec,
    pub comparison: ComparisonContract,
    pub execution: ExecutionTarget,
    pub artifacts: ArtifactPolicy,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentSpecTemplate {
    pub experiment_id: ExperimentId,
    pub question: ExperimentQuestion,
    pub budget: BudgetSpec,
    pub optimizer: OptimizerSpec,
    pub model: ModelContractSpec,
    pub training_input: TrainingInputSpec,
    pub runtime: RuntimeSurfaceSpec,
    pub comparison: ComparisonContract,
    pub execution: ExecutionTarget,
    pub artifacts: ArtifactPolicy,
}

impl ExperimentSpecTemplate {
    pub fn resolve_variant(
        &self,
        species: SpeciesId,
        variant_name: PrimitiveVariantName,
    ) -> ExperimentSpec {
        ExperimentSpec {
            experiment_id: self.experiment_id.clone(),
            question: self.question.clone(),
            variant: VariantSpec {
                species,
                variant_name,
            },
            model: self.model.clone(),
            training_input: self.training_input.clone(),
            budget: self.budget.clone(),
            optimizer: self.optimizer.clone(),
            runtime: self.runtime.clone(),
            comparison: self.comparison.clone(),
            execution: self.execution.clone(),
            artifacts: self.artifacts.clone(),
        }
    }

    pub fn validate_against_config(&self, config: &TournamentConfig) -> Result<(), FractalError> {
        if !self.budget.matches_config(config) {
            return Err(FractalError::InvalidConfig(
                "experiment budget must match the resolved tournament config".into(),
            ));
        }
        self.model.validate_against_config(config)?;
        self.training_input.validate_against_config(config)?;
        if !self.execution.matches_config(config) {
            return Err(FractalError::InvalidConfig(
                "experiment execution target must match the resolved tournament config".into(),
            ));
        }
        if self.optimizer != config.optimizer {
            return Err(FractalError::InvalidConfig(
                "experiment optimizer must match the resolved tournament config".into(),
            ));
        }
        if self.runtime.launch_policy != config.launch_policy {
            return Err(FractalError::InvalidConfig(
                "experiment launch policy must match the resolved tournament config".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesPresetOverride {
    pub species: SpeciesId,
    pub train_batch_size: Option<usize>,
    pub eval_batch_size: Option<usize>,
    pub train_steps_per_species: Option<usize>,
    pub train_token_budget: Option<usize>,
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
            train_token_budget: None,
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
        if let Some(train_token_budget) = self.train_token_budget {
            config.train_token_budget = Some(train_token_budget);
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
    pub train_token_budget: Option<usize>,
    pub eval_batches_per_family: usize,
    pub perplexity_eval_batches: Option<usize>,
    pub arc_eval_batches: Option<usize>,
    pub learning_rate: f64,
    pub optimizer: OptimizerSpec,
    pub launch_policy: LaunchPolicySpec,
    pub seed: u64,
    pub run_timeout: Option<Duration>,
    pub generator_depth_config: GeneratorDepthConfig,
    pub execution_backend: ComputeBackend,
    pub execution_mode: ExecutionMode,
    pub parallelism: usize,
    pub species_overrides: Vec<SpeciesPresetOverride>,
    pub experiment: Option<ExperimentSpecTemplate>,
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
            train_token_budget: None,
            eval_batches_per_family: 1,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
        if self.train_steps_per_species == 0 {
            return Err(FractalError::InvalidConfig(
                "train_steps_per_species must be greater than zero".into(),
            ));
        }
        if self.train_token_budget == Some(0) {
            return Err(FractalError::InvalidConfig(
                "train_token_budget must be greater than zero when configured".into(),
            ));
        }
        if self.eval_batches_per_family == 0 {
            return Err(FractalError::InvalidConfig(
                "eval_batches_per_family must be greater than zero".into(),
            ));
        }
        if self.perplexity_eval_batches == Some(0) {
            return Err(FractalError::InvalidConfig(
                "perplexity_eval_batches must be greater than zero when configured".into(),
            ));
        }
        if self.arc_eval_batches == Some(0) {
            return Err(FractalError::InvalidConfig(
                "arc_eval_batches must be greater than zero when configured".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(FractalError::InvalidConfig(
                "learning_rate must be greater than zero".into(),
            ));
        }
        if (self.learning_rate - self.optimizer.peak_learning_rate).abs() >= f64::EPSILON {
            return Err(FractalError::InvalidConfig(
                "learning_rate must match optimizer peak_learning_rate".into(),
            ));
        }
        self.optimizer.validate()?;
        self.launch_policy.validate()?;
        resolve_precision_profile(&self.execution_backend, &self.launch_policy.precision)?;
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
            if override_config.train_token_budget == Some(0) {
                return Err(FractalError::InvalidConfig(
                    "species override train_token_budget must be greater than zero".into(),
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

        if let Some(experiment) = &self.experiment {
            experiment.validate_against_config(self)?;
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
            train_token_budget: None,
            eval_batches_per_family: 8,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
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
                train_token_budget: None,
                ..SpeciesPresetOverride::for_species(SpeciesId::Ifs)
            }],
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 5e-4,
            optimizer: OptimizerSpec::legacy_adam(5e-4),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::polish_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 2,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_steps_per_species: 1,
            train_token_budget: None,
            eval_batches_per_family: 1,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::default(),
            execution_backend: ComputeBackend::CpuCandle,
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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
            train_token_budget: None,
            eval_batches_per_family: 4,
            perplexity_eval_batches: None,
            arc_eval_batches: None,
            learning_rate: 1e-3,
            optimizer: OptimizerSpec::legacy_adam(1e-3),
            launch_policy: LaunchPolicySpec::legacy_default(),
            seed: 42,
            run_timeout: None,
            generator_depth_config: GeneratorDepthConfig::stress_top_candidates(),
            execution_backend: ComputeBackend::default_for_current_platform(),
            execution_mode: ExecutionMode::Sequential,
            parallelism: 4,
            species_overrides: Vec::new(),
            experiment: None,
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

    pub fn with_optimizer(mut self, optimizer: OptimizerSpec) -> Self {
        self.learning_rate = optimizer.peak_learning_rate;
        self.optimizer = optimizer;
        self
    }

    pub fn with_launch_policy(mut self, launch_policy: LaunchPolicySpec) -> Self {
        self.launch_policy = launch_policy;
        self
    }

    pub fn with_experiment(mut self, experiment: ExperimentSpecTemplate) -> Self {
        self.experiment = Some(experiment);
        self
    }

    pub fn resolved_experiment(
        &self,
        species: SpeciesId,
        variant_name: PrimitiveVariantName,
    ) -> Option<ExperimentSpec> {
        self.experiment
            .as_ref()
            .map(|experiment| experiment.resolve_variant(species, variant_name))
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

    pub fn effective_perplexity_eval_batches(&self) -> usize {
        self.perplexity_eval_batches
            .unwrap_or(self.eval_batches_per_family)
    }

    pub fn effective_arc_eval_batches(&self) -> usize {
        self.arc_eval_batches
            .unwrap_or(self.eval_batches_per_family)
    }

    fn max_effective_eval_batches_per_family(&self) -> usize {
        self.effective_perplexity_eval_batches()
            .max(self.effective_arc_eval_batches())
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointArtifactKind {
    Latest,
    Previous,
    Best,
    Final,
}

impl CheckpointArtifactKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Latest => "latest",
            Self::Previous => "previous",
            Self::Best => "best",
            Self::Final => "final",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointArtifact {
    pub kind: CheckpointArtifactKind,
    pub tokens_seen: usize,
    pub completed_steps: usize,
    pub directory: String,
    pub long_context_perplexity: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct InterimEvalSnapshot {
    pub tokens_seen: usize,
    pub completed_steps: usize,
    pub stability_score: Option<f64>,
    pub long_context_perplexity: Option<f64>,
    pub arc_accuracy: Option<f64>,
    pub tokens_per_sec: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct TrainingRuntimeArtifact {
    pub completed_steps: usize,
    pub planned_steps: usize,
    pub train_tokens_seen: usize,
    pub target_train_tokens: usize,
    pub resumed_from_checkpoint: bool,
    pub checkpoints: Vec<CheckpointArtifact>,
    pub interim_evaluations: Vec<InterimEvalSnapshot>,
}

#[derive(Clone, Debug)]
pub struct RunManifest {
    pub variant_name: PrimitiveVariantName,
    pub timeout_budget: Option<Duration>,
    pub config: TournamentConfig,
    pub experiment: Option<ExperimentSpec>,
}

#[derive(Clone, Debug)]
pub struct SpeciesRunArtifact {
    pub stage: SpeciesRunStage,
    pub manifest: RunManifest,
    pub phase_timings: Vec<PhaseTiming>,
    pub training_runtime: TrainingRuntimeArtifact,
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
            (config.max_eval_batch_size() * config.max_effective_eval_batches_per_family()).max(32);
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
        let config = self.config.effective_for_species(definition.id);
        SpeciesRunContext {
            index,
            experiment: config.resolved_experiment(definition.id, definition.variant_name),
            config,
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
                experiment: context.experiment.clone(),
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
