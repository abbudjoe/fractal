use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use fractal_core::{
    build_attention_only_graph_of_experts_model,
    build_attention_only_recurrent_graph_of_experts_model,
    build_reference_ssm_graph_of_experts_model, error::FractalError,
    AttentionOnlyGraphOfExpertsModel, AttentionOnlyRecurrentGraphOfExpertsModel,
    AttentionOnlyRecurrentGraphOfExpertsVariantSpec, GraphOfExpertsControllerSpec,
    GraphOfExpertsRoutingProbe, GraphOfExpertsVariantSpec, RecurrentGraphOfExpertsRoutingProbe,
    ReferenceSsmGraphOfExpertsModel, TokenBatch, GOE_CHANNEL_COUNT,
};

use crate::{
    hybrid_attention_training::{
        HybridAttentionExecutionBackend, HybridAttentionRuntimeMetrics,
        HybridAttentionSmokeTrainReport,
    },
    process_memory_measurement_note, process_memory_metric_kind, process_peak_memory_bytes,
    v2_training::{evaluate_model, load_byte_level_smoke_batches_from_source, next_token_loss},
    ByteLevelSmokeCorpusSource, ByteLevelVocabularyContract, V2SmokeCorpusStats,
    V2SmokeEvalMetrics, V2SmokeTrainModel, V2SmokeTrainStepReport, BYTE_LEVEL_PAD_TOKEN,
    BYTE_LEVEL_VOCAB_SIZE, DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE,
    DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};

pub const DEFAULT_GOE_RESULTS_LEDGER_PATH: &str = "docs/dreegmor-a-am-results-ledger.jsonl";
const GOE_ACTIVE_CHANNEL_MEAN_WEIGHT_FLOOR: f64 = 0.10;
const GOE_ROUTE_TIE_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOfExpertsSmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub variant: GraphOfExpertsVariantSpec,
    pub execution_backend: HybridAttentionExecutionBackend,
    pub seq_len: usize,
    pub window_stride: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub eval_batches: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub data_seed: Option<u64>,
    pub vocabulary: ByteLevelVocabularyContract,
}

impl GraphOfExpertsSmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        variant: GraphOfExpertsVariantSpec,
    ) -> Self {
        Self {
            corpus_source,
            output_dir,
            variant,
            execution_backend: HybridAttentionExecutionBackend::Cpu,
            seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
            batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
            train_steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            eval_holdout_every: DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: 42,
            data_seed: None,
            vocabulary: ByteLevelVocabularyContract::default(),
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.seq_len == 0
            || self.window_stride == 0
            || self.batch_size == 0
            || self.train_steps == 0
            || self.eval_batches == 0
        {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts_smoke_train sizes and counts must be greater than zero"
                    .to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts_smoke_train.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts_smoke_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.variant.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentGraphOfExpertsSmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub variant: AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    pub execution_backend: HybridAttentionExecutionBackend,
    pub seq_len: usize,
    pub window_stride: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub eval_batches: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub data_seed: Option<u64>,
    pub vocabulary: ByteLevelVocabularyContract,
}

impl RecurrentGraphOfExpertsSmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        variant: AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    ) -> Self {
        Self {
            corpus_source,
            output_dir,
            variant,
            execution_backend: HybridAttentionExecutionBackend::Cpu,
            seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
            batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
            train_steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            eval_holdout_every: DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: 42,
            data_seed: None,
            vocabulary: ByteLevelVocabularyContract::default(),
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.seq_len == 0
            || self.window_stride == 0
            || self.batch_size == 0
            || self.train_steps == 0
            || self.eval_batches == 0
        {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts_smoke_train sizes and counts must be greater than zero"
                    .to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts_smoke_train.eval_holdout_every must be at least 2"
                    .to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts_smoke_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.variant.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOfExpertsRoutingSummary {
    pub sampled_tokens: usize,
    pub round_count: usize,
    pub mean_pre_graph_channel_weights: [f64; 2],
    pub mean_channel_weights: [f64; 2],
    pub mean_round_adjustment_l1: Vec<f64>,
    pub winner_counts: [usize; 2],
    pub tied_token_count: usize,
    pub active_channel_count: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_graph_adjustment_l1: f64,
    pub edge_mix_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOfExpertsSmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: GraphOfExpertsSmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub routing: GraphOfExpertsRoutingSummary,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentGraphOfExpertsSmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: RecurrentGraphOfExpertsSmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub routing: GraphOfExpertsRoutingSummary,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphOfExpertsExperimentVariantKind {
    AttentionOnlyBaseline,
    ReferenceSsmBaseline,
    GraphOfExpertsOverAttentionOnly,
    GraphOfExpertsOverReferenceSsm,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOfExpertsExperimentVariantSummary {
    pub kind: GraphOfExpertsExperimentVariantKind,
    pub label: String,
    pub variant_label: String,
    pub model_label: String,
    pub note: String,
    pub seed: u64,
    pub execution_backend: HybridAttentionExecutionBackend,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_step_count: usize,
    pub report_path: PathBuf,
    pub routing: Option<GraphOfExpertsRoutingSummary>,
}

impl GraphOfExpertsExperimentVariantSummary {
    pub fn attention_only_baseline(report: &HybridAttentionSmokeTrainReport) -> Self {
        Self {
            kind: GraphOfExpertsExperimentVariantKind::AttentionOnlyBaseline,
            label: "A".to_string(),
            variant_label: report.config.variant.label.clone(),
            model_label: report.model_label.clone(),
            note: report.note.clone(),
            seed: report.config.seed,
            execution_backend: report.config.execution_backend,
            initial_eval: report.initial_eval.clone(),
            final_eval: report.final_eval.clone(),
            runtime: report.runtime.clone(),
            train_step_count: report.train_steps.len(),
            report_path: report.report_path.clone(),
            routing: None,
        }
    }

    pub fn reference_ssm_baseline(report: &HybridAttentionSmokeTrainReport) -> Self {
        Self {
            kind: GraphOfExpertsExperimentVariantKind::ReferenceSsmBaseline,
            label: "A + M".to_string(),
            variant_label: report.config.variant.label.clone(),
            model_label: report.model_label.clone(),
            note: report.note.clone(),
            seed: report.config.seed,
            execution_backend: report.config.execution_backend,
            initial_eval: report.initial_eval.clone(),
            final_eval: report.final_eval.clone(),
            runtime: report.runtime.clone(),
            train_step_count: report.train_steps.len(),
            report_path: report.report_path.clone(),
            routing: None,
        }
    }

    pub fn goe_attention_only(report: &GraphOfExpertsSmokeTrainReport) -> Self {
        Self {
            kind: GraphOfExpertsExperimentVariantKind::GraphOfExpertsOverAttentionOnly,
            label: "DREEGMOR(A)".to_string(),
            variant_label: report.config.variant.label.clone(),
            model_label: report.model_label.clone(),
            note: report.note.clone(),
            seed: report.config.seed,
            execution_backend: report.config.execution_backend,
            initial_eval: report.initial_eval.clone(),
            final_eval: report.final_eval.clone(),
            runtime: report.runtime.clone(),
            train_step_count: report.train_steps.len(),
            report_path: report.report_path.clone(),
            routing: Some(report.routing.clone()),
        }
    }

    pub fn goe_reference_ssm(report: &GraphOfExpertsSmokeTrainReport) -> Self {
        Self {
            kind: GraphOfExpertsExperimentVariantKind::GraphOfExpertsOverReferenceSsm,
            label: "DREEGMOR(A + M)".to_string(),
            variant_label: report.config.variant.label.clone(),
            model_label: report.model_label.clone(),
            note: report.note.clone(),
            seed: report.config.seed,
            execution_backend: report.config.execution_backend,
            initial_eval: report.initial_eval.clone(),
            final_eval: report.final_eval.clone(),
            runtime: report.runtime.clone(),
            train_step_count: report.train_steps.len(),
            report_path: report.report_path.clone(),
            routing: Some(report.routing.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphOfExpertsExperimentReport {
    pub note: String,
    pub variants: Vec<GraphOfExpertsExperimentVariantSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoeResultsLedgerKind {
    AAmExperimentRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoeResultsLedgerEntry {
    pub schema_version: u32,
    pub recorded_at_unix_seconds: u64,
    pub kind: GoeResultsLedgerKind,
    pub model: String,
    pub note: String,
    pub run_label: Option<String>,
    pub payload: Value,
}

impl GoeResultsLedgerEntry {
    pub fn aam_experiment_run(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &GraphOfExpertsExperimentReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Ok(Self {
            schema_version: 1,
            recorded_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            kind: GoeResultsLedgerKind::AAmExperimentRun,
            model: model.into(),
            note: note.into(),
            run_label,
            payload: serde_json::to_value(report).map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to serialize GoE results ledger payload: {error}"
                ))
            })?,
        })
    }
}

pub fn default_goe_results_ledger_path(repo_root: impl AsRef<Path>) -> PathBuf {
    repo_root.as_ref().join(DEFAULT_GOE_RESULTS_LEDGER_PATH)
}

pub fn resolve_requested_goe_results_ledger_path(
    repo_root: impl AsRef<Path>,
    request: Option<&str>,
) -> Result<Option<PathBuf>, FractalError> {
    let Some(request) = request else {
        return Ok(None);
    };
    if request.trim().is_empty() {
        return Err(FractalError::InvalidConfig(
            "goe_results_ledger.path must not be empty".to_string(),
        ));
    }
    if request == "default" {
        return Ok(Some(default_goe_results_ledger_path(repo_root)));
    }
    Ok(Some(PathBuf::from(request)))
}

pub fn append_goe_results_ledger_entry(
    path: impl AsRef<Path>,
    entry: &GoeResultsLedgerEntry,
) -> Result<(), FractalError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to create GoE results ledger directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to open GoE results ledger {} for append: {error}",
                path.display()
            ))
        })?;
    serde_json::to_writer(&mut file, entry).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize GoE results ledger entry for {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to terminate GoE results ledger entry for {}: {error}",
            path.display()
        ))
    })
}

pub fn read_goe_results_ledger(
    path: impl AsRef<Path>,
) -> Result<Vec<GoeResultsLedgerEntry>, FractalError> {
    let path = path.as_ref();
    let file = fs::File::open(path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to open GoE results ledger {}: {error}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for (line_index, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read GoE results ledger line {} from {}: {error}",
                line_index + 1,
                path.display()
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<GoeResultsLedgerEntry>(&line).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to parse GoE results ledger line {} from {}: {error}",
                line_index + 1,
                path.display()
            ))
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

impl<B: Backend> V2SmokeTrainModel<B> for AttentionOnlyGraphOfExpertsModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

impl<B: Backend> V2SmokeTrainModel<B> for ReferenceSsmGraphOfExpertsModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

impl<B: Backend> V2SmokeTrainModel<B> for AttentionOnlyRecurrentGraphOfExpertsModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

#[derive(Debug)]
struct RoutingProbeTensors<B: Backend> {
    initial_weights: Tensor<B, 3>,
    final_weights: Tensor<B, 3>,
    round_weights: Vec<Tensor<B, 3>>,
}

trait GraphOfExpertsTrainModel<B>:
    V2SmokeTrainModel<B> + AutodiffModule<B> + Module<B> + Clone
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RoutingProbeTensors<B>, FractalError>;
    fn edge_mix_fraction(&self) -> Result<f64, FractalError>;
}

impl<B> GraphOfExpertsTrainModel<B> for AttentionOnlyGraphOfExpertsModel<B>
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RoutingProbeTensors<B>, FractalError> {
        let probe: GraphOfExpertsRoutingProbe<B> = self.routing_probe(input_ids)?;
        Ok(RoutingProbeTensors {
            initial_weights: probe.pre_graph_weights,
            final_weights: probe.final_weights,
            round_weights: probe.round_weights,
        })
    }

    fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        self.edge_mix_fraction()
    }
}

impl<B> GraphOfExpertsTrainModel<B> for ReferenceSsmGraphOfExpertsModel<B>
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RoutingProbeTensors<B>, FractalError> {
        let probe: GraphOfExpertsRoutingProbe<B> = self.routing_probe(input_ids)?;
        Ok(RoutingProbeTensors {
            initial_weights: probe.pre_graph_weights,
            final_weights: probe.final_weights,
            round_weights: probe.round_weights,
        })
    }

    fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        self.edge_mix_fraction()
    }
}

impl<B> GraphOfExpertsTrainModel<B> for AttentionOnlyRecurrentGraphOfExpertsModel<B>
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RoutingProbeTensors<B>, FractalError> {
        let probe: RecurrentGraphOfExpertsRoutingProbe<B> = self.routing_probe(input_ids)?;
        Ok(RoutingProbeTensors {
            initial_weights: probe.initial_weights,
            final_weights: probe.final_weights,
            round_weights: probe.round_weights,
        })
    }

    fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        Ok(0.0)
    }
}

#[derive(Debug)]
struct DreegmorSmokeTrainArtifacts {
    corpus: V2SmokeCorpusStats,
    initial_eval: V2SmokeEvalMetrics,
    final_eval: V2SmokeEvalMetrics,
    routing: GraphOfExpertsRoutingSummary,
    runtime: HybridAttentionRuntimeMetrics,
    train_steps: Vec<V2SmokeTrainStepReport>,
}

pub fn run_attention_only_goe_smoke_train<B>(
    config: GraphOfExpertsSmokeTrainConfig,
    device: &B::Device,
) -> Result<GraphOfExpertsSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let model_label = format!(
        "dreegmor_attention_only_{}",
        controller_model_tag(config.variant.controller.clone())
    );
    let note = attention_only_goe_note(config.variant.controller.clone());
    let model = build_attention_only_graph_of_experts_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    let artifacts = train_dreegmor_model(
        model,
        &config.corpus_source,
        config.seq_len,
        config.window_stride,
        config.eval_holdout_every,
        config.batch_size,
        config.eval_batches,
        config.train_steps,
        config.learning_rate,
        config.seed,
        config.data_seed,
        device,
    )?;
    write_graph_of_experts_report(config, &model_label, &note, artifacts)
}

pub fn run_reference_ssm_goe_smoke_train<B>(
    config: GraphOfExpertsSmokeTrainConfig,
    device: &B::Device,
) -> Result<GraphOfExpertsSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let model_label = format!(
        "dreegmor_reference_ssm_{}",
        controller_model_tag(config.variant.controller.clone())
    );
    let note = reference_ssm_goe_note(config.variant.controller.clone());
    let model = build_reference_ssm_graph_of_experts_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    let artifacts = train_dreegmor_model(
        model,
        &config.corpus_source,
        config.seq_len,
        config.window_stride,
        config.eval_holdout_every,
        config.batch_size,
        config.eval_batches,
        config.train_steps,
        config.learning_rate,
        config.seed,
        config.data_seed,
        device,
    )?;
    write_graph_of_experts_report(config, &model_label, &note, artifacts)
}

pub fn run_attention_only_recurrent_goe_smoke_train<B>(
    config: RecurrentGraphOfExpertsSmokeTrainConfig,
    device: &B::Device,
) -> Result<RecurrentGraphOfExpertsSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let model_label = format!(
        "dreegmor_recurrent_attention_only_{}",
        recurrent_router_model_tag(config.variant.router.clone())
    );
    let note = attention_only_recurrent_goe_note(config.variant.router.clone());
    let model = build_attention_only_recurrent_graph_of_experts_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    let artifacts = train_dreegmor_model(
        model,
        &config.corpus_source,
        config.seq_len,
        config.window_stride,
        config.eval_holdout_every,
        config.batch_size,
        config.eval_batches,
        config.train_steps,
        config.learning_rate,
        config.seed,
        config.data_seed,
        device,
    )?;
    write_recurrent_graph_of_experts_report(config, &model_label, &note, artifacts)
}

fn train_dreegmor_model<B, M>(
    model: M,
    corpus_source: &ByteLevelSmokeCorpusSource,
    seq_len: usize,
    window_stride: usize,
    eval_holdout_every: usize,
    batch_size: usize,
    eval_batch_limit: usize,
    train_steps: usize,
    learning_rate: f64,
    seed: u64,
    data_seed: Option<u64>,
    device: &B::Device,
) -> Result<DreegmorSmokeTrainArtifacts, FractalError>
where
    B: AutodiffBackend,
    M: GraphOfExpertsTrainModel<B>,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    B::seed(device, seed);
    let (corpus, train_batches, eval_batches) = load_byte_level_smoke_batches_from_source::<B>(
        corpus_source,
        seq_len,
        window_stride,
        eval_holdout_every,
        batch_size,
        data_seed,
        device,
    )?;
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![BYTE_LEVEL_PAD_TOKEN]))
        .init(device);
    let mut optimizer: OptimizerAdaptor<Adam, M, B> = AdamConfig::new().init::<B, M>();
    let selected_eval_tokens = eval_batches
        .iter()
        .take(eval_batch_limit.min(eval_batches.len()))
        .map(|batch| batch.token_count)
        .sum::<usize>();
    let baseline_process_memory = process_peak_memory_bytes();
    let mut peak_process_memory_bytes = baseline_process_memory;
    let total_start = Instant::now();
    let initial_eval_start = Instant::now();
    let initial_eval = evaluate_model(&model, &criterion, &eval_batches, eval_batch_limit)?;
    let initial_eval_wall_time_ms = initial_eval_start.elapsed().as_secs_f64() * 1000.0;
    peak_process_memory_bytes = peak_process_memory_bytes.max(process_peak_memory_bytes());

    let mut model = model;
    let mut seen_tokens = 0usize;
    let mut train_step_reports = Vec::with_capacity(train_steps);
    let train_start = Instant::now();
    for step in 0..train_steps {
        let batch = &train_batches[step % train_batches.len()];
        let loss = next_token_loss(&model, batch, &criterion)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(learning_rate, model, grads);
        seen_tokens += batch.token_count;
        train_step_reports.push(V2SmokeTrainStepReport {
            step: step + 1,
            learning_rate,
            train_loss,
            train_perplexity: train_loss.exp(),
            seen_tokens,
        });
        peak_process_memory_bytes = peak_process_memory_bytes.max(process_peak_memory_bytes());
    }
    let train_wall_time_ms = train_start.elapsed().as_secs_f64() * 1000.0;
    let final_eval_start = Instant::now();
    let final_eval = evaluate_model(&model, &criterion, &eval_batches, eval_batch_limit)?;
    let final_eval_wall_time_ms = final_eval_start.elapsed().as_secs_f64() * 1000.0;
    peak_process_memory_bytes = peak_process_memory_bytes.max(process_peak_memory_bytes());
    let routing = summarize_routing(&model, &eval_batches, eval_batch_limit)?;
    let total_wall_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let runtime = HybridAttentionRuntimeMetrics {
        total_wall_time_ms,
        initial_eval_wall_time_ms,
        train_wall_time_ms,
        final_eval_wall_time_ms,
        train_tokens_seen: seen_tokens,
        eval_tokens_per_pass: selected_eval_tokens,
        train_tokens_per_second: tokens_per_second(seen_tokens, train_wall_time_ms),
        overall_tokens_per_second: tokens_per_second(
            seen_tokens + (selected_eval_tokens * 2),
            total_wall_time_ms,
        ),
        process_memory_metric: process_memory_metric_kind(),
        peak_process_memory_bytes,
        peak_process_memory_delta_bytes: peak_process_memory_bytes
            .saturating_sub(baseline_process_memory),
        cuda_device_memory: None,
        memory_note: process_memory_measurement_note("around train/eval phases"),
    };
    Ok(DreegmorSmokeTrainArtifacts {
        corpus,
        initial_eval,
        final_eval,
        routing,
        runtime,
        train_steps: train_step_reports,
    })
}

fn summarize_routing<B, M>(
    model: &M,
    eval_batches: &[TokenBatch<B>],
    eval_batch_limit: usize,
) -> Result<GraphOfExpertsRoutingSummary, FractalError>
where
    B: AutodiffBackend,
    M: GraphOfExpertsTrainModel<B>,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    let mut pre_graph_weight_sums = [0.0f64; 2];
    let mut weight_sums = [0.0f64; 2];
    let mut round_adjustment_sums = Vec::<f64>::new();
    let mut round_count = 0usize;
    let mut winner_counts = [0usize; 2];
    let mut tied_token_count = 0usize;
    let mut route_entropy_bits_sum = 0.0f64;
    let mut winner_margin_sum = 0.0f64;
    let mut graph_adjustment_l1_sum = 0.0f64;
    let mut sampled_tokens = 0usize;
    for batch in eval_batches
        .iter()
        .take(eval_batch_limit.min(eval_batches.len()))
    {
        let probe = model.routing_probe(batch.input_ids.clone())?;
        round_count = round_count.max(probe.round_weights.len());
        let pre_graph_values = probe
            .initial_weights
            .to_data()
            .to_vec::<f32>()
            .map_err(invalid_state_from_data("graph_of_experts.initial_weights"))?;
        let final_values = probe
            .final_weights
            .to_data()
            .to_vec::<f32>()
            .map_err(invalid_state_from_data("graph_of_experts.routing_weights"))?;
        let mut round_values = Vec::with_capacity(probe.round_weights.len());
        for (round_index, weights) in probe.round_weights.into_iter().enumerate() {
            let values = weights
                .to_data()
                .to_vec::<f32>()
                .map_err(invalid_state_from_data("graph_of_experts.round_weights"))?;
            if round_adjustment_sums.len() <= round_index {
                round_adjustment_sums.push(0.0);
            }
            round_values.push(values);
        }
        for round_index in 0..round_values.len().saturating_sub(1) {
            for (current_pair, next_pair) in round_values[round_index]
                .chunks_exact(GOE_CHANNEL_COUNT)
                .zip(round_values[round_index + 1].chunks_exact(GOE_CHANNEL_COUNT))
            {
                round_adjustment_sums[round_index] +=
                    (f64::from((current_pair[0] - next_pair[0]).abs())
                        + f64::from((current_pair[1] - next_pair[1]).abs()))
                        / GOE_CHANNEL_COUNT as f64;
            }
        }
        for (pre_pair, final_pair) in pre_graph_values
            .chunks_exact(GOE_CHANNEL_COUNT)
            .zip(final_values.chunks_exact(GOE_CHANNEL_COUNT))
        {
            pre_graph_weight_sums[0] += pre_pair[0] as f64;
            pre_graph_weight_sums[1] += pre_pair[1] as f64;
            weight_sums[0] += final_pair[0] as f64;
            weight_sums[1] += final_pair[1] as f64;
            match decisive_winner(final_pair) {
                Some(winner) => winner_counts[winner] += 1,
                None => tied_token_count += 1,
            }
            route_entropy_bits_sum += route_entropy_bits(final_pair);
            winner_margin_sum += f64::from((final_pair[0] - final_pair[1]).abs());
            graph_adjustment_l1_sum += (f64::from((pre_pair[0] - final_pair[0]).abs())
                + f64::from((pre_pair[1] - final_pair[1]).abs()))
                / GOE_CHANNEL_COUNT as f64;
            sampled_tokens += 1;
        }
    }
    let normalizer = sampled_tokens.max(1) as f64;
    let mean_channel_weights = [weight_sums[0] / normalizer, weight_sums[1] / normalizer];
    Ok(GraphOfExpertsRoutingSummary {
        sampled_tokens,
        round_count,
        mean_pre_graph_channel_weights: [
            pre_graph_weight_sums[0] / normalizer,
            pre_graph_weight_sums[1] / normalizer,
        ],
        mean_channel_weights,
        mean_round_adjustment_l1: round_adjustment_sums
            .into_iter()
            .take(round_count.saturating_sub(1))
            .map(|sum| sum / normalizer)
            .collect(),
        winner_counts,
        tied_token_count,
        active_channel_count: active_channel_count(mean_channel_weights),
        mean_route_entropy_bits: route_entropy_bits_sum / normalizer,
        mean_winner_margin: winner_margin_sum / normalizer,
        mean_graph_adjustment_l1: graph_adjustment_l1_sum / normalizer,
        edge_mix_fraction: model.edge_mix_fraction()?,
    })
}

fn write_graph_of_experts_report(
    config: GraphOfExpertsSmokeTrainConfig,
    model_label: &str,
    note: &str,
    artifacts: DreegmorSmokeTrainArtifacts,
) -> Result<GraphOfExpertsSmokeTrainReport, FractalError> {
    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create GoE output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let report_path = config.output_dir.join("report.json");
    let report = GraphOfExpertsSmokeTrainReport {
        model_label: model_label.to_string(),
        note: note.to_string(),
        config,
        corpus: artifacts.corpus,
        initial_eval: artifacts.initial_eval,
        final_eval: artifacts.final_eval,
        routing: artifacts.routing,
        runtime: artifacts.runtime,
        train_steps: artifacts.train_steps,
        report_path: report_path.clone(),
    };
    write_report(&report, &report_path)?;
    Ok(report)
}

fn write_recurrent_graph_of_experts_report(
    config: RecurrentGraphOfExpertsSmokeTrainConfig,
    model_label: &str,
    note: &str,
    artifacts: DreegmorSmokeTrainArtifacts,
) -> Result<RecurrentGraphOfExpertsSmokeTrainReport, FractalError> {
    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create recurrent GoE output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let report_path = config.output_dir.join("report.json");
    let report = RecurrentGraphOfExpertsSmokeTrainReport {
        model_label: model_label.to_string(),
        note: note.to_string(),
        config,
        corpus: artifacts.corpus,
        initial_eval: artifacts.initial_eval,
        final_eval: artifacts.final_eval,
        routing: artifacts.routing,
        runtime: artifacts.runtime,
        train_steps: artifacts.train_steps,
        report_path: report_path.clone(),
    };
    write_recurrent_report(&report, &report_path)?;
    Ok(report)
}

fn controller_model_tag(controller: GraphOfExpertsControllerSpec) -> &'static str {
    match controller.label_suffix() {
        "uniform-average" => "uniform_average",
        "routed-no-graph-mix" => "routed_no_graph_mix",
        "routed-graph-mix" => "routed_graph_mix",
        _ => "invalid",
    }
}

fn attention_only_goe_note(controller: GraphOfExpertsControllerSpec) -> String {
    format!(
        "Exploratory DREEGMOR over the frozen A surface using two dense attention-only experts with controller structure {} on the shared byte-level smoke lane",
        controller.label_suffix()
    )
}

fn reference_ssm_goe_note(controller: GraphOfExpertsControllerSpec) -> String {
    format!(
        "Exploratory DREEGMOR over the frozen A + M surface using two dense reference-SSM experts with controller structure {} on the shared byte-level smoke lane",
        controller.label_suffix()
    )
}

fn recurrent_router_model_tag(router: fractal_core::RecurrentRouterSpec) -> String {
    router.label_suffix().replace('-', "_")
}

fn attention_only_recurrent_goe_note(router: fractal_core::RecurrentRouterSpec) -> String {
    format!(
        "Exploratory recurrent-routing DREEGMOR over the frozen A surface using two dense attention-only experts with virtual-node controller structure {} on the shared byte-level smoke lane",
        router.label_suffix()
    )
}

fn route_entropy_bits(weights: &[f32]) -> f64 {
    weights
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| {
            let probability = probability as f64;
            -probability * probability.log2()
        })
        .sum()
}

fn decisive_winner(weights: &[f32]) -> Option<usize> {
    if weights.len() != GOE_CHANNEL_COUNT
        || (weights[0] - weights[1]).abs() <= GOE_ROUTE_TIE_EPSILON
    {
        None
    } else if weights[1] > weights[0] {
        Some(1)
    } else {
        Some(0)
    }
}

fn active_channel_count(mean_channel_weights: [f64; GOE_CHANNEL_COUNT]) -> usize {
    mean_channel_weights
        .iter()
        .copied()
        .filter(|weight| *weight >= GOE_ACTIVE_CHANNEL_MEAN_WEIGHT_FLOOR)
        .count()
}

fn write_report(report: &GraphOfExpertsSmokeTrainReport, path: &Path) -> Result<(), FractalError> {
    let rendered = serde_json::to_string_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!("failed to serialize GoE smoke report: {error}"))
    })?;
    fs::write(path, rendered).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write GoE smoke report {}: {error}",
            path.display()
        ))
    })
}

fn write_recurrent_report(
    report: &RecurrentGraphOfExpertsSmokeTrainReport,
    path: &Path,
) -> Result<(), FractalError> {
    let rendered = serde_json::to_string_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize recurrent GoE smoke report: {error}"
        ))
    })?;
    fs::write(path, rendered).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write recurrent GoE smoke report {}: {error}",
            path.display()
        ))
    })
}

fn tokens_per_second(tokens: usize, wall_time_ms: f64) -> f64 {
    if wall_time_ms <= f64::EPSILON {
        0.0
    } else {
        tokens as f64 / (wall_time_ms / 1000.0)
    }
}

fn invalid_state_from_data(
    subject: &'static str,
) -> impl FnOnce(burn::tensor::DataError) -> FractalError {
    move |error| {
        FractalError::InvalidState(format!(
            "{subject} could not be materialized for inspection: {error}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{
        append_goe_results_ledger_entry, default_goe_results_ledger_path, read_goe_results_ledger,
        resolve_requested_goe_results_ledger_path, GoeResultsLedgerEntry, GoeResultsLedgerKind,
        GraphOfExpertsExperimentReport, GraphOfExpertsExperimentVariantKind,
        GraphOfExpertsExperimentVariantSummary,
    };
    use crate::{
        HybridAttentionExecutionBackend, HybridAttentionRuntimeMetrics, V2SmokeEvalMetrics,
    };

    #[test]
    fn route_entropy_bits_matches_two_channel_bounds() {
        assert!((super::route_entropy_bits(&[0.5, 0.5]) - 1.0).abs() < 1.0e-6);
        assert!(super::route_entropy_bits(&[1.0, 0.0]).abs() < 1.0e-6);
    }

    #[test]
    fn decisive_winner_treats_balanced_routes_as_ties() {
        assert_eq!(super::decisive_winner(&[0.5, 0.5]), None);
        assert_eq!(super::decisive_winner(&[0.50000006, 0.5]), None);
        assert_eq!(super::decisive_winner(&[0.6, 0.4]), Some(0));
        assert_eq!(super::decisive_winner(&[0.4, 0.6]), Some(1));
    }

    #[test]
    fn active_channel_count_requires_nontrivial_mean_weight() {
        assert_eq!(super::active_channel_count([0.5, 0.5]), 2);
        assert_eq!(super::active_channel_count([0.9, 0.1]), 2);
        assert_eq!(super::active_channel_count([0.95, 0.05]), 1);
    }

    #[test]
    fn default_goe_ledger_path_points_to_docs_jsonl() {
        let path = default_goe_results_ledger_path("/tmp/fractal");
        assert_eq!(
            path,
            PathBuf::from("/tmp/fractal/docs/dreegmor-a-am-results-ledger.jsonl")
        );
    }

    #[test]
    fn resolve_requested_goe_ledger_supports_default() {
        let path =
            resolve_requested_goe_results_ledger_path("/tmp/fractal", Some("default")).unwrap();
        assert_eq!(
            path,
            Some(PathBuf::from(
                "/tmp/fractal/docs/dreegmor-a-am-results-ledger.jsonl"
            ))
        );
    }

    #[test]
    fn append_and_read_goe_ledger_round_trip() {
        let root = std::env::temp_dir().join(format!("fractal-goe-ledger-{}", std::process::id()));
        let ledger_path = root.join("ledger.jsonl");
        let report = GraphOfExpertsExperimentReport {
            note: "goe".to_string(),
            variants: vec![GraphOfExpertsExperimentVariantSummary {
                kind: GraphOfExpertsExperimentVariantKind::AttentionOnlyBaseline,
                label: "A".to_string(),
                variant_label: "attention-only".to_string(),
                model_label: "baseline".to_string(),
                note: "baseline".to_string(),
                seed: 42,
                execution_backend: HybridAttentionExecutionBackend::Cpu,
                initial_eval: V2SmokeEvalMetrics {
                    mean_loss: 1.0,
                    perplexity: 2.0,
                    batch_count: 1,
                },
                final_eval: V2SmokeEvalMetrics {
                    mean_loss: 0.5,
                    perplexity: 1.5,
                    batch_count: 1,
                },
                runtime: HybridAttentionRuntimeMetrics {
                    total_wall_time_ms: 1.0,
                    initial_eval_wall_time_ms: 1.0,
                    train_wall_time_ms: 1.0,
                    final_eval_wall_time_ms: 1.0,
                    train_tokens_seen: 1,
                    eval_tokens_per_pass: 1,
                    train_tokens_per_second: 1.0,
                    overall_tokens_per_second: 1.0,
                    process_memory_metric: crate::ProcessMemoryMetricKind::PeakRss,
                    peak_process_memory_bytes: 1,
                    peak_process_memory_delta_bytes: 1,
                    cuda_device_memory: None,
                    memory_note: "note".to_string(),
                },
                train_step_count: 1,
                report_path: PathBuf::from("/tmp/report.json"),
                routing: None,
            }],
        };
        let entry = GoeResultsLedgerEntry::aam_experiment_run(
            "dreegmor_a_am_experiment",
            "note",
            &report,
            Some("smoke".to_string()),
        )
        .unwrap();
        append_goe_results_ledger_entry(&ledger_path, &entry).unwrap();
        let entries = read_goe_results_ledger(&ledger_path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, GoeResultsLedgerKind::AAmExperimentRun);
        assert_eq!(entries[0].run_label.as_deref(), Some("smoke"));
        let _ = fs::remove_file(ledger_path);
        let _ = fs::remove_dir_all(root);
    }
}
