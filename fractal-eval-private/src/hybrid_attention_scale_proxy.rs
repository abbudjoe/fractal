use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
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

use fractal_core::{
    build_attention_only_recurrent_scale_proxy_model, build_attention_only_scale_proxy_model,
    error::FractalError, AttentionOnlyRecurrentScaleProxyModel,
    AttentionOnlyRecurrentScaleProxyVariantSpec, AttentionOnlyScaleProxyModel,
    AttentionOnlyScaleProxyVariantSpec, ScaleProxyRoutingProbe, TokenBatch,
    SCALE_PROXY_CHANNEL_COUNT,
};

use crate::{
    hybrid_attention_training::{
        HybridAttentionExecutionBackend, HybridAttentionRuntimeMetrics,
        HYBRID_ATTENTION_RUNTIME_MEMORY_NOTE,
    },
    process_memory_metric_kind, process_peak_memory_bytes,
    v2_training::{evaluate_model, load_byte_level_smoke_batches_from_source, next_token_loss},
    ByteLevelSmokeCorpusSource, ByteLevelVocabularyContract, V2SmokeCorpusStats,
    V2SmokeEvalMetrics, V2SmokeTrainModel, V2SmokeTrainStepReport, BYTE_LEVEL_PAD_TOKEN,
    BYTE_LEVEL_VOCAB_SIZE, DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE,
    DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};

const SCALE_PROXY_ACTIVE_CHANNEL_MEAN_WEIGHT_FLOOR: f64 = 0.10;
const SCALE_PROXY_ROUTE_TIE_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleProxySmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub variant: AttentionOnlyScaleProxyVariantSpec,
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

impl ScaleProxySmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        variant: AttentionOnlyScaleProxyVariantSpec,
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
                "scale_proxy_smoke_train sizes and counts must be greater than zero".to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "scale_proxy_smoke_train.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "scale_proxy_smoke_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "scale_proxy_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.variant.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentScaleProxySmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub variant: AttentionOnlyRecurrentScaleProxyVariantSpec,
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

impl RecurrentScaleProxySmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        variant: AttentionOnlyRecurrentScaleProxyVariantSpec,
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
                "recurrent_scale_proxy_smoke_train sizes and counts must be greater than zero"
                    .to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "recurrent_scale_proxy_smoke_train.eval_holdout_every must be at least 2"
                    .to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "recurrent_scale_proxy_smoke_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "recurrent_scale_proxy_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.variant.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleProxyRoutingSummary {
    pub sampled_tokens: usize,
    pub round_count: usize,
    pub mean_initial_channel_weights: [f64; SCALE_PROXY_CHANNEL_COUNT],
    pub mean_final_channel_weights: [f64; SCALE_PROXY_CHANNEL_COUNT],
    pub mean_round_adjustment_l1: Vec<f64>,
    pub winner_counts: [usize; SCALE_PROXY_CHANNEL_COUNT],
    pub tied_token_count: usize,
    pub active_channel_count: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_controller_adjustment_l1: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleProxySmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: ScaleProxySmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub routing: ScaleProxyRoutingSummary,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecurrentScaleProxySmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: RecurrentScaleProxySmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub routing: ScaleProxyRoutingSummary,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

impl<B: Backend> V2SmokeTrainModel<B> for AttentionOnlyScaleProxyModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

impl<B: Backend> V2SmokeTrainModel<B> for AttentionOnlyRecurrentScaleProxyModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

#[derive(Debug)]
struct ScaleProxyProbeTensors<B: Backend> {
    initial_weights: Tensor<B, 3>,
    final_weights: Tensor<B, 3>,
    round_weights: Vec<Tensor<B, 3>>,
}

trait ScaleProxyTrainModel<B>: V2SmokeTrainModel<B> + AutodiffModule<B> + Module<B> + Clone
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<ScaleProxyProbeTensors<B>, FractalError>;
}

impl<B> ScaleProxyTrainModel<B> for AttentionOnlyScaleProxyModel<B>
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<ScaleProxyProbeTensors<B>, FractalError> {
        let probe: ScaleProxyRoutingProbe<B> = self.routing_probe(input_ids)?;
        Ok(ScaleProxyProbeTensors {
            initial_weights: probe.initial_weights,
            final_weights: probe.final_weights,
            round_weights: probe.round_weights,
        })
    }
}

impl<B> ScaleProxyTrainModel<B> for AttentionOnlyRecurrentScaleProxyModel<B>
where
    B: AutodiffBackend,
    <Self as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<ScaleProxyProbeTensors<B>, FractalError> {
        let probe: ScaleProxyRoutingProbe<B> = self.routing_probe(input_ids)?;
        Ok(ScaleProxyProbeTensors {
            initial_weights: probe.initial_weights,
            final_weights: probe.final_weights,
            round_weights: probe.round_weights,
        })
    }
}

#[derive(Debug)]
struct ScaleProxySmokeTrainArtifacts {
    corpus: V2SmokeCorpusStats,
    initial_eval: V2SmokeEvalMetrics,
    final_eval: V2SmokeEvalMetrics,
    routing: ScaleProxyRoutingSummary,
    runtime: HybridAttentionRuntimeMetrics,
    train_steps: Vec<V2SmokeTrainStepReport>,
}

pub fn run_attention_only_scale_proxy_smoke_train<B>(
    config: ScaleProxySmokeTrainConfig,
    device: &B::Device,
) -> Result<ScaleProxySmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let model_label = "dreegmor_scale_proxy_attention_only_one_shot";
    let note = attention_only_scale_proxy_note(&config.variant);
    let model = build_attention_only_scale_proxy_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    let artifacts = train_scale_proxy_model(
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
    write_scale_proxy_report(config, model_label, &note, artifacts)
}

pub fn run_attention_only_recurrent_scale_proxy_smoke_train<B>(
    config: RecurrentScaleProxySmokeTrainConfig,
    device: &B::Device,
) -> Result<RecurrentScaleProxySmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let model_label = format!(
        "dreegmor_scale_proxy_attention_only_recurrent_{}",
        config.variant.router.label_suffix().replace('-', "_")
    );
    let note = attention_only_recurrent_scale_proxy_note(&config.variant);
    let model = build_attention_only_recurrent_scale_proxy_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    let artifacts = train_scale_proxy_model(
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
    write_recurrent_scale_proxy_report(config, &model_label, &note, artifacts)
}

fn train_scale_proxy_model<B, M>(
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
) -> Result<ScaleProxySmokeTrainArtifacts, FractalError>
where
    B: AutodiffBackend,
    M: ScaleProxyTrainModel<B>,
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
        memory_note: HYBRID_ATTENTION_RUNTIME_MEMORY_NOTE.to_string(),
    };
    Ok(ScaleProxySmokeTrainArtifacts {
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
) -> Result<ScaleProxyRoutingSummary, FractalError>
where
    B: AutodiffBackend,
    M: ScaleProxyTrainModel<B>,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    let mut initial_weight_sums = [0.0f64; SCALE_PROXY_CHANNEL_COUNT];
    let mut final_weight_sums = [0.0f64; SCALE_PROXY_CHANNEL_COUNT];
    let mut round_adjustment_sums = Vec::<f64>::new();
    let mut round_count = 0usize;
    let mut winner_counts = [0usize; SCALE_PROXY_CHANNEL_COUNT];
    let mut tied_token_count = 0usize;
    let mut route_entropy_bits_sum = 0.0f64;
    let mut winner_margin_sum = 0.0f64;
    let mut controller_adjustment_l1_sum = 0.0f64;
    let mut sampled_tokens = 0usize;

    for batch in eval_batches
        .iter()
        .take(eval_batch_limit.min(eval_batches.len()))
    {
        let probe = model.routing_probe(batch.input_ids.clone())?;
        round_count = round_count.max(probe.round_weights.len());
        let initial_values = probe
            .initial_weights
            .to_data()
            .to_vec::<f32>()
            .map_err(invalid_state_from_data("scale_proxy.initial_weights"))?;
        let final_values = probe
            .final_weights
            .to_data()
            .to_vec::<f32>()
            .map_err(invalid_state_from_data("scale_proxy.final_weights"))?;
        let mut round_values = Vec::with_capacity(probe.round_weights.len());
        for (round_index, weights) in probe.round_weights.into_iter().enumerate() {
            let values = weights
                .to_data()
                .to_vec::<f32>()
                .map_err(invalid_state_from_data("scale_proxy.round_weights"))?;
            if round_adjustment_sums.len() <= round_index {
                round_adjustment_sums.push(0.0);
            }
            round_values.push(values);
        }
        for round_index in 0..round_values.len().saturating_sub(1) {
            for (current_pair, next_pair) in round_values[round_index]
                .chunks_exact(SCALE_PROXY_CHANNEL_COUNT)
                .zip(round_values[round_index + 1].chunks_exact(SCALE_PROXY_CHANNEL_COUNT))
            {
                round_adjustment_sums[round_index] +=
                    (f64::from((current_pair[0] - next_pair[0]).abs())
                        + f64::from((current_pair[1] - next_pair[1]).abs()))
                        / SCALE_PROXY_CHANNEL_COUNT as f64;
            }
        }
        for (initial_pair, final_pair) in initial_values
            .chunks_exact(SCALE_PROXY_CHANNEL_COUNT)
            .zip(final_values.chunks_exact(SCALE_PROXY_CHANNEL_COUNT))
        {
            initial_weight_sums[0] += initial_pair[0] as f64;
            initial_weight_sums[1] += initial_pair[1] as f64;
            final_weight_sums[0] += final_pair[0] as f64;
            final_weight_sums[1] += final_pair[1] as f64;
            match decisive_winner(final_pair) {
                Some(winner) => winner_counts[winner] += 1,
                None => tied_token_count += 1,
            }
            route_entropy_bits_sum += route_entropy_bits(final_pair);
            winner_margin_sum += f64::from((final_pair[0] - final_pair[1]).abs());
            controller_adjustment_l1_sum += (f64::from((initial_pair[0] - final_pair[0]).abs())
                + f64::from((initial_pair[1] - final_pair[1]).abs()))
                / SCALE_PROXY_CHANNEL_COUNT as f64;
            sampled_tokens += 1;
        }
    }
    let normalizer = sampled_tokens.max(1) as f64;
    let mean_final_channel_weights = [
        final_weight_sums[0] / normalizer,
        final_weight_sums[1] / normalizer,
    ];
    Ok(ScaleProxyRoutingSummary {
        sampled_tokens,
        round_count,
        mean_initial_channel_weights: [
            initial_weight_sums[0] / normalizer,
            initial_weight_sums[1] / normalizer,
        ],
        mean_final_channel_weights,
        mean_round_adjustment_l1: round_adjustment_sums
            .into_iter()
            .take(round_count.saturating_sub(1))
            .map(|sum| sum / normalizer)
            .collect(),
        winner_counts,
        tied_token_count,
        active_channel_count: active_channel_count(mean_final_channel_weights),
        mean_route_entropy_bits: route_entropy_bits_sum / normalizer,
        mean_winner_margin: winner_margin_sum / normalizer,
        mean_controller_adjustment_l1: controller_adjustment_l1_sum / normalizer,
    })
}

fn write_scale_proxy_report(
    config: ScaleProxySmokeTrainConfig,
    model_label: &str,
    note: &str,
    artifacts: ScaleProxySmokeTrainArtifacts,
) -> Result<ScaleProxySmokeTrainReport, FractalError> {
    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create scale-proxy output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let report_path = config.output_dir.join("report.json");
    let report = ScaleProxySmokeTrainReport {
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

fn write_recurrent_scale_proxy_report(
    config: RecurrentScaleProxySmokeTrainConfig,
    model_label: &str,
    note: &str,
    artifacts: ScaleProxySmokeTrainArtifacts,
) -> Result<RecurrentScaleProxySmokeTrainReport, FractalError> {
    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create recurrent scale-proxy output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let report_path = config.output_dir.join("report.json");
    let report = RecurrentScaleProxySmokeTrainReport {
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

fn attention_only_scale_proxy_note(variant: &AttentionOnlyScaleProxyVariantSpec) -> String {
    format!(
        "Exploratory DREEGMOR scale-proxy over the A surface using a shared trunk with one internally expertized attention block at layer {} and hidden-state one-shot routing on the shared byte-level smoke lane",
        variant.expert_layer_index + 1
    )
}

fn attention_only_recurrent_scale_proxy_note(
    variant: &AttentionOnlyRecurrentScaleProxyVariantSpec,
) -> String {
    format!(
        "Exploratory DREEGMOR scale-proxy over the A surface using a shared trunk with one internally expertized attention block at layer {} and recurrent hidden-state routing {} on the shared byte-level smoke lane",
        variant.expert_layer_index + 1,
        variant.router.label_suffix()
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
    if weights.len() != SCALE_PROXY_CHANNEL_COUNT
        || (weights[0] - weights[1]).abs() <= SCALE_PROXY_ROUTE_TIE_EPSILON
    {
        None
    } else if weights[1] > weights[0] {
        Some(1)
    } else {
        Some(0)
    }
}

fn active_channel_count(mean_channel_weights: [f64; SCALE_PROXY_CHANNEL_COUNT]) -> usize {
    mean_channel_weights
        .iter()
        .copied()
        .filter(|weight| *weight >= SCALE_PROXY_ACTIVE_CHANNEL_MEAN_WEIGHT_FLOOR)
        .count()
}

fn write_report(report: &ScaleProxySmokeTrainReport, path: &Path) -> Result<(), FractalError> {
    let rendered = serde_json::to_string_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize scale-proxy smoke report: {error}"
        ))
    })?;
    fs::write(path, rendered).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write scale-proxy smoke report {}: {error}",
            path.display()
        ))
    })
}

fn write_recurrent_report(
    report: &RecurrentScaleProxySmokeTrainReport,
    path: &Path,
) -> Result<(), FractalError> {
    let rendered = serde_json::to_string_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize recurrent scale-proxy smoke report: {error}"
        ))
    })?;
    fs::write(path, rendered).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write recurrent scale-proxy smoke report {}: {error}",
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
