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
    tensor::backend::{AutodiffBackend, Backend},
};
use serde::{Deserialize, Serialize};

use fractal_core::{
    build_attention_only_hybrid_attention_model, build_primitive_hybrid_attention_model,
    build_rust_mamba3_reference_hybrid_attention_model, error::FractalError,
    AttentionOnlyHybridAttentionModel, HybridAttentionVariantKind, HybridAttentionVariantSpec,
    PrimitiveHybridAttentionModel, RustMamba3ReferenceHybridAttentionModel,
};

use crate::{
    v2_training::{evaluate_model, load_byte_level_smoke_batches_from_source, next_token_loss},
    ByteLevelSmokeCorpusSource, ByteLevelVocabularyContract, V2SmokeCorpusStats,
    V2SmokeEvalMetrics, V2SmokeTrainModel, V2SmokeTrainStepReport, BYTE_LEVEL_PAD_TOKEN,
    BYTE_LEVEL_VOCAB_SIZE,
};

pub const DEFAULT_V3A_SMOKE_SEQ_LEN: usize = 32;
pub const DEFAULT_V3A_SMOKE_WINDOW_STRIDE: usize = DEFAULT_V3A_SMOKE_SEQ_LEN;
pub const DEFAULT_V3A_SMOKE_BATCH_SIZE: usize = 1;
pub const DEFAULT_V3A_SMOKE_TRAIN_STEPS: usize = 8;
pub const DEFAULT_V3A_SMOKE_EVAL_BATCHES: usize = 2;
pub const DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY: usize = 10;
pub const DEFAULT_V3A_SMOKE_LEARNING_RATE: f64 = 1e-3;
pub const DEFAULT_V3A_SMOKE_SEED: u64 = 42;

pub const HYBRID_ATTENTION_RUNTIME_MEMORY_NOTE: &str =
    "process RSS metrics are sampled from getrusage(RUSAGE_SELF) around train/eval phases; they are useful for trend detection, not precise kernel-level attribution";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridAttentionExecutionBackend {
    Cpu,
    Metal,
}

impl HybridAttentionExecutionBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridAttentionSmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub variant: HybridAttentionVariantSpec,
    pub execution_backend: HybridAttentionExecutionBackend,
    pub seq_len: usize,
    pub window_stride: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub eval_batches: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub vocabulary: ByteLevelVocabularyContract,
}

impl HybridAttentionSmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        variant: HybridAttentionVariantSpec,
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
            seed: DEFAULT_V3A_SMOKE_SEED,
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
                "hybrid_attention_smoke_train sizes and counts must be greater than zero"
                    .to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_smoke_train.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_smoke_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.variant.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridAttentionSmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: HybridAttentionSmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub runtime: HybridAttentionRuntimeMetrics,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridAttentionRuntimeMetrics {
    pub total_wall_time_ms: f64,
    pub initial_eval_wall_time_ms: f64,
    pub train_wall_time_ms: f64,
    pub final_eval_wall_time_ms: f64,
    pub train_tokens_seen: usize,
    pub eval_tokens_per_pass: usize,
    pub train_tokens_per_second: f64,
    pub overall_tokens_per_second: f64,
    pub peak_rss_bytes: u64,
    pub peak_rss_delta_bytes: u64,
    pub memory_note: String,
}

impl<B: Backend> V2SmokeTrainModel<B> for AttentionOnlyHybridAttentionModel<B> {
    fn forward_logits(
        &self,
        input_ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    ) -> Result<burn::tensor::Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }
}

impl<B: Backend> V2SmokeTrainModel<B> for PrimitiveHybridAttentionModel<B> {
    fn forward_logits(
        &self,
        input_ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    ) -> Result<burn::tensor::Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }
}

impl<B: Backend> V2SmokeTrainModel<B> for RustMamba3ReferenceHybridAttentionModel<B> {
    fn forward_logits(
        &self,
        input_ids: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    ) -> Result<burn::tensor::Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }
}

pub fn run_attention_only_hybrid_attention_smoke_train<B>(
    config: HybridAttentionSmokeTrainConfig,
    device: &B::Device,
) -> Result<HybridAttentionSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_attention_only_hybrid_attention_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    run_hybrid_attention_smoke_train_with_model(
        model,
        config,
        "v3a_attention_only_baseline",
        "Path 1 attention-only baseline on the shared byte-level smoke lane",
        device,
    )
}

pub fn run_primitive_hybrid_attention_smoke_train<B>(
    config: HybridAttentionSmokeTrainConfig,
    device: &B::Device,
) -> Result<HybridAttentionSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_primitive_hybrid_attention_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    run_hybrid_attention_smoke_train_with_model(
        model,
        config,
        "v3a_primitive_hybrid_baseline",
        "Path 1 primitive-hybrid baseline on the shared byte-level smoke lane",
        device,
    )
}

pub fn run_reference_ssm_hybrid_attention_smoke_train<B>(
    config: HybridAttentionSmokeTrainConfig,
    device: &B::Device,
) -> Result<HybridAttentionSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_rust_mamba3_reference_hybrid_attention_model::<B>(
        config.vocabulary.vocab_size,
        &config.variant,
        device,
    )?;
    run_hybrid_attention_smoke_train_with_model(
        model,
        config,
        "v3a_reference_ssm_rust_mamba3_baseline",
        "Path 1 reference SSM hybrid baseline using the faithful Rust Mamba-3-style lane on the shared byte-level smoke lane",
        device,
    )
}

fn run_hybrid_attention_smoke_train_with_model<B, M>(
    model: M,
    config: HybridAttentionSmokeTrainConfig,
    model_label: &str,
    note: &str,
    device: &B::Device,
) -> Result<HybridAttentionSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
    M: V2SmokeTrainModel<B> + AutodiffModule<B> + Module<B> + Clone,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    config.validate()?;
    B::seed(device, config.seed);
    let (corpus, train_batches, eval_batches) = load_byte_level_smoke_batches_from_source::<B>(
        &config.corpus_source,
        config.seq_len,
        config.window_stride,
        config.eval_holdout_every,
        config.batch_size,
        device,
    )?;
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![BYTE_LEVEL_PAD_TOKEN]))
        .init(device);
    let mut optimizer: OptimizerAdaptor<Adam, M, B> = AdamConfig::new().init::<B, M>();
    let selected_eval_tokens = eval_batches
        .iter()
        .take(config.eval_batches.min(eval_batches.len()))
        .map(|batch| batch.token_count)
        .sum::<usize>();
    let baseline_rss = process_peak_rss_bytes();
    let mut peak_rss_bytes = baseline_rss;
    let total_start = Instant::now();
    let initial_eval_start = Instant::now();
    let initial_eval = evaluate_model(&model, &criterion, &eval_batches, config.eval_batches)?;
    let initial_eval_wall_time_ms = initial_eval_start.elapsed().as_secs_f64() * 1000.0;
    peak_rss_bytes = peak_rss_bytes.max(process_peak_rss_bytes());

    let mut model = model;
    let mut seen_tokens = 0usize;
    let mut train_steps = Vec::with_capacity(config.train_steps);
    let train_start = Instant::now();
    for step in 0..config.train_steps {
        let batch = &train_batches[step % train_batches.len()];
        let loss = next_token_loss(&model, batch, &criterion)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        seen_tokens += batch.token_count;
        train_steps.push(V2SmokeTrainStepReport {
            step: step + 1,
            learning_rate: config.learning_rate,
            train_loss,
            train_perplexity: train_loss.exp(),
            seen_tokens,
        });
        peak_rss_bytes = peak_rss_bytes.max(process_peak_rss_bytes());
    }
    let train_wall_time_ms = train_start.elapsed().as_secs_f64() * 1000.0;
    let final_eval_start = Instant::now();
    let final_eval = evaluate_model(&model, &criterion, &eval_batches, config.eval_batches)?;
    let final_eval_wall_time_ms = final_eval_start.elapsed().as_secs_f64() * 1000.0;
    peak_rss_bytes = peak_rss_bytes.max(process_peak_rss_bytes());
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
        peak_rss_bytes,
        peak_rss_delta_bytes: peak_rss_bytes.saturating_sub(baseline_rss),
        memory_note: HYBRID_ATTENTION_RUNTIME_MEMORY_NOTE.to_string(),
    };

    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create hybrid attention output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let report_path = config.output_dir.join("report.json");
    let report = HybridAttentionSmokeTrainReport {
        model_label: model_label.to_string(),
        note: note.to_string(),
        config,
        corpus,
        initial_eval,
        final_eval,
        runtime,
        train_steps,
        report_path: report_path.clone(),
    };
    write_report(&report, &report_path)?;
    Ok(report)
}

fn write_report(report: &HybridAttentionSmokeTrainReport, path: &Path) -> Result<(), FractalError> {
    let rendered = serde_json::to_string_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize hybrid attention smoke report: {error}"
        ))
    })?;
    fs::write(path, rendered).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write hybrid attention smoke report {}: {error}",
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

fn process_peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    #[cfg(target_os = "macos")]
    {
        usage.ru_maxrss as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        (usage.ru_maxrss as u64) * 1024
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HybridAttentionMatrixVariantOutcome {
    Executed(Box<HybridAttentionSmokeTrainReport>),
    Skipped {
        label: String,
        kind: HybridAttentionVariantKind,
        reason: String,
    },
    RequiredMissing {
        label: String,
        kind: HybridAttentionVariantKind,
        reason: String,
    },
}
