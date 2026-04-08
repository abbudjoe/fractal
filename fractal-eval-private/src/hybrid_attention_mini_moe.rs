use std::{
    fs,
    path::PathBuf,
    time::Instant,
};

use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use serde::{Deserialize, Serialize};

use fractal_core::hybrid_attention::{
    BenchmarkPolicy, ConfiguredMiniMoeFfn, ExecutionIsolationMode, MiniMoeBackendKind,
    MiniMoeBackendSpec, MiniMoeDataSpec, MiniMoeEvalMetrics, MiniMoeEvalSpec, MiniMoeModel,
    MiniMoeRunArtifact, MiniMoeRunManifest, MiniMoeSurfaceSpec, MiniMoeSystemMetrics,
    MiniMoeTrainMetrics, MiniMoeTrainSpec,
};
use fractal_core::{error::FractalError, TokenBatch};

use crate::{
    process_memory_metric_kind, process_peak_memory_bytes,
    v2_training::{evaluate_model, load_byte_level_smoke_batches_from_source, next_token_loss},
    ByteLevelSmokeCorpusSource, ByteLevelVocabularyContract, V2SmokeCorpusStats,
    V2SmokeEvalMetrics, V2SmokeTrainModel, V2SmokeTrainStepReport, BYTE_LEVEL_PAD_TOKEN,
    BYTE_LEVEL_VOCAB_SIZE, DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE,
    DEFAULT_V3A_SMOKE_SEED, DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS,
    DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeSmokeTrainConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub manifest: MiniMoeRunManifest,
    pub vocabulary: ByteLevelVocabularyContract,
}

impl MiniMoeSmokeTrainConfig {
    pub fn new(
        corpus_source: ByteLevelSmokeCorpusSource,
        output_dir: PathBuf,
        surface: MiniMoeSurfaceSpec,
    ) -> Result<Self, FractalError> {
        surface.validate()?;
        Ok(Self {
            corpus_source,
            output_dir,
            manifest: MiniMoeRunManifest {
                resolved_layout: surface.resolve_layout()?,
                resolved_dispatch: surface.resolve_dispatch_contract(),
                surface,
                data: MiniMoeDataSpec {
                    seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
                    window_stride: DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
                },
                train: MiniMoeTrainSpec {
                    steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
                    batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
                    learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
                    model_seed: DEFAULT_V3A_SMOKE_SEED,
                    data_seed: None,
                },
                eval: MiniMoeEvalSpec {
                    eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
                    full_eval_pass: false,
                },
                backend: MiniMoeBackendSpec {
                    backend: MiniMoeBackendKind::Cpu,
                },
                benchmark_policy: BenchmarkPolicy::Smoke,
                isolation_mode: ExecutionIsolationMode::SharedProcess,
            },
            vocabulary: ByteLevelVocabularyContract::default(),
        })
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.corpus_source.validate()?;
        self.manifest.validate()?;
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN
            || self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE
        {
            return Err(FractalError::InvalidConfig(
                "mini_moe_smoke_train must remain on the shared byte-level vocabulary contract"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeSmokeTrainReport {
    pub model_label: String,
    pub note: String,
    pub config: MiniMoeSmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub artifact: MiniMoeRunArtifact,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub report_path: PathBuf,
}

impl<B: Backend> V2SmokeTrainModel<B> for MiniMoeModel<B, ConfiguredMiniMoeFfn<B>> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_logits(input_ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

pub fn run_mini_moe_smoke_train<B>(
    config: MiniMoeSmokeTrainConfig,
    device: &B::Device,
) -> Result<MiniMoeSmokeTrainReport, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    B::seed(device, config.manifest.train.model_seed);
    fs::create_dir_all(&config.output_dir).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to create mini-moe output directory {}: {error}",
            config.output_dir.display()
        ))
    })?;

    let artifacts = run_mini_moe_smoke_train_artifacts::<B>(&config, device)?;
    let report_path = config.output_dir.join("report.json");
    let report = MiniMoeSmokeTrainReport {
        model_label: config.manifest.surface.architecture.label.clone(),
        note: "Mini-MoE smoke train with bounded routing trace summaries aggregated over selected eval batches".to_string(),
        config,
        corpus: artifacts.corpus,
        initial_eval: artifacts.initial_eval,
        final_eval: artifacts.final_eval,
        artifact: artifacts.artifact,
        train_steps: artifacts.train_steps,
        report_path: report_path.clone(),
    };
    let report_json = serde_json::to_vec_pretty(&report).map_err(|error| {
        FractalError::InvalidState(format!("failed to serialize mini-moe report: {error}"))
    })?;
    fs::write(&report_path, report_json).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write mini-moe report {}: {error}",
            report_path.display()
        ))
    })?;
    Ok(report)
}

#[derive(Debug)]
struct MiniMoeSmokeTrainArtifacts {
    corpus: V2SmokeCorpusStats,
    initial_eval: V2SmokeEvalMetrics,
    final_eval: V2SmokeEvalMetrics,
    artifact: MiniMoeRunArtifact,
    train_steps: Vec<V2SmokeTrainStepReport>,
}

fn run_mini_moe_smoke_train_artifacts<B>(
    config: &MiniMoeSmokeTrainConfig,
    device: &B::Device,
) -> Result<MiniMoeSmokeTrainArtifacts, FractalError>
where
    B: AutodiffBackend,
{
    let manifest = &config.manifest;
    let (corpus, train_batches, eval_batches) = load_byte_level_smoke_batches_from_source::<B>(
        &config.corpus_source,
        manifest.data.seq_len,
        manifest.data.window_stride,
        DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
        manifest.train.batch_size,
        manifest.train.data_seed,
        device,
    )?;
    let eval_batch_limit = resolve_eval_batch_limit(&manifest.eval, eval_batches.len());
    let selected_eval_tokens = eval_batches
        .iter()
        .take(eval_batch_limit)
        .map(|batch| batch.token_count)
        .sum::<usize>();
    let model = MiniMoeModel::<B, ConfiguredMiniMoeFfn<B>>::new(&manifest.surface, device)?;
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![BYTE_LEVEL_PAD_TOKEN]))
        .init(device);
    let mut optimizer = AdamConfig::new().init::<B, MiniMoeModel<B, ConfiguredMiniMoeFfn<B>>>();

    let baseline_process_memory = process_peak_memory_bytes();
    let mut peak_process_memory_bytes = baseline_process_memory;
    let total_start = Instant::now();

    let initial_eval_start = Instant::now();
    let initial_eval = evaluate_model(&model, &criterion, &eval_batches, eval_batch_limit)?;
    let initial_eval_wall_time_ms = initial_eval_start.elapsed().as_secs_f64() * 1000.0;
    peak_process_memory_bytes = peak_process_memory_bytes.max(process_peak_memory_bytes());

    let mut model = model;
    let mut seen_tokens = 0usize;
    let mut train_step_reports = Vec::with_capacity(manifest.train.steps);
    let train_start = Instant::now();
    for step in 0..manifest.train.steps {
        let batch = &train_batches[step % train_batches.len()];
        let loss = next_token_loss(&model, batch, &criterion)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(manifest.train.learning_rate, model, grads);
        seen_tokens += batch.token_count;
        train_step_reports.push(V2SmokeTrainStepReport {
            step: step + 1,
            learning_rate: manifest.train.learning_rate,
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

    let summary = summarize_mini_moe(&model, &eval_batches, eval_batch_limit)?;
    let total_wall_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    let artifact = MiniMoeRunArtifact {
        manifest: manifest.clone(),
        summary,
        train_metrics: MiniMoeTrainMetrics {
            initial_loss: initial_eval.mean_loss,
            final_loss: final_eval.mean_loss,
            load_balance_aux_loss: None,
        },
        eval_metrics: MiniMoeEvalMetrics {
            final_loss: final_eval.mean_loss,
            perplexity: final_eval.perplexity,
        },
        system_metrics: MiniMoeSystemMetrics {
            train_tokens_per_second: Some(tokens_per_second(seen_tokens, train_wall_time_ms)),
            eval_tokens_per_second: Some(tokens_per_second(
                selected_eval_tokens,
                initial_eval_wall_time_ms + final_eval_wall_time_ms,
            )),
            overall_tokens_per_second: Some(tokens_per_second(
                seen_tokens + (selected_eval_tokens * 2),
                total_wall_time_ms,
            )),
            process_memory_metric: Some(process_memory_metric_kind().as_str().to_string()),
            peak_process_memory_mb: Some(
                peak_process_memory_bytes.saturating_sub(baseline_process_memory) as f64
                    / (1024.0 * 1024.0),
            ),
        },
    };

    Ok(MiniMoeSmokeTrainArtifacts {
        corpus,
        initial_eval,
        final_eval,
        artifact,
        train_steps: train_step_reports,
    })
}

fn summarize_mini_moe<B>(
    model: &MiniMoeModel<B, ConfiguredMiniMoeFfn<B>>,
    eval_batches: &[TokenBatch<B>],
    eval_batch_limit: usize,
) -> Result<fractal_core::hybrid_attention::MiniMoeReportSummary, FractalError>
where
    B: AutodiffBackend,
{
    let mut aggregate = fractal_core::hybrid_attention::MiniMoeTraceBundle {
        layer_summaries: Vec::new(),
        dispatch_summaries: Vec::new(),
        controller_round_summaries: Vec::new(),
        sampled_token_traces: Vec::new(),
    };
    for batch in eval_batches.iter().take(eval_batch_limit) {
        let output = model.forward_with_trace(batch.input_ids.clone())?;
        aggregate.merge(output.trace_bundle);
    }
    Ok(aggregate.into_report_summary())
}

fn resolve_eval_batch_limit(eval: &MiniMoeEvalSpec, available_eval_batches: usize) -> usize {
    if eval.full_eval_pass {
        available_eval_batches
    } else {
        eval.eval_batches.min(available_eval_batches)
    }
}

fn tokens_per_second(tokens: usize, wall_time_ms: f64) -> f64 {
    if tokens == 0 || !wall_time_ms.is_finite() || wall_time_ms <= f64::EPSILON {
        0.0
    } else {
        tokens as f64 / (wall_time_ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    type TestBackend = fractal_core::CpuTrainBackend;

    #[test]
    fn mini_moe_smoke_config_defaults_validate() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fractal-eval-private should live under workspace root");
        let corpus_source = crate::default_v3a_fineweb_stage0_canary_corpus_source(repo_root)
            .expect("default canary corpus should exist in the workspace");
        let config = MiniMoeSmokeTrainConfig::new(
            corpus_source,
            repo_root.join("artifacts/tests/mini-moe-smoke-config"),
            MiniMoeSurfaceSpec::phase1_reference_default(),
        )
        .expect("config should build");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn mini_moe_smoke_train_runs_single_step() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("fractal-eval-private should live under workspace root");
        let corpus_source = crate::default_v3a_fineweb_stage0_canary_corpus_source(repo_root)
            .expect("default canary corpus should exist in the workspace");
        let mut config = MiniMoeSmokeTrainConfig::new(
            corpus_source,
            repo_root.join("artifacts/tests/mini-moe-smoke-train"),
            MiniMoeSurfaceSpec::phase1_reference_default(),
        )
        .expect("config should build");
        config.manifest.data.seq_len = 32;
        config.manifest.data.window_stride = 32;
        config.manifest.train.steps = 1;
        config.manifest.train.batch_size = 1;
        config.manifest.eval.eval_batches = 1;

        let device = Default::default();
        let report = run_mini_moe_smoke_train::<TestBackend>(config, &device)
            .expect("single-step mini-moe smoke train should succeed");
        assert!(report.report_path.is_file());
        assert_eq!(report.artifact.summary.routing.layer_count, 8);
    }
}
