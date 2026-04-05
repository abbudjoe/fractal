use std::path::{Path, PathBuf};

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor, TensorData},
};
use serde::Serialize;

use fractal_core::{
    error::FractalError, BaselineRescueAttentionBlock, GatheredRetrievalContext, LanguageModelHead,
    RescueAttentionBlock, RescueAttentionInput,
};

use crate::v2_training::{ensure_empty_output_dir, resolve_written_artifact, write_json_report};
use crate::{
    build_baseline_hybrid_rescue_model, run_hybrid_rescue_prevalidation_with_modes,
    BaselineHybridRescueModel, BaselineHybridRescueModelConfig, HybridRescueMetrics,
    HybridRescuePrevalidationReport, HybridRescueProbeMode, HybridRescueProbeSuite,
    V2CheckpointArtifacts, V2CheckpointKind,
};

pub const DEFAULT_HYBRID_RESCUE_FROZEN_STEPS: usize = 128;
pub const DEFAULT_HYBRID_RESCUE_FROZEN_EVAL_HOLDOUT_EVERY: usize = 2;
pub const DEFAULT_HYBRID_RESCUE_FROZEN_LEARNING_RATE: f64 = 5e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridRescueFrozenEvalModeSet {
    TrainingOnly,
    TrainingVsLocal,
    InitialFour,
}

impl HybridRescueFrozenEvalModeSet {
    pub fn modes(self, training_mode: HybridRescueProbeMode) -> Vec<HybridRescueProbeMode> {
        match self {
            Self::TrainingOnly => vec![training_mode],
            Self::TrainingVsLocal => vec![HybridRescueProbeMode::LocalOnly, training_mode],
            Self::InitialFour => HybridRescueProbeMode::INITIAL_FOUR.to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescueFrozenTrainConfig {
    pub output_dir: PathBuf,
    pub model: BaselineHybridRescueModelConfig,
    pub suites: Vec<HybridRescueProbeSuite>,
    pub training_mode: HybridRescueProbeMode,
    pub eval_mode_set: HybridRescueFrozenEvalModeSet,
    pub include_train_probe_reports: bool,
    pub train_steps: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
}

impl HybridRescueFrozenTrainConfig {
    pub fn new(output_dir: PathBuf, suites: Vec<HybridRescueProbeSuite>) -> Self {
        Self {
            output_dir,
            model: BaselineHybridRescueModelConfig::default(),
            suites,
            training_mode: HybridRescueProbeMode::OracleRemoteWithOracleExactTokenSubset,
            eval_mode_set: HybridRescueFrozenEvalModeSet::TrainingVsLocal,
            include_train_probe_reports: false,
            train_steps: DEFAULT_HYBRID_RESCUE_FROZEN_STEPS,
            eval_holdout_every: DEFAULT_HYBRID_RESCUE_FROZEN_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_HYBRID_RESCUE_FROZEN_LEARNING_RATE,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.model.validate()?;
        if self.suites.is_empty() {
            return Err(FractalError::InvalidConfig(
                "hybrid_rescue_frozen_train.suites must contain at least one suite".to_string(),
            ));
        }
        if self.train_steps == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_rescue_frozen_train.train_steps must be greater than zero".to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "hybrid_rescue_frozen_train.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "hybrid_rescue_frozen_train.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if self.training_mode == HybridRescueProbeMode::LocalOnly {
            return Err(FractalError::InvalidConfig(
                "hybrid_rescue_frozen_train.training_mode must include remote retrieval"
                    .to_string(),
            ));
        }
        for suite in &self.suites {
            suite.validate_for_model(
                self.model.backbone.vocab_size,
                self.model.backbone.leaf_size,
            )?;
        }
        let _ = split_hybrid_suites(&self.suites, self.eval_holdout_every)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HybridRescueFrozenSuiteSplit {
    pub kind: crate::HybridRescueSuiteKind,
    pub train_sample_names: Vec<String>,
    pub eval_sample_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HybridRescueFrozenSplitStats {
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub suites: Vec<HybridRescueFrozenSuiteSplit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HybridRescueFrozenEvalMetrics {
    pub sample_count: usize,
    pub accuracy: f32,
    pub mean_target_rank: f32,
    pub mean_target_logit: f32,
    pub mean_loss: f32,
    pub mean_local_attention_mass: f32,
    pub mean_remote_attention_mass: f32,
    pub evidence_span_recall_rate: f32,
    pub mean_evidence_token_recall: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HybridRescueFrozenTrainStepReport {
    pub step: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub seen_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescueFrozenTrainReport {
    pub config: HybridRescueFrozenTrainConfig,
    pub split: HybridRescueFrozenSplitStats,
    pub initial_train_probe: Option<HybridRescuePrevalidationReport>,
    pub initial_eval_probe: HybridRescuePrevalidationReport,
    pub initial_eval_metrics: HybridRescueFrozenEvalMetrics,
    pub final_train_probe: Option<HybridRescuePrevalidationReport>,
    pub final_eval_probe: HybridRescuePrevalidationReport,
    pub final_eval_metrics: HybridRescueFrozenEvalMetrics,
    pub best_eval_metrics: HybridRescueFrozenEvalMetrics,
    pub best_checkpoint_kind: V2CheckpointKind,
    pub train_steps: Vec<HybridRescueFrozenTrainStepReport>,
    pub checkpoint: V2CheckpointArtifacts,
}

#[derive(Debug)]
pub struct HybridRescueFrozenTrainResult<M> {
    pub model: M,
    pub report: HybridRescueFrozenTrainReport,
}

#[derive(Debug, Clone)]
struct FrozenHybridTrainingSample<B: Backend> {
    sample: crate::SyntheticProbeSample,
    input: RescueAttentionInput<B>,
}

#[derive(Debug, Clone)]
struct FrozenHybridSuiteSplit {
    train_suites: Vec<HybridRescueProbeSuite>,
    eval_suites: Vec<HybridRescueProbeSuite>,
    stats: HybridRescueFrozenSplitStats,
}

type BaselineFrozenHybridBackbone<B> = fractal_core::FractalV2Model<
    B,
    fractal_core::BaselineLocalTrunk<B>,
    fractal_core::BaselineLeafSummarizer<B>,
    fractal_core::BaselineTreeMergeCell<B>,
    fractal_core::BaselineFractalRouterHead<B>,
    fractal_core::BaselineExactLeafRead<B>,
    fractal_core::v2::BaselineReadFusion<B>,
>;

pub fn run_baseline_hybrid_rescue_frozen_train<B>(
    config: HybridRescueFrozenTrainConfig,
    device: &B::Device,
) -> Result<HybridRescueFrozenTrainResult<BaselineHybridRescueModel<B>>, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_baseline_hybrid_rescue_model::<B>(config.model, device)?;
    run_hybrid_rescue_frozen_train_with_model(model, config, device)
}

pub fn run_hybrid_rescue_frozen_train_with_model<B>(
    model: BaselineHybridRescueModel<B>,
    config: HybridRescueFrozenTrainConfig,
    device: &B::Device,
) -> Result<HybridRescueFrozenTrainResult<BaselineHybridRescueModel<B>>, FractalError>
where
    B: AutodiffBackend,
{
    config.validate()?;
    let split = split_hybrid_suites(&config.suites, config.eval_holdout_every)?;
    let eval_modes = config.eval_mode_set.modes(config.training_mode);
    let initial_train_probe = if config.include_train_probe_reports {
        Some(run_hybrid_rescue_prevalidation_with_modes(
            &model,
            &split.train_suites,
            &eval_modes,
            device,
        )?)
    } else {
        None
    };
    let initial_eval_probe = run_hybrid_rescue_prevalidation_with_modes(
        &model,
        &split.eval_suites,
        &eval_modes,
        device,
    )?;
    let initial_eval_metrics =
        hybrid_eval_metrics_for_mode(&initial_eval_probe, config.training_mode)?;
    let initial_model = model.clone();
    let backbone = model.backbone().clone();
    let output_head = model.output().clone();
    let initial_rescue_attention = model.rescue_attention().clone();
    let initial_optimizer = AdamConfig::new().init::<B, BaselineRescueAttentionBlock<B>>();
    let criterion = CrossEntropyLossConfig::new().init(device);
    let train_samples =
        prepare_frozen_training_samples(&model, &split.train_suites, config.training_mode, device)?;
    if train_samples.is_empty() {
        return Err(FractalError::InvalidConfig(
            "hybrid_rescue_frozen_train split must contain at least one training sample"
                .to_string(),
        ));
    }

    let mut optimizer = AdamConfig::new().init::<B, BaselineRescueAttentionBlock<B>>();
    let mut rescue_attention = initial_rescue_attention.clone();
    let mut train_steps = Vec::with_capacity(config.train_steps);

    for step in 0..config.train_steps {
        let sample = &train_samples[step % train_samples.len()];
        let loss =
            frozen_rescue_sample_loss(&rescue_attention, &output_head, sample, &criterion, device)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &rescue_attention);
        rescue_attention = optimizer.step(config.learning_rate, rescue_attention, grads);
        let seen_samples = step + 1;
        train_steps.push(HybridRescueFrozenTrainStepReport {
            step: seen_samples,
            learning_rate: config.learning_rate,
            train_loss,
            seen_samples,
        });
    }

    let final_model = build_model_from_parts(backbone.clone(), rescue_attention.clone())?;
    let final_train_probe = if config.include_train_probe_reports {
        Some(run_hybrid_rescue_prevalidation_with_modes(
            &final_model,
            &split.train_suites,
            &eval_modes,
            device,
        )?)
    } else {
        None
    };
    let final_eval_probe = run_hybrid_rescue_prevalidation_with_modes(
        &final_model,
        &split.eval_suites,
        &eval_modes,
        device,
    )?;
    let final_eval_metrics = hybrid_eval_metrics_for_mode(&final_eval_probe, config.training_mode)?;
    let (best_eval_metrics, best_checkpoint_kind, best_model, best_optimizer) =
        if final_eval_metrics.mean_loss <= initial_eval_metrics.mean_loss {
            (final_eval_metrics, V2CheckpointKind::FinalEval, None, None)
        } else {
            (
                initial_eval_metrics,
                V2CheckpointKind::InitialEval,
                Some(&initial_model),
                Some(&initial_optimizer),
            )
        };
    let checkpoint = persist_hybrid_rescue_checkpoint_artifacts(
        &final_model,
        best_model,
        &optimizer,
        best_optimizer,
        &config.output_dir,
    )?;
    let report = HybridRescueFrozenTrainReport {
        config,
        split: split.stats,
        initial_train_probe,
        initial_eval_probe,
        initial_eval_metrics,
        final_train_probe,
        final_eval_probe,
        final_eval_metrics,
        best_eval_metrics,
        best_checkpoint_kind,
        train_steps,
        checkpoint,
    };
    write_hybrid_rescue_train_report(&report)?;

    Ok(HybridRescueFrozenTrainResult {
        model: final_model,
        report,
    })
}

fn build_model_from_parts<B: Backend>(
    backbone: BaselineFrozenHybridBackbone<B>,
    rescue_attention: BaselineRescueAttentionBlock<B>,
) -> Result<BaselineHybridRescueModel<B>, FractalError> {
    fractal_core::FractalHybridRescuePrevalidationModel::new(backbone, rescue_attention)
}

fn prepare_frozen_training_samples<B: AutodiffBackend>(
    model: &BaselineHybridRescueModel<B>,
    suites: &[HybridRescueProbeSuite],
    training_mode: HybridRescueProbeMode,
    device: &B::Device,
) -> Result<Vec<FrozenHybridTrainingSample<B>>, FractalError> {
    let mut samples = Vec::new();
    for suite in suites {
        for sample in &suite.samples {
            let input_ids = Tensor::<B, 2, Int>::from_data(
                TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
                device,
            );
            let oracle_spans = training_mode
                .requires_oracle()
                .then(|| vec![Some(sample.evidence_span); sample.input_ids.len()]);
            let prepared = model.prepare_rescue_steps_with_mode_and_oracle_spans(
                input_ids,
                training_mode.runtime_mode(),
                oracle_spans.as_deref(),
            )?;
            let prepared_step = prepared.get(sample.target_position).ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "hybrid_rescue_frozen_train target position {} is out of bounds for sample '{}'",
                    sample.target_position, sample.name
                ))
            })?;
            samples.push(FrozenHybridTrainingSample {
                sample: sample.clone(),
                input: detach_rescue_input(prepared_step.input())?,
            });
        }
    }
    Ok(samples)
}

fn detach_rescue_input<B: AutodiffBackend>(
    input: RescueAttentionInput<B>,
) -> Result<RescueAttentionInput<B>, FractalError> {
    let gathered_remote = detach_gathered_context(input.gathered_remote())?;
    RescueAttentionInput::new(
        input.mode(),
        input.query_state().detach(),
        input.query_positions(),
        input.local_token_states().detach(),
        input.local_token_positions(),
        input.local_token_mask(),
        gathered_remote,
    )
}

fn detach_gathered_context<B: AutodiffBackend>(
    context: &GatheredRetrievalContext<B>,
) -> Result<GatheredRetrievalContext<B>, FractalError> {
    let shape = context.shape();
    GatheredRetrievalContext::from_tensors(
        shape.provenance,
        shape.layout,
        context.token_states().detach(),
        context.absolute_positions(),
        context.source_span_starts(),
        context.source_span_ends(),
        context.token_mask(),
    )
}

fn frozen_rescue_sample_loss<B: AutodiffBackend>(
    rescue_attention: &BaselineRescueAttentionBlock<B>,
    output_head: &LanguageModelHead<B>,
    sample: &FrozenHybridTrainingSample<B>,
    criterion: &CrossEntropyLoss<B>,
    device: &B::Device,
) -> Result<Tensor<B, 1>, FractalError> {
    let rescue_output = rescue_attention.attend(sample.input.clone())?;
    let logits = output_head.forward(rescue_output.updated_state());
    let [batch_size, vocab_size] = logits.dims();
    if batch_size != 1 {
        return Err(FractalError::Shape(format!(
            "hybrid_rescue_frozen_train expected batch size 1, got {batch_size}"
        )));
    }
    let target = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![sample.sample.target_token_id], [1]),
        device,
    );
    if vocab_size == 0 {
        return Err(FractalError::Shape(
            "hybrid_rescue_frozen_train logits vocab must be non-zero".to_string(),
        ));
    }
    Ok(criterion.forward(logits, target))
}

fn split_hybrid_suites(
    suites: &[HybridRescueProbeSuite],
    eval_holdout_every: usize,
) -> Result<FrozenHybridSuiteSplit, FractalError> {
    let mut train_suites = Vec::with_capacity(suites.len());
    let mut eval_suites = Vec::with_capacity(suites.len());
    let mut suite_stats = Vec::with_capacity(suites.len());
    let mut train_sample_count = 0usize;
    let mut eval_sample_count = 0usize;

    for suite in suites {
        let mut train_samples = Vec::new();
        let mut eval_samples = Vec::new();
        for (index, sample) in suite.samples.iter().cloned().enumerate() {
            if index % eval_holdout_every == 0 {
                eval_samples.push(sample);
            } else {
                train_samples.push(sample);
            }
        }
        if train_samples.is_empty() || eval_samples.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_rescue_frozen_train suite {:?} split produced train={} eval={} with eval_holdout_every={}",
                suite.kind,
                train_samples.len(),
                eval_samples.len(),
                eval_holdout_every
            )));
        }
        train_sample_count += train_samples.len();
        eval_sample_count += eval_samples.len();
        suite_stats.push(HybridRescueFrozenSuiteSplit {
            kind: suite.kind,
            train_sample_names: train_samples
                .iter()
                .map(|sample| sample.name.clone())
                .collect(),
            eval_sample_names: eval_samples
                .iter()
                .map(|sample| sample.name.clone())
                .collect(),
        });
        train_suites.push(HybridRescueProbeSuite {
            kind: suite.kind,
            leaf_size: suite.leaf_size,
            samples: train_samples,
        });
        eval_suites.push(HybridRescueProbeSuite {
            kind: suite.kind,
            leaf_size: suite.leaf_size,
            samples: eval_samples,
        });
    }

    Ok(FrozenHybridSuiteSplit {
        train_suites,
        eval_suites,
        stats: HybridRescueFrozenSplitStats {
            train_sample_count,
            eval_sample_count,
            suites: suite_stats,
        },
    })
}

pub fn hybrid_eval_metrics_for_mode(
    report: &HybridRescuePrevalidationReport,
    mode: HybridRescueProbeMode,
) -> Result<HybridRescueFrozenEvalMetrics, FractalError> {
    let mut sample_count = 0usize;
    let mut totals = HybridRescueMetrics {
        accuracy: 0.0,
        mean_target_rank: 0.0,
        mean_target_logit: 0.0,
        mean_loss: 0.0,
        mean_local_attention_mass: 0.0,
        mean_remote_attention_mass: 0.0,
        evidence_span_recall_rate: 0.0,
        mean_evidence_token_recall: 0.0,
    };

    for suite in &report.suites {
        let mode_report = suite
            .mode_reports
            .iter()
            .find(|candidate| candidate.mode == mode)
            .ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "hybrid_rescue_frozen_train report missing mode {:?} for suite {:?}",
                    mode, suite.kind
                ))
            })?;
        let suite_count = suite.sample_count as f32;
        sample_count += suite.sample_count;
        totals.accuracy += mode_report.metrics.accuracy * suite_count;
        totals.mean_target_rank += mode_report.metrics.mean_target_rank * suite_count;
        totals.mean_target_logit += mode_report.metrics.mean_target_logit * suite_count;
        totals.mean_loss += mode_report.metrics.mean_loss * suite_count;
        totals.mean_local_attention_mass +=
            mode_report.metrics.mean_local_attention_mass * suite_count;
        totals.mean_remote_attention_mass +=
            mode_report.metrics.mean_remote_attention_mass * suite_count;
        totals.evidence_span_recall_rate +=
            mode_report.metrics.evidence_span_recall_rate * suite_count;
        totals.mean_evidence_token_recall +=
            mode_report.metrics.mean_evidence_token_recall * suite_count;
    }

    if sample_count == 0 {
        return Err(FractalError::InvalidConfig(
            "hybrid_rescue_frozen_train eval report must contain at least one sample".to_string(),
        ));
    }
    let denom = sample_count as f32;
    Ok(HybridRescueFrozenEvalMetrics {
        sample_count,
        accuracy: totals.accuracy / denom,
        mean_target_rank: totals.mean_target_rank / denom,
        mean_target_logit: totals.mean_target_logit / denom,
        mean_loss: totals.mean_loss / denom,
        mean_local_attention_mass: totals.mean_local_attention_mass / denom,
        mean_remote_attention_mass: totals.mean_remote_attention_mass / denom,
        evidence_span_recall_rate: totals.evidence_span_recall_rate / denom,
        mean_evidence_token_recall: totals.mean_evidence_token_recall / denom,
    })
}

fn persist_hybrid_rescue_checkpoint_artifacts<B, M>(
    final_model: &M,
    best_model: Option<&M>,
    final_optimizer: &OptimizerAdaptor<Adam, BaselineRescueAttentionBlock<B>, B>,
    best_optimizer: Option<&OptimizerAdaptor<Adam, BaselineRescueAttentionBlock<B>, B>>,
    output_dir: &Path,
) -> Result<V2CheckpointArtifacts, FractalError>
where
    B: AutodiffBackend,
    M: Module<B> + Clone + AutodiffModule<B>,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    ensure_empty_output_dir(output_dir)?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let final_model_stem = output_dir.join("model");
    let best_model_stem = output_dir.join("best-model");
    let final_optimizer_stem = output_dir.join("optimizer");
    let best_optimizer_stem = output_dir.join("best-optimizer");
    final_model
        .clone()
        .save_file(final_model_stem.clone(), &recorder)
        .map_err(hybrid_recorder_error)?;
    let best_model_path = if let Some(best_model) = best_model {
        best_model
            .clone()
            .save_file(best_model_stem.clone(), &recorder)
            .map_err(hybrid_recorder_error)?;
        resolve_written_artifact(output_dir, "best-model")?
    } else {
        resolve_written_artifact(output_dir, "model")?
    };
    recorder
        .record(final_optimizer.to_record(), final_optimizer_stem.clone())
        .map(|_| ())
        .map_err(hybrid_recorder_error)?;
    let best_optimizer_path = if let Some(best_optimizer) = best_optimizer {
        recorder
            .record(best_optimizer.to_record(), best_optimizer_stem.clone())
            .map(|_| ())
            .map_err(hybrid_recorder_error)?;
        resolve_written_artifact(output_dir, "best-optimizer")?
    } else {
        resolve_written_artifact(output_dir, "optimizer")?
    };

    Ok(V2CheckpointArtifacts {
        directory: output_dir.to_path_buf(),
        final_model_path: resolve_written_artifact(output_dir, "model")?,
        best_model_path,
        final_optimizer_path: resolve_written_artifact(output_dir, "optimizer")?,
        best_optimizer_path,
        report_path: output_dir.join("report.json"),
    })
}

fn write_hybrid_rescue_train_report(
    report: &HybridRescueFrozenTrainReport,
) -> Result<(), FractalError> {
    write_json_report(
        &report.checkpoint.report_path,
        "hybrid rescue frozen training report",
        report,
    )
}

fn hybrid_recorder_error(error: impl core::fmt::Display) -> FractalError {
    FractalError::InvalidState(format!(
        "failed to persist hybrid rescue training artifact: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::{backend::Autodiff, backend::Candle};

    use super::*;
    use crate::default_hybrid_rescue_prevalidation_suites;

    type TestBackend = Autodiff<Candle<f32, i64>>;

    #[test]
    fn frozen_hybrid_rescue_train_persists_checkpoint_and_reports_probe_splits() {
        let root = unique_temp_dir("hybrid-rescue-frozen-train");
        let output_dir = root.join("artifacts");
        let suites = default_hybrid_rescue_prevalidation_suites()
            .unwrap()
            .into_iter()
            .filter(|suite| suite.kind == crate::HybridRescueSuiteKind::Copy)
            .collect::<Vec<_>>();
        let mut config = HybridRescueFrozenTrainConfig::new(output_dir, suites);
        config.train_steps = 1;
        config.eval_holdout_every = 2;

        let device = <TestBackend as Backend>::Device::default();
        let result =
            run_baseline_hybrid_rescue_frozen_train::<TestBackend>(config, &device).unwrap();

        assert_eq!(result.report.split.train_sample_count, 1);
        assert_eq!(result.report.split.eval_sample_count, 1);
        assert!(result.report.initial_eval_metrics.mean_loss.is_finite());
        assert!(result.report.final_eval_metrics.mean_loss.is_finite());
        assert!(result.report.checkpoint.final_model_path.exists());
        assert!(result.report.checkpoint.best_model_path.exists());
        assert!(result.report.checkpoint.report_path.exists());
        assert_eq!(result.report.initial_eval_probe.suites.len(), 1);
        assert_eq!(result.report.final_eval_probe.suites.len(), 1);
        assert!(result.report.initial_train_probe.is_none());
        assert!(result.report.final_train_probe.is_none());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn frozen_hybrid_rescue_train_rejects_local_only_training_mode() {
        let root = unique_temp_dir("hybrid-rescue-frozen-train-local-only");
        let suites = default_hybrid_rescue_prevalidation_suites().unwrap();
        let mut config = HybridRescueFrozenTrainConfig::new(root.join("artifacts"), suites);
        config.training_mode = HybridRescueProbeMode::LocalOnly;

        let error = config.validate().unwrap_err();
        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("remote retrieval")
        ));
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nonce}"))
    }
}
