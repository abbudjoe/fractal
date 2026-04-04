use std::{collections::BTreeMap, path::PathBuf};

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor, TensorData},
};
use serde::Serialize;

use fractal_core::{
    error::FractalError, ExactLeafRead, FractalRouterHead, FractalV2Model, LeafSummarizer,
    LocalTrunk, ReadFusion, TreeMergeCell,
};

use crate::v2_synthetic::{projection_breakdown_for_sample, SyntheticProbeModel};
use crate::v2_training::{
    persist_v2_checkpoint_artifacts, write_json_report, V2CheckpointArtifacts, V2CheckpointKind,
};
use crate::{
    build_baseline_v2_synthetic_model, run_v2_synthetic_probe_suites_with_modes,
    BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig, SyntheticProbeKind,
    SyntheticProbeMode, SyntheticProbeReport, SyntheticProbeSample, SyntheticProbeSuite,
};

pub const DEFAULT_V2_SUPERVISED_SYNTHETIC_STEPS: usize = 128;
pub const DEFAULT_V2_SUPERVISED_SYNTHETIC_EVAL_HOLDOUT_EVERY: usize = 2;
pub const DEFAULT_V2_SUPERVISED_SYNTHETIC_LEARNING_RATE: f64 = 5e-3;
pub const V2_SUPERVISED_SYNTHETIC_LEAF_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2SupervisedSyntheticTrainConfig {
    pub output_dir: PathBuf,
    pub model: BaselineV2SyntheticModelConfig,
    pub suites: Vec<SyntheticProbeSuite>,
    pub training_mode: SyntheticProbeMode,
    pub train_steps: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
}

impl V2SupervisedSyntheticTrainConfig {
    pub fn new(output_dir: PathBuf, suites: Vec<SyntheticProbeSuite>) -> Self {
        Self {
            output_dir,
            model: BaselineV2SyntheticModelConfig::default(),
            suites,
            training_mode: SyntheticProbeMode::TreePlusExactRead,
            train_steps: DEFAULT_V2_SUPERVISED_SYNTHETIC_STEPS,
            eval_holdout_every: DEFAULT_V2_SUPERVISED_SYNTHETIC_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V2_SUPERVISED_SYNTHETIC_LEARNING_RATE,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.model.validate()?;
        if self.suites.is_empty() {
            return Err(FractalError::InvalidConfig(
                "v2_supervised_synthetic.suites must contain at least one suite".to_string(),
            ));
        }
        if self.train_steps == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_supervised_synthetic.train_steps must be greater than zero".to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "v2_supervised_synthetic.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "v2_supervised_synthetic.learning_rate must be finite and greater than zero"
                    .to_string(),
            ));
        }
        if !self.training_mode.supports(SyntheticProbeMode::TreeOnly) {
            return Err(FractalError::InvalidConfig(format!(
                "v2_supervised_synthetic.training_mode must require tree memory, got {:?}",
                self.training_mode
            )));
        }
        for suite in &self.suites {
            suite.validate_for_model(self.model.vocab_size, self.model.leaf_size)?;
        }
        let _ = split_supervised_suites(&self.suites, self.eval_holdout_every)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct V2SupervisedSyntheticSuiteSplit {
    pub kind: SyntheticProbeKind,
    pub train_sample_names: Vec<String>,
    pub eval_sample_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct V2SupervisedSyntheticSplitStats {
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub suites: Vec<V2SupervisedSyntheticSuiteSplit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct V2SupervisedSyntheticEvalMetrics {
    pub sample_count: usize,
    pub accuracy: f32,
    pub mean_target_logit: f32,
    pub mean_loss: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct V2SupervisedSyntheticTrainStepReport {
    pub step: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub seen_samples: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2SupervisedSyntheticTrainReport {
    pub config: V2SupervisedSyntheticTrainConfig,
    pub split: V2SupervisedSyntheticSplitStats,
    pub initial_train_probe: SyntheticProbeReport,
    pub initial_eval_probe: SyntheticProbeReport,
    pub initial_eval_metrics: V2SupervisedSyntheticEvalMetrics,
    pub final_train_probe: SyntheticProbeReport,
    pub final_eval_probe: SyntheticProbeReport,
    pub final_eval_metrics: V2SupervisedSyntheticEvalMetrics,
    pub best_eval_metrics: V2SupervisedSyntheticEvalMetrics,
    pub best_checkpoint_kind: V2CheckpointKind,
    pub train_steps: Vec<V2SupervisedSyntheticTrainStepReport>,
    pub checkpoint: V2CheckpointArtifacts,
}

#[derive(Debug)]
pub struct V2SupervisedSyntheticTrainResult<M> {
    pub model: M,
    pub report: V2SupervisedSyntheticTrainReport,
}

pub trait V2SupervisedSyntheticTrainModel<B: Backend> {
    fn vocab_size(&self) -> usize;
    fn leaf_size(&self) -> usize;

    fn probe_logits_for_sample(
        &self,
        sample: &SyntheticProbeSample,
        mode: SyntheticProbeMode,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, FractalError>;
}

impl<B, LT, LS, TM, RH, ER, RF> V2SupervisedSyntheticTrainModel<B>
    for FractalV2Model<B, LT, LS, TM, RH, ER, RF>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    ER: ExactLeafRead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
{
    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }

    fn leaf_size(&self) -> usize {
        self.shape().local_trunk.leaf_size
    }

    fn probe_logits_for_sample(
        &self,
        sample: &SyntheticProbeSample,
        mode: SyntheticProbeMode,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, FractalError> {
        sample.validate_for_vocab(
            V2SupervisedSyntheticTrainModel::vocab_size(self),
            V2SupervisedSyntheticTrainModel::leaf_size(self),
        )?;
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
            device,
        );
        Ok(projection_breakdown_for_sample(self, sample, mode, input_ids)?.fused_logits())
    }
}

pub fn run_baseline_v2_supervised_synthetic_train<B>(
    config: V2SupervisedSyntheticTrainConfig,
    device: &B::Device,
) -> Result<V2SupervisedSyntheticTrainResult<BaselineV2SyntheticModel<B>>, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_baseline_v2_synthetic_model::<B>(config.model, device)?;
    run_v2_supervised_synthetic_train_with_model_and_modes(
        model,
        config,
        &SyntheticProbeMode::ALL_WITH_ORACLE,
        device,
    )
}

pub fn run_baseline_v2_supervised_synthetic_train_with_modes<B>(
    config: V2SupervisedSyntheticTrainConfig,
    report_modes: &[SyntheticProbeMode],
    device: &B::Device,
) -> Result<V2SupervisedSyntheticTrainResult<BaselineV2SyntheticModel<B>>, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_baseline_v2_synthetic_model::<B>(config.model, device)?;
    run_v2_supervised_synthetic_train_with_model_and_modes(model, config, report_modes, device)
}

pub fn run_v2_supervised_synthetic_train_with_model<B, M>(
    model: M,
    config: V2SupervisedSyntheticTrainConfig,
    device: &B::Device,
) -> Result<V2SupervisedSyntheticTrainResult<M>, FractalError>
where
    B: AutodiffBackend,
    M: V2SupervisedSyntheticTrainModel<B>
        + SyntheticProbeModel<Device = B::Device>
        + AutodiffModule<B>
        + Module<B>
        + Clone,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    run_v2_supervised_synthetic_train_with_model_and_modes(
        model,
        config,
        &SyntheticProbeMode::ALL_WITH_ORACLE,
        device,
    )
}

pub fn run_v2_supervised_synthetic_train_with_model_and_modes<B, M>(
    model: M,
    config: V2SupervisedSyntheticTrainConfig,
    report_modes: &[SyntheticProbeMode],
    device: &B::Device,
) -> Result<V2SupervisedSyntheticTrainResult<M>, FractalError>
where
    B: AutodiffBackend,
    M: V2SupervisedSyntheticTrainModel<B>
        + SyntheticProbeModel<Device = B::Device>
        + AutodiffModule<B>
        + Module<B>
        + Clone,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    config.validate()?;
    if report_modes.is_empty() {
        return Err(FractalError::InvalidConfig(
            "v2_supervised_synthetic.report_modes must not be empty".to_string(),
        ));
    }
    if !report_modes.contains(&config.training_mode) {
        return Err(FractalError::InvalidConfig(format!(
            "v2_supervised_synthetic.report_modes must include training_mode {:?}",
            config.training_mode
        )));
    }
    let split = split_supervised_suites(&config.suites, config.eval_holdout_every)?;
    let initial_train_probe = run_v2_synthetic_probe_suites_with_modes(
        &model,
        &split.train_suites,
        report_modes,
        device,
    )?;
    let initial_eval_probe =
        run_v2_synthetic_probe_suites_with_modes(&model, &split.eval_suites, report_modes, device)?;
    let initial_eval_metrics = eval_metrics_for_mode(&initial_eval_probe, config.training_mode)?;
    let initial_model = model.clone();
    let initial_optimizer = AdamConfig::new().init::<B, M>();
    let criterion = CrossEntropyLossConfig::new().init(device);
    let mut optimizer = AdamConfig::new().init::<B, M>();
    let mut model = model;
    let train_samples = flatten_samples(&split.train_suites);
    let mut train_steps = Vec::with_capacity(config.train_steps);

    for step in 0..config.train_steps {
        let sample = &train_samples[step % train_samples.len()];
        let loss = supervised_probe_loss(&model, sample, config.training_mode, &criterion, device)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        let seen_samples = step + 1;
        train_steps.push(V2SupervisedSyntheticTrainStepReport {
            step: seen_samples,
            learning_rate: config.learning_rate,
            train_loss,
            seen_samples,
        });
    }

    let final_train_probe = run_v2_synthetic_probe_suites_with_modes(
        &model,
        &split.train_suites,
        report_modes,
        device,
    )?;
    let final_eval_probe =
        run_v2_synthetic_probe_suites_with_modes(&model, &split.eval_suites, report_modes, device)?;
    let final_eval_metrics = eval_metrics_for_mode(&final_eval_probe, config.training_mode)?;
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
    let checkpoint = persist_v2_checkpoint_artifacts(
        &model,
        best_model,
        &optimizer,
        best_optimizer,
        &config.output_dir,
    )?;
    let report = V2SupervisedSyntheticTrainReport {
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
    write_supervised_report(&report)?;

    Ok(V2SupervisedSyntheticTrainResult { model, report })
}

#[derive(Debug, Clone)]
struct SupervisedSyntheticSuiteSplit {
    train_suites: Vec<SyntheticProbeSuite>,
    eval_suites: Vec<SyntheticProbeSuite>,
    stats: V2SupervisedSyntheticSplitStats,
}

fn split_supervised_suites(
    suites: &[SyntheticProbeSuite],
    eval_holdout_every: usize,
) -> Result<SupervisedSyntheticSuiteSplit, FractalError> {
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
                "v2_supervised_synthetic suite {:?} split produced train={} eval={} with eval_holdout_every={}",
                suite.kind,
                train_samples.len(),
                eval_samples.len(),
                eval_holdout_every
            )));
        }

        train_sample_count += train_samples.len();
        eval_sample_count += eval_samples.len();
        suite_stats.push(V2SupervisedSyntheticSuiteSplit {
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
        train_suites.push(SyntheticProbeSuite {
            kind: suite.kind,
            leaf_size: suite.leaf_size,
            samples: train_samples,
        });
        eval_suites.push(SyntheticProbeSuite {
            kind: suite.kind,
            leaf_size: suite.leaf_size,
            samples: eval_samples,
        });
    }

    Ok(SupervisedSyntheticSuiteSplit {
        train_suites,
        eval_suites,
        stats: V2SupervisedSyntheticSplitStats {
            train_sample_count,
            eval_sample_count,
            suites: suite_stats,
        },
    })
}

fn flatten_samples(suites: &[SyntheticProbeSuite]) -> Vec<SyntheticProbeSample> {
    suites
        .iter()
        .flat_map(|suite| suite.samples.iter().cloned())
        .collect()
}

fn supervised_probe_loss<B, M>(
    model: &M,
    sample: &SyntheticProbeSample,
    mode: SyntheticProbeMode,
    criterion: &CrossEntropyLoss<B>,
    device: &B::Device,
) -> Result<Tensor<B, 1>, FractalError>
where
    B: AutodiffBackend,
    M: V2SupervisedSyntheticTrainModel<B>,
{
    let logits = model.probe_logits_for_sample(sample, mode, device)?;
    let [batch_size, vocab_size] = logits.dims();
    if batch_size != 1 {
        return Err(FractalError::Shape(format!(
            "v2 supervised synthetic expected batch size 1 per sample, got {batch_size}"
        )));
    }
    if vocab_size != model.vocab_size() {
        return Err(FractalError::Shape(format!(
            "v2 supervised synthetic vocab mismatch: logits vocab {} vs model vocab {}",
            vocab_size,
            model.vocab_size()
        )));
    }
    let target =
        Tensor::<B, 1, Int>::from_data(TensorData::new(vec![sample.target_token_id], [1]), device);
    Ok(criterion.forward(logits, target))
}

fn eval_metrics_for_mode(
    report: &SyntheticProbeReport,
    mode: SyntheticProbeMode,
) -> Result<V2SupervisedSyntheticEvalMetrics, FractalError> {
    let mut sample_count = 0usize;
    let mut correct_count = 0usize;
    let mut total_target_logit = 0.0f32;
    let mut total_loss = 0.0f32;
    for suite in &report.suites {
        let mode_report = suite.mode_report(mode).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "v2 supervised synthetic report missing mode {:?} for suite {:?}",
                mode, suite.kind
            ))
        })?;
        sample_count += suite.sample_count;
        correct_count += mode_report
            .sample_results
            .iter()
            .filter(|sample| sample.correct)
            .count();
        total_target_logit += mode_report.metrics.mean_target_logit * suite.sample_count as f32;
        total_loss += mode_report.metrics.mean_loss * suite.sample_count as f32;
    }
    if sample_count == 0 {
        return Err(FractalError::InvalidConfig(
            "v2 supervised synthetic eval report must contain at least one sample".to_string(),
        ));
    }
    Ok(V2SupervisedSyntheticEvalMetrics {
        sample_count,
        accuracy: correct_count as f32 / sample_count as f32,
        mean_target_logit: total_target_logit / sample_count as f32,
        mean_loss: total_loss / sample_count as f32,
    })
}

fn write_supervised_report(report: &V2SupervisedSyntheticTrainReport) -> Result<(), FractalError> {
    write_json_report(
        &report.checkpoint.report_path,
        "v2 supervised synthetic training report",
        report,
    )
}

pub fn filter_synthetic_probe_suites(
    suites: Vec<SyntheticProbeSuite>,
    kinds: &[SyntheticProbeKind],
) -> Vec<SyntheticProbeSuite> {
    if kinds.is_empty() {
        return suites;
    }
    let selected = kinds
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    suites
        .into_iter()
        .filter(|suite| selected.contains(&suite.kind))
        .collect()
}

pub fn mode_eval_summary_by_kind(
    report: &SyntheticProbeReport,
    mode: SyntheticProbeMode,
) -> Result<BTreeMap<SyntheticProbeKind, V2SupervisedSyntheticEvalMetrics>, FractalError> {
    let mut by_kind = BTreeMap::new();
    for suite in &report.suites {
        let mode_report = suite.mode_report(mode).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "v2 supervised synthetic report missing mode {:?} for suite {:?}",
                mode, suite.kind
            ))
        })?;
        by_kind.insert(
            suite.kind,
            V2SupervisedSyntheticEvalMetrics {
                sample_count: suite.sample_count,
                accuracy: mode_report.metrics.accuracy,
                mean_target_logit: mode_report.metrics.mean_target_logit,
                mean_loss: mode_report.metrics.mean_loss,
            },
        );
    }
    Ok(by_kind)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::{backend::Autodiff, backend::Candle};

    use super::*;
    use crate::default_v2_synthetic_probe_suites;

    type TestBackend = Autodiff<Candle<f32, i64>>;

    #[test]
    fn supervised_train_persists_checkpoint_and_reports_probe_splits() {
        let root = unique_temp_dir("v2-supervised-synthetic");
        let output_dir = root.join("artifacts");
        let suites = filter_synthetic_probe_suites(
            default_v2_synthetic_probe_suites(),
            &[SyntheticProbeKind::Copy],
        );
        let mut config = V2SupervisedSyntheticTrainConfig::new(output_dir, suites);
        config.train_steps = 1;
        config.eval_holdout_every = 2;

        let device = <TestBackend as Backend>::Device::default();
        let result = run_baseline_v2_supervised_synthetic_train_with_modes::<TestBackend>(
            config,
            &[SyntheticProbeMode::TreePlusExactRead],
            &device,
        )
        .unwrap();

        assert_eq!(result.report.split.train_sample_count, 1);
        assert_eq!(result.report.split.eval_sample_count, 1);
        assert!(result.report.initial_eval_metrics.mean_loss.is_finite());
        assert!(result.report.final_eval_metrics.mean_loss.is_finite());
        assert!(result.report.checkpoint.final_model_path.exists());
        assert!(result.report.checkpoint.best_model_path.exists());
        assert!(result.report.checkpoint.report_path.exists());
        assert_eq!(result.report.initial_eval_probe.suites.len(), 1);
        assert_eq!(result.report.final_eval_probe.suites.len(), 1);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn supervised_train_rejects_suite_split_without_train_and_eval_samples() {
        let root = unique_temp_dir("v2-supervised-synthetic-invalid-split");
        let suite = SyntheticProbeSuite {
            kind: SyntheticProbeKind::Copy,
            leaf_size: 16,
            samples: vec![default_v2_synthetic_probe_suites()[0].samples[0].clone()],
        };
        let mut config = V2SupervisedSyntheticTrainConfig::new(root.join("artifacts"), vec![suite]);
        config.eval_holdout_every = 2;

        let error = config.validate().unwrap_err();
        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("split produced")
        ));
    }

    #[test]
    fn supervised_train_rejects_report_modes_that_omit_training_mode() {
        let root = unique_temp_dir("v2-supervised-synthetic-missing-training-mode");
        let output_dir = root.join("artifacts");
        let suites = filter_synthetic_probe_suites(
            default_v2_synthetic_probe_suites(),
            &[SyntheticProbeKind::Copy],
        );
        let mut config = V2SupervisedSyntheticTrainConfig::new(output_dir, suites);
        config.train_steps = 1;
        config.eval_holdout_every = 2;

        let device = <TestBackend as Backend>::Device::default();
        let error = run_baseline_v2_supervised_synthetic_train_with_modes::<TestBackend>(
            config,
            &[SyntheticProbeMode::TreeOnly],
            &device,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message)
                if message.contains("report_modes must include training_mode")
        ));
    }

    #[test]
    fn filter_synthetic_probe_suites_keeps_requested_kinds() {
        let filtered = filter_synthetic_probe_suites(
            default_v2_synthetic_probe_suites(),
            &[SyntheticProbeKind::Copy, SyntheticProbeKind::NoisyRetrieval],
        );

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].kind, SyntheticProbeKind::Copy);
        assert_eq!(filtered[1].kind, SyntheticProbeKind::NoisyRetrieval);
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }
}
