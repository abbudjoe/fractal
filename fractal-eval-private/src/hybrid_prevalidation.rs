use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    nn::Initializer,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use fractal_core::{
    error::FractalError, BaselineRescueAttentionBlock, BaselineRescueAttentionConfig,
    FractalHybridRescuePrevalidationModel, HybridRescuePrevalidationMode, TokenSpan,
    PHASE1_LEAF_SIZE, PHASE1_LOCAL_WINDOW_SIZE, PHASE1_ROUTED_SPAN_COUNT,
    PHASE1_TOTAL_TOKEN_BUDGET,
};

use crate::{
    build_baseline_v2_synthetic_model, v2_synthetic::score_probe_sample,
    v2_synthetic_probe_suites_for_leaf_size, SyntheticProbeKind, SyntheticProbeSample,
};

const DEFAULT_HYBRID_RESULTS_LEDGER_PATH: &str = "docs/v3-results-ledger.jsonl";
const MQAR_SENTINEL: i64 = 7;
const MQAR_QUERY_SENTINEL: i64 = 8;

pub type BaselineHybridRescueModel<B> = FractalHybridRescuePrevalidationModel<
    B,
    fractal_core::BaselineLocalTrunk<B>,
    fractal_core::BaselineLeafSummarizer<B>,
    fractal_core::BaselineTreeMergeCell<B>,
    fractal_core::BaselineFractalRouterHead<B>,
    fractal_core::BaselineExactLeafRead<B>,
    fractal_core::v2::BaselineReadFusion<B>,
    BaselineRescueAttentionBlock<B>,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineHybridRescueModelConfig {
    pub backbone: crate::BaselineV2SyntheticModelConfig,
    pub attention_dim: usize,
}

impl Default for BaselineHybridRescueModelConfig {
    fn default() -> Self {
        Self {
            backbone: crate::BaselineV2SyntheticModelConfig {
                root_count: 1,
                total_root_state_dim: 12,
                total_root_readout_dim: 8,
                leaf_size: PHASE1_LEAF_SIZE,
                routing_head_count: 1,
                beam_width: PHASE1_ROUTED_SPAN_COUNT,
                top_leaf_reads: PHASE1_ROUTED_SPAN_COUNT,
                exact_read_head_count: 1,
                ..crate::BaselineV2SyntheticModelConfig::default()
            },
            attention_dim: 4,
        }
    }
}

impl BaselineHybridRescueModelConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.backbone.validate()?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.root_count",
            self.backbone.root_count,
            1,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.leaf_size",
            self.backbone.leaf_size,
            PHASE1_LEAF_SIZE,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.top_leaf_reads",
            self.backbone.top_leaf_reads,
            PHASE1_ROUTED_SPAN_COUNT,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.beam_width",
            self.backbone.beam_width,
            PHASE1_ROUTED_SPAN_COUNT,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.routing_head_count",
            self.backbone.routing_head_count,
            1,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.backbone.token_dim",
            self.backbone.token_dim,
            self.backbone.root_readout_dim(),
        )?;
        ensure_nonzero(
            "hybrid_rescue_prevalidation.attention_dim",
            self.attention_dim,
        )?;
        Ok(())
    }
}

pub fn build_baseline_hybrid_rescue_model<B: Backend>(
    config: BaselineHybridRescueModelConfig,
    device: &B::Device,
) -> Result<BaselineHybridRescueModel<B>, FractalError> {
    config.validate()?;
    let backbone = build_baseline_v2_synthetic_model::<B>(config.backbone, device)?;
    let rescue_attention = BaselineRescueAttentionConfig {
        token_state_dim: config.backbone.root_readout_dim(),
        attention_dim: config.attention_dim,
        local_window_size: PHASE1_LOCAL_WINDOW_SIZE,
        routed_span_count: PHASE1_ROUTED_SPAN_COUNT,
        leaf_size: config.backbone.leaf_size,
        sink_token_count: 0,
        total_token_budget: PHASE1_TOTAL_TOKEN_BUDGET,
        initializer: Initializer::Uniform {
            min: -0.08,
            max: 0.08,
        },
    }
    .try_init(device)?;

    FractalHybridRescuePrevalidationModel::new(backbone, rescue_attention)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridRescueProbeMode {
    LocalOnly,
    RoutedRemote,
    OracleRemote,
    OracleRemoteWithOracleExactTokenSubset,
}

impl HybridRescueProbeMode {
    pub const INITIAL_FOUR: [Self; 4] = [
        Self::LocalOnly,
        Self::RoutedRemote,
        Self::OracleRemote,
        Self::OracleRemoteWithOracleExactTokenSubset,
    ];

    pub fn runtime_mode(self) -> HybridRescuePrevalidationMode {
        match self {
            Self::LocalOnly => HybridRescuePrevalidationMode::LocalOnly,
            Self::RoutedRemote => HybridRescuePrevalidationMode::RoutedRemote,
            Self::OracleRemote => HybridRescuePrevalidationMode::OracleRemote,
            Self::OracleRemoteWithOracleExactTokenSubset => {
                HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset
            }
        }
    }

    pub fn requires_oracle(self) -> bool {
        matches!(
            self,
            Self::OracleRemote | Self::OracleRemoteWithOracleExactTokenSubset
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridRescueSuiteKind {
    Mqar,
    Copy,
    Induction,
    RetrievalHeavy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HybridRescueProbeSuite {
    pub kind: HybridRescueSuiteKind,
    pub leaf_size: usize,
    pub samples: Vec<SyntheticProbeSample>,
}

impl HybridRescueProbeSuite {
    pub fn validate_for_model(
        &self,
        vocab_size: usize,
        leaf_size: usize,
    ) -> Result<(), FractalError> {
        ensure_nonzero(
            "hybrid_rescue_prevalidation.suite.leaf_size",
            self.leaf_size,
        )?;
        ensure_match(
            "hybrid_rescue_prevalidation.suite.model_leaf_size",
            self.leaf_size,
            leaf_size,
        )?;
        if self.samples.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_rescue_prevalidation suite '{:?}' must contain at least one sample",
                self.kind
            )));
        }
        for sample in &self.samples {
            sample.validate_for_vocab(vocab_size, self.leaf_size)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescueSampleResult {
    pub sample_name: String,
    pub predicted_token_id: i64,
    pub target_token_id: i64,
    pub target_rank: usize,
    pub correct: bool,
    pub target_logit: f32,
    pub loss: f32,
    pub mean_local_attention_mass: f32,
    pub mean_remote_attention_mass: f32,
    pub evidence_span_recalled: bool,
    pub evidence_token_recall: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HybridRescueMetrics {
    pub accuracy: f32,
    pub mean_target_rank: f32,
    pub mean_target_logit: f32,
    pub mean_loss: f32,
    pub mean_local_attention_mass: f32,
    pub mean_remote_attention_mass: f32,
    pub evidence_span_recall_rate: f32,
    pub mean_evidence_token_recall: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescueModeReport {
    pub mode: HybridRescueProbeMode,
    pub metrics: HybridRescueMetrics,
    pub sample_results: Vec<HybridRescueSampleResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescueSuiteReport {
    pub kind: HybridRescueSuiteKind,
    pub sample_count: usize,
    pub mode_reports: Vec<HybridRescueModeReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HybridRescuePrevalidationReport {
    pub suites: Vec<HybridRescueSuiteReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HybridResultsLedgerKind {
    RescuePrevalidationProbe,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridResultsLedgerEntry {
    pub schema_version: u32,
    pub recorded_at_unix_seconds: u64,
    pub kind: HybridResultsLedgerKind,
    pub model: String,
    pub note: String,
    pub run_label: Option<String>,
    pub payload: Value,
}

impl HybridResultsLedgerEntry {
    pub fn rescue_prevalidation_probe(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &HybridRescuePrevalidationReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Ok(Self {
            schema_version: 1,
            recorded_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            kind: HybridResultsLedgerKind::RescuePrevalidationProbe,
            model: model.into(),
            note: note.into(),
            run_label,
            payload: serde_json::to_value(report).map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to serialize hybrid results ledger payload: {error}"
                ))
            })?,
        })
    }
}

pub fn default_hybrid_results_ledger_path(repo_root: impl AsRef<Path>) -> PathBuf {
    repo_root
        .as_ref()
        .join(DEFAULT_HYBRID_RESULTS_LEDGER_PATH)
}

pub fn resolve_requested_hybrid_results_ledger_path(
    repo_root: impl AsRef<Path>,
    request: Option<&str>,
) -> Result<Option<PathBuf>, FractalError> {
    let Some(request) = request else {
        return Ok(None);
    };
    if request.trim().is_empty() {
        return Err(FractalError::InvalidConfig(
            "hybrid_results_ledger.path must not be empty".to_string(),
        ));
    }
    if request == "default" {
        return Ok(Some(default_hybrid_results_ledger_path(repo_root)));
    }
    Ok(Some(PathBuf::from(request)))
}

pub fn append_hybrid_results_ledger_entry(
    path: impl AsRef<Path>,
    entry: &HybridResultsLedgerEntry,
) -> Result<(), FractalError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to create hybrid results ledger directory {}: {error}",
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
                "failed to open hybrid results ledger {} for append: {error}",
                path.display()
            ))
        })?;
    serde_json::to_writer(&mut file, entry).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize hybrid results ledger entry for {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to terminate hybrid results ledger entry for {}: {error}",
            path.display()
        ))
    })
}

pub fn default_hybrid_rescue_prevalidation_suites(
) -> Result<Vec<HybridRescueProbeSuite>, FractalError> {
    hybrid_rescue_prevalidation_suites_for_leaf_size(PHASE1_LEAF_SIZE)
}

pub fn hybrid_rescue_prevalidation_suites_for_leaf_size(
    leaf_size: usize,
) -> Result<Vec<HybridRescueProbeSuite>, FractalError> {
    let suites = v2_synthetic_probe_suites_for_leaf_size(leaf_size)?;
    let copy = take_single_suite(&suites, SyntheticProbeKind::Copy)?;
    let induction = take_single_suite(&suites, SyntheticProbeKind::Induction)?;
    let associative = take_single_suite(&suites, SyntheticProbeKind::AssociativeRecall)?;
    let noisy = take_single_suite(&suites, SyntheticProbeKind::NoisyRetrieval)?;
    let far = take_single_suite(&suites, SyntheticProbeKind::FarTokenComparison)?;

    Ok(vec![
        mqar_probe_suite_for_leaf_size(leaf_size)?,
        HybridRescueProbeSuite {
            kind: HybridRescueSuiteKind::Copy,
            leaf_size,
            samples: copy.samples.clone(),
        },
        HybridRescueProbeSuite {
            kind: HybridRescueSuiteKind::Induction,
            leaf_size,
            samples: induction.samples.clone(),
        },
        HybridRescueProbeSuite {
            kind: HybridRescueSuiteKind::RetrievalHeavy,
            leaf_size,
            samples: associative
                .samples
                .iter()
                .chain(noisy.samples.iter())
                .chain(far.samples.iter())
                .cloned()
                .collect(),
        },
    ])
}

pub fn run_hybrid_rescue_prevalidation_with_modes<B: Backend>(
    model: &BaselineHybridRescueModel<B>,
    suites: &[HybridRescueProbeSuite],
    modes: &[HybridRescueProbeMode],
    device: &B::Device,
) -> Result<HybridRescuePrevalidationReport, FractalError> {
    if modes.is_empty() {
        return Err(FractalError::InvalidConfig(
            "hybrid_rescue_prevalidation.modes must not be empty".to_string(),
        ));
    }
    let mut reports = Vec::with_capacity(suites.len());
    for suite in suites {
        reports.push(run_hybrid_rescue_prevalidation_suite_with_modes(
            model, suite, modes, device,
        )?);
    }
    Ok(HybridRescuePrevalidationReport { suites: reports })
}

pub fn run_hybrid_rescue_prevalidation_suite_with_modes<B: Backend>(
    model: &BaselineHybridRescueModel<B>,
    suite: &HybridRescueProbeSuite,
    modes: &[HybridRescueProbeMode],
    device: &B::Device,
) -> Result<HybridRescueSuiteReport, FractalError> {
    suite.validate_for_model(model.backbone().shape().vocab_size, model.shape().rescue_attention.leaf_size)?;
    let mut mode_reports = Vec::with_capacity(modes.len());
    for &mode in modes {
        let mut sample_results = Vec::with_capacity(suite.samples.len());
        for sample in &suite.samples {
            sample_results.push(run_hybrid_rescue_sample(model, sample, mode, device)?);
        }
        mode_reports.push(HybridRescueModeReport {
            mode,
            metrics: aggregate_hybrid_metrics(&sample_results),
            sample_results,
        });
    }
    Ok(HybridRescueSuiteReport {
        kind: suite.kind,
        sample_count: suite.samples.len(),
        mode_reports,
    })
}

pub fn run_baseline_hybrid_rescue_prevalidation<B: Backend>(
    config: BaselineHybridRescueModelConfig,
    suites: &[HybridRescueProbeSuite],
    modes: &[HybridRescueProbeMode],
    device: &B::Device,
) -> Result<HybridRescuePrevalidationReport, FractalError> {
    let model = build_baseline_hybrid_rescue_model::<B>(config, device)?;
    run_hybrid_rescue_prevalidation_with_modes(&model, suites, modes, device)
}

fn run_hybrid_rescue_sample<B: Backend>(
    model: &BaselineHybridRescueModel<B>,
    sample: &SyntheticProbeSample,
    mode: HybridRescueProbeMode,
    device: &B::Device,
) -> Result<HybridRescueSampleResult, FractalError> {
    sample.validate_for_vocab(model.backbone().shape().vocab_size, model.shape().rescue_attention.leaf_size)?;
    let input_ids = Tensor::<B, 2, Int>::from_data(
        TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
        device,
    );
    let oracle_spans = mode
        .requires_oracle()
        .then(|| vec![Some(sample.evidence_span); sample.input_ids.len()]);
    let output = model.forward_with_mode_and_oracle_spans(
        input_ids,
        mode.runtime_mode(),
        oracle_spans.as_deref(),
    )?;
    let step = output.steps().get(sample.target_position).ok_or_else(|| {
        FractalError::InvalidState(format!(
            "hybrid_rescue_prevalidation target position {} is out of bounds for {} steps",
            sample.target_position,
            output.steps().len()
        ))
    })?;
    let logits = output
        .logits()
        .slice([
            0..1,
            sample.target_position..sample.target_position + 1,
            0..model.backbone().shape().vocab_size,
        ])
        .reshape([model.backbone().shape().vocab_size])
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read hybrid rescue logits for sample '{}': {error}",
                sample.name
            ))
        })?;
    let base = score_probe_sample(sample, &logits, model.backbone().shape().vocab_size)?;
    let target_rank = target_rank(&logits, sample.target_token_id)?;
    let candidate_recall = step
        .candidate_recall()
        .and_then(|recalls| recalls.first())
        .cloned();

    Ok(HybridRescueSampleResult {
        sample_name: base.sample_name,
        predicted_token_id: base.predicted_token_id,
        target_token_id: base.target_token_id,
        target_rank,
        correct: base.correct,
        target_logit: base.target_logit,
        loss: base.loss,
        mean_local_attention_mass: step.attention_diagnostics().mean_local_attention_mass,
        mean_remote_attention_mass: step.attention_diagnostics().mean_remote_attention_mass,
        evidence_span_recalled: candidate_recall
            .as_ref()
            .map(|recall| recall.evidence_span_recalled)
            .unwrap_or(false),
        evidence_token_recall: candidate_recall
            .as_ref()
            .map(GatheredRecallLike::evidence_token_recall)
            .unwrap_or(0.0),
    })
}

fn aggregate_hybrid_metrics(sample_results: &[HybridRescueSampleResult]) -> HybridRescueMetrics {
    let sample_count = sample_results.len().max(1) as f32;
    let (
        correct_count,
        target_rank_sum,
        target_logit_sum,
        loss_sum,
        local_attention_sum,
        remote_attention_sum,
        evidence_span_recall_count,
        evidence_token_recall_sum,
    ) = sample_results.iter().fold(
        (0usize, 0usize, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize, 0.0f32),
        |acc, result| {
            (
                acc.0 + usize::from(result.correct),
                acc.1 + result.target_rank,
                acc.2 + result.target_logit,
                acc.3 + result.loss,
                acc.4 + result.mean_local_attention_mass,
                acc.5 + result.mean_remote_attention_mass,
                acc.6 + usize::from(result.evidence_span_recalled),
                acc.7 + result.evidence_token_recall,
            )
        },
    );

    HybridRescueMetrics {
        accuracy: correct_count as f32 / sample_count,
        mean_target_rank: target_rank_sum as f32 / sample_count,
        mean_target_logit: target_logit_sum / sample_count,
        mean_loss: loss_sum / sample_count,
        mean_local_attention_mass: local_attention_sum / sample_count,
        mean_remote_attention_mass: remote_attention_sum / sample_count,
        evidence_span_recall_rate: evidence_span_recall_count as f32 / sample_count,
        mean_evidence_token_recall: evidence_token_recall_sum / sample_count,
    }
}

fn take_single_suite(
    suites: &[crate::SyntheticProbeSuite],
    kind: SyntheticProbeKind,
) -> Result<&crate::SyntheticProbeSuite, FractalError> {
    suites.iter().find(|suite| suite.kind == kind).ok_or_else(|| {
        FractalError::InvalidState(format!(
            "failed to locate synthetic probe suite for {:?}",
            kind
        ))
    })
}

fn mqar_probe_suite_for_leaf_size(leaf_size: usize) -> Result<HybridRescueProbeSuite, FractalError> {
    ensure_match("hybrid_rescue_prevalidation.mqar.leaf_size", leaf_size, PHASE1_LEAF_SIZE)?;
    Ok(HybridRescueProbeSuite {
        kind: HybridRescueSuiteKind::Mqar,
        leaf_size,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::AssociativeRecall,
                "mqar-alpha",
                vec![
                    MQAR_SENTINEL, 11, 31, 12, 32, 13, 33, 14, 34, 21, 22, 23, 24, 25, 26, 27,
                    MQAR_QUERY_SENTINEL, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                    MQAR_QUERY_SENTINEL, 12,
                ],
                32,
                TokenSpan::new(4, 5)?,
                4,
                crate::SyntheticProbeMode::TreeOnly,
            )?,
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::AssociativeRecall,
                "mqar-beta",
                vec![
                    MQAR_SENTINEL, 15, 35, 16, 36, 17, 37, 18, 38, 28, 29, 30, 31, 32, 33, 34,
                    MQAR_QUERY_SENTINEL, 54, 55, 56, 57, 58, 59, 20, 19, 18, 17, 16, 15, 14,
                    MQAR_QUERY_SENTINEL, 18,
                ],
                38,
                TokenSpan::new(8, 9)?,
                8,
                crate::SyntheticProbeMode::TreeOnly,
            )?,
        ],
    })
}

fn target_rank(logits: &[f32], target_token_id: i64) -> Result<usize, FractalError> {
    let target_index = usize::try_from(target_token_id).map_err(|_| {
        FractalError::InvalidConfig(format!(
            "hybrid_rescue_prevalidation target token id {} cannot be converted to usize",
            target_token_id
        ))
    })?;
    let target_logit = *logits.get(target_index).ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "hybrid_rescue_prevalidation target token {} is outside vocab size {}",
            target_token_id,
            logits.len()
        ))
    })?;
    Ok(1 + logits.iter().filter(|value| **value > target_logit).count())
}

trait GatheredRecallLike {
    fn evidence_token_recall(&self) -> f32;
}

impl GatheredRecallLike for fractal_core::GatheredCandidateRecall {
    fn evidence_token_recall(&self) -> f32 {
        self.evidence_token_recall()
    }
}

fn ensure_nonzero(name: &str, value: usize) -> Result<(), FractalError> {
    if value == 0 {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be greater than zero"
        )));
    }
    Ok(())
}

fn ensure_match(name: &str, actual: usize, expected: usize) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;

    use super::*;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn baseline_hybrid_rescue_default_config_is_phase1_compatible() {
        BaselineHybridRescueModelConfig::default().validate().unwrap();
    }

    #[test]
    fn default_hybrid_suites_include_required_phase1_labels() {
        let suites = default_hybrid_rescue_prevalidation_suites().unwrap();
        let kinds = suites.iter().map(|suite| suite.kind).collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                HybridRescueSuiteKind::Mqar,
                HybridRescueSuiteKind::Copy,
                HybridRescueSuiteKind::Induction,
                HybridRescueSuiteKind::RetrievalHeavy
            ]
        );
    }

    #[test]
    fn hybrid_prevalidation_runs_initial_four_modes() {
        let device = <TestBackend as Backend>::Device::default();
        let model = build_baseline_hybrid_rescue_model::<TestBackend>(
            BaselineHybridRescueModelConfig::default(),
            &device,
        )
        .unwrap();
        let suites = default_hybrid_rescue_prevalidation_suites().unwrap();
        let report = run_hybrid_rescue_prevalidation_with_modes(
            &model,
            &suites[..1],
            &HybridRescueProbeMode::INITIAL_FOUR,
            &device,
        )
        .unwrap();

        assert_eq!(report.suites.len(), 1);
        assert_eq!(report.suites[0].mode_reports.len(), 4);
        assert_eq!(report.suites[0].sample_count, 2);
    }

    #[test]
    fn hybrid_results_ledger_default_path_points_at_v3_file() {
        let path = default_hybrid_results_ledger_path("/tmp/fractal");
        assert_eq!(path, PathBuf::from("/tmp/fractal/docs/v3-results-ledger.jsonl"));
    }
}
