use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use fractal_core::{
    error::FractalError, ExactLeafRead, FractalRouterHead, FractalV2MemoryMode, FractalV2Model,
    LeafSummarizer, LocalTrunk, ReadFusion, TokenSpan, TreeMergeCell,
};

const MIN_V2_PROBE_VOCAB_SIZE: usize = 64;
const COPY_SENTINEL: i64 = 1;
const ASSOC_SENTINEL: i64 = 2;
const INDUCTION_SENTINEL: i64 = 3;
const NOISY_SENTINEL: i64 = 4;
const COMPARE_SENTINEL: i64 = 5;
const QUERY_SENTINEL: i64 = 6;
const LESS_THAN_TOKEN: i64 = 60;
const EQUAL_TOKEN: i64 = 61;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyntheticProbeKind {
    Copy,
    AssociativeRecall,
    Induction,
    NoisyRetrieval,
    FarTokenComparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyntheticProbeMode {
    NoMemory,
    TreeOnly,
    TreePlusExactRead,
}

impl SyntheticProbeMode {
    pub const ALL: [Self; 3] = [Self::NoMemory, Self::TreeOnly, Self::TreePlusExactRead];

    pub fn memory_mode(self) -> FractalV2MemoryMode {
        match self {
            Self::NoMemory => FractalV2MemoryMode::NoMemory,
            Self::TreeOnly => FractalV2MemoryMode::TreeOnly,
            Self::TreePlusExactRead => FractalV2MemoryMode::TreePlusExactRead,
        }
    }

    pub fn supports(self, minimum_mode: Self) -> bool {
        self.rank() >= minimum_mode.rank()
    }

    fn rank(self) -> usize {
        match self {
            Self::NoMemory => 0,
            Self::TreeOnly => 1,
            Self::TreePlusExactRead => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticProbeSample {
    pub kind: SyntheticProbeKind,
    pub name: String,
    pub input_ids: Vec<i64>,
    pub target_position: usize,
    pub target_token_id: i64,
    pub evidence_span: TokenSpan,
    pub minimum_mode: SyntheticProbeMode,
}

impl SyntheticProbeSample {
    pub fn final_target(
        kind: SyntheticProbeKind,
        name: impl Into<String>,
        input_ids: Vec<i64>,
        target_token_id: i64,
        evidence_span: TokenSpan,
        minimum_mode: SyntheticProbeMode,
    ) -> Result<Self, FractalError> {
        if input_ids.is_empty() {
            return Err(FractalError::InvalidConfig(
                "synthetic_probe_sample.input_ids must not be empty".to_string(),
            ));
        }
        let target_position = input_ids.len() - 1;

        Ok(Self {
            kind,
            name: name.into(),
            input_ids,
            target_position,
            target_token_id,
            evidence_span,
            minimum_mode,
        })
    }

    pub fn validate_for_vocab(
        &self,
        vocab_size: usize,
        leaf_size: usize,
    ) -> Result<(), FractalError> {
        if self.name.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "synthetic_probe_sample.name must not be empty".to_string(),
            ));
        }
        ensure_vocab_capacity(vocab_size)?;
        ensure_nonzero("synthetic_probe_sample.leaf_size", leaf_size, &self.name)?;
        if self.input_ids.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' must contain at least one token",
                self.name
            )));
        }
        if self.target_position >= self.input_ids.len() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' target_position {} is out of bounds for input length {}",
                self.name,
                self.target_position,
                self.input_ids.len()
            )));
        }
        for token_id in &self.input_ids {
            ensure_token_in_vocab(
                *token_id,
                vocab_size,
                &format!("synthetic_probe_sample '{}'.input_ids", self.name),
            )?;
        }
        ensure_token_in_vocab(
            self.target_token_id,
            vocab_size,
            &format!("synthetic_probe_sample '{}'.target_token_id", self.name),
        )?;
        if self.evidence_span.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' evidence_span must not be empty",
                self.name
            )));
        }
        if self.evidence_span.end() > self.input_ids.len() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' evidence_span {:?} exceeds input length {}",
                self.name,
                self.evidence_span,
                self.input_ids.len()
            )));
        }
        if self.evidence_span.end() > self.target_position {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' evidence_span {:?} must end before target position {}",
                self.name, self.evidence_span, self.target_position
            )));
        }
        if self.evidence_span.end() > leaf_size {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' evidence_span {:?} must lie within the first sealed leaf of size {}",
                self.name,
                self.evidence_span,
                leaf_size
            )));
        }
        if self.target_position < leaf_size {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' target_position {} must fall after the first sealed leaf of size {}",
                self.name,
                self.target_position,
                leaf_size
            )));
        }
        if matches!(self.minimum_mode, SyntheticProbeMode::NoMemory) {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' minimum_mode must require memory",
                self.name
            )));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticProbeSuite {
    pub kind: SyntheticProbeKind,
    pub leaf_size: usize,
    pub samples: Vec<SyntheticProbeSample>,
}

impl SyntheticProbeSuite {
    pub fn validate_for_vocab(&self, vocab_size: usize) -> Result<(), FractalError> {
        ensure_vocab_capacity(vocab_size)?;
        ensure_nonzero("synthetic_probe_suite.leaf_size", self.leaf_size, "")?;
        if self.samples.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_suite '{:?}' must contain at least one sample",
                self.kind
            )));
        }
        for sample in &self.samples {
            if sample.kind != self.kind {
                return Err(FractalError::InvalidConfig(format!(
                    "synthetic_probe_suite '{:?}' contains mismatched sample kind '{:?}'",
                    self.kind, sample.kind
                )));
            }
            sample.validate_for_vocab(vocab_size, self.leaf_size)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticProbeSampleResult {
    pub sample_name: String,
    pub predicted_token_id: i64,
    pub target_token_id: i64,
    pub correct: bool,
    pub target_logit: f32,
    pub loss: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticProbeMetrics {
    pub accuracy: f32,
    pub mean_target_logit: f32,
    pub mean_loss: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticProbeModeReport {
    pub mode: SyntheticProbeMode,
    pub metrics: SyntheticProbeMetrics,
    pub sample_results: Vec<SyntheticProbeSampleResult>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticProbeSuiteReport {
    pub kind: SyntheticProbeKind,
    pub sample_count: usize,
    pub mode_reports: Vec<SyntheticProbeModeReport>,
}

impl SyntheticProbeSuiteReport {
    pub fn mode_report(&self, mode: SyntheticProbeMode) -> Option<&SyntheticProbeModeReport> {
        self.mode_reports.iter().find(|report| report.mode == mode)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticProbeReport {
    pub suites: Vec<SyntheticProbeSuiteReport>,
}

pub trait SyntheticProbeModel {
    type Device;

    fn vocab_size(&self) -> usize;

    fn logits_for_sample(
        &self,
        sample: &SyntheticProbeSample,
        mode: SyntheticProbeMode,
        device: &Self::Device,
    ) -> Result<Vec<f32>, FractalError>;
}

impl<B, LT, LS, TM, RH, ER, RF> SyntheticProbeModel for FractalV2Model<B, LT, LS, TM, RH, ER, RF>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    ER: ExactLeafRead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
{
    type Device = B::Device;

    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }

    fn logits_for_sample(
        &self,
        sample: &SyntheticProbeSample,
        mode: SyntheticProbeMode,
        device: &Self::Device,
    ) -> Result<Vec<f32>, FractalError> {
        sample.validate_for_vocab(self.vocab_size(), self.shape().local_trunk.leaf_size)?;

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
            device,
        );
        let logits = self
            .forward_with_memory_mode(input_ids, mode.memory_mode())?
            .logits()
            .narrow(1, sample.target_position, 1)
            .reshape([self.vocab_size()]);

        logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(invalid_state_from_data("synthetic_probe.logits"))
    }
}

pub fn run_v2_synthetic_probe_suites<M: SyntheticProbeModel>(
    model: &M,
    suites: &[SyntheticProbeSuite],
    device: &M::Device,
) -> Result<SyntheticProbeReport, FractalError> {
    let mut reports = Vec::with_capacity(suites.len());
    for suite in suites {
        reports.push(run_v2_synthetic_probe_suite(model, suite, device)?);
    }

    Ok(SyntheticProbeReport { suites: reports })
}

pub fn run_v2_synthetic_probe_suite<M: SyntheticProbeModel>(
    model: &M,
    suite: &SyntheticProbeSuite,
    device: &M::Device,
) -> Result<SyntheticProbeSuiteReport, FractalError> {
    suite.validate_for_vocab(model.vocab_size())?;

    let mut mode_reports = Vec::with_capacity(SyntheticProbeMode::ALL.len());
    for mode in SyntheticProbeMode::ALL {
        let mut sample_results = Vec::with_capacity(suite.samples.len());
        for sample in &suite.samples {
            let logits = model.logits_for_sample(sample, mode, device)?;
            sample_results.push(score_probe_sample(sample, &logits, model.vocab_size())?);
        }
        mode_reports.push(SyntheticProbeModeReport {
            mode,
            metrics: aggregate_probe_metrics(&sample_results),
            sample_results,
        });
    }

    Ok(SyntheticProbeSuiteReport {
        kind: suite.kind,
        sample_count: suite.samples.len(),
        mode_reports,
    })
}

pub fn default_v2_synthetic_probe_suites() -> Vec<SyntheticProbeSuite> {
    vec![
        copy_probe_suite(),
        associative_recall_probe_suite(),
        induction_probe_suite(),
        noisy_retrieval_probe_suite(),
        far_token_comparison_probe_suite(),
    ]
}

fn score_probe_sample(
    sample: &SyntheticProbeSample,
    logits: &[f32],
    vocab_size: usize,
) -> Result<SyntheticProbeSampleResult, FractalError> {
    ensure_match("synthetic_probe.logits_vocab", logits.len(), vocab_size)?;
    let predicted_token_id = predicted_token_id(logits)?;
    let target_index = usize::try_from(sample.target_token_id).map_err(|_| {
        FractalError::InvalidConfig(format!(
            "synthetic_probe target token id {} cannot be converted to usize",
            sample.target_token_id
        ))
    })?;
    let target_logit = *logits.get(target_index).ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "synthetic_probe target token {} is outside vocab size {}",
            sample.target_token_id, vocab_size
        ))
    })?;

    Ok(SyntheticProbeSampleResult {
        sample_name: sample.name.clone(),
        predicted_token_id,
        target_token_id: sample.target_token_id,
        correct: predicted_token_id == sample.target_token_id,
        target_logit,
        loss: negative_log_likelihood(logits, sample.target_token_id)?,
    })
}

fn aggregate_probe_metrics(sample_results: &[SyntheticProbeSampleResult]) -> SyntheticProbeMetrics {
    let sample_count = sample_results.len().max(1) as f32;
    let (correct_count, target_logit_sum, loss_sum) = sample_results.iter().fold(
        (0usize, 0.0f32, 0.0f32),
        |(correct_acc, logit_acc, loss_acc), result| {
            (
                correct_acc + usize::from(result.correct),
                logit_acc + result.target_logit,
                loss_acc + result.loss,
            )
        },
    );

    SyntheticProbeMetrics {
        accuracy: correct_count as f32 / sample_count,
        mean_target_logit: target_logit_sum / sample_count,
        mean_loss: loss_sum / sample_count,
    }
}

fn negative_log_likelihood(logits: &[f32], target_token_id: i64) -> Result<f32, FractalError> {
    let target_index = usize::try_from(target_token_id).map_err(|_| {
        FractalError::InvalidConfig(format!(
            "synthetic_probe target token id {} cannot be converted to usize",
            target_token_id
        ))
    })?;
    let target_logit = *logits.get(target_index).ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "synthetic_probe target token {} is outside logits width {}",
            target_token_id,
            logits.len()
        ))
    })?;
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let log_sum_exp = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .sum::<f32>()
        .max(1.0e-12)
        .ln()
        + max_logit;

    Ok(log_sum_exp - target_logit)
}

fn predicted_token_id(logits: &[f32]) -> Result<i64, FractalError> {
    let (predicted_index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| {
            FractalError::InvalidState("synthetic_probe logits are empty".to_string())
        })?;

    Ok(predicted_index as i64)
}

fn ensure_match(name: &str, actual: usize, expected: usize) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} expected {expected}, got {actual}"
        )));
    }

    Ok(())
}

fn ensure_vocab_capacity(vocab_size: usize) -> Result<(), FractalError> {
    if vocab_size < MIN_V2_PROBE_VOCAB_SIZE {
        return Err(FractalError::InvalidConfig(format!(
            "synthetic_probe requires vocab_size >= {}, got {}",
            MIN_V2_PROBE_VOCAB_SIZE, vocab_size
        )));
    }

    Ok(())
}

fn ensure_nonzero(name: &str, value: usize, context: &str) -> Result<(), FractalError> {
    if value == 0 {
        let suffix = if context.is_empty() {
            String::new()
        } else {
            format!(" for '{context}'")
        };
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be greater than zero{suffix}"
        )));
    }

    Ok(())
}

fn ensure_token_in_vocab(token_id: i64, vocab_size: usize, name: &str) -> Result<(), FractalError> {
    if token_id < 0
        || usize::try_from(token_id)
            .ok()
            .filter(|id| *id < vocab_size)
            .is_none()
    {
        return Err(FractalError::InvalidConfig(format!(
            "{name} token id {} is outside vocab size {}",
            token_id, vocab_size
        )));
    }

    Ok(())
}

fn invalid_state_from_data(
    label: &'static str,
) -> impl FnOnce(burn::tensor::DataError) -> FractalError {
    move |error| FractalError::InvalidState(format!("{label} data conversion failed: {error}"))
}

fn span(start: usize, end: usize) -> TokenSpan {
    TokenSpan::new(start, end).expect("default synthetic spans must be valid")
}

fn copy_probe_suite() -> SyntheticProbeSuite {
    SyntheticProbeSuite {
        kind: SyntheticProbeKind::Copy,
        leaf_size: 16,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::Copy,
                "copy-alpha",
                vec![
                    COPY_SENTINEL,
                    41,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    QUERY_SENTINEL,
                    11,
                ],
                41,
                span(1, 2),
                SyntheticProbeMode::TreePlusExactRead,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::Copy,
                "copy-beta",
                vec![
                    COPY_SENTINEL,
                    42,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    43,
                    44,
                    45,
                    46,
                    QUERY_SENTINEL,
                    12,
                ],
                42,
                span(1, 2),
                SyntheticProbeMode::TreePlusExactRead,
            )
            .unwrap(),
        ],
    }
}

fn associative_recall_probe_suite() -> SyntheticProbeSuite {
    SyntheticProbeSuite {
        kind: SyntheticProbeKind::AssociativeRecall,
        leaf_size: 16,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::AssociativeRecall,
                "assoc-alpha",
                vec![
                    ASSOC_SENTINEL,
                    11,
                    31,
                    12,
                    32,
                    13,
                    33,
                    14,
                    34,
                    15,
                    35,
                    16,
                    36,
                    17,
                    37,
                    18,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    QUERY_SENTINEL,
                    14,
                ],
                34,
                span(7, 9),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::AssociativeRecall,
                "assoc-beta",
                vec![
                    ASSOC_SENTINEL,
                    19,
                    41,
                    20,
                    42,
                    21,
                    43,
                    22,
                    44,
                    23,
                    45,
                    24,
                    46,
                    25,
                    47,
                    26,
                    48,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    QUERY_SENTINEL,
                    23,
                ],
                45,
                span(9, 11),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
        ],
    }
}

fn induction_probe_suite() -> SyntheticProbeSuite {
    SyntheticProbeSuite {
        kind: SyntheticProbeKind::Induction,
        leaf_size: 16,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::Induction,
                "induction-alpha",
                vec![
                    INDUCTION_SENTINEL,
                    11,
                    21,
                    31,
                    11,
                    21,
                    32,
                    12,
                    22,
                    33,
                    13,
                    23,
                    34,
                    14,
                    24,
                    35,
                    15,
                    25,
                    36,
                    16,
                    26,
                    37,
                    17,
                    27,
                    38,
                    18,
                    28,
                    39,
                    11,
                    21,
                    QUERY_SENTINEL,
                    31,
                ],
                11,
                span(1, 4),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::Induction,
                "induction-beta",
                vec![
                    INDUCTION_SENTINEL,
                    12,
                    22,
                    41,
                    12,
                    22,
                    42,
                    13,
                    23,
                    43,
                    14,
                    24,
                    44,
                    15,
                    25,
                    45,
                    16,
                    26,
                    46,
                    17,
                    27,
                    47,
                    18,
                    28,
                    48,
                    19,
                    29,
                    49,
                    12,
                    22,
                    QUERY_SENTINEL,
                    41,
                ],
                12,
                span(1, 4),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
        ],
    }
}

fn noisy_retrieval_probe_suite() -> SyntheticProbeSuite {
    SyntheticProbeSuite {
        kind: SyntheticProbeKind::NoisyRetrieval,
        leaf_size: 16,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::NoisyRetrieval,
                "noisy-alpha",
                vec![
                    NOISY_SENTINEL,
                    11,
                    51,
                    12,
                    52,
                    13,
                    53,
                    14,
                    54,
                    15,
                    55,
                    16,
                    56,
                    17,
                    57,
                    18,
                    58,
                    11,
                    59,
                    12,
                    60,
                    13,
                    61,
                    14,
                    62,
                    15,
                    63,
                    16,
                    52,
                    17,
                    QUERY_SENTINEL,
                    16,
                ],
                56,
                span(11, 13),
                SyntheticProbeMode::TreePlusExactRead,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::NoisyRetrieval,
                "noisy-beta",
                vec![
                    NOISY_SENTINEL,
                    21,
                    41,
                    22,
                    42,
                    23,
                    43,
                    24,
                    44,
                    25,
                    45,
                    26,
                    46,
                    27,
                    47,
                    28,
                    48,
                    21,
                    49,
                    22,
                    50,
                    23,
                    51,
                    24,
                    52,
                    25,
                    53,
                    26,
                    54,
                    27,
                    QUERY_SENTINEL,
                    27,
                ],
                47,
                span(13, 15),
                SyntheticProbeMode::TreePlusExactRead,
            )
            .unwrap(),
        ],
    }
}

fn far_token_comparison_probe_suite() -> SyntheticProbeSuite {
    SyntheticProbeSuite {
        kind: SyntheticProbeKind::FarTokenComparison,
        leaf_size: 16,
        samples: vec![
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::FarTokenComparison,
                "compare-alpha",
                vec![
                    COMPARE_SENTINEL,
                    19,
                    41,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    52,
                    QUERY_SENTINEL,
                    19,
                ],
                LESS_THAN_TOKEN,
                span(1, 2),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::FarTokenComparison,
                "compare-beta",
                vec![
                    COMPARE_SENTINEL,
                    44,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    44,
                    QUERY_SENTINEL,
                    44,
                ],
                EQUAL_TOKEN,
                span(1, 2),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::Candle, nn::Initializer};
    use fractal_core::{
        v2::{BaselineReadFusion, BaselineReadFusionConfig},
        BaselineExactLeafReadConfig, BaselineFractalRouterHeadConfig, BaselineLeafSummarizerConfig,
        BaselineLocalTrunkConfig, BaselineTreeMergeCellConfig, FractalV2Components,
    };

    type TestBackend = Candle<f32, i64>;
    type BaselineV2Model<B> = FractalV2Model<
        B,
        fractal_core::BaselineLocalTrunk<B>,
        fractal_core::BaselineLeafSummarizer<B>,
        fractal_core::BaselineTreeMergeCell<B>,
        fractal_core::BaselineFractalRouterHead<B>,
        fractal_core::BaselineExactLeafRead<B>,
        BaselineReadFusion<B>,
    >;

    struct FixtureModel {
        vocab_size: usize,
        leaf_size: usize,
    }

    impl FixtureModel {
        fn predicted_token_for(
            &self,
            sample: &SyntheticProbeSample,
            mode: SyntheticProbeMode,
        ) -> i64 {
            if mode.supports(sample.minimum_mode) {
                sample.target_token_id
            } else {
                wrong_token(sample.target_token_id)
            }
        }
    }

    impl SyntheticProbeModel for FixtureModel {
        type Device = ();

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn logits_for_sample(
            &self,
            sample: &SyntheticProbeSample,
            mode: SyntheticProbeMode,
            _device: &Self::Device,
        ) -> Result<Vec<f32>, FractalError> {
            sample.validate_for_vocab(self.vocab_size, self.leaf_size)?;
            let predicted = self.predicted_token_for(sample, mode);
            let mut logits = vec![-4.0f32; self.vocab_size];
            logits[usize::try_from(predicted).unwrap()] = 8.0;
            logits[usize::try_from(sample.target_token_id).unwrap()] =
                if predicted == sample.target_token_id {
                    8.0
                } else {
                    2.0
                };

            Ok(logits)
        }
    }

    fn wrong_token(target: i64) -> i64 {
        if target == EQUAL_TOKEN {
            LESS_THAN_TOKEN
        } else {
            target + 1
        }
    }

    fn baseline_v2_model<B: Backend>(device: &B::Device) -> BaselineV2Model<B> {
        FractalV2Model::new(
            64,
            8,
            FractalV2Components {
                local_trunk: BaselineLocalTrunkConfig::new(8, 2, 6, 4, 16)
                    .try_init(device)
                    .unwrap(),
                leaf_summarizer: BaselineLeafSummarizerConfig {
                    readout_dim: 4,
                    leaf_size: 16,
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    token_cache_key_dim: 4,
                    token_cache_value_dim: 6,
                }
                .try_init(device)
                .unwrap(),
                tree_merge_cell: BaselineTreeMergeCellConfig {
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    scale_embedding_dim: 4,
                }
                .try_init(device)
                .unwrap(),
                router: BaselineFractalRouterHeadConfig {
                    query_dim: 4,
                    key_dim: 4,
                    head_count: 2,
                    beam_width: 2,
                    top_leaf_reads: 2,
                    allow_early_stop: false,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
                exact_read: BaselineExactLeafReadConfig {
                    query_dim: 4,
                    key_dim: 4,
                    value_dim: 6,
                    head_count: 2,
                    top_leaf_reads: 2,
                    leaf_size: 16,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
                read_fusion: BaselineReadFusionConfig {
                    root_count: 2,
                    root_readout_dim: 4,
                    routed_value_dim: 5,
                    exact_read_value_dim: 6,
                    fused_readout_dim: 8,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
            },
            device,
        )
        .unwrap()
    }

    #[test]
    fn default_v2_synthetic_probe_suites_cover_all_required_tasks() {
        let suites = default_v2_synthetic_probe_suites();
        let kinds = suites.iter().map(|suite| suite.kind).collect::<Vec<_>>();

        assert_eq!(
            kinds,
            vec![
                SyntheticProbeKind::Copy,
                SyntheticProbeKind::AssociativeRecall,
                SyntheticProbeKind::Induction,
                SyntheticProbeKind::NoisyRetrieval,
                SyntheticProbeKind::FarTokenComparison,
            ]
        );
        assert!(suites.iter().all(|suite| suite.samples.len() >= 2));
        assert!(suites.iter().all(|suite| suite.leaf_size == 16));
        for suite in &suites {
            suite.validate_for_vocab(64).unwrap();
            assert!(suite
                .samples
                .iter()
                .all(|sample| sample.input_ids.len() > suite.leaf_size));
        }
    }

    #[test]
    fn synthetic_probe_suite_rejects_mismatched_sample_kind() {
        let suite = SyntheticProbeSuite {
            kind: SyntheticProbeKind::Copy,
            leaf_size: 16,
            samples: vec![SyntheticProbeSample::final_target(
                SyntheticProbeKind::AssociativeRecall,
                "bad",
                vec![
                    ASSOC_SENTINEL,
                    11,
                    31,
                    QUERY_SENTINEL,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],
                31,
                span(1, 3),
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap()],
        };

        let error = suite.validate_for_vocab(64).unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("mismatched sample kind")
        ));
    }

    #[test]
    fn synthetic_probe_harness_distinguishes_memory_modes_with_fixture_model() {
        let fixture = FixtureModel {
            vocab_size: 64,
            leaf_size: 16,
        };
        let report =
            run_v2_synthetic_probe_suites(&fixture, &default_v2_synthetic_probe_suites(), &())
                .unwrap();

        let copy = report
            .suites
            .iter()
            .find(|suite| suite.kind == SyntheticProbeKind::Copy)
            .unwrap();
        assert_eq!(
            copy.mode_report(SyntheticProbeMode::TreePlusExactRead)
                .unwrap()
                .metrics
                .accuracy,
            1.0
        );
        assert_eq!(
            copy.mode_report(SyntheticProbeMode::TreeOnly)
                .unwrap()
                .metrics
                .accuracy,
            0.0
        );

        let assoc = report
            .suites
            .iter()
            .find(|suite| suite.kind == SyntheticProbeKind::AssociativeRecall)
            .unwrap();
        assert_eq!(
            assoc
                .mode_report(SyntheticProbeMode::NoMemory)
                .unwrap()
                .metrics
                .accuracy,
            0.0
        );
        assert_eq!(
            assoc
                .mode_report(SyntheticProbeMode::TreeOnly)
                .unwrap()
                .metrics
                .accuracy,
            1.0
        );

        let noisy = report
            .suites
            .iter()
            .find(|suite| suite.kind == SyntheticProbeKind::NoisyRetrieval)
            .unwrap();
        assert_eq!(
            noisy
                .mode_report(SyntheticProbeMode::TreePlusExactRead)
                .unwrap()
                .metrics
                .accuracy,
            1.0
        );
        assert_eq!(
            noisy
                .mode_report(SyntheticProbeMode::TreeOnly)
                .unwrap()
                .metrics
                .accuracy,
            0.0
        );
    }

    #[test]
    fn default_v2_synthetic_probe_suites_are_memory_separating_on_real_v2_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);

        for suite in default_v2_synthetic_probe_suites() {
            suite.validate_for_vocab(model.vocab_size()).unwrap();
            for sample in &suite.samples {
                let input_ids = Tensor::<TestBackend, 2, Int>::from_data(
                    TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
                    &device,
                );
                let no_memory = model
                    .forward_retrieval_trace_with_memory_mode(
                        input_ids.clone(),
                        FractalV2MemoryMode::NoMemory,
                    )
                    .unwrap();
                let tree_only = model
                    .forward_retrieval_trace_with_memory_mode(
                        input_ids.clone(),
                        FractalV2MemoryMode::TreeOnly,
                    )
                    .unwrap();
                let full = model
                    .forward_retrieval_trace_with_memory_mode(
                        input_ids,
                        FractalV2MemoryMode::TreePlusExactRead,
                    )
                    .unwrap();

                let no_memory_step = no_memory.steps().last().unwrap();
                assert!(
                    no_memory
                        .final_state()
                        .leaf_token_cache()
                        .shared_spans()
                        .is_empty(),
                    "no-memory mode unexpectedly sealed leaves for sample '{}'",
                    sample.name
                );
                assert!(
                    no_memory.final_state().tree().levels().is_empty(),
                    "no-memory mode unexpectedly built a tree for sample '{}'",
                    sample.name
                );
                assert!(
                    mask_values(no_memory_step.routed().selected_leaf_mask())
                        .into_iter()
                        .all(|selected| !selected),
                    "no-memory mode unexpectedly routed memory for sample '{}'",
                    sample.name
                );
                assert!(
                    mask_values(no_memory_step.exact_read().selected_token_mask())
                        .into_iter()
                        .all(|selected| !selected),
                    "no-memory mode unexpectedly exact-read tokens for sample '{}'",
                    sample.name
                );

                let tree_only_step = tree_only.steps().last().unwrap();
                assert!(
                    tree_only
                        .final_state()
                        .leaf_token_cache()
                        .shared_spans()
                        .iter()
                        .any(|sealed_span| span_covers(sealed_span, sample.evidence_span)),
                    "tree-only mode did not materialize evidence span {:?} for sample '{}'",
                    sample.evidence_span,
                    sample.name
                );
                assert!(
                    tree_only
                        .final_state()
                        .tree()
                        .level(0)
                        .unwrap()
                        .shared_spans()
                        .iter()
                        .any(|sealed_span| span_covers(sealed_span, sample.evidence_span)),
                    "tree-only mode did not surface evidence span {:?} in level-0 summaries for sample '{}'",
                    sample.evidence_span,
                    sample.name
                );
                assert!(
                    mask_values(tree_only_step.routed().selected_leaf_mask())
                        .into_iter()
                        .any(|selected| selected),
                    "tree-only mode failed to route any sealed leaf for sample '{}'",
                    sample.name
                );
                assert!(
                    mask_values(tree_only_step.exact_read().selected_token_mask())
                        .into_iter()
                        .all(|selected| !selected),
                    "tree-only mode unexpectedly performed exact read for sample '{}'",
                    sample.name
                );

                let full_step = full.steps().last().unwrap();
                assert!(
                    full.final_state()
                        .leaf_token_cache()
                        .shared_spans()
                        .iter()
                        .any(|sealed_span| span_covers(sealed_span, sample.evidence_span)),
                    "full mode did not materialize evidence span {:?} for sample '{}'",
                    sample.evidence_span,
                    sample.name
                );
                assert!(
                    mask_values(full_step.routed().selected_leaf_mask())
                        .into_iter()
                        .any(|selected| selected),
                    "full mode failed to route any sealed leaf for sample '{}'",
                    sample.name
                );
                assert!(
                    mask_values(full_step.exact_read().selected_token_mask())
                        .into_iter()
                        .any(|selected| selected),
                    "full mode failed to perform exact read for sample '{}'",
                    sample.name
                );
            }
        }
    }

    #[test]
    fn synthetic_probe_harness_runs_against_baseline_v2_model() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let report =
            run_v2_synthetic_probe_suites(&model, &default_v2_synthetic_probe_suites(), &device)
                .unwrap();

        assert_eq!(report.suites.len(), 5);
        for suite in &report.suites {
            assert_eq!(
                suite.sample_count,
                suite.mode_reports[0].sample_results.len()
            );
            assert_eq!(suite.mode_reports.len(), SyntheticProbeMode::ALL.len());
            for mode_report in &suite.mode_reports {
                assert!(mode_report.metrics.accuracy.is_finite());
                assert!(mode_report.metrics.mean_target_logit.is_finite());
                assert!(mode_report.metrics.mean_loss.is_finite());
            }
        }
    }

    fn mask_values(tensor: Tensor<TestBackend, 3, burn::tensor::Bool>) -> Vec<bool> {
        tensor
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
    }

    fn span_covers(outer: &TokenSpan, inner: TokenSpan) -> bool {
        outer.start() <= inner.start() && outer.end() >= inner.end()
    }
}
