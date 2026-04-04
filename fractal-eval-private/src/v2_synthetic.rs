use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use serde::Serialize;
use std::collections::BTreeMap;

use fractal_core::{
    error::FractalError, BatchHeadRoute, ExactLeafRead, FractalRouteOutput, FractalRouterHead,
    FractalRoutingDiagnostics, FractalV2MemoryMode, FractalV2Model, HeadRouteTrace, LeafSummarizer,
    LocalTrunk, ReadFusion, TokenSpan, TreeMergeCell,
};

use crate::{
    build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig,
};

pub const MIN_V2_PROBE_VOCAB_SIZE: usize = 64;
const COPY_SENTINEL: i64 = 1;
const ASSOC_SENTINEL: i64 = 2;
const INDUCTION_SENTINEL: i64 = 3;
const NOISY_SENTINEL: i64 = 4;
const COMPARE_SENTINEL: i64 = 5;
const QUERY_SENTINEL: i64 = 6;
const LESS_THAN_TOKEN: i64 = 60;
const EQUAL_TOKEN: i64 = 61;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticProbeKind {
    Copy,
    AssociativeRecall,
    Induction,
    NoisyRetrieval,
    FarTokenComparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticProbeMode {
    NoMemory,
    SummariesOnly,
    TreeOnly,
    TreePlusExactRead,
    OracleTreeOnly,
    OracleTreePlusExactRead,
    OracleTreePlusOracleExactRead,
}

impl SyntheticProbeMode {
    pub const ALL: [Self; 4] = [
        Self::NoMemory,
        Self::SummariesOnly,
        Self::TreeOnly,
        Self::TreePlusExactRead,
    ];

    pub const ALL_WITH_ORACLE: [Self; 7] = [
        Self::NoMemory,
        Self::SummariesOnly,
        Self::TreeOnly,
        Self::TreePlusExactRead,
        Self::OracleTreeOnly,
        Self::OracleTreePlusExactRead,
        Self::OracleTreePlusOracleExactRead,
    ];

    pub fn memory_mode(self) -> Option<FractalV2MemoryMode> {
        match self {
            Self::NoMemory => Some(FractalV2MemoryMode::NoMemory),
            Self::SummariesOnly => Some(FractalV2MemoryMode::SummariesOnly),
            Self::TreeOnly => Some(FractalV2MemoryMode::TreeOnly),
            Self::TreePlusExactRead => Some(FractalV2MemoryMode::TreePlusExactRead),
            Self::OracleTreeOnly
            | Self::OracleTreePlusExactRead
            | Self::OracleTreePlusOracleExactRead => None,
        }
    }

    pub fn supports(self, minimum_mode: Self) -> bool {
        match minimum_mode {
            Self::NoMemory => true,
            Self::SummariesOnly => !matches!(self, Self::NoMemory),
            Self::TreeOnly => matches!(
                self,
                Self::TreeOnly
                    | Self::TreePlusExactRead
                    | Self::OracleTreeOnly
                    | Self::OracleTreePlusExactRead
                    | Self::OracleTreePlusOracleExactRead
            ),
            Self::TreePlusExactRead => {
                matches!(
                    self,
                    Self::TreePlusExactRead
                        | Self::OracleTreePlusExactRead
                        | Self::OracleTreePlusOracleExactRead
                )
            }
            Self::OracleTreeOnly => {
                matches!(
                    self,
                    Self::OracleTreeOnly
                        | Self::OracleTreePlusExactRead
                        | Self::OracleTreePlusOracleExactRead
                )
            }
            Self::OracleTreePlusExactRead => matches!(
                self,
                Self::OracleTreePlusExactRead | Self::OracleTreePlusOracleExactRead
            ),
            Self::OracleTreePlusOracleExactRead => {
                matches!(self, Self::OracleTreePlusOracleExactRead)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SyntheticProbeSample {
    pub kind: SyntheticProbeKind,
    pub name: String,
    pub input_ids: Vec<i64>,
    pub target_position: usize,
    pub target_token_id: i64,
    pub evidence_span: TokenSpan,
    pub oracle_exact_token_position: usize,
    pub minimum_mode: SyntheticProbeMode,
}

impl SyntheticProbeSample {
    pub fn final_target(
        kind: SyntheticProbeKind,
        name: impl Into<String>,
        input_ids: Vec<i64>,
        target_token_id: i64,
        evidence_span: TokenSpan,
        oracle_exact_token_position: usize,
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
            oracle_exact_token_position,
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
        let minimum_input_len = leaf_size.checked_mul(2).ok_or_else(|| {
            FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' minimum input length overflowed leaf_size {}",
                self.name, leaf_size
            ))
        })?;
        if self.input_ids.len() < minimum_input_len {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' input length {} must span at least two sealed leaves of size {}",
                self.name,
                self.input_ids.len(),
                leaf_size
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
        if self.oracle_exact_token_position >= self.input_ids.len() {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' oracle_exact_token_position {} is out of bounds for input length {}",
                self.name,
                self.oracle_exact_token_position,
                self.input_ids.len()
            )));
        }
        if self.oracle_exact_token_position < self.evidence_span.start()
            || self.oracle_exact_token_position >= self.evidence_span.end()
        {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_sample '{}' oracle_exact_token_position {} must lie within evidence_span {:?}",
                self.name,
                self.oracle_exact_token_position,
                self.evidence_span
            )));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SyntheticProbeSuite {
    pub kind: SyntheticProbeKind,
    pub leaf_size: usize,
    pub samples: Vec<SyntheticProbeSample>,
}

impl SyntheticProbeSuite {
    pub fn validate_for_model(
        &self,
        vocab_size: usize,
        leaf_size: usize,
    ) -> Result<(), FractalError> {
        ensure_vocab_capacity(vocab_size)?;
        ensure_nonzero("synthetic_probe_suite.leaf_size", self.leaf_size, "")?;
        if self.leaf_size != leaf_size {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe_suite '{:?}' leaf_size {} must match model leaf_size {}",
                self.kind, self.leaf_size, leaf_size
            )));
        }
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SyntheticProbeSampleResult {
    pub sample_name: String,
    pub predicted_token_id: i64,
    pub target_token_id: i64,
    pub correct: bool,
    pub target_logit: f32,
    pub loss: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SyntheticProbeMetrics {
    pub accuracy: f32,
    pub mean_target_logit: f32,
    pub mean_loss: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SyntheticProbeModeReport {
    pub mode: SyntheticProbeMode,
    pub metrics: SyntheticProbeMetrics,
    pub sample_results: Vec<SyntheticProbeSampleResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SyntheticProbeReport {
    pub suites: Vec<SyntheticProbeSuiteReport>,
}

pub fn baseline_v2_synthetic_model<B: Backend>(
    device: &B::Device,
) -> Result<BaselineV2SyntheticModel<B>, FractalError> {
    build_baseline_v2_synthetic_model(BaselineV2SyntheticModelConfig::default(), device)
}

pub trait SyntheticProbeModel {
    type Device;

    fn vocab_size(&self) -> usize;
    fn leaf_size(&self) -> usize;

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

    fn leaf_size(&self) -> usize {
        self.shape().local_trunk.leaf_size
    }

    fn logits_for_sample(
        &self,
        sample: &SyntheticProbeSample,
        mode: SyntheticProbeMode,
        device: &Self::Device,
    ) -> Result<Vec<f32>, FractalError> {
        sample.validate_for_vocab(self.vocab_size(), self.leaf_size())?;

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
            device,
        );
        let logits = match mode.memory_mode() {
            Some(memory_mode) => self
                .forward_with_memory_mode(input_ids, memory_mode)?
                .logits()
                .narrow(1, sample.target_position, 1)
                .reshape([self.vocab_size()]),
            None => oracle_logits_for_sample(self, sample, mode, input_ids)?,
        };

        logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(invalid_state_from_data("synthetic_probe.logits"))
    }
}

fn oracle_logits_for_sample<B, LT, LS, TM, RH, ER, RF>(
    model: &FractalV2Model<B, LT, LS, TM, RH, ER, RF>,
    sample: &SyntheticProbeSample,
    mode: SyntheticProbeMode,
    input_ids: Tensor<B, 2, Int>,
) -> Result<Tensor<B, 1>, FractalError>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    ER: ExactLeafRead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
{
    let memory_mode = match mode {
        SyntheticProbeMode::OracleTreeOnly => FractalV2MemoryMode::TreeOnly,
        SyntheticProbeMode::OracleTreePlusExactRead
        | SyntheticProbeMode::OracleTreePlusOracleExactRead => {
            FractalV2MemoryMode::TreePlusExactRead
        }
        _ => {
            return Err(FractalError::InvalidConfig(format!(
                "synthetic_probe oracle logits require an oracle mode, got {:?}",
                mode
            )))
        }
    };
    let trace =
        model.forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)?;
    let step = trace.steps().get(sample.target_position).ok_or_else(|| {
        FractalError::InvalidState(format!(
            "synthetic_probe target position {} is out of bounds for {} retrieval steps",
            sample.target_position,
            trace.steps().len()
        ))
    })?;
    let query_position = sample.target_position.checked_add(1).ok_or_else(|| {
        FractalError::InvalidState("synthetic_probe query position overflowed".to_string())
    })?;
    let root_readouts = step.root_readouts();
    let active_root_count = root_readouts.dims()[1];
    let query = model.routing_query_from_root_readouts(root_readouts.clone(), active_root_count)?;
    let oracle_routed = oracle_route_output(
        model.router().shape(),
        trace.final_state().tree(),
        sample.evidence_span,
        query_position,
        &query.device(),
    )?;
    let oracle_exact = match mode {
        SyntheticProbeMode::OracleTreePlusOracleExactRead => oracle_exact_read_output(
            model.exact_read().shape(),
            &oracle_routed,
            trace.final_state().leaf_token_cache(),
            sample.oracle_exact_token_position,
            query_position,
            &query.device(),
        )?,
        _ => model.exact_read().read(
            query,
            query_position,
            &oracle_routed,
            trace.final_state().leaf_token_cache(),
        )?,
    };
    let logits = model.project_retrieval_step_logits_with_memory_mode(
        root_readouts,
        &oracle_routed,
        &oracle_exact,
        memory_mode,
    )?;

    Ok(logits.reshape([model.shape().vocab_size]))
}

fn oracle_route_output<B: Backend>(
    shape: fractal_core::FractalRouterHeadShape,
    tree: &fractal_core::TreeSummaryState<B>,
    evidence_span: TokenSpan,
    query_position: usize,
    device: &B::Device,
) -> Result<FractalRouteOutput<B>, FractalError> {
    let level_zero = tree.level(0).ok_or_else(|| {
        FractalError::InvalidState(
            "oracle synthetic probe requires a sealed level-0 summary tree".to_string(),
        )
    })?;
    let oracle_leaf_index = level_zero
        .shared_spans()
        .iter()
        .position(|span| span_covers(span, evidence_span) && span.end() <= query_position)
        .ok_or_else(|| {
            FractalError::InvalidState(format!(
                "oracle synthetic probe could not find a sealed evidence leaf covering [{}, {}) before query position {}",
                evidence_span.start(),
                evidence_span.end(),
                query_position
            ))
        })?;
    let oracle_leaf_span = level_zero.shared_spans()[oracle_leaf_index];
    let [batch_size, _, value_dim] = level_zero.values().dims();
    let total_slots = batch_size * shape.head_count * shape.top_leaf_reads;
    let mut selected_leaf_indices = vec![0i64; total_slots];
    let mut selected_leaf_mask = vec![false; total_slots];
    let mut selected_leaf_scores = vec![0.0f32; total_slots];
    for batch_index in 0..batch_size {
        for head_index in 0..shape.head_count {
            let flat_index = (batch_index * shape.head_count + head_index) * shape.top_leaf_reads;
            selected_leaf_indices[flat_index] = oracle_leaf_index as i64;
            selected_leaf_mask[flat_index] = true;
            selected_leaf_scores[flat_index] = 1.0;
        }
    }

    let selected_leaf_value = level_zero
        .values()
        .slice([
            0..batch_size,
            oracle_leaf_index..oracle_leaf_index + 1,
            0..value_dim,
        ])
        .reshape([batch_size, 1, 1, value_dim])
        .repeat(&[1, shape.head_count, 1, 1]);
    let selected_leaf_values = if shape.top_leaf_reads == 1 {
        selected_leaf_value
    } else {
        Tensor::cat(
            vec![
                selected_leaf_value,
                Tensor::<B, 4>::zeros(
                    [
                        batch_size,
                        shape.head_count,
                        shape.top_leaf_reads - 1,
                        value_dim,
                    ],
                    device,
                ),
            ],
            2,
        )
    };

    let selected_span_distance = query_position.saturating_sub(oracle_leaf_span.end());
    FractalRouteOutput::from_parts(
        Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                selected_leaf_indices,
                [batch_size, shape.head_count, shape.top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 3, burn::tensor::Bool>::from_data(
            TensorData::new(
                selected_leaf_mask,
                [batch_size, shape.head_count, shape.top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 3>::from_data(
            TensorData::new(
                selected_leaf_scores,
                [batch_size, shape.head_count, shape.top_leaf_reads],
            ),
            device,
        ),
        selected_leaf_values,
        (0..shape.head_count)
            .map(|_| HeadRouteTrace {
                batch_routes: (0..batch_size)
                    .map(|_| BatchHeadRoute {
                        steps: Vec::new(),
                        selected_leaf_indices: vec![oracle_leaf_index],
                        selected_leaf_spans: vec![oracle_leaf_span],
                        selected_leaf_scores: vec![1.0],
                    })
                    .collect(),
            })
            .collect(),
        FractalRoutingDiagnostics {
            routing_depth_histogram: vec![fractal_core::RoutingHistogramBin {
                value: 1,
                count: batch_size * shape.head_count,
            }],
            candidate_entropy_per_head: vec![0.0; shape.head_count],
            selected_span_distance_histogram: vec![fractal_core::RoutingHistogramBin {
                value: selected_span_distance,
                count: batch_size * shape.head_count,
            }],
            head_agreement_rate: 1.0,
            head_disagreement_rate: 0.0,
        },
    )
}

fn oracle_exact_read_output<B: Backend>(
    shape: fractal_core::ExactLeafReadShape,
    routed: &FractalRouteOutput<B>,
    leaf_token_cache: &fractal_core::LeafTokenCache<B>,
    oracle_exact_token_position: usize,
    query_position: usize,
    device: &B::Device,
) -> Result<fractal_core::ExactLeafReadOutput<B>, FractalError> {
    let selected_leaf_indices = routed.selected_leaf_indices();
    let selected_leaf_mask = routed.selected_leaf_mask();
    let [batch_size, head_count, top_leaf_reads] = selected_leaf_indices.dims();
    let leaf_index_data = selected_leaf_indices
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(invalid_state_from_data(
            "oracle_exact_read.selected_leaf_indices",
        ))?;
    let leaf_mask_data = selected_leaf_mask
        .clone()
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(invalid_state_from_data(
            "oracle_exact_read.selected_leaf_mask",
        ))?;
    let [cache_batch_size, cache_leaf_count, cache_leaf_size, value_dim] =
        leaf_token_cache.values().dims();
    if cache_batch_size != batch_size {
        return Err(FractalError::InvalidState(format!(
            "oracle exact read batch size mismatch: route batch {} vs cache batch {}",
            batch_size, cache_batch_size
        )));
    }
    if cache_leaf_size != shape.leaf_size {
        return Err(FractalError::InvalidState(format!(
            "oracle exact read leaf size mismatch: exact read {} vs cache {}",
            shape.leaf_size, cache_leaf_size
        )));
    }
    if value_dim != shape.value_dim {
        return Err(FractalError::InvalidState(format!(
            "oracle exact read value dim mismatch: exact read {} vs cache {}",
            shape.value_dim, value_dim
        )));
    }
    let cache_values = leaf_token_cache
        .values()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data(
            "oracle_exact_read.leaf_token_cache_values",
        ))?;
    let mut selected_token_indices = vec![-1i64; batch_size * head_count * top_leaf_reads];
    let mut selected_token_absolute_positions =
        vec![-1i64; batch_size * head_count * top_leaf_reads];
    let mut selected_token_scores = vec![0.0f32; batch_size * head_count * top_leaf_reads];
    let mut attention_weights =
        vec![0.0f32; batch_size * head_count * top_leaf_reads * shape.leaf_size];
    let mut read_values = vec![0.0f32; batch_size * head_count * top_leaf_reads * shape.value_dim];
    let mut selected_local_positions = Vec::new();

    for flat_index in 0..leaf_mask_data.len() {
        if !leaf_mask_data[flat_index] {
            continue;
        }
        let leaf_index = usize::try_from(leaf_index_data[flat_index]).map_err(|_| {
            FractalError::InvalidState(format!(
                "oracle exact read received negative selected leaf index {}",
                leaf_index_data[flat_index]
            ))
        })?;
        if leaf_index >= cache_leaf_count {
            return Err(FractalError::InvalidState(format!(
                "oracle exact read selected leaf index {leaf_index} but cache only holds {cache_leaf_count} leaves"
            )));
        }
        let selected_span = leaf_token_cache.shared_spans()[leaf_index];
        if oracle_exact_token_position < selected_span.start()
            || oracle_exact_token_position >= selected_span.end()
        {
            return Err(FractalError::InvalidState(format!(
                "oracle exact read token position {} does not lie within selected span [{}, {})",
                oracle_exact_token_position,
                selected_span.start(),
                selected_span.end()
            )));
        }
        if oracle_exact_token_position >= query_position {
            return Err(FractalError::InvalidState(format!(
                "oracle exact read token position {} must be before query position {}",
                oracle_exact_token_position, query_position
            )));
        }

        let local_index = oracle_exact_token_position - selected_span.start();
        selected_token_indices[flat_index] = local_index as i64;
        selected_token_absolute_positions[flat_index] = oracle_exact_token_position as i64;
        selected_token_scores[flat_index] = 1.0;
        attention_weights[flat_index * shape.leaf_size + local_index] = 1.0;
        selected_local_positions.push(local_index);

        let batch_index = flat_index / (head_count * top_leaf_reads);
        let slot_index = flat_index % top_leaf_reads;
        let head_index = (flat_index / top_leaf_reads) % head_count;
        let cache_offset = ((batch_index * cache_leaf_count + leaf_index) * shape.leaf_size
            + local_index)
            * shape.value_dim;
        let read_offset = ((batch_index * head_count + head_index) * top_leaf_reads + slot_index)
            * shape.value_dim;
        read_values[read_offset..read_offset + shape.value_dim]
            .copy_from_slice(&cache_values[cache_offset..cache_offset + shape.value_dim]);
    }

    fractal_core::ExactLeafReadOutput::new(
        Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                selected_token_indices,
                [batch_size, head_count, top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                selected_token_absolute_positions,
                [batch_size, head_count, top_leaf_reads],
            ),
            device,
        ),
        selected_leaf_mask,
        Tensor::<B, 3>::from_data(
            TensorData::new(
                selected_token_scores,
                [batch_size, head_count, top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 4>::from_data(
            TensorData::new(
                attention_weights,
                [batch_size, head_count, top_leaf_reads, shape.leaf_size],
            ),
            device,
        ),
        Tensor::<B, 4>::from_data(
            TensorData::new(
                read_values,
                [batch_size, head_count, top_leaf_reads, shape.value_dim],
            ),
            device,
        ),
        fractal_core::ExactLeafReadDiagnostics {
            fraction_using_exact_read: leaf_mask_data.iter().filter(|selected| **selected).count()
                as f32
                / leaf_mask_data.len() as f32,
            selected_token_position_histogram: exact_read_histogram(&selected_local_positions),
            average_attention_entropy_per_head: vec![0.0; head_count],
            average_top_token_probability_per_head: vec![1.0; head_count],
        },
    )
}

pub fn run_v2_synthetic_probe_suites<M: SyntheticProbeModel>(
    model: &M,
    suites: &[SyntheticProbeSuite],
    device: &M::Device,
) -> Result<SyntheticProbeReport, FractalError> {
    run_v2_synthetic_probe_suites_with_modes(model, suites, &SyntheticProbeMode::ALL, device)
}

pub fn run_v2_synthetic_probe_suites_with_modes<M: SyntheticProbeModel>(
    model: &M,
    suites: &[SyntheticProbeSuite],
    modes: &[SyntheticProbeMode],
    device: &M::Device,
) -> Result<SyntheticProbeReport, FractalError> {
    let mut reports = Vec::with_capacity(suites.len());
    for suite in suites {
        reports.push(run_v2_synthetic_probe_suite_with_modes(
            model, suite, modes, device,
        )?);
    }

    Ok(SyntheticProbeReport { suites: reports })
}

pub fn run_v2_synthetic_probe_suite<M: SyntheticProbeModel>(
    model: &M,
    suite: &SyntheticProbeSuite,
    device: &M::Device,
) -> Result<SyntheticProbeSuiteReport, FractalError> {
    run_v2_synthetic_probe_suite_with_modes(model, suite, &SyntheticProbeMode::ALL, device)
}

pub fn run_v2_synthetic_probe_suite_with_modes<M: SyntheticProbeModel>(
    model: &M,
    suite: &SyntheticProbeSuite,
    modes: &[SyntheticProbeMode],
    device: &M::Device,
) -> Result<SyntheticProbeSuiteReport, FractalError> {
    suite.validate_for_model(model.vocab_size(), model.leaf_size())?;
    if modes.is_empty() {
        return Err(FractalError::InvalidConfig(
            "synthetic_probe.modes must not be empty".to_string(),
        ));
    }

    let mut mode_reports = Vec::with_capacity(modes.len());
    for &mode in modes {
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

fn span_covers(outer: &TokenSpan, inner: TokenSpan) -> bool {
    outer.start() <= inner.start() && outer.end() >= inner.end()
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

fn exact_read_histogram(values: &[usize]) -> Vec<fractal_core::ExactReadHistogramBin> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for &value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
        .into_iter()
        .map(|(value, count)| fractal_core::ExactReadHistogramBin { value, count })
        .collect()
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
                1,
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
                1,
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
                8,
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
                10,
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
                3,
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
                3,
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
                    16,
                    56,
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
                    18,
                    QUERY_SENTINEL,
                    16,
                ],
                56,
                span(1, 3),
                2,
                SyntheticProbeMode::TreePlusExactRead,
            )
            .unwrap(),
            SyntheticProbeSample::final_target(
                SyntheticProbeKind::NoisyRetrieval,
                "noisy-beta",
                vec![
                    NOISY_SENTINEL,
                    27,
                    47,
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
                    28,
                    QUERY_SENTINEL,
                    27,
                ],
                47,
                span(1, 3),
                2,
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
                1,
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
                1,
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::marker::PhantomData;

    use burn::{
        backend::Candle,
        module::Module,
        nn::Initializer,
        tensor::{Bool, Int, Tensor, TensorData},
    };
    use fractal_core::{
        v2::{BaselineReadFusion, BaselineReadFusionConfig},
        BaselineLeafSummarizerConfig, BaselineLocalTrunkConfig, BaselineTreeMergeCellConfig,
        BatchHeadRoute, ExactLeafReadDiagnostics, ExactLeafReadOutput, ExactLeafReadShape,
        FractalRouteOutput, FractalRouterHeadShape, FractalRoutingDiagnostics, FractalV2Components,
        HeadRouteTrace,
    };

    type TestBackend = Candle<f32, i64>;
    type StructuralV2Model<B> = FractalV2Model<
        B,
        fractal_core::BaselineLocalTrunk<B>,
        fractal_core::BaselineLeafSummarizer<B>,
        fractal_core::BaselineTreeMergeCell<B>,
        StructuralProbeRouter<B>,
        StructuralProbeExactRead<B>,
        BaselineReadFusion<B>,
    >;

    #[derive(Module, Debug)]
    struct StructuralProbeRouter<B: Backend> {
        shape: FractalRouterHeadShape,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StructuralProbeExactRead<B: Backend> {
        shape: ExactLeafReadShape,
        selected_local_index: usize,
        fill_value: f32,
        _marker: PhantomData<B>,
    }

    impl<B: Backend> StructuralProbeRouter<B> {
        fn new(shape: FractalRouterHeadShape) -> Self {
            Self {
                shape,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StructuralProbeExactRead<B> {
        fn new(shape: ExactLeafReadShape, selected_local_index: usize, fill_value: f32) -> Self {
            Self {
                shape,
                selected_local_index,
                fill_value,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> FractalRouterHead<B> for StructuralProbeRouter<B> {
        fn shape(&self) -> FractalRouterHeadShape {
            self.shape
        }

        fn route(
            &self,
            query: Tensor<B, 2>,
            _query_position: usize,
            tree: &fractal_core::TreeSummaryState<B>,
        ) -> Result<FractalRouteOutput<B>, FractalError> {
            let [batch_size, query_dim] = query.dims();
            assert_eq!(query_dim, self.shape.query_dim);
            let level_zero = tree.level(0);
            let leaf_count = level_zero.map(|level| level.node_count()).unwrap_or(0);
            let has_tree = leaf_count > 0;
            let selected_leaf_index = if has_tree { 0i64 } else { -1i64 };
            let selected_leaf_mask = has_tree;
            let value_dim = tree.value_dim();

            FractalRouteOutput::from_parts(
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(
                        vec![
                            selected_leaf_index;
                            batch_size * self.shape.head_count * self.shape.top_leaf_reads
                        ],
                        [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 3, Bool>::from_data(
                    TensorData::new(
                        vec![
                            selected_leaf_mask;
                            batch_size * self.shape.head_count * self.shape.top_leaf_reads
                        ],
                        [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 3>::from_data(
                    TensorData::new(
                        vec![
                            if has_tree { 1.0 } else { 0.0 };
                            batch_size * self.shape.head_count * self.shape.top_leaf_reads
                        ],
                        [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 4>::zeros(
                    [
                        batch_size,
                        self.shape.head_count,
                        self.shape.top_leaf_reads,
                        value_dim,
                    ],
                    &query.device(),
                ),
                (0..self.shape.head_count)
                    .map(|_| HeadRouteTrace {
                        batch_routes: (0..batch_size)
                            .map(|_| BatchHeadRoute {
                                steps: Vec::new(),
                                selected_leaf_indices: if has_tree { vec![0] } else { Vec::new() },
                                selected_leaf_spans: if has_tree {
                                    vec![level_zero.unwrap().shared_spans()[0]]
                                } else {
                                    Vec::new()
                                },
                                selected_leaf_scores: if has_tree { vec![1.0] } else { Vec::new() },
                            })
                            .collect(),
                    })
                    .collect(),
                FractalRoutingDiagnostics {
                    routing_depth_histogram: Vec::new(),
                    candidate_entropy_per_head: vec![0.0; self.shape.head_count],
                    selected_span_distance_histogram: Vec::new(),
                    head_agreement_rate: 1.0,
                    head_disagreement_rate: 0.0,
                },
            )
        }
    }

    impl<B: Backend> ExactLeafRead<B> for StructuralProbeExactRead<B> {
        fn shape(&self) -> ExactLeafReadShape {
            self.shape
        }

        fn read(
            &self,
            query: Tensor<B, 2>,
            _query_position: usize,
            routed: &FractalRouteOutput<B>,
            leaf_token_cache: &fractal_core::LeafTokenCache<B>,
        ) -> Result<ExactLeafReadOutput<B>, FractalError> {
            let [batch_size, query_dim] = query.dims();
            assert_eq!(query_dim, self.shape.query_dim);
            let selected_leaf_indices = routed.selected_leaf_indices();
            let selected_leaf_mask = routed.selected_leaf_mask();
            let [mask_batch_size, head_count, top_leaf_reads] = selected_leaf_mask.dims();
            assert_eq!(mask_batch_size, batch_size);

            let leaf_indices = selected_leaf_indices
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap();
            let mask = selected_leaf_mask
                .clone()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap();
            let local_index = self.selected_local_index.min(self.shape.leaf_size - 1) as i64;
            let mut selected_token_indices = vec![-1i64; batch_size * head_count * top_leaf_reads];
            let mut selected_token_positions =
                vec![-1i64; batch_size * head_count * top_leaf_reads];
            let mut selected_token_scores = vec![0.0f32; batch_size * head_count * top_leaf_reads];
            let mut attention_weights =
                vec![0.0f32; batch_size * head_count * top_leaf_reads * self.shape.leaf_size];
            let mut read_values =
                vec![0.0f32; batch_size * head_count * top_leaf_reads * self.shape.value_dim];

            for (flat_index, is_selected) in mask.iter().copied().enumerate() {
                if !is_selected {
                    continue;
                }
                let leaf_index = usize::try_from(leaf_indices[flat_index]).unwrap();
                selected_token_indices[flat_index] = local_index;
                selected_token_positions[flat_index] =
                    (leaf_token_cache.shared_spans()[leaf_index].start() + local_index as usize)
                        as i64;
                selected_token_scores[flat_index] = 1.0;
                let attention_offset = flat_index * self.shape.leaf_size + local_index as usize;
                attention_weights[attention_offset] = 1.0;
                let value_offset = flat_index * self.shape.value_dim;
                read_values[value_offset..value_offset + self.shape.value_dim]
                    .fill(self.fill_value);
            }

            ExactLeafReadOutput::new(
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(
                        selected_token_indices,
                        [batch_size, head_count, top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(
                        selected_token_positions,
                        [batch_size, head_count, top_leaf_reads],
                    ),
                    &query.device(),
                ),
                selected_leaf_mask.clone(),
                Tensor::<B, 3>::from_data(
                    TensorData::new(
                        selected_token_scores,
                        [batch_size, head_count, top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        attention_weights,
                        [batch_size, head_count, top_leaf_reads, self.shape.leaf_size],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        read_values,
                        [batch_size, head_count, top_leaf_reads, self.shape.value_dim],
                    ),
                    &query.device(),
                ),
                ExactLeafReadDiagnostics {
                    fraction_using_exact_read: if mask.iter().any(|value| *value) {
                        1.0
                    } else {
                        0.0
                    },
                    selected_token_position_histogram: Vec::new(),
                    average_attention_entropy_per_head: vec![0.0; head_count],
                    average_top_token_probability_per_head: vec![1.0; head_count],
                },
            )
        }
    }

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

        fn leaf_size(&self) -> usize {
            self.leaf_size
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

    fn structural_v2_model<B: Backend>(device: &B::Device) -> StructuralV2Model<B> {
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
                router: StructuralProbeRouter::new(FractalRouterHeadShape {
                    query_dim: 4,
                    key_dim: 4,
                    head_count: 1,
                    beam_width: 1,
                    top_leaf_reads: 1,
                    allow_early_stop: false,
                }),
                exact_read: StructuralProbeExactRead::new(
                    ExactLeafReadShape {
                        query_dim: 4,
                        key_dim: 4,
                        value_dim: 6,
                        head_count: 1,
                        top_leaf_reads: 1,
                        leaf_size: 16,
                    },
                    1,
                    1.0,
                ),
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
            suite.validate_for_model(64, 16).unwrap();
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
                2,
                SyntheticProbeMode::TreeOnly,
            )
            .unwrap()],
        };

        let error = suite.validate_for_model(64, 16).unwrap_err();

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
    fn synthetic_probe_mode_supports_oracle_modes_explicitly() {
        assert!(SyntheticProbeMode::OracleTreeOnly.supports(SyntheticProbeMode::TreeOnly));
        assert!(!SyntheticProbeMode::OracleTreeOnly.supports(SyntheticProbeMode::TreePlusExactRead));
        assert!(SyntheticProbeMode::OracleTreePlusExactRead
            .supports(SyntheticProbeMode::TreePlusExactRead));
        assert!(SyntheticProbeMode::OracleTreePlusOracleExactRead
            .supports(SyntheticProbeMode::TreePlusExactRead));
        assert!(SyntheticProbeMode::OracleTreePlusOracleExactRead
            .supports(SyntheticProbeMode::OracleTreePlusExactRead));
        assert!(SyntheticProbeMode::OracleTreePlusExactRead.supports(SyntheticProbeMode::TreeOnly));
        assert!(!SyntheticProbeMode::NoMemory.supports(SyntheticProbeMode::TreeOnly));
    }

    #[test]
    fn default_v2_synthetic_probe_suites_are_memory_separating_on_real_v2_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = structural_v2_model::<TestBackend>(&device);

        for suite in default_v2_synthetic_probe_suites() {
            suite
                .validate_for_model(model.vocab_size(), model.leaf_size())
                .unwrap();
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
                let tree_only_route_spans = selected_route_spans(
                    tree_only_step.routed(),
                    tree_only
                        .final_state()
                        .tree()
                        .level(0)
                        .unwrap()
                        .shared_spans(),
                );
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
                    tree_only_route_spans
                        .iter()
                        .any(|span| span_overlaps(span, sample.evidence_span)),
                    "tree-only mode failed to route the evidence span {:?} for sample '{}'",
                    sample.evidence_span,
                    sample.name
                );
                assert!(
                    tree_only_route_spans.len()
                        < tree_only.final_state().sealed_leaves().shared_spans().len(),
                    "tree-only mode degenerated into dense retrieval for sample '{}'",
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
                let full_route_spans = selected_route_spans(
                    full_step.routed(),
                    full.final_state().tree().level(0).unwrap().shared_spans(),
                );
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
                    full_route_spans
                        .iter()
                        .any(|span| span_overlaps(span, sample.evidence_span)),
                    "full mode failed to route the evidence span {:?} for sample '{}'",
                    sample.evidence_span,
                    sample.name
                );
                if matches!(sample.minimum_mode, SyntheticProbeMode::TreePlusExactRead) {
                    assert!(
                        selected_absolute_positions(full_step.exact_read())
                            .into_iter()
                            .any(|position| sample.evidence_span.start() <= position
                                && position < sample.evidence_span.end()),
                        "full mode exact read did not intersect the evidence span {:?} for sample '{}'",
                        sample.evidence_span,
                        sample.name
                    );
                }
            }
        }
    }

    #[test]
    fn oracle_route_output_selects_the_evidence_leaf() {
        let device = <TestBackend as Backend>::Device::default();
        let model = structural_v2_model::<TestBackend>(&device);
        let suite = copy_probe_suite();
        let sample = &suite.samples[0];
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
            &device,
        );
        let trace = model
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)
            .unwrap();
        let step = trace.steps().last().unwrap();
        let query = model
            .routing_query_from_root_readouts(
                step.root_readouts(),
                model.shape().local_trunk.root_count,
            )
            .unwrap();
        let routed = oracle_route_output(
            model.router().shape(),
            trace.final_state().tree(),
            sample.evidence_span,
            sample.target_position + 1,
            &query.device(),
        )
        .unwrap();
        let selected_spans = selected_route_spans(
            &routed,
            trace.final_state().tree().level(0).unwrap().shared_spans(),
        );

        assert_eq!(
            selected_spans,
            vec![trace.final_state().tree().level(0).unwrap().shared_spans()[0]]
        );
        assert!(selected_spans
            .iter()
            .all(|span| span_covers(span, sample.evidence_span)));
    }

    #[test]
    fn oracle_exact_read_output_selects_the_explicit_evidence_token() {
        let device = <TestBackend as Backend>::Device::default();
        let model = structural_v2_model::<TestBackend>(&device);
        let suite = noisy_retrieval_probe_suite();
        let sample = &suite.samples[0];
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(sample.input_ids.clone(), [1, sample.input_ids.len()]),
            &device,
        );
        let trace = model
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)
            .unwrap();
        let step = trace.steps().last().unwrap();
        let query = model
            .routing_query_from_root_readouts(
                step.root_readouts(),
                model.shape().local_trunk.root_count,
            )
            .unwrap();
        let routed = oracle_route_output(
            model.router().shape(),
            trace.final_state().tree(),
            sample.evidence_span,
            sample.target_position + 1,
            &query.device(),
        )
        .unwrap();
        let exact = oracle_exact_read_output(
            model.exact_read().shape(),
            &routed,
            trace.final_state().leaf_token_cache(),
            sample.oracle_exact_token_position,
            sample.target_position + 1,
            &query.device(),
        )
        .unwrap();

        assert_eq!(
            selected_absolute_positions(&exact),
            vec![sample.oracle_exact_token_position]
        );
    }

    #[test]
    fn oracle_probe_modes_extend_the_fixture_harness() {
        let fixture = FixtureModel {
            vocab_size: 64,
            leaf_size: 16,
        };
        let modes = [
            SyntheticProbeMode::OracleTreeOnly,
            SyntheticProbeMode::OracleTreePlusExactRead,
            SyntheticProbeMode::OracleTreePlusOracleExactRead,
        ];
        let report = run_v2_synthetic_probe_suites_with_modes(
            &fixture,
            &default_v2_synthetic_probe_suites(),
            &modes,
            &(),
        )
        .unwrap();

        let copy = report
            .suites
            .iter()
            .find(|suite| suite.kind == SyntheticProbeKind::Copy)
            .unwrap();
        assert_eq!(
            copy.mode_report(SyntheticProbeMode::OracleTreeOnly)
                .unwrap()
                .metrics
                .accuracy,
            0.0
        );
        assert_eq!(
            copy.mode_report(SyntheticProbeMode::OracleTreePlusExactRead)
                .unwrap()
                .metrics
                .accuracy,
            1.0
        );
        assert_eq!(
            copy.mode_report(SyntheticProbeMode::OracleTreePlusOracleExactRead)
                .unwrap()
                .metrics
                .accuracy,
            1.0
        );
    }

    #[test]
    fn synthetic_probe_harness_runs_against_baseline_v2_model() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_synthetic_model::<TestBackend>(&device).unwrap();
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

    fn selected_route_spans(
        output: &FractalRouteOutput<TestBackend>,
        level_zero_spans: &[TokenSpan],
    ) -> Vec<TokenSpan> {
        let indices = output
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let mask = mask_values(output.selected_leaf_mask());

        indices
            .into_iter()
            .zip(mask)
            .filter_map(|(index, selected)| {
                if !selected {
                    return None;
                }
                usize::try_from(index)
                    .ok()
                    .and_then(|leaf_index| level_zero_spans.get(leaf_index).copied())
            })
            .collect()
    }

    fn selected_absolute_positions(
        output: &fractal_core::ExactLeafReadOutput<TestBackend>,
    ) -> Vec<usize> {
        let positions = output
            .selected_token_absolute_positions()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let mask = mask_values(output.selected_token_mask());

        positions
            .into_iter()
            .zip(mask)
            .filter_map(|(position, selected)| {
                if selected {
                    usize::try_from(position).ok()
                } else {
                    None
                }
            })
            .collect()
    }

    fn span_covers(outer: &TokenSpan, inner: TokenSpan) -> bool {
        outer.start() <= inner.start() && outer.end() >= inner.end()
    }

    fn span_overlaps(left: &TokenSpan, right: TokenSpan) -> bool {
        left.start() < right.end() && right.start() < left.end()
    }
}
