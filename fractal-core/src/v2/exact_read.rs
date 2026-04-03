use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::{router::FractalRouteOutput, state::LeafTokenCache};

const EXACT_READ_INIT_MIN: f64 = -0.08;
const EXACT_READ_INIT_MAX: f64 = 0.08;
const MASKED_SCORE_FLOOR: f64 = -1.0e9;

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactLeafReadShape {
    pub query_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub head_count: usize,
    pub top_leaf_reads: usize,
    pub leaf_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactReadHistogramBin {
    pub value: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExactLeafReadDiagnostics {
    pub fraction_using_exact_read: f32,
    pub selected_token_position_histogram: Vec<ExactReadHistogramBin>,
    pub average_attention_entropy_per_head: Vec<f32>,
    pub average_top_token_probability_per_head: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ExactLeafReadOutput<B: Backend> {
    selected_token_indices: Tensor<B, 3, Int>,
    selected_token_absolute_positions: Tensor<B, 3, Int>,
    selected_token_mask: Tensor<B, 3, Bool>,
    selected_token_scores: Tensor<B, 3>,
    attention_weights: Tensor<B, 4>,
    read_values: Tensor<B, 4>,
    diagnostics: ExactLeafReadDiagnostics,
}

impl<B: Backend> ExactLeafReadOutput<B> {
    pub(crate) fn new(
        selected_token_indices: Tensor<B, 3, Int>,
        selected_token_absolute_positions: Tensor<B, 3, Int>,
        selected_token_mask: Tensor<B, 3, Bool>,
        selected_token_scores: Tensor<B, 3>,
        attention_weights: Tensor<B, 4>,
        read_values: Tensor<B, 4>,
        diagnostics: ExactLeafReadDiagnostics,
    ) -> Result<Self, FractalError> {
        let [batch_size, head_count, top_leaf_reads] = selected_token_indices.dims();
        ensure_nonzero("exact_read_output.batch_size", batch_size)?;
        ensure_nonzero("exact_read_output.head_count", head_count)?;
        ensure_nonzero("exact_read_output.top_leaf_reads", top_leaf_reads)?;
        ensure_dims3(
            "exact_read_output.selected_token_absolute_positions",
            selected_token_absolute_positions.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;
        ensure_dims3(
            "exact_read_output.selected_token_mask",
            selected_token_mask.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;
        ensure_dims3(
            "exact_read_output.selected_token_scores",
            selected_token_scores.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;
        let [weight_batch_size, weight_head_count, weight_top_leaf_reads, leaf_size] =
            attention_weights.dims();
        ensure_match(
            "exact_read_output.attention_weights.batch_size",
            weight_batch_size,
            batch_size,
        )?;
        ensure_match(
            "exact_read_output.attention_weights.head_count",
            weight_head_count,
            head_count,
        )?;
        ensure_match(
            "exact_read_output.attention_weights.top_leaf_reads",
            weight_top_leaf_reads,
            top_leaf_reads,
        )?;
        ensure_nonzero("exact_read_output.attention_weights.leaf_size", leaf_size)?;
        let [value_batch_size, value_head_count, value_top_leaf_reads, value_dim] =
            read_values.dims();
        ensure_match(
            "exact_read_output.read_values.batch_size",
            value_batch_size,
            batch_size,
        )?;
        ensure_match(
            "exact_read_output.read_values.head_count",
            value_head_count,
            head_count,
        )?;
        ensure_match(
            "exact_read_output.read_values.top_leaf_reads",
            value_top_leaf_reads,
            top_leaf_reads,
        )?;
        ensure_nonzero("exact_read_output.read_values.value_dim", value_dim)?;
        ensure_match(
            "exact_read_output.attention_entropy_per_head",
            diagnostics.average_attention_entropy_per_head.len(),
            head_count,
        )?;
        ensure_match(
            "exact_read_output.top_token_probability_per_head",
            diagnostics.average_top_token_probability_per_head.len(),
            head_count,
        )?;

        Ok(Self {
            selected_token_indices,
            selected_token_absolute_positions,
            selected_token_mask,
            selected_token_scores,
            attention_weights,
            read_values,
            diagnostics,
        })
    }

    pub fn selected_token_indices(&self) -> Tensor<B, 3, Int> {
        self.selected_token_indices.clone()
    }

    pub fn selected_token_absolute_positions(&self) -> Tensor<B, 3, Int> {
        self.selected_token_absolute_positions.clone()
    }

    pub fn selected_token_mask(&self) -> Tensor<B, 3, Bool> {
        self.selected_token_mask.clone()
    }

    pub fn selected_token_scores(&self) -> Tensor<B, 3> {
        self.selected_token_scores.clone()
    }

    pub fn attention_weights(&self) -> Tensor<B, 4> {
        self.attention_weights.clone()
    }

    pub fn read_values(&self) -> Tensor<B, 4> {
        self.read_values.clone()
    }

    pub fn diagnostics(&self) -> &ExactLeafReadDiagnostics {
        &self.diagnostics
    }
}

pub trait ExactLeafRead<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> ExactLeafReadShape;

    fn read(
        &self,
        query: Tensor<B, 2>,
        query_position: usize,
        routed: &FractalRouteOutput<B>,
        leaf_token_cache: &LeafTokenCache<B>,
    ) -> Result<ExactLeafReadOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineExactLeafReadConfig {
    pub query_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub head_count: usize,
    pub top_leaf_reads: usize,
    pub leaf_size: usize,
    #[config(
        default = "Initializer::Uniform { min: EXACT_READ_INIT_MIN, max: EXACT_READ_INIT_MAX }"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct BaselineExactLeafRead<B: Backend> {
    query_projection: StructuredProjection<B>,
    shape: ExactLeafReadShape,
}

impl BaselineExactLeafReadConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("baseline_exact_read.query_dim", self.query_dim)?;
        ensure_nonzero("baseline_exact_read.key_dim", self.key_dim)?;
        ensure_nonzero("baseline_exact_read.value_dim", self.value_dim)?;
        ensure_nonzero("baseline_exact_read.head_count", self.head_count)?;
        ensure_nonzero("baseline_exact_read.top_leaf_reads", self.top_leaf_reads)?;
        ensure_nonzero("baseline_exact_read.leaf_size", self.leaf_size)
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineExactLeafRead<B> {
        self.try_init(device)
            .unwrap_or_else(|error| panic!("invalid baseline exact read config: {error}"))
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineExactLeafRead<B>, FractalError> {
        self.validate()?;
        let query_projection =
            StructuredProjectionConfig::new(self.query_dim, self.head_count * self.key_dim)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(self.initializer.clone())
                .init(device);

        Ok(BaselineExactLeafRead {
            query_projection,
            shape: ExactLeafReadShape {
                query_dim: self.query_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                head_count: self.head_count,
                top_leaf_reads: self.top_leaf_reads,
                leaf_size: self.leaf_size,
            },
        })
    }
}

impl<B: Backend> BaselineExactLeafRead<B> {
    fn project_head_queries(&self, query: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, query_dim] = query.dims();
        assert_eq!(
            query_dim, self.shape.query_dim,
            "exact read query width mismatch: expected {}, got {}",
            self.shape.query_dim, query_dim
        );

        self.query_projection.forward(query).reshape([
            batch_size,
            self.shape.head_count,
            self.shape.key_dim,
        ])
    }
}

impl<B: Backend> ExactLeafRead<B> for BaselineExactLeafRead<B> {
    fn shape(&self) -> ExactLeafReadShape {
        self.shape
    }

    fn read(
        &self,
        query: Tensor<B, 2>,
        query_position: usize,
        routed: &FractalRouteOutput<B>,
        leaf_token_cache: &LeafTokenCache<B>,
    ) -> Result<ExactLeafReadOutput<B>, FractalError> {
        let [batch_size, query_dim] = query.dims();
        ensure_nonzero("exact_read.query.batch_size", batch_size)?;
        ensure_match(
            "exact_read.query.query_dim",
            query_dim,
            self.shape.query_dim,
        )?;

        let selected_leaf_indices = routed.selected_leaf_indices();
        let selected_leaf_mask = routed.selected_leaf_mask();
        let [route_batch_size, route_head_count, route_top_leaf_reads] =
            selected_leaf_indices.dims();
        ensure_match("exact_read.routed.batch_size", route_batch_size, batch_size)?;
        ensure_match(
            "exact_read.routed.head_count",
            route_head_count,
            self.shape.head_count,
        )?;
        ensure_match(
            "exact_read.routed.top_leaf_reads",
            route_top_leaf_reads,
            self.shape.top_leaf_reads,
        )?;
        ensure_dims3(
            "exact_read.routed.selected_leaf_mask",
            selected_leaf_mask.dims(),
            [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
        )?;

        let cache_shape =
            leaf_token_cache.shape(super::state::BatchTimelineMode::LockstepSharedTimeline);
        ensure_match(
            "exact_read.leaf_token_cache.batch_size",
            cache_shape.batch_size,
            batch_size,
        )?;
        ensure_match(
            "exact_read.leaf_token_cache.tokens_per_leaf",
            cache_shape.tokens_per_leaf,
            self.shape.leaf_size,
        )?;
        ensure_match(
            "exact_read.leaf_token_cache.key_dim",
            cache_shape.key_dim,
            self.shape.key_dim,
        )?;
        ensure_match(
            "exact_read.leaf_token_cache.value_dim",
            cache_shape.value_dim,
            self.shape.value_dim,
        )?;

        let device = query.device();
        let head_queries = self.project_head_queries(query);
        let leaf_indices_data = selected_leaf_indices
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(invalid_state_from_data("exact_read.selected_leaf_indices"))?;
        let leaf_mask_data = selected_leaf_mask
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .map_err(invalid_state_from_data("exact_read.selected_leaf_mask"))?;
        let leaf_count = leaf_token_cache.shared_spans().len();
        let mut selected_token_index_data =
            vec![-1i64; batch_size * self.shape.head_count * self.shape.top_leaf_reads];
        let mut selected_token_absolute_data =
            vec![-1i64; batch_size * self.shape.head_count * self.shape.top_leaf_reads];
        let mut selected_token_score_data =
            vec![0.0f32; batch_size * self.shape.head_count * self.shape.top_leaf_reads];
        let mut local_token_positions = Vec::new();
        let mut entropy_sum_per_head = vec![0.0f32; self.shape.head_count];
        let mut top_probability_sum_per_head = vec![0.0f32; self.shape.head_count];
        let mut active_reads_per_head = vec![0usize; self.shape.head_count];
        let mut active_read_count = 0usize;
        let mut head_value_batches = Vec::with_capacity(self.shape.head_count);
        let mut head_attention_batches = Vec::with_capacity(self.shape.head_count);

        for head_index in 0..self.shape.head_count {
            let mut batch_value_rows = Vec::with_capacity(batch_size);
            let mut batch_attention_rows = Vec::with_capacity(batch_size);

            for batch_index in 0..batch_size {
                let mut slot_values = Vec::with_capacity(self.shape.top_leaf_reads);
                let mut slot_attention_weights = Vec::with_capacity(self.shape.top_leaf_reads);

                for slot in 0..self.shape.top_leaf_reads {
                    let flat_index = flat_selection_index(
                        batch_index,
                        head_index,
                        slot,
                        self.shape.head_count,
                        self.shape.top_leaf_reads,
                    );

                    if !leaf_mask_data[flat_index] {
                        slot_values.push(Tensor::<B, 2>::zeros([1, self.shape.value_dim], &device));
                        slot_attention_weights
                            .push(Tensor::<B, 2>::zeros([1, self.shape.leaf_size], &device));
                        continue;
                    }

                    let leaf_index =
                        usize::try_from(leaf_indices_data[flat_index]).map_err(|_| {
                            FractalError::InvalidState(format!(
                                "exact read received negative selected leaf index {}",
                                leaf_indices_data[flat_index]
                            ))
                        })?;
                    if leaf_index >= leaf_count {
                        return Err(FractalError::InvalidState(format!(
                            "exact read selected leaf index {leaf_index} but cache only holds {leaf_count} sealed leaves"
                        )));
                    }

                    let selected_span = leaf_token_cache.shared_spans()[leaf_index];
                    if selected_span.end() > query_position {
                        return Err(FractalError::InvalidState(format!(
                            "exact read selected future sealed span [{}, {}) for query position {}",
                            selected_span.start(),
                            selected_span.end(),
                            query_position
                        )));
                    }

                    let leaf_keys = leaf_token_cache
                        .keys()
                        .slice([
                            batch_index..batch_index + 1,
                            leaf_index..leaf_index + 1,
                            0..self.shape.leaf_size,
                            0..self.shape.key_dim,
                        ])
                        .reshape([self.shape.leaf_size, self.shape.key_dim]);
                    let leaf_values = leaf_token_cache
                        .values()
                        .slice([
                            batch_index..batch_index + 1,
                            leaf_index..leaf_index + 1,
                            0..self.shape.leaf_size,
                            0..self.shape.value_dim,
                        ])
                        .reshape([self.shape.leaf_size, self.shape.value_dim]);
                    let leaf_mask = leaf_token_cache
                        .mask()
                        .slice([
                            batch_index..batch_index + 1,
                            leaf_index..leaf_index + 1,
                            0..self.shape.leaf_size,
                        ])
                        .reshape([self.shape.leaf_size]);
                    let head_query = head_queries
                        .clone()
                        .slice([
                            batch_index..batch_index + 1,
                            head_index..head_index + 1,
                            0..self.shape.key_dim,
                        ])
                        .reshape([1, self.shape.key_dim]);
                    let raw_scores = (leaf_keys.clone()
                        * Tensor::cat(vec![head_query.clone(); self.shape.leaf_size], 0))
                    .sum_dim(1)
                    .mul_scalar(1.0 / (self.shape.key_dim as f64).sqrt())
                    .reshape([self.shape.leaf_size]);
                    let masked_scores = Tensor::<B, 1>::zeros([self.shape.leaf_size], &device)
                        .add_scalar(MASKED_SCORE_FLOOR)
                        .mask_where(leaf_mask, raw_scores);
                    let attention = softmax(masked_scores.reshape([1, self.shape.leaf_size]), 1)
                        .reshape([self.shape.leaf_size]);
                    let repeated_attention = attention
                        .clone()
                        .reshape([self.shape.leaf_size, 1])
                        .repeat(&[1, self.shape.value_dim]);
                    let read_value = (leaf_values * repeated_attention)
                        .sum_dim(0)
                        .reshape([1, self.shape.value_dim]);
                    let attention_data = attention
                        .clone()
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .map_err(invalid_state_from_data("exact_read.attention_weights"))?;
                    let (selected_token_index, selected_token_score) =
                        argmax_probability(&attention_data);
                    let selected_token_absolute_position = selected_span
                        .start()
                        .checked_add(selected_token_index)
                        .ok_or_else(|| {
                            FractalError::InvalidState(
                                "exact read selected token position overflowed".to_string(),
                            )
                        })?;

                    selected_token_index_data[flat_index] = selected_token_index as i64;
                    selected_token_absolute_data[flat_index] =
                        selected_token_absolute_position as i64;
                    selected_token_score_data[flat_index] = selected_token_score;
                    local_token_positions.push(selected_token_index);
                    entropy_sum_per_head[head_index] += entropy(&attention_data);
                    top_probability_sum_per_head[head_index] += selected_token_score;
                    active_reads_per_head[head_index] += 1;
                    active_read_count += 1;
                    slot_values.push(read_value);
                    slot_attention_weights.push(attention.reshape([1, self.shape.leaf_size]));
                }

                batch_value_rows.push(Tensor::cat(slot_values, 0).reshape([
                    1,
                    self.shape.top_leaf_reads,
                    self.shape.value_dim,
                ]));
                batch_attention_rows.push(Tensor::cat(slot_attention_weights, 0).reshape([
                    1,
                    self.shape.top_leaf_reads,
                    self.shape.leaf_size,
                ]));
            }

            head_value_batches.push(Tensor::cat(batch_value_rows, 0).reshape([
                batch_size,
                1,
                self.shape.top_leaf_reads,
                self.shape.value_dim,
            ]));
            head_attention_batches.push(Tensor::cat(batch_attention_rows, 0).reshape([
                batch_size,
                1,
                self.shape.top_leaf_reads,
                self.shape.leaf_size,
            ]));
        }

        let read_values = Tensor::cat(head_value_batches, 1);
        let attention_weights = Tensor::cat(head_attention_batches, 1);
        let total_slots = batch_size * self.shape.head_count * self.shape.top_leaf_reads;
        let diagnostics = ExactLeafReadDiagnostics {
            fraction_using_exact_read: active_read_count as f32 / total_slots as f32,
            selected_token_position_histogram: histogram(&local_token_positions),
            average_attention_entropy_per_head: entropy_sum_per_head
                .into_iter()
                .zip(active_reads_per_head.iter().copied())
                .map(
                    |(sum, count)| {
                        if count == 0 {
                            0.0
                        } else {
                            sum / count as f32
                        }
                    },
                )
                .collect(),
            average_top_token_probability_per_head: top_probability_sum_per_head
                .into_iter()
                .zip(active_reads_per_head.iter().copied())
                .map(
                    |(sum, count)| {
                        if count == 0 {
                            0.0
                        } else {
                            sum / count as f32
                        }
                    },
                )
                .collect(),
        };

        ExactLeafReadOutput::new(
            Tensor::<B, 3, Int>::from_data(
                TensorData::new(
                    selected_token_index_data,
                    [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                ),
                &device,
            ),
            Tensor::<B, 3, Int>::from_data(
                TensorData::new(
                    selected_token_absolute_data,
                    [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                ),
                &device,
            ),
            Tensor::<B, 3, Bool>::from_data(
                TensorData::new(
                    leaf_mask_data,
                    [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                ),
                &device,
            ),
            Tensor::<B, 3>::from_data(
                TensorData::new(
                    selected_token_score_data,
                    [batch_size, self.shape.head_count, self.shape.top_leaf_reads],
                ),
                &device,
            ),
            attention_weights,
            read_values,
            diagnostics,
        )
    }
}

fn flat_selection_index(
    batch_index: usize,
    head_index: usize,
    slot: usize,
    head_count: usize,
    top_leaf_reads: usize,
) -> usize {
    ((batch_index * head_count + head_index) * top_leaf_reads) + slot
}

fn argmax_probability(values: &[f32]) -> (usize, f32) {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().copied().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }

    (best_index, best_value)
}

fn entropy(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn histogram(values: &[usize]) -> Vec<ExactReadHistogramBin> {
    let mut buckets = std::collections::BTreeMap::new();
    for value in values {
        *buckets.entry(*value).or_insert(0usize) += 1;
    }

    buckets
        .into_iter()
        .map(|(value, count)| ExactReadHistogramBin { value, count })
        .collect()
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

fn ensure_dims3(name: &str, actual: [usize; 3], expected: [usize; 3]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {:?}, got {:?}",
            expected, actual
        )));
    }

    Ok(())
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
    use burn::{
        backend::Candle,
        tensor::{Tensor, TensorData},
    };

    use crate::v2::{
        model::FractalV2ModelShape,
        read_fusion::ReadFusionShape,
        router::{
            BatchHeadRoute, FractalRouteOutput, FractalRouterHeadShape, FractalRoutingDiagnostics,
            HeadRouteTrace,
        },
        state::{FractalV2StateLayout, LeafTokenCacheRecord, TokenSpan},
        tree::TreeMergeCellShape,
        LeafSummarizerShape, LocalTrunkShape,
    };

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn test_layout(batch_size: usize, leaf_size: usize) -> FractalV2StateLayout {
        FractalV2StateLayout::from_model_shape(
            FractalV2ModelShape {
                vocab_size: 32,
                token_dim: 1,
                local_trunk: LocalTrunkShape {
                    token_dim: 1,
                    root_count: 1,
                    root_state_dim: 1,
                    root_readout_dim: 1,
                    leaf_size,
                },
                leaf_summarizer: LeafSummarizerShape {
                    readout_dim: 1,
                    leaf_size,
                    summary_dim: 1,
                    key_dim: 1,
                    value_dim: 2,
                    token_cache_key_dim: 1,
                    token_cache_value_dim: 2,
                },
                tree_merge_cell: TreeMergeCellShape {
                    summary_dim: 1,
                    key_dim: 1,
                    value_dim: 2,
                    scale_embedding_dim: 1,
                },
                router: FractalRouterHeadShape {
                    query_dim: 1,
                    key_dim: 1,
                    head_count: 1,
                    beam_width: 1,
                    top_leaf_reads: 1,
                    allow_early_stop: false,
                },
                read_fusion: ReadFusionShape {
                    root_count: 1,
                    root_readout_dim: 1,
                    retrieved_value_dim: 2,
                    fused_readout_dim: 2,
                },
                exact_read: ExactLeafReadShape {
                    query_dim: 1,
                    key_dim: 1,
                    value_dim: 2,
                    head_count: 1,
                    top_leaf_reads: 1,
                    leaf_size,
                },
            },
            batch_size,
        )
        .unwrap()
    }

    fn test_cache(device: &<TestBackend as Backend>::Device) -> LeafTokenCache<TestBackend> {
        let leaf_size = 4;
        let layout = test_layout(1, leaf_size);
        LeafTokenCache::from_record(
            LeafTokenCacheRecord {
                keys: Tensor::<TestBackend, 4>::from_data(
                    TensorData::from([[
                        [[0.1], [-0.2], [0.0], [0.3]],
                        [[0.0], [0.5], [2.0], [-1.0]],
                    ]]),
                    device,
                ),
                values: Tensor::<TestBackend, 4>::from_data(
                    TensorData::from([[
                        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
                        [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
                    ]]),
                    device,
                ),
                mask: Tensor::<TestBackend, 3, Bool>::from_data(
                    TensorData::from([[[true, true, true, true], [true, true, true, true]]]),
                    device,
                ),
                shared_spans: vec![TokenSpan::new(0, 4).unwrap(), TokenSpan::new(4, 8).unwrap()],
            },
            layout,
        )
        .unwrap()
    }

    fn routed_output(
        device: &<TestBackend as Backend>::Device,
        leaf_index: i64,
        mask: bool,
    ) -> FractalRouteOutput<TestBackend> {
        FractalRouteOutput::from_parts(
            Tensor::<TestBackend, 3, Int>::from_data(TensorData::from([[[leaf_index]]]), device),
            Tensor::<TestBackend, 3, Bool>::from_data(TensorData::from([[[mask]]]), device),
            Tensor::<TestBackend, 3>::from_data(TensorData::from([[[1.0f32]]]), device),
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[0.0f32, 0.0f32]]]]), device),
            vec![HeadRouteTrace {
                batch_routes: vec![BatchHeadRoute {
                    steps: Vec::new(),
                    selected_leaf_indices: if mask {
                        vec![leaf_index as usize]
                    } else {
                        Vec::new()
                    },
                    selected_leaf_spans: if mask {
                        vec![TokenSpan::new(
                            (leaf_index as usize) * 4,
                            ((leaf_index as usize) + 1) * 4,
                        )
                        .unwrap()]
                    } else {
                        Vec::new()
                    },
                    selected_leaf_scores: if mask { vec![1.0] } else { Vec::new() },
                }],
            }],
            FractalRoutingDiagnostics {
                routing_depth_histogram: Vec::new(),
                candidate_entropy_per_head: vec![0.0],
                selected_span_distance_histogram: Vec::new(),
                head_agreement_rate: 1.0,
                head_disagreement_rate: 0.0,
            },
        )
        .unwrap()
    }

    fn exact_read(device: &<TestBackend as Backend>::Device) -> BaselineExactLeafRead<TestBackend> {
        BaselineExactLeafReadConfig::new(1, 1, 2, 1, 1, 4)
            .with_initializer(Initializer::Constant { value: 1.0 })
            .init(device)
    }

    #[test]
    fn baseline_exact_leaf_read_selects_token_within_routed_leaf() {
        let device = Default::default();
        let cache = test_cache(&device);
        let routed = routed_output(&device, 1, true);
        let query = Tensor::<TestBackend, 2>::from_data(TensorData::from([[2.0f32]]), &device);

        let output = exact_read(&device).read(query, 8, &routed, &cache).unwrap();

        assert_eq!(
            output
                .selected_token_indices()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            output
                .selected_token_absolute_positions()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            vec![6]
        );
        let read_values = output
            .read_values()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let attention = output
            .attention_weights()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let expected_first =
            attention[0] * 5.0 + attention[1] * 6.0 + attention[2] * 7.0 + attention[3] * 8.0;
        let expected_second =
            attention[0] * 50.0 + attention[1] * 60.0 + attention[2] * 70.0 + attention[3] * 80.0;
        assert!((attention.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        assert!((read_values[0] - expected_first).abs() < 1.0e-4);
        assert!((read_values[1] - expected_second).abs() < 1.0e-4);
        assert_eq!(output.diagnostics().fraction_using_exact_read, 1.0);
        assert_eq!(
            output.diagnostics().selected_token_position_histogram,
            vec![ExactReadHistogramBin { value: 2, count: 1 }]
        );
    }

    #[test]
    fn baseline_exact_leaf_read_rejects_unsealed_leaf_indices() {
        let device = Default::default();
        let cache = test_cache(&device);
        let routed = routed_output(&device, 2, true);
        let query = Tensor::<TestBackend, 2>::from_data(TensorData::from([[2.0f32]]), &device);

        let error = exact_read(&device)
            .read(query, 8, &routed, &cache)
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidState(message) if message.contains("selected leaf index 2"))
        );
    }

    #[test]
    fn baseline_exact_leaf_read_rejects_future_sealed_spans() {
        let device = Default::default();
        let cache = test_cache(&device);
        let routed = routed_output(&device, 1, true);
        let query = Tensor::<TestBackend, 2>::from_data(TensorData::from([[2.0f32]]), &device);

        let error = exact_read(&device)
            .read(query, 7, &routed, &cache)
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidState(message) if message.contains("future sealed span"))
        );
    }

    #[test]
    fn baseline_exact_leaf_read_zeroes_inactive_routed_slots() {
        let device = Default::default();
        let cache = test_cache(&device);
        let routed = FractalRouteOutput::from_parts(
            Tensor::<TestBackend, 3, Int>::from_data(TensorData::from([[[1, -1]]]), &device),
            Tensor::<TestBackend, 3, Bool>::from_data(TensorData::from([[[true, false]]]), &device),
            Tensor::<TestBackend, 3>::from_data(TensorData::from([[[1.0f32, 0.0f32]]]), &device),
            Tensor::<TestBackend, 4>::from_data(
                TensorData::from([[[[0.0f32, 0.0f32], [0.0f32, 0.0f32]]]]),
                &device,
            ),
            vec![HeadRouteTrace {
                batch_routes: vec![BatchHeadRoute {
                    steps: Vec::new(),
                    selected_leaf_indices: vec![1],
                    selected_leaf_spans: vec![TokenSpan::new(4, 8).unwrap()],
                    selected_leaf_scores: vec![1.0],
                }],
            }],
            FractalRoutingDiagnostics {
                routing_depth_histogram: Vec::new(),
                candidate_entropy_per_head: vec![0.0],
                selected_span_distance_histogram: Vec::new(),
                head_agreement_rate: 1.0,
                head_disagreement_rate: 0.0,
            },
        )
        .unwrap();
        let query = Tensor::<TestBackend, 2>::from_data(TensorData::from([[2.0f32]]), &device);
        let output = BaselineExactLeafReadConfig::new(1, 1, 2, 1, 2, 4)
            .with_initializer(Initializer::Constant { value: 1.0 })
            .init(&device)
            .read(query, 8, &routed, &cache)
            .unwrap();

        assert_eq!(
            output
                .selected_token_mask()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap(),
            vec![true, false]
        );
        assert_eq!(
            output
                .selected_token_indices()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            vec![2, -1]
        );
        assert_eq!(
            output
                .selected_token_absolute_positions()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            vec![6, -1]
        );
        let read_values = output
            .read_values()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        assert_eq!(&read_values[2..4], &[0.0, 0.0]);
        assert!((output.diagnostics().fraction_using_exact_read - 0.5).abs() < 1.0e-6);
    }
}
