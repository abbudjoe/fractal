use burn::{
    config::Config,
    module::{Module, Param},
    nn::Initializer,
    tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::state::{TokenSpan, TreeNodeAddress, TreeSummaryState};

const ROUTER_INIT_MIN: f64 = -0.08;
const ROUTER_INIT_MAX: f64 = 0.08;
const HEAD_BIAS_SCALE: f32 = 4.0;

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalRouterHeadShape {
    pub query_dim: usize,
    pub key_dim: usize,
    pub head_count: usize,
    pub beam_width: usize,
    pub top_leaf_reads: usize,
    pub allow_early_stop: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingHistogramBin {
    pub value: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchRouteStep {
    pub level: usize,
    pub considered_candidate_indices: Vec<usize>,
    pub considered_candidate_spans: Vec<TokenSpan>,
    pub surviving_candidate_indices: Vec<usize>,
    pub surviving_candidate_spans: Vec<TokenSpan>,
    pub surviving_candidate_scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchHeadRoute {
    pub steps: Vec<BatchRouteStep>,
    pub selected_leaf_indices: Vec<usize>,
    pub selected_leaf_spans: Vec<TokenSpan>,
    pub selected_leaf_scores: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeadRouteTrace {
    pub batch_routes: Vec<BatchHeadRoute>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FractalRoutingDiagnostics {
    pub routing_depth_histogram: Vec<RoutingHistogramBin>,
    pub candidate_entropy_per_head: Vec<f32>,
    pub selected_span_distance_histogram: Vec<RoutingHistogramBin>,
    pub head_agreement_rate: f32,
    pub head_disagreement_rate: f32,
}

#[derive(Debug, Clone)]
pub struct FractalRouteOutput<B: Backend> {
    selected_leaf_indices: Tensor<B, 3, Int>,
    selected_leaf_mask: Tensor<B, 3, Bool>,
    selected_leaf_scores: Tensor<B, 3>,
    selected_leaf_values: Tensor<B, 4>,
    traces: Vec<HeadRouteTrace>,
    diagnostics: FractalRoutingDiagnostics,
}

impl<B: Backend> FractalRouteOutput<B> {
    pub fn selected_leaf_indices(&self) -> Tensor<B, 3, Int> {
        self.selected_leaf_indices.clone()
    }

    pub fn selected_leaf_mask(&self) -> Tensor<B, 3, Bool> {
        self.selected_leaf_mask.clone()
    }

    pub fn selected_leaf_scores(&self) -> Tensor<B, 3> {
        self.selected_leaf_scores.clone()
    }

    pub fn selected_leaf_values(&self) -> Tensor<B, 4> {
        self.selected_leaf_values.clone()
    }

    pub fn traces(&self) -> &[HeadRouteTrace] {
        &self.traces
    }

    pub fn diagnostics(&self) -> &FractalRoutingDiagnostics {
        &self.diagnostics
    }
}

pub trait FractalRouterHead<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> FractalRouterHeadShape;

    fn route(
        &self,
        query: Tensor<B, 2>,
        query_position: usize,
        tree: &TreeSummaryState<B>,
    ) -> Result<FractalRouteOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineFractalRouterHeadConfig {
    pub query_dim: usize,
    pub key_dim: usize,
    pub head_count: usize,
    pub beam_width: usize,
    pub top_leaf_reads: usize,
    #[config(default = false)]
    pub allow_early_stop: bool,
    #[config(default = "Initializer::Uniform { min: ROUTER_INIT_MIN, max: ROUTER_INIT_MAX }")]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct BaselineFractalRouterHead<B: Backend> {
    query_projection: StructuredProjection<B>,
    head_bias: Param<Tensor<B, 2>>,
    shape: FractalRouterHeadShape,
}

impl BaselineFractalRouterHeadConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("baseline_router.query_dim", self.query_dim)?;
        ensure_nonzero("baseline_router.key_dim", self.key_dim)?;
        ensure_nonzero("baseline_router.head_count", self.head_count)?;
        ensure_nonzero("baseline_router.beam_width", self.beam_width)?;
        ensure_nonzero("baseline_router.top_leaf_reads", self.top_leaf_reads)?;
        if self.allow_early_stop {
            return Err(FractalError::InvalidConfig(
                "baseline_router.allow_early_stop must remain false in v1".to_string(),
            ));
        }
        if self.top_leaf_reads > self.beam_width {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_router.top_leaf_reads ({}) must not exceed beam_width ({})",
                self.top_leaf_reads, self.beam_width
            )));
        }

        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineFractalRouterHead<B> {
        self.try_init(device)
            .unwrap_or_else(|error| panic!("invalid baseline router config: {error}"))
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineFractalRouterHead<B>, FractalError> {
        self.validate()?;
        let query_projection =
            StructuredProjectionConfig::new(self.query_dim, self.head_count * self.key_dim)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(self.initializer.clone())
                .init(device);

        Ok(BaselineFractalRouterHead {
            query_projection,
            head_bias: init_head_bias(self.head_count, self.key_dim, device),
            shape: FractalRouterHeadShape {
                query_dim: self.query_dim,
                key_dim: self.key_dim,
                head_count: self.head_count,
                beam_width: self.beam_width,
                top_leaf_reads: self.top_leaf_reads,
                allow_early_stop: false,
            },
        })
    }
}

impl<B: Backend> BaselineFractalRouterHead<B> {
    fn project_head_queries(&self, query: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch_size, query_dim] = query.dims();
        assert_eq!(
            query_dim, self.shape.query_dim,
            "router query width mismatch: expected {}, got {}",
            self.shape.query_dim, query_dim
        );

        self.query_projection.forward(query).reshape([
            batch_size,
            self.shape.head_count,
            self.shape.key_dim,
        ]) + self
            .head_bias
            .val()
            .reshape([1, self.shape.head_count, self.shape.key_dim])
    }
}

impl<B: Backend> FractalRouterHead<B> for BaselineFractalRouterHead<B> {
    fn shape(&self) -> FractalRouterHeadShape {
        self.shape
    }

    fn route(
        &self,
        query: Tensor<B, 2>,
        query_position: usize,
        tree: &TreeSummaryState<B>,
    ) -> Result<FractalRouteOutput<B>, FractalError> {
        let [batch_size, query_dim] = query.dims();
        ensure_nonzero("router.query.batch_size", batch_size)?;
        ensure_match("router.query.query_dim", query_dim, self.shape.query_dim)?;
        ensure_match("router.tree.key_dim", tree.key_dim(), self.shape.key_dim)?;

        if tree.root_address().is_none() {
            return Ok(empty_route_output(
                batch_size,
                self.shape,
                tree.value_dim(),
                &query.device(),
            ));
        }

        let head_queries = self.project_head_queries(query);
        let selection_count = self.shape.top_leaf_reads;
        let mut selected_index_data =
            vec![-1i64; batch_size * self.shape.head_count * selection_count];
        let mut selected_score_data =
            vec![0.0f32; batch_size * self.shape.head_count * selection_count];
        let mut selected_mask_data =
            vec![false; batch_size * self.shape.head_count * selection_count];
        let mut selections =
            vec![vec![vec![None; selection_count]; batch_size]; self.shape.head_count];
        let mut traces = Vec::with_capacity(self.shape.head_count);
        let mut candidate_entropy_per_head = Vec::with_capacity(self.shape.head_count);
        let mut routing_depths = Vec::with_capacity(batch_size * self.shape.head_count);
        let mut selected_distances = Vec::new();
        let mut primary_leaf_indices = vec![vec![None; batch_size]; self.shape.head_count];

        for head_index in 0..self.shape.head_count {
            let mut batch_routes = Vec::with_capacity(batch_size);
            let mut entropy_sum = 0.0f32;
            let mut entropy_count = 0usize;

            for batch_index in 0..batch_size {
                let head_query = head_queries
                    .clone()
                    .slice([
                        batch_index..batch_index + 1,
                        head_index..head_index + 1,
                        0..self.shape.key_dim,
                    ])
                    .reshape([1, self.shape.key_dim]);
                let routed =
                    route_single_batch(tree, head_query, query_position, self.shape, batch_index)?;

                for (slot, leaf_index) in routed.selected_leaf_indices.iter().enumerate() {
                    let flat_index = selection_flat_index(
                        batch_index,
                        head_index,
                        slot,
                        batch_size,
                        self.shape.head_count,
                        selection_count,
                    );
                    selected_index_data[flat_index] = *leaf_index as i64;
                    selected_score_data[flat_index] = routed.selected_leaf_scores[slot];
                    selected_mask_data[flat_index] = true;
                    selections[head_index][batch_index][slot] = Some(*leaf_index);
                }

                if let Some(primary) = routed.selected_leaf_indices.first().copied() {
                    primary_leaf_indices[head_index][batch_index] = Some(primary);
                }

                entropy_sum += routed.entropy_sum;
                entropy_count += routed.entropy_count;
                routing_depths.push(routed.depth);
                for span in &routed.selected_leaf_spans {
                    selected_distances.push(selected_span_distance(query_position, *span)?);
                }
                batch_routes.push(routed.into_batch_route());
            }

            candidate_entropy_per_head.push(if entropy_count == 0 {
                0.0
            } else {
                entropy_sum / entropy_count as f32
            });
            traces.push(HeadRouteTrace { batch_routes });
        }

        let selected_leaf_values = gather_selected_leaf_values(
            tree,
            &selections,
            batch_size,
            self.shape.head_count,
            selection_count,
        )?;
        let selected_leaf_indices = Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                selected_index_data,
                [batch_size, self.shape.head_count, selection_count],
            ),
            &selected_leaf_values.device(),
        );
        let selected_leaf_scores = Tensor::<B, 3>::from_data(
            TensorData::new(
                selected_score_data,
                [batch_size, self.shape.head_count, selection_count],
            ),
            &selected_leaf_values.device(),
        );
        let selected_leaf_mask = Tensor::<B, 3, Bool>::from_data(
            TensorData::new(
                selected_mask_data,
                [batch_size, self.shape.head_count, selection_count],
            ),
            &selected_leaf_values.device(),
        );
        let head_agreement_rate = pairwise_head_agreement(&primary_leaf_indices);
        let diagnostics = FractalRoutingDiagnostics {
            routing_depth_histogram: histogram(&routing_depths),
            candidate_entropy_per_head,
            selected_span_distance_histogram: histogram(&selected_distances),
            head_agreement_rate,
            head_disagreement_rate: 1.0 - head_agreement_rate,
        };

        Ok(FractalRouteOutput {
            selected_leaf_indices,
            selected_leaf_mask,
            selected_leaf_scores,
            selected_leaf_values,
            traces,
            diagnostics,
        })
    }
}

#[derive(Debug)]
struct RoutedBatchSelection {
    steps: Vec<BatchRouteStep>,
    selected_leaf_indices: Vec<usize>,
    selected_leaf_spans: Vec<TokenSpan>,
    selected_leaf_scores: Vec<f32>,
    entropy_sum: f32,
    entropy_count: usize,
    depth: usize,
}

impl RoutedBatchSelection {
    fn into_batch_route(self) -> BatchHeadRoute {
        BatchHeadRoute {
            steps: self.steps,
            selected_leaf_indices: self.selected_leaf_indices,
            selected_leaf_spans: self.selected_leaf_spans,
            selected_leaf_scores: self.selected_leaf_scores,
        }
    }
}

fn route_single_batch<B: Backend>(
    tree: &TreeSummaryState<B>,
    head_query: Tensor<B, 2>,
    query_position: usize,
    shape: FractalRouterHeadShape,
    batch_index: usize,
) -> Result<RoutedBatchSelection, FractalError> {
    let Some(root) = tree.root_address() else {
        return Ok(RoutedBatchSelection {
            steps: Vec::new(),
            selected_leaf_indices: Vec::new(),
            selected_leaf_spans: Vec::new(),
            selected_leaf_scores: Vec::new(),
            entropy_sum: 0.0,
            entropy_count: 0,
            depth: 0,
        });
    };
    let mut active = vec![root];
    let mut steps = Vec::new();
    let mut entropy_sum = 0.0f32;
    let mut entropy_count = 0usize;

    while active[0].level() > 0 {
        let considered = collect_child_candidates(tree, &active, query_position)?;
        if considered.is_empty() {
            let depth = steps.len();
            return Ok(RoutedBatchSelection {
                steps,
                selected_leaf_indices: Vec::new(),
                selected_leaf_spans: Vec::new(),
                selected_leaf_scores: Vec::new(),
                entropy_sum,
                entropy_count,
                depth,
            });
        }
        let next_is_leaf = considered[0].level() == 0;
        let keep = if next_is_leaf {
            shape.top_leaf_reads.min(considered.len())
        } else {
            shape.beam_width.min(considered.len())
        };
        let scored =
            score_candidates_for_batch(tree, head_query.clone(), &considered, batch_index)?;
        let selected = select_top_candidates(scored, &considered, keep)?;
        entropy_sum += entropy(&selected.surviving_scores);
        entropy_count += 1;
        active = selected.survivors;
        steps.push(BatchRouteStep {
            level: active[0].level(),
            considered_candidate_indices: considered.iter().map(|node| node.index()).collect(),
            considered_candidate_spans: considered.iter().map(|node| node.shared_span()).collect(),
            surviving_candidate_indices: active.iter().map(|node| node.index()).collect(),
            surviving_candidate_spans: active.iter().map(|node| node.shared_span()).collect(),
            surviving_candidate_scores: selected.surviving_scores,
        });
    }

    let selected_leaf_indices = active.iter().map(|node| node.index()).collect::<Vec<_>>();
    let selected_leaf_spans = active
        .iter()
        .map(|node| node.shared_span())
        .collect::<Vec<_>>();
    let selected_leaf_scores = steps
        .last()
        .map(|step| step.surviving_candidate_scores.clone())
        .unwrap_or_else(|| vec![1.0]);
    let depth = steps.len();

    Ok(RoutedBatchSelection {
        steps,
        selected_leaf_indices,
        selected_leaf_spans,
        selected_leaf_scores,
        entropy_sum,
        entropy_count,
        depth,
    })
}

#[derive(Debug)]
struct SelectedCandidates {
    survivors: Vec<TreeNodeAddress>,
    surviving_scores: Vec<f32>,
}

fn select_top_candidates<B: Backend>(
    scores: Tensor<B, 1>,
    candidates: &[TreeNodeAddress],
    keep: usize,
) -> Result<SelectedCandidates, FractalError> {
    ensure_nonzero("router.select.keep", keep)?;
    let candidate_count = candidates.len();
    ensure_nonzero("router.select.candidate_count", candidate_count)?;
    ensure_at_most("router.select.keep", keep, candidate_count)?;
    let (top_scores, top_indices) = scores
        .reshape([1, candidate_count])
        .topk_with_indices(keep, 1);
    let normalized = softmax(top_scores, 1);
    let surviving_scores = normalized
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data("router surviving scores"))?;
    let surviving_indices = top_indices
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(invalid_state_from_data("router surviving indices"))?;
    let mut survivors = Vec::with_capacity(keep);

    for candidate_index in surviving_indices {
        let index = usize::try_from(candidate_index).map_err(|_| {
            FractalError::InvalidState(format!(
                "router produced negative candidate index {candidate_index}"
            ))
        })?;
        survivors.push(candidates[index]);
    }

    Ok(SelectedCandidates {
        survivors,
        surviving_scores,
    })
}

fn collect_child_candidates<B: Backend>(
    tree: &TreeSummaryState<B>,
    active: &[TreeNodeAddress],
    query_position: usize,
) -> Result<Vec<TreeNodeAddress>, FractalError> {
    let mut children = Vec::new();
    for node in active {
        children.extend(
            tree.child_addresses(*node)?
                .into_iter()
                .filter(|child| child.shared_span().end() <= query_position),
        );
    }

    Ok(children)
}

fn score_candidates_for_batch<B: Backend>(
    tree: &TreeSummaryState<B>,
    head_query: Tensor<B, 2>,
    candidates: &[TreeNodeAddress],
    batch_index: usize,
) -> Result<Tensor<B, 1>, FractalError> {
    let key_dim = tree.key_dim();
    let mut keys = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let level = tree.level(candidate.level()).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "tree level {} does not exist while scoring router candidates",
                candidate.level()
            ))
        })?;
        let [batch_size, _, stored_key_dim] = level.keys().dims();
        ensure_match("router.score.key_dim", stored_key_dim, key_dim)?;
        if batch_index >= batch_size {
            return Err(FractalError::InvalidState(format!(
                "router batch index {batch_index} is out of bounds for batch size {batch_size}"
            )));
        }
        keys.push(
            level
                .keys()
                .slice([
                    batch_index..batch_index + 1,
                    candidate.index()..candidate.index() + 1,
                    0..key_dim,
                ])
                .reshape([1, key_dim]),
        );
    }

    let repeated_query = vec![head_query; candidates.len()];
    Ok((Tensor::cat(keys, 0) * Tensor::cat(repeated_query, 0))
        .sum_dim(1)
        .mul_scalar(1.0 / (key_dim as f64).sqrt())
        .reshape([candidates.len()]))
}

fn gather_selected_leaf_values<B: Backend>(
    tree: &TreeSummaryState<B>,
    selections: &[Vec<Vec<Option<usize>>>],
    batch_size: usize,
    head_count: usize,
    top_k: usize,
) -> Result<Tensor<B, 4>, FractalError> {
    let value_dim = tree.value_dim();
    let device = selections_device(tree)?;
    let level0 = tree.level(0).ok_or_else(|| {
        FractalError::InvalidState(
            "router cannot gather selected leaf values from an empty tree".to_string(),
        )
    })?;
    let mut head_batches = Vec::with_capacity(head_count);

    for head_selection in selections {
        let mut batch_rows = Vec::with_capacity(batch_size);
        for (batch_index, selected_indices) in head_selection.iter().enumerate() {
            let mut selected_values = Vec::with_capacity(top_k);
            for selected in selected_indices {
                selected_values.push(match selected {
                    Some(leaf_index) => level0
                        .values()
                        .slice([
                            batch_index..batch_index + 1,
                            *leaf_index..*leaf_index + 1,
                            0..value_dim,
                        ])
                        .reshape([1, value_dim]),
                    None => Tensor::<B, 2>::zeros([1, value_dim], &device),
                });
            }
            batch_rows.push(Tensor::cat(selected_values, 0).reshape([1, top_k, value_dim]));
        }
        head_batches.push(Tensor::cat(batch_rows, 0).reshape([batch_size, 1, top_k, value_dim]));
    }

    Ok(Tensor::cat(head_batches, 1))
}

fn selections_device<B: Backend>(tree: &TreeSummaryState<B>) -> Result<B::Device, FractalError> {
    let level0 = tree.level(0).ok_or_else(|| {
        FractalError::InvalidState(
            "router requires level 0 to resolve the output device".to_string(),
        )
    })?;
    Ok(level0.values().device())
}

fn empty_route_output<B: Backend>(
    batch_size: usize,
    shape: FractalRouterHeadShape,
    value_dim: usize,
    device: &B::Device,
) -> FractalRouteOutput<B> {
    FractalRouteOutput {
        selected_leaf_indices: Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                vec![-1i64; batch_size * shape.head_count * shape.top_leaf_reads],
                [batch_size, shape.head_count, shape.top_leaf_reads],
            ),
            device,
        ),
        selected_leaf_mask: Tensor::<B, 3, Bool>::from_data(
            TensorData::new(
                vec![false; batch_size * shape.head_count * shape.top_leaf_reads],
                [batch_size, shape.head_count, shape.top_leaf_reads],
            ),
            device,
        ),
        selected_leaf_scores: Tensor::<B, 3>::zeros(
            [batch_size, shape.head_count, shape.top_leaf_reads],
            device,
        ),
        selected_leaf_values: Tensor::<B, 4>::zeros(
            [
                batch_size,
                shape.head_count,
                shape.top_leaf_reads,
                value_dim,
            ],
            device,
        ),
        traces: (0..shape.head_count)
            .map(|_| HeadRouteTrace {
                batch_routes: (0..batch_size)
                    .map(|_| BatchHeadRoute {
                        steps: Vec::new(),
                        selected_leaf_indices: Vec::new(),
                        selected_leaf_spans: Vec::new(),
                        selected_leaf_scores: Vec::new(),
                    })
                    .collect(),
            })
            .collect(),
        diagnostics: FractalRoutingDiagnostics {
            routing_depth_histogram: Vec::new(),
            candidate_entropy_per_head: vec![0.0; shape.head_count],
            selected_span_distance_histogram: Vec::new(),
            head_agreement_rate: 1.0,
            head_disagreement_rate: 0.0,
        },
    }
}

fn pairwise_head_agreement(primary_leaf_indices: &[Vec<Option<usize>>]) -> f32 {
    let mut total_pairs = 0usize;
    let mut agreeing_pairs = 0usize;

    for head_index in 0..primary_leaf_indices.len() {
        for other_head_index in head_index + 1..primary_leaf_indices.len() {
            for (left, right) in primary_leaf_indices[head_index]
                .iter()
                .zip(primary_leaf_indices[other_head_index].iter())
            {
                let Some(left) = *left else {
                    continue;
                };
                let Some(right) = *right else {
                    continue;
                };
                total_pairs += 1;
                if left == right {
                    agreeing_pairs += 1;
                }
            }
        }
    }

    if total_pairs == 0 {
        1.0
    } else {
        agreeing_pairs as f32 / total_pairs as f32
    }
}

fn histogram(values: &[usize]) -> Vec<RoutingHistogramBin> {
    let mut buckets = std::collections::BTreeMap::new();
    for value in values {
        *buckets.entry(*value).or_insert(0usize) += 1;
    }

    buckets
        .into_iter()
        .map(|(value, count)| RoutingHistogramBin { value, count })
        .collect()
}

fn entropy(probabilities: &[f32]) -> f32 {
    probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

fn selected_span_distance(query_position: usize, span: TokenSpan) -> Result<usize, FractalError> {
    query_position.checked_sub(span.end()).ok_or_else(|| {
        FractalError::InvalidState(format!(
            "router selected future span [{}, {}) for query position {}",
            span.start(),
            span.end(),
            query_position
        ))
    })
}

fn selection_flat_index(
    batch_index: usize,
    head_index: usize,
    slot: usize,
    _batch_size: usize,
    head_count: usize,
    top_k: usize,
) -> usize {
    ((batch_index * head_count + head_index) * top_k) + slot
}

fn init_head_bias<B: Backend>(
    head_count: usize,
    key_dim: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let mut values = Vec::with_capacity(head_count * key_dim);
    for head_index in 0..head_count {
        let angle = core::f32::consts::TAU * head_index as f32 / head_count.max(1) as f32;
        for dim_index in 0..key_dim {
            let phase = angle + dim_index as f32 * core::f32::consts::FRAC_PI_2;
            values.push(phase.cos() * HEAD_BIAS_SCALE);
        }
    }

    Param::from_data(TensorData::new(values, [head_count, key_dim]), device)
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

fn ensure_at_most(name: &str, actual: usize, expected_max: usize) -> Result<(), FractalError> {
    if actual > expected_max {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be at most {expected_max}, got {actual}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        tensor::{TensorData, Tolerance},
    };

    use super::*;
    use crate::v2::{
        FractalV2ModelShape, FractalV2StateLayout, LeafSummarizerShape, LocalTrunkShape,
        ReadFusionShape, TreeLevelStoreRecord, TreeMergeCellShape, TreeSummaryStateRecord,
    };

    type TestBackend = Candle<f32, i64>;

    fn test_model_shape() -> FractalV2ModelShape {
        FractalV2ModelShape {
            vocab_size: 32,
            token_dim: 8,
            local_trunk: LocalTrunkShape {
                token_dim: 8,
                root_count: 2,
                root_state_dim: 6,
                root_readout_dim: 2,
                leaf_size: 16,
            },
            leaf_summarizer: LeafSummarizerShape {
                readout_dim: 2,
                leaf_size: 16,
                summary_dim: 3,
                key_dim: 2,
                value_dim: 3,
                token_cache_key_dim: 2,
                token_cache_value_dim: 3,
            },
            tree_merge_cell: TreeMergeCellShape {
                summary_dim: 3,
                key_dim: 2,
                value_dim: 3,
                scale_embedding_dim: 4,
            },
            router: FractalRouterHeadShape {
                query_dim: 2,
                key_dim: 2,
                head_count: 4,
                beam_width: 2,
                top_leaf_reads: 2,
                allow_early_stop: false,
            },
            read_fusion: ReadFusionShape {
                root_count: 2,
                root_readout_dim: 2,
                retrieved_value_dim: 3,
                fused_readout_dim: 5,
            },
        }
    }

    fn manual_tree(device: &<TestBackend as Backend>::Device) -> TreeSummaryState<TestBackend> {
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 1).unwrap();
        TreeSummaryState::from_record(
            TreeSummaryStateRecord {
                levels: vec![
                    TreeLevelStoreRecord {
                        summaries: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    1.0, 0.0, 0.0, //
                                    0.0, 1.0, 0.0, //
                                    -1.0, 0.0, 0.0, //
                                    0.0, -1.0, 0.0,
                                ],
                                [1, 4, 3],
                            ),
                            device,
                        ),
                        keys: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    1.0, 0.0, //
                                    0.0, 1.0, //
                                    -1.0, 0.0, //
                                    0.0, -1.0,
                                ],
                                [1, 4, 2],
                            ),
                            device,
                        ),
                        values: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    10.0, 0.0, 0.0, //
                                    0.0, 10.0, 0.0, //
                                    -10.0, 0.0, 0.0, //
                                    0.0, -10.0, 0.0,
                                ],
                                [1, 4, 3],
                            ),
                            device,
                        ),
                        level: 0,
                        shared_spans: vec![
                            TokenSpan::new(0, 16).unwrap(),
                            TokenSpan::new(16, 32).unwrap(),
                            TokenSpan::new(32, 48).unwrap(),
                            TokenSpan::new(48, 64).unwrap(),
                        ],
                    },
                    TreeLevelStoreRecord {
                        summaries: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    0.5, 0.5, 0.0, //
                                    -0.5, -0.5, 0.0,
                                ],
                                [1, 2, 3],
                            ),
                            device,
                        ),
                        keys: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    0.8, 0.2, //
                                    -0.2, -0.8,
                                ],
                                [1, 2, 2],
                            ),
                            device,
                        ),
                        values: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(
                                vec![
                                    5.0, 5.0, 0.0, //
                                    -5.0, -5.0, 0.0,
                                ],
                                [1, 2, 3],
                            ),
                            device,
                        ),
                        level: 1,
                        shared_spans: vec![
                            TokenSpan::new(0, 32).unwrap(),
                            TokenSpan::new(32, 64).unwrap(),
                        ],
                    },
                    TreeLevelStoreRecord {
                        summaries: Tensor::<TestBackend, 3>::zeros([1, 1, 3], device),
                        keys: Tensor::<TestBackend, 3>::from_data(
                            TensorData::new(vec![0.0, 0.0], [1, 1, 2]),
                            device,
                        ),
                        values: Tensor::<TestBackend, 3>::zeros([1, 1, 3], device),
                        level: 2,
                        shared_spans: vec![TokenSpan::new(0, 64).unwrap()],
                    },
                ],
            },
            layout,
        )
        .unwrap()
    }

    fn test_router(
        device: &<TestBackend as Backend>::Device,
    ) -> BaselineFractalRouterHead<TestBackend> {
        BaselineFractalRouterHeadConfig::new(2, 2, 4, 2, 2)
            .with_initializer(Initializer::Constant { value: 1.0 })
            .init(device)
    }

    #[test]
    fn baseline_router_routes_only_to_sealed_leaf_nodes_and_enforces_beam_width() {
        let device = Default::default();
        let tree = manual_tree(&device);
        let router = test_router(&device);
        let query = Tensor::<TestBackend, 2>::from_data([[3.0, 0.0]], &device);

        let routed = router.route(query, 64, &tree).unwrap();
        let selected_indices = routed
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let selected_mask = routed
            .selected_leaf_mask()
            .to_data()
            .into_vec::<bool>()
            .unwrap();
        let level0_spans = tree.level(0).unwrap().shared_spans();

        for (flat_index, is_selected) in selected_mask.iter().enumerate() {
            if !is_selected {
                continue;
            }
            let selected = usize::try_from(selected_indices[flat_index]).unwrap();
            assert!(selected < level0_spans.len());
        }

        for trace in routed.traces() {
            let batch_trace = &trace.batch_routes[0];
            for step in &batch_trace.steps {
                assert!(step.surviving_candidate_indices.len() <= 2);
                let probability_sum: f32 = step.surviving_candidate_scores.iter().sum();
                assert!((probability_sum - 1.0).abs() < 1.0e-5);
            }
        }
    }

    #[test]
    fn baseline_router_is_query_dependent() {
        let device = Default::default();
        let tree = manual_tree(&device);
        let router = test_router(&device);
        let positive = router
            .route(
                Tensor::<TestBackend, 2>::from_data([[3.0, 0.0]], &device),
                64,
                &tree,
            )
            .unwrap();
        let negative = router
            .route(
                Tensor::<TestBackend, 2>::from_data([[-3.0, 0.0]], &device),
                64,
                &tree,
            )
            .unwrap();

        let positive_indices = positive
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let negative_indices = negative
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();

        assert_ne!(positive_indices[0], negative_indices[0]);
    }

    #[test]
    fn baseline_router_rejects_future_leaf_spans_for_earlier_query_positions() {
        let device = Default::default();
        let tree = manual_tree(&device);
        let router = test_router(&device);
        let routed = router
            .route(
                Tensor::<TestBackend, 2>::from_data([[-3.0, 0.0]], &device),
                32,
                &tree,
            )
            .unwrap();
        let selected_indices = routed
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let selected_mask = routed
            .selected_leaf_mask()
            .to_data()
            .into_vec::<bool>()
            .unwrap();

        for (flat_index, is_selected) in selected_mask.iter().enumerate() {
            if !is_selected {
                continue;
            }
            let selected = usize::try_from(selected_indices[flat_index]).unwrap();
            assert!(selected < 2);
        }
    }

    #[test]
    fn baseline_router_is_deterministic_for_fixed_inputs() {
        let device = Default::default();
        let tree = manual_tree(&device);
        let router = test_router(&device);
        let query = Tensor::<TestBackend, 2>::from_data([[1.5, -0.5]], &device);

        let first = router.route(query.clone(), 64, &tree).unwrap();
        let second = router.route(query, 64, &tree).unwrap();

        assert_eq!(
            first
                .selected_leaf_indices()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            second
                .selected_leaf_indices()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap()
        );
        assert_eq!(
            first
                .selected_leaf_mask()
                .to_data()
                .into_vec::<bool>()
                .unwrap(),
            second
                .selected_leaf_mask()
                .to_data()
                .into_vec::<bool>()
                .unwrap()
        );
        first
            .selected_leaf_values()
            .to_data()
            .assert_approx_eq::<f32>(
                &second.selected_leaf_values().to_data(),
                Tolerance::default(),
            );
    }

    #[test]
    fn baseline_router_heads_do_not_all_choose_the_same_path_by_default() {
        let device = Default::default();
        let tree = manual_tree(&device);
        let router = test_router(&device);
        let query = Tensor::<TestBackend, 2>::from_data([[3.0, 0.0]], &device);

        let routed = router.route(query, 64, &tree).unwrap();
        let indices = routed
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();
        let head_top1 = indices
            .chunks_exact(2)
            .map(|chunk| chunk[0])
            .collect::<Vec<_>>();
        let unique = head_top1
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();

        assert!(unique.len() > 1);
        assert!(routed.diagnostics().head_disagreement_rate > 0.0);
    }
}
