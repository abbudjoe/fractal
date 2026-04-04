use std::{
    collections::{BTreeMap, BTreeSet},
    hint::black_box,
    time::Instant,
};

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
use serde::Serialize;

use fractal_core::{
    error::FractalError,
    summarize_root_readout_sequence,
    v2::{FractalV2ForwardOutput, FractalV2RetrievalTrace},
    ExactLeafRead, FractalRouterHead, LocalTrunk, RoutingHistogramBin, TreeNodeBatch,
};

use crate::{
    build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig,
};

pub const DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V2BenchmarkSurface {
    TokenAppend,
    LeafSealing,
    TreeUpdate,
    Routing,
    ExactLeafRead,
    ForwardPass,
}

impl V2BenchmarkSurface {
    pub const ALL: [Self; 6] = [
        Self::TokenAppend,
        Self::LeafSealing,
        Self::TreeUpdate,
        Self::Routing,
        Self::ExactLeafRead,
        Self::ForwardPass,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct V2BenchmarkConfig {
    pub sequence_lengths: Vec<usize>,
    pub leaf_size: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
}

impl V2BenchmarkConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.sequence_lengths.is_empty() {
            return Err(FractalError::InvalidConfig(
                "v2_benchmark.sequence_lengths must not be empty".to_string(),
            ));
        }
        if self.sequence_lengths.contains(&0) {
            return Err(FractalError::InvalidConfig(
                "v2_benchmark.sequence_lengths must all be greater than zero".to_string(),
            ));
        }
        if self.leaf_size == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_benchmark.leaf_size must be greater than zero".to_string(),
            ));
        }
        if let Some(sequence_length) = self
            .sequence_lengths
            .iter()
            .copied()
            .find(|sequence_length| *sequence_length < self.leaf_size)
        {
            return Err(FractalError::InvalidConfig(format!(
                "v2_benchmark.sequence_lengths must all be at least leaf_size {} (got {})",
                self.leaf_size, sequence_length
            )));
        }
        if self.iterations == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_benchmark.iterations must be greater than zero".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for V2BenchmarkConfig {
    fn default() -> Self {
        Self {
            sequence_lengths: DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS.to_vec(),
            leaf_size: 16,
            iterations: 3,
            warmup_iterations: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LeafUsageBin {
    pub leaf_index: usize,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2ObservabilitySnapshot {
    pub routing_sparsity: f32,
    pub root_collapse_mean_pairwise_cosine_similarity: f32,
    pub exact_read_usage: f32,
    pub mean_retrieval_distance: f32,
    pub tree_depth_reached: usize,
    pub level0_leaf_count: usize,
    pub head_agreement_rate: f32,
    pub has_dead_or_unused_tree_nodes: bool,
    pub selected_leaf_usage: Vec<V2LeafUsageBin>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2BenchmarkEntry {
    pub surface: V2BenchmarkSurface,
    pub sequence_length: usize,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub logical_tokens_per_iteration: usize,
    pub total_wall_time_ms: f64,
    pub mean_wall_time_ms: f64,
    pub tokens_per_sec: f64,
    pub peak_rss_bytes: u64,
    pub peak_rss_delta_bytes: u64,
    pub observability: V2ObservabilitySnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2BenchmarkReport {
    pub model: String,
    pub note: String,
    pub config: V2BenchmarkConfig,
    pub entries: Vec<V2BenchmarkEntry>,
}

pub fn run_baseline_v2_benchmark_suite<B: Backend>(
    config: V2BenchmarkConfig,
    device: &B::Device,
) -> Result<V2BenchmarkReport, FractalError> {
    config.validate()?;
    let model = build_baseline_v2_synthetic_model::<B>(
        BaselineV2SyntheticModelConfig::default().with_leaf_size(config.leaf_size),
        device,
    )?;
    run_v2_benchmark_suite_for_model(
        &model,
        config,
        "baseline_v2_random_init".to_string(),
        "process RSS metrics are sampled from getrusage(RUSAGE_SELF) after warmup; they are useful for trend detection, not precise kernel-level attribution".to_string(),
        device,
    )
}

pub fn run_v2_benchmark_suite_for_model<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    config: V2BenchmarkConfig,
    model_label: String,
    note: String,
    device: &B::Device,
) -> Result<V2BenchmarkReport, FractalError> {
    config.validate()?;
    let mut entries = Vec::new();

    for sequence_length in &config.sequence_lengths {
        let observability = observe_sequence(model, *sequence_length, device)?;
        for surface in V2BenchmarkSurface::ALL {
            entries.push(benchmark_surface(
                model,
                surface,
                *sequence_length,
                &config,
                &observability,
                device,
            )?);
        }
    }

    Ok(V2BenchmarkReport {
        model: model_label,
        note,
        config,
        entries,
    })
}

fn benchmark_surface<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    surface: V2BenchmarkSurface,
    sequence_length: usize,
    config: &V2BenchmarkConfig,
    observability: &V2ObservabilitySnapshot,
    device: &B::Device,
) -> Result<V2BenchmarkEntry, FractalError> {
    let timing = match surface {
        V2BenchmarkSurface::TokenAppend => benchmark_token_append(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
        V2BenchmarkSurface::LeafSealing => benchmark_leaf_sealing(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
        V2BenchmarkSurface::TreeUpdate => benchmark_tree_update(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
        V2BenchmarkSurface::Routing => benchmark_routing(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
        V2BenchmarkSurface::ExactLeafRead => benchmark_exact_leaf_read(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
        V2BenchmarkSurface::ForwardPass => benchmark_forward_pass(
            model,
            sequence_length,
            config.iterations,
            config.warmup_iterations,
            device,
        )?,
    };
    let logical_tokens_per_iteration = logical_tokens_per_iteration(surface, sequence_length);

    Ok(V2BenchmarkEntry {
        surface,
        sequence_length,
        iterations: config.iterations,
        warmup_iterations: config.warmup_iterations,
        logical_tokens_per_iteration,
        total_wall_time_ms: timing.total_wall_time_ms,
        mean_wall_time_ms: timing.mean_wall_time_ms,
        tokens_per_sec: timing.tokens_per_sec(logical_tokens_per_iteration),
        peak_rss_bytes: timing.peak_rss_bytes,
        peak_rss_delta_bytes: timing.peak_rss_delta_bytes,
        observability: observability.clone(),
    })
}

fn logical_tokens_per_iteration(surface: V2BenchmarkSurface, sequence_length: usize) -> usize {
    match surface {
        V2BenchmarkSurface::ForwardPass => sequence_length,
        _ => 1,
    }
}

#[derive(Debug, Clone, Copy)]
struct BenchmarkTiming {
    total_wall_time_ms: f64,
    mean_wall_time_ms: f64,
    peak_rss_bytes: u64,
    peak_rss_delta_bytes: u64,
}

impl BenchmarkTiming {
    fn tokens_per_sec(self, logical_tokens_per_iteration: usize) -> f64 {
        let total_seconds = self.total_wall_time_ms / 1000.0;
        if total_seconds <= f64::EPSILON {
            0.0
        } else {
            logical_tokens_per_iteration as f64 * (1000.0 / self.mean_wall_time_ms.max(1.0e-9))
        }
    }
}

fn benchmark_prepared<Prep, Run, Prepared, Output>(
    iterations: usize,
    warmup_iterations: usize,
    mut prepare: Prep,
    mut run: Run,
) -> Result<BenchmarkTiming, FractalError>
where
    Prep: FnMut() -> Result<Prepared, FractalError>,
    Run: FnMut(Prepared) -> Result<Output, FractalError>,
{
    for _ in 0..warmup_iterations {
        black_box(run(prepare()?)?);
    }

    let baseline_rss = process_peak_rss_bytes();
    let mut total_ms = 0.0f64;
    let mut peak_rss_bytes = baseline_rss;
    for _ in 0..iterations {
        let prepared = prepare()?;
        let start = Instant::now();
        black_box(run(prepared)?);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        total_ms += elapsed_ms;
        peak_rss_bytes = peak_rss_bytes.max(process_peak_rss_bytes());
    }

    Ok(BenchmarkTiming {
        total_wall_time_ms: total_ms,
        mean_wall_time_ms: total_ms / iterations as f64,
        peak_rss_bytes,
        peak_rss_delta_bytes: peak_rss_bytes.saturating_sub(baseline_rss),
    })
}

fn benchmark_forward_pass<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || benchmark_input_ids(sequence_length, model.shape().vocab_size, device),
        |input_ids| -> Result<FractalV2ForwardOutput<B>, FractalError> { model.forward(input_ids) },
    )
}

fn benchmark_token_append<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || prepare_token_append_fixture(model, sequence_length, device),
        |fixture| {
            let mut state = fixture.state;
            state.append_root_readouts_with_active_root_count(
                fixture.next_root_readouts,
                model.shape().local_trunk.root_count,
                model.leaf_summarizer(),
                model.tree_merge_cell(),
            )
        },
    )
}

fn benchmark_leaf_sealing<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || prepare_leaf_sealing_fixture(model, sequence_length, device),
        |fixture| {
            let mut state = fixture.state;
            let sealed = state.append_root_readouts_with_active_root_count(
                fixture.next_root_readouts,
                model.shape().local_trunk.root_count,
                model.leaf_summarizer(),
                model.tree_merge_cell(),
            )?;
            sealed.ok_or_else(|| {
                FractalError::InvalidState(
                    "leaf sealing benchmark expected a sealed leaf materialization".to_string(),
                )
            })
        },
    )
}

fn benchmark_tree_update<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || prepare_tree_update_fixture(model, sequence_length, device),
        |fixture| {
            let mut tree = fixture.tree;
            tree.append_sealed_leaf(fixture.node, fixture.shared_span, model.tree_merge_cell())?;
            Ok(tree)
        },
    )
}

fn benchmark_routing<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || prepare_route_fixture(model, sequence_length, device),
        |fixture| {
            model
                .router()
                .route(fixture.query, fixture.query_position, &fixture.tree)
        },
    )
}

fn benchmark_exact_leaf_read<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    iterations: usize,
    warmup_iterations: usize,
    device: &B::Device,
) -> Result<BenchmarkTiming, FractalError> {
    benchmark_prepared(
        iterations,
        warmup_iterations,
        || prepare_exact_read_fixture(model, sequence_length, device),
        |fixture| {
            model.exact_read().read(
                fixture.query,
                fixture.query_position,
                &fixture.routed,
                &fixture.leaf_token_cache,
            )
        },
    )
}

#[derive(Debug, Clone)]
struct TokenAppendFixture<B: Backend> {
    state: fractal_core::FractalV2State<B>,
    next_root_readouts: Tensor<B, 3>,
}

#[derive(Debug, Clone)]
struct TreeUpdateFixture<B: Backend> {
    tree: fractal_core::TreeSummaryState<B>,
    node: TreeNodeBatch<B>,
    shared_span: fractal_core::TokenSpan,
}

#[derive(Debug, Clone)]
struct RouteFixture<B: Backend> {
    tree: fractal_core::TreeSummaryState<B>,
    query: Tensor<B, 2>,
    query_position: usize,
}

#[derive(Debug, Clone)]
struct ExactReadFixture<B: Backend> {
    leaf_token_cache: fractal_core::LeafTokenCache<B>,
    query: Tensor<B, 2>,
    query_position: usize,
    routed: fractal_core::FractalRouteOutput<B>,
}

fn observe_sequence<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<V2ObservabilitySnapshot, FractalError> {
    let input_ids = benchmark_input_ids(sequence_length, model.shape().vocab_size, device)?;
    let trace = model.forward_retrieval_trace(input_ids)?;
    let root_diagnostics = summarize_root_readout_sequence(trace_root_readouts(&trace)?)?;
    let mut sealed_leaf_count = 0usize;
    let mut routing_sparsity_sum = 0.0f32;
    let mut routing_sparsity_count = 0usize;
    let mut exact_read_usage_sum = 0.0f32;
    let mut retrieval_distance_total = 0usize;
    let mut retrieval_distance_count = 0usize;
    let mut head_agreement_sum = 0.0f32;
    let mut selected_leaf_usage = BTreeMap::<usize, usize>::new();

    for step in trace.steps() {
        if step.sealed_leaf().is_some() {
            sealed_leaf_count += 1;
        }
        exact_read_usage_sum += step.exact_read().diagnostics().fraction_using_exact_read;
        let route_diag = step.routed().diagnostics();
        head_agreement_sum += route_diag.head_agreement_rate;
        let selected_leaf_count = selected_leaf_count(step.routed())?;
        accumulate_selected_leaf_usage(step.routed(), &mut selected_leaf_usage)?;
        if sealed_leaf_count > 0 {
            let density = selected_leaf_count as f32 / sealed_leaf_count as f32;
            routing_sparsity_sum += 1.0 - density.min(1.0);
            routing_sparsity_count += 1;
        }
        let (distance_total, distance_count) =
            histogram_weighted_total(&route_diag.selected_span_distance_histogram);
        retrieval_distance_total += distance_total;
        retrieval_distance_count += distance_count;
    }

    let tree_diag = trace.final_state().tree().diagnostics();
    let step_count = trace.steps().len().max(1) as f32;

    Ok(V2ObservabilitySnapshot {
        routing_sparsity: if routing_sparsity_count == 0 {
            1.0
        } else {
            routing_sparsity_sum / routing_sparsity_count as f32
        },
        root_collapse_mean_pairwise_cosine_similarity: root_diagnostics
            .mean_pairwise_cosine_similarity,
        exact_read_usage: exact_read_usage_sum / step_count,
        mean_retrieval_distance: if retrieval_distance_count == 0 {
            0.0
        } else {
            retrieval_distance_total as f32 / retrieval_distance_count as f32
        },
        tree_depth_reached: tree_diag.tree_depth_reached,
        level0_leaf_count: trace.final_state().sealed_leaves().shared_spans().len(),
        head_agreement_rate: head_agreement_sum / step_count,
        has_dead_or_unused_tree_nodes: tree_diag.has_dead_or_unused_nodes,
        selected_leaf_usage: selected_leaf_usage
            .into_iter()
            .map(|(leaf_index, count)| V2LeafUsageBin { leaf_index, count })
            .collect(),
    })
}

fn trace_root_readouts<B: Backend>(
    trace: &FractalV2RetrievalTrace<B>,
) -> Result<Tensor<B, 4>, FractalError> {
    let first_step = trace.steps().first().ok_or_else(|| {
        FractalError::InvalidState("benchmark trace must contain at least one step".to_string())
    })?;
    let [batch_size, root_count, readout_dim] = first_step.root_readouts().dims();
    let stacked = trace
        .steps()
        .iter()
        .map(|step| {
            step.root_readouts()
                .reshape([batch_size, 1, root_count, readout_dim])
        })
        .collect::<Vec<_>>();

    Ok(Tensor::cat(stacked, 1))
}

fn prepare_token_append_fixture<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<TokenAppendFixture<B>, FractalError> {
    let input_ids = benchmark_input_ids(sequence_length, model.shape().vocab_size, device)?;
    let trace = model.forward_retrieval_trace(input_ids)?;
    let state = trace.final_state().clone();
    let next_root_readouts =
        next_token_root_readouts(model, state.roots().clone(), sequence_length, device)?;

    Ok(TokenAppendFixture {
        state,
        next_root_readouts,
    })
}

fn prepare_leaf_sealing_fixture<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<TokenAppendFixture<B>, FractalError> {
    if sequence_length < model.shape().local_trunk.leaf_size {
        return Err(FractalError::InvalidConfig(format!(
            "leaf sealing benchmark requires sequence_length >= {}, got {}",
            model.shape().local_trunk.leaf_size,
            sequence_length
        )));
    }
    let prefill_length = sequence_length - 1;
    let input_ids = benchmark_input_ids(prefill_length, model.shape().vocab_size, device)?;
    let trace = model.forward_retrieval_trace(input_ids)?;
    let state = trace.final_state().clone();
    let next_root_readouts =
        next_token_root_readouts(model, state.roots().clone(), prefill_length, device)?;

    Ok(TokenAppendFixture {
        state,
        next_root_readouts,
    })
}

fn prepare_tree_update_fixture<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<TreeUpdateFixture<B>, FractalError> {
    let sealing_fixture = prepare_leaf_sealing_fixture(model, sequence_length, device)?;
    let tree = sealing_fixture.state.tree().clone();
    let mut materialize_state = sealing_fixture.state;
    let sealed_leaf = materialize_state
        .append_root_readouts_with_active_root_count(
            sealing_fixture.next_root_readouts,
            model.shape().local_trunk.root_count,
            model.leaf_summarizer(),
            model.tree_merge_cell(),
        )?
        .ok_or_else(|| {
            FractalError::InvalidState(
                "tree update benchmark expected a sealed leaf materialization".to_string(),
            )
        })?;

    Ok(TreeUpdateFixture {
        tree,
        node: TreeNodeBatch::from_tensors(
            sealed_leaf.summary(),
            sealed_leaf.key(),
            sealed_leaf.value(),
        )?,
        shared_span: sealed_leaf.shared_span(),
    })
}

fn prepare_route_fixture<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<RouteFixture<B>, FractalError> {
    let input_ids = benchmark_input_ids(sequence_length, model.shape().vocab_size, device)?;
    let trace = model.forward_retrieval_trace(input_ids)?;
    let final_step = trace.steps().last().ok_or_else(|| {
        FractalError::InvalidState(
            "routing benchmark trace must contain at least one step".to_string(),
        )
    })?;

    Ok(RouteFixture {
        tree: trace.final_state().tree().clone(),
        query: model.routing_query_from_root_readouts(
            final_step.root_readouts(),
            model.shape().local_trunk.root_count,
        )?,
        query_position: sequence_length,
    })
}

fn prepare_exact_read_fixture<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<ExactReadFixture<B>, FractalError> {
    let route_fixture = prepare_route_fixture(model, sequence_length, device)?;
    let leaf_token_cache = prefilled_leaf_token_cache(model, sequence_length, device)?;
    let routed = model.router().route(
        route_fixture.query.clone(),
        route_fixture.query_position,
        &route_fixture.tree,
    )?;

    Ok(ExactReadFixture {
        leaf_token_cache,
        query: route_fixture.query,
        query_position: route_fixture.query_position,
        routed,
    })
}

fn prefilled_leaf_token_cache<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    sequence_length: usize,
    device: &B::Device,
) -> Result<fractal_core::LeafTokenCache<B>, FractalError> {
    let input_ids = benchmark_input_ids(sequence_length, model.shape().vocab_size, device)?;
    let trace = model.forward_retrieval_trace(input_ids)?;
    Ok(trace.final_state().leaf_token_cache().clone())
}

fn next_token_root_readouts<B: Backend>(
    model: &BaselineV2SyntheticModel<B>,
    roots: fractal_core::MultiRootState<B>,
    token_index: usize,
    device: &B::Device,
) -> Result<Tensor<B, 3>, FractalError> {
    let next_token_id = benchmark_token_id(token_index, model.shape().vocab_size);
    let input_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(vec![next_token_id], [1, 1]), device);
    let token_embedding = model
        .embedding()
        .forward(input_ids)
        .reshape([1, model.shape().token_dim]);
    let step = model.local_trunk().step(token_embedding, roots)?;
    Ok(step.root_readouts())
}

fn benchmark_input_ids<B: Backend>(
    sequence_length: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2, Int>, FractalError> {
    if vocab_size < 2 {
        return Err(FractalError::InvalidConfig(format!(
            "benchmark_input_ids requires vocab_size >= 2, got {}",
            vocab_size
        )));
    }
    let input_ids = (0..sequence_length)
        .map(|index| benchmark_token_id(index, vocab_size))
        .collect::<Vec<_>>();
    Ok(Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_ids, [1, sequence_length]),
        device,
    ))
}

fn benchmark_token_id(index: usize, vocab_size: usize) -> i64 {
    (((index * 13) % (vocab_size - 1)) + 1) as i64
}

fn selected_leaf_count<B: Backend>(
    routed: &fractal_core::FractalRouteOutput<B>,
) -> Result<usize, FractalError> {
    let indices = routed
        .selected_leaf_indices()
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(invalid_state_from_data("benchmark.selected_leaf_indices"))?;
    let mask = routed
        .selected_leaf_mask()
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(invalid_state_from_data("benchmark.selected_leaf_mask"))?;
    let mut unique = BTreeSet::new();
    for (index, selected) in indices.into_iter().zip(mask.into_iter()) {
        if selected && index >= 0 {
            unique.insert(index);
        }
    }

    Ok(unique.len())
}

fn histogram_weighted_total(histogram: &[RoutingHistogramBin]) -> (usize, usize) {
    histogram
        .iter()
        .fold((0usize, 0usize), |(total, count), bin| {
            (total + bin.value * bin.count, count + bin.count)
        })
}

fn accumulate_selected_leaf_usage<B: Backend>(
    routed: &fractal_core::FractalRouteOutput<B>,
    selected_leaf_usage: &mut BTreeMap<usize, usize>,
) -> Result<(), FractalError> {
    let indices = routed
        .selected_leaf_indices()
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(invalid_state_from_data("benchmark.selected_leaf_indices"))?;
    let mask = routed
        .selected_leaf_mask()
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(invalid_state_from_data("benchmark.selected_leaf_mask"))?;

    for (index, selected) in indices.into_iter().zip(mask.into_iter()) {
        if selected && index >= 0 {
            *selected_leaf_usage.entry(index as usize).or_default() += 1;
        }
    }

    Ok(())
}

fn invalid_state_from_data(
    label: &'static str,
) -> impl FnOnce(burn::tensor::DataError) -> FractalError {
    move |error| FractalError::InvalidState(format!("{label} data conversion failed: {error}"))
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

#[cfg(test)]
mod tests {
    use burn::backend::Candle;

    use super::*;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn benchmark_config_rejects_zero_iterations() {
        let error = V2BenchmarkConfig {
            sequence_lengths: vec![32],
            leaf_size: 16,
            iterations: 0,
            warmup_iterations: 0,
        }
        .validate()
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("iterations")
        ));
    }

    #[test]
    fn benchmark_config_rejects_leaf_size_larger_than_sequence_length() {
        let error = V2BenchmarkConfig {
            sequence_lengths: vec![32],
            leaf_size: 64,
            iterations: 1,
            warmup_iterations: 0,
        }
        .validate()
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("leaf_size 64")
        ));
    }

    #[test]
    fn baseline_v2_benchmark_suite_runs_on_small_length() {
        let device = Default::default();
        let report = run_baseline_v2_benchmark_suite::<TestBackend>(
            V2BenchmarkConfig {
                sequence_lengths: vec![32],
                leaf_size: 16,
                iterations: 1,
                warmup_iterations: 0,
            },
            &device,
        )
        .unwrap();

        assert_eq!(report.entries.len(), V2BenchmarkSurface::ALL.len());
        assert!(report
            .entries
            .iter()
            .all(|entry| entry.sequence_length == 32));
        assert_eq!(report.config.leaf_size, 16);
        assert!(report
            .entries
            .iter()
            .all(|entry| entry.observability.level0_leaf_count > 0));
        assert!(report
            .entries
            .iter()
            .all(|entry| entry.mean_wall_time_ms.is_finite()));
    }

    #[test]
    fn baseline_v2_benchmark_suite_supports_exploratory_leaf_sizes() {
        let device = Default::default();
        let report = run_baseline_v2_benchmark_suite::<TestBackend>(
            V2BenchmarkConfig {
                sequence_lengths: vec![64],
                leaf_size: 32,
                iterations: 1,
                warmup_iterations: 0,
            },
            &device,
        )
        .unwrap();

        assert_eq!(report.config.leaf_size, 32);
        assert!(report
            .entries
            .iter()
            .all(|entry| entry.sequence_length == 64));
    }
}
