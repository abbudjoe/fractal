use std::collections::BTreeMap;

use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
};

use super::{
    auditor::{
        summarize_head_contexts, CausalMemoryAuditPlan, CausalMemoryAuditReport,
        CausalMemoryAuditSampleReport, CausalMemoryDeltaMetrics, CausalMemoryEvaluationContext,
        CausalMemoryHeadContext, CausalMemoryIntervention, CausalMemoryInterventionResult,
    },
    exact_read::{ExactLeafRead, ExactLeafReadShape},
    leaf::{LeafSummarizer, LeafSummarizerShape},
    local_trunk::{
        summarize_root_readout_sequence, LocalTrunk, LocalTrunkDiagnostics, LocalTrunkShape,
    },
    read_fusion::{ReadFusion, ReadFusionAblation, ReadFusionInput, ReadFusionShape},
    router::{FractalRouterHead, FractalRouterHeadShape},
    state::{FractalV2State, MergeCheckpointPolicy, MultiRootState, SealedLeafMaterialization},
    tree::{TreeMergeCell, TreeMergeCellShape},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalV2ModelShape {
    pub vocab_size: usize,
    pub token_dim: usize,
    pub local_trunk: LocalTrunkShape,
    pub leaf_summarizer: LeafSummarizerShape,
    pub tree_merge_cell: TreeMergeCellShape,
    pub router: FractalRouterHeadShape,
    pub exact_read: ExactLeafReadShape,
    pub read_fusion: ReadFusionShape,
}

impl FractalV2ModelShape {
    pub(crate) fn validate(self) -> Result<Self, FractalError> {
        validate_fractal_v2_model_shape(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FractalV2MemoryMode {
    NoMemory,
    SummariesOnly,
    TreeOnly,
    TreePlusExactRead,
}

impl FractalV2MemoryMode {
    fn include_leaf_memory(self) -> bool {
        !matches!(self, Self::NoMemory)
    }

    fn include_tree_routing(self) -> bool {
        matches!(self, Self::TreeOnly | Self::TreePlusExactRead)
    }

    fn include_exact_read(self) -> bool {
        matches!(self, Self::TreePlusExactRead)
    }

    fn read_fusion_ablation(self) -> ReadFusionAblation {
        match self {
            Self::NoMemory => ReadFusionAblation::without_memory_reads(),
            Self::SummariesOnly => ReadFusionAblation::without_memory_reads(),
            Self::TreeOnly => ReadFusionAblation::without_exact_read_values(),
            Self::TreePlusExactRead => ReadFusionAblation::full(),
        }
    }
}

#[derive(Debug)]
pub struct FractalV2Components<LT, LS, TM, RH, ER, RF> {
    pub local_trunk: LT,
    pub leaf_summarizer: LS,
    pub tree_merge_cell: TM,
    pub router: RH,
    pub exact_read: ER,
    pub read_fusion: RF,
}

#[derive(Debug, Clone)]
pub struct FractalV2LocalBaselineOutput<B: Backend> {
    root_readouts: Tensor<B, 4>,
    mean_readouts: Tensor<B, 3>,
    final_state: MultiRootState<B>,
    diagnostics: LocalTrunkDiagnostics,
}

#[derive(Debug, Clone)]
pub struct FractalV2RetrievalStepOutput<B: Backend> {
    root_readouts: Tensor<B, 3>,
    sealed_leaf: Option<SealedLeafMaterialization<B>>,
    routed: crate::v2::FractalRouteOutput<B>,
    exact_read: crate::v2::ExactLeafReadOutput<B>,
}

impl<B: Backend> FractalV2RetrievalStepOutput<B> {
    pub fn root_readouts(&self) -> Tensor<B, 3> {
        self.root_readouts.clone()
    }

    pub fn sealed_leaf(&self) -> Option<&SealedLeafMaterialization<B>> {
        self.sealed_leaf.as_ref()
    }

    pub fn routed(&self) -> &crate::v2::FractalRouteOutput<B> {
        &self.routed
    }

    pub fn exact_read(&self) -> &crate::v2::ExactLeafReadOutput<B> {
        &self.exact_read
    }
}

#[derive(Debug, Clone)]
pub struct FractalV2RetrievalTrace<B: Backend> {
    steps: Vec<FractalV2RetrievalStepOutput<B>>,
    final_state: FractalV2State<B>,
}

#[derive(Debug, Clone)]
pub struct FractalV2ForwardOutput<B: Backend> {
    fused_readouts: Tensor<B, 3>,
    logits: Tensor<B, 3>,
    final_state: FractalV2State<B>,
}

impl<B: Backend> FractalV2RetrievalTrace<B> {
    pub fn steps(&self) -> &[FractalV2RetrievalStepOutput<B>] {
        &self.steps
    }

    pub fn final_state(&self) -> &FractalV2State<B> {
        &self.final_state
    }
}

impl<B: Backend> FractalV2ForwardOutput<B> {
    pub fn fused_readouts(&self) -> Tensor<B, 3> {
        self.fused_readouts.clone()
    }

    pub fn logits(&self) -> Tensor<B, 3> {
        self.logits.clone()
    }

    pub fn final_state(&self) -> &FractalV2State<B> {
        &self.final_state
    }
}

impl<B: Backend> FractalV2LocalBaselineOutput<B> {
    pub fn root_readouts(&self) -> Tensor<B, 4> {
        self.root_readouts.clone()
    }

    pub fn mean_readouts(&self) -> Tensor<B, 3> {
        self.mean_readouts.clone()
    }

    pub fn final_state(&self) -> &MultiRootState<B> {
        &self.final_state
    }

    pub fn diagnostics(&self) -> &LocalTrunkDiagnostics {
        &self.diagnostics
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalV2LocalBaselineShape {
    pub vocab_size: usize,
    pub token_dim: usize,
    pub local_trunk: LocalTrunkShape,
}

#[derive(Module, Debug)]
pub struct FractalV2LocalBaselineModel<B: Backend, LT: Module<B>> {
    embedding: Embedding<B>,
    local_trunk: LT,
    vocab_size: usize,
    token_dim: usize,
    root_count: usize,
    root_state_dim: usize,
    root_readout_dim: usize,
    leaf_size: usize,
}

#[derive(Module, Debug)]
pub struct FractalV2Model<
    B: Backend,
    LT: Module<B>,
    LS: Module<B>,
    TM: Module<B>,
    RH: Module<B>,
    ER: Module<B>,
    RF: Module<B>,
> {
    embedding: Embedding<B>,
    local_trunk: LT,
    leaf_summarizer: LS,
    tree_merge_cell: TM,
    router: RH,
    exact_read: ER,
    read_fusion: RF,
    output: LanguageModelHead<B>,
    vocab_size: usize,
    token_dim: usize,
    root_count: usize,
    root_state_dim: usize,
    root_readout_dim: usize,
    leaf_size: usize,
    summary_dim: usize,
    key_dim: usize,
    value_dim: usize,
    token_cache_key_dim: usize,
    token_cache_value_dim: usize,
    scale_embedding_dim: usize,
    routing_head_count: usize,
    beam_width: usize,
    top_leaf_reads: usize,
    allow_early_stop: bool,
    exact_read_query_dim: usize,
    exact_read_key_dim: usize,
    exact_read_value_dim: usize,
    exact_read_head_count: usize,
    exact_read_leaf_size: usize,
    fused_readout_dim: usize,
}

impl<B, LT, LS, TM, RH, ER, RF> FractalV2Model<B, LT, LS, TM, RH, ER, RF>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    ER: ExactLeafRead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
{
    pub fn new(
        vocab_size: usize,
        token_dim: usize,
        components: FractalV2Components<LT, LS, TM, RH, ER, RF>,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let FractalV2Components {
            local_trunk,
            leaf_summarizer,
            tree_merge_cell,
            router,
            exact_read,
            read_fusion,
        } = components;
        let shape = FractalV2ModelShape {
            vocab_size,
            token_dim,
            local_trunk: local_trunk.shape(),
            leaf_summarizer: leaf_summarizer.shape(),
            tree_merge_cell: tree_merge_cell.shape(),
            router: router.shape(),
            exact_read: exact_read.shape(),
            read_fusion: read_fusion.shape(),
        }
        .validate()?;
        let embedding = EmbeddingConfig::new(vocab_size, token_dim).init(device);
        let output = LanguageModelHeadConfig::new(shape.read_fusion.fused_readout_dim, vocab_size)
            .init(device);

        Ok(Self {
            embedding,
            local_trunk,
            leaf_summarizer,
            tree_merge_cell,
            router,
            exact_read,
            read_fusion,
            output,
            vocab_size: shape.vocab_size,
            token_dim: shape.token_dim,
            root_count: shape.local_trunk.root_count,
            root_state_dim: shape.local_trunk.root_state_dim,
            root_readout_dim: shape.local_trunk.root_readout_dim,
            leaf_size: shape.local_trunk.leaf_size,
            summary_dim: shape.leaf_summarizer.summary_dim,
            key_dim: shape.leaf_summarizer.key_dim,
            value_dim: shape.leaf_summarizer.value_dim,
            token_cache_key_dim: shape.leaf_summarizer.token_cache_key_dim,
            token_cache_value_dim: shape.leaf_summarizer.token_cache_value_dim,
            scale_embedding_dim: shape.tree_merge_cell.scale_embedding_dim,
            routing_head_count: shape.router.head_count,
            beam_width: shape.router.beam_width,
            top_leaf_reads: shape.router.top_leaf_reads,
            allow_early_stop: shape.router.allow_early_stop,
            exact_read_query_dim: shape.exact_read.query_dim,
            exact_read_key_dim: shape.exact_read.key_dim,
            exact_read_value_dim: shape.exact_read.value_dim,
            exact_read_head_count: shape.exact_read.head_count,
            exact_read_leaf_size: shape.exact_read.leaf_size,
            fused_readout_dim: shape.read_fusion.fused_readout_dim,
        })
    }

    pub fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub fn local_trunk(&self) -> &LT {
        &self.local_trunk
    }

    pub fn leaf_summarizer(&self) -> &LS {
        &self.leaf_summarizer
    }

    pub fn tree_merge_cell(&self) -> &TM {
        &self.tree_merge_cell
    }

    pub fn router(&self) -> &RH {
        &self.router
    }

    pub fn exact_read(&self) -> &ER {
        &self.exact_read
    }

    pub fn read_fusion(&self) -> &RF {
        &self.read_fusion
    }

    pub fn output(&self) -> &LanguageModelHead<B> {
        &self.output
    }

    pub fn routing_query_from_root_readouts(
        &self,
        root_readouts: Tensor<B, 3>,
        active_root_count: usize,
    ) -> Result<Tensor<B, 2>, FractalError> {
        summarize_query_from_roots(
            root_readouts,
            active_root_count,
            self.root_count,
            self.root_readout_dim,
        )
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        self.forward_with_memory_mode(input_ids, FractalV2MemoryMode::TreePlusExactRead)
    }

    pub fn forward_with_memory_mode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        memory_mode: FractalV2MemoryMode,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        let trace = self.forward_retrieval_trace_internal(input_ids, None, memory_mode)?;
        self.forward_from_retrieval_trace(trace, memory_mode.read_fusion_ablation())
    }

    pub fn forward_with_ablation(
        &self,
        input_ids: Tensor<B, 2, Int>,
        ablation: ReadFusionAblation,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        let trace = self.forward_retrieval_trace_internal(
            input_ids,
            ablation.active_root_count(),
            FractalV2MemoryMode::TreePlusExactRead,
        )?;
        self.forward_from_retrieval_trace(trace, ablation)
    }

    pub fn forward_retrieval_trace(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<FractalV2RetrievalTrace<B>, FractalError> {
        self.forward_retrieval_trace_with_memory_mode(
            input_ids,
            FractalV2MemoryMode::TreePlusExactRead,
        )
    }

    pub fn forward_retrieval_trace_with_memory_mode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        memory_mode: FractalV2MemoryMode,
    ) -> Result<FractalV2RetrievalTrace<B>, FractalError> {
        self.forward_retrieval_trace_internal(input_ids, None, memory_mode)
    }

    pub fn forward_retrieval_trace_with_ablation(
        &self,
        input_ids: Tensor<B, 2, Int>,
        ablation: ReadFusionAblation,
    ) -> Result<FractalV2RetrievalTrace<B>, FractalError> {
        self.forward_retrieval_trace_internal(
            input_ids,
            ablation.active_root_count(),
            FractalV2MemoryMode::TreePlusExactRead,
        )
    }

    fn forward_retrieval_trace_internal(
        &self,
        input_ids: Tensor<B, 2, Int>,
        active_root_count: Option<usize>,
        memory_mode: FractalV2MemoryMode,
    ) -> Result<FractalV2RetrievalTrace<B>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        ensure_nonzero("fractal_v2_retrieval_trace.batch_size", batch_size)?;
        ensure_nonzero("fractal_v2_retrieval_trace.seq_len", seq_len)?;

        let shape = self.shape();
        let device = input_ids.device();
        let embeddings = self.embedding.forward(input_ids);
        let mut roots =
            MultiRootState::zeros_for_local_trunk(batch_size, shape.local_trunk, &device)?;
        let mut state = FractalV2State::for_model_shape(
            shape,
            batch_size,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: self.leaf_size,
            },
            &device,
        )?;
        let mut steps = Vec::with_capacity(seq_len);

        for position in 0..seq_len {
            let token_embedding = embeddings
                .clone()
                .narrow(1, position, 1)
                .reshape([batch_size, self.token_dim]);
            let local_step = self.local_trunk.step(token_embedding, roots)?;
            let root_readouts = local_step.root_readouts();
            roots = local_step.into_next_state();
            state.update_roots(roots.clone())?;
            let effective_root_count = active_root_count.unwrap_or(self.root_count);
            let sealed_leaf = if memory_mode.include_leaf_memory() {
                state.append_root_readouts_with_active_root_count(
                    root_readouts.clone(),
                    effective_root_count,
                    &self.leaf_summarizer,
                    &self.tree_merge_cell,
                )?
            } else {
                None
            };
            let query = summarize_query_from_roots(
                root_readouts.clone(),
                effective_root_count,
                self.root_count,
                self.root_readout_dim,
            )?;
            let query_position = position + 1;
            let routed = if memory_mode.include_tree_routing() {
                self.router
                    .route(query.clone(), query_position, state.tree())?
            } else {
                empty_route_output(
                    batch_size,
                    self.routing_head_count,
                    self.top_leaf_reads,
                    self.value_dim,
                    &device,
                )?
            };
            let exact_read = if memory_mode.include_exact_read() {
                self.exact_read
                    .read(query, query_position, &routed, state.leaf_token_cache())?
            } else {
                empty_exact_read_output(
                    batch_size,
                    self.exact_read_head_count,
                    self.top_leaf_reads,
                    self.exact_read_leaf_size,
                    self.exact_read_value_dim,
                    &device,
                )?
            };
            steps.push(FractalV2RetrievalStepOutput {
                root_readouts,
                sealed_leaf,
                routed,
                exact_read,
            });
        }

        Ok(FractalV2RetrievalTrace {
            steps,
            final_state: state,
        })
    }

    pub fn audit_causal_memory(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
        plan: &CausalMemoryAuditPlan,
    ) -> Result<CausalMemoryAuditReport, FractalError> {
        let plan = plan.clone().validate()?;
        let [batch_size, seq_len] = input_ids.dims();
        ensure_nonzero("fractal_v2_audit.batch_size", batch_size)?;
        ensure_nonzero("fractal_v2_audit.seq_len", seq_len)?;
        ensure_match(
            "fractal_v2_audit.target_batch_size",
            target_ids.dims()[0],
            batch_size,
        )?;
        ensure_match(
            "fractal_v2_audit.target_seq_len",
            target_ids.dims()[1],
            seq_len,
        )?;

        let mut samples_by_position = BTreeMap::<usize, Vec<_>>::new();
        for sample in plan.samples() {
            if sample.batch_index >= batch_size {
                return Err(FractalError::InvalidConfig(format!(
                    "causal_memory_audit sample batch {} is out of bounds for batch size {}",
                    sample.batch_index, batch_size
                )));
            }
            if sample.position >= seq_len {
                return Err(FractalError::InvalidConfig(format!(
                    "causal_memory_audit sample position {} is out of bounds for sequence length {}",
                    sample.position, seq_len
                )));
            }
            samples_by_position
                .entry(sample.position)
                .or_default()
                .push(sample.clone());
        }

        let shape = self.shape();
        let device = input_ids.device();
        let embeddings = self.embedding.forward(input_ids);
        let target_data = target_ids
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(invalid_state_from_data("fractal_v2_audit.target_ids"))?;
        let mut roots =
            MultiRootState::zeros_for_local_trunk(batch_size, shape.local_trunk, &device)?;
        let mut state = FractalV2State::for_model_shape(
            shape,
            batch_size,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: self.leaf_size,
            },
            &device,
        )?;
        let mut sample_reports = Vec::new();

        for position in 0..seq_len {
            let token_embedding = embeddings
                .clone()
                .narrow(1, position, 1)
                .reshape([batch_size, self.token_dim]);
            let local_step = self.local_trunk.step(token_embedding, roots)?;
            let root_readouts = local_step.root_readouts();
            roots = local_step.into_next_state();
            state.update_roots(roots.clone())?;
            let _sealed_leaf = state.append_root_readouts_with_active_root_count(
                root_readouts.clone(),
                self.root_count,
                &self.leaf_summarizer,
                &self.tree_merge_cell,
            )?;
            let query = summarize_query_from_roots(
                root_readouts.clone(),
                self.root_count,
                self.root_count,
                self.root_readout_dim,
            )?;
            let query_position = position + 1;
            let routed = self
                .router
                .route(query.clone(), query_position, state.tree())?;
            let exact_read = self.exact_read.read(
                query.clone(),
                query_position,
                &routed,
                state.leaf_token_cache(),
            )?;
            let reference_logits = self.project_step_logits(
                root_readouts.clone(),
                routed.selected_leaf_values(),
                routed.selected_leaf_scores(),
                routed.selected_leaf_mask(),
                exact_read.read_values(),
                exact_read.selected_token_mask(),
            )?;

            let Some(samples) = samples_by_position.get(&position) else {
                continue;
            };

            for sample in samples {
                let target_token_id = target_token_at(
                    &target_data,
                    batch_size,
                    seq_len,
                    sample.batch_index,
                    sample.position,
                )?;
                let reference_logits_row = logits_for_batch(
                    reference_logits.clone(),
                    sample.batch_index,
                    self.vocab_size,
                )?;
                let reference_loss =
                    negative_log_likelihood(&reference_logits_row, target_token_id)?;
                let reference_target_logit =
                    target_logit(&reference_logits_row, target_token_id, self.vocab_size)?;
                let reference_head_contexts = head_contexts_for_route(
                    &routed,
                    sample.batch_index,
                    query_position,
                    state
                        .tree()
                        .level(0)
                        .map(|level| level.shared_spans())
                        .unwrap_or(&[]),
                )?;
                let reference_context = summarize_head_contexts(&reference_head_contexts);
                let reference_metrics = AuditMetricReference {
                    task_family: &sample.task_family,
                    context: reference_context,
                    head_contexts: &reference_head_contexts,
                    reference_loss,
                    reference_target_logit,
                    reference_logits: &reference_logits_row,
                    batch_index: sample.batch_index,
                    target_token_id,
                    vocab_size: self.vocab_size,
                };
                let mut interventions = Vec::new();

                if plan.include_no_tree_read() {
                    let logits = self.project_step_logits(
                        root_readouts.clone(),
                        Tensor::<B, 4>::zeros(routed.selected_leaf_values().dims(), &device),
                        routed.selected_leaf_scores(),
                        routed.selected_leaf_mask(),
                        exact_read.read_values(),
                        exact_read.selected_token_mask(),
                    )?;
                    interventions.push(intervention_result(
                        CausalMemoryIntervention::NoTreeRead,
                        reference_metrics,
                        logits,
                    )?);
                }

                if plan.include_no_exact_leaf_read() {
                    let logits = self.project_step_logits(
                        root_readouts.clone(),
                        routed.selected_leaf_values(),
                        routed.selected_leaf_scores(),
                        routed.selected_leaf_mask(),
                        Tensor::<B, 4>::zeros(exact_read.read_values().dims(), &device),
                        exact_read.selected_token_mask(),
                    )?;
                    interventions.push(intervention_result(
                        CausalMemoryIntervention::NoExactLeafRead,
                        reference_metrics,
                        logits,
                    )?);
                }

                if plan.include_next_best_span_substitution() {
                    match next_best_route_for_batch(
                        &routed,
                        state.tree(),
                        sample.batch_index,
                        query_position,
                    )? {
                        Some(next_best) => {
                            let exact = self.exact_read.read(
                                query.clone(),
                                query_position,
                                &next_best,
                                state.leaf_token_cache(),
                            )?;
                            let next_best_head_contexts = head_contexts_for_route(
                                &next_best,
                                sample.batch_index,
                                query_position,
                                state
                                    .tree()
                                    .level(0)
                                    .map(|level| level.shared_spans())
                                    .unwrap_or(&[]),
                            )?;
                            let logits = self.project_step_logits(
                                root_readouts.clone(),
                                next_best.selected_leaf_values(),
                                next_best.selected_leaf_scores(),
                                next_best.selected_leaf_mask(),
                                exact.read_values(),
                                exact.selected_token_mask(),
                            )?;
                            interventions.push(intervention_result(
                                CausalMemoryIntervention::NextBestSpanSubstitution,
                                AuditMetricReference {
                                    context: summarize_head_contexts(&next_best_head_contexts),
                                    head_contexts: &next_best_head_contexts,
                                    ..reference_metrics
                                },
                                logits,
                            )?);
                        }
                        None => interventions.push(CausalMemoryInterventionResult {
                            intervention: CausalMemoryIntervention::NextBestSpanSubstitution,
                            applied: false,
                            context: None,
                            head_contexts: Vec::new(),
                            metrics: None,
                        }),
                    }
                }

                if plan.include_root_drop() {
                    for root_index in 0..self.root_count {
                        let logits = self.project_step_logits(
                            zero_root_readout_for_batch(
                                root_readouts.clone(),
                                sample.batch_index,
                                root_index,
                            )?,
                            routed.selected_leaf_values(),
                            routed.selected_leaf_scores(),
                            routed.selected_leaf_mask(),
                            exact_read.read_values(),
                            exact_read.selected_token_mask(),
                        )?;
                        interventions.push(intervention_result(
                            CausalMemoryIntervention::RootDrop { root_index },
                            reference_metrics,
                            logits,
                        )?);
                    }
                }

                sample_reports.push(CausalMemoryAuditSampleReport {
                    sample: sample.clone(),
                    reference_context,
                    reference_head_contexts,
                    target_token_id,
                    reference_loss,
                    reference_target_logit,
                    interventions,
                });
            }
        }

        Ok(CausalMemoryAuditReport::from_sample_reports(sample_reports))
    }

    fn forward_from_retrieval_trace(
        &self,
        trace: FractalV2RetrievalTrace<B>,
        ablation: ReadFusionAblation,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        let FractalV2RetrievalTrace { steps, final_state } = trace;
        let seq_len = steps.len();
        if seq_len == 0 {
            return Err(FractalError::InvalidConfig(
                "fractal_v2_forward_trace requires at least one retrieval step".to_string(),
            ));
        }
        let [batch_size, _, _] = steps[0].root_readouts().dims();
        let mut fused_steps = Vec::with_capacity(seq_len);

        for step in &steps {
            let fusion_input = ReadFusionInput::new(
                step.root_readouts(),
                step.routed().selected_leaf_values(),
                step.routed().selected_leaf_scores(),
                step.routed().selected_leaf_mask(),
                step.exact_read().read_values(),
                step.exact_read().selected_token_mask(),
            )?;
            let fusion = self.read_fusion.fuse(&fusion_input, ablation)?;
            fused_steps.push(fusion.fused_readout().reshape([
                batch_size,
                1,
                self.fused_readout_dim,
            ]));
        }

        let fused_readouts = Tensor::cat(fused_steps, 1);
        let logits = self.output.forward(fused_readouts.clone());

        Ok(FractalV2ForwardOutput {
            fused_readouts,
            logits,
            final_state,
        })
    }

    fn project_step_logits(
        &self,
        root_readouts: Tensor<B, 3>,
        routed_values: Tensor<B, 4>,
        routed_scores: Tensor<B, 3>,
        routed_mask: Tensor<B, 3, burn::tensor::Bool>,
        exact_read_values: Tensor<B, 4>,
        exact_read_mask: Tensor<B, 3, burn::tensor::Bool>,
    ) -> Result<Tensor<B, 2>, FractalError> {
        let [batch_size, _, _] = root_readouts.dims();
        let fusion_input = ReadFusionInput::new(
            root_readouts,
            routed_values,
            routed_scores,
            routed_mask,
            exact_read_values,
            exact_read_mask,
        )?;
        let fusion = self
            .read_fusion
            .fuse(&fusion_input, ReadFusionAblation::full())?;
        Ok(self
            .output
            .forward(
                fusion
                    .fused_readout()
                    .reshape([batch_size, 1, self.fused_readout_dim]),
            )
            .reshape([batch_size, self.vocab_size]))
    }

    pub fn shape(&self) -> FractalV2ModelShape {
        FractalV2ModelShape {
            vocab_size: self.vocab_size,
            token_dim: self.token_dim,
            local_trunk: LocalTrunkShape {
                token_dim: self.token_dim,
                root_count: self.root_count,
                root_state_dim: self.root_state_dim,
                root_readout_dim: self.root_readout_dim,
                leaf_size: self.leaf_size,
            },
            leaf_summarizer: LeafSummarizerShape {
                readout_dim: self.root_readout_dim,
                leaf_size: self.leaf_size,
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                token_cache_key_dim: self.token_cache_key_dim,
                token_cache_value_dim: self.token_cache_value_dim,
            },
            tree_merge_cell: TreeMergeCellShape {
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                scale_embedding_dim: self.scale_embedding_dim,
            },
            router: FractalRouterHeadShape {
                query_dim: self.root_readout_dim,
                key_dim: self.key_dim,
                head_count: self.routing_head_count,
                beam_width: self.beam_width,
                top_leaf_reads: self.top_leaf_reads,
                allow_early_stop: self.allow_early_stop,
            },
            exact_read: ExactLeafReadShape {
                query_dim: self.exact_read_query_dim,
                key_dim: self.exact_read_key_dim,
                value_dim: self.exact_read_value_dim,
                head_count: self.exact_read_head_count,
                top_leaf_reads: self.top_leaf_reads,
                leaf_size: self.exact_read_leaf_size,
            },
            read_fusion: ReadFusionShape {
                root_count: self.root_count,
                root_readout_dim: self.root_readout_dim,
                routed_value_dim: self.value_dim,
                exact_read_value_dim: self.token_cache_value_dim,
                fused_readout_dim: self.fused_readout_dim,
            },
        }
    }
}

fn summarize_query_from_roots<B: Backend>(
    root_readouts: Tensor<B, 3>,
    active_root_count: usize,
    total_root_count: usize,
    root_readout_dim: usize,
) -> Result<Tensor<B, 2>, FractalError> {
    if active_root_count == 0 || active_root_count > total_root_count {
        return Err(FractalError::InvalidConfig(format!(
            "fractal_v2_query.active_root_count must be within 1..={total_root_count}, got {active_root_count}"
        )));
    }

    let [batch_size, root_count, actual_root_readout_dim] = root_readouts.dims();
    ensure_match("fractal_v2_query.root_count", root_count, total_root_count)?;
    ensure_match(
        "fractal_v2_query.root_readout_dim",
        actual_root_readout_dim,
        root_readout_dim,
    )?;

    Ok(root_readouts
        .narrow(1, 0, active_root_count)
        .sum_dim(1)
        .mul_scalar(1.0 / active_root_count as f64)
        .reshape([batch_size, root_readout_dim]))
}

fn target_token_at(
    target_data: &[i64],
    batch_size: usize,
    seq_len: usize,
    batch_index: usize,
    position: usize,
) -> Result<i64, FractalError> {
    if batch_index >= batch_size || position >= seq_len {
        return Err(FractalError::InvalidState(format!(
            "target lookup [{batch_index}, {position}] is out of bounds for [{batch_size}, {seq_len}]"
        )));
    }
    Ok(target_data[batch_index * seq_len + position])
}

fn logits_for_batch<B: Backend>(
    logits: Tensor<B, 2>,
    batch_index: usize,
    vocab_size: usize,
) -> Result<Vec<f32>, FractalError> {
    let [batch_size, actual_vocab_size] = logits.dims();
    ensure_match("audit_logits.vocab_size", actual_vocab_size, vocab_size)?;
    if batch_index >= batch_size {
        return Err(FractalError::InvalidState(format!(
            "audit_logits batch {batch_index} is out of bounds for batch size {batch_size}"
        )));
    }
    logits
        .slice([batch_index..batch_index + 1, 0..vocab_size])
        .reshape([vocab_size])
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data("audit_logits"))
}

fn target_logit(
    logits: &[f32],
    target_token_id: i64,
    vocab_size: usize,
) -> Result<f32, FractalError> {
    let target_index = usize::try_from(target_token_id).map_err(|_| {
        FractalError::InvalidConfig(format!(
            "target token id {target_token_id} must be non-negative"
        ))
    })?;
    if target_index >= vocab_size {
        return Err(FractalError::InvalidConfig(format!(
            "target token id {target_token_id} is out of bounds for vocab size {vocab_size}"
        )));
    }
    Ok(logits[target_index])
}

fn negative_log_likelihood(logits: &[f32], target_token_id: i64) -> Result<f32, FractalError> {
    let target_index = usize::try_from(target_token_id).map_err(|_| {
        FractalError::InvalidConfig(format!(
            "target token id {target_token_id} must be non-negative"
        ))
    })?;
    if target_index >= logits.len() {
        return Err(FractalError::InvalidConfig(format!(
            "target token id {target_token_id} is out of bounds for {} logits",
            logits.len()
        )));
    }

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let log_sum_exp = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .sum::<f32>()
        .ln()
        + max_logit;

    Ok(log_sum_exp - logits[target_index])
}

fn kl_divergence(reference_logits: &[f32], perturbed_logits: &[f32]) -> Result<f32, FractalError> {
    if reference_logits.len() != perturbed_logits.len() {
        return Err(FractalError::InvalidState(format!(
            "cannot compare KL divergence for mismatched logit widths: {} vs {}",
            reference_logits.len(),
            perturbed_logits.len()
        )));
    }
    let reference_probs = softmax_vec(reference_logits);
    let perturbed_probs = softmax_vec(perturbed_logits);

    Ok(reference_probs
        .iter()
        .zip(perturbed_probs.iter())
        .map(|(reference, perturbed)| {
            let reference = (*reference).max(1.0e-12);
            let perturbed = (*perturbed).max(1.0e-12);
            reference * (reference.ln() - perturbed.ln())
        })
        .sum())
}

fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let exp_values = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect::<Vec<_>>();
    let denominator = exp_values.iter().sum::<f32>().max(1.0e-12);
    exp_values
        .into_iter()
        .map(|value| value / denominator)
        .collect()
}

fn head_contexts_for_route<B: Backend>(
    routed: &crate::v2::FractalRouteOutput<B>,
    batch_index: usize,
    query_position: usize,
    level_zero_spans: &[crate::v2::TokenSpan],
) -> Result<Vec<CausalMemoryHeadContext>, FractalError> {
    let mut head_contexts = Vec::with_capacity(routed.traces().len());
    let selected_leaf_indices = routed
        .selected_leaf_indices()
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "route output selected_leaf_indices data conversion failed: {error}"
            ))
        })?;
    let selected_leaf_mask = routed
        .selected_leaf_mask()
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "route output selected_leaf_mask data conversion failed: {error}"
            ))
        })?;
    let [batch_size, head_count, top_leaf_reads] = routed.selected_leaf_indices().dims();
    if batch_index >= batch_size {
        return Err(FractalError::InvalidState(format!(
            "route output batch {} is out of bounds for batch size {}",
            batch_index, batch_size
        )));
    }

    for (head_index, head_trace) in routed.traces().iter().enumerate() {
        let batch_route = head_trace.batch_routes.get(batch_index).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "route output batch {} is out of bounds for {} batch routes",
                batch_index,
                head_trace.batch_routes.len()
            ))
        })?;
        let mut runtime_selected_leaf_indices = Vec::new();
        let mut runtime_span_distances = Vec::new();
        for slot in 0..top_leaf_reads {
            let flat_index = ((batch_index * head_count + head_index) * top_leaf_reads) + slot;
            if !selected_leaf_mask[flat_index] {
                continue;
            }
            let leaf_index = usize::try_from(selected_leaf_indices[flat_index]).map_err(|_| {
                FractalError::InvalidState(format!(
                    "route output selected_leaf_indices[{batch_index}][{head_index}][{slot}] must be non-negative when selected"
                ))
            })?;
            let span = *level_zero_spans.get(leaf_index).ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "route output selected_leaf_indices[{batch_index}][{head_index}][{slot}]={} is out of bounds for {} level-0 spans",
                    leaf_index,
                    level_zero_spans.len()
                ))
            })?;
            runtime_selected_leaf_indices.push(leaf_index);
            runtime_span_distances.push(selected_span_distance(query_position, span)?);
        }
        head_contexts.push(CausalMemoryHeadContext {
            head_index,
            routing_depth: batch_route.steps.len(),
            span_distances: runtime_span_distances,
            selected_leaf_indices: runtime_selected_leaf_indices,
        });
    }

    Ok(head_contexts)
}

fn selected_span_distance(
    query_position: usize,
    span: crate::v2::TokenSpan,
) -> Result<usize, FractalError> {
    if span.end() > query_position {
        return Err(FractalError::InvalidState(format!(
            "selected span [{}, {}) ends after query position {}",
            span.start(),
            span.end(),
            query_position
        )));
    }

    Ok(query_position - span.end())
}

fn empty_route_output<B: Backend>(
    batch_size: usize,
    head_count: usize,
    top_leaf_reads: usize,
    value_dim: usize,
    device: &B::Device,
) -> Result<crate::v2::FractalRouteOutput<B>, FractalError> {
    crate::v2::FractalRouteOutput::from_parts(
        Tensor::<B, 3, Int>::zeros([batch_size, head_count, top_leaf_reads], device),
        Tensor::<B, 3, Bool>::zeros([batch_size, head_count, top_leaf_reads], device),
        Tensor::<B, 3>::zeros([batch_size, head_count, top_leaf_reads], device),
        Tensor::<B, 4>::zeros([batch_size, head_count, top_leaf_reads, value_dim], device),
        (0..head_count)
            .map(|_| crate::v2::HeadRouteTrace {
                batch_routes: (0..batch_size)
                    .map(|_| crate::v2::BatchHeadRoute {
                        steps: Vec::new(),
                        selected_leaf_indices: Vec::new(),
                        selected_leaf_spans: Vec::new(),
                        selected_leaf_scores: Vec::new(),
                    })
                    .collect(),
            })
            .collect(),
        crate::v2::FractalRoutingDiagnostics {
            routing_depth_histogram: Vec::new(),
            candidate_entropy_per_head: vec![0.0; head_count],
            selected_span_distance_histogram: Vec::new(),
            head_agreement_rate: 1.0,
            head_disagreement_rate: 0.0,
        },
    )
}

fn empty_exact_read_output<B: Backend>(
    batch_size: usize,
    head_count: usize,
    top_leaf_reads: usize,
    leaf_size: usize,
    value_dim: usize,
    device: &B::Device,
) -> Result<crate::v2::ExactLeafReadOutput<B>, FractalError> {
    crate::v2::ExactLeafReadOutput::new(
        Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                vec![-1i64; batch_size * head_count * top_leaf_reads],
                [batch_size, head_count, top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                vec![-1i64; batch_size * head_count * top_leaf_reads],
                [batch_size, head_count, top_leaf_reads],
            ),
            device,
        ),
        Tensor::<B, 3, Bool>::zeros([batch_size, head_count, top_leaf_reads], device),
        Tensor::<B, 3>::zeros([batch_size, head_count, top_leaf_reads], device),
        Tensor::<B, 4>::zeros([batch_size, head_count, top_leaf_reads, leaf_size], device),
        Tensor::<B, 4>::zeros([batch_size, head_count, top_leaf_reads, value_dim], device),
        crate::v2::ExactLeafReadDiagnostics {
            fraction_using_exact_read: 0.0,
            selected_token_position_histogram: Vec::new(),
            average_attention_entropy_per_head: vec![0.0; head_count],
            average_top_token_probability_per_head: vec![0.0; head_count],
        },
    )
}

#[derive(Clone, Copy)]
struct AuditMetricReference<'a> {
    task_family: &'a crate::v2::CausalMemoryTaskFamily,
    context: CausalMemoryEvaluationContext,
    head_contexts: &'a [CausalMemoryHeadContext],
    reference_loss: f32,
    reference_target_logit: f32,
    reference_logits: &'a [f32],
    batch_index: usize,
    target_token_id: i64,
    vocab_size: usize,
}

fn intervention_result<B: Backend>(
    intervention: CausalMemoryIntervention,
    reference: AuditMetricReference<'_>,
    perturbed_logits: Tensor<B, 2>,
) -> Result<CausalMemoryInterventionResult, FractalError> {
    let perturbed_logits = logits_for_batch(
        perturbed_logits,
        reference.batch_index,
        reference.vocab_size,
    )?;
    let perturbed_loss = negative_log_likelihood(&perturbed_logits, reference.target_token_id)?;
    let perturbed_target_logit = target_logit(
        &perturbed_logits,
        reference.target_token_id,
        reference.vocab_size,
    )?;
    let retrieval_accuracy_delta = retrieval_accuracy_delta(
        reference.task_family,
        reference.reference_logits,
        &perturbed_logits,
        reference.target_token_id,
        reference.vocab_size,
    )?;

    Ok(CausalMemoryInterventionResult {
        intervention,
        applied: true,
        context: Some(reference.context),
        head_contexts: reference.head_contexts.to_vec(),
        metrics: Some(CausalMemoryDeltaMetrics {
            loss_delta: perturbed_loss - reference.reference_loss,
            target_logit_delta: reference.reference_target_logit - perturbed_target_logit,
            kl_divergence: kl_divergence(reference.reference_logits, &perturbed_logits)?,
            retrieval_accuracy_delta,
            perplexity_delta: perturbed_loss.exp() - reference.reference_loss.exp(),
        }),
    })
}

fn retrieval_accuracy_delta(
    task_family: &crate::v2::CausalMemoryTaskFamily,
    reference_logits: &[f32],
    perturbed_logits: &[f32],
    target_token_id: i64,
    vocab_size: usize,
) -> Result<Option<f32>, FractalError> {
    match task_family {
        crate::v2::CausalMemoryTaskFamily::OrdinaryLm => Ok(None),
        crate::v2::CausalMemoryTaskFamily::Copy
        | crate::v2::CausalMemoryTaskFamily::AssociativeRecall
        | crate::v2::CausalMemoryTaskFamily::Induction
        | crate::v2::CausalMemoryTaskFamily::NoisyRetrieval
        | crate::v2::CausalMemoryTaskFamily::Custom(_) => Ok(Some(
            target_prediction_accuracy(reference_logits, target_token_id, vocab_size)?
                - target_prediction_accuracy(perturbed_logits, target_token_id, vocab_size)?,
        )),
    }
}

fn target_prediction_accuracy(
    logits: &[f32],
    target_token_id: i64,
    vocab_size: usize,
) -> Result<f32, FractalError> {
    Ok(
        if predicted_token_id(logits, vocab_size)? == target_token_id {
            1.0
        } else {
            0.0
        },
    )
}

fn predicted_token_id(logits: &[f32], vocab_size: usize) -> Result<i64, FractalError> {
    ensure_match("predicted_token_id.vocab_size", logits.len(), vocab_size)?;
    let (predicted_index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| {
            FractalError::InvalidState("cannot predict from empty logits".to_string())
        })?;

    Ok(predicted_index as i64)
}

fn zero_root_readout_for_batch<B: Backend>(
    root_readouts: Tensor<B, 3>,
    batch_index: usize,
    root_index: usize,
) -> Result<Tensor<B, 3>, FractalError> {
    let [batch_size, root_count, root_readout_dim] = root_readouts.dims();
    if batch_index >= batch_size || root_index >= root_count {
        return Err(FractalError::InvalidState(format!(
            "cannot zero root readout for batch {} root {} in [{}, {}, {}]",
            batch_index, root_index, batch_size, root_count, root_readout_dim
        )));
    }
    let mut values = root_readouts
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data("root_readouts"))?;
    for dim_index in 0..root_readout_dim {
        let flat_index = (batch_index * root_count + root_index) * root_readout_dim + dim_index;
        values[flat_index] = 0.0;
    }

    Ok(Tensor::<B, 3>::from_data(
        burn::tensor::TensorData::new(values, [batch_size, root_count, root_readout_dim]),
        &root_readouts.device(),
    ))
}

fn next_best_route_for_batch<B: Backend>(
    routed: &crate::v2::FractalRouteOutput<B>,
    tree: &crate::v2::TreeSummaryState<B>,
    batch_index: usize,
    query_position: usize,
) -> Result<Option<crate::v2::FractalRouteOutput<B>>, FractalError> {
    let selected_leaf_indices = routed.selected_leaf_indices();
    let selected_leaf_mask = routed.selected_leaf_mask();
    let selected_leaf_scores = routed.selected_leaf_scores();
    let selected_leaf_values = routed.selected_leaf_values();
    let [batch_size, head_count, top_leaf_reads] = selected_leaf_indices.dims();
    let [value_batch_size, value_head_count, value_top_leaf_reads, value_dim] =
        selected_leaf_values.dims();
    ensure_match("next_best.value_batch_size", value_batch_size, batch_size)?;
    ensure_match("next_best.value_head_count", value_head_count, head_count)?;
    ensure_match(
        "next_best.value_top_leaf_reads",
        value_top_leaf_reads,
        top_leaf_reads,
    )?;
    if batch_index >= batch_size || top_leaf_reads < 2 {
        return Ok(None);
    }

    let mut index_data = selected_leaf_indices
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(invalid_state_from_data("next_best.selected_leaf_indices"))?;
    let mut mask_data = selected_leaf_mask
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(invalid_state_from_data("next_best.selected_leaf_mask"))?;
    let mut score_data = selected_leaf_scores
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data("next_best.selected_leaf_scores"))?;
    let mut value_data = selected_leaf_values
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(invalid_state_from_data("next_best.selected_leaf_values"))?;
    let mut traces = routed.traces().to_vec();
    let diagnostics = routed.diagnostics().clone();

    for head_index in 0..head_count {
        let primary_flat =
            selection_flat_index(batch_index, head_index, 0, head_count, top_leaf_reads);
        let batch_route = traces
            .get_mut(head_index)
            .and_then(|trace| trace.batch_routes.get_mut(batch_index))
            .ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "missing route trace for head {} batch {} while building next-best route",
                    head_index, batch_index
                ))
            })?;
        let final_step = batch_route.steps.last().ok_or_else(|| {
            FractalError::InvalidState(format!(
                "missing final routing step for head {} batch {} while building next-best route",
                head_index, batch_index
            ))
        })?;
        if final_step.level != 0 {
            return Err(FractalError::InvalidState(
                "next-best substitution requires the final routing step to target leaf nodes"
                    .to_string(),
            ));
        }
        let selected_set = batch_route
            .selected_leaf_indices
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let mut alternate = None;
        for candidate_slot in top_scored_candidate_slots(&final_step.considered_candidate_scores) {
            let leaf_index = final_step.considered_candidate_indices[candidate_slot];
            if selected_set.contains(&leaf_index) {
                continue;
            }
            let span = final_step.considered_candidate_spans[candidate_slot];
            if span.end() > query_position {
                return Err(FractalError::InvalidState(format!(
                    "next-best leaf span [{}, {}) ends after query position {}",
                    span.start(),
                    span.end(),
                    query_position
                )));
            }
            alternate = Some((
                leaf_index,
                final_step.considered_candidate_scores[candidate_slot],
                span,
            ));
            break;
        }
        let Some((next_index, _next_score, next_span)) = alternate else {
            return Ok(None);
        };
        index_data[primary_flat] = next_index as i64;
        mask_data[primary_flat] = true;
        let next_value = tree
            .level(0)
            .ok_or_else(|| {
                FractalError::InvalidState(
                    "next-best substitution requires tree level 0 to be present".to_string(),
                )
            })?
            .values()
            .slice([
                batch_index..batch_index + 1,
                next_index..next_index + 1,
                0..value_dim,
            ])
            .reshape([value_dim])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(invalid_state_from_data("next_best.selected_leaf_value"))?;
        for (dim_index, value) in next_value.iter().copied().enumerate().take(value_dim) {
            let primary_value_flat = ((((batch_index * head_count) + head_index) * top_leaf_reads)
                * value_dim)
                + dim_index;
            value_data[primary_value_flat] = value;
        }
        batch_route.selected_leaf_indices[0] = next_index;
        batch_route.selected_leaf_spans[0] = next_span;
        let active_slot_count = batch_route.selected_leaf_indices.len();
        let mut raw_selected_scores = Vec::with_capacity(active_slot_count);
        for slot in 0..active_slot_count {
            let leaf_index = if slot == 0 {
                next_index
            } else {
                batch_route.selected_leaf_indices[slot]
            };
            raw_selected_scores.push(raw_score_for_considered_leaf(final_step, leaf_index)?);
        }
        let normalized_selected_scores = softmax_vec(&raw_selected_scores);
        for (slot, score) in normalized_selected_scores.iter().copied().enumerate() {
            let flat_index =
                selection_flat_index(batch_index, head_index, slot, head_count, top_leaf_reads);
            score_data[flat_index] = score;
            batch_route.selected_leaf_scores[slot] = score;
        }
        for slot in active_slot_count..top_leaf_reads {
            let flat_index =
                selection_flat_index(batch_index, head_index, slot, head_count, top_leaf_reads);
            score_data[flat_index] = 0.0;
        }
    }

    Ok(Some(crate::v2::FractalRouteOutput::from_parts(
        Tensor::<B, 3, Int>::from_data(
            burn::tensor::TensorData::new(index_data, [batch_size, head_count, top_leaf_reads]),
            &selected_leaf_indices.device(),
        ),
        Tensor::<B, 3, burn::tensor::Bool>::from_data(
            burn::tensor::TensorData::new(mask_data, [batch_size, head_count, top_leaf_reads]),
            &selected_leaf_mask.device(),
        ),
        Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new(score_data, [batch_size, head_count, top_leaf_reads]),
            &selected_leaf_scores.device(),
        ),
        Tensor::<B, 4>::from_data(
            burn::tensor::TensorData::new(
                value_data,
                [batch_size, head_count, top_leaf_reads, value_dim],
            ),
            &selected_leaf_values.device(),
        ),
        traces,
        diagnostics,
    )?))
}

fn top_scored_candidate_slots(scores: &[f32]) -> Vec<usize> {
    let mut slots = scores.iter().copied().enumerate().collect::<Vec<_>>();
    slots.sort_by(|(_, left), (_, right)| right.total_cmp(left));
    slots.into_iter().map(|(index, _)| index).collect()
}

fn raw_score_for_considered_leaf(
    final_step: &crate::v2::BatchRouteStep,
    leaf_index: usize,
) -> Result<f32, FractalError> {
    final_step
        .considered_candidate_indices
        .iter()
        .position(|candidate| *candidate == leaf_index)
        .map(|slot| final_step.considered_candidate_scores[slot])
        .ok_or_else(|| {
            FractalError::InvalidState(format!(
                "leaf {} was not present in the final considered candidate set",
                leaf_index
            ))
        })
}

fn selection_flat_index(
    batch_index: usize,
    head_index: usize,
    slot: usize,
    head_count: usize,
    top_leaf_reads: usize,
) -> usize {
    ((batch_index * head_count + head_index) * top_leaf_reads) + slot
}

fn invalid_state_from_data(
    surface: &'static str,
) -> impl Fn(burn::tensor::DataError) -> FractalError + Copy {
    move |error| {
        FractalError::InvalidState(format!(
            "{surface} could not be inspected through tensor data: {error}"
        ))
    }
}

impl<B, LT> FractalV2LocalBaselineModel<B, LT>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
{
    pub fn new(
        vocab_size: usize,
        token_dim: usize,
        local_trunk: LT,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let shape = FractalV2LocalBaselineShape {
            vocab_size,
            token_dim,
            local_trunk: local_trunk.shape(),
        }
        .validate()?;
        let embedding = EmbeddingConfig::new(vocab_size, token_dim).init(device);

        Ok(Self {
            embedding,
            local_trunk,
            vocab_size: shape.vocab_size,
            token_dim: shape.token_dim,
            root_count: shape.local_trunk.root_count,
            root_state_dim: shape.local_trunk.root_state_dim,
            root_readout_dim: shape.local_trunk.root_readout_dim,
            leaf_size: shape.local_trunk.leaf_size,
        })
    }

    pub fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub fn local_trunk(&self) -> &LT {
        &self.local_trunk
    }

    pub fn shape(&self) -> FractalV2LocalBaselineShape {
        FractalV2LocalBaselineShape {
            vocab_size: self.vocab_size,
            token_dim: self.token_dim,
            local_trunk: LocalTrunkShape {
                token_dim: self.token_dim,
                root_count: self.root_count,
                root_state_dim: self.root_state_dim,
                root_readout_dim: self.root_readout_dim,
                leaf_size: self.leaf_size,
            },
        }
    }

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<FractalV2LocalBaselineOutput<B>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        ensure_nonzero("local_baseline.batch_size", batch_size)?;
        ensure_nonzero("local_baseline.seq_len", seq_len)?;

        let device = input_ids.device();
        let embeddings = self.embedding.forward(input_ids);
        let mut state =
            MultiRootState::zeros_for_local_trunk(batch_size, self.shape().local_trunk, &device)?;
        let mut root_readouts = Vec::with_capacity(seq_len);

        for position in 0..seq_len {
            let token_embedding = embeddings
                .clone()
                .narrow(1, position, 1)
                .reshape([batch_size, self.token_dim]);
            let step = self.local_trunk.step(token_embedding, state)?;
            root_readouts.push(step.root_readouts().reshape([
                batch_size,
                1,
                self.root_count,
                self.root_readout_dim,
            ]));
            state = step.into_next_state();
        }

        let root_readouts = Tensor::cat(root_readouts, 1);
        let mean_readouts = root_readouts
            .clone()
            .sum_dim(2)
            .mul_scalar(1.0 / self.root_count as f64)
            .reshape([batch_size, seq_len, self.root_readout_dim]);
        let diagnostics = summarize_root_readout_sequence(root_readouts.clone())?;

        Ok(FractalV2LocalBaselineOutput {
            root_readouts,
            mean_readouts,
            final_state: state,
            diagnostics,
        })
    }
}

impl FractalV2LocalBaselineShape {
    pub(crate) fn validate(self) -> Result<Self, FractalError> {
        ensure_nonzero("local_baseline.vocab_size", self.vocab_size)?;
        ensure_nonzero("local_baseline.token_dim", self.token_dim)?;
        ensure_nonzero(
            "local_baseline.local_trunk.token_dim",
            self.local_trunk.token_dim,
        )?;
        ensure_nonzero(
            "local_baseline.local_trunk.root_count",
            self.local_trunk.root_count,
        )?;
        ensure_nonzero(
            "local_baseline.local_trunk.root_state_dim",
            self.local_trunk.root_state_dim,
        )?;
        ensure_nonzero(
            "local_baseline.local_trunk.root_readout_dim",
            self.local_trunk.root_readout_dim,
        )?;
        ensure_nonzero(
            "local_baseline.local_trunk.leaf_size",
            self.local_trunk.leaf_size,
        )?;
        ensure_match(
            "local_baseline.local_trunk.token_dim",
            self.local_trunk.token_dim,
            self.token_dim,
        )?;

        Ok(self)
    }
}

fn validate_fractal_v2_model_shape(
    shape: FractalV2ModelShape,
) -> Result<FractalV2ModelShape, FractalError> {
    let FractalV2ModelShape {
        vocab_size,
        token_dim,
        local_trunk,
        leaf_summarizer: leaf,
        tree_merge_cell: tree,
        router,
        exact_read,
        read_fusion,
    } = shape;

    ensure_nonzero("vocab_size", vocab_size)?;
    ensure_nonzero("token_dim", token_dim)?;
    ensure_nonzero("local_trunk.token_dim", local_trunk.token_dim)?;
    ensure_nonzero("local_trunk.root_count", local_trunk.root_count)?;
    ensure_nonzero("local_trunk.root_state_dim", local_trunk.root_state_dim)?;
    ensure_nonzero("local_trunk.root_readout_dim", local_trunk.root_readout_dim)?;
    ensure_nonzero("local_trunk.leaf_size", local_trunk.leaf_size)?;
    ensure_nonzero("leaf_summarizer.readout_dim", leaf.readout_dim)?;
    ensure_nonzero("leaf_summarizer.leaf_size", leaf.leaf_size)?;
    ensure_nonzero("leaf_summarizer.summary_dim", leaf.summary_dim)?;
    ensure_nonzero("leaf_summarizer.key_dim", leaf.key_dim)?;
    ensure_nonzero("leaf_summarizer.value_dim", leaf.value_dim)?;
    ensure_nonzero(
        "leaf_summarizer.token_cache_key_dim",
        leaf.token_cache_key_dim,
    )?;
    ensure_nonzero(
        "leaf_summarizer.token_cache_value_dim",
        leaf.token_cache_value_dim,
    )?;
    ensure_nonzero("tree_merge_cell.summary_dim", tree.summary_dim)?;
    ensure_nonzero("tree_merge_cell.key_dim", tree.key_dim)?;
    ensure_nonzero("tree_merge_cell.value_dim", tree.value_dim)?;
    ensure_nonzero(
        "tree_merge_cell.scale_embedding_dim",
        tree.scale_embedding_dim,
    )?;
    ensure_nonzero("router.query_dim", router.query_dim)?;
    ensure_nonzero("router.key_dim", router.key_dim)?;
    ensure_nonzero("router.head_count", router.head_count)?;
    ensure_nonzero("router.beam_width", router.beam_width)?;
    ensure_nonzero("router.top_leaf_reads", router.top_leaf_reads)?;
    if router.allow_early_stop {
        return Err(FractalError::InvalidConfig(
            "router.allow_early_stop must remain false in v1".to_string(),
        ));
    }
    ensure_nonzero("exact_read.query_dim", exact_read.query_dim)?;
    ensure_nonzero("exact_read.key_dim", exact_read.key_dim)?;
    ensure_nonzero("exact_read.value_dim", exact_read.value_dim)?;
    ensure_nonzero("exact_read.head_count", exact_read.head_count)?;
    ensure_nonzero("exact_read.top_leaf_reads", exact_read.top_leaf_reads)?;
    ensure_nonzero("exact_read.leaf_size", exact_read.leaf_size)?;
    ensure_nonzero("read_fusion.root_count", read_fusion.root_count)?;
    ensure_nonzero("read_fusion.root_readout_dim", read_fusion.root_readout_dim)?;
    ensure_nonzero("read_fusion.routed_value_dim", read_fusion.routed_value_dim)?;
    ensure_nonzero(
        "read_fusion.exact_read_value_dim",
        read_fusion.exact_read_value_dim,
    )?;
    ensure_nonzero(
        "read_fusion.fused_readout_dim",
        read_fusion.fused_readout_dim,
    )?;

    ensure_match("local_trunk.token_dim", local_trunk.token_dim, token_dim)?;
    ensure_match(
        "leaf_summarizer.readout_dim",
        leaf.readout_dim,
        local_trunk.root_readout_dim,
    )?;
    ensure_match(
        "leaf_summarizer.leaf_size",
        leaf.leaf_size,
        local_trunk.leaf_size,
    )?;
    ensure_match(
        "tree_merge_cell.summary_dim",
        tree.summary_dim,
        leaf.summary_dim,
    )?;
    ensure_match("tree_merge_cell.key_dim", tree.key_dim, leaf.key_dim)?;
    ensure_match("tree_merge_cell.value_dim", tree.value_dim, leaf.value_dim)?;
    ensure_match(
        "router.query_dim",
        router.query_dim,
        local_trunk.root_readout_dim,
    )?;
    ensure_match("router.key_dim", router.key_dim, tree.key_dim)?;
    ensure_match(
        "exact_read.query_dim",
        exact_read.query_dim,
        local_trunk.root_readout_dim,
    )?;
    ensure_match(
        "exact_read.key_dim",
        exact_read.key_dim,
        leaf.token_cache_key_dim,
    )?;
    ensure_match(
        "exact_read.value_dim",
        exact_read.value_dim,
        leaf.token_cache_value_dim,
    )?;
    ensure_match(
        "exact_read.head_count",
        exact_read.head_count,
        router.head_count,
    )?;
    ensure_match(
        "exact_read.top_leaf_reads",
        exact_read.top_leaf_reads,
        router.top_leaf_reads,
    )?;
    ensure_match(
        "exact_read.leaf_size",
        exact_read.leaf_size,
        local_trunk.leaf_size,
    )?;
    ensure_match(
        "read_fusion.root_count",
        read_fusion.root_count,
        local_trunk.root_count,
    )?;
    ensure_match(
        "read_fusion.root_readout_dim",
        read_fusion.root_readout_dim,
        local_trunk.root_readout_dim,
    )?;
    ensure_match(
        "read_fusion.routed_value_dim",
        read_fusion.routed_value_dim,
        tree.value_dim,
    )?;
    ensure_match(
        "read_fusion.exact_read_value_dim",
        read_fusion.exact_read_value_dim,
        exact_read.value_dim,
    )?;

    Ok(FractalV2ModelShape {
        vocab_size,
        token_dim,
        local_trunk,
        leaf_summarizer: leaf,
        tree_merge_cell: tree,
        router,
        exact_read,
        read_fusion,
    })
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
    use core::marker::PhantomData;

    use burn::{
        backend::Candle,
        module::Module,
        tensor::{Tensor, TensorData},
    };

    use super::*;
    use crate::{
        registry::gradient_l2_norm,
        registry::CpuTrainBackend,
        v2::local_trunk::{BaselineLocalTrunk, BaselineLocalTrunkConfig, LocalTrunkStepOutput},
        v2::{
            BaselineExactLeafRead, BaselineExactLeafReadConfig, BaselineFractalRouterHead,
            BaselineFractalRouterHeadConfig, BaselineLeafSummarizer, BaselineLeafSummarizerConfig,
            BaselineReadFusion, BaselineReadFusionConfig, BaselineTreeMergeCell,
            BaselineTreeMergeCellConfig, ReadFusionOutput,
        },
    };

    type TestBackend = Candle<f32, i64>;
    type TestComponents = FractalV2Components<
        StubLocalTrunk<TestBackend>,
        StubLeafSummarizer<TestBackend>,
        StubTreeMergeCell<TestBackend>,
        StubRouter<TestBackend>,
        StubExactRead<TestBackend>,
        StubReadFusion<TestBackend>,
    >;
    type BaselineModel<B> = FractalV2LocalBaselineModel<B, BaselineLocalTrunk<B>>;
    type BaselineV2Model<B> = FractalV2Model<
        B,
        BaselineLocalTrunk<B>,
        BaselineLeafSummarizer<B>,
        BaselineTreeMergeCell<B>,
        BaselineFractalRouterHead<B>,
        BaselineExactLeafRead<B>,
        BaselineReadFusion<B>,
    >;

    #[derive(Module, Debug)]
    struct StubLocalTrunk<B: Backend> {
        token_dim: usize,
        root_count: usize,
        root_state_dim: usize,
        root_readout_dim: usize,
        leaf_size: usize,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StubLeafSummarizer<B: Backend> {
        readout_dim: usize,
        leaf_size: usize,
        summary_dim: usize,
        key_dim: usize,
        value_dim: usize,
        token_cache_key_dim: usize,
        token_cache_value_dim: usize,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StubTreeMergeCell<B: Backend> {
        summary_dim: usize,
        key_dim: usize,
        value_dim: usize,
        scale_embedding_dim: usize,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StubRouter<B: Backend> {
        query_dim: usize,
        key_dim: usize,
        head_count: usize,
        beam_width: usize,
        top_leaf_reads: usize,
        allow_early_stop: bool,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StubReadFusion<B: Backend> {
        root_count: usize,
        root_readout_dim: usize,
        routed_value_dim: usize,
        exact_read_value_dim: usize,
        fused_readout_dim: usize,
        _marker: PhantomData<B>,
    }

    #[derive(Module, Debug)]
    struct StubExactRead<B: Backend> {
        query_dim: usize,
        key_dim: usize,
        value_dim: usize,
        head_count: usize,
        top_leaf_reads: usize,
        leaf_size: usize,
        fill_value: f32,
        _marker: PhantomData<B>,
    }

    impl<B: Backend> StubLocalTrunk<B> {
        fn new(shape: LocalTrunkShape) -> Self {
            Self {
                token_dim: shape.token_dim,
                root_count: shape.root_count,
                root_state_dim: shape.root_state_dim,
                root_readout_dim: shape.root_readout_dim,
                leaf_size: shape.leaf_size,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StubLeafSummarizer<B> {
        fn new(shape: LeafSummarizerShape) -> Self {
            Self {
                readout_dim: shape.readout_dim,
                leaf_size: shape.leaf_size,
                summary_dim: shape.summary_dim,
                key_dim: shape.key_dim,
                value_dim: shape.value_dim,
                token_cache_key_dim: shape.token_cache_key_dim,
                token_cache_value_dim: shape.token_cache_value_dim,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StubTreeMergeCell<B> {
        fn new(shape: TreeMergeCellShape) -> Self {
            Self {
                summary_dim: shape.summary_dim,
                key_dim: shape.key_dim,
                value_dim: shape.value_dim,
                scale_embedding_dim: shape.scale_embedding_dim,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StubRouter<B> {
        fn new(shape: FractalRouterHeadShape) -> Self {
            Self {
                query_dim: shape.query_dim,
                key_dim: shape.key_dim,
                head_count: shape.head_count,
                beam_width: shape.beam_width,
                top_leaf_reads: shape.top_leaf_reads,
                allow_early_stop: shape.allow_early_stop,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StubReadFusion<B> {
        fn new(shape: ReadFusionShape) -> Self {
            Self {
                root_count: shape.root_count,
                root_readout_dim: shape.root_readout_dim,
                routed_value_dim: shape.routed_value_dim,
                exact_read_value_dim: shape.exact_read_value_dim,
                fused_readout_dim: shape.fused_readout_dim,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> StubExactRead<B> {
        fn new(shape: ExactLeafReadShape, fill_value: f32) -> Self {
            Self {
                query_dim: shape.query_dim,
                key_dim: shape.key_dim,
                value_dim: shape.value_dim,
                head_count: shape.head_count,
                top_leaf_reads: shape.top_leaf_reads,
                leaf_size: shape.leaf_size,
                fill_value,
                _marker: PhantomData,
            }
        }
    }

    impl<B: Backend> LocalTrunk<B> for StubLocalTrunk<B> {
        fn shape(&self) -> LocalTrunkShape {
            LocalTrunkShape {
                token_dim: self.token_dim,
                root_count: self.root_count,
                root_state_dim: self.root_state_dim,
                root_readout_dim: self.root_readout_dim,
                leaf_size: self.leaf_size,
            }
        }

        fn step(
            &self,
            token_embedding: Tensor<B, 2>,
            state: MultiRootState<B>,
        ) -> Result<LocalTrunkStepOutput<B>, FractalError> {
            let [batch_size, token_dim] = token_embedding.dims();
            let base_signal = token_embedding
                .clone()
                .narrow(1, 0, 1)
                .reshape([batch_size, 1, 1])
                .repeat(&[1, self.root_count, self.root_readout_dim]);
            let root_offsets = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (0..self.root_count * self.root_readout_dim)
                        .map(|index| {
                            let root_index = index / self.root_readout_dim;
                            root_index as f32 + 1.0
                        })
                        .collect::<Vec<_>>(),
                    [1, self.root_count, self.root_readout_dim],
                ),
                &token_embedding.device(),
            )
            .repeat(&[batch_size, 1, 1]);
            let token_scale = (token_dim.max(1) as f64).recip();
            let root_readouts = base_signal.mul_scalar(token_scale) + root_offsets;
            let next_state = MultiRootState::from_tensors(
                state.recurrent(),
                root_readouts.clone(),
                root_readouts.clone(),
            )?;

            Ok(LocalTrunkStepOutput::new(next_state, root_readouts))
        }
    }

    impl<B: Backend> LeafSummarizer<B> for StubLeafSummarizer<B> {
        fn shape(&self) -> LeafSummarizerShape {
            LeafSummarizerShape {
                readout_dim: self.readout_dim,
                leaf_size: self.leaf_size,
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                token_cache_key_dim: self.token_cache_key_dim,
                token_cache_value_dim: self.token_cache_value_dim,
            }
        }

        fn summarize_sealed_leaf(
            &self,
            token_readouts: Tensor<B, 4>,
        ) -> Result<crate::v2::LeafSummarizerOutput<B>, FractalError> {
            let [batch_size, _root_count, leaf_size, readout_dim] = token_readouts.dims();
            ensure_match("stub_leaf_summarizer.leaf_size", leaf_size, self.leaf_size)?;
            ensure_match(
                "stub_leaf_summarizer.readout_dim",
                readout_dim,
                self.readout_dim,
            )?;

            Ok(crate::v2::LeafSummarizerOutput::new(
                Tensor::<B, 2>::zeros([batch_size, self.summary_dim], &token_readouts.device()),
                Tensor::<B, 2>::zeros([batch_size, self.key_dim], &token_readouts.device()),
                Tensor::<B, 2>::zeros([batch_size, self.value_dim], &token_readouts.device()),
                Tensor::<B, 3>::zeros(
                    [batch_size, leaf_size, self.token_cache_key_dim],
                    &token_readouts.device(),
                ),
                Tensor::<B, 3>::zeros(
                    [batch_size, leaf_size, self.token_cache_value_dim],
                    &token_readouts.device(),
                ),
                Tensor::<B, 2, burn::tensor::Bool>::ones(
                    [batch_size, leaf_size],
                    &token_readouts.device(),
                ),
            ))
        }
    }

    impl<B: Backend> TreeMergeCell<B> for StubTreeMergeCell<B> {
        fn shape(&self) -> TreeMergeCellShape {
            TreeMergeCellShape {
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                scale_embedding_dim: self.scale_embedding_dim,
            }
        }

        fn merge_pair(
            &self,
            left: crate::v2::TreeNodeBatch<B>,
            _right: crate::v2::TreeNodeBatch<B>,
            _level: usize,
        ) -> Result<crate::v2::TreeMergeOutput<B>, FractalError> {
            let [batch_size, summary_dim] = left.summary().dims();
            let [key_batch_size, key_dim] = left.key().dims();
            let [value_batch_size, value_dim] = left.value().dims();
            ensure_match(
                "stub_tree_merge_cell.key_batch_size",
                key_batch_size,
                batch_size,
            )?;
            ensure_match(
                "stub_tree_merge_cell.value_batch_size",
                value_batch_size,
                batch_size,
            )?;
            ensure_match(
                "stub_tree_merge_cell.summary_dim",
                summary_dim,
                self.summary_dim,
            )?;
            ensure_match("stub_tree_merge_cell.key_dim", key_dim, self.key_dim)?;
            ensure_match("stub_tree_merge_cell.value_dim", value_dim, self.value_dim)?;

            Ok(crate::v2::TreeMergeOutput::new(
                Tensor::<B, 2>::zeros([batch_size, self.summary_dim], &left.summary().device()),
                Tensor::<B, 2>::zeros([batch_size, self.key_dim], &left.summary().device()),
                Tensor::<B, 2>::zeros([batch_size, self.value_dim], &left.summary().device()),
            ))
        }
    }

    impl<B: Backend> FractalRouterHead<B> for StubRouter<B> {
        fn shape(&self) -> FractalRouterHeadShape {
            FractalRouterHeadShape {
                query_dim: self.query_dim,
                key_dim: self.key_dim,
                head_count: self.head_count,
                beam_width: self.beam_width,
                top_leaf_reads: self.top_leaf_reads,
                allow_early_stop: self.allow_early_stop,
            }
        }

        fn route(
            &self,
            query: Tensor<B, 2>,
            _query_position: usize,
            tree: &crate::v2::TreeSummaryState<B>,
        ) -> Result<crate::v2::router::FractalRouteOutput<B>, FractalError> {
            let [batch_size, query_dim] = query.dims();
            ensure_match("stub_router.query_dim", query_dim, self.query_dim)?;
            let value_dim = tree.value_dim();
            let leaf_count = tree.level(0).map(|level| level.node_count()).unwrap_or(0);
            let has_tree = tree.root_address().is_some();
            let query_signal = query
                .clone()
                .slice([0..batch_size, 0..1])
                .reshape([batch_size]);
            let query_signal = query_signal
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .unwrap();
            let mut selected_leaf_indices =
                vec![-1i64; batch_size * self.head_count * self.top_leaf_reads];
            let mut selected_leaf_mask =
                vec![false; batch_size * self.head_count * self.top_leaf_reads];
            let mut selected_leaf_scores =
                vec![0.0f32; batch_size * self.head_count * self.top_leaf_reads];
            if has_tree {
                for (batch_index, signal) in query_signal.iter().enumerate().take(batch_size) {
                    let primary_leaf_index = if leaf_count > 1 && *signal >= 1.25 {
                        1
                    } else {
                        0
                    };
                    for head_index in 0..self.head_count {
                        let flat_index =
                            (batch_index * self.head_count + head_index) * self.top_leaf_reads;
                        selected_leaf_indices[flat_index] = primary_leaf_index as i64;
                        selected_leaf_mask[flat_index] = true;
                        selected_leaf_scores[flat_index] = 1.0;
                    }
                }
            }
            crate::v2::router::FractalRouteOutput::from_parts(
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(
                        selected_leaf_indices,
                        [batch_size, self.head_count, self.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 3, burn::tensor::Bool>::from_data(
                    TensorData::new(
                        selected_leaf_mask,
                        [batch_size, self.head_count, self.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 3>::from_data(
                    TensorData::new(
                        selected_leaf_scores,
                        [batch_size, self.head_count, self.top_leaf_reads],
                    ),
                    &query.device(),
                ),
                Tensor::<B, 4>::zeros(
                    [batch_size, self.head_count, self.top_leaf_reads, value_dim],
                    &query.device(),
                )
                .add_scalar(if has_tree { 0.5 } else { 0.0 }),
                (0..self.head_count)
                    .map(|_| crate::v2::HeadRouteTrace {
                        batch_routes: (0..batch_size)
                            .map(|batch_index| crate::v2::BatchHeadRoute {
                                steps: Vec::new(),
                                selected_leaf_indices: if has_tree {
                                    vec![if leaf_count > 1 && query_signal[batch_index] >= 1.25 {
                                        1
                                    } else {
                                        0
                                    }]
                                } else {
                                    Vec::new()
                                },
                                selected_leaf_spans: if has_tree {
                                    let start =
                                        if leaf_count > 1 && query_signal[batch_index] >= 1.25 {
                                            16
                                        } else {
                                            0
                                        };
                                    vec![crate::v2::TokenSpan::new(start, start + 16).unwrap()]
                                } else {
                                    Vec::new()
                                },
                                selected_leaf_scores: if has_tree { vec![1.0] } else { Vec::new() },
                            })
                            .collect(),
                    })
                    .collect(),
                crate::v2::FractalRoutingDiagnostics {
                    routing_depth_histogram: Vec::new(),
                    candidate_entropy_per_head: vec![0.0; self.head_count],
                    selected_span_distance_histogram: Vec::new(),
                    head_agreement_rate: 1.0,
                    head_disagreement_rate: 0.0,
                },
            )
        }
    }

    impl<B: Backend> ReadFusion<B> for StubReadFusion<B> {
        fn shape(&self) -> ReadFusionShape {
            ReadFusionShape {
                root_count: self.root_count,
                root_readout_dim: self.root_readout_dim,
                routed_value_dim: self.routed_value_dim,
                exact_read_value_dim: self.exact_read_value_dim,
                fused_readout_dim: self.fused_readout_dim,
            }
        }

        fn fuse(
            &self,
            input: &ReadFusionInput<B>,
            ablation: ReadFusionAblation,
        ) -> Result<ReadFusionOutput<B>, FractalError> {
            let [batch_size, root_count, root_readout_dim] = input.root_readouts().dims();
            ensure_match("stub_read_fusion.root_count", root_count, self.root_count)?;
            ensure_match(
                "stub_read_fusion.root_readout_dim",
                root_readout_dim,
                self.root_readout_dim,
            )?;
            let active_root_count = ablation.active_root_count().unwrap_or(root_count);
            if active_root_count == 0 || active_root_count > root_count {
                return Err(FractalError::InvalidConfig(format!(
                    "stub_read_fusion.active_root_count must be within 1..={root_count}, got {active_root_count}"
                )));
            }

            let root_summary = input
                .root_readouts()
                .narrow(1, 0, active_root_count)
                .sum_dim(1)
                .mul_scalar(1.0 / active_root_count as f64)
                .reshape([batch_size, self.root_readout_dim]);

            let routed_summary = summarize_stub_retrieved_values(
                input.routed_values(),
                input.routed_scores(),
                input.routed_mask(),
                self.routed_value_dim,
                ablation.include_routed_values(),
            )?;
            let exact_read_summary = summarize_stub_retrieved_values(
                input.exact_read_values(),
                input.routed_scores(),
                input.exact_read_mask(),
                self.exact_read_value_dim,
                ablation.include_exact_read_values(),
            )?;

            let root_scalar = root_summary
                .clone()
                .sum_dim(1)
                .reshape([batch_size, 1])
                .repeat(&[1, self.fused_readout_dim]);
            let routed_scalar = routed_summary
                .clone()
                .sum_dim(1)
                .reshape([batch_size, 1])
                .repeat(&[1, self.fused_readout_dim]);
            let exact_read_scalar = exact_read_summary
                .clone()
                .sum_dim(1)
                .reshape([batch_size, 1])
                .repeat(&[1, self.fused_readout_dim]);

            ReadFusionOutput::new(
                root_scalar.clone() + routed_scalar.clone() + exact_read_scalar.clone(),
                root_scalar,
                routed_scalar,
                exact_read_scalar,
                root_summary,
                routed_summary,
                exact_read_summary,
            )
        }
    }

    impl<B: Backend> ExactLeafRead<B> for StubExactRead<B> {
        fn shape(&self) -> ExactLeafReadShape {
            ExactLeafReadShape {
                query_dim: self.query_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                head_count: self.head_count,
                top_leaf_reads: self.top_leaf_reads,
                leaf_size: self.leaf_size,
            }
        }

        fn read(
            &self,
            _query: Tensor<B, 2>,
            _query_position: usize,
            routed: &crate::v2::FractalRouteOutput<B>,
            leaf_token_cache: &crate::v2::LeafTokenCache<B>,
        ) -> Result<crate::v2::ExactLeafReadOutput<B>, FractalError> {
            let selected_leaf_indices = routed.selected_leaf_indices();
            let selected_leaf_mask = routed.selected_leaf_mask();
            let [batch_size, head_count, top_leaf_reads] = selected_leaf_indices.dims();
            let leaf_mask_data = selected_leaf_mask
                .clone()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap();
            let leaf_index_data = selected_leaf_indices
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap();
            let mut local_indices = vec![-1i64; batch_size * head_count * top_leaf_reads];
            let mut absolute_positions = vec![-1i64; batch_size * head_count * top_leaf_reads];
            let mut token_scores = vec![0.0f32; batch_size * head_count * top_leaf_reads];
            let mut attention_weights =
                vec![0.0f32; batch_size * head_count * top_leaf_reads * self.leaf_size];
            let mut read_values =
                vec![0.0f32; batch_size * head_count * top_leaf_reads * self.value_dim];
            for (flat_index, is_active) in leaf_mask_data.iter().copied().enumerate() {
                if !is_active {
                    continue;
                }
                let leaf_index = usize::try_from(leaf_index_data[flat_index]).unwrap();
                local_indices[flat_index] = 0;
                absolute_positions[flat_index] =
                    leaf_token_cache.shared_spans()[leaf_index].start() as i64;
                token_scores[flat_index] = 1.0;
                let attention_offset = flat_index * self.leaf_size;
                attention_weights[attention_offset] = 1.0;
                let value_offset = flat_index * self.value_dim;
                read_values[value_offset..value_offset + self.value_dim].fill(self.fill_value);
            }

            crate::v2::ExactLeafReadOutput::new(
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(local_indices, [batch_size, head_count, top_leaf_reads]),
                    &selected_leaf_mask.device(),
                ),
                Tensor::<B, 3, Int>::from_data(
                    TensorData::new(absolute_positions, [batch_size, head_count, top_leaf_reads]),
                    &selected_leaf_mask.device(),
                ),
                selected_leaf_mask.clone(),
                Tensor::<B, 3>::from_data(
                    TensorData::new(token_scores, [batch_size, head_count, top_leaf_reads]),
                    &selected_leaf_mask.device(),
                ),
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        attention_weights,
                        [batch_size, head_count, top_leaf_reads, self.leaf_size],
                    ),
                    &selected_leaf_mask.device(),
                ),
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        read_values,
                        [batch_size, head_count, top_leaf_reads, self.value_dim],
                    ),
                    &selected_leaf_mask.device(),
                ),
                crate::v2::ExactLeafReadDiagnostics {
                    fraction_using_exact_read: if leaf_mask_data.iter().any(|value| *value) {
                        1.0 / top_leaf_reads as f32
                    } else {
                        0.0
                    },
                    selected_token_position_histogram: if leaf_mask_data.iter().any(|value| *value)
                    {
                        vec![crate::v2::ExactReadHistogramBin {
                            value: 0,
                            count: leaf_mask_data.iter().filter(|value| **value).count(),
                        }]
                    } else {
                        Vec::new()
                    },
                    average_attention_entropy_per_head: vec![0.0; head_count],
                    average_top_token_probability_per_head: vec![0.0; head_count],
                },
            )
        }
    }

    fn summarize_stub_retrieved_values<B: Backend>(
        values: Tensor<B, 4>,
        scores: Tensor<B, 3>,
        mask: Tensor<B, 3, burn::tensor::Bool>,
        expected_value_dim: usize,
        include_values: bool,
    ) -> Result<Tensor<B, 2>, FractalError> {
        let [batch_size, head_count, top_leaf_reads, value_dim] = values.dims();
        ensure_match("stub_read_fusion.value_dim", value_dim, expected_value_dim)?;
        assert_dims3(
            "stub_read_fusion.scores",
            scores.dims(),
            [batch_size, head_count, top_leaf_reads],
        );
        assert_dims3(
            "stub_read_fusion.mask",
            mask.dims(),
            [batch_size, head_count, top_leaf_reads],
        );
        if !include_values {
            return Ok(Tensor::<B, 2>::zeros(
                [batch_size, expected_value_dim],
                &values.device(),
            ));
        }

        let mask = mask
            .reshape([batch_size, head_count, top_leaf_reads, 1])
            .repeat(&[1, 1, 1, value_dim]);
        let masked_values = Tensor::<B, 4>::zeros(
            [batch_size, head_count, top_leaf_reads, value_dim],
            &values.device(),
        )
        .mask_where(mask, values);
        let score_broadcast = scores
            .reshape([batch_size, head_count, top_leaf_reads, 1])
            .repeat(&[1, 1, 1, value_dim]);

        Ok((masked_values * score_broadcast)
            .sum_dim(2)
            .sum_dim(1)
            .mul_scalar(1.0 / head_count as f64)
            .reshape([batch_size, expected_value_dim]))
    }

    fn valid_components_with_exact_read_fill(fill_value: f32) -> TestComponents {
        FractalV2Components {
            local_trunk: StubLocalTrunk::<TestBackend>::new(LocalTrunkShape {
                token_dim: 128,
                root_count: 2,
                root_state_dim: 96,
                root_readout_dim: 64,
                leaf_size: 16,
            }),
            leaf_summarizer: StubLeafSummarizer::<TestBackend>::new(LeafSummarizerShape {
                readout_dim: 64,
                leaf_size: 16,
                summary_dim: 80,
                key_dim: 48,
                value_dim: 72,
                token_cache_key_dim: 48,
                token_cache_value_dim: 56,
            }),
            tree_merge_cell: StubTreeMergeCell::<TestBackend>::new(TreeMergeCellShape {
                summary_dim: 80,
                key_dim: 48,
                value_dim: 72,
                scale_embedding_dim: 12,
            }),
            router: StubRouter::<TestBackend>::new(FractalRouterHeadShape {
                query_dim: 64,
                key_dim: 48,
                head_count: 4,
                beam_width: 2,
                top_leaf_reads: 2,
                allow_early_stop: false,
            }),
            exact_read: StubExactRead::<TestBackend>::new(
                ExactLeafReadShape {
                    query_dim: 64,
                    key_dim: 48,
                    value_dim: 56,
                    head_count: 4,
                    top_leaf_reads: 2,
                    leaf_size: 16,
                },
                fill_value,
            ),
            read_fusion: StubReadFusion::<TestBackend>::new(ReadFusionShape {
                root_count: 2,
                root_readout_dim: 64,
                routed_value_dim: 72,
                exact_read_value_dim: 56,
                fused_readout_dim: 96,
            }),
        }
    }

    fn valid_components() -> TestComponents {
        valid_components_with_exact_read_fill(1.0)
    }

    fn baseline_model<B: Backend>(root_count: usize, device: &B::Device) -> BaselineModel<B> {
        FractalV2LocalBaselineModel::new(
            64,
            8,
            BaselineLocalTrunkConfig::new(8, root_count, 6, 4, 16)
                .try_init(device)
                .unwrap(),
            device,
        )
        .unwrap()
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
                    initializer: burn::nn::Initializer::Uniform {
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
                    initializer: burn::nn::Initializer::Uniform {
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
                    initializer: burn::nn::Initializer::Uniform {
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

    fn token_ids<B: Backend>(
        values: &[i64],
        shape: [usize; 2],
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), shape), device)
    }

    fn assert_invalid_config(components: TestComponents, expected_field: &str) {
        let device = <TestBackend as Backend>::Device::default();
        let error = FractalV2Model::new(32_000, 128, components, &device).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains(expected_field))
        );
    }

    fn assert_dims3(name: &str, actual: [usize; 3], expected: [usize; 3]) {
        assert_eq!(actual, expected, "{name} mismatch");
    }

    #[test]
    fn fractal_v2_model_builds_from_consistent_boundaries() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();

        assert_eq!(
            model.shape(),
            FractalV2ModelShape {
                vocab_size: 32_000,
                token_dim: 128,
                local_trunk: LocalTrunkShape {
                    token_dim: 128,
                    root_count: 2,
                    root_state_dim: 96,
                    root_readout_dim: 64,
                    leaf_size: 16,
                },
                leaf_summarizer: LeafSummarizerShape {
                    readout_dim: 64,
                    leaf_size: 16,
                    summary_dim: 80,
                    key_dim: 48,
                    value_dim: 72,
                    token_cache_key_dim: 48,
                    token_cache_value_dim: 56,
                },
                tree_merge_cell: TreeMergeCellShape {
                    summary_dim: 80,
                    key_dim: 48,
                    value_dim: 72,
                    scale_embedding_dim: 12,
                },
                router: FractalRouterHeadShape {
                    query_dim: 64,
                    key_dim: 48,
                    head_count: 4,
                    beam_width: 2,
                    top_leaf_reads: 2,
                    allow_early_stop: false,
                },
                exact_read: ExactLeafReadShape {
                    query_dim: 64,
                    key_dim: 48,
                    value_dim: 56,
                    head_count: 4,
                    top_leaf_reads: 2,
                    leaf_size: 16,
                },
                read_fusion: ReadFusionShape {
                    root_count: 2,
                    root_readout_dim: 64,
                    routed_value_dim: 72,
                    exact_read_value_dim: 56,
                    fused_readout_dim: 96,
                },
            }
        );
        assert_eq!(model.output().logical_dims(), [96, 32_000]);
    }

    #[test]
    fn fractal_v2_model_rejects_mismatched_router_query_dim() {
        let mut components = valid_components();
        components.router.query_dim = 63;

        assert_invalid_config(components, "router.query_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_root_state_dim() {
        let mut components = valid_components();
        components.local_trunk.root_state_dim = 0;

        assert_invalid_config(components, "local_trunk.root_state_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_leaf_summary_dim() {
        let mut components = valid_components();
        components.leaf_summarizer.summary_dim = 0;

        assert_invalid_config(components, "leaf_summarizer.summary_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_token_cache_value_dim() {
        let mut components = valid_components();
        components.leaf_summarizer.token_cache_value_dim = 0;
        components.read_fusion.exact_read_value_dim = 0;

        assert_invalid_config(components, "leaf_summarizer.token_cache_value_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_scale_embedding_dim() {
        let mut components = valid_components();
        components.tree_merge_cell.scale_embedding_dim = 0;

        assert_invalid_config(components, "tree_merge_cell.scale_embedding_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_exact_read_value_dim() {
        let mut components = valid_components();
        components.exact_read.value_dim = 0;
        components.read_fusion.exact_read_value_dim = 0;

        assert_invalid_config(components, "exact_read.value_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_fused_readout_dim() {
        let mut components = valid_components();
        components.read_fusion.fused_readout_dim = 0;

        assert_invalid_config(components, "read_fusion.fused_readout_dim");
    }

    #[test]
    fn fractal_v2_model_shape_uses_validated_contract_not_live_component_state() {
        let device = <TestBackend as Backend>::Device::default();
        let mut model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();

        model.router.query_dim = 17;
        model.exact_read.key_dim = 23;
        model.read_fusion.fused_readout_dim = 23;

        assert_eq!(model.shape().router.query_dim, 64);
        assert_eq!(model.shape().exact_read.key_dim, 48);
        assert_eq!(model.shape().read_fusion.fused_readout_dim, 96);
        assert_eq!(model.output().logical_dims(), [96, 32_000]);
    }

    #[test]
    fn fractal_v2_local_baseline_forward_returns_inspectable_multi_root_outputs() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_model::<TestBackend>(2, &device);
        let input_ids = token_ids::<TestBackend>(&[1, 2, 3, 4, 4, 3, 2, 1], [2, 4], &device);

        let output = model.forward(input_ids).unwrap();

        assert_eq!(output.root_readouts().dims(), [2, 4, 2, 4]);
        assert_eq!(output.mean_readouts().dims(), [2, 4, 4]);
        assert_eq!(output.final_state().shape().batch_size, 2);
        assert_eq!(output.final_state().shape().root_count, 2);
        assert_eq!(output.diagnostics().per_root.len(), 2);
        assert!(output
            .diagnostics()
            .mean_pairwise_cosine_similarity
            .is_finite());
    }

    #[test]
    fn fractal_v2_local_baseline_is_causal_over_sequence_positions() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_model::<TestBackend>(2, &device);
        let input_a = token_ids::<TestBackend>(&[1, 2, 3, 4], [1, 4], &device);
        let input_b = token_ids::<TestBackend>(&[1, 2, 9, 4], [1, 4], &device);

        let output_a = model.forward(input_a).unwrap();
        let output_b = model.forward(input_b).unwrap();
        let prefix_a = output_a
            .root_readouts()
            .narrow(1, 0, 2)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let prefix_b = output_b
            .root_readouts()
            .narrow(1, 0, 2)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();

        assert_eq!(prefix_a, prefix_b);
    }

    #[test]
    fn fractal_v2_model_forward_retrieval_trace_reaches_exact_read_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let trace = model.forward_retrieval_trace(input_ids).unwrap();

        assert_eq!(trace.steps().len(), 16);
        let final_step = trace.steps().last().unwrap();
        assert!(final_step.sealed_leaf().is_some());
        assert_eq!(
            final_step
                .routed()
                .selected_leaf_mask()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap(),
            vec![true, false, true, false, true, false, true, false]
        );
        assert_eq!(
            final_step
                .exact_read()
                .selected_token_mask()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap(),
            vec![true, false, true, false, true, false, true, false]
        );
        assert_eq!(
            final_step
                .exact_read()
                .selected_token_absolute_positions()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .unwrap(),
            vec![0, -1, 0, -1, 0, -1, 0, -1]
        );
        assert_eq!(
            trace.final_state().leaf_token_cache().shared_spans().len(),
            1
        );
    }

    #[test]
    fn fractal_v2_model_forward_retrieval_trace_supports_no_exact_read_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let enabled = FractalV2Model::new(
            32_000,
            128,
            valid_components_with_exact_read_fill(1.0),
            &device,
        )
        .unwrap();
        let disabled = FractalV2Model::new(
            32_000,
            128,
            valid_components_with_exact_read_fill(0.0),
            &device,
        )
        .unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let enabled_trace = enabled.forward_retrieval_trace(input_ids.clone()).unwrap();
        let disabled_trace = disabled.forward_retrieval_trace(input_ids).unwrap();
        let enabled_final = enabled_trace.steps().last().unwrap();
        let disabled_final = disabled_trace.steps().last().unwrap();

        assert_eq!(
            enabled_final
                .routed()
                .selected_leaf_mask()
                .to_data()
                .convert::<bool>(),
            disabled_final
                .routed()
                .selected_leaf_mask()
                .to_data()
                .convert::<bool>()
        );
        let enabled_values = enabled_final
            .exact_read()
            .read_values()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let disabled_values = disabled_final
            .exact_read()
            .read_values()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        assert!(enabled_values.iter().any(|value| *value > 0.0));
        assert!(disabled_values.iter().all(|value| *value == 0.0));
        assert_ne!(enabled_values, disabled_values);
    }

    #[test]
    fn fractal_v2_memory_mode_no_memory_skips_tree_and_exact_read_execution() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );

        let trace = model
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::NoMemory)
            .unwrap();
        let final_step = trace.steps().last().unwrap();

        assert!(trace
            .final_state()
            .sealed_leaves()
            .shared_spans()
            .is_empty());
        assert!(trace
            .final_state()
            .leaf_token_cache()
            .shared_spans()
            .is_empty());
        assert!(trace.final_state().tree().levels().is_empty());
        assert!(final_step
            .routed()
            .selected_leaf_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .all(|value| !*value));
        assert!(final_step
            .exact_read()
            .selected_token_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .all(|value| !*value));
    }

    #[test]
    fn fractal_v2_memory_mode_tree_only_routes_memory_without_exact_read() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );

        let trace = model
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)
            .unwrap();
        let final_step = trace.steps().last().unwrap();

        assert_eq!(
            trace.final_state().leaf_token_cache().shared_spans(),
            &[
                crate::v2::TokenSpan::new(0, 16).unwrap(),
                crate::v2::TokenSpan::new(16, 32).unwrap(),
            ]
        );
        assert_eq!(
            trace.final_state().tree().level(0).unwrap().shared_spans(),
            &[
                crate::v2::TokenSpan::new(0, 16).unwrap(),
                crate::v2::TokenSpan::new(16, 32).unwrap(),
            ]
        );
        assert!(final_step
            .routed()
            .selected_leaf_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .any(|value| *value));
        assert!(final_step
            .exact_read()
            .selected_token_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .all(|value| !*value));
    }

    #[test]
    fn fractal_v2_memory_mode_summaries_only_builds_tree_without_routing() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );

        let trace = model
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::SummariesOnly)
            .unwrap();
        let final_step = trace.steps().last().unwrap();

        assert_eq!(
            trace.final_state().leaf_token_cache().shared_spans(),
            &[
                crate::v2::TokenSpan::new(0, 16).unwrap(),
                crate::v2::TokenSpan::new(16, 32).unwrap(),
            ]
        );
        assert_eq!(
            trace.final_state().tree().level(0).unwrap().shared_spans(),
            &[
                crate::v2::TokenSpan::new(0, 16).unwrap(),
                crate::v2::TokenSpan::new(16, 32).unwrap(),
            ]
        );
        assert!(final_step
            .routed()
            .selected_leaf_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .all(|value| !*value));
        assert!(final_step
            .exact_read()
            .selected_token_mask()
            .to_data()
            .convert::<bool>()
            .into_vec::<bool>()
            .unwrap()
            .iter()
            .all(|value| !*value));
    }

    #[test]
    fn fractal_v2_model_forward_produces_end_to_end_logits() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let output = model.forward(input_ids).unwrap();

        assert_eq!(output.fused_readouts().dims(), [1, 16, 96]);
        assert_eq!(output.logits().dims(), [1, 16, 32_000]);
        assert_eq!(output.final_state().shape().roots.batch_size, 1);
        assert_eq!(output.final_state().shape().layout.leaf_size(), 16);
    }

    #[test]
    fn fractal_v2_model_forward_supports_zero_routed_value_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let enabled = model.forward(input_ids.clone()).unwrap();
        let ablated = model
            .forward_with_ablation(input_ids, ReadFusionAblation::without_routed_values())
            .unwrap();

        assert_ne!(
            enabled.fused_readouts().to_data().convert::<f32>(),
            ablated.fused_readouts().to_data().convert::<f32>()
        );
        assert_ne!(
            enabled.logits().to_data().convert::<f32>(),
            ablated.logits().to_data().convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_model_forward_supports_zero_exact_read_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let enabled = model.forward(input_ids.clone()).unwrap();
        let ablated = model
            .forward_with_ablation(input_ids, ReadFusionAblation::without_exact_read_values())
            .unwrap();

        assert_ne!(
            enabled.fused_readouts().to_data().convert::<f32>(),
            ablated.fused_readouts().to_data().convert::<f32>()
        );
        assert_ne!(
            enabled.logits().to_data().convert::<f32>(),
            ablated.logits().to_data().convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_model_forward_supports_zero_extra_roots_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 16],
            &device,
        );

        let enabled = model.forward(input_ids.clone()).unwrap();
        let single_root = model
            .forward_with_ablation(input_ids, ReadFusionAblation::with_active_root_count(1))
            .unwrap();

        assert_ne!(
            enabled.fused_readouts().to_data().convert::<f32>(),
            single_root.fused_readouts().to_data().convert::<f32>()
        );
        assert_ne!(
            enabled.logits().to_data().convert::<f32>(),
            single_root.logits().to_data().convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_retrieval_trace_extra_root_ablation_changes_retrieval_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2Model::new(32_000, 128, valid_components(), &device).unwrap();
        let input_ids = token_ids::<TestBackend>(
            &(1..=32).map(i64::from).collect::<Vec<_>>(),
            [1, 32],
            &device,
        );

        let enabled = model
            .forward_retrieval_trace_with_ablation(input_ids.clone(), ReadFusionAblation::full())
            .unwrap();
        let single_root = model
            .forward_retrieval_trace_with_ablation(
                input_ids,
                ReadFusionAblation::with_active_root_count(1),
            )
            .unwrap();
        let enabled_final = enabled.steps().last().unwrap();
        let single_root_final = single_root.steps().last().unwrap();

        assert_ne!(
            enabled_final
                .routed()
                .selected_leaf_indices()
                .to_data()
                .convert::<i64>(),
            single_root_final
                .routed()
                .selected_leaf_indices()
                .to_data()
                .convert::<i64>()
        );
        assert_ne!(
            enabled_final
                .exact_read()
                .selected_token_absolute_positions()
                .to_data()
                .convert::<i64>(),
            single_root_final
                .exact_read()
                .selected_token_absolute_positions()
                .to_data()
                .convert::<i64>()
        );
    }

    #[test]
    fn fractal_v2_baseline_stack_forward_runs_end_to_end() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );

        let output = model.forward(input_ids.clone()).unwrap();
        let no_routed = model
            .forward_with_ablation(
                input_ids.clone(),
                ReadFusionAblation::without_routed_values(),
            )
            .unwrap();
        let no_exact = model
            .forward_with_ablation(
                input_ids.clone(),
                ReadFusionAblation::without_exact_read_values(),
            )
            .unwrap();
        let one_root = model
            .forward_with_ablation(input_ids, ReadFusionAblation::with_active_root_count(1))
            .unwrap();

        assert_eq!(output.fused_readouts().dims(), [1, 32, 8]);
        assert_eq!(output.logits().dims(), [1, 32, 64]);
        assert_ne!(
            output.logits().to_data().convert::<f32>(),
            no_routed.logits().to_data().convert::<f32>()
        );
        assert_ne!(
            output.logits().to_data().convert::<f32>(),
            no_exact.logits().to_data().convert::<f32>()
        );
        assert_ne!(
            output.logits().to_data().convert::<f32>(),
            one_root.logits().to_data().convert::<f32>()
        );
    }

    fn repeated_target_ids<B: Backend>(
        token_id: i64,
        shape: [usize; 2],
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![token_id; shape[0] * shape[1]], shape),
            device,
        )
    }

    fn intervention_result_for(
        report: &CausalMemoryAuditSampleReport,
        intervention: CausalMemoryIntervention,
    ) -> &CausalMemoryInterventionResult {
        report
            .interventions
            .iter()
            .find(|result| result.intervention == intervention)
            .expect("missing intervention result")
    }

    #[test]
    fn fractal_v2_causal_memory_audit_reports_measurable_utilities() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=64)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 64],
            &device,
        );
        let target_ids = repeated_target_ids::<TestBackend>(7, [1, 64], &device);
        let plan = CausalMemoryAuditPlan::all(vec![crate::v2::CausalMemoryAuditSample {
            batch_index: 0,
            position: 63,
            task_family: crate::v2::CausalMemoryTaskFamily::OrdinaryLm,
        }])
        .unwrap();

        let report = model
            .audit_causal_memory(input_ids, target_ids, &plan)
            .unwrap();

        assert_eq!(report.sample_reports.len(), 1);
        assert!(report.tree_retrieval_utility.is_some());
        assert!(report.exact_leaf_read_utility.is_some());
        assert_eq!(report.utility_by_root.len(), 2);
        let sample = &report.sample_reports[0];
        let no_tree = intervention_result_for(sample, CausalMemoryIntervention::NoTreeRead);
        let no_exact = intervention_result_for(sample, CausalMemoryIntervention::NoExactLeafRead);
        let next_best =
            intervention_result_for(sample, CausalMemoryIntervention::NextBestSpanSubstitution);
        let root_drops = sample
            .interventions
            .iter()
            .filter(|result| {
                matches!(
                    result.intervention,
                    CausalMemoryIntervention::RootDrop { .. }
                )
            })
            .collect::<Vec<_>>();

        assert!(no_tree.applied);
        assert!(no_exact.applied);
        assert!(next_best.applied);
        assert_eq!(root_drops.len(), 2);
        assert_eq!(
            next_best.context.unwrap().routing_depth,
            sample.reference_context.routing_depth
        );
        assert!(no_tree.metrics.unwrap().kl_divergence.is_finite());
        assert!(no_exact.metrics.unwrap().kl_divergence.is_finite());
        assert!(next_best.metrics.unwrap().kl_divergence.is_finite());
    }

    #[test]
    fn fractal_v2_causal_memory_audit_tracks_all_selected_slots_per_head() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=64)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 64],
            &device,
        );
        let target_ids = repeated_target_ids::<TestBackend>(7, [1, 64], &device);
        let plan = CausalMemoryAuditPlan::all(vec![crate::v2::CausalMemoryAuditSample {
            batch_index: 0,
            position: 63,
            task_family: crate::v2::CausalMemoryTaskFamily::OrdinaryLm,
        }])
        .unwrap();

        let report = model
            .audit_causal_memory(input_ids, target_ids, &plan)
            .unwrap();
        let sample = &report.sample_reports[0];

        assert_eq!(sample.reference_head_contexts.len(), 2);
        assert!(sample
            .reference_head_contexts
            .iter()
            .all(|context| context.selected_leaf_indices.len() == 2));
        assert!(sample
            .reference_head_contexts
            .iter()
            .all(|context| context.span_distances.len() == 2));
        assert!(report.utility_by_selected_leaf.len() >= 2);
        assert!(report.utility_by_span_distance.len() >= 2);
    }

    #[test]
    fn fractal_v2_next_best_route_renormalizes_selected_scores() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=64)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 64],
            &device,
        );

        let trace = model.forward_retrieval_trace(input_ids).unwrap();
        let final_step = trace.steps().last().unwrap();
        let next_best =
            next_best_route_for_batch(final_step.routed(), trace.final_state().tree(), 0, 64)
                .unwrap()
                .expect("expected an unselected next-best leaf on the final step");
        let next_best_scores = next_best
            .selected_leaf_scores()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let next_best_indices = next_best
            .selected_leaf_indices()
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .unwrap();

        for head_index in 0..2 {
            let batch_route = &next_best.traces()[head_index].batch_routes[0];
            let final_route_step = batch_route.steps.last().unwrap();
            let selected_raw_scores = batch_route
                .selected_leaf_indices
                .iter()
                .map(|leaf_index| raw_score_for_considered_leaf(final_route_step, *leaf_index))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let expected = softmax_vec(&selected_raw_scores);
            let score_base = head_index * 2;

            assert!(
                (next_best_scores[score_base] + next_best_scores[score_base + 1] - 1.0).abs()
                    < 1.0e-5
            );
            assert_eq!(
                next_best_indices[score_base] as usize,
                batch_route.selected_leaf_indices[0]
            );
            assert_eq!(
                next_best_indices[score_base + 1] as usize,
                batch_route.selected_leaf_indices[1]
            );
            assert!((next_best_scores[score_base] - expected[0]).abs() < 1.0e-5);
            assert!((next_best_scores[score_base + 1] - expected[1]).abs() < 1.0e-5);
        }
    }

    #[test]
    fn fractal_v2_causal_memory_audit_reports_retrieval_accuracy_for_retrieval_tasks() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=64)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 64],
            &device,
        );
        let target_ids = repeated_target_ids::<TestBackend>(7, [1, 64], &device);
        let plan = CausalMemoryAuditPlan::all(vec![crate::v2::CausalMemoryAuditSample {
            batch_index: 0,
            position: 63,
            task_family: crate::v2::CausalMemoryTaskFamily::Copy,
        }])
        .unwrap();

        let report = model
            .audit_causal_memory(input_ids, target_ids, &plan)
            .unwrap();

        assert!(report.sample_reports[0]
            .interventions
            .iter()
            .filter(|result| result.applied)
            .all(|result| result
                .metrics
                .and_then(|metrics| metrics.retrieval_accuracy_delta)
                .is_some()));
        assert_eq!(report.utility_by_task_family.len(), 1);
        assert!(report.utility_by_task_family[0]
            .stats
            .average_retrieval_accuracy_delta
            .is_some());
    }

    #[test]
    fn fractal_v2_causal_memory_audit_no_tree_matches_manual_final_step_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );
        let target_ids = repeated_target_ids::<TestBackend>(5, [1, 32], &device);
        let plan = CausalMemoryAuditPlan::all(vec![crate::v2::CausalMemoryAuditSample {
            batch_index: 0,
            position: 31,
            task_family: crate::v2::CausalMemoryTaskFamily::OrdinaryLm,
        }])
        .unwrap();

        let audit = model
            .audit_causal_memory(input_ids.clone(), target_ids.clone(), &plan)
            .unwrap();
        let full = model.forward(input_ids.clone()).unwrap();
        let no_tree = model
            .forward_with_ablation(input_ids, ReadFusionAblation::without_routed_values())
            .unwrap();
        let sample = &audit.sample_reports[0];
        let result = intervention_result_for(sample, CausalMemoryIntervention::NoTreeRead);
        let target = 5i64;
        let full_logits = full
            .logits()
            .slice([0..1, 31..32, 0..64])
            .reshape([64])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let no_tree_logits = no_tree
            .logits()
            .slice([0..1, 31..32, 0..64])
            .reshape([64])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let expected_loss_delta = negative_log_likelihood(&no_tree_logits, target).unwrap()
            - negative_log_likelihood(&full_logits, target).unwrap();
        let expected_target_logit_delta = target_logit(&full_logits, target, 64).unwrap()
            - target_logit(&no_tree_logits, target, 64).unwrap();

        assert!((result.metrics.unwrap().loss_delta - expected_loss_delta).abs() < 1.0e-5);
        assert!(
            (result.metrics.unwrap().target_logit_delta - expected_target_logit_delta).abs()
                < 1.0e-5
        );
    }

    #[test]
    fn fractal_v2_causal_memory_audit_no_exact_matches_manual_final_step_ablation() {
        let device = <TestBackend as Backend>::Device::default();
        let model = baseline_v2_model::<TestBackend>(&device);
        let input_ids = token_ids::<TestBackend>(
            &(1..=32)
                .map(|value| (value % 63 + 1) as i64)
                .collect::<Vec<_>>(),
            [1, 32],
            &device,
        );
        let target_ids = repeated_target_ids::<TestBackend>(11, [1, 32], &device);
        let plan = CausalMemoryAuditPlan::all(vec![crate::v2::CausalMemoryAuditSample {
            batch_index: 0,
            position: 31,
            task_family: crate::v2::CausalMemoryTaskFamily::OrdinaryLm,
        }])
        .unwrap();

        let audit = model
            .audit_causal_memory(input_ids.clone(), target_ids.clone(), &plan)
            .unwrap();
        let full = model.forward(input_ids.clone()).unwrap();
        let no_exact = model
            .forward_with_ablation(input_ids, ReadFusionAblation::without_exact_read_values())
            .unwrap();
        let sample = &audit.sample_reports[0];
        let result = intervention_result_for(sample, CausalMemoryIntervention::NoExactLeafRead);
        let target = 11i64;
        let full_logits = full
            .logits()
            .slice([0..1, 31..32, 0..64])
            .reshape([64])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let no_exact_logits = no_exact
            .logits()
            .slice([0..1, 31..32, 0..64])
            .reshape([64])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let expected_loss_delta = negative_log_likelihood(&no_exact_logits, target).unwrap()
            - negative_log_likelihood(&full_logits, target).unwrap();
        let expected_target_logit_delta = target_logit(&full_logits, target, 64).unwrap()
            - target_logit(&no_exact_logits, target, 64).unwrap();

        assert!((result.metrics.unwrap().loss_delta - expected_loss_delta).abs() < 1.0e-5);
        assert!(
            (result.metrics.unwrap().target_logit_delta - expected_target_logit_delta).abs()
                < 1.0e-5
        );
    }

    #[test]
    fn zero_root_readout_for_batch_only_zeroes_the_selected_root() {
        let device = <TestBackend as Backend>::Device::default();
        let root_readouts = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..8).map(|value| value as f32).collect::<Vec<_>>(),
                [1, 2, 4],
            ),
            &device,
        );

        let zeroed = zero_root_readout_for_batch(root_readouts, 0, 1).unwrap();
        let values = zeroed.to_data().convert::<f32>().into_vec::<f32>().unwrap();

        assert_eq!(&values[0..4], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(&values[4..8], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fractal_v2_local_baseline_training_step_backpropagates() {
        let device = <CpuTrainBackend as Backend>::Device::default();
        let model = baseline_model::<CpuTrainBackend>(2, &device);
        let input_ids = token_ids::<CpuTrainBackend>(&[1, 2, 3, 4, 4, 3, 2, 1], [2, 4], &device);

        let loss = model.forward(input_ids).unwrap().mean_readouts().sum();
        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &model);

        assert!(gradient_l2_norm(&model, &grads) > 0.0);
    }

    #[test]
    fn fractal_v2_local_baseline_runs_for_single_and_multi_root_configs() {
        let device = <TestBackend as Backend>::Device::default();
        let input_ids = token_ids::<TestBackend>(&[1, 2, 3, 4], [1, 4], &device);
        let single_root_model = baseline_model::<TestBackend>(1, &device);
        let multi_root_model = baseline_model::<TestBackend>(2, &device);

        let single = single_root_model.forward(input_ids.clone()).unwrap();
        let multi = multi_root_model.forward(input_ids).unwrap();

        assert_eq!(single.root_readouts().dims(), [1, 4, 1, 4]);
        assert_eq!(multi.root_readouts().dims(), [1, 4, 2, 4]);
        assert_eq!(single.diagnostics().per_root.len(), 1);
        assert_eq!(multi.diagnostics().per_root.len(), 2);
        assert!(multi.diagnostics().mean_pairwise_cosine_similarity < 0.9999);
    }

    #[test]
    fn fractal_v2_local_baseline_model_accepts_nondefault_leaf_size() {
        let device = <TestBackend as Backend>::Device::default();
        let model = FractalV2LocalBaselineModel::new(
            64,
            8,
            StubLocalTrunk::<TestBackend>::new(LocalTrunkShape {
                token_dim: 8,
                root_count: 2,
                root_state_dim: 6,
                root_readout_dim: 4,
                leaf_size: 8,
            }),
            &device,
        )
        .unwrap();

        assert_eq!(model.shape().local_trunk.leaf_size, 8);
    }
}
