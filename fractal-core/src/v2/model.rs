use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Tensor},
};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
};

use super::{
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

    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        self.forward_with_ablation(input_ids, ReadFusionAblation::default())
    }

    pub fn forward_with_ablation(
        &self,
        input_ids: Tensor<B, 2, Int>,
        ablation: ReadFusionAblation,
    ) -> Result<FractalV2ForwardOutput<B>, FractalError> {
        let trace = self.forward_retrieval_trace_with_ablation(input_ids, ablation)?;
        self.forward_from_retrieval_trace(trace, ablation)
    }

    pub fn forward_retrieval_trace(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<FractalV2RetrievalTrace<B>, FractalError> {
        self.forward_retrieval_trace_with_ablation(input_ids, ReadFusionAblation::default())
    }

    pub fn forward_retrieval_trace_with_ablation(
        &self,
        input_ids: Tensor<B, 2, Int>,
        ablation: ReadFusionAblation,
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
            let sealed_leaf = state.append_root_readouts(
                root_readouts.clone(),
                &self.leaf_summarizer,
                &self.tree_merge_cell,
            )?;
            let query = summarize_query_from_roots(
                root_readouts.clone(),
                ablation.active_root_count().unwrap_or(self.root_count),
                self.root_count,
                self.root_readout_dim,
            )?;
            let query_position = position + 1;
            let routed = self
                .router
                .route(query.clone(), query_position, state.tree())?;
            let exact_read =
                self.exact_read
                    .read(query, query_position, &routed, state.leaf_token_cache())?;
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
        if self.local_trunk.leaf_size != 16 {
            return Err(FractalError::InvalidConfig(format!(
                "local_baseline.local_trunk.leaf_size must equal 16 in phase 3, got {}",
                self.local_trunk.leaf_size
            )));
        }
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
    fn fractal_v2_local_baseline_model_rejects_non_phase_three_leaf_size() {
        let device = <TestBackend as Backend>::Device::default();
        let error = FractalV2LocalBaselineModel::new(
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
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("local_baseline.local_trunk.leaf_size"))
        );
    }
}
