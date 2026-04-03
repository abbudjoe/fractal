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
    leaf::{LeafSummarizer, LeafSummarizerShape},
    local_trunk::{
        summarize_root_readout_sequence, LocalTrunk, LocalTrunkDiagnostics, LocalTrunkShape,
    },
    read_fusion::{ReadFusion, ReadFusionShape},
    router::{FractalRouterHead, FractalRouterHeadShape},
    state::MultiRootState,
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
    pub read_fusion: ReadFusionShape,
}

impl FractalV2ModelShape {
    pub(crate) fn validate(self) -> Result<Self, FractalError> {
        validate_fractal_v2_model_shape(
            self.vocab_size,
            self.token_dim,
            self.local_trunk,
            self.leaf_summarizer,
            self.tree_merge_cell,
            self.router,
            self.read_fusion,
        )
    }
}

#[derive(Debug)]
pub struct FractalV2Components<LT, LS, TM, RH, RF> {
    pub local_trunk: LT,
    pub leaf_summarizer: LS,
    pub tree_merge_cell: TM,
    pub router: RH,
    pub read_fusion: RF,
}

#[derive(Debug, Clone)]
pub struct FractalV2LocalBaselineOutput<B: Backend> {
    root_readouts: Tensor<B, 4>,
    mean_readouts: Tensor<B, 3>,
    final_state: MultiRootState<B>,
    diagnostics: LocalTrunkDiagnostics,
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
    RF: Module<B>,
> {
    embedding: Embedding<B>,
    local_trunk: LT,
    leaf_summarizer: LS,
    tree_merge_cell: TM,
    router: RH,
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
    fused_readout_dim: usize,
}

impl<B, LT, LS, TM, RH, RF> FractalV2Model<B, LT, LS, TM, RH, RF>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
{
    pub fn new(
        vocab_size: usize,
        token_dim: usize,
        components: FractalV2Components<LT, LS, TM, RH, RF>,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let FractalV2Components {
            local_trunk,
            leaf_summarizer,
            tree_merge_cell,
            router,
            read_fusion,
        } = components;
        let shape = FractalV2ModelShape {
            vocab_size,
            token_dim,
            local_trunk: local_trunk.shape(),
            leaf_summarizer: leaf_summarizer.shape(),
            tree_merge_cell: tree_merge_cell.shape(),
            router: router.shape(),
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

    pub fn read_fusion(&self) -> &RF {
        &self.read_fusion
    }

    pub fn output(&self) -> &LanguageModelHead<B> {
        &self.output
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
            read_fusion: ReadFusionShape {
                root_count: self.root_count,
                root_readout_dim: self.root_readout_dim,
                retrieved_value_dim: self.token_cache_value_dim,
                fused_readout_dim: self.fused_readout_dim,
            },
        }
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
    vocab_size: usize,
    token_dim: usize,
    local_trunk: LocalTrunkShape,
    leaf: LeafSummarizerShape,
    tree: TreeMergeCellShape,
    router: FractalRouterHeadShape,
    read_fusion: ReadFusionShape,
) -> Result<FractalV2ModelShape, FractalError> {
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
    ensure_nonzero("read_fusion.root_count", read_fusion.root_count)?;
    ensure_nonzero("read_fusion.root_readout_dim", read_fusion.root_readout_dim)?;
    ensure_nonzero(
        "read_fusion.retrieved_value_dim",
        read_fusion.retrieved_value_dim,
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
        "read_fusion.retrieved_value_dim",
        read_fusion.retrieved_value_dim,
        leaf.token_cache_value_dim,
    )?;

    Ok(FractalV2ModelShape {
        vocab_size,
        token_dim,
        local_trunk,
        leaf_summarizer: leaf,
        tree_merge_cell: tree,
        router,
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
    };

    type TestBackend = Candle<f32, i64>;
    type TestComponents = FractalV2Components<
        StubLocalTrunk<TestBackend>,
        StubLeafSummarizer<TestBackend>,
        StubTreeMergeCell<TestBackend>,
        StubRouter<TestBackend>,
        StubReadFusion<TestBackend>,
    >;
    type BaselineModel<B> = FractalV2LocalBaselineModel<B, BaselineLocalTrunk<B>>;

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
        retrieved_value_dim: usize,
        fused_readout_dim: usize,
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
                retrieved_value_dim: shape.retrieved_value_dim,
                fused_readout_dim: shape.fused_readout_dim,
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
            let [batch_size, _] = token_embedding.dims();
            let root_readouts = Tensor::<B, 3>::zeros(
                [batch_size, self.root_count, self.root_readout_dim],
                &token_embedding.device(),
            );
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
    }

    impl<B: Backend> ReadFusion<B> for StubReadFusion<B> {
        fn shape(&self) -> ReadFusionShape {
            ReadFusionShape {
                root_count: self.root_count,
                root_readout_dim: self.root_readout_dim,
                retrieved_value_dim: self.retrieved_value_dim,
                fused_readout_dim: self.fused_readout_dim,
            }
        }
    }

    fn valid_components() -> TestComponents {
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
            read_fusion: StubReadFusion::<TestBackend>::new(ReadFusionShape {
                root_count: 2,
                root_readout_dim: 64,
                retrieved_value_dim: 56,
                fused_readout_dim: 96,
            }),
        }
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
                read_fusion: ReadFusionShape {
                    root_count: 2,
                    root_readout_dim: 64,
                    retrieved_value_dim: 56,
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
        components.read_fusion.retrieved_value_dim = 0;

        assert_invalid_config(components, "leaf_summarizer.token_cache_value_dim");
    }

    #[test]
    fn fractal_v2_model_rejects_zero_width_scale_embedding_dim() {
        let mut components = valid_components();
        components.tree_merge_cell.scale_embedding_dim = 0;

        assert_invalid_config(components, "tree_merge_cell.scale_embedding_dim");
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
        model.read_fusion.fused_readout_dim = 23;

        assert_eq!(model.shape().router.query_dim, 64);
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
