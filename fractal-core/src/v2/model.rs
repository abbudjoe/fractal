use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::backend::Backend,
};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
};

use super::{
    leaf::{LeafSummarizer, LeafSummarizerShape},
    local_trunk::{LocalTrunk, LocalTrunkShape},
    read_fusion::{ReadFusion, ReadFusionShape},
    router::{FractalRouterHead, FractalRouterHeadShape},
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

#[derive(Debug)]
pub struct FractalV2Components<LT, LS, TM, RH, RF> {
    pub local_trunk: LT,
    pub leaf_summarizer: LS,
    pub tree_merge_cell: TM,
    pub router: RH,
    pub read_fusion: RF,
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
        let shape = Self::resolve_shape(
            vocab_size,
            token_dim,
            local_trunk.shape(),
            leaf_summarizer.shape(),
            tree_merge_cell.shape(),
            router.shape(),
            read_fusion.shape(),
        )?;
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
                token_dim: self.token_dim,
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

    fn resolve_shape(
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
        ensure_nonzero("leaf_summarizer.token_dim", leaf.token_dim)?;
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
        ensure_match("leaf_summarizer.token_dim", leaf.token_dim, token_dim)?;
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

    use burn::{backend::Candle, module::Module};

    use super::*;

    type TestBackend = Candle<f32, i64>;
    type TestComponents = FractalV2Components<
        StubLocalTrunk<TestBackend>,
        StubLeafSummarizer<TestBackend>,
        StubTreeMergeCell<TestBackend>,
        StubRouter<TestBackend>,
        StubReadFusion<TestBackend>,
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
        token_dim: usize,
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
                token_dim: shape.token_dim,
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
    }

    impl<B: Backend> LeafSummarizer<B> for StubLeafSummarizer<B> {
        fn shape(&self) -> LeafSummarizerShape {
            LeafSummarizerShape {
                token_dim: self.token_dim,
                leaf_size: self.leaf_size,
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                token_cache_key_dim: self.token_cache_key_dim,
                token_cache_value_dim: self.token_cache_value_dim,
            }
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
                token_dim: 128,
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
                    token_dim: 128,
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
}
