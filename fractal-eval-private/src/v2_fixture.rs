use burn::{nn::Initializer, tensor::backend::Backend};

use fractal_core::{
    error::FractalError,
    v2::{BaselineReadFusion, BaselineReadFusionConfig},
    BaselineExactLeafRead, BaselineExactLeafReadConfig, BaselineFractalRouterHead,
    BaselineFractalRouterHeadConfig, BaselineLeafSummarizer, BaselineLeafSummarizerConfig,
    BaselineLocalTrunk, BaselineLocalTrunkConfig, BaselineTreeMergeCell,
    BaselineTreeMergeCellConfig, FractalV2Components, FractalV2Model,
};

pub type BaselineV2SyntheticModel<B> = FractalV2Model<
    B,
    BaselineLocalTrunk<B>,
    BaselineLeafSummarizer<B>,
    BaselineTreeMergeCell<B>,
    BaselineFractalRouterHead<B>,
    BaselineExactLeafRead<B>,
    BaselineReadFusion<B>,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineV2SyntheticModelConfig {
    pub vocab_size: usize,
    pub token_dim: usize,
}

impl BaselineV2SyntheticModelConfig {
    pub const fn new(vocab_size: usize, token_dim: usize) -> Self {
        Self {
            vocab_size,
            token_dim,
        }
    }
}

impl Default for BaselineV2SyntheticModelConfig {
    fn default() -> Self {
        Self::new(64, 8)
    }
}

pub fn build_baseline_v2_synthetic_model<B: Backend>(
    config: BaselineV2SyntheticModelConfig,
    device: &B::Device,
) -> Result<BaselineV2SyntheticModel<B>, FractalError> {
    FractalV2Model::new(
        config.vocab_size,
        config.token_dim,
        FractalV2Components {
            local_trunk: BaselineLocalTrunkConfig::new(config.token_dim, 2, 6, 4, 16)
                .try_init(device)?,
            leaf_summarizer: BaselineLeafSummarizerConfig {
                readout_dim: 4,
                leaf_size: 16,
                summary_dim: 6,
                key_dim: 4,
                value_dim: 5,
                token_cache_key_dim: 4,
                token_cache_value_dim: 6,
            }
            .try_init(device)?,
            tree_merge_cell: BaselineTreeMergeCellConfig {
                summary_dim: 6,
                key_dim: 4,
                value_dim: 5,
                scale_embedding_dim: 4,
            }
            .try_init(device)?,
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
            .try_init(device)?,
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
            .try_init(device)?,
            read_fusion: BaselineReadFusionConfig {
                root_count: 2,
                root_readout_dim: 4,
                routed_value_dim: 5,
                exact_read_value_dim: 6,
                fused_readout_dim: config.token_dim,
                initializer: Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                },
            }
            .try_init(device)?,
        },
        device,
    )
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;

    use super::*;
    use crate::{default_v2_synthetic_probe_suites, SyntheticProbeModel};

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn baseline_v2_fixture_matches_default_probe_contract() {
        let device = Default::default();
        let model = build_baseline_v2_synthetic_model::<TestBackend>(
            BaselineV2SyntheticModelConfig::default(),
            &device,
        )
        .unwrap();

        for suite in default_v2_synthetic_probe_suites() {
            suite
                .validate_for_model(model.vocab_size(), model.leaf_size())
                .unwrap();
        }
    }
}
