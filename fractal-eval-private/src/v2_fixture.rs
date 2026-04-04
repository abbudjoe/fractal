use burn::{nn::Initializer, tensor::backend::Backend};
use serde::Serialize;

use fractal_core::{
    error::FractalError,
    v2::{BaselineReadFusion, BaselineReadFusionConfig},
    BaselineExactLeafRead, BaselineExactLeafReadConfig, BaselineFractalRouterHead,
    BaselineFractalRouterHeadConfig, BaselineLeafSummarizer, BaselineLeafSummarizerConfig,
    BaselineLocalTrunk, BaselineLocalTrunkConfig, BaselineTreeMergeCell,
    BaselineTreeMergeCellConfig, FractalV2Components, FractalV2Model,
};

use crate::v2_synthetic::MIN_V2_PROBE_VOCAB_SIZE;

pub type BaselineV2SyntheticModel<B> = FractalV2Model<
    B,
    BaselineLocalTrunk<B>,
    BaselineLeafSummarizer<B>,
    BaselineTreeMergeCell<B>,
    BaselineFractalRouterHead<B>,
    BaselineExactLeafRead<B>,
    BaselineReadFusion<B>,
>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BaselineV2SyntheticModelConfig {
    pub vocab_size: usize,
    pub token_dim: usize,
    pub root_count: usize,
    pub total_root_state_dim: usize,
    pub total_root_readout_dim: usize,
    pub leaf_size: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub token_cache_key_dim: usize,
    pub token_cache_value_dim: usize,
    pub scale_embedding_dim: usize,
    pub routing_head_count: usize,
    pub beam_width: usize,
    pub top_leaf_reads: usize,
    pub exact_read_head_count: usize,
}

impl BaselineV2SyntheticModelConfig {
    pub const fn new(vocab_size: usize, token_dim: usize) -> Self {
        Self {
            vocab_size,
            token_dim,
            root_count: 2,
            total_root_state_dim: 12,
            total_root_readout_dim: 8,
            leaf_size: 16,
            summary_dim: 6,
            key_dim: 4,
            value_dim: 5,
            token_cache_key_dim: 4,
            token_cache_value_dim: 6,
            scale_embedding_dim: 4,
            routing_head_count: 2,
            beam_width: 2,
            top_leaf_reads: 2,
            exact_read_head_count: 2,
        }
    }

    pub const fn with_leaf_size(self, leaf_size: usize) -> Self {
        Self { leaf_size, ..self }
    }

    pub const fn with_root_count_preserving_total_budget(self, root_count: usize) -> Self {
        Self { root_count, ..self }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.root_count == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model.root_count must be greater than zero".to_string(),
            ));
        }
        if self.total_root_state_dim == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model.total_root_state_dim must be greater than zero"
                    .to_string(),
            ));
        }
        if self.total_root_readout_dim == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model.total_root_readout_dim must be greater than zero"
                    .to_string(),
            ));
        }
        if !self.total_root_state_dim.is_multiple_of(self.root_count) {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_v2_synthetic_model.total_root_state_dim {} must be divisible by root_count {}",
                self.total_root_state_dim, self.root_count
            )));
        }
        if !self.total_root_readout_dim.is_multiple_of(self.root_count) {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_v2_synthetic_model.total_root_readout_dim {} must be divisible by root_count {}",
                self.total_root_readout_dim, self.root_count
            )));
        }
        if self.leaf_size == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model.leaf_size must be greater than zero".to_string(),
            ));
        }
        if self.routing_head_count == 0 || self.exact_read_head_count == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model head counts must be greater than zero".to_string(),
            ));
        }
        if self.beam_width == 0 || self.top_leaf_reads == 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_v2_synthetic_model beam_width and top_leaf_reads must be greater than zero".to_string(),
            ));
        }
        if self.top_leaf_reads > self.beam_width {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_v2_synthetic_model.top_leaf_reads ({}) must not exceed beam_width ({})",
                self.top_leaf_reads, self.beam_width
            )));
        }

        Ok(())
    }

    pub const fn root_state_dim(&self) -> usize {
        self.total_root_state_dim / self.root_count
    }

    pub const fn root_readout_dim(&self) -> usize {
        self.total_root_readout_dim / self.root_count
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
    config.validate()?;
    if config.vocab_size < MIN_V2_PROBE_VOCAB_SIZE {
        return Err(FractalError::InvalidConfig(format!(
            "baseline_v2_synthetic_model.vocab_size must be at least {}, got {}",
            MIN_V2_PROBE_VOCAB_SIZE, config.vocab_size
        )));
    }
    let root_state_dim = config.root_state_dim();
    let root_readout_dim = config.root_readout_dim();

    FractalV2Model::new(
        config.vocab_size,
        config.token_dim,
        FractalV2Components {
            local_trunk: BaselineLocalTrunkConfig::new(
                config.token_dim,
                config.root_count,
                root_state_dim,
                root_readout_dim,
                config.leaf_size,
            )
            .try_init(device)?,
            leaf_summarizer: BaselineLeafSummarizerConfig {
                readout_dim: root_readout_dim,
                leaf_size: config.leaf_size,
                summary_dim: config.summary_dim,
                key_dim: config.key_dim,
                value_dim: config.value_dim,
                token_cache_key_dim: config.token_cache_key_dim,
                token_cache_value_dim: config.token_cache_value_dim,
            }
            .try_init(device)?,
            tree_merge_cell: BaselineTreeMergeCellConfig {
                summary_dim: config.summary_dim,
                key_dim: config.key_dim,
                value_dim: config.value_dim,
                scale_embedding_dim: config.scale_embedding_dim,
            }
            .try_init(device)?,
            router: BaselineFractalRouterHeadConfig {
                query_dim: root_readout_dim,
                key_dim: config.key_dim,
                head_count: config.routing_head_count,
                beam_width: config.beam_width,
                top_leaf_reads: config.top_leaf_reads,
                allow_early_stop: false,
                initializer: Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                },
            }
            .try_init(device)?,
            exact_read: BaselineExactLeafReadConfig {
                query_dim: root_readout_dim,
                key_dim: config.token_cache_key_dim,
                value_dim: config.token_cache_value_dim,
                head_count: config.exact_read_head_count,
                top_leaf_reads: config.top_leaf_reads,
                leaf_size: config.leaf_size,
                initializer: Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                },
            }
            .try_init(device)?,
            read_fusion: BaselineReadFusionConfig {
                root_count: config.root_count,
                root_readout_dim,
                routed_value_dim: config.value_dim,
                exact_read_value_dim: config.token_cache_value_dim,
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

    #[test]
    fn baseline_v2_fixture_rejects_vocab_below_probe_minimum() {
        let device = Default::default();
        let error = build_baseline_v2_synthetic_model::<TestBackend>(
            BaselineV2SyntheticModelConfig::new(MIN_V2_PROBE_VOCAB_SIZE - 1, 8),
            &device,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message)
                if message.contains("baseline_v2_synthetic_model.vocab_size")
        ));
    }

    #[test]
    fn baseline_v2_fixture_supports_single_root_equal_budget_variant() {
        let device = Default::default();
        let base = BaselineV2SyntheticModelConfig::default();
        let single_root = base.with_root_count_preserving_total_budget(1);
        let model = build_baseline_v2_synthetic_model::<TestBackend>(single_root, &device).unwrap();

        assert_eq!(model.shape().local_trunk.root_count, 1);
        assert_eq!(
            model.shape().local_trunk.root_state_dim * model.shape().local_trunk.root_count,
            base.total_root_state_dim
        );
        assert_eq!(
            model.shape().local_trunk.root_readout_dim * model.shape().local_trunk.root_count,
            base.total_root_readout_dim
        );
        assert_eq!(
            model.shape().router.query_dim,
            single_root.root_readout_dim()
        );
    }

    #[test]
    fn baseline_v2_fixture_rejects_uneven_root_budget_split() {
        let device = Default::default();
        let error = build_baseline_v2_synthetic_model::<TestBackend>(
            BaselineV2SyntheticModelConfig {
                total_root_state_dim: 11,
                ..BaselineV2SyntheticModelConfig::default()
            },
            &device,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message)
                if message.contains("total_root_state_dim")
        ));
    }

    #[test]
    fn baseline_v2_fixture_supports_nondefault_leaf_size_for_exploratory_benchmarks() {
        let device = Default::default();
        let model = build_baseline_v2_synthetic_model::<TestBackend>(
            BaselineV2SyntheticModelConfig::default().with_leaf_size(32),
            &device,
        )
        .unwrap();

        assert_eq!(model.shape().local_trunk.leaf_size, 32);
        assert_eq!(model.shape().leaf_summarizer.leaf_size, 32);
        assert_eq!(model.shape().exact_read.leaf_size, 32);
    }
}
