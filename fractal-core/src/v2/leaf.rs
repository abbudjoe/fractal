use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    tensor::{backend::Backend, Bool, Int, Tensor},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

const LEAF_INIT_MIN: f64 = -0.08;
const LEAF_INIT_MAX: f64 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafSummarizerShape {
    pub readout_dim: usize,
    pub leaf_size: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub token_cache_key_dim: usize,
    pub token_cache_value_dim: usize,
}

type LeafSummarizerParts<B> = (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 2, Bool>,
);

#[derive(Debug, Clone)]
pub struct LeafSummarizerOutput<B: Backend> {
    summary: Tensor<B, 2>,
    key: Tensor<B, 2>,
    value: Tensor<B, 2>,
    token_keys: Tensor<B, 3>,
    token_values: Tensor<B, 3>,
    token_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> LeafSummarizerOutput<B> {
    pub(crate) fn new(
        summary: Tensor<B, 2>,
        key: Tensor<B, 2>,
        value: Tensor<B, 2>,
        token_keys: Tensor<B, 3>,
        token_values: Tensor<B, 3>,
        token_mask: Tensor<B, 2, Bool>,
    ) -> Self {
        Self {
            summary,
            key,
            value,
            token_keys,
            token_values,
            token_mask,
        }
    }

    pub fn summary(&self) -> Tensor<B, 2> {
        self.summary.clone()
    }

    pub fn key(&self) -> Tensor<B, 2> {
        self.key.clone()
    }

    pub fn value(&self) -> Tensor<B, 2> {
        self.value.clone()
    }

    pub fn token_keys(&self) -> Tensor<B, 3> {
        self.token_keys.clone()
    }

    pub fn token_values(&self) -> Tensor<B, 3> {
        self.token_values.clone()
    }

    pub fn token_mask(&self) -> Tensor<B, 2, Bool> {
        self.token_mask.clone()
    }

    pub(crate) fn into_parts(self) -> LeafSummarizerParts<B> {
        (
            self.summary,
            self.key,
            self.value,
            self.token_keys,
            self.token_values,
            self.token_mask,
        )
    }
}

pub trait LeafSummarizer<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> LeafSummarizerShape;

    fn summarize_sealed_leaf(
        &self,
        token_readouts: Tensor<B, 4>,
    ) -> Result<LeafSummarizerOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineLeafSummarizerConfig {
    pub readout_dim: usize,
    pub leaf_size: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub token_cache_key_dim: usize,
    pub token_cache_value_dim: usize,
}

#[derive(Module, Debug)]
pub struct BaselineLeafSummarizer<B: Backend> {
    summary: StructuredProjection<B>,
    key: StructuredProjection<B>,
    value: StructuredProjection<B>,
    token_key: StructuredProjection<B>,
    token_value: StructuredProjection<B>,
    readout_dim: usize,
    leaf_size: usize,
    summary_dim: usize,
    key_dim: usize,
    value_dim: usize,
    token_cache_key_dim: usize,
    token_cache_value_dim: usize,
}

impl BaselineLeafSummarizerConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("baseline_leaf_summarizer.readout_dim", self.readout_dim)?;
        ensure_nonzero("baseline_leaf_summarizer.leaf_size", self.leaf_size)?;
        ensure_nonzero("baseline_leaf_summarizer.summary_dim", self.summary_dim)?;
        ensure_nonzero("baseline_leaf_summarizer.key_dim", self.key_dim)?;
        ensure_nonzero("baseline_leaf_summarizer.value_dim", self.value_dim)?;
        ensure_nonzero(
            "baseline_leaf_summarizer.token_cache_key_dim",
            self.token_cache_key_dim,
        )?;
        ensure_nonzero(
            "baseline_leaf_summarizer.token_cache_value_dim",
            self.token_cache_value_dim,
        )?;
        if self.leaf_size != 16 {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_leaf_summarizer.leaf_size must equal 16 in phase 4, got {}",
                self.leaf_size
            )));
        }

        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineLeafSummarizer<B> {
        self.try_init(device).unwrap_or_else(|error| {
            panic!("invalid baseline leaf summarizer config: {error}");
        })
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineLeafSummarizer<B>, FractalError> {
        self.validate()?;
        let initializer = Initializer::Uniform {
            min: LEAF_INIT_MIN,
            max: LEAF_INIT_MAX,
        };
        let projection = |d_input, d_output| -> StructuredProjection<B> {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(initializer.clone())
                .init(device)
        };

        Ok(BaselineLeafSummarizer {
            summary: projection(self.readout_dim, self.summary_dim),
            key: projection(self.readout_dim, self.key_dim),
            value: projection(self.readout_dim, self.value_dim),
            token_key: projection(self.readout_dim, self.token_cache_key_dim),
            token_value: projection(self.readout_dim, self.token_cache_value_dim),
            readout_dim: self.readout_dim,
            leaf_size: self.leaf_size,
            summary_dim: self.summary_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            token_cache_key_dim: self.token_cache_key_dim,
            token_cache_value_dim: self.token_cache_value_dim,
        })
    }
}

impl<B: Backend> LeafSummarizer<B> for BaselineLeafSummarizer<B> {
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
    ) -> Result<LeafSummarizerOutput<B>, FractalError> {
        let [batch_size, root_count, leaf_size, readout_dim] = token_readouts.dims();
        ensure_nonzero("leaf_summarizer.batch_size", batch_size)?;
        ensure_nonzero("leaf_summarizer.root_count", root_count)?;
        ensure_match("leaf_summarizer.leaf_size", leaf_size, self.leaf_size)?;
        ensure_match("leaf_summarizer.readout_dim", readout_dim, self.readout_dim)?;

        let per_token_readouts = token_readouts
            .sum_dim(1)
            .mul_scalar(1.0 / root_count as f64)
            .reshape([batch_size, leaf_size, self.readout_dim]);
        let leaf_context = per_token_readouts
            .clone()
            .sum_dim(1)
            .mul_scalar(1.0 / leaf_size as f64)
            .reshape([batch_size, self.readout_dim]);
        let token_flat = per_token_readouts
            .clone()
            .reshape([batch_size * leaf_size, self.readout_dim]);

        Ok(LeafSummarizerOutput::new(
            self.summary.forward(leaf_context.clone()),
            self.key.forward(leaf_context.clone()),
            self.value.forward(leaf_context),
            self.token_key.forward(token_flat.clone()).reshape([
                batch_size,
                leaf_size,
                self.token_cache_key_dim,
            ]),
            self.token_value.forward(token_flat).reshape([
                batch_size,
                leaf_size,
                self.token_cache_value_dim,
            ]),
            full_true_mask([batch_size, leaf_size], &per_token_readouts.device()),
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

fn full_true_mask<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Bool> {
    Tensor::<B, D, Int>::ones(shape, device).greater_elem(0)
}

#[cfg(test)]
mod tests {
    use burn::{backend::Candle, tensor::TensorData};

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn test_input(device: &<TestBackend as Backend>::Device) -> Tensor<TestBackend, 4> {
        Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [
                    [[1.0, 0.0, 2.0], [2.0, 1.0, 0.0]],
                    [[3.0, 1.0, 1.0], [0.0, 2.0, 1.0]],
                ],
                [
                    [[0.5, 1.5, 2.5], [1.5, 0.5, 1.0]],
                    [[2.5, 0.5, 1.5], [3.5, 2.5, 0.5]],
                ],
            ]),
            device,
        )
    }

    #[test]
    fn baseline_leaf_summarizer_produces_expected_shapes() {
        let device = Default::default();
        let summarizer = BaselineLeafSummarizerConfig::new(3, 2, 5, 4, 6, 7, 8)
            .try_init::<TestBackend>(&device)
            .unwrap_err();
        assert!(
            matches!(summarizer, FractalError::InvalidConfig(message) if message.contains("leaf_size must equal 16"))
        );
    }

    #[test]
    fn baseline_leaf_summarizer_is_deterministic_for_same_input() {
        let device = Default::default();
        let summarizer =
            BaselineLeafSummarizerConfig::new(3, 16, 5, 4, 6, 7, 8).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 4>::zeros([2, 2, 16, 3], &device)
            .slice_assign([0..2, 0..2, 0..2, 0..3], test_input(&device));

        let first = summarizer.summarize_sealed_leaf(input.clone()).unwrap();
        let second = summarizer.summarize_sealed_leaf(input).unwrap();

        assert_eq!(
            first.summary().to_data().convert::<f32>(),
            second.summary().to_data().convert::<f32>()
        );
        assert_eq!(
            first.key().to_data().convert::<f32>(),
            second.key().to_data().convert::<f32>()
        );
        assert_eq!(
            first.value().to_data().convert::<f32>(),
            second.value().to_data().convert::<f32>()
        );
        assert_eq!(
            first.token_keys().to_data().convert::<f32>(),
            second.token_keys().to_data().convert::<f32>()
        );
        assert_eq!(
            first.token_values().to_data().convert::<f32>(),
            second.token_values().to_data().convert::<f32>()
        );
        assert_eq!(
            first.token_mask().to_data().convert::<bool>(),
            second.token_mask().to_data().convert::<bool>()
        );
    }

    #[test]
    fn baseline_leaf_summarizer_masks_every_token_in_a_sealed_leaf() {
        let device = Default::default();
        let summarizer =
            BaselineLeafSummarizerConfig::new(3, 16, 5, 4, 6, 7, 8).init::<TestBackend>(&device);
        let base = test_input(&device);
        let zeros = Tensor::<TestBackend, 4>::zeros([2, 2, 14, 3], &device);
        let input = Tensor::<TestBackend, 4>::zeros([2, 2, 16, 3], &device)
            .slice_assign([0..2, 0..2, 0..2, 0..3], base)
            .slice_assign([0..2, 0..2, 2..16, 0..3], zeros);

        let output = summarizer.summarize_sealed_leaf(input).unwrap();

        assert_eq!(output.summary().dims(), [2, 5]);
        assert_eq!(output.key().dims(), [2, 4]);
        assert_eq!(output.value().dims(), [2, 6]);
        assert_eq!(output.token_keys().dims(), [2, 16, 7]);
        assert_eq!(output.token_values().dims(), [2, 16, 8]);
        assert_eq!(output.token_mask().dims(), [2, 16]);
        assert_eq!(
            output
                .token_mask()
                .to_data()
                .convert::<bool>()
                .into_vec::<bool>()
                .unwrap(),
            vec![true; 32]
        );
    }
}
