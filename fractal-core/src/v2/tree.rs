use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    tensor::{backend::Backend, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

const TREE_INIT_MIN: f64 = -0.08;
const TREE_INIT_MAX: f64 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeMergeCellShape {
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub scale_embedding_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeSummaryDiagnostics {
    pub nodes_per_level: Vec<usize>,
    pub tree_depth_reached: usize,
    pub has_dead_or_unused_nodes: bool,
}

#[derive(Debug, Clone)]
pub struct TreeNodeBatch<B: Backend> {
    summary: Tensor<B, 2>,
    key: Tensor<B, 2>,
    value: Tensor<B, 2>,
}

impl<B: Backend> TreeNodeBatch<B> {
    pub fn from_tensors(
        summary: Tensor<B, 2>,
        key: Tensor<B, 2>,
        value: Tensor<B, 2>,
    ) -> Result<Self, FractalError> {
        let [batch_size, summary_dim] = summary.dims();
        let [key_batch_size, key_dim] = key.dims();
        let [value_batch_size, value_dim] = value.dims();
        ensure_nonzero("tree_node.batch_size", batch_size)?;
        ensure_nonzero("tree_node.summary_dim", summary_dim)?;
        ensure_nonzero("tree_node.key_dim", key_dim)?;
        ensure_nonzero("tree_node.value_dim", value_dim)?;
        ensure_match("tree_node.key_batch_size", key_batch_size, batch_size)?;
        ensure_match("tree_node.value_batch_size", value_batch_size, batch_size)?;

        Ok(Self {
            summary,
            key,
            value,
        })
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
}

#[derive(Debug, Clone)]
pub struct TreeMergeOutput<B: Backend> {
    summary: Tensor<B, 2>,
    key: Tensor<B, 2>,
    value: Tensor<B, 2>,
}

impl<B: Backend> TreeMergeOutput<B> {
    pub fn new(summary: Tensor<B, 2>, key: Tensor<B, 2>, value: Tensor<B, 2>) -> Self {
        Self {
            summary,
            key,
            value,
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

    pub fn into_node(self) -> Result<TreeNodeBatch<B>, FractalError> {
        TreeNodeBatch::from_tensors(self.summary, self.key, self.value)
    }
}

pub trait TreeMergeCell<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> TreeMergeCellShape;

    fn merge_pair(
        &self,
        left: TreeNodeBatch<B>,
        right: TreeNodeBatch<B>,
        level: usize,
    ) -> Result<TreeMergeOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineTreeMergeCellConfig {
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub scale_embedding_dim: usize,
}

#[derive(Module, Debug)]
pub struct BaselineTreeMergeCell<B: Backend> {
    summary: StructuredProjection<B>,
    key: StructuredProjection<B>,
    value: StructuredProjection<B>,
    summary_dim: usize,
    key_dim: usize,
    value_dim: usize,
    scale_embedding_dim: usize,
}

impl BaselineTreeMergeCellConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("baseline_tree_merge_cell.summary_dim", self.summary_dim)?;
        ensure_nonzero("baseline_tree_merge_cell.key_dim", self.key_dim)?;
        ensure_nonzero("baseline_tree_merge_cell.value_dim", self.value_dim)?;
        ensure_nonzero(
            "baseline_tree_merge_cell.scale_embedding_dim",
            self.scale_embedding_dim,
        )
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineTreeMergeCell<B> {
        self.try_init(device)
            .unwrap_or_else(|error| panic!("invalid baseline tree merge cell config: {error}"))
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineTreeMergeCell<B>, FractalError> {
        self.validate()?;
        let initializer = Initializer::Uniform {
            min: TREE_INIT_MIN,
            max: TREE_INIT_MAX,
        };
        let input_dim = self
            .summary_dim
            .checked_mul(2)
            .and_then(|value| value.checked_add(self.scale_embedding_dim))
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "baseline_tree_merge_cell input dim overflow".to_string(),
                )
            })?;
        let projection = |d_input, d_output| -> StructuredProjection<B> {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(initializer.clone())
                .init(device)
        };

        Ok(BaselineTreeMergeCell {
            summary: projection(input_dim, self.summary_dim),
            key: projection(input_dim, self.key_dim),
            value: projection(input_dim, self.value_dim),
            summary_dim: self.summary_dim,
            key_dim: self.key_dim,
            value_dim: self.value_dim,
            scale_embedding_dim: self.scale_embedding_dim,
        })
    }
}

impl<B: Backend> TreeMergeCell<B> for BaselineTreeMergeCell<B> {
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
        left: TreeNodeBatch<B>,
        right: TreeNodeBatch<B>,
        level: usize,
    ) -> Result<TreeMergeOutput<B>, FractalError> {
        let [batch_size, left_summary_dim] = left.summary().dims();
        let [right_batch_size, right_summary_dim] = right.summary().dims();
        ensure_nonzero("baseline_tree_merge_cell.batch_size", batch_size)?;
        ensure_match(
            "baseline_tree_merge_cell.right_batch_size",
            right_batch_size,
            batch_size,
        )?;
        ensure_match(
            "baseline_tree_merge_cell.left_summary_dim",
            left_summary_dim,
            self.summary_dim,
        )?;
        ensure_match(
            "baseline_tree_merge_cell.right_summary_dim",
            right_summary_dim,
            self.summary_dim,
        )?;
        let left = TreeNodeBatch::from_tensors(left.summary(), left.key(), left.value())?;
        let right = TreeNodeBatch::from_tensors(right.summary(), right.key(), right.value())?;
        ensure_match(
            "baseline_tree_merge_cell.left_key_dim",
            left.key().dims()[1],
            self.key_dim,
        )?;
        ensure_match(
            "baseline_tree_merge_cell.right_key_dim",
            right.key().dims()[1],
            self.key_dim,
        )?;
        ensure_match(
            "baseline_tree_merge_cell.left_value_dim",
            left.value().dims()[1],
            self.value_dim,
        )?;
        ensure_match(
            "baseline_tree_merge_cell.right_value_dim",
            right.value().dims()[1],
            self.value_dim,
        )?;

        let mean_summary = (left.summary() + right.summary()).mul_scalar(0.5);
        let detail_summary = right.summary() - left.summary();
        let scale = scale_embedding(
            batch_size,
            level,
            self.scale_embedding_dim,
            &mean_summary.device(),
        );
        let context = Tensor::cat(vec![mean_summary, detail_summary, scale], 1);

        Ok(TreeMergeOutput::new(
            self.summary.forward(context.clone()).tanh(),
            self.key.forward(context.clone()),
            self.value.forward(context),
        ))
    }
}

fn scale_embedding<B: Backend>(
    batch_size: usize,
    level: usize,
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(batch_size * dim);
    let level = level as f32 + 1.0;
    let denom_base = 10_000.0f32;
    let dim_f = dim as f32;

    for _ in 0..batch_size {
        for index in 0..dim {
            let frequency = denom_base.powf((2 * (index / 2)) as f32 / dim_f.max(1.0));
            let angle = level / frequency;
            values.push(if index % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            });
        }
    }

    Tensor::<B, 2>::from_data(TensorData::new(values, [batch_size, dim]), device)
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
    use burn::backend::Candle;

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn test_node(
        batch_size: usize,
        summary_dim: usize,
        key_dim: usize,
        value_dim: usize,
        device: &<TestBackend as Backend>::Device,
    ) -> TreeNodeBatch<TestBackend> {
        TreeNodeBatch::from_tensors(
            Tensor::<TestBackend, 2>::zeros([batch_size, summary_dim], device),
            Tensor::<TestBackend, 2>::zeros([batch_size, key_dim], device),
            Tensor::<TestBackend, 2>::zeros([batch_size, value_dim], device),
        )
        .unwrap()
    }

    #[test]
    fn baseline_tree_merge_cell_rejects_zero_scale_embedding_dim() {
        let device = <TestBackend as Backend>::Device::default();
        let error = BaselineTreeMergeCellConfig::new(8, 6, 10, 0)
            .try_init::<TestBackend>(&device)
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("scale_embedding_dim"))
        );
    }

    #[test]
    fn baseline_tree_merge_cell_merge_pair_produces_expected_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let cell = BaselineTreeMergeCellConfig::new(8, 6, 10, 4).init::<TestBackend>(&device);
        let output = cell
            .merge_pair(
                test_node(3, 8, 6, 10, &device),
                test_node(3, 8, 6, 10, &device),
                1,
            )
            .unwrap();

        assert_eq!(output.summary().dims(), [3, 8]);
        assert_eq!(output.key().dims(), [3, 6]);
        assert_eq!(output.value().dims(), [3, 10]);
    }

    #[test]
    fn baseline_tree_merge_cell_is_deterministic_for_same_inputs() {
        let device = <TestBackend as Backend>::Device::default();
        let cell = BaselineTreeMergeCellConfig::new(8, 6, 10, 4).init::<TestBackend>(&device);
        let left = test_node(2, 8, 6, 10, &device);
        let right = test_node(2, 8, 6, 10, &device);

        let first = cell.merge_pair(left.clone(), right.clone(), 2).unwrap();
        let second = cell.merge_pair(left, right, 2).unwrap();

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
    }
}
