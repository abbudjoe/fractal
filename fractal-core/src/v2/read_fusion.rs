use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    tensor::{backend::Backend, Bool, Tensor},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

const READ_FUSION_INIT_MIN: f64 = -0.08;
const READ_FUSION_INIT_MAX: f64 = 0.08;

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadFusionShape {
    pub root_count: usize,
    pub root_readout_dim: usize,
    pub routed_value_dim: usize,
    pub exact_read_value_dim: usize,
    pub fused_readout_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadFusionAblation {
    include_routed_values: bool,
    include_exact_read_values: bool,
    active_root_count: Option<usize>,
}

impl ReadFusionAblation {
    pub const fn full() -> Self {
        Self {
            include_routed_values: true,
            include_exact_read_values: true,
            active_root_count: None,
        }
    }

    pub const fn without_routed_values() -> Self {
        Self {
            include_routed_values: false,
            ..Self::full()
        }
    }

    pub const fn without_exact_read_values() -> Self {
        Self {
            include_exact_read_values: false,
            ..Self::full()
        }
    }

    pub const fn with_active_root_count(active_root_count: usize) -> Self {
        Self {
            include_routed_values: true,
            include_exact_read_values: true,
            active_root_count: Some(active_root_count),
        }
    }

    pub const fn include_routed_values(&self) -> bool {
        self.include_routed_values
    }

    pub const fn include_exact_read_values(&self) -> bool {
        self.include_exact_read_values
    }

    pub const fn active_root_count(&self) -> Option<usize> {
        self.active_root_count
    }
}

impl Default for ReadFusionAblation {
    fn default() -> Self {
        Self::full()
    }
}

#[derive(Debug, Clone)]
pub struct ReadFusionInput<B: Backend> {
    root_readouts: Tensor<B, 3>,
    routed_values: Tensor<B, 4>,
    routed_scores: Tensor<B, 3>,
    routed_mask: Tensor<B, 3, Bool>,
    exact_read_values: Tensor<B, 4>,
    exact_read_mask: Tensor<B, 3, Bool>,
}

impl<B: Backend> ReadFusionInput<B> {
    pub fn new(
        root_readouts: Tensor<B, 3>,
        routed_values: Tensor<B, 4>,
        routed_scores: Tensor<B, 3>,
        routed_mask: Tensor<B, 3, Bool>,
        exact_read_values: Tensor<B, 4>,
        exact_read_mask: Tensor<B, 3, Bool>,
    ) -> Result<Self, FractalError> {
        let [batch_size, root_count, root_readout_dim] = root_readouts.dims();
        ensure_nonzero("read_fusion_input.batch_size", batch_size)?;
        ensure_nonzero("read_fusion_input.root_count", root_count)?;
        ensure_nonzero("read_fusion_input.root_readout_dim", root_readout_dim)?;

        let [routed_batch_size, head_count, top_leaf_reads, routed_value_dim] =
            routed_values.dims();
        ensure_match(
            "read_fusion_input.routed_values.batch_size",
            routed_batch_size,
            batch_size,
        )?;
        ensure_nonzero("read_fusion_input.head_count", head_count)?;
        ensure_nonzero("read_fusion_input.top_leaf_reads", top_leaf_reads)?;
        ensure_nonzero("read_fusion_input.routed_value_dim", routed_value_dim)?;
        ensure_dims3(
            "read_fusion_input.routed_scores",
            routed_scores.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;
        ensure_dims3(
            "read_fusion_input.routed_mask",
            routed_mask.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;

        let [exact_batch_size, exact_head_count, exact_top_leaf_reads, exact_read_value_dim] =
            exact_read_values.dims();
        ensure_match(
            "read_fusion_input.exact_read_values.batch_size",
            exact_batch_size,
            batch_size,
        )?;
        ensure_match(
            "read_fusion_input.exact_read_values.head_count",
            exact_head_count,
            head_count,
        )?;
        ensure_match(
            "read_fusion_input.exact_read_values.top_leaf_reads",
            exact_top_leaf_reads,
            top_leaf_reads,
        )?;
        ensure_nonzero(
            "read_fusion_input.exact_read_value_dim",
            exact_read_value_dim,
        )?;
        ensure_dims3(
            "read_fusion_input.exact_read_mask",
            exact_read_mask.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;

        Ok(Self {
            root_readouts,
            routed_values,
            routed_scores,
            routed_mask,
            exact_read_values,
            exact_read_mask,
        })
    }

    pub fn root_readouts(&self) -> Tensor<B, 3> {
        self.root_readouts.clone()
    }

    pub fn routed_values(&self) -> Tensor<B, 4> {
        self.routed_values.clone()
    }

    pub fn routed_scores(&self) -> Tensor<B, 3> {
        self.routed_scores.clone()
    }

    pub fn routed_mask(&self) -> Tensor<B, 3, Bool> {
        self.routed_mask.clone()
    }

    pub fn exact_read_values(&self) -> Tensor<B, 4> {
        self.exact_read_values.clone()
    }

    pub fn exact_read_mask(&self) -> Tensor<B, 3, Bool> {
        self.exact_read_mask.clone()
    }
}

#[derive(Debug, Clone)]
pub struct ReadFusionOutput<B: Backend> {
    fused_readout: Tensor<B, 2>,
    root_lane: Tensor<B, 2>,
    routed_lane: Tensor<B, 2>,
    exact_read_lane: Tensor<B, 2>,
    root_summary: Tensor<B, 2>,
    routed_summary: Tensor<B, 2>,
    exact_read_summary: Tensor<B, 2>,
}

impl<B: Backend> ReadFusionOutput<B> {
    pub fn new(
        fused_readout: Tensor<B, 2>,
        root_lane: Tensor<B, 2>,
        routed_lane: Tensor<B, 2>,
        exact_read_lane: Tensor<B, 2>,
        root_summary: Tensor<B, 2>,
        routed_summary: Tensor<B, 2>,
        exact_read_summary: Tensor<B, 2>,
    ) -> Result<Self, FractalError> {
        let [batch_size, fused_readout_dim] = fused_readout.dims();
        ensure_nonzero("read_fusion_output.batch_size", batch_size)?;
        ensure_nonzero("read_fusion_output.fused_readout_dim", fused_readout_dim)?;
        ensure_dims2(
            "read_fusion_output.root_lane",
            root_lane.dims(),
            [batch_size, fused_readout_dim],
        )?;
        ensure_dims2(
            "read_fusion_output.routed_lane",
            routed_lane.dims(),
            [batch_size, fused_readout_dim],
        )?;
        ensure_dims2(
            "read_fusion_output.exact_read_lane",
            exact_read_lane.dims(),
            [batch_size, fused_readout_dim],
        )?;
        let [root_summary_batch_size, root_summary_dim] = root_summary.dims();
        ensure_match(
            "read_fusion_output.root_summary.batch_size",
            root_summary_batch_size,
            batch_size,
        )?;
        ensure_nonzero(
            "read_fusion_output.root_summary.root_readout_dim",
            root_summary_dim,
        )?;
        let [routed_summary_batch_size, routed_summary_dim] = routed_summary.dims();
        ensure_match(
            "read_fusion_output.routed_summary.batch_size",
            routed_summary_batch_size,
            batch_size,
        )?;
        ensure_nonzero(
            "read_fusion_output.routed_summary.routed_value_dim",
            routed_summary_dim,
        )?;
        let [exact_summary_batch_size, exact_summary_dim] = exact_read_summary.dims();
        ensure_match(
            "read_fusion_output.exact_read_summary.batch_size",
            exact_summary_batch_size,
            batch_size,
        )?;
        ensure_nonzero(
            "read_fusion_output.exact_read_summary.exact_read_value_dim",
            exact_summary_dim,
        )?;

        Ok(Self {
            fused_readout,
            root_lane,
            routed_lane,
            exact_read_lane,
            root_summary,
            routed_summary,
            exact_read_summary,
        })
    }

    pub fn fused_readout(&self) -> Tensor<B, 2> {
        self.fused_readout.clone()
    }

    pub fn root_lane(&self) -> Tensor<B, 2> {
        self.root_lane.clone()
    }

    pub fn routed_lane(&self) -> Tensor<B, 2> {
        self.routed_lane.clone()
    }

    pub fn exact_read_lane(&self) -> Tensor<B, 2> {
        self.exact_read_lane.clone()
    }

    pub fn root_summary(&self) -> Tensor<B, 2> {
        self.root_summary.clone()
    }

    pub fn routed_summary(&self) -> Tensor<B, 2> {
        self.routed_summary.clone()
    }

    pub fn exact_read_summary(&self) -> Tensor<B, 2> {
        self.exact_read_summary.clone()
    }
}

pub trait ReadFusion<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> ReadFusionShape;

    fn fuse(
        &self,
        input: &ReadFusionInput<B>,
        ablation: ReadFusionAblation,
    ) -> Result<ReadFusionOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineReadFusionConfig {
    pub root_count: usize,
    pub root_readout_dim: usize,
    pub routed_value_dim: usize,
    pub exact_read_value_dim: usize,
    pub fused_readout_dim: usize,
    #[config(
        default = "Initializer::Uniform { min: READ_FUSION_INIT_MIN, max: READ_FUSION_INIT_MAX }"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct BaselineReadFusion<B: Backend> {
    root_projection: StructuredProjection<B>,
    routed_projection: StructuredProjection<B>,
    exact_read_projection: StructuredProjection<B>,
    shape: ReadFusionShape,
}

impl BaselineReadFusionConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("baseline_read_fusion.root_count", self.root_count)?;
        ensure_nonzero(
            "baseline_read_fusion.root_readout_dim",
            self.root_readout_dim,
        )?;
        ensure_nonzero(
            "baseline_read_fusion.routed_value_dim",
            self.routed_value_dim,
        )?;
        ensure_nonzero(
            "baseline_read_fusion.exact_read_value_dim",
            self.exact_read_value_dim,
        )?;
        ensure_nonzero(
            "baseline_read_fusion.fused_readout_dim",
            self.fused_readout_dim,
        )
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineReadFusion<B> {
        self.try_init(device)
            .unwrap_or_else(|error| panic!("invalid baseline read fusion config: {error}"))
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineReadFusion<B>, FractalError> {
        self.validate()?;
        let root_projection =
            StructuredProjectionConfig::new(self.root_readout_dim, self.fused_readout_dim)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_bias(false)
                .with_initializer(self.initializer.clone())
                .init(device);
        let routed_projection =
            StructuredProjectionConfig::new(self.routed_value_dim, self.fused_readout_dim)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_bias(false)
                .with_initializer(self.initializer.clone())
                .init(device);
        let exact_read_projection =
            StructuredProjectionConfig::new(self.exact_read_value_dim, self.fused_readout_dim)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_bias(false)
                .with_initializer(self.initializer.clone())
                .init(device);

        Ok(BaselineReadFusion {
            root_projection,
            routed_projection,
            exact_read_projection,
            shape: ReadFusionShape {
                root_count: self.root_count,
                root_readout_dim: self.root_readout_dim,
                routed_value_dim: self.routed_value_dim,
                exact_read_value_dim: self.exact_read_value_dim,
                fused_readout_dim: self.fused_readout_dim,
            },
        })
    }
}

impl<B: Backend> BaselineReadFusion<B> {
    fn summarize_roots(
        &self,
        root_readouts: Tensor<B, 3>,
        active_root_count: usize,
    ) -> Result<Tensor<B, 2>, FractalError> {
        let [batch_size, root_count, root_readout_dim] = root_readouts.dims();
        ensure_match(
            "baseline_read_fusion.root_readouts.root_count",
            root_count,
            self.shape.root_count,
        )?;
        ensure_match(
            "baseline_read_fusion.root_readouts.root_readout_dim",
            root_readout_dim,
            self.shape.root_readout_dim,
        )?;
        if active_root_count == 0 || active_root_count > root_count {
            return Err(FractalError::InvalidConfig(format!(
                "baseline_read_fusion.active_root_count must be within 1..={root_count}, got {active_root_count}"
            )));
        }

        Ok(root_readouts
            .narrow(1, 0, active_root_count)
            .sum_dim(1)
            .mul_scalar(1.0 / active_root_count as f64)
            .reshape([batch_size, self.shape.root_readout_dim]))
    }

    fn summarize_weighted_reads(
        &self,
        lane_name: &str,
        values: Tensor<B, 4>,
        scores: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
        expected_value_dim: usize,
    ) -> Result<Tensor<B, 2>, FractalError> {
        let [batch_size, head_count, top_leaf_reads, value_dim] = values.dims();
        ensure_nonzero(&format!("{lane_name}.batch_size"), batch_size)?;
        ensure_nonzero(&format!("{lane_name}.head_count"), head_count)?;
        ensure_nonzero(&format!("{lane_name}.top_leaf_reads"), top_leaf_reads)?;
        ensure_match(
            &format!("{lane_name}.value_dim"),
            value_dim,
            expected_value_dim,
        )?;
        ensure_dims3(
            &format!("{lane_name}.scores"),
            scores.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;
        ensure_dims3(
            &format!("{lane_name}.mask"),
            mask.dims(),
            [batch_size, head_count, top_leaf_reads],
        )?;

        let device = values.device();
        let mask = mask
            .reshape([batch_size, head_count, top_leaf_reads, 1])
            .repeat(&[1, 1, 1, value_dim]);
        let masked_values =
            Tensor::<B, 4>::zeros([batch_size, head_count, top_leaf_reads, value_dim], &device)
                .mask_where(mask, values);
        let score_broadcast = scores
            .reshape([batch_size, head_count, top_leaf_reads, 1])
            .repeat(&[1, 1, 1, value_dim]);

        Ok((masked_values * score_broadcast)
            .sum_dim(2)
            .sum_dim(1)
            .mul_scalar(1.0 / (head_count * top_leaf_reads) as f64)
            .reshape([batch_size, value_dim]))
    }
}

impl<B: Backend> ReadFusion<B> for BaselineReadFusion<B> {
    fn shape(&self) -> ReadFusionShape {
        self.shape
    }

    fn fuse(
        &self,
        input: &ReadFusionInput<B>,
        ablation: ReadFusionAblation,
    ) -> Result<ReadFusionOutput<B>, FractalError> {
        let [batch_size, root_count, _root_readout_dim] = input.root_readouts().dims();
        let active_root_count = ablation.active_root_count.unwrap_or(root_count);
        let root_summary = self.summarize_roots(input.root_readouts(), active_root_count)?;
        let routed_summary = if ablation.include_routed_values {
            self.summarize_weighted_reads(
                "baseline_read_fusion.routed_values",
                input.routed_values(),
                input.routed_scores(),
                input.routed_mask(),
                self.shape.routed_value_dim,
            )?
        } else {
            Tensor::<B, 2>::zeros(
                [batch_size, self.shape.routed_value_dim],
                &root_summary.device(),
            )
        };
        let exact_read_summary = if ablation.include_exact_read_values {
            self.summarize_weighted_reads(
                "baseline_read_fusion.exact_read_values",
                input.exact_read_values(),
                input.routed_scores(),
                input.exact_read_mask(),
                self.shape.exact_read_value_dim,
            )?
        } else {
            Tensor::<B, 2>::zeros(
                [batch_size, self.shape.exact_read_value_dim],
                &root_summary.device(),
            )
        };

        let root_lane = self.root_projection.forward(root_summary.clone());
        let routed_lane = self.routed_projection.forward(routed_summary.clone());
        let exact_read_lane = self
            .exact_read_projection
            .forward(exact_read_summary.clone());
        let fused_readout = root_lane.clone() + routed_lane.clone() + exact_read_lane.clone();

        ReadFusionOutput::new(
            fused_readout,
            root_lane,
            routed_lane,
            exact_read_lane,
            root_summary,
            routed_summary,
            exact_read_summary,
        )
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

fn ensure_dims2(name: &str, actual: [usize; 2], expected: [usize; 2]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected:?}, got {actual:?}"
        )));
    }

    Ok(())
}

fn ensure_dims3(name: &str, actual: [usize; 3], expected: [usize; 3]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected:?}, got {actual:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        nn::Initializer,
        tensor::{Tensor, TensorData},
    };

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn baseline_fusion() -> BaselineReadFusion<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        BaselineReadFusionConfig {
            root_count: 2,
            root_readout_dim: 3,
            routed_value_dim: 2,
            exact_read_value_dim: 2,
            fused_readout_dim: 4,
            initializer: Initializer::Constant { value: 0.25 },
        }
        .init::<TestBackend>(&device)
    }

    fn fusion_input() -> ReadFusionInput<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        ReadFusionInput::new(
            Tensor::<TestBackend, 3>::from_data(
                TensorData::new(
                    vec![
                        1.0, 2.0, 3.0, //
                        3.0, 4.0, 5.0,
                    ],
                    [1, 2, 3],
                ),
                &device,
            ),
            Tensor::<TestBackend, 4>::from_data(
                TensorData::new(
                    vec![
                        2.0, 4.0, //
                        1.0, 3.0,
                    ],
                    [1, 1, 2, 2],
                ),
                &device,
            ),
            Tensor::<TestBackend, 3>::from_data(
                TensorData::new(vec![1.0, 0.5], [1, 1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 3, Bool>::from_data(
                TensorData::new(vec![true, false], [1, 1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 4>::from_data(
                TensorData::new(
                    vec![
                        5.0, 1.0, //
                        7.0, 3.0,
                    ],
                    [1, 1, 2, 2],
                ),
                &device,
            ),
            Tensor::<TestBackend, 3, Bool>::from_data(
                TensorData::new(vec![true, false], [1, 1, 2]),
                &device,
            ),
        )
        .unwrap()
    }

    #[test]
    fn read_fusion_input_rejects_mismatched_exact_read_shape() {
        let device = <TestBackend as Backend>::Device::default();
        let error = ReadFusionInput::new(
            Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
            Tensor::<TestBackend, 4>::zeros([1, 1, 2, 4], &device),
            Tensor::<TestBackend, 3>::zeros([1, 1, 2], &device),
            Tensor::<TestBackend, 3, Bool>::ones([1, 1, 2], &device),
            Tensor::<TestBackend, 4>::zeros([1, 2, 2, 4], &device),
            Tensor::<TestBackend, 3, Bool>::ones([1, 2, 2], &device),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("read_fusion_input.exact_read_values.head_count"))
        );
    }

    #[test]
    fn baseline_read_fusion_keeps_source_lanes_explicit() {
        let fusion = baseline_fusion();
        let output = fusion
            .fuse(&fusion_input(), ReadFusionAblation::full())
            .unwrap();

        assert_eq!(output.fused_readout().dims(), [1, 4]);
        assert_eq!(output.root_lane().dims(), [1, 4]);
        assert_eq!(output.routed_lane().dims(), [1, 4]);
        assert_eq!(output.exact_read_lane().dims(), [1, 4]);
        assert_eq!(output.root_summary().dims(), [1, 3]);
        assert_eq!(output.routed_summary().dims(), [1, 2]);
        assert_eq!(output.exact_read_summary().dims(), [1, 2]);
    }

    #[test]
    fn baseline_read_fusion_supports_routed_and_exact_ablation() {
        let fusion = baseline_fusion();
        let input = fusion_input();
        let full = fusion.fuse(&input, ReadFusionAblation::full()).unwrap();
        let without_routed = fusion
            .fuse(&input, ReadFusionAblation::without_routed_values())
            .unwrap();
        let without_exact = fusion
            .fuse(&input, ReadFusionAblation::without_exact_read_values())
            .unwrap();

        assert!(without_routed
            .routed_lane()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
        assert!(without_exact
            .exact_read_lane()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()
            .iter()
            .all(|value| *value == 0.0));
        assert_ne!(
            full.fused_readout().to_data().convert::<f32>(),
            without_routed.fused_readout().to_data().convert::<f32>()
        );
        assert_ne!(
            full.fused_readout().to_data().convert::<f32>(),
            without_exact.fused_readout().to_data().convert::<f32>()
        );
    }

    #[test]
    fn baseline_read_fusion_supports_root_count_ablation() {
        let fusion = baseline_fusion();
        let input = fusion_input();
        let full = fusion.fuse(&input, ReadFusionAblation::full()).unwrap();
        let single_root = fusion
            .fuse(&input, ReadFusionAblation::with_active_root_count(1))
            .unwrap();

        assert_ne!(
            full.root_summary().to_data().convert::<f32>(),
            single_root.root_summary().to_data().convert::<f32>()
        );
        assert_ne!(
            full.fused_readout().to_data().convert::<f32>(),
            single_root.fused_readout().to_data().convert::<f32>()
        );
    }
}
