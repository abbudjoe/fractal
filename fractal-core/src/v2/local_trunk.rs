use burn::{
    config::Config,
    module::{Module, Param},
    nn::Initializer,
    tensor::{activation::sigmoid, backend::Backend, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::state::MultiRootState;

const CONTRACTIVE_INIT_MIN: f64 = -0.08;
const CONTRACTIVE_INIT_MAX: f64 = 0.08;

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTrunkShape {
    pub token_dim: usize,
    pub root_count: usize,
    pub root_state_dim: usize,
    pub root_readout_dim: usize,
    pub leaf_size: usize,
}

#[derive(Debug, Clone)]
pub struct LocalTrunkStepOutput<B: Backend> {
    next_state: MultiRootState<B>,
    root_readouts: Tensor<B, 3>,
}

impl<B: Backend> LocalTrunkStepOutput<B> {
    pub(crate) fn new(next_state: MultiRootState<B>, root_readouts: Tensor<B, 3>) -> Self {
        Self {
            next_state,
            root_readouts,
        }
    }

    pub fn next_state(&self) -> &MultiRootState<B> {
        &self.next_state
    }

    pub fn into_next_state(self) -> MultiRootState<B> {
        self.next_state
    }

    pub fn root_readouts(&self) -> Tensor<B, 3> {
        self.root_readouts.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootActivationStats {
    pub mean_l2_norm: f32,
    pub mean_activation: f32,
    pub activation_std: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LocalTrunkDiagnostics {
    pub per_root: Vec<RootActivationStats>,
    pub mean_pairwise_cosine_similarity: f32,
}

pub trait LocalTrunk<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> LocalTrunkShape;

    fn step(
        &self,
        token_embedding: Tensor<B, 2>,
        state: MultiRootState<B>,
    ) -> Result<LocalTrunkStepOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineLocalTrunkConfig {
    pub token_dim: usize,
    pub root_count: usize,
    pub root_state_dim: usize,
    pub root_readout_dim: usize,
    pub leaf_size: usize,
}

#[derive(Module, Debug)]
pub struct BaselineLocalTrunk<B: Backend> {
    token_drive: StructuredProjection<B>,
    token_gate: StructuredProjection<B>,
    recurrent_drive: StructuredProjection<B>,
    recurrent_gate: StructuredProjection<B>,
    readout: StructuredProjection<B>,
    root_bias: Param<Tensor<B, 2>>,
    shape: LocalTrunkShape,
}

impl BaselineLocalTrunkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineLocalTrunk<B> {
        let projection_initializer = Initializer::Uniform {
            min: CONTRACTIVE_INIT_MIN,
            max: CONTRACTIVE_INIT_MAX,
        };
        let projection = |d_input, d_output| -> StructuredProjection<B> {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(projection_initializer.clone())
                .init(device)
        };

        BaselineLocalTrunk {
            token_drive: projection(self.token_dim, self.root_state_dim),
            token_gate: projection(self.token_dim, self.root_state_dim),
            recurrent_drive: projection(self.root_state_dim, self.root_state_dim),
            recurrent_gate: projection(self.root_state_dim, self.root_state_dim),
            readout: projection(self.root_state_dim, self.root_readout_dim),
            root_bias: init_root_bias(self.root_count, self.root_state_dim, device),
            shape: LocalTrunkShape {
                token_dim: self.token_dim,
                root_count: self.root_count,
                root_state_dim: self.root_state_dim,
                root_readout_dim: self.root_readout_dim,
                leaf_size: self.leaf_size,
            },
        }
    }
}

impl<B: Backend> LocalTrunk<B> for BaselineLocalTrunk<B> {
    fn shape(&self) -> LocalTrunkShape {
        self.shape
    }

    fn step(
        &self,
        token_embedding: Tensor<B, 2>,
        state: MultiRootState<B>,
    ) -> Result<LocalTrunkStepOutput<B>, FractalError> {
        let [batch_size, token_dim] = token_embedding.dims();
        ensure_match(
            "local_trunk.token_embedding.dim",
            token_dim,
            self.shape.token_dim,
        )?;

        let state_shape = state.shape();
        ensure_match(
            "local_trunk.state.batch_size",
            state_shape.batch_size,
            batch_size,
        )?;
        ensure_match(
            "local_trunk.state.root_count",
            state_shape.root_count,
            self.shape.root_count,
        )?;
        ensure_match(
            "local_trunk.state.recurrent_dim",
            state_shape.recurrent_dim,
            self.shape.root_state_dim,
        )?;
        ensure_match(
            "local_trunk.state.intent_dim",
            state_shape.intent_dim,
            self.shape.root_readout_dim,
        )?;

        let repeated_token_drive = expand_token_projection(
            self.token_drive.forward(token_embedding.clone()),
            batch_size,
            self.shape.root_count,
            self.shape.root_state_dim,
        );
        let repeated_token_gate = expand_token_projection(
            self.token_gate.forward(token_embedding),
            batch_size,
            self.shape.root_count,
            self.shape.root_state_dim,
        );
        let root_bias = expand_root_bias(
            self.root_bias.val(),
            batch_size,
            self.shape.root_count,
            self.shape.root_state_dim,
        );
        let recurrent = state.recurrent();
        let recurrent_flat = recurrent.clone().reshape([
            batch_size * self.shape.root_count,
            self.shape.root_state_dim,
        ]);
        let recurrent_drive = self
            .recurrent_drive
            .forward(recurrent_flat.clone())
            .reshape([batch_size, self.shape.root_count, self.shape.root_state_dim]);
        let recurrent_gate = self.recurrent_gate.forward(recurrent_flat).reshape([
            batch_size,
            self.shape.root_count,
            self.shape.root_state_dim,
        ]);
        let gate = sigmoid(repeated_token_gate + recurrent_gate);
        let candidate = (repeated_token_drive + recurrent_drive + root_bias).tanh();
        let keep = gate.clone().mul_scalar(-1.0).add_scalar(1.0);
        let next_recurrent = recurrent * keep + candidate * gate;
        let root_readouts = self
            .readout
            .forward(next_recurrent.clone().reshape([
                batch_size * self.shape.root_count,
                self.shape.root_state_dim,
            ]))
            .reshape([
                batch_size,
                self.shape.root_count,
                self.shape.root_readout_dim,
            ]);
        let next_state = MultiRootState::from_tensors(
            next_recurrent,
            root_readouts.clone(),
            root_readouts.clone(),
        )?;

        Ok(LocalTrunkStepOutput::new(next_state, root_readouts))
    }
}

pub fn summarize_root_readout_sequence<B: Backend>(
    root_readouts: Tensor<B, 4>,
) -> Result<LocalTrunkDiagnostics, FractalError> {
    let [batch_size, seq_len, root_count, readout_dim] = root_readouts.dims();
    ensure_nonzero("local_trunk.root_readouts.batch_size", batch_size)?;
    ensure_nonzero("local_trunk.root_readouts.seq_len", seq_len)?;
    ensure_nonzero("local_trunk.root_readouts.root_count", root_count)?;
    ensure_nonzero("local_trunk.root_readouts.readout_dim", readout_dim)?;

    let values = root_readouts
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "local_trunk root readouts could not be inspected for diagnostics: {error}"
            ))
        })?;
    let sample_count = batch_size * seq_len;
    let mut per_root = Vec::with_capacity(root_count);

    for root_index in 0..root_count {
        let mut norm_sum = 0.0f32;
        let mut activation_sum = 0.0f64;
        let mut activation_sq_sum = 0.0f64;
        let mut activation_count = 0usize;

        for batch_index in 0..batch_size {
            for step_index in 0..seq_len {
                let mut norm_sq = 0.0f32;
                for readout_index in 0..readout_dim {
                    let value = values[flat_root_readout_index(
                        batch_index,
                        step_index,
                        root_index,
                        readout_index,
                        seq_len,
                        root_count,
                        readout_dim,
                    )];
                    norm_sq += value * value;
                    activation_sum += f64::from(value);
                    activation_sq_sum += f64::from(value * value);
                    activation_count += 1;
                }
                norm_sum += (norm_sq + 1.0e-12).sqrt();
            }
        }

        let mean_activation = (activation_sum / activation_count as f64) as f32;
        let mean_square = activation_sq_sum / activation_count as f64;
        let variance = (mean_square - f64::from(mean_activation).powi(2)).max(0.0) as f32;
        per_root.push(RootActivationStats {
            mean_l2_norm: norm_sum / sample_count as f32,
            mean_activation,
            activation_std: variance.sqrt(),
        });
    }

    let mean_pairwise_cosine_similarity = if root_count < 2 {
        0.0
    } else {
        let mut pairwise_sum = 0.0f32;
        let mut pairwise_count = 0usize;
        for batch_index in 0..batch_size {
            for step_index in 0..seq_len {
                for left_root in 0..root_count {
                    for right_root in (left_root + 1)..root_count {
                        let mut dot = 0.0f32;
                        let mut left_norm_sq = 0.0f32;
                        let mut right_norm_sq = 0.0f32;
                        for readout_index in 0..readout_dim {
                            let left = values[flat_root_readout_index(
                                batch_index,
                                step_index,
                                left_root,
                                readout_index,
                                seq_len,
                                root_count,
                                readout_dim,
                            )];
                            let right = values[flat_root_readout_index(
                                batch_index,
                                step_index,
                                right_root,
                                readout_index,
                                seq_len,
                                root_count,
                                readout_dim,
                            )];
                            dot += left * right;
                            left_norm_sq += left * left;
                            right_norm_sq += right * right;
                        }
                        let denom = ((left_norm_sq + 1.0e-12).sqrt()
                            * (right_norm_sq + 1.0e-12).sqrt())
                        .max(1.0e-12);
                        pairwise_sum += dot / denom;
                        pairwise_count += 1;
                    }
                }
            }
        }
        pairwise_sum / pairwise_count as f32
    };

    Ok(LocalTrunkDiagnostics {
        per_root,
        mean_pairwise_cosine_similarity,
    })
}

fn expand_token_projection<B: Backend>(
    projection: Tensor<B, 2>,
    batch_size: usize,
    root_count: usize,
    state_dim: usize,
) -> Tensor<B, 3> {
    projection
        .reshape([batch_size, 1, state_dim])
        .repeat(&[1, root_count, 1])
}

fn expand_root_bias<B: Backend>(
    root_bias: Tensor<B, 2>,
    batch_size: usize,
    root_count: usize,
    state_dim: usize,
) -> Tensor<B, 3> {
    root_bias
        .reshape([1, root_count, state_dim])
        .repeat(&[batch_size, 1, 1])
}

fn init_root_bias<B: Backend>(
    root_count: usize,
    state_dim: usize,
    device: &B::Device,
) -> Param<Tensor<B, 2>> {
    let center = (root_count.saturating_sub(1)) as f32 / 2.0;
    let mut values = Vec::with_capacity(root_count * state_dim);
    for root_index in 0..root_count {
        let root_offset = (root_index as f32 - center) * 0.05;
        for dim_index in 0..state_dim {
            let sign = if dim_index % 2 == 0 { 1.0 } else { -1.0 };
            values.push(root_offset * sign);
        }
    }

    Param::from_data(TensorData::new(values, [root_count, state_dim]), device)
}

fn flat_root_readout_index(
    batch_index: usize,
    step_index: usize,
    root_index: usize,
    readout_index: usize,
    seq_len: usize,
    root_count: usize,
    readout_dim: usize,
) -> usize {
    (((batch_index * seq_len + step_index) * root_count + root_index) * readout_dim) + readout_index
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
    use burn::{backend::Candle, tensor::TensorData};

    use super::*;
    use crate::{registry::CpuTrainBackend, v2::state::MultiRootState};

    type TestBackend = Candle<f32, i64>;

    fn baseline_trunk<B: Backend>() -> BaselineLocalTrunk<B> {
        let device = Default::default();
        BaselineLocalTrunkConfig::new(8, 2, 6, 4, 16).init::<B>(&device)
    }

    #[test]
    fn baseline_local_trunk_step_updates_roots_with_expected_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let trunk = baseline_trunk::<TestBackend>();
        let token_embedding =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![0.5f32; 16], [2, 8]), &device);
        let state = MultiRootState::from_tensors(
            Tensor::<TestBackend, 3>::zeros([2, 2, 6], &device),
            Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device),
            Tensor::<TestBackend, 3>::zeros([2, 2, 4], &device),
        )
        .unwrap();

        let step = trunk.step(token_embedding, state).unwrap();

        assert_eq!(step.next_state().shape().batch_size, 2);
        assert_eq!(step.next_state().shape().root_count, 2);
        assert_eq!(step.next_state().shape().recurrent_dim, 6);
        assert_eq!(step.root_readouts().dims(), [2, 2, 4]);
    }

    #[test]
    fn baseline_local_trunk_root_biases_prevent_trivial_initial_collapse() {
        let device = <TestBackend as Backend>::Device::default();
        let trunk = baseline_trunk::<TestBackend>();
        let token_embedding =
            Tensor::<TestBackend, 2>::from_data(TensorData::new(vec![1.0f32; 8], [1, 8]), &device);
        let state = MultiRootState::from_tensors(
            Tensor::<TestBackend, 3>::zeros([1, 2, 6], &device),
            Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device),
            Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device),
        )
        .unwrap();

        let readouts = trunk
            .step(token_embedding, state)
            .unwrap()
            .root_readouts()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();

        assert!(readouts[0..4] != readouts[4..8]);
    }

    #[test]
    fn root_readout_sequence_diagnostics_report_pairwise_similarity_and_stats() {
        let device = <TestBackend as Backend>::Device::default();
        let root_readouts = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0f32, 0.0, 0.0, 1.0, //
                    0.0, 1.0, 1.0, 0.0, //
                ],
                [1, 2, 2, 2],
            ),
            &device,
        );

        let diagnostics = summarize_root_readout_sequence(root_readouts).unwrap();

        assert_eq!(diagnostics.per_root.len(), 2);
        assert!(diagnostics.per_root[0].mean_l2_norm > 0.0);
        assert!(diagnostics.per_root[1].mean_l2_norm > 0.0);
        assert!(diagnostics.mean_pairwise_cosine_similarity.abs() < 1.0);
    }

    #[test]
    fn baseline_local_trunk_backward_smoke_produces_gradients() {
        let device = <CpuTrainBackend as Backend>::Device::default();
        let trunk = baseline_trunk::<CpuTrainBackend>();
        let token_embedding = Tensor::<CpuTrainBackend, 2>::from_data(
            TensorData::new(vec![0.25f32; 16], [2, 8]),
            &device,
        );
        let state = MultiRootState::from_tensors(
            Tensor::<CpuTrainBackend, 3>::zeros([2, 2, 6], &device),
            Tensor::<CpuTrainBackend, 3>::zeros([2, 2, 4], &device),
            Tensor::<CpuTrainBackend, 3>::zeros([2, 2, 4], &device),
        )
        .unwrap();

        let loss = trunk
            .step(token_embedding, state)
            .unwrap()
            .root_readouts()
            .sum();
        let grads = burn::optim::GradientsParams::from_grads(loss.backward(), &trunk);

        assert!(crate::registry::gradient_l2_norm(&trunk, &grads) > 0.0);
    }
}
