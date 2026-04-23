use burn::{
    module::{Ignored, Module, Param},
    nn::{
        transformer::{
            PositionWiseFeedForward, PositionWiseFeedForwardConfig, TransformerEncoder,
            TransformerEncoderConfig, TransformerEncoderInput,
        },
        Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

use super::common::local_causal_mask;
use super::mamba3_baseline::{SimpleRmsNorm, DEFAULT_RUST_MAMBA3_NORM_EPS};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
    state::{FractalState, StateLayout},
    HybridAttentionLayerRole, HybridAttentionVariantSpec, PrimitiveHybridNormMode,
    PrimitiveHybridPrimitive, PrimitiveHybridReadoutMode, PrimitiveHybridResidualMode,
    PrimitiveHybridWrapperSymmetryMode,
};

const DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionModelShape {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub head_count: usize,
    pub local_window: usize,
    pub total_layers: usize,
}

impl HybridAttentionModelShape {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.vocab_size == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.vocab_size must be greater than zero".to_string(),
            ));
        }
        if self.d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.d_model must be greater than zero".to_string(),
            ));
        }
        if self.d_ff == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.d_ff must be greater than zero".to_string(),
            ));
        }
        if self.head_count == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.head_count must be greater than zero".to_string(),
            ));
        }
        if !self.d_model.is_multiple_of(self.head_count) {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention_model.d_model {} must be divisible by head_count {}",
                self.d_model, self.head_count
            )));
        }
        if self.local_window == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.local_window must be greater than zero".to_string(),
            ));
        }
        if self.total_layers == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_model.total_layers must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SequencePrimitiveStepResult<B: Backend> {
    pub next_state: Tensor<B, 2>,
    pub emitted_output: Tensor<B, 2>,
}

#[derive(Debug)]
pub struct SequencePrimitiveScanResult<B: Backend> {
    pub emitted_outputs: Tensor<B, 3>,
    pub final_state: Tensor<B, 2>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrimitiveHybridBlockConfig {
    primitive_kind: PrimitiveHybridPrimitive,
    residual_mode: PrimitiveHybridResidualMode,
    readout_mode: PrimitiveHybridReadoutMode,
    norm_mode: PrimitiveHybridNormMode,
    wrapper_symmetry_mode: PrimitiveHybridWrapperSymmetryMode,
}

#[derive(Module, Debug)]
pub struct AttentionOnlyHybridAttentionModel<B: Backend> {
    embedding: Embedding<B>,
    encoder: TransformerEncoder<B>,
    output: LanguageModelHead<B>,
    vocab_size: usize,
    d_model: usize,
    head_count: usize,
    local_window: usize,
    total_layers: usize,
}

impl<B: Backend> AttentionOnlyHybridAttentionModel<B> {
    pub fn new(shape: HybridAttentionModelShape, device: &B::Device) -> Result<Self, FractalError> {
        shape.validate()?;
        let encoder = TransformerEncoderConfig::new(
            shape.d_model,
            shape.d_ff,
            shape.head_count,
            shape.total_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .with_initializer(Initializer::Uniform {
            min: -0.08,
            max: 0.08,
        })
        .init(device);
        Ok(Self {
            embedding: EmbeddingConfig::new(shape.vocab_size, shape.d_model)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            encoder,
            output: LanguageModelHeadConfig::new(shape.d_model, shape.vocab_size)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            vocab_size: shape.vocab_size,
            d_model: shape.d_model,
            head_count: shape.head_count,
            local_window: shape.local_window,
            total_layers: shape.total_layers,
        })
    }

    pub fn shape(&self) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_ff: self.d_model * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
            head_count: self.head_count,
            local_window: self.local_window,
            total_layers: self.total_layers,
        }
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let embeddings = self.embedding.forward(input_ids);
        let mask =
            local_causal_mask::<B>(batch_size, seq_len, self.local_window, &embeddings.device());
        let hidden = self
            .encoder
            .forward(TransformerEncoderInput::new(embeddings).mask_attn(mask));
        Ok(self.output.forward(hidden))
    }
}

#[derive(Module, Debug)]
pub struct ContractiveSequenceMixer<B: Backend> {
    gate_projection: StructuredProjection<B>,
    state_projection: StructuredProjection<B>,
    input_projection: StructuredProjection<B>,
}

impl<B: Backend> ContractiveSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Self {
            gate_projection: projection.init(device),
            state_projection: projection.init(device),
            input_projection: projection.init(device),
        }
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let gate = gated_sigmoid(self.gate_projection.forward(input.clone()));
        let mix = self.state_projection.forward(state.clone())
            + self.input_projection.forward(input.clone());
        let next_state = gate.clone() * mix + one_minus(gate) * state.clone();
        SequencePrimitiveStepResult {
            emitted_output: next_state.clone(),
            next_state,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, d_model] = inputs.dims();
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..d_model])
                .reshape([batch_size, d_model]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, d_model]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct P2RotaryReadoutSequenceMixer<B: Backend> {
    update_gate_projection: StructuredProjection<B>,
    state_transform_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    candidate_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    output_projection: StructuredProjection<B>,
    d_model: usize,
}

impl<B: Backend> P2RotaryReadoutSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(FractalError::InvalidConfig(format!(
                "p2_rotary_readout_sequence_mixer requires a positive even d_model, got {d_model}"
            )));
        }
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, d_model / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            update_gate_projection: projection.init(device),
            state_transform_projection: projection.init(device),
            angle_projection: angle_projection.init(device),
            candidate_projection: projection.init(device),
            output_gate_projection: projection.init(device),
            output_projection: projection.init(device),
            d_model,
        })
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let update_gate = gated_sigmoid(self.update_gate_projection.forward(input.clone()));
        let transformed_state = rotate_state_pairs(
            self.state_transform_projection.forward(state.clone()),
            self.angle_projection.forward(input.clone()),
        );
        let candidate = self.candidate_projection.forward(input.clone()).tanh();
        let next_state =
            update_gate.clone() * transformed_state + one_minus(update_gate) * candidate;
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        let emitted_output = output_gate * self.output_projection.forward(next_state.clone());
        SequencePrimitiveStepResult {
            next_state,
            emitted_output,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, width] = inputs.dims();
        debug_assert_eq!(width, self.d_model);
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..width])
                .reshape([batch_size, width]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, width]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct P23RotaryCarryBlendReadoutSequenceMixer<B: Backend> {
    update_gate_projection: StructuredProjection<B>,
    state_transform_projection: StructuredProjection<B>,
    carry_state_projection: StructuredProjection<B>,
    dynamics_mix_gate_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    candidate_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    output_projection: StructuredProjection<B>,
    d_model: usize,
}

impl<B: Backend> P23RotaryCarryBlendReadoutSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(FractalError::InvalidConfig(format!(
                "p2_3_rotary_carry_blend_readout_sequence_mixer requires a positive even d_model, got {d_model}"
            )));
        }
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, d_model / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            update_gate_projection: projection.init(device),
            state_transform_projection: projection.init(device),
            carry_state_projection: projection.init(device),
            dynamics_mix_gate_projection: projection.init(device),
            angle_projection: angle_projection.init(device),
            candidate_projection: projection.init(device),
            output_gate_projection: projection.init(device),
            output_projection: projection.init(device),
            d_model,
        })
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let update_gate = gated_sigmoid(self.update_gate_projection.forward(input.clone()));
        let dynamics_mix_gate =
            gated_sigmoid(self.dynamics_mix_gate_projection.forward(input.clone()));
        let rotated_state = rotate_state_pairs(
            self.state_transform_projection.forward(state.clone()),
            self.angle_projection.forward(input.clone()),
        );
        let carried_state = self.carry_state_projection.forward(state.clone()).tanh();
        let transformed_state = dynamics_mix_gate.clone() * rotated_state
            + one_minus(dynamics_mix_gate) * carried_state;
        let candidate = self.candidate_projection.forward(input.clone()).tanh();
        let next_state =
            update_gate.clone() * transformed_state + one_minus(update_gate) * candidate;
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        let emitted_output = output_gate * self.output_projection.forward(next_state.clone());
        SequencePrimitiveStepResult {
            next_state,
            emitted_output,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, width] = inputs.dims();
        debug_assert_eq!(width, self.d_model);
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..width])
                .reshape([batch_size, width]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, width]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct P20RotaryStateOutputSequenceMixer<B: Backend> {
    update_gate_projection: StructuredProjection<B>,
    state_transform_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    candidate_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    d_model: usize,
}

impl<B: Backend> P20RotaryStateOutputSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(FractalError::InvalidConfig(format!(
                "p2_0_rotary_state_output_sequence_mixer requires a positive even d_model, got {d_model}"
            )));
        }
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, d_model / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            update_gate_projection: projection.init(device),
            state_transform_projection: projection.init(device),
            angle_projection: angle_projection.init(device),
            candidate_projection: projection.init(device),
            output_gate_projection: projection.init(device),
            d_model,
        })
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let update_gate = gated_sigmoid(self.update_gate_projection.forward(input.clone()));
        let transformed_state = rotate_state_pairs(
            self.state_transform_projection.forward(state.clone()),
            self.angle_projection.forward(input.clone()),
        );
        let candidate = self.candidate_projection.forward(input.clone()).tanh();
        let next_state =
            update_gate.clone() * transformed_state + one_minus(update_gate) * candidate;
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        let emitted_output = output_gate * next_state.clone();
        SequencePrimitiveStepResult {
            next_state,
            emitted_output,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, width] = inputs.dims();
        debug_assert_eq!(width, self.d_model);
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..width])
                .reshape([batch_size, width]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, width]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct P21WideLatentSequenceMixer<B: Backend> {
    update_gate_projection: StructuredProjection<B>,
    state_transform_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    candidate_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    latent_dim: usize,
    d_model: usize,
}

impl<B: Backend> P21WideLatentSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "p2_1_wide_latent_sequence_mixer requires a positive d_model".to_string(),
            ));
        }
        let latent_dim = d_model.checked_mul(2).ok_or_else(|| {
            FractalError::InvalidConfig(format!(
                "p2_1_wide_latent_sequence_mixer latent width overflowed for d_model {d_model}"
            ))
        })?;
        let state_projection = StructuredProjectionConfig::new(latent_dim, latent_dim)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let input_to_latent = StructuredProjectionConfig::new(d_model, latent_dim)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, latent_dim / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let output_gate = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            update_gate_projection: input_to_latent.init(device),
            state_transform_projection: state_projection.init(device),
            angle_projection: angle_projection.init(device),
            candidate_projection: input_to_latent.init(device),
            output_gate_projection: output_gate.init(device),
            latent_dim,
            d_model,
        })
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let update_gate = gated_sigmoid(self.update_gate_projection.forward(input.clone()));
        let transformed_state = rotate_state_pairs(
            self.state_transform_projection.forward(state.clone()),
            self.angle_projection.forward(input.clone()),
        );
        let candidate = self.candidate_projection.forward(input.clone()).tanh();
        let next_state =
            update_gate.clone() * transformed_state + one_minus(update_gate) * candidate;
        let readout = leading_state_slice(next_state.clone(), self.d_model);
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        let emitted_output = output_gate * readout;
        SequencePrimitiveStepResult {
            next_state,
            emitted_output,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, width] = inputs.dims();
        debug_assert_eq!(width, self.d_model);
        debug_assert_eq!(initial_state.dims()[1], self.latent_dim);
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..width])
                .reshape([batch_size, width]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, self.d_model]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct P22WideLatentReadoutSequenceMixer<B: Backend> {
    update_gate_projection: StructuredProjection<B>,
    state_transform_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    candidate_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    output_projection: StructuredProjection<B>,
    latent_dim: usize,
    d_model: usize,
}

impl<B: Backend> P22WideLatentReadoutSequenceMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "p2_2_wide_latent_readout_sequence_mixer requires a positive d_model".to_string(),
            ));
        }
        let latent_dim = d_model.checked_mul(2).ok_or_else(|| {
            FractalError::InvalidConfig(format!(
                "p2_2_wide_latent_readout_sequence_mixer latent width overflowed for d_model {d_model}"
            ))
        })?;
        let state_projection = StructuredProjectionConfig::new(latent_dim, latent_dim)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let input_to_latent = StructuredProjectionConfig::new(d_model, latent_dim)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, latent_dim / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let latent_to_output = StructuredProjectionConfig::new(latent_dim, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let output_gate = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            update_gate_projection: input_to_latent.init(device),
            state_transform_projection: state_projection.init(device),
            angle_projection: angle_projection.init(device),
            candidate_projection: input_to_latent.init(device),
            output_gate_projection: output_gate.init(device),
            output_projection: latent_to_output.init(device),
            latent_dim,
            d_model,
        })
    }

    pub fn step(
        &self,
        state: &Tensor<B, 2>,
        input: &Tensor<B, 2>,
    ) -> SequencePrimitiveStepResult<B> {
        let update_gate = gated_sigmoid(self.update_gate_projection.forward(input.clone()));
        let transformed_state = rotate_state_pairs(
            self.state_transform_projection.forward(state.clone()),
            self.angle_projection.forward(input.clone()),
        );
        let candidate = self.candidate_projection.forward(input.clone()).tanh();
        let next_state =
            update_gate.clone() * transformed_state + one_minus(update_gate) * candidate;
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        let emitted_output = output_gate * self.output_projection.forward(next_state.clone());
        SequencePrimitiveStepResult {
            next_state,
            emitted_output,
        }
    }

    pub fn scan(
        &self,
        initial_state: Tensor<B, 2>,
        inputs: Tensor<B, 3>,
    ) -> SequencePrimitiveScanResult<B> {
        let [batch_size, seq_len, width] = inputs.dims();
        debug_assert_eq!(width, self.d_model);
        debug_assert_eq!(initial_state.dims()[1], self.latent_dim);
        let mut state = initial_state;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = inputs
                .clone()
                .slice([0..batch_size, position..position + 1, 0..width])
                .reshape([batch_size, width]);
            let step = self.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([batch_size, 1, self.d_model]));
        }
        SequencePrimitiveScanResult {
            emitted_outputs: Tensor::cat(outputs, 1),
            final_state: state,
        }
    }
}

#[derive(Module, Debug)]
pub struct RotarySelectiveStateMixer<B: Backend> {
    decay_projection: StructuredProjection<B>,
    angle_projection: StructuredProjection<B>,
    input_projection: StructuredProjection<B>,
    output_gate_projection: StructuredProjection<B>,
    output_projection: StructuredProjection<B>,
    d_model: usize,
}

impl<B: Backend> RotarySelectiveStateMixer<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || !d_model.is_multiple_of(2) {
            return Err(FractalError::InvalidConfig(format!(
                "rotary_selective_state_mixer requires a positive even d_model, got {d_model}"
            )));
        }
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let angle_projection = StructuredProjectionConfig::new(d_model, d_model / 2)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            decay_projection: projection.init(device),
            angle_projection: angle_projection.init(device),
            input_projection: projection.init(device),
            output_gate_projection: projection.init(device),
            output_projection: projection.init(device),
            d_model,
        })
    }

    pub fn forward(&self, state: &Tensor<B, 2>, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        let decay = gated_sigmoid(self.decay_projection.forward(input.clone()));
        let candidate = self.input_projection.forward(input.clone()).tanh();
        let rotated =
            rotate_state_pairs(state.clone(), self.angle_projection.forward(input.clone()));
        let next_state = decay.clone() * rotated + one_minus(decay) * candidate;
        let output_gate = gated_sigmoid(self.output_gate_projection.forward(input.clone()));
        output_gate * self.output_projection.forward(next_state)
    }

    pub fn next_state(&self, state: &Tensor<B, 2>, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        let decay = gated_sigmoid(self.decay_projection.forward(input.clone()));
        let candidate = self.input_projection.forward(input.clone()).tanh();
        let rotated =
            rotate_state_pairs(state.clone(), self.angle_projection.forward(input.clone()));
        decay.clone() * rotated + one_minus(decay) * candidate
    }
}

#[derive(Module, Debug)]
pub struct PrimitiveMixerBlock<B: Backend> {
    input_norm: Option<LayerNorm<B>>,
    output_norm: Option<LayerNorm<B>>,
    input_rms_norm: Option<SimpleRmsNorm<B>>,
    output_rms_norm: Option<SimpleRmsNorm<B>>,
    primitive_kind: Ignored<PrimitiveHybridPrimitive>,
    residual_mode: Ignored<PrimitiveHybridResidualMode>,
    readout_mode: Ignored<PrimitiveHybridReadoutMode>,
    norm_mode: Ignored<PrimitiveHybridNormMode>,
    wrapper_symmetry_mode: Ignored<PrimitiveHybridWrapperSymmetryMode>,
    p1_contractive: Option<ContractiveSequenceMixer<B>>,
    p20_rotary_state_output: Option<P20RotaryStateOutputSequenceMixer<B>>,
    p2_rotary_readout: Option<P2RotaryReadoutSequenceMixer<B>>,
    p23_rotary_carry_blend_readout: Option<P23RotaryCarryBlendReadoutSequenceMixer<B>>,
    p21_wide_latent: Option<P21WideLatentSequenceMixer<B>>,
    p22_wide_latent_readout: Option<P22WideLatentReadoutSequenceMixer<B>>,
    residual_scale: Option<Param<Tensor<B, 1>>>,
    residual_gate_projection: Option<StructuredProjection<B>>,
    wrapper_readout_projection: Option<StructuredProjection<B>>,
    wrapper_readout_norm: Option<LayerNorm<B>>,
    wrapper_post_readout_norm: Option<LayerNorm<B>>,
    wrapper_residual_renorm: Option<LayerNorm<B>>,
    feedforward: PositionWiseFeedForward<B>,
    d_model: usize,
}

impl<B: Backend> PrimitiveMixerBlock<B> {
    fn new(
        d_model: usize,
        d_ff: usize,
        config: PrimitiveHybridBlockConfig,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        if d_model == 0 || d_ff == 0 {
            return Err(FractalError::InvalidConfig(
                "primitive_mixer_block dimensions must be greater than zero".to_string(),
            ));
        }
        let (
            p1_contractive,
            p20_rotary_state_output,
            p2_rotary_readout,
            p23_rotary_carry_blend_readout,
            p21_wide_latent,
            p22_wide_latent_readout,
        ) = match config.primitive_kind {
            PrimitiveHybridPrimitive::P1Contractive => (
                Some(ContractiveSequenceMixer::new(d_model, device)),
                None,
                None,
                None,
                None,
                None,
            ),
            PrimitiveHybridPrimitive::P20RotaryStateOutput => (
                None,
                Some(P20RotaryStateOutputSequenceMixer::new(d_model, device)?),
                None,
                None,
                None,
                None,
            ),
            PrimitiveHybridPrimitive::P2RotaryReadout => (
                None,
                None,
                Some(P2RotaryReadoutSequenceMixer::new(d_model, device)?),
                None,
                None,
                None,
            ),
            PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout => (
                None,
                None,
                None,
                Some(P23RotaryCarryBlendReadoutSequenceMixer::new(
                    d_model, device,
                )?),
                None,
                None,
            ),
            PrimitiveHybridPrimitive::P21WideLatent => (
                None,
                None,
                None,
                None,
                Some(P21WideLatentSequenceMixer::new(d_model, device)?),
                None,
            ),
            PrimitiveHybridPrimitive::P22WideLatentReadout => (
                None,
                None,
                None,
                None,
                None,
                Some(P22WideLatentReadoutSequenceMixer::new(d_model, device)?),
            ),
        };
        let projection = StructuredProjectionConfig::new(d_model, d_model)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let (residual_scale, residual_gate_projection) = match config.residual_mode {
            PrimitiveHybridResidualMode::PlainAdd => (None, None),
            PrimitiveHybridResidualMode::ScaledAdd => (
                Some(Param::from_data(
                    TensorData::new(vec![0.5f32; d_model], [d_model]),
                    device,
                )),
                None,
            ),
            PrimitiveHybridResidualMode::GatedAdd => (None, Some(projection.init(device))),
        };
        let (wrapper_readout_projection, wrapper_readout_norm) = match config.readout_mode {
            PrimitiveHybridReadoutMode::Direct => (None, None),
            PrimitiveHybridReadoutMode::Projected => (Some(projection.init(device)), None),
            PrimitiveHybridReadoutMode::ProjectedNorm => (
                Some(projection.init(device)),
                Some(LayerNormConfig::new(d_model).init(device)),
            ),
        };
        let (wrapper_post_readout_norm, wrapper_residual_renorm) = match config.norm_mode {
            PrimitiveHybridNormMode::PreNormOnly => (None, None),
            PrimitiveHybridNormMode::PostReadoutNorm => {
                (Some(LayerNormConfig::new(d_model).init(device)), None)
            }
            PrimitiveHybridNormMode::ResidualSideRenorm => {
                (None, Some(LayerNormConfig::new(d_model).init(device)))
            }
        };
        let (input_norm, output_norm, input_rms_norm, output_rms_norm) =
            match config.wrapper_symmetry_mode {
                PrimitiveHybridWrapperSymmetryMode::Standard => (
                    Some(LayerNormConfig::new(d_model).init(device)),
                    Some(LayerNormConfig::new(d_model).init(device)),
                    None,
                    None,
                ),
                PrimitiveHybridWrapperSymmetryMode::MambaRms => (
                    None,
                    None,
                    Some(SimpleRmsNorm::new(
                        d_model,
                        DEFAULT_RUST_MAMBA3_NORM_EPS,
                        device,
                    )?),
                    Some(SimpleRmsNorm::new(
                        d_model,
                        DEFAULT_RUST_MAMBA3_NORM_EPS,
                        device,
                    )?),
                ),
            };
        Ok(Self {
            input_norm,
            output_norm,
            input_rms_norm,
            output_rms_norm,
            primitive_kind: Ignored(config.primitive_kind),
            residual_mode: Ignored(config.residual_mode),
            readout_mode: Ignored(config.readout_mode),
            norm_mode: Ignored(config.norm_mode),
            wrapper_symmetry_mode: Ignored(config.wrapper_symmetry_mode),
            p1_contractive,
            p20_rotary_state_output,
            p2_rotary_readout,
            p23_rotary_carry_blend_readout,
            p21_wide_latent,
            p22_wide_latent_readout,
            residual_scale,
            residual_gate_projection,
            wrapper_readout_projection,
            wrapper_readout_norm,
            wrapper_post_readout_norm,
            wrapper_residual_renorm,
            feedforward: PositionWiseFeedForwardConfig::new(d_model, d_ff)
                .with_dropout(0.0)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            d_model,
        })
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, _seq_len, width] = input.dims();
        if width != self.d_model {
            return Err(FractalError::Shape(format!(
                "primitive_mixer_block expected width {}, got {}",
                self.d_model, width
            )));
        }

        let normed = match *self.wrapper_symmetry_mode {
            PrimitiveHybridWrapperSymmetryMode::Standard => self
                .input_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing standard input norm".to_string(),
                    )
                })?
                .forward(input.clone()),
            PrimitiveHybridWrapperSymmetryMode::MambaRms => self
                .input_rms_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing mamba-rms input norm".to_string(),
                    )
                })?
                .forward3(input.clone()),
        };
        let device = normed.device();
        let state_width = (*self.primitive_kind).state_width(self.d_model);
        let initial_state = zero_flat_state::<B>(batch_size, state_width, &device)?;
        let mixed = match *self.primitive_kind {
            PrimitiveHybridPrimitive::P1Contractive => {
                self.p1_contractive
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P1Contractive module".to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
            PrimitiveHybridPrimitive::P20RotaryStateOutput => {
                self.p20_rotary_state_output
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P20RotaryStateOutput module".to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
            PrimitiveHybridPrimitive::P2RotaryReadout => {
                self.p2_rotary_readout
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P2RotaryReadout module".to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
            PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout => {
                self.p23_rotary_carry_blend_readout
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P23RotaryCarryBlendReadout module"
                                .to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
            PrimitiveHybridPrimitive::P21WideLatent => {
                self.p21_wide_latent
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P21WideLatent module".to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
            PrimitiveHybridPrimitive::P22WideLatentReadout => {
                self.p22_wide_latent_readout
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing P22WideLatentReadout module".to_string(),
                        )
                    })?
                    .scan(initial_state, normed.clone())
                    .emitted_outputs
            }
        };
        let readout = match *self.readout_mode {
            PrimitiveHybridReadoutMode::Direct => mixed,
            PrimitiveHybridReadoutMode::Projected => self
                .wrapper_readout_projection
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing projected readout projection".to_string(),
                    )
                })?
                .forward(mixed),
            PrimitiveHybridReadoutMode::ProjectedNorm => self
                .wrapper_readout_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing projected-norm readout norm".to_string(),
                    )
                })?
                .forward(
                    self.wrapper_readout_projection
                        .as_ref()
                        .ok_or_else(|| {
                            FractalError::InvalidState(
                                "primitive_mixer_block missing projected-norm readout projection"
                                    .to_string(),
                            )
                        })?
                        .forward(mixed),
                ),
        };
        let readout = match *self.norm_mode {
            PrimitiveHybridNormMode::PreNormOnly => readout,
            PrimitiveHybridNormMode::PostReadoutNorm => self
                .wrapper_post_readout_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing post-readout norm".to_string(),
                    )
                })?
                .forward(readout),
            PrimitiveHybridNormMode::ResidualSideRenorm => readout,
        };
        let residual = match *self.residual_mode {
            PrimitiveHybridResidualMode::PlainAdd => input + readout,
            PrimitiveHybridResidualMode::ScaledAdd => {
                let scale = self
                    .residual_scale
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "primitive_mixer_block missing scaled residual parameter".to_string(),
                        )
                    })?
                    .val()
                    .reshape([1, 1, self.d_model]);
                input + readout * scale
            }
            PrimitiveHybridResidualMode::GatedAdd => {
                let gate = gated_sigmoid::<B, 3>(
                    self.residual_gate_projection
                        .as_ref()
                        .ok_or_else(|| {
                            FractalError::InvalidState(
                                "primitive_mixer_block missing gated residual projection"
                                    .to_string(),
                            )
                        })?
                        .forward(normed.clone()),
                );
                input + gate * readout
            }
        };
        let residual = match *self.norm_mode {
            PrimitiveHybridNormMode::PreNormOnly | PrimitiveHybridNormMode::PostReadoutNorm => {
                residual
            }
            PrimitiveHybridNormMode::ResidualSideRenorm => self
                .wrapper_residual_renorm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing residual-side renorm".to_string(),
                    )
                })?
                .forward(residual),
        };
        let ff_input = match *self.wrapper_symmetry_mode {
            PrimitiveHybridWrapperSymmetryMode::Standard => self
                .output_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing standard output norm".to_string(),
                    )
                })?
                .forward(residual.clone()),
            PrimitiveHybridWrapperSymmetryMode::MambaRms => self
                .output_rms_norm
                .as_ref()
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "primitive_mixer_block missing mamba-rms output norm".to_string(),
                    )
                })?
                .forward3(residual.clone()),
        };
        let ff = self.feedforward.forward(ff_input);
        Ok(residual + ff)
    }
}

#[derive(Module, Debug)]
pub struct ReferenceSsmMixerBlock<B: Backend> {
    input_norm: LayerNorm<B>,
    output_norm: LayerNorm<B>,
    reference_ssm: RotarySelectiveStateMixer<B>,
    feedforward: PositionWiseFeedForward<B>,
    d_model: usize,
}

impl<B: Backend> ReferenceSsmMixerBlock<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || d_ff == 0 {
            return Err(FractalError::InvalidConfig(
                "reference_ssm_mixer_block dimensions must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            input_norm: LayerNormConfig::new(d_model).init(device),
            output_norm: LayerNormConfig::new(d_model).init(device),
            reference_ssm: RotarySelectiveStateMixer::new(d_model, device)?,
            feedforward: PositionWiseFeedForwardConfig::new(d_model, d_ff)
                .with_dropout(0.0)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            d_model,
        })
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len, width] = input.dims();
        if width != self.d_model {
            return Err(FractalError::Shape(format!(
                "reference_ssm_mixer_block expected width {}, got {}",
                self.d_model, width
            )));
        }

        let normed = self.input_norm.forward(input.clone());
        let device = normed.device();
        let mut state =
            FractalState::<B>::zeros(StateLayout::Flat, batch_size, self.d_model, &device)?
                .flat()?;
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = normed
                .clone()
                .slice([0..batch_size, position..position + 1, 0..self.d_model])
                .reshape([batch_size, self.d_model]);
            let y_t = self.reference_ssm.forward(&state, &x_t);
            state = self.reference_ssm.next_state(&state, &x_t);
            outputs.push(y_t.reshape([batch_size, 1, self.d_model]));
        }
        let mixed = Tensor::cat(outputs, 1);
        let residual = input + mixed;
        let ff = self
            .feedforward
            .forward(self.output_norm.forward(residual.clone()));
        Ok(residual + ff)
    }
}

#[derive(Module, Debug)]
pub struct PrimitiveHybridAttentionModel<B: Backend> {
    embedding: Embedding<B>,
    attention_layers: Vec<TransformerEncoder<B>>,
    primitive_layers: Vec<PrimitiveMixerBlock<B>>,
    layer_schedule: Ignored<Vec<HybridAttentionLayerRole>>,
    output: LanguageModelHead<B>,
    vocab_size: usize,
    d_model: usize,
    head_count: usize,
    local_window: usize,
    total_layers: usize,
}

impl<B: Backend> PrimitiveHybridAttentionModel<B> {
    fn new(
        shape: HybridAttentionModelShape,
        layer_schedule: &[HybridAttentionLayerRole],
        block_config: PrimitiveHybridBlockConfig,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        shape.validate()?;
        if shape.total_layers != layer_schedule.len() {
            return Err(FractalError::InvalidConfig(format!(
                "primitive_hybrid_attention_model expected total_layers {} to match schedule length {}",
                shape.total_layers,
                layer_schedule.len()
            )));
        }
        if layer_schedule.iter().any(|role| {
            !matches!(
                role,
                HybridAttentionLayerRole::ExactAttention | HybridAttentionLayerRole::Primitive
            )
        }) {
            return Err(FractalError::InvalidConfig(
                "primitive_hybrid_attention_model schedule may contain only exact-attention and primitive layers".to_string(),
            ));
        }
        let attention_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ExactAttention))
            .count();
        let primitive_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::Primitive))
            .count();
        if attention_count == 0 || primitive_count == 0 {
            return Err(FractalError::InvalidConfig(
                "primitive_hybrid_attention_model schedule must contain at least one exact-attention and one primitive layer".to_string(),
            ));
        }

        let attention_layers = (0..attention_count)
            .map(|_| {
                TransformerEncoderConfig::new(shape.d_model, shape.d_ff, shape.head_count, 1)
                    .with_dropout(0.0)
                    .with_norm_first(true)
                    .with_initializer(Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    })
                    .init(device)
            })
            .collect();
        let mut primitive_layers = Vec::with_capacity(primitive_count);
        for _ in 0..primitive_count {
            primitive_layers.push(PrimitiveMixerBlock::new(
                shape.d_model,
                shape.d_ff,
                block_config,
                device,
            )?);
        }

        Ok(Self {
            embedding: EmbeddingConfig::new(shape.vocab_size, shape.d_model)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            attention_layers,
            primitive_layers,
            layer_schedule: Ignored(layer_schedule.to_vec()),
            output: LanguageModelHeadConfig::new(shape.d_model, shape.vocab_size)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            vocab_size: shape.vocab_size,
            d_model: shape.d_model,
            head_count: shape.head_count,
            local_window: shape.local_window,
            total_layers: shape.total_layers,
        })
    }

    pub fn shape(&self) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_ff: self.d_model * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
            head_count: self.head_count,
            local_window: self.local_window,
            total_layers: self.total_layers,
        }
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids);
        let mask = local_causal_mask::<B>(batch_size, seq_len, self.local_window, &hidden.device());
        let mut attention_index = 0;
        let mut primitive_index = 0;
        for role in self.layer_schedule.iter() {
            match role {
                HybridAttentionLayerRole::ExactAttention => {
                    hidden = self.attention_layers[attention_index]
                        .forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone()));
                    attention_index += 1;
                }
                HybridAttentionLayerRole::Primitive => {
                    hidden = self.primitive_layers[primitive_index].forward(hidden)?;
                    primitive_index += 1;
                }
                HybridAttentionLayerRole::ReferenceSsm => {
                    return Err(FractalError::InvalidConfig(
                        "primitive_hybrid_attention_model schedule contained an unexpected reference-SSM layer at runtime".to_string(),
                    ));
                }
            }
        }
        Ok(self.output.forward(hidden))
    }
}

#[derive(Module, Debug)]
pub struct ReferenceSsmHybridAttentionModel<B: Backend> {
    embedding: Embedding<B>,
    attention_layers: Vec<TransformerEncoder<B>>,
    reference_layers: Vec<ReferenceSsmMixerBlock<B>>,
    layer_schedule: Ignored<Vec<HybridAttentionLayerRole>>,
    output: LanguageModelHead<B>,
    vocab_size: usize,
    d_model: usize,
    head_count: usize,
    local_window: usize,
    total_layers: usize,
}

impl<B: Backend> ReferenceSsmHybridAttentionModel<B> {
    pub fn new(
        shape: HybridAttentionModelShape,
        layer_schedule: &[HybridAttentionLayerRole],
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        shape.validate()?;
        if !shape.d_model.is_multiple_of(2) {
            return Err(FractalError::InvalidConfig(format!(
                "reference_ssm_hybrid_attention_model requires even d_model, got {}",
                shape.d_model
            )));
        }
        if shape.total_layers != layer_schedule.len() {
            return Err(FractalError::InvalidConfig(format!(
                "reference_ssm_hybrid_attention_model expected total_layers {} to match schedule length {}",
                shape.total_layers,
                layer_schedule.len()
            )));
        }
        if layer_schedule.iter().any(|role| {
            !matches!(
                role,
                HybridAttentionLayerRole::ExactAttention | HybridAttentionLayerRole::ReferenceSsm
            )
        }) {
            return Err(FractalError::InvalidConfig(
                "reference_ssm_hybrid_attention_model schedule may contain only exact-attention and reference-SSM layers".to_string(),
            ));
        }
        let attention_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ExactAttention))
            .count();
        let reference_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ReferenceSsm))
            .count();
        if attention_count == 0 || reference_count == 0 {
            return Err(FractalError::InvalidConfig(
                "reference_ssm_hybrid_attention_model schedule must contain at least one exact-attention and one reference-SSM layer".to_string(),
            ));
        }

        let attention_layers = (0..attention_count)
            .map(|_| {
                TransformerEncoderConfig::new(shape.d_model, shape.d_ff, shape.head_count, 1)
                    .with_dropout(0.0)
                    .with_norm_first(true)
                    .with_initializer(Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    })
                    .init(device)
            })
            .collect();
        let mut reference_layers = Vec::with_capacity(reference_count);
        for _ in 0..reference_count {
            reference_layers.push(ReferenceSsmMixerBlock::new(
                shape.d_model,
                shape.d_ff,
                device,
            )?);
        }

        Ok(Self {
            embedding: EmbeddingConfig::new(shape.vocab_size, shape.d_model)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            attention_layers,
            reference_layers,
            layer_schedule: Ignored(layer_schedule.to_vec()),
            output: LanguageModelHeadConfig::new(shape.d_model, shape.vocab_size)
                .with_initializer(Initializer::Uniform {
                    min: -0.08,
                    max: 0.08,
                })
                .init(device),
            vocab_size: shape.vocab_size,
            d_model: shape.d_model,
            head_count: shape.head_count,
            local_window: shape.local_window,
            total_layers: shape.total_layers,
        })
    }

    pub fn shape(&self) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_ff: self.d_model * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
            head_count: self.head_count,
            local_window: self.local_window,
            total_layers: self.total_layers,
        }
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids);
        let mask = local_causal_mask::<B>(batch_size, seq_len, self.local_window, &hidden.device());
        let mut attention_index = 0;
        let mut reference_index = 0;
        for role in self.layer_schedule.iter() {
            match role {
                HybridAttentionLayerRole::ExactAttention => {
                    hidden = self.attention_layers[attention_index]
                        .forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone()));
                    attention_index += 1;
                }
                HybridAttentionLayerRole::ReferenceSsm => {
                    hidden = self.reference_layers[reference_index].forward(hidden)?;
                    reference_index += 1;
                }
                HybridAttentionLayerRole::Primitive => {
                    return Err(FractalError::InvalidConfig(
                        "reference_ssm_hybrid_attention_model schedule contained an unexpected primitive layer at runtime".to_string(),
                    ));
                }
            }
        }
        Ok(self.output.forward(hidden))
    }
}

pub fn build_attention_only_hybrid_attention_model<B: Backend>(
    vocab_size: usize,
    variant: &HybridAttentionVariantSpec,
    device: &B::Device,
) -> Result<AttentionOnlyHybridAttentionModel<B>, FractalError> {
    let shape = HybridAttentionModelShape {
        vocab_size,
        d_model: variant.hidden_dim,
        d_ff: variant.hidden_dim * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
        head_count: variant.head_count,
        local_window: variant.local_window,
        total_layers: variant.total_layers(),
    };
    AttentionOnlyHybridAttentionModel::new(shape, device)
}

pub fn build_primitive_hybrid_attention_model<B: Backend>(
    vocab_size: usize,
    variant: &HybridAttentionVariantSpec,
    device: &B::Device,
) -> Result<PrimitiveHybridAttentionModel<B>, FractalError> {
    let primitive_kind = variant.primitive.ok_or_else(|| {
        FractalError::InvalidConfig(
            "primitive_hybrid variant must set primitive before building the model".to_string(),
        )
    })?;
    let residual_mode = variant.primitive_residual_mode.ok_or_else(|| {
        FractalError::InvalidConfig(
            "primitive_hybrid variant must set primitive_residual_mode before building the model"
                .to_string(),
        )
    })?;
    let readout_mode = variant.primitive_readout_mode.ok_or_else(|| {
        FractalError::InvalidConfig(
            "primitive_hybrid variant must set primitive_readout_mode before building the model"
                .to_string(),
        )
    })?;
    let norm_mode = variant.primitive_norm_mode.ok_or_else(|| {
        FractalError::InvalidConfig(
            "primitive_hybrid variant must set primitive_norm_mode before building the model"
                .to_string(),
        )
    })?;
    let wrapper_symmetry_mode =
        variant
            .primitive_wrapper_symmetry_mode
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "primitive_hybrid variant must set primitive_wrapper_symmetry_mode before building the model"
                        .to_string(),
                )
            })?;
    let block_config = PrimitiveHybridBlockConfig {
        primitive_kind,
        residual_mode,
        readout_mode,
        norm_mode,
        wrapper_symmetry_mode,
    };
    let shape = HybridAttentionModelShape {
        vocab_size,
        d_model: variant.hidden_dim,
        d_ff: variant.hidden_dim * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
        head_count: variant.head_count,
        local_window: variant.local_window,
        total_layers: variant.total_layers(),
    };
    PrimitiveHybridAttentionModel::new(shape, &variant.layer_schedule, block_config, device)
}

pub fn build_reference_ssm_hybrid_attention_model<B: Backend>(
    vocab_size: usize,
    variant: &HybridAttentionVariantSpec,
    device: &B::Device,
) -> Result<ReferenceSsmHybridAttentionModel<B>, FractalError> {
    let shape = HybridAttentionModelShape {
        vocab_size,
        d_model: variant.hidden_dim,
        d_ff: variant.hidden_dim * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
        head_count: variant.head_count,
        local_window: variant.local_window,
        total_layers: variant.total_layers(),
    };
    ReferenceSsmHybridAttentionModel::new(shape, &variant.layer_schedule, device)
}

fn zero_flat_state<B: Backend>(
    batch_size: usize,
    d_model: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>, FractalError> {
    FractalState::<B>::zeros(StateLayout::Flat, batch_size, d_model, device)?.flat()
}

fn rotate_state_pairs<B: Backend>(state: Tensor<B, 2>, angles: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch_size, width] = state.dims();
    let pair_count = width / 2;
    let state_pairs = state.reshape([batch_size, pair_count, 2]);
    let first = state_pairs
        .clone()
        .slice([0..batch_size, 0..pair_count, 0..1])
        .reshape([batch_size, pair_count]);
    let second = state_pairs
        .slice([0..batch_size, 0..pair_count, 1..2])
        .reshape([batch_size, pair_count]);
    let cos = angles.clone().cos();
    let sin = angles.sin();
    let rotated_first = first.clone() * cos.clone() - second.clone() * sin.clone();
    let rotated_second = first * sin + second * cos;
    Tensor::cat(
        vec![
            rotated_first.reshape([batch_size, pair_count, 1]),
            rotated_second.reshape([batch_size, pair_count, 1]),
        ],
        2,
    )
    .reshape([batch_size, width])
}

fn leading_state_slice<B: Backend>(state: Tensor<B, 2>, d_model: usize) -> Tensor<B, 2> {
    let [batch_size, _width] = state.dims();
    state.slice([0..batch_size, 0..d_model])
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    use burn::tensor::{Int, Tensor};

    use super::super::common::local_causal_mask;
    use super::{
        build_attention_only_hybrid_attention_model, build_primitive_hybrid_attention_model,
        build_reference_ssm_hybrid_attention_model, rotate_state_pairs, zero_flat_state,
        P20RotaryStateOutputSequenceMixer, P21WideLatentSequenceMixer,
        P22WideLatentReadoutSequenceMixer, P23RotaryCarryBlendReadoutSequenceMixer,
        P2RotaryReadoutSequenceMixer,
    };
    use crate::{
        phase1_hybrid_attention_baseline_matrix, phase1_p20_candidate_variant,
        phase1_p21_candidate_variant, phase1_p22_candidate_variant, phase1_p23_candidate_variant,
        phase1_p2_candidate_variant, phase1_p2_interface_candidate_variant,
        PrimitiveHybridNormMode, PrimitiveHybridPrimitive, PrimitiveHybridReadoutMode,
        PrimitiveHybridResidualMode, PrimitiveHybridWrapperSymmetryMode,
    };

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn attention_only_model_returns_logits() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_attention_only_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.attention_only,
            &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_returns_logits() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_primitive_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.primitive_hybrid,
            &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p2_variant() {
        let device = Default::default();
        let variant = phase1_p2_candidate_variant();
        let model =
            build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p20_variant() {
        let device = Default::default();
        let variant = phase1_p20_candidate_variant();
        let model =
            build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p21_variant() {
        let device = Default::default();
        let variant = phase1_p21_candidate_variant();
        let model =
            build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p22_variant() {
        let device = Default::default();
        let variant = phase1_p22_candidate_variant();
        let model =
            build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p23_variant() {
        let device = Default::default();
        let variant = phase1_p23_candidate_variant();
        let model =
            build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn primitive_hybrid_model_supports_p2_interface_residual_variants() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        for residual_mode in [
            PrimitiveHybridResidualMode::PlainAdd,
            PrimitiveHybridResidualMode::ScaledAdd,
            PrimitiveHybridResidualMode::GatedAdd,
        ] {
            let variant = phase1_p2_interface_candidate_variant(
                PrimitiveHybridPrimitive::P2RotaryReadout,
                residual_mode,
                PrimitiveHybridReadoutMode::Direct,
                PrimitiveHybridNormMode::PreNormOnly,
                PrimitiveHybridWrapperSymmetryMode::Standard,
            );
            let model =
                build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device)
                    .unwrap();
            assert_eq!(
                model.forward_logits(input.clone()).unwrap().dims(),
                [2, 8, 257]
            );
        }
    }

    #[test]
    fn primitive_hybrid_model_supports_p2_interface_readout_variants() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        for readout_mode in [
            PrimitiveHybridReadoutMode::Direct,
            PrimitiveHybridReadoutMode::Projected,
            PrimitiveHybridReadoutMode::ProjectedNorm,
        ] {
            let variant = phase1_p2_interface_candidate_variant(
                PrimitiveHybridPrimitive::P2RotaryReadout,
                PrimitiveHybridResidualMode::PlainAdd,
                readout_mode,
                PrimitiveHybridNormMode::PreNormOnly,
                PrimitiveHybridWrapperSymmetryMode::Standard,
            );
            let model =
                build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device)
                    .unwrap();
            assert_eq!(
                model.forward_logits(input.clone()).unwrap().dims(),
                [2, 8, 257]
            );
        }
    }

    #[test]
    fn primitive_hybrid_model_supports_p2_interface_norm_variants() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        for norm_mode in [
            PrimitiveHybridNormMode::PreNormOnly,
            PrimitiveHybridNormMode::PostReadoutNorm,
            PrimitiveHybridNormMode::ResidualSideRenorm,
        ] {
            let variant = phase1_p2_interface_candidate_variant(
                PrimitiveHybridPrimitive::P2RotaryReadout,
                PrimitiveHybridResidualMode::PlainAdd,
                PrimitiveHybridReadoutMode::Direct,
                norm_mode,
                PrimitiveHybridWrapperSymmetryMode::Standard,
            );
            let model =
                build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device)
                    .unwrap();
            assert_eq!(
                model.forward_logits(input.clone()).unwrap().dims(),
                [2, 8, 257]
            );
        }
    }

    #[test]
    fn primitive_hybrid_model_supports_p2_wrapper_symmetry_variants() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        for wrapper_symmetry_mode in [
            PrimitiveHybridWrapperSymmetryMode::Standard,
            PrimitiveHybridWrapperSymmetryMode::MambaRms,
        ] {
            let variant = phase1_p2_interface_candidate_variant(
                PrimitiveHybridPrimitive::P2RotaryReadout,
                PrimitiveHybridResidualMode::PlainAdd,
                PrimitiveHybridReadoutMode::Direct,
                PrimitiveHybridNormMode::PreNormOnly,
                wrapper_symmetry_mode,
            );
            let model =
                build_primitive_hybrid_attention_model::<TestBackend>(257, &variant, &device)
                    .unwrap();
            assert_eq!(
                model.forward_logits(input.clone()).unwrap().dims(),
                [2, 8, 257]
            );
        }
    }

    #[test]
    fn reference_ssm_hybrid_model_returns_logits() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_reference_ssm_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.reference_ssm_hybrid,
            &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn local_causal_mask_respects_window() {
        let device = Default::default();
        let mask = local_causal_mask::<TestBackend>(1, 4, 2, &device);
        let values = mask.to_data().to_vec::<bool>().unwrap();
        assert_eq!(
            values,
            vec![
                false, true, true, true, false, false, true, true, true, false, false, true, true,
                true, false, false,
            ]
        );
    }

    #[test]
    fn rotate_state_pairs_applies_pairwise_rotation() {
        let device = Default::default();
        let state = Tensor::<TestBackend, 2>::from_data([[1.0, 0.0, 0.0, 1.0]], &device);
        let angles =
            Tensor::<TestBackend, 2>::from_data([[core::f32::consts::FRAC_PI_2, 0.0]], &device);
        let rotated = rotate_state_pairs(state, angles);
        let values = rotated.to_data().to_vec::<f32>().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-4);
        assert!((values[1] - 1.0).abs() < 1e-4);
        assert!((values[2] - 0.0).abs() < 1e-4);
        assert!((values[3] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn p2_rotary_readout_scan_matches_manual_step_loop() {
        let device = Default::default();
        let mixer = P2RotaryReadoutSequenceMixer::<TestBackend>::new(4, &device).unwrap();
        let initial_state = zero_flat_state::<TestBackend>(1, 4, &device).unwrap();
        let inputs = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.1, -0.2, 0.3, -0.4],
                [0.5, 0.2, -0.1, 0.7],
                [-0.3, 0.6, 0.4, -0.2],
            ]],
            &device,
        );

        let scan = mixer.scan(initial_state.clone(), inputs.clone());

        let mut state = initial_state;
        let mut outputs = Vec::new();
        for position in 0..3 {
            let x_t = inputs
                .clone()
                .slice([0..1, position..position + 1, 0..4])
                .reshape([1, 4]);
            let step = mixer.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([1, 1, 4]));
        }
        let manual_outputs = Tensor::cat(outputs, 1);

        let scan_output = scan.emitted_outputs.to_data().to_vec::<f32>().unwrap();
        let manual_output = manual_outputs.to_data().to_vec::<f32>().unwrap();
        let scan_state = scan.final_state.to_data().to_vec::<f32>().unwrap();
        let manual_state = state.to_data().to_vec::<f32>().unwrap();

        for (lhs, rhs) in scan_output.iter().zip(manual_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        for (lhs, rhs) in scan_state.iter().zip(manual_state.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn p20_rotary_state_output_scan_matches_manual_step_loop() {
        let device = Default::default();
        let mixer = P20RotaryStateOutputSequenceMixer::<TestBackend>::new(4, &device).unwrap();
        let initial_state = zero_flat_state::<TestBackend>(1, 4, &device).unwrap();
        let inputs = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.1, 0.2, -0.3, 0.4],
                [-0.5, 0.1, 0.7, -0.2],
                [0.3, -0.4, 0.6, 0.2],
            ]],
            &device,
        );

        let scan = mixer.scan(initial_state.clone(), inputs.clone());

        let mut state = initial_state;
        let mut outputs = Vec::new();
        for position in 0..3 {
            let x_t = inputs
                .clone()
                .slice([0..1, position..position + 1, 0..4])
                .reshape([1, 4]);
            let step = mixer.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([1, 1, 4]));
        }
        let manual_outputs = Tensor::cat(outputs, 1);

        let scan_output = scan.emitted_outputs.to_data().to_vec::<f32>().unwrap();
        let manual_output = manual_outputs.to_data().to_vec::<f32>().unwrap();
        let scan_state = scan.final_state.to_data().to_vec::<f32>().unwrap();
        let manual_state = state.to_data().to_vec::<f32>().unwrap();

        for (lhs, rhs) in scan_output.iter().zip(manual_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        for (lhs, rhs) in scan_state.iter().zip(manual_state.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn p23_rotary_carry_blend_readout_scan_matches_manual_step_loop() {
        let device = Default::default();
        let mixer =
            P23RotaryCarryBlendReadoutSequenceMixer::<TestBackend>::new(4, &device).unwrap();
        let initial_state = zero_flat_state::<TestBackend>(1, 4, &device).unwrap();
        let inputs = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.2, -0.1, 0.4, -0.3],
                [0.6, 0.1, -0.2, 0.5],
                [-0.4, 0.7, 0.3, -0.1],
            ]],
            &device,
        );

        let scan = mixer.scan(initial_state.clone(), inputs.clone());

        let mut state = initial_state;
        let mut outputs = Vec::new();
        for position in 0..3 {
            let x_t = inputs
                .clone()
                .slice([0..1, position..position + 1, 0..4])
                .reshape([1, 4]);
            let step = mixer.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([1, 1, 4]));
        }
        let manual_outputs = Tensor::cat(outputs, 1);

        let scan_output = scan.emitted_outputs.to_data().to_vec::<f32>().unwrap();
        let manual_output = manual_outputs.to_data().to_vec::<f32>().unwrap();
        let scan_state = scan.final_state.to_data().to_vec::<f32>().unwrap();
        let manual_state = state.to_data().to_vec::<f32>().unwrap();

        for (lhs, rhs) in scan_output.iter().zip(manual_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        for (lhs, rhs) in scan_state.iter().zip(manual_state.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn p21_wide_latent_scan_matches_manual_step_loop() {
        let device = Default::default();
        let mixer = P21WideLatentSequenceMixer::<TestBackend>::new(4, &device).unwrap();
        let initial_state = zero_flat_state::<TestBackend>(1, 8, &device).unwrap();
        let inputs = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.2, -0.1, 0.3, 0.4],
                [-0.5, 0.2, 0.1, -0.7],
                [0.6, -0.4, 0.2, 0.1],
            ]],
            &device,
        );

        let scan = mixer.scan(initial_state.clone(), inputs.clone());

        let mut state = initial_state;
        let mut outputs = Vec::new();
        for position in 0..3 {
            let x_t = inputs
                .clone()
                .slice([0..1, position..position + 1, 0..4])
                .reshape([1, 4]);
            let step = mixer.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([1, 1, 4]));
        }
        let manual_outputs = Tensor::cat(outputs, 1);

        let scan_output = scan.emitted_outputs.to_data().to_vec::<f32>().unwrap();
        let manual_output = manual_outputs.to_data().to_vec::<f32>().unwrap();
        let scan_state = scan.final_state.to_data().to_vec::<f32>().unwrap();
        let manual_state = state.to_data().to_vec::<f32>().unwrap();

        for (lhs, rhs) in scan_output.iter().zip(manual_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        for (lhs, rhs) in scan_state.iter().zip(manual_state.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }

    #[test]
    fn p22_wide_latent_readout_scan_matches_manual_step_loop() {
        let device = Default::default();
        let mixer = P22WideLatentReadoutSequenceMixer::<TestBackend>::new(4, &device).unwrap();
        let initial_state = zero_flat_state::<TestBackend>(1, 8, &device).unwrap();
        let inputs = Tensor::<TestBackend, 3>::from_data(
            [[
                [0.2, -0.1, 0.3, 0.4],
                [-0.5, 0.2, 0.1, -0.7],
                [0.6, -0.4, 0.2, 0.1],
            ]],
            &device,
        );

        let scan = mixer.scan(initial_state.clone(), inputs.clone());

        let mut state = initial_state;
        let mut outputs = Vec::new();
        for position in 0..3 {
            let x_t = inputs
                .clone()
                .slice([0..1, position..position + 1, 0..4])
                .reshape([1, 4]);
            let step = mixer.step(&state, &x_t);
            state = step.next_state;
            outputs.push(step.emitted_output.reshape([1, 1, 4]));
        }
        let manual_outputs = Tensor::cat(outputs, 1);

        let scan_output = scan.emitted_outputs.to_data().to_vec::<f32>().unwrap();
        let manual_output = manual_outputs.to_data().to_vec::<f32>().unwrap();
        let scan_state = scan.final_state.to_data().to_vec::<f32>().unwrap();
        let manual_state = state.to_data().to_vec::<f32>().unwrap();

        for (lhs, rhs) in scan_output.iter().zip(manual_output.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
        for (lhs, rhs) in scan_state.iter().zip(manual_state.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }
}
