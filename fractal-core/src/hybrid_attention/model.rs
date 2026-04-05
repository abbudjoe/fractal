use burn::{
    module::{Ignored, Module},
    nn::{
        transformer::{
            PositionWiseFeedForward, PositionWiseFeedForwardConfig, TransformerEncoder,
            TransformerEncoderConfig, TransformerEncoderInput,
        },
        Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Bool, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
    state::{FractalState, StateLayout},
    HybridAttentionLayerRole, HybridAttentionVariantSpec,
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

    pub fn forward(&self, state: &Tensor<B, 2>, input: &Tensor<B, 2>) -> Tensor<B, 2> {
        let gate = gated_sigmoid(self.gate_projection.forward(input.clone()));
        let mix = self.state_projection.forward(state.clone())
            + self.input_projection.forward(input.clone());
        gate.clone() * mix + one_minus(gate) * state.clone()
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
    input_norm: LayerNorm<B>,
    output_norm: LayerNorm<B>,
    primitive: ContractiveSequenceMixer<B>,
    feedforward: PositionWiseFeedForward<B>,
    d_model: usize,
}

impl<B: Backend> PrimitiveMixerBlock<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 || d_ff == 0 {
            return Err(FractalError::InvalidConfig(
                "primitive_mixer_block dimensions must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            input_norm: LayerNormConfig::new(d_model).init(device),
            output_norm: LayerNormConfig::new(d_model).init(device),
            primitive: ContractiveSequenceMixer::new(d_model, device),
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
                "primitive_mixer_block expected width {}, got {}",
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
            state = self.primitive.forward(&state, &x_t);
            outputs.push(state.clone().reshape([batch_size, 1, self.d_model]));
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
    pub fn new(
        shape: HybridAttentionModelShape,
        layer_schedule: &[HybridAttentionLayerRole],
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
            primitive_layers.push(PrimitiveMixerBlock::new(shape.d_model, shape.d_ff, device)?);
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
    let shape = HybridAttentionModelShape {
        vocab_size,
        d_model: variant.hidden_dim,
        d_ff: variant.hidden_dim * DEFAULT_HYBRID_ATTENTION_FEEDFORWARD_MULTIPLIER,
        head_count: variant.head_count,
        local_window: variant.local_window,
        total_layers: variant.total_layers(),
    };
    PrimitiveHybridAttentionModel::new(shape, &variant.layer_schedule, device)
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

fn local_causal_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    local_window: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mut data = Vec::with_capacity(batch_size * seq_len * seq_len);
    for _ in 0..batch_size {
        for query in 0..seq_len {
            let earliest_visible = query.saturating_sub(local_window.saturating_sub(1));
            for key in 0..seq_len {
                data.push(key < earliest_visible || key > query);
            }
        }
    }
    Tensor::from_data(
        TensorData::new(data, [batch_size, seq_len, seq_len]),
        device,
    )
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

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    use burn::tensor::{Int, Tensor};

    use super::{
        build_attention_only_hybrid_attention_model, build_primitive_hybrid_attention_model,
        build_reference_ssm_hybrid_attention_model, local_causal_mask, rotate_state_pairs,
    };
    use crate::phase1_hybrid_attention_baseline_matrix;

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
}
