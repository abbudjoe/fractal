use burn::{
    module::{Ignored, Module},
    nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Initializer,
    },
    tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor},
};
use serde::{Deserialize, Serialize};

use super::{
    common::local_causal_mask, HybridAttentionModelShape, HybridAttentionVariantKind,
    HybridAttentionVariantSpec,
};
use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    phase1_hybrid_attention_baseline_matrix,
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::recurrent_router::{
    RecurrentRouterFeedbackMode, RecurrentRouterPrimitiveKind, RecurrentRouterSelectionMode,
    RecurrentRouterSpec, VirtualNodeRecurrentRouter,
};

const SCALE_PROXY_INIT_MIN: f64 = -0.08;
const SCALE_PROXY_INIT_MAX: f64 = 0.08;
pub const SCALE_PROXY_CHANNEL_COUNT: usize = 2;
pub const DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionOnlyScaleProxyVariantSpec {
    pub label: String,
    pub base_variant: HybridAttentionVariantSpec,
    pub expert_layer_index: usize,
}

impl AttentionOnlyScaleProxyVariantSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.label.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "attention_only_scale_proxy.variant.label must be non-empty".to_string(),
            ));
        }
        self.base_variant.validate()?;
        if self.base_variant.kind != HybridAttentionVariantKind::AttentionOnly {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_scale_proxy.variant[{}] requires an attention-only base variant, got {:?}",
                self.label, self.base_variant.kind
            )));
        }
        let total_layers = self.base_variant.total_layers();
        if self.expert_layer_index >= total_layers {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_scale_proxy.variant[{}].expert_layer_index {} must be less than total_layers {}",
                self.label, self.expert_layer_index, total_layers
            )));
        }
        Ok(())
    }

    pub fn shape(&self, vocab_size: usize) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size,
            d_model: self.base_variant.hidden_dim,
            d_ff: self.base_variant.hidden_dim * 4,
            head_count: self.base_variant.head_count,
            local_window: self.base_variant.local_window,
            total_layers: self.base_variant.total_layers(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionOnlyRecurrentScaleProxyVariantSpec {
    pub label: String,
    pub base_variant: HybridAttentionVariantSpec,
    pub expert_layer_index: usize,
    pub router: RecurrentRouterSpec,
}

impl AttentionOnlyRecurrentScaleProxyVariantSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.label.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "attention_only_recurrent_scale_proxy.variant.label must be non-empty".to_string(),
            ));
        }
        self.base_variant.validate()?;
        if self.base_variant.kind != HybridAttentionVariantKind::AttentionOnly {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_recurrent_scale_proxy.variant[{}] requires an attention-only base variant, got {:?}",
                self.label, self.base_variant.kind
            )));
        }
        let total_layers = self.base_variant.total_layers();
        if self.expert_layer_index >= total_layers {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_recurrent_scale_proxy.variant[{}].expert_layer_index {} must be less than total_layers {}",
                self.label, self.expert_layer_index, total_layers
            )));
        }
        self.router.validate()?;
        if self.router.channel_count != SCALE_PROXY_CHANNEL_COUNT {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_recurrent_scale_proxy.variant[{}].router.channel_count must remain {}, got {}",
                self.label, SCALE_PROXY_CHANNEL_COUNT, self.router.channel_count
            )));
        }
        Ok(())
    }

    pub fn shape(&self, vocab_size: usize) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size,
            d_model: self.base_variant.hidden_dim,
            d_ff: self.base_variant.hidden_dim * 4,
            head_count: self.base_variant.head_count,
            local_window: self.base_variant.local_window,
            total_layers: self.base_variant.total_layers(),
        }
    }
}

pub fn scale_proxy_one_shot_over_attention_only_variant() -> AttentionOnlyScaleProxyVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    AttentionOnlyScaleProxyVariantSpec {
        label: format!(
            "dreegmor-scale-proxy-over-a-one-shot-layer{}",
            DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX + 1
        ),
        base_variant: matrix.attention_only,
        expert_layer_index: DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX,
    }
}

pub fn scale_proxy_recurrent_over_attention_only_variant(
) -> AttentionOnlyRecurrentScaleProxyVariantSpec {
    scale_proxy_recurrent_over_attention_only_variant_with_router(
        RecurrentRouterSpec::minimal_dense_gru(),
    )
}

pub fn scale_proxy_recurrent_over_attention_only_variant_with_router(
    router: RecurrentRouterSpec,
) -> AttentionOnlyRecurrentScaleProxyVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    AttentionOnlyRecurrentScaleProxyVariantSpec {
        label: format!(
            "dreegmor-scale-proxy-over-a-recurrent-{}-layer{}",
            router.label_suffix(),
            DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX + 1
        ),
        base_variant: matrix.attention_only,
        expert_layer_index: DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX,
        router,
    }
}

#[derive(Debug)]
pub struct ScaleProxyRoutingProbe<B: Backend> {
    pub initial_weights: Tensor<B, 3>,
    pub final_weights: Tensor<B, 3>,
    pub round_weights: Vec<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
struct TwoChannelHiddenStateRouter<B: Backend> {
    route_projection: StructuredProjection<B>,
}

impl<B: Backend> TwoChannelHiddenStateRouter<B> {
    fn new(d_model: usize, device: &B::Device) -> Result<Self, FractalError> {
        if d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "attention_only_scale_proxy.router requires positive d_model".to_string(),
            ));
        }
        Ok(Self {
            route_projection: StructuredProjectionConfig::new(d_model, SCALE_PROXY_CHANNEL_COUNT)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .init(device),
        })
    }

    fn routing_probe(&self, hidden: Tensor<B, 3>) -> ScaleProxyRoutingProbe<B> {
        let weights = softmax(self.route_projection.forward(hidden), 2);
        ScaleProxyRoutingProbe {
            initial_weights: weights.clone(),
            final_weights: weights.clone(),
            round_weights: vec![weights],
        }
    }
}

#[derive(Module, Debug)]
struct TwoChannelHiddenStateRecurrentRouter<B: Backend> {
    contract: Ignored<VirtualNodeRecurrentRouter>,
    token_state_projection: StructuredProjection<B>,
    token_route_projection: StructuredProjection<B>,
    state_route_projection: StructuredProjection<B>,
    route_feedback_projection: StructuredProjection<B>,
    reset_gate_projection: StructuredProjection<B>,
    update_gate_projection: StructuredProjection<B>,
    candidate_input_projection: StructuredProjection<B>,
    candidate_state_projection: StructuredProjection<B>,
}

impl<B: Backend> TwoChannelHiddenStateRecurrentRouter<B> {
    fn new(
        d_model: usize,
        router: RecurrentRouterSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let contract = VirtualNodeRecurrentRouter::new(d_model, router.clone())?;
        if router.primitive_kind != RecurrentRouterPrimitiveKind::GruVirtualNode {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_recurrent_scale_proxy.router only supports gru-virtual-node in the minimal runtime, got {:?}",
                router.primitive_kind
            )));
        }
        if router.feedback_mode != RecurrentRouterFeedbackMode::RouterStateOnly {
            return Err(FractalError::InvalidConfig(
                "attention_only_recurrent_scale_proxy.router expert-feedback is not implemented yet"
                    .to_string(),
            ));
        }
        if router.selection_mode != RecurrentRouterSelectionMode::DenseSoftmax {
            return Err(FractalError::InvalidConfig(format!(
                "attention_only_recurrent_scale_proxy.router only supports dense-softmax in the minimal runtime, got {:?}",
                router.selection_mode
            )));
        }
        let projection = |d_input, d_output| {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .init(device)
        };
        Ok(Self {
            contract: Ignored(contract),
            token_state_projection: projection(d_model, router.state_width),
            token_route_projection: projection(router.state_width, SCALE_PROXY_CHANNEL_COUNT),
            state_route_projection: projection(router.state_width, SCALE_PROXY_CHANNEL_COUNT),
            route_feedback_projection: projection(SCALE_PROXY_CHANNEL_COUNT, router.state_width),
            reset_gate_projection: projection(router.state_width, router.state_width),
            update_gate_projection: projection(router.state_width, router.state_width),
            candidate_input_projection: projection(router.state_width, router.state_width),
            candidate_state_projection: projection(router.state_width, router.state_width),
        })
    }

    fn routing_probe(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<ScaleProxyRoutingProbe<B>, FractalError> {
        let [batch_size, seq_len, _width] = hidden.dims();
        let router = &self.contract;
        let token_state = self.token_state_projection.forward(hidden);
        let token_logits = self.token_route_projection.forward(token_state.clone());
        let pooled_token_state = mean_over_tokens(token_state).tanh();
        let mut state = pooled_token_state.clone();
        let mut round_weights = Vec::with_capacity(router.spec.round_count);

        for round_index in 0..router.spec.round_count {
            let state_bias = self
                .state_route_projection
                .forward(state.clone())
                .reshape([batch_size, 1, SCALE_PROXY_CHANNEL_COUNT])
                .repeat(&[1, seq_len, 1]);
            let weights = softmax(token_logits.clone() + state_bias, 2);
            round_weights.push(weights.clone());

            if round_index + 1 < router.spec.round_count {
                let route_summary = mean_over_tokens(weights);
                let controller_drive = pooled_token_state.clone()
                    + self.route_feedback_projection.forward(route_summary);
                let reset_gate =
                    gated_sigmoid(self.reset_gate_projection.forward(controller_drive.clone()));
                let update_gate = gated_sigmoid(
                    self.update_gate_projection
                        .forward(controller_drive.clone()),
                );
                let candidate = (self.candidate_input_projection.forward(controller_drive)
                    + reset_gate * self.candidate_state_projection.forward(state.clone()))
                .tanh();
                state = update_gate.clone() * state + one_minus(update_gate) * candidate;
            }
        }

        let initial_weights = round_weights.first().cloned().ok_or_else(|| {
            FractalError::InvalidState(
                "attention_only_recurrent_scale_proxy.router produced no routing rounds"
                    .to_string(),
            )
        })?;
        let final_weights = round_weights.last().cloned().ok_or_else(|| {
            FractalError::InvalidState(
                "attention_only_recurrent_scale_proxy.router produced no final routing weights"
                    .to_string(),
            )
        })?;
        Ok(ScaleProxyRoutingProbe {
            initial_weights,
            final_weights,
            round_weights,
        })
    }
}

#[derive(Module, Debug)]
pub struct AttentionOnlyScaleProxyModel<B: Backend> {
    embedding: Embedding<B>,
    shared_prefix_layers: Vec<TransformerEncoder<B>>,
    router: TwoChannelHiddenStateRouter<B>,
    expert_a: TransformerEncoder<B>,
    expert_b: TransformerEncoder<B>,
    shared_suffix_layers: Vec<TransformerEncoder<B>>,
    output: LanguageModelHead<B>,
    variant: Ignored<AttentionOnlyScaleProxyVariantSpec>,
}

impl<B: Backend> AttentionOnlyScaleProxyModel<B> {
    pub fn new(
        vocab_size: usize,
        variant: &AttentionOnlyScaleProxyVariantSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        variant.validate()?;
        let shape = variant.shape(vocab_size);
        Ok(Self {
            embedding: shared_embedding(shape, device),
            shared_prefix_layers: build_attention_layers(shape, variant.expert_layer_index, device),
            router: TwoChannelHiddenStateRouter::new(shape.d_model, device)?,
            expert_a: build_attention_layer(shape, device),
            expert_b: build_attention_layer(shape, device),
            shared_suffix_layers: build_attention_layers(
                shape,
                shape.total_layers - variant.expert_layer_index - 1,
                device,
            ),
            output: shared_output_head(shape, device, true),
            variant: Ignored(variant.clone()),
        })
    }

    pub fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<ScaleProxyRoutingProbe<B>, FractalError> {
        let (hidden, _mask) = self.hidden_before_expert(input_ids);
        Ok(self.router.routing_probe(hidden))
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let (hidden, mask) = self.hidden_before_expert(input_ids);
        let probe = self.router.routing_probe(hidden.clone());
        let mixed = mix_expert_outputs(
            self.expert_a
                .forward(TransformerEncoderInput::new(hidden.clone()).mask_attn(mask.clone())),
            self.expert_b
                .forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone())),
            probe.final_weights,
        );
        Ok(self
            .output
            .forward(forward_layers(mixed, &self.shared_suffix_layers, mask)))
    }

    pub fn vocab_size(&self) -> usize {
        self.output.logical_dims()[1]
    }

    fn hidden_before_expert(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3, Bool>) {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids);
        let mask = local_causal_mask::<B>(
            batch_size,
            seq_len,
            self.variant.base_variant.local_window,
            &hidden.device(),
        );
        hidden = forward_layers(hidden, &self.shared_prefix_layers, mask.clone());
        (hidden, mask)
    }
}

#[derive(Module, Debug)]
pub struct AttentionOnlyRecurrentScaleProxyModel<B: Backend> {
    embedding: Embedding<B>,
    shared_prefix_layers: Vec<TransformerEncoder<B>>,
    router: TwoChannelHiddenStateRecurrentRouter<B>,
    expert_a: TransformerEncoder<B>,
    expert_b: TransformerEncoder<B>,
    shared_suffix_layers: Vec<TransformerEncoder<B>>,
    output: LanguageModelHead<B>,
    variant: Ignored<AttentionOnlyRecurrentScaleProxyVariantSpec>,
}

impl<B: Backend> AttentionOnlyRecurrentScaleProxyModel<B> {
    pub fn new(
        vocab_size: usize,
        variant: &AttentionOnlyRecurrentScaleProxyVariantSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        variant.validate()?;
        let shape = variant.shape(vocab_size);
        Ok(Self {
            embedding: shared_embedding(shape, device),
            shared_prefix_layers: build_attention_layers(shape, variant.expert_layer_index, device),
            router: TwoChannelHiddenStateRecurrentRouter::new(
                shape.d_model,
                variant.router.clone(),
                device,
            )?,
            expert_a: build_attention_layer(shape, device),
            expert_b: build_attention_layer(shape, device),
            shared_suffix_layers: build_attention_layers(
                shape,
                shape.total_layers - variant.expert_layer_index - 1,
                device,
            ),
            output: shared_output_head(shape, device, true),
            variant: Ignored(variant.clone()),
        })
    }

    pub fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<ScaleProxyRoutingProbe<B>, FractalError> {
        let (hidden, _mask) = self.hidden_before_expert(input_ids);
        self.router.routing_probe(hidden)
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let (hidden, mask) = self.hidden_before_expert(input_ids);
        let probe = self.router.routing_probe(hidden.clone())?;
        let mixed = mix_expert_outputs(
            self.expert_a
                .forward(TransformerEncoderInput::new(hidden.clone()).mask_attn(mask.clone())),
            self.expert_b
                .forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone())),
            probe.final_weights,
        );
        Ok(self
            .output
            .forward(forward_layers(mixed, &self.shared_suffix_layers, mask)))
    }

    pub fn vocab_size(&self) -> usize {
        self.output.logical_dims()[1]
    }

    fn hidden_before_expert(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3, Bool>) {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids);
        let mask = local_causal_mask::<B>(
            batch_size,
            seq_len,
            self.variant.base_variant.local_window,
            &hidden.device(),
        );
        hidden = forward_layers(hidden, &self.shared_prefix_layers, mask.clone());
        (hidden, mask)
    }
}

pub fn build_attention_only_scale_proxy_model<B: Backend>(
    vocab_size: usize,
    variant: &AttentionOnlyScaleProxyVariantSpec,
    device: &B::Device,
) -> Result<AttentionOnlyScaleProxyModel<B>, FractalError> {
    AttentionOnlyScaleProxyModel::new(vocab_size, variant, device)
}

pub fn build_attention_only_recurrent_scale_proxy_model<B: Backend>(
    vocab_size: usize,
    variant: &AttentionOnlyRecurrentScaleProxyVariantSpec,
    device: &B::Device,
) -> Result<AttentionOnlyRecurrentScaleProxyModel<B>, FractalError> {
    AttentionOnlyRecurrentScaleProxyModel::new(vocab_size, variant, device)
}

fn build_attention_layer<B: Backend>(
    shape: HybridAttentionModelShape,
    device: &B::Device,
) -> TransformerEncoder<B> {
    TransformerEncoderConfig::new(shape.d_model, shape.d_ff, shape.head_count, 1)
        .with_dropout(0.0)
        .with_norm_first(true)
        .with_initializer(Initializer::Uniform {
            min: SCALE_PROXY_INIT_MIN,
            max: SCALE_PROXY_INIT_MAX,
        })
        .init(device)
}

fn build_attention_layers<B: Backend>(
    shape: HybridAttentionModelShape,
    count: usize,
    device: &B::Device,
) -> Vec<TransformerEncoder<B>> {
    (0..count)
        .map(|_| build_attention_layer(shape, device))
        .collect()
}

fn shared_embedding<B: Backend>(
    shape: HybridAttentionModelShape,
    device: &B::Device,
) -> Embedding<B> {
    EmbeddingConfig::new(shape.vocab_size, shape.d_model)
        .with_initializer(Initializer::Uniform {
            min: SCALE_PROXY_INIT_MIN,
            max: SCALE_PROXY_INIT_MAX,
        })
        .init(device)
}

fn shared_output_head<B: Backend>(
    shape: HybridAttentionModelShape,
    device: &B::Device,
    bias: bool,
) -> LanguageModelHead<B> {
    LanguageModelHeadConfig::new(shape.d_model, shape.vocab_size)
        .with_bias(bias)
        .with_initializer(Initializer::Uniform {
            min: SCALE_PROXY_INIT_MIN,
            max: SCALE_PROXY_INIT_MAX,
        })
        .init(device)
}

fn forward_layers<B: Backend>(
    mut hidden: Tensor<B, 3>,
    layers: &[TransformerEncoder<B>],
    mask: Tensor<B, 3, Bool>,
) -> Tensor<B, 3> {
    for layer in layers {
        hidden = layer.forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone()));
    }
    hidden
}

fn mean_over_tokens<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, width] = tensor.dims();
    tensor
        .sum_dim(1)
        .reshape([batch_size, width])
        .mul_scalar(1.0 / seq_len as f64)
}

fn mix_expert_outputs<B: Backend>(
    expert_a: Tensor<B, 3>,
    expert_b: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, width] = expert_a.dims();
    debug_assert_eq!(expert_b.dims(), [batch_size, seq_len, width]);
    debug_assert_eq!(
        weights.dims(),
        [batch_size, seq_len, SCALE_PROXY_CHANNEL_COUNT]
    );
    let stacked = Tensor::cat(
        vec![
            expert_a.reshape([batch_size, seq_len, 1, width]),
            expert_b.reshape([batch_size, seq_len, 1, width]),
        ],
        2,
    );
    let expanded_weights = weights
        .reshape([batch_size, seq_len, SCALE_PROXY_CHANNEL_COUNT, 1])
        .repeat(&[1, 1, 1, width]);
    (stacked * expanded_weights)
        .sum_dim(2)
        .reshape([batch_size, seq_len, width])
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Int, Tensor};
    use burn::{backend::Candle, module::Module};

    use super::{
        build_attention_only_recurrent_scale_proxy_model, build_attention_only_scale_proxy_model,
        scale_proxy_one_shot_over_attention_only_variant,
        scale_proxy_recurrent_over_attention_only_variant,
        AttentionOnlyRecurrentScaleProxyVariantSpec, AttentionOnlyScaleProxyVariantSpec,
        DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX,
    };
    use crate::{
        build_attention_only_graph_of_experts_model,
        goe_over_attention_only_variant_with_controller, phase1_hybrid_attention_baseline_matrix,
        GraphOfExpertsControllerSpec,
    };

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn scale_proxy_variant_reuses_attention_surface() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let variant: AttentionOnlyScaleProxyVariantSpec =
            scale_proxy_one_shot_over_attention_only_variant();
        assert_eq!(variant.base_variant, matrix.attention_only);
        assert_eq!(
            variant.expert_layer_index,
            DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX
        );
    }

    #[test]
    fn scale_proxy_one_shot_model_returns_logits_and_weights() {
        let device = Default::default();
        let variant = scale_proxy_one_shot_over_attention_only_variant();
        let model =
            build_attention_only_scale_proxy_model::<TestBackend>(257, &variant, &device).unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(
            model.forward_logits(input.clone()).unwrap().dims(),
            [2, 8, 257]
        );
        let probe = model.routing_probe(input).unwrap();
        assert_eq!(probe.initial_weights.dims(), [2, 8, 2]);
        assert_eq!(probe.round_weights.len(), 1);
    }

    #[test]
    fn scale_proxy_recurrent_model_returns_logits_and_rounds() {
        let device = Default::default();
        let variant: AttentionOnlyRecurrentScaleProxyVariantSpec =
            scale_proxy_recurrent_over_attention_only_variant();
        let model =
            build_attention_only_recurrent_scale_proxy_model::<TestBackend>(257, &variant, &device)
                .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(
            model.forward_logits(input.clone()).unwrap().dims(),
            [2, 8, 257]
        );
        let probe = model.routing_probe(input).unwrap();
        assert_eq!(probe.final_weights.dims(), [2, 8, 2]);
        assert_eq!(probe.round_weights.len(), variant.router.round_count);
    }

    #[test]
    fn scale_proxy_model_uses_fewer_params_than_whole_backbone_dreegmor() {
        let device = Default::default();
        let scale_proxy = build_attention_only_scale_proxy_model::<TestBackend>(
            257,
            &scale_proxy_one_shot_over_attention_only_variant(),
            &device,
        )
        .unwrap();
        let whole_backbone = build_attention_only_graph_of_experts_model::<TestBackend>(
            257,
            &goe_over_attention_only_variant_with_controller(
                GraphOfExpertsControllerSpec::routed_no_graph_mix(),
            ),
            &device,
        )
        .unwrap();
        assert!(scale_proxy.num_params() < whole_backbone.num_params());
    }
}
