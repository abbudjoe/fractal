use burn::{
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Initializer},
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};
use serde::{Deserialize, Serialize};

use super::{
    build_attention_only_hybrid_attention_model, AttentionOnlyHybridAttentionModel,
    HybridAttentionModelShape, HybridAttentionVariantKind, HybridAttentionVariantSpec,
};
use crate::{
    error::FractalError,
    phase1_hybrid_attention_baseline_matrix,
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::recurrent_router::{
    RecurrentRouterFeedbackMode, RecurrentRouterPrimitiveKind, RecurrentRouterSelectionMode,
    RecurrentRouterSpec, VirtualNodeRecurrentRouter,
};

const RECURRENT_ROUTER_INIT_MIN: f64 = -0.08;
const RECURRENT_ROUTER_INIT_MAX: f64 = 0.08;
pub const RECURRENT_DREEGMOR_CHANNEL_COUNT: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionOnlyRecurrentGraphOfExpertsVariantSpec {
    pub label: String,
    pub base_variant: HybridAttentionVariantSpec,
    pub router: RecurrentRouterSpec,
}

impl AttentionOnlyRecurrentGraphOfExpertsVariantSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.label.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts.variant.label must be non-empty".to_string(),
            ));
        }
        self.base_variant.validate()?;
        if self.base_variant.kind != HybridAttentionVariantKind::AttentionOnly {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_graph_of_experts.variant[{}] requires an attention-only base variant, got {:?}",
                self.label, self.base_variant.kind
            )));
        }
        self.router.validate()?;
        if self.router.channel_count != RECURRENT_DREEGMOR_CHANNEL_COUNT {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_graph_of_experts.variant[{}] channel_count must remain {}, got {}",
                self.label, RECURRENT_DREEGMOR_CHANNEL_COUNT, self.router.channel_count
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

pub fn recurrent_goe_over_attention_only_variant() -> AttentionOnlyRecurrentGraphOfExpertsVariantSpec
{
    recurrent_goe_over_attention_only_variant_with_router(RecurrentRouterSpec::minimal_dense_gru())
}

pub fn recurrent_goe_over_attention_only_variant_with_router(
    router: RecurrentRouterSpec,
) -> AttentionOnlyRecurrentGraphOfExpertsVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    AttentionOnlyRecurrentGraphOfExpertsVariantSpec {
        label: format!("dreegmor-recurrent-over-a-{}", router.label_suffix()),
        base_variant: matrix.attention_only,
        router,
    }
}

#[derive(Debug)]
pub struct RecurrentGraphOfExpertsRoutingProbe<B: Backend> {
    pub initial_weights: Tensor<B, 3>,
    pub final_weights: Tensor<B, 3>,
    pub round_weights: Vec<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
struct TwoChannelVirtualNodeRecurrentRouter<B: Backend> {
    contract: Ignored<VirtualNodeRecurrentRouter>,
    token_embedding: Embedding<B>,
    token_state_projection: StructuredProjection<B>,
    token_route_projection: StructuredProjection<B>,
    state_route_projection: StructuredProjection<B>,
    route_feedback_projection: StructuredProjection<B>,
    reset_gate_projection: StructuredProjection<B>,
    update_gate_projection: StructuredProjection<B>,
    candidate_input_projection: StructuredProjection<B>,
    candidate_state_projection: StructuredProjection<B>,
}

impl<B: Backend> TwoChannelVirtualNodeRecurrentRouter<B> {
    fn new(
        vocab_size: usize,
        d_model: usize,
        router: RecurrentRouterSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let contract = VirtualNodeRecurrentRouter::new(d_model, router.clone())?;
        if router.primitive_kind != RecurrentRouterPrimitiveKind::GruVirtualNode {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_graph_of_experts.router only supports gru-virtual-node in the minimal runtime, got {:?}",
                router.primitive_kind
            )));
        }
        if router.feedback_mode != RecurrentRouterFeedbackMode::RouterStateOnly {
            return Err(FractalError::InvalidConfig(
                "recurrent_graph_of_experts.router expert-feedback is not implemented yet"
                    .to_string(),
            ));
        }
        if router.selection_mode != RecurrentRouterSelectionMode::DenseSoftmax {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_graph_of_experts.router only supports dense-softmax in the minimal runtime, got {:?}",
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
            token_embedding: EmbeddingConfig::new(vocab_size, d_model)
                .with_initializer(Initializer::Uniform {
                    min: RECURRENT_ROUTER_INIT_MIN,
                    max: RECURRENT_ROUTER_INIT_MAX,
                })
                .init(device),
            token_state_projection: projection(d_model, router.state_width),
            token_route_projection: projection(
                router.state_width,
                RECURRENT_DREEGMOR_CHANNEL_COUNT,
            ),
            state_route_projection: projection(
                router.state_width,
                RECURRENT_DREEGMOR_CHANNEL_COUNT,
            ),
            route_feedback_projection: projection(
                RECURRENT_DREEGMOR_CHANNEL_COUNT,
                router.state_width,
            ),
            reset_gate_projection: projection(router.state_width, router.state_width),
            update_gate_projection: projection(router.state_width, router.state_width),
            candidate_input_projection: projection(router.state_width, router.state_width),
            candidate_state_projection: projection(router.state_width, router.state_width),
        })
    }

    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RecurrentGraphOfExpertsRoutingProbe<B>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let router = &self.contract;
        let token_state = self
            .token_state_projection
            .forward(self.token_embedding.forward(input_ids));
        let token_logits = self.token_route_projection.forward(token_state.clone());
        let pooled_token_state = mean_over_tokens(token_state).tanh();
        let mut state = pooled_token_state.clone();
        let mut round_weights = Vec::with_capacity(router.spec.round_count);

        for round_index in 0..router.spec.round_count {
            let state_bias = self
                .state_route_projection
                .forward(state.clone())
                .reshape([batch_size, 1, RECURRENT_DREEGMOR_CHANNEL_COUNT])
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
                "recurrent_graph_of_experts.router produced no routing rounds".to_string(),
            )
        })?;
        let final_weights = round_weights.last().cloned().ok_or_else(|| {
            FractalError::InvalidState(
                "recurrent_graph_of_experts.router produced no final routing weights".to_string(),
            )
        })?;
        Ok(RecurrentGraphOfExpertsRoutingProbe {
            initial_weights,
            final_weights,
            round_weights,
        })
    }
}

#[derive(Module, Debug)]
pub struct AttentionOnlyRecurrentGraphOfExpertsModel<B: Backend> {
    router: TwoChannelVirtualNodeRecurrentRouter<B>,
    expert_a: AttentionOnlyHybridAttentionModel<B>,
    expert_b: AttentionOnlyHybridAttentionModel<B>,
    variant: Ignored<AttentionOnlyRecurrentGraphOfExpertsVariantSpec>,
}

impl<B: Backend> AttentionOnlyRecurrentGraphOfExpertsModel<B> {
    pub fn new(
        vocab_size: usize,
        variant: &AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        variant.validate()?;
        let shape = variant.shape(vocab_size);
        Ok(Self {
            router: TwoChannelVirtualNodeRecurrentRouter::new(
                vocab_size,
                shape.d_model,
                variant.router.clone(),
                device,
            )?,
            expert_a: build_attention_only_hybrid_attention_model::<B>(
                vocab_size,
                &variant.base_variant,
                device,
            )?,
            expert_b: build_attention_only_hybrid_attention_model::<B>(
                vocab_size,
                &variant.base_variant,
                device,
            )?,
            variant: Ignored(variant.clone()),
        })
    }

    pub fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<RecurrentGraphOfExpertsRoutingProbe<B>, FractalError> {
        self.router.routing_probe(input_ids)
    }

    pub fn route_weights(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        Ok(self.routing_probe(input_ids)?.final_weights)
    }

    pub fn vocab_size(&self) -> usize {
        self.expert_a.shape().vocab_size
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let weights = self.route_weights(input_ids.clone())?;
        let expert_a = self.expert_a.forward_logits(input_ids.clone())?;
        let expert_b = self.expert_b.forward_logits(input_ids)?;
        Ok(combine_two_channel_logits(expert_a, expert_b, weights))
    }
}

pub fn build_attention_only_recurrent_graph_of_experts_model<B: Backend>(
    vocab_size: usize,
    variant: &AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    device: &B::Device,
) -> Result<AttentionOnlyRecurrentGraphOfExpertsModel<B>, FractalError> {
    AttentionOnlyRecurrentGraphOfExpertsModel::new(vocab_size, variant, device)
}

fn mean_over_tokens<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, width] = tensor.dims();
    tensor
        .sum_dim(1)
        .reshape([batch_size, width])
        .mul_scalar(1.0 / seq_len as f64)
}

fn combine_two_channel_logits<B: Backend>(
    expert_a: Tensor<B, 3>,
    expert_b: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, vocab_size] = expert_a.dims();
    debug_assert_eq!(expert_b.dims(), [batch_size, seq_len, vocab_size]);
    debug_assert_eq!(
        weights.dims(),
        [batch_size, seq_len, RECURRENT_DREEGMOR_CHANNEL_COUNT]
    );
    let stacked = Tensor::cat(
        vec![
            expert_a.reshape([batch_size, seq_len, 1, vocab_size]),
            expert_b.reshape([batch_size, seq_len, 1, vocab_size]),
        ],
        2,
    );
    let expanded_weights = weights
        .reshape([batch_size, seq_len, RECURRENT_DREEGMOR_CHANNEL_COUNT, 1])
        .repeat(&[1, 1, 1, vocab_size]);
    (stacked * expanded_weights)
        .sum_dim(2)
        .reshape([batch_size, seq_len, vocab_size])
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    use burn::tensor::{Int, Tensor};

    use super::{
        build_attention_only_recurrent_graph_of_experts_model,
        recurrent_goe_over_attention_only_variant,
        recurrent_goe_over_attention_only_variant_with_router,
    };
    use crate::hybrid_attention::recurrent_router::RecurrentRouterSpec;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn recurrent_attention_only_variant_reuses_attention_surface() {
        let variant = recurrent_goe_over_attention_only_variant();
        variant.validate().unwrap();
        assert_eq!(
            variant.base_variant.kind,
            crate::hybrid_attention::HybridAttentionVariantKind::AttentionOnly
        );
    }

    #[test]
    fn recurrent_attention_only_model_returns_logits_and_rounds() {
        let device = Default::default();
        let variant = recurrent_goe_over_attention_only_variant();
        let model = build_attention_only_recurrent_graph_of_experts_model::<TestBackend>(
            257, &variant, &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(
            model.forward_logits(input.clone()).unwrap().dims(),
            [2, 8, 257]
        );
        let probe = model.routing_probe(input).unwrap();
        assert_eq!(probe.round_weights.len(), variant.router.round_count);
        assert_eq!(probe.final_weights.dims(), [2, 8, 2]);
    }

    #[test]
    fn recurrent_attention_only_runtime_rejects_feedback_until_implemented() {
        let device = Default::default();
        let variant = recurrent_goe_over_attention_only_variant_with_router(
            RecurrentRouterSpec::minimal_dense_gru_with_expert_feedback(),
        );
        let error = build_attention_only_recurrent_graph_of_experts_model::<TestBackend>(
            257, &variant, &device,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("expert-feedback is not implemented"));
    }
}
