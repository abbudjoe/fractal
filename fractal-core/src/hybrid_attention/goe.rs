use burn::{
    module::{Ignored, Module, Param},
    nn::{Embedding, EmbeddingConfig, Initializer},
    tensor::{
        activation::{sigmoid, softmax},
        backend::Backend,
        Int, Tensor, TensorData,
    },
};
use serde::{Deserialize, Serialize};

use super::{
    build_attention_only_hybrid_attention_model,
    build_rust_mamba3_reference_hybrid_attention_model, AttentionOnlyHybridAttentionModel,
    HybridAttentionModelShape, HybridAttentionVariantKind, HybridAttentionVariantSpec,
    RustMamba3ReferenceHybridAttentionModel,
};
use crate::{
    error::FractalError,
    phase1_hybrid_attention_baseline_matrix,
    primitives::one_minus,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

const GOE_INIT_MIN: f64 = -0.08;
const GOE_INIT_MAX: f64 = 0.08;
const GOE_MAX_NEIGHBOR_MIX: f64 = 0.25;
const GOE_NEIGHBOR_MIX_INIT_LOGIT: f32 = -4.0;
pub const GOE_CHANNEL_COUNT: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphOfExpertsBackboneKind {
    AttentionOnly,
    ReferenceSsmHybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphOfExpertsRoutingMode {
    UniformAverage,
    TokenLocalRouter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphOfExpertsTopology {
    None,
    TwoNodeLine,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphOfExpertsControllerSpec {
    pub routing_mode: GraphOfExpertsRoutingMode,
    pub topology: GraphOfExpertsTopology,
    pub channel_count: usize,
}

impl GraphOfExpertsControllerSpec {
    pub const fn uniform_average() -> Self {
        Self {
            routing_mode: GraphOfExpertsRoutingMode::UniformAverage,
            topology: GraphOfExpertsTopology::None,
            channel_count: GOE_CHANNEL_COUNT,
        }
    }

    pub const fn routed_no_graph_mix() -> Self {
        Self {
            routing_mode: GraphOfExpertsRoutingMode::TokenLocalRouter,
            topology: GraphOfExpertsTopology::None,
            channel_count: GOE_CHANNEL_COUNT,
        }
    }

    pub const fn two_channel_line() -> Self {
        Self {
            routing_mode: GraphOfExpertsRoutingMode::TokenLocalRouter,
            topology: GraphOfExpertsTopology::TwoNodeLine,
            channel_count: GOE_CHANNEL_COUNT,
        }
    }

    pub const fn label_suffix(&self) -> &'static str {
        match (self.routing_mode, self.topology) {
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::None) => {
                "uniform-average"
            }
            (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::None) => {
                "routed-no-graph-mix"
            }
            (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::TwoNodeLine) => {
                "routed-graph-mix"
            }
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::TwoNodeLine) => {
                "invalid"
            }
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.channel_count != GOE_CHANNEL_COUNT {
            return Err(FractalError::InvalidConfig(format!(
                "graph_of_experts.controller.channel_count must remain {GOE_CHANNEL_COUNT} for the minimal scaffold, got {}",
                self.channel_count
            )));
        }
        match (self.routing_mode, self.topology) {
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::None)
            | (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::None)
            | (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::TwoNodeLine) => {
                Ok(())
            }
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::TwoNodeLine) => {
                Err(FractalError::InvalidConfig(
                    "graph_of_experts.controller uniform-average may not enable graph topology"
                        .to_string(),
                ))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphOfExpertsVariantSpec {
    pub label: String,
    pub backbone_kind: GraphOfExpertsBackboneKind,
    pub controller: GraphOfExpertsControllerSpec,
    pub base_variant: HybridAttentionVariantSpec,
}

impl GraphOfExpertsVariantSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.label.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts.variant.label must be non-empty".to_string(),
            ));
        }
        self.controller.validate()?;
        self.base_variant.validate()?;
        match (self.backbone_kind, self.base_variant.kind) {
            (
                GraphOfExpertsBackboneKind::AttentionOnly,
                HybridAttentionVariantKind::AttentionOnly,
            )
            | (
                GraphOfExpertsBackboneKind::ReferenceSsmHybrid,
                HybridAttentionVariantKind::ReferenceSsmHybrid,
            ) => Ok(()),
            (GraphOfExpertsBackboneKind::AttentionOnly, actual) => Err(
                FractalError::InvalidConfig(format!(
                    "graph_of_experts.variant[{}] attention-only backbone requires an attention-only base variant, got {actual:?}",
                    self.label
                )),
            ),
            (GraphOfExpertsBackboneKind::ReferenceSsmHybrid, actual) => Err(
                FractalError::InvalidConfig(format!(
                    "graph_of_experts.variant[{}] reference-ssm backbone requires a reference-ssm-hybrid base variant, got {actual:?}",
                    self.label
                )),
            ),
        }
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

pub fn goe_over_attention_only_variant() -> GraphOfExpertsVariantSpec {
    goe_over_attention_only_variant_with_controller(GraphOfExpertsControllerSpec::two_channel_line())
}

pub fn goe_over_attention_only_variant_with_controller(
    controller: GraphOfExpertsControllerSpec,
) -> GraphOfExpertsVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    GraphOfExpertsVariantSpec {
        label: format!("dreegmor-over-a-{}", controller.label_suffix()),
        backbone_kind: GraphOfExpertsBackboneKind::AttentionOnly,
        controller,
        base_variant: matrix.attention_only,
    }
}

pub fn goe_over_reference_ssm_variant() -> GraphOfExpertsVariantSpec {
    goe_over_reference_ssm_variant_with_controller(GraphOfExpertsControllerSpec::two_channel_line())
}

pub fn goe_over_reference_ssm_variant_with_controller(
    controller: GraphOfExpertsControllerSpec,
) -> GraphOfExpertsVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    GraphOfExpertsVariantSpec {
        label: format!("dreegmor-over-a-plus-m-{}", controller.label_suffix()),
        backbone_kind: GraphOfExpertsBackboneKind::ReferenceSsmHybrid,
        controller,
        base_variant: matrix.reference_ssm_hybrid,
    }
}

#[derive(Debug)]
pub struct GraphOfExpertsRoutingProbe<B: Backend> {
    pub pre_graph_weights: Tensor<B, 3>,
    pub final_weights: Tensor<B, 3>,
    pub round_weights: Vec<Tensor<B, 3>>,
}

#[derive(Module, Debug)]
struct TwoChannelGraphOfExpertsRouter<B: Backend> {
    routing_mode: Ignored<GraphOfExpertsRoutingMode>,
    topology: Ignored<GraphOfExpertsTopology>,
    token_embedding: Option<Embedding<B>>,
    route_projection: Option<StructuredProjection<B>>,
    neighbor_mix_logit: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> TwoChannelGraphOfExpertsRouter<B> {
    fn new(
        vocab_size: usize,
        d_model: usize,
        controller: GraphOfExpertsControllerSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        if vocab_size == 0 || d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "graph_of_experts.router requires positive vocab_size and d_model".to_string(),
            ));
        }
        controller.validate()?;
        let (token_embedding, route_projection) = match controller.routing_mode {
            GraphOfExpertsRoutingMode::UniformAverage => (None, None),
            GraphOfExpertsRoutingMode::TokenLocalRouter => (
                Some(
                    EmbeddingConfig::new(vocab_size, d_model)
                        .with_initializer(Initializer::Uniform {
                            min: GOE_INIT_MIN,
                            max: GOE_INIT_MAX,
                        })
                        .init(device),
                ),
                Some(
                    StructuredProjectionConfig::new(d_model, GOE_CHANNEL_COUNT)
                        .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                        .init(device),
                ),
            ),
        };
        let neighbor_mix_logit = match controller.topology {
            GraphOfExpertsTopology::None => None,
            GraphOfExpertsTopology::TwoNodeLine => Some(Param::from_data(
                TensorData::new(vec![GOE_NEIGHBOR_MIX_INIT_LOGIT], [1]),
                device,
            )),
        };
        Ok(Self {
            routing_mode: Ignored(controller.routing_mode),
            topology: Ignored(controller.topology),
            token_embedding,
            route_projection,
            neighbor_mix_logit,
        })
    }

    fn routing_probe(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<GraphOfExpertsRoutingProbe<B>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let pre_graph_weights = match *self.routing_mode {
            GraphOfExpertsRoutingMode::UniformAverage => {
                uniform_channel_weights(batch_size, seq_len, &input_ids.device())
            }
            GraphOfExpertsRoutingMode::TokenLocalRouter => {
                let route_logits = self.route_logits(input_ids.clone())?;
                softmax(route_logits, 2)
            }
        };
        let final_weights = match (*self.routing_mode, *self.topology) {
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::None)
            | (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::None) => {
                pre_graph_weights.clone()
            }
            (GraphOfExpertsRoutingMode::TokenLocalRouter, GraphOfExpertsTopology::TwoNodeLine) => {
                let route_logits = self.route_logits(input_ids)?;
                let neighbor_logits = swap_two_channel_order(route_logits.clone());
                let neighbor_mix = self.neighbor_mix_tensor(batch_size, seq_len)?;
                let mixed_logits = route_logits.clone() * one_minus(neighbor_mix.clone())
                    + neighbor_logits * neighbor_mix;
                softmax(mixed_logits, 2)
            }
            (GraphOfExpertsRoutingMode::UniformAverage, GraphOfExpertsTopology::TwoNodeLine) => {
                return Err(FractalError::InvalidConfig(
                    "graph_of_experts.router uniform-average may not enable graph topology"
                        .to_string(),
                ))
            }
        };
        Ok(GraphOfExpertsRoutingProbe {
            pre_graph_weights: pre_graph_weights.clone(),
            final_weights: final_weights.clone(),
            round_weights: vec![pre_graph_weights, final_weights],
        })
    }

    fn route_weights(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        Ok(self.routing_probe(input_ids)?.final_weights)
    }

    fn route_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        let token_embedding = self.token_embedding.as_ref().ok_or_else(|| {
            FractalError::InvalidState(
                "graph_of_experts.router token_embedding missing for token-local routing"
                    .to_string(),
            )
        })?;
        let route_projection = self.route_projection.as_ref().ok_or_else(|| {
            FractalError::InvalidState(
                "graph_of_experts.router route_projection missing for token-local routing"
                    .to_string(),
            )
        })?;
        Ok(route_projection.forward(token_embedding.forward(input_ids)))
    }

    fn neighbor_mix_tensor(
        &self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let neighbor_mix_logit = self.neighbor_mix_logit.as_ref().ok_or_else(|| {
            FractalError::InvalidState(
                "graph_of_experts.router neighbor_mix_logit missing for graph topology".to_string(),
            )
        })?;
        Ok(sigmoid(neighbor_mix_logit.val())
            .mul_scalar(GOE_MAX_NEIGHBOR_MIX)
            .reshape([1, 1, 1])
            .repeat(&[batch_size, seq_len, GOE_CHANNEL_COUNT]))
    }

    fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        match self.neighbor_mix_logit.as_ref() {
            Some(neighbor_mix_logit) => {
                let mix = sigmoid(neighbor_mix_logit.val())
                    .mul_scalar(GOE_MAX_NEIGHBOR_MIX)
                    .to_data()
                    .to_vec::<f32>()
                    .map_err(invalid_state_from_data(
                        "graph_of_experts.router.edge_mix_fraction",
                    ))?;
                Ok(mix.first().copied().unwrap_or_default() as f64)
            }
            None => Ok(0.0),
        }
    }
}

#[derive(Module, Debug)]
pub struct AttentionOnlyGraphOfExpertsModel<B: Backend> {
    router: TwoChannelGraphOfExpertsRouter<B>,
    expert_a: AttentionOnlyHybridAttentionModel<B>,
    expert_b: AttentionOnlyHybridAttentionModel<B>,
    variant: Ignored<GraphOfExpertsVariantSpec>,
}

impl<B: Backend> AttentionOnlyGraphOfExpertsModel<B> {
    pub fn new(
        vocab_size: usize,
        variant: &GraphOfExpertsVariantSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        variant.validate()?;
        if !matches!(
            variant.backbone_kind,
            GraphOfExpertsBackboneKind::AttentionOnly
        ) {
            return Err(FractalError::InvalidConfig(format!(
                "graph_of_experts.variant[{}] must use an attention-only backbone for AttentionOnlyGraphOfExpertsModel",
                variant.label
            )));
        }
        let shape = variant.shape(vocab_size);
        Ok(Self {
            router: TwoChannelGraphOfExpertsRouter::new(
                vocab_size,
                shape.d_model,
                variant.controller.clone(),
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
    ) -> Result<GraphOfExpertsRoutingProbe<B>, FractalError> {
        self.router.routing_probe(input_ids)
    }

    pub fn route_weights(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        self.router.route_weights(input_ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.expert_a.shape().vocab_size
    }

    pub fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        self.router.edge_mix_fraction()
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

#[derive(Module, Debug)]
pub struct ReferenceSsmGraphOfExpertsModel<B: Backend> {
    router: TwoChannelGraphOfExpertsRouter<B>,
    expert_a: RustMamba3ReferenceHybridAttentionModel<B>,
    expert_b: RustMamba3ReferenceHybridAttentionModel<B>,
    variant: Ignored<GraphOfExpertsVariantSpec>,
}

impl<B: Backend> ReferenceSsmGraphOfExpertsModel<B> {
    pub fn new(
        vocab_size: usize,
        variant: &GraphOfExpertsVariantSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        variant.validate()?;
        if !matches!(
            variant.backbone_kind,
            GraphOfExpertsBackboneKind::ReferenceSsmHybrid
        ) {
            return Err(FractalError::InvalidConfig(format!(
                "graph_of_experts.variant[{}] must use a reference-ssm-hybrid backbone for ReferenceSsmGraphOfExpertsModel",
                variant.label
            )));
        }
        let shape = variant.shape(vocab_size);
        Ok(Self {
            router: TwoChannelGraphOfExpertsRouter::new(
                vocab_size,
                shape.d_model,
                variant.controller.clone(),
                device,
            )?,
            expert_a: build_rust_mamba3_reference_hybrid_attention_model::<B>(
                vocab_size,
                &variant.base_variant,
                device,
            )?,
            expert_b: build_rust_mamba3_reference_hybrid_attention_model::<B>(
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
    ) -> Result<GraphOfExpertsRoutingProbe<B>, FractalError> {
        self.router.routing_probe(input_ids)
    }

    pub fn route_weights(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        self.router.route_weights(input_ids)
    }

    pub fn vocab_size(&self) -> usize {
        self.expert_a.shape().vocab_size
    }

    pub fn edge_mix_fraction(&self) -> Result<f64, FractalError> {
        self.router.edge_mix_fraction()
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

pub fn build_attention_only_graph_of_experts_model<B: Backend>(
    vocab_size: usize,
    variant: &GraphOfExpertsVariantSpec,
    device: &B::Device,
) -> Result<AttentionOnlyGraphOfExpertsModel<B>, FractalError> {
    AttentionOnlyGraphOfExpertsModel::new(vocab_size, variant, device)
}

pub fn build_reference_ssm_graph_of_experts_model<B: Backend>(
    vocab_size: usize,
    variant: &GraphOfExpertsVariantSpec,
    device: &B::Device,
) -> Result<ReferenceSsmGraphOfExpertsModel<B>, FractalError> {
    ReferenceSsmGraphOfExpertsModel::new(vocab_size, variant, device)
}

fn swap_two_channel_order<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch_size, seq_len, channel_count] = tensor.dims();
    debug_assert_eq!(channel_count, GOE_CHANNEL_COUNT);
    Tensor::cat(
        vec![
            tensor.clone().slice([0..batch_size, 0..seq_len, 1..2]),
            tensor.slice([0..batch_size, 0..seq_len, 0..1]),
        ],
        2,
    )
}

fn uniform_channel_weights<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(
            vec![0.5f32; batch_size * seq_len * GOE_CHANNEL_COUNT],
            [batch_size, seq_len, GOE_CHANNEL_COUNT],
        ),
        device,
    )
}

fn combine_two_channel_logits<B: Backend>(
    expert_a: Tensor<B, 3>,
    expert_b: Tensor<B, 3>,
    weights: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [batch_size, seq_len, vocab_size] = expert_a.dims();
    debug_assert_eq!(expert_b.dims(), [batch_size, seq_len, vocab_size]);
    debug_assert_eq!(weights.dims(), [batch_size, seq_len, GOE_CHANNEL_COUNT]);
    let stacked = Tensor::cat(
        vec![
            expert_a.reshape([batch_size, seq_len, 1, vocab_size]),
            expert_b.reshape([batch_size, seq_len, 1, vocab_size]),
        ],
        2,
    );
    let expanded_weights = weights
        .reshape([batch_size, seq_len, GOE_CHANNEL_COUNT, 1])
        .repeat(&[1, 1, 1, vocab_size]);
    (stacked * expanded_weights)
        .sum_dim(2)
        .reshape([batch_size, seq_len, vocab_size])
}

fn invalid_state_from_data(
    subject: &'static str,
) -> impl FnOnce(burn::tensor::DataError) -> FractalError {
    move |error| {
        FractalError::InvalidState(format!(
            "{subject} could not be materialized for inspection: {error}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    use burn::tensor::{Int, Tensor};

    use super::{
        build_attention_only_graph_of_experts_model, build_reference_ssm_graph_of_experts_model,
        goe_over_attention_only_variant, goe_over_attention_only_variant_with_controller,
        goe_over_reference_ssm_variant, GraphOfExpertsBackboneKind, GraphOfExpertsControllerSpec,
        GraphOfExpertsRoutingMode, GraphOfExpertsTopology, GraphOfExpertsVariantSpec,
        GOE_CHANNEL_COUNT,
    };
    use crate::phase1_hybrid_attention_baseline_matrix;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn goe_attention_only_variant_reuses_frozen_attention_surface() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let variant = goe_over_attention_only_variant();

        assert_eq!(
            variant.backbone_kind,
            GraphOfExpertsBackboneKind::AttentionOnly
        );
        assert_eq!(variant.base_variant, matrix.attention_only);
        variant.validate().unwrap();
    }

    #[test]
    fn goe_reference_variant_reuses_frozen_reference_surface() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let variant = goe_over_reference_ssm_variant();

        assert_eq!(
            variant.backbone_kind,
            GraphOfExpertsBackboneKind::ReferenceSsmHybrid
        );
        assert_eq!(variant.base_variant, matrix.reference_ssm_hybrid);
        variant.validate().unwrap();
    }

    #[test]
    fn controller_supports_incremental_structure_ladder() {
        for controller in [
            GraphOfExpertsControllerSpec::uniform_average(),
            GraphOfExpertsControllerSpec::routed_no_graph_mix(),
            GraphOfExpertsControllerSpec::two_channel_line(),
        ] {
            controller.validate().unwrap();
        }
    }

    #[test]
    fn controller_rejects_uniform_average_with_graph_topology() {
        let controller = GraphOfExpertsControllerSpec {
            routing_mode: GraphOfExpertsRoutingMode::UniformAverage,
            topology: GraphOfExpertsTopology::TwoNodeLine,
            channel_count: GOE_CHANNEL_COUNT,
        };
        assert!(controller.validate().is_err());
    }

    #[test]
    fn goe_attention_only_model_returns_logits() {
        let device = Default::default();
        let variant = goe_over_attention_only_variant();
        let model =
            build_attention_only_graph_of_experts_model::<TestBackend>(257, &variant, &device)
                .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn goe_reference_model_returns_logits() {
        let device = Default::default();
        let variant = goe_over_reference_ssm_variant();
        let model =
            build_reference_ssm_graph_of_experts_model::<TestBackend>(257, &variant, &device)
                .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn routing_weights_stay_normalized() {
        let device = Default::default();
        let variant = goe_over_attention_only_variant();
        let model =
            build_attention_only_graph_of_experts_model::<TestBackend>(257, &variant, &device)
                .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let weights = model.route_weights(input).unwrap();
        let sums = weights
            .clone()
            .sum_dim(2)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(weights.dims(), [1, 4, GOE_CHANNEL_COUNT]);
        for value in sums {
            assert!((value - 1.0).abs() < 1.0e-4);
        }
    }

    #[test]
    fn uniform_average_controller_emits_balanced_weights() {
        let device = Default::default();
        let variant = goe_over_attention_only_variant_with_controller(
            GraphOfExpertsControllerSpec::uniform_average(),
        );
        let model =
            build_attention_only_graph_of_experts_model::<TestBackend>(257, &variant, &device)
                .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let weights = model.route_weights(input).unwrap();
        let values = weights.to_data().to_vec::<f32>().unwrap();
        for pair in values.chunks_exact(2) {
            assert!((pair[0] - 0.5).abs() < 1.0e-6);
            assert!((pair[1] - 0.5).abs() < 1.0e-6);
        }
    }

    #[test]
    fn validation_rejects_mismatched_backbone_and_base_variant() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let variant = GraphOfExpertsVariantSpec {
            label: "bad".to_string(),
            backbone_kind: GraphOfExpertsBackboneKind::AttentionOnly,
            controller: GraphOfExpertsControllerSpec::two_channel_line(),
            base_variant: matrix.reference_ssm_hybrid,
        };
        assert!(variant.validate().is_err());
    }
}
