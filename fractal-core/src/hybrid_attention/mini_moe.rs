use std::collections::BTreeSet;

use burn::{
    module::{Ignored, Module, ModuleDisplay},
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Embedding, EmbeddingConfig, Initializer, LayerNorm, LayerNormConfig,
    },
    tensor::{
        activation::{gelu, relu, softmax},
        backend::Backend,
        Bool, IndexingUpdateOp, Int, Tensor, TensorData,
    },
};
use serde::{Deserialize, Serialize};

use super::common::local_causal_mask;
use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

const MINI_MOE_SCHEMA_VERSION: u32 = 1;
const MINI_MOE_INIT_MIN: f64 = -0.08;
const MINI_MOE_INIT_MAX: f64 = 0.08;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeSurfaceSpec {
    pub architecture: MiniMoeArchitectureSpec,
    pub runtime: MiniMoeRuntimeSpec,
    pub observability: MiniMoeObservabilitySpec,
}

impl MiniMoeSurfaceSpec {
    pub fn phase1_reference_default() -> Self {
        Self {
            architecture: MiniMoeArchitectureSpec {
                schema_version: MINI_MOE_SCHEMA_VERSION,
                preset: Some(MiniMoePreset::Phase1Reference),
                label: "dreegmor-mini-moe-reference".to_string(),
                backbone: MiniMoeBackboneSpec {
                    vocab_size: 257,
                    hidden_dim: 128,
                    head_count: 4,
                    total_layers: 8,
                    local_window: 256,
                    ffn_multiplier: 4,
                },
                moe: MiniMoeStackSpec {
                    experts_per_block: 4,
                    active_experts_per_token: 1,
                    moe_layer_schedule: MiniMoeLayerSchedule::AllLayers,
                    expert_ffn_multiplier: 4,
                    load_balance_loss_weight: 1.0e-2,
                },
                router: MiniMoeRouterSpec::OneShot(OneShotRouterSpec {}),
            },
            runtime: MiniMoeRuntimeSpec {
                dispatch: MiniMoeDispatchSpec {
                    mode: MiniMoeDispatchMode::SparseTopK,
                },
            },
            observability: MiniMoeObservabilitySpec {
                sampling: TraceSamplingPolicy {
                    token_trace_budget_per_layer: 8,
                    expert_example_budget_per_layer: 4,
                    deterministic: true,
                    sample_seed: 42,
                },
                capture_round_summaries: true,
            },
        }
    }

    pub fn phase1_recurrent_default() -> Self {
        let mut surface = Self::phase1_reference_default();
        surface.architecture.preset = Some(MiniMoePreset::Phase1Recurrent);
        surface.architecture.label = "dreegmor-mini-moe-recurrent".to_string();
        surface.architecture.router =
            MiniMoeRouterSpec::RecurrentPreExpert(RecurrentPreExpertRouterSpec {
                round_count: 2,
                state_dim: 64,
            });
        surface
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        self.architecture.validate()?;
        self.runtime.validate()?;
        self.observability.validate()?;
        let resolved_dispatch =
            self.runtime
                .resolve_dispatch_contract(self.architecture.moe.active_experts_per_token);
        if resolved_dispatch.active_experts_per_token
            != self.architecture.moe.active_experts_per_token
        {
            return Err(FractalError::InvalidConfig(
                "mini_moe.surface resolved active_experts_per_token must match architecture"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub fn resolve_layout(&self) -> Result<ResolvedMiniMoeLayout, FractalError> {
        self.architecture
            .moe
            .moe_layer_schedule
            .resolve(self.architecture.backbone.total_layers)
    }

    pub fn resolve_dispatch_contract(&self) -> ResolvedDispatchContract {
        self.runtime
            .resolve_dispatch_contract(self.architecture.moe.active_experts_per_token)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeArchitectureSpec {
    pub schema_version: u32,
    pub preset: Option<MiniMoePreset>,
    pub label: String,
    pub backbone: MiniMoeBackboneSpec,
    pub moe: MiniMoeStackSpec,
    pub router: MiniMoeRouterSpec,
}

impl MiniMoeArchitectureSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.schema_version != MINI_MOE_SCHEMA_VERSION {
            return Err(FractalError::InvalidConfig(format!(
                "mini_moe.architecture.schema_version must be {}, got {}",
                MINI_MOE_SCHEMA_VERSION, self.schema_version
            )));
        }
        if self.label.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "mini_moe.architecture.label must be non-empty".to_string(),
            ));
        }
        self.backbone.validate()?;
        self.moe.validate()?;
        self.router.validate(&self.moe)?;
        let resolved = self
            .moe
            .moe_layer_schedule
            .resolve(self.backbone.total_layers)?;
        if resolved.moe_layers.is_empty() {
            return Err(FractalError::InvalidConfig(
                "mini_moe.architecture must resolve at least one moe layer".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeBackboneSpec {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub head_count: usize,
    pub total_layers: usize,
    pub local_window: usize,
    pub ffn_multiplier: usize,
}

impl MiniMoeBackboneSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.vocab_size == 0
            || self.hidden_dim == 0
            || self.head_count == 0
            || self.total_layers == 0
            || self.local_window == 0
            || self.ffn_multiplier == 0
        {
            return Err(FractalError::InvalidConfig(
                "mini_moe.backbone dimensions and counts must be greater than zero".to_string(),
            ));
        }
        if self.hidden_dim % self.head_count != 0 {
            return Err(FractalError::InvalidConfig(format!(
                "mini_moe.backbone.hidden_dim {} must be divisible by head_count {}",
                self.hidden_dim, self.head_count
            )));
        }
        Ok(())
    }

    pub fn dense_ffn_width(&self) -> usize {
        self.hidden_dim * self.ffn_multiplier
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeStackSpec {
    pub experts_per_block: usize,
    pub active_experts_per_token: usize,
    pub moe_layer_schedule: MiniMoeLayerSchedule,
    pub expert_ffn_multiplier: usize,
    pub load_balance_loss_weight: f64,
}

impl MiniMoeStackSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.experts_per_block < 2 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.moe.experts_per_block must be at least 2".to_string(),
            ));
        }
        if self.active_experts_per_token == 0
            || self.active_experts_per_token > self.experts_per_block
        {
            return Err(FractalError::InvalidConfig(format!(
                "mini_moe.moe.active_experts_per_token must be between 1 and experts_per_block {}, got {}",
                self.experts_per_block, self.active_experts_per_token
            )));
        }
        if self.expert_ffn_multiplier == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.moe.expert_ffn_multiplier must be greater than zero".to_string(),
            ));
        }
        if !(self.load_balance_loss_weight.is_finite() && self.load_balance_loss_weight >= 0.0) {
            return Err(FractalError::InvalidConfig(
                "mini_moe.moe.load_balance_loss_weight must be finite and non-negative"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeLayerSchedule {
    AllLayers,
    EveryN { n: usize },
    Explicit(Vec<usize>),
}

impl MiniMoeLayerSchedule {
    pub fn resolve(&self, total_layers: usize) -> Result<ResolvedMiniMoeLayout, FractalError> {
        if total_layers == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.schedule requires total_layers > 0".to_string(),
            ));
        }
        let mut moe_layers = match self {
            Self::AllLayers => (0..total_layers).collect::<Vec<_>>(),
            Self::EveryN { n } => {
                if *n == 0 {
                    return Err(FractalError::InvalidConfig(
                        "mini_moe.schedule.every_n.n must be greater than zero".to_string(),
                    ));
                }
                (0..total_layers).step_by(*n).collect::<Vec<_>>()
            }
            Self::Explicit(indices) => indices.clone(),
        };
        moe_layers.sort_unstable();
        moe_layers.dedup();
        if moe_layers.is_empty() {
            return Err(FractalError::InvalidConfig(
                "mini_moe.schedule resolved no moe layers".to_string(),
            ));
        }
        if let Some(invalid) = moe_layers.iter().find(|&&index| index >= total_layers) {
            return Err(FractalError::InvalidConfig(format!(
                "mini_moe.schedule layer index {} must be less than total_layers {}",
                invalid, total_layers
            )));
        }
        let dense_layers = (0..total_layers)
            .filter(|index| !moe_layers.contains(index))
            .collect::<Vec<_>>();
        Ok(ResolvedMiniMoeLayout {
            moe_layers,
            dense_layers,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeLayerSchedulePreset {
    AllLayers,
    AlternatingFromZero,
    AlternatingFromOne,
}

impl MiniMoeLayerSchedulePreset {
    pub fn lower(self, total_layers: usize) -> MiniMoeLayerSchedule {
        match self {
            Self::AllLayers => MiniMoeLayerSchedule::AllLayers,
            Self::AlternatingFromZero => {
                MiniMoeLayerSchedule::Explicit((0..total_layers).step_by(2).collect())
            }
            Self::AlternatingFromOne => {
                MiniMoeLayerSchedule::Explicit((1..total_layers).step_by(2).collect())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedMiniMoeLayout {
    pub moe_layers: Vec<usize>,
    pub dense_layers: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeRouterSpec {
    OneShot(OneShotRouterSpec),
    RecurrentPreExpert(RecurrentPreExpertRouterSpec),
}

impl MiniMoeRouterSpec {
    pub fn validate(&self, moe: &MiniMoeStackSpec) -> Result<(), FractalError> {
        match self {
            Self::OneShot(spec) => spec.validate(),
            Self::RecurrentPreExpert(spec) => spec.validate(moe),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OneShotRouterSpec {}

impl OneShotRouterSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentPreExpertRouterSpec {
    pub round_count: usize,
    pub state_dim: usize,
}

impl RecurrentPreExpertRouterSpec {
    pub fn validate(&self, _moe: &MiniMoeStackSpec) -> Result<(), FractalError> {
        if self.round_count < 2 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.recurrent_pre_expert_router.round_count must be at least 2"
                    .to_string(),
            ));
        }
        if self.state_dim == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.recurrent_pre_expert_router.state_dim must be greater than zero"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeRuntimeSpec {
    pub dispatch: MiniMoeDispatchSpec,
}

impl MiniMoeRuntimeSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.dispatch.validate()
    }

    pub fn resolve_dispatch_contract(
        &self,
        active_experts_per_token: usize,
    ) -> ResolvedDispatchContract {
        ResolvedDispatchContract {
            mode: self.dispatch.mode,
            active_experts_per_token,
            tie_break: TieBreakPolicy::LowestIndex,
            capacity: DispatchCapacityPolicy::Unlimited,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeDispatchSpec {
    pub mode: MiniMoeDispatchMode,
}

impl MiniMoeDispatchSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeDispatchMode {
    SparseTopK,
    DenseDebug,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TieBreakPolicy {
    LowestIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchCapacityPolicy {
    Unlimited,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDispatchContract {
    pub mode: MiniMoeDispatchMode,
    pub active_experts_per_token: usize,
    pub tie_break: TieBreakPolicy,
    pub capacity: DispatchCapacityPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoePreset {
    Phase1Reference,
    Phase1Recurrent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedAttentionSpec {
    pub hidden_dim: usize,
    pub head_count: usize,
    pub local_window: usize,
}

#[derive(Debug)]
pub struct AttentionForwardOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub trace: Option<AttentionTrace<B>>,
}

#[derive(Debug)]
pub struct AttentionTrace<B: Backend> {
    pub attention_scores: Option<Tensor<B, 4>>,
}

pub trait AttentionSublayer<B: Backend>: Module<B> + ModuleDisplay + Clone {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Result<AttentionForwardOutput<B>, FractalError>;
}

#[derive(Module, Debug)]
pub struct SharedAttentionSublayer<B: Backend> {
    spec: Ignored<SharedAttentionSpec>,
    inner: MultiHeadAttention<B>,
}

impl<B: Backend> SharedAttentionSublayer<B> {
    pub fn new(spec: SharedAttentionSpec, device: &B::Device) -> Result<Self, FractalError> {
        if spec.hidden_dim == 0 || spec.head_count == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.shared_attention requires positive hidden_dim and head_count"
                    .to_string(),
            ));
        }
        if spec.hidden_dim % spec.head_count != 0 {
            return Err(FractalError::InvalidConfig(format!(
                "mini_moe.shared_attention hidden_dim {} must be divisible by head_count {}",
                spec.hidden_dim, spec.head_count
            )));
        }
        let inner = MultiHeadAttentionConfig::new(spec.hidden_dim, spec.head_count)
            .with_dropout(0.0)
            .with_initializer(Initializer::Uniform {
                min: MINI_MOE_INIT_MIN,
                max: MINI_MOE_INIT_MAX,
            })
            .init(device);
        Ok(Self {
            spec: Ignored(spec),
            inner,
        })
    }
}

impl<B: Backend> AttentionSublayer<B> for SharedAttentionSublayer<B> {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Result<AttentionForwardOutput<B>, FractalError> {
        let output = self.inner.forward(MhaInput::self_attn(hidden).mask_attn(mask));
        Ok(AttentionForwardOutput {
            hidden: output.context,
            trace: Some(AttentionTrace {
                attention_scores: Some(output.weights),
            }),
        })
    }
}

#[derive(Debug)]
pub struct FfnForwardOutput<B: Backend, T> {
    pub hidden: Tensor<B, 3>,
    pub trace: T,
}

pub trait FfnSublayer<B: Backend>: Module<B> + ModuleDisplay + Clone {
    type Trace;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertId {
    pub layer_index: usize,
    pub expert_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertActivationKind {
    Gelu,
    Silu,
    Relu,
    SquaredRelu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertFfnSpec {
    pub hidden_dim: usize,
    pub expansion_dim: usize,
    pub activation: ExpertActivationKind,
    pub gated: bool,
}

#[derive(Module, Debug)]
pub struct ExpertFfn<B: Backend> {
    id: Ignored<ExpertId>,
    spec: Ignored<ExpertFfnSpec>,
    up_projection: StructuredProjection<B>,
    gate_projection: Option<StructuredProjection<B>>,
    down_projection: StructuredProjection<B>,
}

impl<B: Backend> ExpertFfn<B> {
    pub fn new(
        id: ExpertId,
        spec: ExpertFfnSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        if spec.hidden_dim == 0 || spec.expansion_dim == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.expert_ffn requires positive hidden_dim and expansion_dim"
                    .to_string(),
            ));
        }
        let projection = |d_input, d_output| {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(Initializer::Uniform {
                    min: MINI_MOE_INIT_MIN,
                    max: MINI_MOE_INIT_MAX,
                })
                .init(device)
        };
        Ok(Self {
            id: Ignored(id),
            spec: Ignored(spec.clone()),
            up_projection: projection(spec.hidden_dim, spec.expansion_dim),
            gate_projection: spec.gated.then(|| projection(spec.hidden_dim, spec.expansion_dim)),
            down_projection: projection(spec.expansion_dim, spec.hidden_dim),
        })
    }

    pub fn forward(&self, hidden: Tensor<B, 2>) -> Result<Tensor<B, 2>, FractalError> {
        let mut activated = activate(self.spec.activation.clone(), self.up_projection.forward(hidden.clone()));
        if let Some(gate_projection) = &self.gate_projection {
            let gate = silu_like(gate_projection.forward(hidden));
            activated = activated * gate;
        }
        Ok(self.down_projection.forward(activated))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteSiteId {
    pub layer_index: usize,
}

pub struct PreExpertRouterInput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub site: RouteSiteId,
}

pub trait PreExpertRouterController<B: Backend>: Module<B> + ModuleDisplay + Clone {
    fn route(
        &self,
        input: PreExpertRouterInput<B>,
    ) -> Result<RoutePlan<B>, FractalError>;
}

#[derive(Debug)]
pub struct RoutePlan<B: Backend> {
    pub expert_logits: Tensor<B, 3>,
    pub expert_weights: Tensor<B, 3>,
    pub round_summaries: Vec<RouteRoundSummary<B>>,
}

#[derive(Debug)]
pub struct RouteRoundSummary<B: Backend> {
    pub expert_logits: Tensor<B, 3>,
    pub expert_weights: Tensor<B, 3>,
}

#[derive(Debug)]
pub struct ExpertDispatchAssignment<B: Backend> {
    pub expert_index: usize,
    pub token_indices: Tensor<B, 1, Int>,
    pub slot_indices: Tensor<B, 1, Int>,
}

#[derive(Debug)]
pub struct DispatchPlan<B: Backend> {
    pub site: RouteSiteId,
    pub mode: MiniMoeDispatchMode,
    pub selected_expert_indices: Tensor<B, 3, Int>,
    pub selected_expert_weights: Tensor<B, 3>,
    pub assignments: Vec<ExpertDispatchAssignment<B>>,
}

#[derive(Module, Debug, Clone)]
pub struct MoeDispatcher {
    contract: Ignored<ResolvedDispatchContract>,
}

impl MoeDispatcher {
    pub fn new(contract: ResolvedDispatchContract) -> Self {
        Self {
            contract: Ignored(contract),
        }
    }

    pub fn compile<B: Backend>(
        &self,
        site: RouteSiteId,
        route_plan: &RoutePlan<B>,
    ) -> Result<DispatchPlan<B>, FractalError> {
        let [batch_size, seq_len, expert_count] = route_plan.expert_weights.dims();
        let device = route_plan.expert_weights.device();
        let selected_expert_indices = match self.contract.mode {
            MiniMoeDispatchMode::SparseTopK => {
                let adjusted = apply_tie_break_bias(
                    route_plan.expert_weights.clone(),
                    self.contract.tie_break,
                    expert_count,
                );
                let (_top_weights, top_indices) =
                    adjusted.topk_with_indices(self.contract.active_experts_per_token, 2);
                top_indices
            }
            MiniMoeDispatchMode::DenseDebug => dense_debug_indices::<B>(
                batch_size,
                seq_len,
                expert_count,
                &device,
            ),
        };
        let selected_expert_weights = route_plan
            .expert_weights
            .clone()
            .gather(2, selected_expert_indices.clone());
        let assignments = build_dispatch_assignments::<B>(
            &selected_expert_indices,
            expert_count,
            &device,
        )?;
        Ok(DispatchPlan {
            site,
            mode: self.contract.mode,
            selected_expert_indices,
            selected_expert_weights,
            assignments,
        })
    }

    pub fn dispatch<B: Backend>(
        &self,
        hidden: Tensor<B, 3>,
        routed_experts: &[ExpertFfn<B>],
        dispatch_plan: &DispatchPlan<B>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len, hidden_dim] = hidden.dims();
        let flat_count = batch_size * seq_len;
        let hidden_flat = hidden.reshape([flat_count, hidden_dim]);
        let selected_width = dispatch_plan.selected_expert_weights.dims()[2];
        let selected_weights = dispatch_plan
            .selected_expert_weights
            .clone()
            .reshape([flat_count, selected_width]);
        let mut output_flat = Tensor::<B, 2>::zeros([flat_count, hidden_dim], &hidden_flat.device());

        for assignment in &dispatch_plan.assignments {
            if assignment.token_indices.dims()[0] == 0 {
                continue;
            }
            let expert = routed_experts.get(assignment.expert_index).ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "mini_moe.dispatch missing expert {} for site {}",
                    assignment.expert_index, dispatch_plan.site.layer_index
                ))
            })?;
            let token_count = assignment.token_indices.dims()[0];
            let hidden_indices = assignment
                .token_indices
                .clone()
                .reshape([token_count, 1])
                .repeat(&[1, hidden_dim]);
            let hidden_subset = hidden_flat.clone().gather(0, hidden_indices.clone());
            let expert_output = expert.forward(hidden_subset)?;

            let row_indices = assignment
                .token_indices
                .clone()
                .reshape([token_count, 1])
                .repeat(&[1, selected_width]);
            let row_subset = selected_weights.clone().gather(0, row_indices);
            let scalar_weights = row_subset.gather(
                1,
                assignment.slot_indices.clone().reshape([token_count, 1]),
            );
            let weighted = expert_output
                * scalar_weights
                    .reshape([token_count, 1])
                    .repeat(&[1, hidden_dim]);
            output_flat =
                output_flat.scatter(0, hidden_indices, weighted, IndexingUpdateOp::Add);
        }

        Ok(output_flat.reshape([batch_size, seq_len, hidden_dim]))
    }

    pub fn contract(&self) -> &ResolvedDispatchContract {
        &self.contract
    }
}

#[derive(Module, Debug)]
pub struct OneShotRouter<B: Backend> {
    route_projection: StructuredProjection<B>,
}

impl<B: Backend> OneShotRouter<B> {
    pub fn new(
        hidden_dim: usize,
        experts_per_block: usize,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        if hidden_dim == 0 || experts_per_block < 2 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.one_shot_router requires positive hidden_dim and at least 2 experts"
                    .to_string(),
            ));
        }
        Ok(Self {
            route_projection: StructuredProjectionConfig::new(hidden_dim, experts_per_block)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(Initializer::Uniform {
                    min: MINI_MOE_INIT_MIN,
                    max: MINI_MOE_INIT_MAX,
                })
                .init(device),
        })
    }
}

impl<B: Backend> PreExpertRouterController<B> for OneShotRouter<B> {
    fn route(
        &self,
        input: PreExpertRouterInput<B>,
    ) -> Result<RoutePlan<B>, FractalError> {
        let expert_logits = self.route_projection.forward(input.hidden);
        let expert_weights = softmax(expert_logits.clone(), 2);
        Ok(RoutePlan {
            expert_logits: expert_logits.clone(),
            expert_weights: expert_weights.clone(),
            round_summaries: vec![RouteRoundSummary {
                expert_logits,
                expert_weights,
            }],
        })
    }
}

#[derive(Module, Debug)]
pub struct RecurrentPreExpertRouter<B: Backend> {
    token_state_projection: StructuredProjection<B>,
    token_route_projection: StructuredProjection<B>,
    state_route_projection: StructuredProjection<B>,
    route_feedback_projection: StructuredProjection<B>,
    reset_gate_projection: StructuredProjection<B>,
    update_gate_projection: StructuredProjection<B>,
    candidate_input_projection: StructuredProjection<B>,
    candidate_state_projection: StructuredProjection<B>,
    spec: Ignored<RecurrentPreExpertRouterSpec>,
    experts_per_block: usize,
}

impl<B: Backend> RecurrentPreExpertRouter<B> {
    pub fn new(
        hidden_dim: usize,
        experts_per_block: usize,
        spec: RecurrentPreExpertRouterSpec,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        spec.validate(&MiniMoeStackSpec {
            experts_per_block,
            active_experts_per_token: 1,
            moe_layer_schedule: MiniMoeLayerSchedule::AllLayers,
            expert_ffn_multiplier: 1,
            load_balance_loss_weight: 0.0,
        })?;
        let projection = |d_input, d_output| {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(Initializer::Uniform {
                    min: MINI_MOE_INIT_MIN,
                    max: MINI_MOE_INIT_MAX,
                })
                .init(device)
        };
        Ok(Self {
            token_state_projection: projection(hidden_dim, spec.state_dim),
            token_route_projection: projection(spec.state_dim, experts_per_block),
            state_route_projection: projection(spec.state_dim, experts_per_block),
            route_feedback_projection: projection(experts_per_block, spec.state_dim),
            reset_gate_projection: projection(spec.state_dim, spec.state_dim),
            update_gate_projection: projection(spec.state_dim, spec.state_dim),
            candidate_input_projection: projection(spec.state_dim, spec.state_dim),
            candidate_state_projection: projection(spec.state_dim, spec.state_dim),
            spec: Ignored(spec),
            experts_per_block,
        })
    }
}

impl<B: Backend> PreExpertRouterController<B> for RecurrentPreExpertRouter<B> {
    fn route(
        &self,
        input: PreExpertRouterInput<B>,
    ) -> Result<RoutePlan<B>, FractalError> {
        let [batch_size, seq_len, _hidden_dim] = input.hidden.dims();
        let token_state = self.token_state_projection.forward(input.hidden);
        let token_logits = self.token_route_projection.forward(token_state.clone());
        let pooled_token_state = mean_over_tokens(token_state).tanh();
        let mut state = pooled_token_state.clone();
        let mut round_summaries = Vec::with_capacity(self.spec.round_count);

        for round_index in 0..self.spec.round_count {
            let state_bias = self
                .state_route_projection
                .forward(state.clone())
                .reshape([batch_size, 1, self.experts_per_block])
                .repeat(&[1, seq_len, 1]);
            let expert_logits = token_logits.clone() + state_bias;
            let expert_weights = softmax(expert_logits.clone(), 2);
            round_summaries.push(RouteRoundSummary {
                expert_logits: expert_logits.clone(),
                expert_weights: expert_weights.clone(),
            });

            if round_index + 1 < self.spec.round_count {
                let route_summary = mean_over_tokens(expert_weights);
                let controller_drive =
                    pooled_token_state.clone() + self.route_feedback_projection.forward(route_summary);
                let reset_gate =
                    gated_sigmoid(self.reset_gate_projection.forward(controller_drive.clone()));
                let update_gate =
                    gated_sigmoid(self.update_gate_projection.forward(controller_drive.clone()));
                let candidate = (self.candidate_input_projection.forward(controller_drive)
                    + reset_gate * self.candidate_state_projection.forward(state.clone()))
                .tanh();
                state = update_gate.clone() * state + one_minus(update_gate) * candidate;
            }
        }

        let final_summary = round_summaries.last().ok_or_else(|| {
            FractalError::InvalidState(
                "mini_moe.recurrent_pre_expert_router produced no routing rounds".to_string(),
            )
        })?;
        Ok(RoutePlan {
            expert_logits: final_summary.expert_logits.clone(),
            expert_weights: final_summary.expert_weights.clone(),
            round_summaries,
        })
    }
}

#[derive(Debug)]
pub struct MoeFfnForwardOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub route_plan: RoutePlan<B>,
    pub dispatch_plan: DispatchPlan<B>,
}

#[derive(Debug)]
pub enum MiniMoeFfnTrace<B: Backend> {
    Dense,
    Moe {
        route_plan: RoutePlan<B>,
        dispatch_plan: DispatchPlan<B>,
    },
}

#[derive(Module, Debug)]
pub struct DenseFfnSublayer<B: Backend> {
    expert: ExpertFfn<B>,
}

impl<B: Backend> DenseFfnSublayer<B> {
    pub fn new(layer_index: usize, spec: ExpertFfnSpec, device: &B::Device) -> Result<Self, FractalError> {
        Ok(Self {
            expert: ExpertFfn::new(
                ExpertId {
                    layer_index,
                    expert_index: 0,
                },
                spec,
                device,
            )?,
        })
    }
}

impl<B: Backend> FfnSublayer<B> for DenseFfnSublayer<B> {
    type Trace = MiniMoeFfnTrace<B>;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError> {
        let [batch_size, seq_len, hidden_dim] = hidden.dims();
        let output = self
            .expert
            .forward(hidden.reshape([batch_size * seq_len, hidden_dim]))?
            .reshape([batch_size, seq_len, hidden_dim]);
        Ok(FfnForwardOutput {
            hidden: output,
            trace: MiniMoeFfnTrace::Dense,
        })
    }
}

#[derive(Module, Debug)]
pub struct MoeFfnSublayer<B: Backend, R> {
    site: Ignored<RouteSiteId>,
    router: R,
    routed_experts: Vec<ExpertFfn<B>>,
    dispatcher: Ignored<MoeDispatcher>,
}

impl<B: Backend, R: PreExpertRouterController<B>> MoeFfnSublayer<B, R> {
    pub fn new(
        site: RouteSiteId,
        router: R,
        routed_experts: Vec<ExpertFfn<B>>,
        dispatcher: MoeDispatcher,
    ) -> Result<Self, FractalError> {
        if routed_experts.is_empty() {
            return Err(FractalError::InvalidConfig(
                "mini_moe.moe_ffn requires at least one routed expert".to_string(),
            ));
        }
        Ok(Self {
            site: Ignored(site),
            router,
            routed_experts,
            dispatcher: Ignored(dispatcher),
        })
    }

    pub fn forward_moe(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<MoeFfnForwardOutput<B>, FractalError> {
        let route_plan = self.router.route(PreExpertRouterInput {
            hidden: hidden.clone(),
            site: self.site.0.clone(),
        })?;
        let dispatch_plan = self.dispatcher.compile(self.site.0.clone(), &route_plan)?;
        let mixed = self
            .dispatcher
            .dispatch(hidden, &self.routed_experts, &dispatch_plan)?;
        Ok(MoeFfnForwardOutput {
            hidden: mixed,
            route_plan,
            dispatch_plan,
        })
    }
}

impl<B: Backend, R: PreExpertRouterController<B>> FfnSublayer<B> for MoeFfnSublayer<B, R> {
    type Trace = MiniMoeFfnTrace<B>;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError> {
        let output = self.forward_moe(hidden)?;
        Ok(FfnForwardOutput {
            hidden: output.hidden,
            trace: MiniMoeFfnTrace::Moe {
                route_plan: output.route_plan,
                dispatch_plan: output.dispatch_plan,
            },
        })
    }
}

#[derive(Module, Debug)]
pub struct OneShotMoeFfnSublayer<B: Backend> {
    inner: MoeFfnSublayer<B, OneShotRouter<B>>,
}

impl<B: Backend> OneShotMoeFfnSublayer<B> {
    pub fn new(
        site: RouteSiteId,
        hidden_dim: usize,
        moe: &MiniMoeStackSpec,
        dispatch: ResolvedDispatchContract,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let expert_spec = ExpertFfnSpec {
            hidden_dim,
            expansion_dim: hidden_dim * moe.expert_ffn_multiplier,
            activation: ExpertActivationKind::Gelu,
            gated: false,
        };
        let routed_experts = (0..moe.experts_per_block)
            .map(|expert_index| {
                ExpertFfn::new(
                    ExpertId {
                        layer_index: site.layer_index,
                        expert_index,
                    },
                    expert_spec.clone(),
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            inner: MoeFfnSublayer::new(
                site,
                OneShotRouter::new(hidden_dim, moe.experts_per_block, device)?,
                routed_experts,
                MoeDispatcher::new(dispatch),
            )?,
        })
    }
}

impl<B: Backend> FfnSublayer<B> for OneShotMoeFfnSublayer<B> {
    type Trace = MiniMoeFfnTrace<B>;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError> {
        self.inner.forward(hidden)
    }
}

#[derive(Module, Debug)]
pub struct RecurrentMoeFfnSublayer<B: Backend> {
    inner: MoeFfnSublayer<B, RecurrentPreExpertRouter<B>>,
}

impl<B: Backend> RecurrentMoeFfnSublayer<B> {
    pub fn new(
        site: RouteSiteId,
        hidden_dim: usize,
        moe: &MiniMoeStackSpec,
        router: &RecurrentPreExpertRouterSpec,
        dispatch: ResolvedDispatchContract,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let expert_spec = ExpertFfnSpec {
            hidden_dim,
            expansion_dim: hidden_dim * moe.expert_ffn_multiplier,
            activation: ExpertActivationKind::Gelu,
            gated: false,
        };
        let routed_experts = (0..moe.experts_per_block)
            .map(|expert_index| {
                ExpertFfn::new(
                    ExpertId {
                        layer_index: site.layer_index,
                        expert_index,
                    },
                    expert_spec.clone(),
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            inner: MoeFfnSublayer::new(
                site,
                RecurrentPreExpertRouter::new(
                    hidden_dim,
                    moe.experts_per_block,
                    router.clone(),
                    device,
                )?,
                routed_experts,
                MoeDispatcher::new(dispatch),
            )?,
        })
    }
}

impl<B: Backend> FfnSublayer<B> for RecurrentMoeFfnSublayer<B> {
    type Trace = MiniMoeFfnTrace<B>;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError> {
        self.inner.forward(hidden)
    }
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum ConfiguredMiniMoeFfn<B: Backend> {
    Dense(DenseFfnSublayer<B>),
    OneShotMoe(OneShotMoeFfnSublayer<B>),
    RecurrentMoe(RecurrentMoeFfnSublayer<B>),
}

impl<B: Backend> FfnSublayer<B> for ConfiguredMiniMoeFfn<B> {
    type Trace = MiniMoeFfnTrace<B>;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError> {
        match self {
            Self::Dense(ffn) => ffn.forward(hidden),
            Self::OneShotMoe(ffn) => ffn.forward(hidden),
            Self::RecurrentMoe(ffn) => ffn.forward(hidden),
        }
    }
}

#[derive(Debug)]
pub struct BlockForwardOutput<B: Backend, T> {
    pub hidden: Tensor<B, 3>,
    pub ffn_trace: T,
}

#[derive(Module, Debug)]
pub struct MiniTransformerBlock<B: Backend, F> {
    pre_attention_norm: LayerNorm<B>,
    attention: SharedAttentionSublayer<B>,
    pre_ffn_norm: LayerNorm<B>,
    ffn: F,
}

impl<B: Backend, F: FfnSublayer<B>> MiniTransformerBlock<B, F> {
    pub fn new(
        hidden_dim: usize,
        attention: SharedAttentionSublayer<B>,
        ffn: F,
        device: &B::Device,
    ) -> Self {
        Self {
            pre_attention_norm: LayerNormConfig::new(hidden_dim).init(device),
            attention,
            pre_ffn_norm: LayerNormConfig::new(hidden_dim).init(device),
            ffn,
        }
    }

    pub fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Result<BlockForwardOutput<B, F::Trace>, FractalError> {
        let attention_input = self.pre_attention_norm.forward(hidden.clone());
        let attention_output = self.attention.forward(attention_input, mask)?;
        let hidden = hidden + attention_output.hidden;
        let ffn_input = self.pre_ffn_norm.forward(hidden.clone());
        let ffn_output = self.ffn.forward(ffn_input)?;
        Ok(BlockForwardOutput {
            hidden: hidden + ffn_output.hidden,
            ffn_trace: ffn_output.trace,
        })
    }
}

#[derive(Debug)]
pub struct MiniMoeForwardOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub trace_bundle: MiniMoeTraceBundle,
}

#[derive(Module, Debug)]
pub struct MiniMoeModel<B: Backend, F> {
    embedding: Embedding<B>,
    blocks: Vec<MiniTransformerBlock<B, F>>,
    output: LanguageModelHead<B>,
    architecture: Ignored<MiniMoeArchitectureSpec>,
    runtime: Ignored<MiniMoeRuntimeSpec>,
    observability: Ignored<MiniMoeObservabilitySpec>,
    resolved_layout: Ignored<ResolvedMiniMoeLayout>,
}

impl<B: Backend> MiniMoeModel<B, ConfiguredMiniMoeFfn<B>> {
    pub fn new(surface: &MiniMoeSurfaceSpec, device: &B::Device) -> Result<Self, FractalError> {
        surface.validate()?;
        let architecture = surface.architecture.clone();
        let resolved_layout = surface.resolve_layout()?;
        let resolved_dispatch = surface.resolve_dispatch_contract();
        let embedding = EmbeddingConfig::new(
            architecture.backbone.vocab_size,
            architecture.backbone.hidden_dim,
        )
        .with_initializer(Initializer::Uniform {
            min: MINI_MOE_INIT_MIN,
            max: MINI_MOE_INIT_MAX,
        })
        .init(device);
        let attention_spec = SharedAttentionSpec {
            hidden_dim: architecture.backbone.hidden_dim,
            head_count: architecture.backbone.head_count,
            local_window: architecture.backbone.local_window,
        };
        let dense_spec = ExpertFfnSpec {
            hidden_dim: architecture.backbone.hidden_dim,
            expansion_dim: architecture.backbone.dense_ffn_width(),
            activation: ExpertActivationKind::Gelu,
            gated: false,
        };
        let mut blocks = Vec::with_capacity(architecture.backbone.total_layers);
        for layer_index in 0..architecture.backbone.total_layers {
            let site = RouteSiteId { layer_index };
            let attention = SharedAttentionSublayer::new(attention_spec.clone(), device)?;
            let ffn = if resolved_layout.moe_layers.contains(&layer_index) {
                match &architecture.router {
                    MiniMoeRouterSpec::OneShot(_) => ConfiguredMiniMoeFfn::OneShotMoe(
                        OneShotMoeFfnSublayer::new(
                            site,
                            architecture.backbone.hidden_dim,
                            &architecture.moe,
                            resolved_dispatch.clone(),
                            device,
                        )?,
                    ),
                    MiniMoeRouterSpec::RecurrentPreExpert(router) => {
                        ConfiguredMiniMoeFfn::RecurrentMoe(RecurrentMoeFfnSublayer::new(
                            site,
                            architecture.backbone.hidden_dim,
                            &architecture.moe,
                            router,
                            resolved_dispatch.clone(),
                            device,
                        )?)
                    }
                }
            } else {
                ConfiguredMiniMoeFfn::Dense(DenseFfnSublayer::new(
                    layer_index,
                    dense_spec.clone(),
                    device,
                )?)
            };
            blocks.push(MiniTransformerBlock::new(
                architecture.backbone.hidden_dim,
                attention,
                ffn,
                device,
            ));
        }
        let output = LanguageModelHeadConfig::new(
            architecture.backbone.hidden_dim,
            architecture.backbone.vocab_size,
        )
        .with_initializer(Initializer::Uniform {
            min: MINI_MOE_INIT_MIN,
            max: MINI_MOE_INIT_MAX,
        })
        .init(device);
        Ok(Self {
            embedding,
            blocks,
            output,
            architecture: Ignored(architecture),
            runtime: Ignored(surface.runtime.clone()),
            observability: Ignored(surface.observability.clone()),
            resolved_layout: Ignored(resolved_layout),
        })
    }
}

impl<B: Backend, F: FfnSublayer<B, Trace = MiniMoeFfnTrace<B>>> MiniMoeModel<B, F> {
    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        Ok(self.forward_with_trace(input_ids)?.logits)
    }

    pub fn forward_with_trace(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<MiniMoeForwardOutput<B>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids.clone());
        let mask = local_causal_mask::<B>(
            batch_size,
            seq_len,
            self.architecture.backbone.local_window,
            &hidden.device(),
        );
        let mut sink = MiniMoeTraceCollector::new(
            self.architecture.backbone.vocab_size,
            self.observability.0.clone(),
        );
        for block in &self.blocks {
            let output = block.forward(hidden, mask.clone())?;
            sink.record_ffn_trace(&output.ffn_trace);
            hidden = output.hidden;
        }
        let logits = self.output.forward(hidden);
        Ok(MiniMoeForwardOutput {
            logits,
            trace_bundle: <MiniMoeTraceCollector as MiniMoeObservabilitySink<B>>::finalize(sink),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.architecture.backbone.vocab_size
    }

    pub fn architecture(&self) -> &MiniMoeArchitectureSpec {
        &self.architecture
    }

    pub fn runtime(&self) -> &MiniMoeRuntimeSpec {
        &self.runtime
    }

    pub fn resolved_layout(&self) -> &ResolvedMiniMoeLayout {
        &self.resolved_layout
    }
}

pub type ReferenceMiniMoeModel<B> = MiniMoeModel<B, ConfiguredMiniMoeFfn<B>>;
pub type RecurrentMiniMoeModel<B> = MiniMoeModel<B, ConfiguredMiniMoeFfn<B>>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeObservabilitySpec {
    pub sampling: TraceSamplingPolicy,
    pub capture_round_summaries: bool,
}

impl MiniMoeObservabilitySpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.sampling.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSamplingPolicy {
    pub token_trace_budget_per_layer: usize,
    pub expert_example_budget_per_layer: usize,
    pub deterministic: bool,
    pub sample_seed: u64,
}

impl TraceSamplingPolicy {
    pub fn validate(&self) -> Result<(), FractalError> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRoutingSummary {
    pub sampled_tokens: usize,
    pub layer_count: usize,
    pub round_count: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_expert_weights: Vec<f64>,
    pub winner_counts: Vec<usize>,
    pub active_expert_count: usize,
    pub mean_round_adjustment_l1: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertUsageSummary {
    pub site: RouteSiteId,
    pub expert_id: usize,
    pub selection_count: usize,
    pub mean_weight: f64,
    pub representative_token_examples: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRouteSummary {
    pub site: RouteSiteId,
    pub sampled_tokens: usize,
    pub expert_usage: Vec<ExpertUsageSummary>,
    pub route_entropy_bits: f64,
    pub reroute_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenRouteTrace {
    pub token_text: String,
    pub site: RouteSiteId,
    pub round_expert_indices: Vec<Vec<usize>>,
    pub round_expert_weights: Vec<Vec<f64>>,
    pub final_expert_indices: Vec<usize>,
    pub confidence_margin: f64,
    pub rerouted: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerRoundSummary {
    pub site: RouteSiteId,
    pub round_index: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_route_adjustment_l1: Option<f64>,
    pub rerouted_token_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DispatchSummary {
    pub site: RouteSiteId,
    pub mode: MiniMoeDispatchMode,
    pub selected_expert_counts: Vec<usize>,
    pub dropped_token_fraction: Option<f64>,
}

#[derive(Debug)]
pub enum MiniMoeTraceEvent<B: Backend> {
    RoutePlanned {
        site: RouteSiteId,
        route_plan: RoutePlan<B>,
    },
    DispatchCompiled {
        dispatch_plan: DispatchPlan<B>,
    },
    TokenExampleObserved {
        site: RouteSiteId,
        token_text: String,
        final_expert_indices: Vec<usize>,
        final_expert_weights: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeTraceBundle {
    pub layer_summaries: Vec<LayerRouteSummary>,
    pub dispatch_summaries: Vec<DispatchSummary>,
    pub controller_round_summaries: Vec<ControllerRoundSummary>,
    pub sampled_token_traces: Vec<TokenRouteTrace>,
}

impl MiniMoeTraceBundle {
    pub fn merge(&mut self, mut other: Self) {
        self.layer_summaries.append(&mut other.layer_summaries);
        self.dispatch_summaries.append(&mut other.dispatch_summaries);
        self.controller_round_summaries
            .append(&mut other.controller_round_summaries);
        self.sampled_token_traces
            .append(&mut other.sampled_token_traces);
    }

    pub fn into_report_summary(self) -> MiniMoeReportSummary {
        let layer_count = self
            .layer_summaries
            .iter()
            .map(|summary| summary.site.layer_index)
            .collect::<BTreeSet<_>>()
            .len();
        let round_count = self
            .controller_round_summaries
            .iter()
            .map(|summary| summary.round_index + 1)
            .max()
            .unwrap_or(0);
        let mut total_tokens = 0usize;
        let mut winner_counts: Vec<usize> = Vec::new();
        let mut mean_expert_weights: Vec<f64> = Vec::new();
        let mut mean_entropy = 0.0f64;
        let mut reroute_adjustments = Vec::new();
        for layer in &self.layer_summaries {
            total_tokens += layer.sampled_tokens;
            mean_entropy += layer.route_entropy_bits * layer.sampled_tokens as f64;
            if winner_counts.len() < layer.expert_usage.len() {
                winner_counts.resize(layer.expert_usage.len(), 0);
                mean_expert_weights.resize(layer.expert_usage.len(), 0.0);
            }
            for usage in &layer.expert_usage {
                winner_counts[usage.expert_id] += usage.selection_count;
                mean_expert_weights[usage.expert_id] +=
                    usage.mean_weight * layer.sampled_tokens as f64;
            }
        }
        if total_tokens > 0 {
            mean_entropy /= total_tokens as f64;
            for weight in &mut mean_expert_weights {
                *weight /= total_tokens as f64;
            }
        }
        for round in &self.controller_round_summaries {
            if let Some(value) = round.mean_route_adjustment_l1 {
                reroute_adjustments.push(value);
            }
        }
        MiniMoeReportSummary {
            routing: MiniMoeRoutingSummary {
                sampled_tokens: total_tokens,
                layer_count,
                round_count,
                mean_route_entropy_bits: mean_entropy,
                mean_winner_margin: if self.controller_round_summaries.is_empty() {
                    0.0
                } else {
                    self.controller_round_summaries
                        .iter()
                        .map(|summary| summary.mean_winner_margin)
                        .sum::<f64>()
                        / self.controller_round_summaries.len() as f64
                },
                mean_expert_weights,
                winner_counts: winner_counts.clone(),
                active_expert_count: winner_counts.iter().filter(|count| **count > 0).count(),
                mean_round_adjustment_l1: reroute_adjustments,
            },
            layers: self.layer_summaries,
            dispatch: self.dispatch_summaries,
            controller_rounds: self.controller_round_summaries,
            sampled_tokens: self.sampled_token_traces,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeReportSummary {
    pub routing: MiniMoeRoutingSummary,
    pub layers: Vec<LayerRouteSummary>,
    pub dispatch: Vec<DispatchSummary>,
    pub controller_rounds: Vec<ControllerRoundSummary>,
    pub sampled_tokens: Vec<TokenRouteTrace>,
}

pub trait MiniMoeObservabilitySink<B: Backend> {
    fn record_route_plan(&mut self, site: RouteSiteId, route_plan: &RoutePlan<B>);
    fn record_dispatch_plan(&mut self, dispatch_plan: &DispatchPlan<B>);
    fn record_token_example(
        &mut self,
        site: RouteSiteId,
        token_text: &str,
        final_expert_indices: &[usize],
        final_expert_weights: &[f64],
    );
    fn finalize(self) -> MiniMoeTraceBundle;
}

#[derive(Debug)]
pub struct MiniMoeTraceCollector {
    layer_summaries: Vec<LayerRouteSummary>,
    dispatch_summaries: Vec<DispatchSummary>,
    controller_round_summaries: Vec<ControllerRoundSummary>,
    sampled_token_traces: Vec<TokenRouteTrace>,
}

impl MiniMoeTraceCollector {
    pub fn new(_vocab_size: usize, _spec: MiniMoeObservabilitySpec) -> Self {
        Self {
            layer_summaries: Vec::new(),
            dispatch_summaries: Vec::new(),
            controller_round_summaries: Vec::new(),
            sampled_token_traces: Vec::new(),
        }
    }

    pub fn record_ffn_trace<B: Backend>(&mut self, trace: &MiniMoeFfnTrace<B>) {
        let MiniMoeFfnTrace::Moe {
            route_plan,
            dispatch_plan,
        } = trace
        else {
            return;
        };

        let site = dispatch_plan.site.clone();
        let [batch_size, seq_len, expert_count] = route_plan.expert_weights.dims();
        let sampled_tokens = batch_size * seq_len;
        let weights = tensor_to_vec_f32(route_plan.expert_weights.clone())
            .unwrap_or_else(|_| vec![0.0; sampled_tokens * expert_count]);
        let final_indices = tensor_to_vec_i64(dispatch_plan.selected_expert_indices.clone())
            .unwrap_or_else(|_| vec![0; sampled_tokens]);
        let selected_width = dispatch_plan.selected_expert_indices.dims()[2];
        let mut winner_counts = vec![0usize; expert_count];
        let mut mean_weights = vec![0.0f64; expert_count];
        let mut entropy_sum = 0.0f64;
        let mut rerouted_count = 0usize;

        for token_index in 0..sampled_tokens {
            let token_slice = &weights[token_index * expert_count..(token_index + 1) * expert_count];
            for (expert_index, weight) in token_slice.iter().enumerate() {
                mean_weights[expert_index] += *weight as f64;
            }
            entropy_sum += entropy_bits(token_slice);
            let winner = final_indices
                .get(token_index * selected_width)
                .copied()
                .unwrap_or_default()
                .max(0) as usize;
            if winner < winner_counts.len() {
                winner_counts[winner] += 1;
            }
            if route_plan.round_summaries.len() > 1 {
                let first_round = tensor_to_vec_i64(select_winners(
                    route_plan.round_summaries.first().unwrap().expert_weights.clone(),
                ))
                .unwrap_or_default();
                if first_round.get(token_index).copied().unwrap_or_default()
                    != winner as i64
                {
                    rerouted_count += 1;
                }
            }
        }
        if sampled_tokens > 0 {
            for weight in &mut mean_weights {
                *weight /= sampled_tokens as f64;
            }
        }

        self.layer_summaries.push(LayerRouteSummary {
            site: site.clone(),
            sampled_tokens,
            expert_usage: winner_counts
                .iter()
                .enumerate()
                .map(|(expert_id, count)| ExpertUsageSummary {
                    site: site.clone(),
                    expert_id,
                    selection_count: *count,
                    mean_weight: mean_weights[expert_id],
                    representative_token_examples: Vec::new(),
                })
                .collect(),
            route_entropy_bits: if sampled_tokens == 0 {
                0.0
            } else {
                entropy_sum / sampled_tokens as f64
            },
            reroute_fraction: if sampled_tokens == 0 {
                0.0
            } else {
                rerouted_count as f64 / sampled_tokens as f64
            },
        });

        self.dispatch_summaries.push(DispatchSummary {
            site: site.clone(),
            mode: dispatch_plan.mode,
            selected_expert_counts: winner_counts.clone(),
            dropped_token_fraction: None,
        });

        for (round_index, round_summary) in route_plan.round_summaries.iter().enumerate() {
            let round_weights = tensor_to_vec_f32(round_summary.expert_weights.clone())
                .unwrap_or_else(|_| vec![0.0; sampled_tokens * expert_count]);
            let mut round_entropy_sum = 0.0f64;
            let mut winner_margin_sum = 0.0f64;
            for token_index in 0..sampled_tokens {
                let token_slice =
                    &round_weights[token_index * expert_count..(token_index + 1) * expert_count];
                round_entropy_sum += entropy_bits(token_slice);
                winner_margin_sum += winner_margin(token_slice);
            }
            let mean_route_adjustment_l1 = if round_index == 0 {
                None
            } else {
                let prev = &route_plan.round_summaries[round_index - 1];
                Some(mean_l1_between(prev.expert_weights.clone(), round_summary.expert_weights.clone()).unwrap_or(0.0))
            };
            self.controller_round_summaries.push(ControllerRoundSummary {
                site: site.clone(),
                round_index,
                mean_route_entropy_bits: if sampled_tokens == 0 {
                    0.0
                } else {
                    round_entropy_sum / sampled_tokens as f64
                },
                mean_winner_margin: if sampled_tokens == 0 {
                    0.0
                } else {
                    winner_margin_sum / sampled_tokens as f64
                },
                mean_route_adjustment_l1,
                rerouted_token_fraction: if sampled_tokens == 0 || round_index == 0 {
                    0.0
                } else {
                    rerouted_count as f64 / sampled_tokens as f64
                },
            });
        }
    }

    pub fn into_report_summary<B: Backend>(self) -> MiniMoeReportSummary {
        <MiniMoeTraceCollector as MiniMoeObservabilitySink<B>>::finalize(self).into_report_summary()
    }
}

impl<B: Backend> MiniMoeObservabilitySink<B> for MiniMoeTraceCollector {
    fn record_route_plan(&mut self, _site: RouteSiteId, _route_plan: &RoutePlan<B>) {}

    fn record_dispatch_plan(&mut self, _dispatch_plan: &DispatchPlan<B>) {}

    fn record_token_example(
        &mut self,
        site: RouteSiteId,
        token_text: &str,
        final_expert_indices: &[usize],
        final_expert_weights: &[f64],
    ) {
        if self.sampled_token_traces.len() >= 16 {
            return;
        }
        self.sampled_token_traces.push(TokenRouteTrace {
            token_text: token_text.to_string(),
            site,
            round_expert_indices: vec![final_expert_indices.to_vec()],
            round_expert_weights: vec![final_expert_weights.to_vec()],
            final_expert_indices: final_expert_indices.to_vec(),
            confidence_margin: winner_margin_f64(final_expert_weights),
            rerouted: false,
        });
    }

    fn finalize(self) -> MiniMoeTraceBundle {
        MiniMoeTraceBundle {
            layer_summaries: self.layer_summaries,
            dispatch_summaries: self.dispatch_summaries,
            controller_round_summaries: self.controller_round_summaries,
            sampled_token_traces: self.sampled_token_traces,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRunManifest {
    pub surface: MiniMoeSurfaceSpec,
    pub resolved_layout: ResolvedMiniMoeLayout,
    pub resolved_dispatch: ResolvedDispatchContract,
    pub data: MiniMoeDataSpec,
    pub train: MiniMoeTrainSpec,
    pub eval: MiniMoeEvalSpec,
    pub backend: MiniMoeBackendSpec,
    pub benchmark_policy: BenchmarkPolicy,
    pub isolation_mode: ExecutionIsolationMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeDataSpec {
    pub seq_len: usize,
    pub window_stride: usize,
}

impl MiniMoeDataSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.seq_len == 0 || self.window_stride == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.data seq_len and window_stride must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeTrainSpec {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub model_seed: u64,
    pub data_seed: Option<u64>,
}

impl MiniMoeTrainSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.steps == 0 || self.batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.train steps and batch_size must be greater than zero".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "mini_moe.train.learning_rate must be finite and greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeEvalSpec {
    pub eval_batches: usize,
    pub full_eval_pass: bool,
}

impl MiniMoeEvalSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if !self.full_eval_pass && self.eval_batches == 0 {
            return Err(FractalError::InvalidConfig(
                "mini_moe.eval.eval_batches must be greater than zero unless full_eval_pass is true"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeBackendSpec {
    pub backend: MiniMoeBackendKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeBackendKind {
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPolicy {
    Smoke,
    Benchmark,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionIsolationMode {
    SharedProcess,
    IsolatedProcess,
}

impl MiniMoeRunManifest {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.surface.validate()?;
        let expected_layout = self.surface.resolve_layout()?;
        if self.resolved_layout != expected_layout {
            return Err(FractalError::InvalidConfig(
                "mini_moe.manifest.resolved_layout must match surface.resolve_layout()"
                    .to_string(),
            ));
        }
        let expected_dispatch = self.surface.resolve_dispatch_contract();
        if self.resolved_dispatch != expected_dispatch {
            return Err(FractalError::InvalidConfig(
                "mini_moe.manifest.resolved_dispatch must match surface.resolve_dispatch_contract()"
                    .to_string(),
            ));
        }
        self.data.validate()?;
        self.train.validate()?;
        self.eval.validate()?;
        if self.benchmark_policy == BenchmarkPolicy::Benchmark
            && self.isolation_mode != ExecutionIsolationMode::IsolatedProcess
        {
            return Err(FractalError::InvalidConfig(
                "mini_moe benchmark runs require isolated process execution".to_string(),
            ));
        }
        if self.benchmark_policy == BenchmarkPolicy::Benchmark
            && self.surface.runtime.dispatch.mode == MiniMoeDispatchMode::DenseDebug
        {
            return Err(FractalError::InvalidConfig(
                "mini_moe benchmark runs forbid dense-debug dispatch".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRunArtifact {
    pub manifest: MiniMoeRunManifest,
    pub summary: MiniMoeReportSummary,
    pub train_metrics: MiniMoeTrainMetrics,
    pub eval_metrics: MiniMoeEvalMetrics,
    pub system_metrics: MiniMoeSystemMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeTrainMetrics {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub load_balance_aux_loss: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeEvalMetrics {
    pub final_loss: f64,
    pub perplexity: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeSystemMetrics {
    pub train_tokens_per_second: Option<f64>,
    pub eval_tokens_per_second: Option<f64>,
    pub overall_tokens_per_second: Option<f64>,
    pub process_memory_metric: Option<String>,
    pub peak_process_memory_mb: Option<f64>,
}

fn activate<B: Backend>(kind: ExpertActivationKind, tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    match kind {
        ExpertActivationKind::Gelu => gelu(tensor),
        ExpertActivationKind::Silu => silu_like(tensor),
        ExpertActivationKind::Relu => relu(tensor),
        ExpertActivationKind::SquaredRelu => {
            let activated = relu(tensor);
            activated.clone() * activated
        }
    }
}

fn silu_like<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    tensor.clone() * gated_sigmoid(tensor)
}

fn mean_over_tokens<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, seq_len, width] = tensor.dims();
    tensor
        .sum_dim(1)
        .reshape([batch_size, width])
        .mul_scalar(1.0 / seq_len as f64)
}

fn apply_tie_break_bias<B: Backend>(
    weights: Tensor<B, 3>,
    policy: TieBreakPolicy,
    expert_count: usize,
) -> Tensor<B, 3> {
    match policy {
        TieBreakPolicy::LowestIndex => {
            let [batch_size, seq_len, _] = weights.dims();
            let mut bias = Vec::with_capacity(expert_count);
            for index in 0..expert_count {
                bias.push(-(index as f32) * 1.0e-7);
            }
            let bias = Tensor::<B, 1>::from_data(TensorData::new(bias, [expert_count]), &weights.device())
                .reshape([1, 1, expert_count])
                .repeat(&[batch_size, seq_len, 1]);
            weights + bias
        }
    }
}

fn dense_debug_indices<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    expert_count: usize,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let mut indices = Vec::with_capacity(batch_size * seq_len * expert_count);
    for _batch in 0..batch_size {
        for _token in 0..seq_len {
            for expert_index in 0..expert_count {
                indices.push(expert_index as i64);
            }
        }
    }
    Tensor::<B, 3, Int>::from_data(
        TensorData::new(indices, [batch_size, seq_len, expert_count]),
        device,
    )
}

fn build_dispatch_assignments<B: Backend>(
    selected_expert_indices: &Tensor<B, 3, Int>,
    expert_count: usize,
    device: &B::Device,
) -> Result<Vec<ExpertDispatchAssignment<B>>, FractalError> {
    let [batch_size, seq_len, selected_width] = selected_expert_indices.dims();
    let data = tensor_to_vec_i64(selected_expert_indices.clone())?;
    let mut per_expert_positions = vec![Vec::<i64>::new(); expert_count];
    let mut per_expert_slots = vec![Vec::<i64>::new(); expert_count];

    for batch_index in 0..batch_size {
        for token_index in 0..seq_len {
            let flat_index = (batch_index * seq_len + token_index) as i64;
            for slot_index in 0..selected_width {
                let offset = (batch_index * seq_len * selected_width)
                    + (token_index * selected_width)
                    + slot_index;
                let expert_index = *data.get(offset).ok_or_else(|| {
                    FractalError::InvalidState(
                        "mini_moe.dispatch failed to decode selected expert indices".to_string(),
                    )
                })?;
                if expert_index < 0 || expert_index as usize >= expert_count {
                    return Err(FractalError::InvalidState(format!(
                        "mini_moe.dispatch selected invalid expert index {}",
                        expert_index
                    )));
                }
                per_expert_positions[expert_index as usize].push(flat_index);
                per_expert_slots[expert_index as usize].push(slot_index as i64);
            }
        }
    }

    Ok((0..expert_count)
        .map(|expert_index| ExpertDispatchAssignment {
            expert_index,
            token_indices: Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    per_expert_positions[expert_index].clone(),
                    [per_expert_positions[expert_index].len()],
                ),
                device,
            ),
            slot_indices: Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    per_expert_slots[expert_index].clone(),
                    [per_expert_slots[expert_index].len()],
                ),
                device,
            ),
        })
        .collect())
}

fn tensor_to_vec_i64<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
) -> Result<Vec<i64>, FractalError> {
    tensor
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| FractalError::InvalidState(format!("tensor data conversion failed: {error}")))
}

fn tensor_to_vec_f32<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, FractalError> {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| FractalError::InvalidState(format!("tensor data conversion failed: {error}")))
}

fn entropy_bits(values: &[f32]) -> f64 {
    values
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| {
            let probability = *value as f64;
            -probability * probability.log2()
        })
        .sum::<f64>()
}

fn winner_margin(values: &[f32]) -> f64 {
    let mut sorted = values.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(core::cmp::Ordering::Equal));
    let winner = sorted.first().copied().unwrap_or_default() as f64;
    let runner_up = sorted.get(1).copied().unwrap_or_default() as f64;
    winner - runner_up
}

fn winner_margin_f64(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| right.partial_cmp(left).unwrap_or(core::cmp::Ordering::Equal));
    let winner = sorted.first().copied().unwrap_or_default();
    let runner_up = sorted.get(1).copied().unwrap_or_default();
    winner - runner_up
}

fn mean_l1_between<B: Backend>(left: Tensor<B, 3>, right: Tensor<B, 3>) -> Result<f64, FractalError> {
    let left = tensor_to_vec_f32(left)?;
    let right = tensor_to_vec_f32(right)?;
    if left.len() != right.len() {
        return Err(FractalError::Shape(
            "mini_moe.mean_l1_between requires matching tensor lengths".to_string(),
        ));
    }
    if left.is_empty() {
        return Ok(0.0);
    }
    Ok(left
        .iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (*lhs as f64 - *rhs as f64).abs())
        .sum::<f64>()
        / left.len() as f64)
}

fn select_winners<B: Backend>(expert_weights: Tensor<B, 3>) -> Tensor<B, 2, Int> {
    let [batch_size, seq_len, _expert_count] = expert_weights.dims();
    expert_weights.argmax(2).reshape([batch_size, seq_len])
}

#[cfg(test)]
mod tests {
    use burn::{backend::Candle, tensor::Tensor};

    use super::*;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn phase1_defaults_validate() {
        let reference = MiniMoeSurfaceSpec::phase1_reference_default();
        let recurrent = MiniMoeSurfaceSpec::phase1_recurrent_default();

        reference.validate().unwrap();
        recurrent.validate().unwrap();
        assert_eq!(reference.architecture.backbone.total_layers, 8);
        assert_eq!(reference.architecture.moe.experts_per_block, 4);
        assert_eq!(reference.resolve_layout().unwrap().moe_layers, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert!(matches!(
            recurrent.architecture.router,
            MiniMoeRouterSpec::RecurrentPreExpert(_)
        ));
    }

    #[test]
    fn schedule_preset_and_resolution_are_canonical() {
        let lowered = MiniMoeLayerSchedulePreset::AlternatingFromOne.lower(8);
        let resolved = lowered.resolve(8).unwrap();
        assert_eq!(resolved.moe_layers, vec![1, 3, 5, 7]);
        assert_eq!(resolved.dense_layers, vec![0, 2, 4, 6]);
    }

    #[test]
    fn dispatcher_prefers_lowest_index_on_ties() {
        let device = Default::default();
        let dispatcher = MoeDispatcher::new(ResolvedDispatchContract {
            mode: MiniMoeDispatchMode::SparseTopK,
            active_experts_per_token: 1,
            tie_break: TieBreakPolicy::LowestIndex,
            capacity: DispatchCapacityPolicy::Unlimited,
        });
        let route_plan = RoutePlan {
            expert_logits: Tensor::<TestBackend, 3>::zeros([1, 2, 4], &device),
            expert_weights: Tensor::<TestBackend, 3>::from_data(
                [[[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0]]],
                &device,
            ),
            round_summaries: Vec::new(),
        };
        let plan = dispatcher
            .compile(RouteSiteId { layer_index: 0 }, &route_plan)
            .unwrap();
        let indices = tensor_to_vec_i64(plan.selected_expert_indices).unwrap();
        assert_eq!(indices, vec![0, 0]);
    }

    #[test]
    fn one_shot_and_recurrent_routers_return_expected_round_counts() {
        let device = Default::default();
        let hidden = Tensor::<TestBackend, 3>::zeros([1, 3, 8], &device);
        let one_shot = OneShotRouter::new(8, 4, &device).unwrap();
        let recurrent =
            RecurrentPreExpertRouter::new(8, 4, RecurrentPreExpertRouterSpec { round_count: 2, state_dim: 6 }, &device)
                .unwrap();
        let one_shot_plan = one_shot
            .route(PreExpertRouterInput {
                hidden: hidden.clone(),
                site: RouteSiteId { layer_index: 0 },
            })
            .unwrap();
        let recurrent_plan = recurrent
            .route(PreExpertRouterInput {
                hidden,
                site: RouteSiteId { layer_index: 0 },
            })
            .unwrap();
        assert_eq!(one_shot_plan.round_summaries.len(), 1);
        assert_eq!(recurrent_plan.round_summaries.len(), 2);
    }

    #[test]
    fn model_builds_and_returns_logits_with_trace() {
        let device = Default::default();
        let surface = MiniMoeSurfaceSpec::phase1_reference_default();
        let model = MiniMoeModel::<TestBackend, ConfiguredMiniMoeFfn<TestBackend>>::new(
            &surface,
            &device,
        )
        .unwrap();
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let output = model.forward_with_trace(input_ids).unwrap();
        assert_eq!(output.logits.dims(), [1, 4, 257]);
        assert_eq!(output.trace_bundle.layer_summaries.len(), 8);
        assert_eq!(output.trace_bundle.dispatch_summaries.len(), 8);
    }

    #[test]
    fn benchmark_manifest_requires_isolated_process() {
        let surface = MiniMoeSurfaceSpec::phase1_reference_default();
        let manifest = MiniMoeRunManifest {
            resolved_layout: surface.resolve_layout().unwrap(),
            resolved_dispatch: surface.resolve_dispatch_contract(),
            surface,
            data: MiniMoeDataSpec {
                seq_len: 32,
                window_stride: 32,
            },
            train: MiniMoeTrainSpec {
                steps: 8,
                batch_size: 1,
                learning_rate: 1.0e-3,
                model_seed: 42,
                data_seed: None,
            },
            eval: MiniMoeEvalSpec {
                eval_batches: 2,
                full_eval_pass: false,
            },
            backend: MiniMoeBackendSpec {
                backend: MiniMoeBackendKind::Cpu,
            },
            benchmark_policy: BenchmarkPolicy::Benchmark,
            isolation_mode: ExecutionIsolationMode::SharedProcess,
        };
        assert!(manifest.validate().is_err());
    }
}
