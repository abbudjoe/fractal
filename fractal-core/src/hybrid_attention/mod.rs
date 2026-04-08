pub mod common;
pub mod config;
pub mod goe;
pub mod goe_recurrent;
pub mod mamba3_baseline;
pub mod model;
pub mod mini_moe;
pub mod recurrent_router;
pub mod scale_proxy;

pub use config::{
    phase1_hybrid_attention_baseline_matrix, phase1_p20_candidate_variant,
    phase1_p21_candidate_variant, phase1_p22_candidate_variant, phase1_p23_candidate_variant,
    phase1_p2_candidate_variant, phase1_p2_factor_candidate_variant,
    phase1_p2_interface_candidate_variant, primitive_from_p2_factors,
    HybridAttentionBaselineMatrix, HybridAttentionComparisonContract,
    HybridAttentionEfficiencyTarget, HybridAttentionLayerRole, HybridAttentionVariantKind,
    HybridAttentionVariantSpec, HybridSequenceKernelContract, HybridSequenceScanMode,
    HybridSequenceStateLayout, P2InternalReadoutFactor, P2LatentWidthFactor,
    PrimitiveHybridNormMode, PrimitiveHybridPrimitive, PrimitiveHybridReadoutMode,
    PrimitiveHybridResidualMode, PrimitiveHybridWrapperSymmetryMode, ReferenceSsmFamily,
    PATH1_PHASE1_LOCAL_WINDOW_SIZE,
};
pub use goe::{
    build_attention_only_graph_of_experts_model, build_reference_ssm_graph_of_experts_model,
    goe_over_attention_only_variant, goe_over_attention_only_variant_with_controller,
    goe_over_reference_ssm_variant, goe_over_reference_ssm_variant_with_controller,
    AttentionOnlyGraphOfExpertsModel, GraphOfExpertsBackboneKind, GraphOfExpertsControllerSpec,
    GraphOfExpertsRoutingMode, GraphOfExpertsRoutingProbe, GraphOfExpertsTopology,
    GraphOfExpertsVariantSpec, ReferenceSsmGraphOfExpertsModel, GOE_CHANNEL_COUNT,
};
pub use goe_recurrent::{
    build_attention_only_recurrent_graph_of_experts_model,
    recurrent_goe_over_attention_only_variant,
    recurrent_goe_over_attention_only_variant_with_router,
    AttentionOnlyRecurrentGraphOfExpertsModel, AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    RecurrentGraphOfExpertsRoutingProbe, RECURRENT_DREEGMOR_CHANNEL_COUNT,
};
pub use mamba3_baseline::{
    build_rust_mamba3_reference_hybrid_attention_model, RustMamba3BaselineConfig,
    RustMamba3DerivedShape, RustMamba3Mixer, RustMamba3MixerBlock,
    RustMamba3ReferenceHybridAttentionModel, RustMamba3RopeFraction,
};
pub use model::{
    build_attention_only_hybrid_attention_model, build_primitive_hybrid_attention_model,
    AttentionOnlyHybridAttentionModel, HybridAttentionModelShape, PrimitiveHybridAttentionModel,
};
pub use mini_moe::*;
pub use recurrent_router::{
    minimal_recurrent_router_experiment_matrix, RecurrentRouterExperimentVariantKind,
    RecurrentRouterExperimentVariantSpec, RecurrentRouterFeedbackMode,
    RecurrentRouterPrimitiveKind, RecurrentRouterSelectionMode, RecurrentRouterSpec,
    VirtualNodeRecurrentRouter, DEFAULT_RECURRENT_ROUTER_ROUND_COUNT,
    DEFAULT_RECURRENT_ROUTER_STATE_WIDTH, MAX_RECURRENT_ROUTER_ROUND_COUNT,
};
pub use scale_proxy::{
    build_attention_only_recurrent_scale_proxy_model, build_attention_only_scale_proxy_model,
    scale_proxy_one_shot_over_attention_only_variant,
    scale_proxy_recurrent_over_attention_only_variant,
    scale_proxy_recurrent_over_attention_only_variant_with_router,
    AttentionOnlyRecurrentScaleProxyModel, AttentionOnlyRecurrentScaleProxyVariantSpec,
    AttentionOnlyScaleProxyModel, AttentionOnlyScaleProxyVariantSpec,
    ScaleProxyRoutingProbe, DEFAULT_SCALE_PROXY_EXPERT_LAYER_INDEX,
    SCALE_PROXY_CHANNEL_COUNT,
};
