pub mod config;
pub mod mamba3_baseline;
pub mod model;

pub use config::{
    phase1_hybrid_attention_baseline_matrix, HybridAttentionBaselineMatrix,
    HybridAttentionComparisonContract, HybridAttentionEfficiencyTarget, HybridAttentionLayerRole,
    HybridAttentionVariantKind, HybridAttentionVariantSpec, PrimitiveHybridPrimitive,
    ReferenceSsmFamily, PATH1_PHASE1_LOCAL_WINDOW_SIZE,
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
