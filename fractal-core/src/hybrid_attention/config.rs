use serde::{Deserialize, Serialize};

use crate::{
    error::FractalError, projection::ProjectionLayoutPolicy, registry::PrimitiveVariantName,
};

pub const PATH1_PHASE1_LOCAL_WINDOW_SIZE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HybridAttentionVariantKind {
    AttentionOnly,
    ReferenceSsmHybrid,
    PrimitiveHybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReferenceSsmFamily {
    Mamba3ProxyV1,
    Mamba3RustV1,
}

impl ReferenceSsmFamily {
    pub const fn kernel_contract(self) -> HybridSequenceKernelContract {
        match self {
            Self::Mamba3ProxyV1 => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::StructuredReferenceSsm,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: true,
            },
            Self::Mamba3RustV1 => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::StructuredReferenceSsm,
                scan_mode: HybridSequenceScanMode::ChunkedSequentialStepLoop,
                explicit_output_readout: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridPrimitive {
    P1Contractive,
    P20RotaryStateOutput,
    P2RotaryReadout,
    P23RotaryCarryBlendReadout,
    P21WideLatent,
    P22WideLatentReadout,
}

impl PrimitiveHybridPrimitive {
    pub const fn variant_name(self) -> PrimitiveVariantName {
        match self {
            Self::P1Contractive => PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
            Self::P20RotaryStateOutput => {
                PrimitiveVariantName::new_unchecked("p2_0_rotary_state_output_v1")
            }
            Self::P2RotaryReadout => PrimitiveVariantName::new_unchecked("p2_rotary_readout_v1"),
            Self::P23RotaryCarryBlendReadout => {
                PrimitiveVariantName::new_unchecked("p2_3_rotary_carry_blend_readout_v1")
            }
            Self::P21WideLatent => PrimitiveVariantName::new_unchecked("p2_1_wide_latent_v1"),
            Self::P22WideLatentReadout => {
                PrimitiveVariantName::new_unchecked("p2_2_wide_latent_readout_v1")
            }
        }
    }

    pub const fn label_root(self) -> &'static str {
        match self {
            Self::P1Contractive => "primitive-hybrid-p1",
            Self::P20RotaryStateOutput => "primitive-hybrid-p2-0",
            Self::P2RotaryReadout => "primitive-hybrid-p2",
            Self::P23RotaryCarryBlendReadout => "primitive-hybrid-p2-3",
            Self::P21WideLatent => "primitive-hybrid-p2-1",
            Self::P22WideLatentReadout => "primitive-hybrid-p2-2",
        }
    }

    pub const fn state_width(self, d_model: usize) -> usize {
        match self {
            Self::P1Contractive
            | Self::P20RotaryStateOutput
            | Self::P2RotaryReadout
            | Self::P23RotaryCarryBlendReadout => d_model,
            Self::P21WideLatent | Self::P22WideLatentReadout => d_model * 2,
        }
    }

    pub const fn has_explicit_internal_readout(self) -> bool {
        match self {
            Self::P1Contractive | Self::P20RotaryStateOutput | Self::P21WideLatent => false,
            Self::P2RotaryReadout
            | Self::P23RotaryCarryBlendReadout
            | Self::P22WideLatentReadout => true,
        }
    }

    pub const fn kernel_contract(self) -> HybridSequenceKernelContract {
        match self {
            Self::P1Contractive => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::ModelWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: false,
            },
            Self::P20RotaryStateOutput => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::ModelWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: false,
            },
            Self::P2RotaryReadout => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::ModelWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: true,
            },
            Self::P23RotaryCarryBlendReadout => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::ModelWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: true,
            },
            Self::P21WideLatent => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::DoubleWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: false,
            },
            Self::P22WideLatentReadout => HybridSequenceKernelContract {
                projection_layout_policy: ProjectionLayoutPolicy::OutputByInput,
                state_layout: HybridSequenceStateLayout::DoubleWidthLatent,
                scan_mode: HybridSequenceScanMode::SequentialStepLoop,
                explicit_output_readout: true,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HybridSequenceStateLayout {
    ModelWidthLatent,
    DoubleWidthLatent,
    StructuredReferenceSsm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HybridSequenceScanMode {
    SequentialStepLoop,
    ChunkedSequentialStepLoop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridSequenceKernelContract {
    pub projection_layout_policy: ProjectionLayoutPolicy,
    pub state_layout: HybridSequenceStateLayout,
    pub scan_mode: HybridSequenceScanMode,
    pub explicit_output_readout: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum P2LatentWidthFactor {
    Base,
    Wide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum P2InternalReadoutFactor {
    DirectState,
    ExplicitProjection,
}

pub const fn primitive_from_p2_factors(
    latent_width: P2LatentWidthFactor,
    internal_readout: P2InternalReadoutFactor,
) -> PrimitiveHybridPrimitive {
    match (latent_width, internal_readout) {
        (P2LatentWidthFactor::Base, P2InternalReadoutFactor::DirectState) => {
            PrimitiveHybridPrimitive::P20RotaryStateOutput
        }
        (P2LatentWidthFactor::Base, P2InternalReadoutFactor::ExplicitProjection) => {
            PrimitiveHybridPrimitive::P2RotaryReadout
        }
        (P2LatentWidthFactor::Wide, P2InternalReadoutFactor::DirectState) => {
            PrimitiveHybridPrimitive::P21WideLatent
        }
        (P2LatentWidthFactor::Wide, P2InternalReadoutFactor::ExplicitProjection) => {
            PrimitiveHybridPrimitive::P22WideLatentReadout
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridResidualMode {
    PlainAdd,
    ScaledAdd,
    GatedAdd,
}

impl PrimitiveHybridResidualMode {
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::PlainAdd => "plain",
            Self::ScaledAdd => "scaled",
            Self::GatedAdd => "gated",
        }
    }

    pub const fn label_suffix(self) -> &'static str {
        match self {
            Self::PlainAdd => "",
            Self::ScaledAdd => "-scaled-residual",
            Self::GatedAdd => "-gated-residual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridReadoutMode {
    Direct,
    Projected,
    ProjectedNorm,
}

impl PrimitiveHybridReadoutMode {
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Projected => "projected",
            Self::ProjectedNorm => "projected-norm",
        }
    }

    pub const fn label_suffix(self) -> &'static str {
        match self {
            Self::Direct => "",
            Self::Projected => "-projected-readout",
            Self::ProjectedNorm => "-projected-norm-readout",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridNormMode {
    PreNormOnly,
    PostReadoutNorm,
    ResidualSideRenorm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridWrapperSymmetryMode {
    Standard,
    MambaRms,
}

impl PrimitiveHybridWrapperSymmetryMode {
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::MambaRms => "mamba-rms",
        }
    }

    pub const fn label_suffix(self) -> &'static str {
        match self {
            Self::Standard => "",
            Self::MambaRms => "-mamba-rms-wrapper",
        }
    }
}

impl PrimitiveHybridNormMode {
    pub const fn cli_name(self) -> &'static str {
        match self {
            Self::PreNormOnly => "pre-norm-only",
            Self::PostReadoutNorm => "post-readout-norm",
            Self::ResidualSideRenorm => "residual-renorm",
        }
    }

    pub const fn label_suffix(self) -> &'static str {
        match self {
            Self::PreNormOnly => "",
            Self::PostReadoutNorm => "-post-readout-norm",
            Self::ResidualSideRenorm => "-residual-renorm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HybridAttentionLayerRole {
    ExactAttention,
    ReferenceSsm,
    Primitive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HybridAttentionEfficiencyTarget {
    DecodeThroughput,
    ActivationMemory,
    KvCacheSize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionComparisonContract {
    pub matched_total_layers: bool,
    pub matched_hidden_dim: bool,
    pub matched_head_count: bool,
    pub matched_local_window: bool,
    pub matched_training_budget: bool,
    pub matched_eval_suites: bool,
    pub primary_efficiency_target: HybridAttentionEfficiencyTarget,
}

impl HybridAttentionComparisonContract {
    pub const fn phase1_default() -> Self {
        Self {
            matched_total_layers: true,
            matched_hidden_dim: true,
            matched_head_count: true,
            matched_local_window: true,
            matched_training_budget: true,
            matched_eval_suites: true,
            primary_efficiency_target: HybridAttentionEfficiencyTarget::DecodeThroughput,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionVariantSpec {
    pub kind: HybridAttentionVariantKind,
    pub label: String,
    pub hidden_dim: usize,
    pub head_count: usize,
    pub local_window: usize,
    pub layer_schedule: Vec<HybridAttentionLayerRole>,
    pub reference_ssm_family: Option<ReferenceSsmFamily>,
    pub primitive: Option<PrimitiveHybridPrimitive>,
    #[serde(default)]
    pub primitive_residual_mode: Option<PrimitiveHybridResidualMode>,
    #[serde(default)]
    pub primitive_readout_mode: Option<PrimitiveHybridReadoutMode>,
    #[serde(default)]
    pub primitive_norm_mode: Option<PrimitiveHybridNormMode>,
    #[serde(default)]
    pub primitive_wrapper_symmetry_mode: Option<PrimitiveHybridWrapperSymmetryMode>,
}

impl HybridAttentionVariantSpec {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.hidden_dim == 0 {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention.variant[{label}].hidden_dim must be greater than zero",
                label = self.label
            )));
        }
        if self.head_count == 0 {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention.variant[{label}].head_count must be greater than zero",
                label = self.label
            )));
        }
        if self.local_window == 0 {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention.variant[{label}].local_window must be greater than zero",
                label = self.label
            )));
        }
        if self.layer_schedule.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention.variant[{label}].layer_schedule must not be empty",
                label = self.label
            )));
        }

        let exact_attention_layers = self
            .layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ExactAttention))
            .count();
        if exact_attention_layers == 0 {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_attention.variant[{label}] must retain at least one exact-attention layer",
                label = self.label
            )));
        }

        let has_reference_ssm = self
            .layer_schedule
            .iter()
            .any(|role| matches!(role, HybridAttentionLayerRole::ReferenceSsm));
        let has_primitive = self
            .layer_schedule
            .iter()
            .any(|role| matches!(role, HybridAttentionLayerRole::Primitive));

        match self.kind {
            HybridAttentionVariantKind::AttentionOnly => {
                if self.reference_ssm_family.is_some()
                    || self.primitive.is_some()
                    || self.primitive_residual_mode.is_some()
                    || self.primitive_readout_mode.is_some()
                    || self.primitive_norm_mode.is_some()
                    || self.primitive_wrapper_symmetry_mode.is_some()
                {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] attention-only baseline must not set reference_ssm_family, primitive, primitive_residual_mode, primitive_readout_mode, primitive_norm_mode, or primitive_wrapper_symmetry_mode",
                        label = self.label
                    )));
                }
                if self
                    .layer_schedule
                    .iter()
                    .any(|role| !matches!(role, HybridAttentionLayerRole::ExactAttention))
                {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] attention-only baseline must contain only exact-attention layers",
                        label = self.label
                    )));
                }
            }
            HybridAttentionVariantKind::ReferenceSsmHybrid => {
                if self.reference_ssm_family.is_none()
                    || self.primitive.is_some()
                    || self.primitive_residual_mode.is_some()
                    || self.primitive_readout_mode.is_some()
                    || self.primitive_norm_mode.is_some()
                    || self.primitive_wrapper_symmetry_mode.is_some()
                {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] reference SSM hybrid must set reference_ssm_family and must not set primitive, primitive_residual_mode, primitive_readout_mode, primitive_norm_mode, or primitive_wrapper_symmetry_mode",
                        label = self.label
                    )));
                }
                if !has_reference_ssm || has_primitive {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] reference SSM hybrid must contain exact-attention and reference-SSM layers only",
                        label = self.label
                    )));
                }
            }
            HybridAttentionVariantKind::PrimitiveHybrid => {
                if self.reference_ssm_family.is_some()
                    || self.primitive.is_none()
                    || self.primitive_residual_mode.is_none()
                    || self.primitive_readout_mode.is_none()
                    || self.primitive_norm_mode.is_none()
                    || self.primitive_wrapper_symmetry_mode.is_none()
                {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] primitive hybrid must set primitive, primitive_residual_mode, primitive_readout_mode, primitive_norm_mode, and primitive_wrapper_symmetry_mode, and must not set reference_ssm_family",
                        label = self.label
                    )));
                }
                if !has_primitive || has_reference_ssm {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] primitive hybrid must contain exact-attention and primitive layers only",
                        label = self.label
                    )));
                }
                match self
                    .primitive
                    .expect("primitive hybrid validation already checked primitive.is_some()")
                {
                    PrimitiveHybridPrimitive::P1Contractive => {}
                    PrimitiveHybridPrimitive::P20RotaryStateOutput
                    | PrimitiveHybridPrimitive::P2RotaryReadout
                    | PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout => {
                        if !self.hidden_dim.is_multiple_of(2) {
                            return Err(FractalError::InvalidConfig(format!(
                                "hybrid_attention.variant[{label}] base-width rotary primitive requires even hidden_dim, got {}",
                                self.hidden_dim,
                                label = self.label
                            )));
                        }
                    }
                    PrimitiveHybridPrimitive::P21WideLatent
                    | PrimitiveHybridPrimitive::P22WideLatentReadout => {
                        let _ = self.hidden_dim.checked_mul(2).ok_or_else(|| {
                            FractalError::InvalidConfig(format!(
                                "hybrid_attention.variant[{label}] P2.1 latent width overflowed for hidden_dim {}",
                                self.hidden_dim,
                                label = self.label
                            ))
                        })?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn total_layers(&self) -> usize {
        self.layer_schedule.len()
    }

    pub fn sequence_kernel_contract(&self) -> Option<HybridSequenceKernelContract> {
        match self.kind {
            HybridAttentionVariantKind::AttentionOnly => None,
            HybridAttentionVariantKind::ReferenceSsmHybrid => self
                .reference_ssm_family
                .map(ReferenceSsmFamily::kernel_contract),
            HybridAttentionVariantKind::PrimitiveHybrid => self
                .primitive
                .map(PrimitiveHybridPrimitive::kernel_contract),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionBaselineMatrix {
    pub comparison: HybridAttentionComparisonContract,
    pub attention_only: HybridAttentionVariantSpec,
    pub reference_ssm_hybrid: HybridAttentionVariantSpec,
    pub primitive_hybrid: HybridAttentionVariantSpec,
}

impl HybridAttentionBaselineMatrix {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.attention_only.validate()?;
        self.reference_ssm_hybrid.validate()?;
        self.primitive_hybrid.validate()?;

        if !matches!(
            self.attention_only.kind,
            HybridAttentionVariantKind::AttentionOnly
        ) {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention.matrix.attention_only must be an attention-only baseline"
                    .to_string(),
            ));
        }
        if !matches!(
            self.reference_ssm_hybrid.kind,
            HybridAttentionVariantKind::ReferenceSsmHybrid
        ) {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention.matrix.reference_ssm_hybrid must be a reference SSM hybrid"
                    .to_string(),
            ));
        }
        if !matches!(
            self.primitive_hybrid.kind,
            HybridAttentionVariantKind::PrimitiveHybrid
        ) {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention.matrix.primitive_hybrid must be a primitive hybrid".to_string(),
            ));
        }

        let reference = &self.attention_only;
        for variant in [&self.reference_ssm_hybrid, &self.primitive_hybrid] {
            if self.comparison.matched_total_layers
                && variant.total_layers() != reference.total_layers()
            {
                return Err(FractalError::InvalidConfig(format!(
                    "hybrid_attention.matrix variant[{label}] must match total layer count {}",
                    reference.total_layers(),
                    label = variant.label
                )));
            }
            if self.comparison.matched_hidden_dim && variant.hidden_dim != reference.hidden_dim {
                return Err(FractalError::InvalidConfig(format!(
                    "hybrid_attention.matrix variant[{label}] must match hidden_dim {}",
                    reference.hidden_dim,
                    label = variant.label
                )));
            }
            if self.comparison.matched_head_count && variant.head_count != reference.head_count {
                return Err(FractalError::InvalidConfig(format!(
                    "hybrid_attention.matrix variant[{label}] must match head_count {}",
                    reference.head_count,
                    label = variant.label
                )));
            }
            if self.comparison.matched_local_window
                && variant.local_window != reference.local_window
            {
                return Err(FractalError::InvalidConfig(format!(
                    "hybrid_attention.matrix variant[{label}] must match local_window {}",
                    reference.local_window,
                    label = variant.label
                )));
            }
        }

        Ok(())
    }
}

pub fn phase1_hybrid_attention_baseline_matrix() -> HybridAttentionBaselineMatrix {
    let hidden_dim = 128;
    let head_count = 4;
    let local_window = PATH1_PHASE1_LOCAL_WINDOW_SIZE;

    let interleaved_reference_schedule = vec![
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::ReferenceSsm,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::ReferenceSsm,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::ReferenceSsm,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::ReferenceSsm,
    ];
    let interleaved_primitive_schedule = vec![
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::Primitive,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::Primitive,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::Primitive,
        HybridAttentionLayerRole::ExactAttention,
        HybridAttentionLayerRole::Primitive,
    ];

    HybridAttentionBaselineMatrix {
        comparison: HybridAttentionComparisonContract::phase1_default(),
        attention_only: HybridAttentionVariantSpec {
            kind: HybridAttentionVariantKind::AttentionOnly,
            label: "attention-only".to_string(),
            hidden_dim,
            head_count,
            local_window,
            layer_schedule: vec![HybridAttentionLayerRole::ExactAttention; 8],
            reference_ssm_family: None,
            primitive: None,
            primitive_residual_mode: None,
            primitive_readout_mode: None,
            primitive_norm_mode: None,
            primitive_wrapper_symmetry_mode: None,
        },
        reference_ssm_hybrid: HybridAttentionVariantSpec {
            kind: HybridAttentionVariantKind::ReferenceSsmHybrid,
            label: "reference-ssm-hybrid".to_string(),
            hidden_dim,
            head_count,
            local_window,
            layer_schedule: interleaved_reference_schedule,
            reference_ssm_family: Some(ReferenceSsmFamily::Mamba3RustV1),
            primitive: None,
            primitive_residual_mode: None,
            primitive_readout_mode: None,
            primitive_norm_mode: None,
            primitive_wrapper_symmetry_mode: None,
        },
        primitive_hybrid: HybridAttentionVariantSpec {
            kind: HybridAttentionVariantKind::PrimitiveHybrid,
            label: "primitive-hybrid".to_string(),
            hidden_dim,
            head_count,
            local_window,
            layer_schedule: interleaved_primitive_schedule,
            reference_ssm_family: None,
            primitive: Some(PrimitiveHybridPrimitive::P1Contractive),
            primitive_residual_mode: Some(PrimitiveHybridResidualMode::PlainAdd),
            primitive_readout_mode: Some(PrimitiveHybridReadoutMode::Direct),
            primitive_norm_mode: Some(PrimitiveHybridNormMode::PreNormOnly),
            primitive_wrapper_symmetry_mode: Some(PrimitiveHybridWrapperSymmetryMode::Standard),
        },
    }
}

pub fn phase1_p2_candidate_variant() -> HybridAttentionVariantSpec {
    phase1_p2_factor_candidate_variant(
        P2LatentWidthFactor::Base,
        P2InternalReadoutFactor::ExplicitProjection,
    )
}

pub fn phase1_p23_candidate_variant() -> HybridAttentionVariantSpec {
    let matrix = phase1_hybrid_attention_baseline_matrix();
    let mut variant = matrix.primitive_hybrid.clone();
    variant.label = PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout
        .label_root()
        .to_string();
    variant.primitive = Some(PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout);
    variant
}

pub fn phase1_p20_candidate_variant() -> HybridAttentionVariantSpec {
    phase1_p2_factor_candidate_variant(
        P2LatentWidthFactor::Base,
        P2InternalReadoutFactor::DirectState,
    )
}

pub fn phase1_p21_candidate_variant() -> HybridAttentionVariantSpec {
    phase1_p2_factor_candidate_variant(
        P2LatentWidthFactor::Wide,
        P2InternalReadoutFactor::DirectState,
    )
}

pub fn phase1_p22_candidate_variant() -> HybridAttentionVariantSpec {
    phase1_p2_factor_candidate_variant(
        P2LatentWidthFactor::Wide,
        P2InternalReadoutFactor::ExplicitProjection,
    )
}

pub fn phase1_p2_factor_candidate_variant(
    latent_width: P2LatentWidthFactor,
    internal_readout: P2InternalReadoutFactor,
) -> HybridAttentionVariantSpec {
    phase1_p2_interface_candidate_variant(
        primitive_from_p2_factors(latent_width, internal_readout),
        PrimitiveHybridResidualMode::PlainAdd,
        PrimitiveHybridReadoutMode::Direct,
        PrimitiveHybridNormMode::PreNormOnly,
        PrimitiveHybridWrapperSymmetryMode::Standard,
    )
}

pub fn phase1_p2_interface_candidate_variant(
    primitive: PrimitiveHybridPrimitive,
    residual_mode: PrimitiveHybridResidualMode,
    readout_mode: PrimitiveHybridReadoutMode,
    norm_mode: PrimitiveHybridNormMode,
    wrapper_symmetry_mode: PrimitiveHybridWrapperSymmetryMode,
) -> HybridAttentionVariantSpec {
    let mut variant = phase1_hybrid_attention_baseline_matrix().primitive_hybrid;
    variant.label = format!(
        "{}{}{}{}{}",
        primitive.label_root(),
        residual_mode.label_suffix(),
        readout_mode.label_suffix(),
        norm_mode.label_suffix(),
        wrapper_symmetry_mode.label_suffix(),
    );
    variant.primitive = Some(primitive);
    variant.primitive_residual_mode = Some(residual_mode);
    variant.primitive_readout_mode = Some(readout_mode);
    variant.primitive_norm_mode = Some(norm_mode);
    variant.primitive_wrapper_symmetry_mode = Some(wrapper_symmetry_mode);
    variant
}

#[cfg(test)]
mod tests {
    use super::{
        phase1_hybrid_attention_baseline_matrix, phase1_p20_candidate_variant,
        phase1_p21_candidate_variant, phase1_p22_candidate_variant, phase1_p23_candidate_variant,
        phase1_p2_candidate_variant, phase1_p2_factor_candidate_variant,
        phase1_p2_interface_candidate_variant, primitive_from_p2_factors, HybridAttentionLayerRole,
        HybridAttentionVariantKind, HybridSequenceScanMode, HybridSequenceStateLayout,
        P2InternalReadoutFactor, P2LatentWidthFactor, PrimitiveHybridNormMode,
        PrimitiveHybridPrimitive, PrimitiveHybridReadoutMode, PrimitiveHybridResidualMode,
        PrimitiveHybridWrapperSymmetryMode, ReferenceSsmFamily,
    };
    use crate::ProjectionLayoutPolicy;

    #[test]
    fn phase1_matrix_is_self_consistent() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        matrix.validate().expect("phase1 matrix should validate");

        assert_eq!(
            matrix.attention_only.kind,
            HybridAttentionVariantKind::AttentionOnly
        );
        assert_eq!(
            matrix.reference_ssm_hybrid.kind,
            HybridAttentionVariantKind::ReferenceSsmHybrid
        );
        assert_eq!(
            matrix.primitive_hybrid.kind,
            HybridAttentionVariantKind::PrimitiveHybrid
        );
        assert_eq!(
            matrix
                .attention_only
                .layer_schedule
                .iter()
                .filter(|role| matches!(role, HybridAttentionLayerRole::ExactAttention))
                .count(),
            8
        );
    }

    #[test]
    fn phase1_p2_candidate_preserves_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let p2 = phase1_p2_candidate_variant();

        assert_eq!(p2.kind, HybridAttentionVariantKind::PrimitiveHybrid);
        assert_eq!(p2.hidden_dim, matrix.primitive_hybrid.hidden_dim);
        assert_eq!(p2.head_count, matrix.primitive_hybrid.head_count);
        assert_eq!(p2.local_window, matrix.primitive_hybrid.local_window);
        assert_eq!(p2.layer_schedule, matrix.primitive_hybrid.layer_schedule);
        assert_eq!(
            p2.primitive,
            Some(PrimitiveHybridPrimitive::P2RotaryReadout)
        );
        assert_eq!(
            p2.primitive_residual_mode,
            Some(PrimitiveHybridResidualMode::PlainAdd)
        );
        assert_eq!(
            p2.primitive_readout_mode,
            Some(PrimitiveHybridReadoutMode::Direct)
        );
        assert_eq!(
            p2.primitive_norm_mode,
            Some(PrimitiveHybridNormMode::PreNormOnly)
        );
        assert_eq!(
            p2.primitive_wrapper_symmetry_mode,
            Some(PrimitiveHybridWrapperSymmetryMode::Standard)
        );
        p2.validate().expect("p2 candidate variant should validate");
    }

    #[test]
    fn phase1_p20_candidate_preserves_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let p20 = phase1_p20_candidate_variant();

        assert_eq!(p20.kind, HybridAttentionVariantKind::PrimitiveHybrid);
        assert_eq!(p20.hidden_dim, matrix.primitive_hybrid.hidden_dim);
        assert_eq!(p20.head_count, matrix.primitive_hybrid.head_count);
        assert_eq!(p20.local_window, matrix.primitive_hybrid.local_window);
        assert_eq!(p20.layer_schedule, matrix.primitive_hybrid.layer_schedule);
        assert_eq!(
            p20.primitive,
            Some(PrimitiveHybridPrimitive::P20RotaryStateOutput)
        );
        p20.validate()
            .expect("p2.0 candidate variant should validate");
    }

    #[test]
    fn phase1_p21_candidate_preserves_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let p21 = phase1_p21_candidate_variant();

        assert_eq!(p21.kind, HybridAttentionVariantKind::PrimitiveHybrid);
        assert_eq!(p21.hidden_dim, matrix.primitive_hybrid.hidden_dim);
        assert_eq!(p21.head_count, matrix.primitive_hybrid.head_count);
        assert_eq!(p21.local_window, matrix.primitive_hybrid.local_window);
        assert_eq!(p21.layer_schedule, matrix.primitive_hybrid.layer_schedule);
        assert_eq!(p21.primitive, Some(PrimitiveHybridPrimitive::P21WideLatent));
        assert_eq!(
            p21.primitive_residual_mode,
            Some(PrimitiveHybridResidualMode::PlainAdd)
        );
        assert_eq!(
            p21.primitive_readout_mode,
            Some(PrimitiveHybridReadoutMode::Direct)
        );
        assert_eq!(
            p21.primitive_norm_mode,
            Some(PrimitiveHybridNormMode::PreNormOnly)
        );
        assert_eq!(
            p21.primitive_wrapper_symmetry_mode,
            Some(PrimitiveHybridWrapperSymmetryMode::Standard)
        );
        p21.validate()
            .expect("p2.1 candidate variant should validate");
    }

    #[test]
    fn phase1_p23_candidate_preserves_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let p23 = phase1_p23_candidate_variant();

        assert_eq!(p23.kind, HybridAttentionVariantKind::PrimitiveHybrid);
        assert_eq!(p23.hidden_dim, matrix.primitive_hybrid.hidden_dim);
        assert_eq!(p23.head_count, matrix.primitive_hybrid.head_count);
        assert_eq!(p23.local_window, matrix.primitive_hybrid.local_window);
        assert_eq!(p23.layer_schedule, matrix.primitive_hybrid.layer_schedule);
        assert_eq!(
            p23.primitive,
            Some(PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout)
        );
        assert_eq!(
            p23.primitive_residual_mode,
            Some(PrimitiveHybridResidualMode::PlainAdd)
        );
        assert_eq!(
            p23.primitive_readout_mode,
            Some(PrimitiveHybridReadoutMode::Direct)
        );
        assert_eq!(
            p23.primitive_norm_mode,
            Some(PrimitiveHybridNormMode::PreNormOnly)
        );
        assert_eq!(
            p23.primitive_wrapper_symmetry_mode,
            Some(PrimitiveHybridWrapperSymmetryMode::Standard)
        );
        p23.validate()
            .expect("p2.3 candidate variant should validate");
    }

    #[test]
    fn phase1_p22_candidate_preserves_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let p22 = phase1_p22_candidate_variant();

        assert_eq!(p22.kind, HybridAttentionVariantKind::PrimitiveHybrid);
        assert_eq!(p22.hidden_dim, matrix.primitive_hybrid.hidden_dim);
        assert_eq!(p22.head_count, matrix.primitive_hybrid.head_count);
        assert_eq!(p22.local_window, matrix.primitive_hybrid.local_window);
        assert_eq!(p22.layer_schedule, matrix.primitive_hybrid.layer_schedule);
        assert_eq!(
            p22.primitive,
            Some(PrimitiveHybridPrimitive::P22WideLatentReadout)
        );
        p22.validate()
            .expect("p2.2 candidate variant should validate");
    }

    #[test]
    fn p2_factor_mapping_covers_all_four_permutations() {
        assert_eq!(
            primitive_from_p2_factors(
                P2LatentWidthFactor::Base,
                P2InternalReadoutFactor::DirectState
            ),
            PrimitiveHybridPrimitive::P20RotaryStateOutput
        );
        assert_eq!(
            primitive_from_p2_factors(
                P2LatentWidthFactor::Base,
                P2InternalReadoutFactor::ExplicitProjection
            ),
            PrimitiveHybridPrimitive::P2RotaryReadout
        );
        assert_eq!(
            primitive_from_p2_factors(
                P2LatentWidthFactor::Wide,
                P2InternalReadoutFactor::DirectState
            ),
            PrimitiveHybridPrimitive::P21WideLatent
        );
        assert_eq!(
            primitive_from_p2_factors(
                P2LatentWidthFactor::Wide,
                P2InternalReadoutFactor::ExplicitProjection
            ),
            PrimitiveHybridPrimitive::P22WideLatentReadout
        );
    }

    #[test]
    fn phase1_p2_factor_candidates_preserve_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        for (latent_width, internal_readout, expected) in [
            (
                P2LatentWidthFactor::Base,
                P2InternalReadoutFactor::DirectState,
                PrimitiveHybridPrimitive::P20RotaryStateOutput,
            ),
            (
                P2LatentWidthFactor::Base,
                P2InternalReadoutFactor::ExplicitProjection,
                PrimitiveHybridPrimitive::P2RotaryReadout,
            ),
            (
                P2LatentWidthFactor::Wide,
                P2InternalReadoutFactor::DirectState,
                PrimitiveHybridPrimitive::P21WideLatent,
            ),
            (
                P2LatentWidthFactor::Wide,
                P2InternalReadoutFactor::ExplicitProjection,
                PrimitiveHybridPrimitive::P22WideLatentReadout,
            ),
        ] {
            let variant = phase1_p2_factor_candidate_variant(latent_width, internal_readout);
            assert_eq!(variant.kind, HybridAttentionVariantKind::PrimitiveHybrid);
            assert_eq!(variant.hidden_dim, matrix.primitive_hybrid.hidden_dim);
            assert_eq!(variant.head_count, matrix.primitive_hybrid.head_count);
            assert_eq!(variant.local_window, matrix.primitive_hybrid.local_window);
            assert_eq!(
                variant.layer_schedule,
                matrix.primitive_hybrid.layer_schedule
            );
            assert_eq!(variant.primitive, Some(expected));
            variant
                .validate()
                .expect("p2 factor candidate should validate");
        }
    }

    #[test]
    fn phase1_p2_interface_variants_preserve_comparison_contract() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
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
            assert_eq!(variant.kind, HybridAttentionVariantKind::PrimitiveHybrid);
            assert_eq!(variant.hidden_dim, matrix.primitive_hybrid.hidden_dim);
            assert_eq!(variant.head_count, matrix.primitive_hybrid.head_count);
            assert_eq!(variant.local_window, matrix.primitive_hybrid.local_window);
            assert_eq!(
                variant.layer_schedule,
                matrix.primitive_hybrid.layer_schedule
            );
            assert_eq!(
                variant.primitive,
                Some(PrimitiveHybridPrimitive::P2RotaryReadout)
            );
            assert_eq!(variant.primitive_residual_mode, Some(residual_mode));
            assert_eq!(
                variant.primitive_readout_mode,
                Some(PrimitiveHybridReadoutMode::Direct)
            );
            assert_eq!(
                variant.primitive_norm_mode,
                Some(PrimitiveHybridNormMode::PreNormOnly)
            );
            assert_eq!(
                variant.primitive_wrapper_symmetry_mode,
                Some(PrimitiveHybridWrapperSymmetryMode::Standard)
            );
            variant
                .validate()
                .expect("p2 interface candidate variant should validate");
        }
    }

    #[test]
    fn all_path1_sequence_mixers_expose_output_by_input_kernel_contracts() {
        for primitive in [
            PrimitiveHybridPrimitive::P1Contractive,
            PrimitiveHybridPrimitive::P20RotaryStateOutput,
            PrimitiveHybridPrimitive::P2RotaryReadout,
            PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout,
            PrimitiveHybridPrimitive::P21WideLatent,
            PrimitiveHybridPrimitive::P22WideLatentReadout,
        ] {
            assert_eq!(
                primitive.kernel_contract().projection_layout_policy,
                ProjectionLayoutPolicy::OutputByInput
            );
        }
        for family in [
            ReferenceSsmFamily::Mamba3ProxyV1,
            ReferenceSsmFamily::Mamba3RustV1,
        ] {
            assert_eq!(
                family.kernel_contract().projection_layout_policy,
                ProjectionLayoutPolicy::OutputByInput
            );
        }
    }

    #[test]
    fn path1_kernel_contracts_make_scan_mode_and_state_layout_explicit() {
        let base = PrimitiveHybridPrimitive::P2RotaryReadout.kernel_contract();
        assert_eq!(base.scan_mode, HybridSequenceScanMode::SequentialStepLoop);
        assert_eq!(
            base.state_layout,
            HybridSequenceStateLayout::ModelWidthLatent
        );
        assert!(base.explicit_output_readout);

        let wide = PrimitiveHybridPrimitive::P22WideLatentReadout.kernel_contract();
        assert_eq!(wide.scan_mode, HybridSequenceScanMode::SequentialStepLoop);
        assert_eq!(
            wide.state_layout,
            HybridSequenceStateLayout::DoubleWidthLatent
        );
        assert!(wide.explicit_output_readout);

        let rust_mamba = ReferenceSsmFamily::Mamba3RustV1.kernel_contract();
        assert_eq!(
            rust_mamba.scan_mode,
            HybridSequenceScanMode::ChunkedSequentialStepLoop
        );
        assert_eq!(
            rust_mamba.state_layout,
            HybridSequenceStateLayout::StructuredReferenceSsm
        );
        assert!(rust_mamba.explicit_output_readout);
    }

    #[test]
    fn variant_sequence_kernel_contract_matches_selected_lane() {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        assert!(matrix.attention_only.sequence_kernel_contract().is_none());
        assert_eq!(
            matrix
                .reference_ssm_hybrid
                .sequence_kernel_contract()
                .expect("reference SSM variant should expose a kernel contract"),
            ReferenceSsmFamily::Mamba3RustV1.kernel_contract()
        );
        assert_eq!(
            phase1_p23_candidate_variant()
                .sequence_kernel_contract()
                .expect("p2.3 candidate should expose a kernel contract"),
            PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout.kernel_contract()
        );
    }
}
