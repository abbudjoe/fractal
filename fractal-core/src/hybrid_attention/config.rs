use serde::{Deserialize, Serialize};

use crate::{error::FractalError, registry::PrimitiveVariantName};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrimitiveHybridPrimitive {
    P1Contractive,
}

impl PrimitiveHybridPrimitive {
    pub const fn variant_name(self) -> PrimitiveVariantName {
        match self {
            Self::P1Contractive => PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
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
                if self.reference_ssm_family.is_some() || self.primitive.is_some() {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] attention-only baseline must not set reference_ssm_family or primitive",
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
                if self.reference_ssm_family.is_none() || self.primitive.is_some() {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] reference SSM hybrid must set reference_ssm_family and must not set primitive",
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
                if self.reference_ssm_family.is_some() || self.primitive.is_none() {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] primitive hybrid must set primitive and must not set reference_ssm_family",
                        label = self.label
                    )));
                }
                if !has_primitive || has_reference_ssm {
                    return Err(FractalError::InvalidConfig(format!(
                        "hybrid_attention.variant[{label}] primitive hybrid must contain exact-attention and primitive layers only",
                        label = self.label
                    )));
                }
            }
        }

        Ok(())
    }

    pub fn total_layers(&self) -> usize {
        self.layer_schedule.len()
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
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        phase1_hybrid_attention_baseline_matrix, HybridAttentionLayerRole,
        HybridAttentionVariantKind,
    };

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
}
