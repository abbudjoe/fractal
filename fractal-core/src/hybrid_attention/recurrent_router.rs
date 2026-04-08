use serde::{Deserialize, Serialize};

use super::goe::GOE_CHANNEL_COUNT;
use crate::error::FractalError;

pub const DEFAULT_RECURRENT_ROUTER_STATE_WIDTH: usize = 64;
pub const DEFAULT_RECURRENT_ROUTER_ROUND_COUNT: usize = 2;
pub const MAX_RECURRENT_ROUTER_ROUND_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecurrentRouterPrimitiveKind {
    GruVirtualNode,
    P1ContractiveVirtualNode,
    P2RotaryReadoutVirtualNode,
    ReferenceSsmVirtualNode,
}

impl RecurrentRouterPrimitiveKind {
    pub const fn label_stem(self) -> &'static str {
        match self {
            Self::GruVirtualNode => "gru-virtual-node",
            Self::P1ContractiveVirtualNode => "p1-virtual-node",
            Self::P2RotaryReadoutVirtualNode => "p2-virtual-node",
            Self::ReferenceSsmVirtualNode => "reference-ssm-virtual-node",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecurrentRouterFeedbackMode {
    RouterStateOnly,
    AggregatedExpertOutput,
}

impl RecurrentRouterFeedbackMode {
    pub const fn label_stem(self) -> &'static str {
        match self {
            Self::RouterStateOnly => "state-only",
            Self::AggregatedExpertOutput => "expert-feedback",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RecurrentRouterSelectionMode {
    DenseSoftmax,
    SparseTopK,
}

impl RecurrentRouterSelectionMode {
    pub const fn label_stem(self) -> &'static str {
        match self {
            Self::DenseSoftmax => "dense-softmax",
            Self::SparseTopK => "sparse-topk",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentRouterSpec {
    pub primitive_kind: RecurrentRouterPrimitiveKind,
    pub feedback_mode: RecurrentRouterFeedbackMode,
    pub selection_mode: RecurrentRouterSelectionMode,
    pub round_count: usize,
    pub state_width: usize,
    pub channel_count: usize,
    pub top_k: usize,
}

impl RecurrentRouterSpec {
    pub const fn minimal_dense_gru() -> Self {
        Self {
            primitive_kind: RecurrentRouterPrimitiveKind::GruVirtualNode,
            feedback_mode: RecurrentRouterFeedbackMode::RouterStateOnly,
            selection_mode: RecurrentRouterSelectionMode::DenseSoftmax,
            round_count: DEFAULT_RECURRENT_ROUTER_ROUND_COUNT,
            state_width: DEFAULT_RECURRENT_ROUTER_STATE_WIDTH,
            channel_count: GOE_CHANNEL_COUNT,
            top_k: GOE_CHANNEL_COUNT,
        }
    }

    pub const fn minimal_dense_gru_with_expert_feedback() -> Self {
        Self {
            primitive_kind: RecurrentRouterPrimitiveKind::GruVirtualNode,
            feedback_mode: RecurrentRouterFeedbackMode::AggregatedExpertOutput,
            selection_mode: RecurrentRouterSelectionMode::DenseSoftmax,
            round_count: DEFAULT_RECURRENT_ROUTER_ROUND_COUNT,
            state_width: DEFAULT_RECURRENT_ROUTER_STATE_WIDTH,
            channel_count: GOE_CHANNEL_COUNT,
            top_k: GOE_CHANNEL_COUNT,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.round_count < 2 || self.round_count > MAX_RECURRENT_ROUTER_ROUND_COUNT {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_router.round_count must be between 2 and {MAX_RECURRENT_ROUTER_ROUND_COUNT}, got {}",
                self.round_count
            )));
        }
        if self.state_width == 0 {
            return Err(FractalError::InvalidConfig(
                "recurrent_router.state_width must be greater than zero".to_string(),
            ));
        }
        if self.channel_count < 2 {
            return Err(FractalError::InvalidConfig(format!(
                "recurrent_router.channel_count must be at least 2, got {}",
                self.channel_count
            )));
        }
        match self.selection_mode {
            RecurrentRouterSelectionMode::DenseSoftmax => {
                if self.top_k != self.channel_count {
                    return Err(FractalError::InvalidConfig(format!(
                        "recurrent_router.top_k must equal channel_count {} for dense-softmax routing, got {}",
                        self.channel_count, self.top_k
                    )));
                }
            }
            RecurrentRouterSelectionMode::SparseTopK => {
                if self.top_k == 0 || self.top_k > self.channel_count {
                    return Err(FractalError::InvalidConfig(format!(
                        "recurrent_router.top_k must be between 1 and channel_count {} for sparse-topk routing, got {}",
                        self.channel_count, self.top_k
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn label_suffix(&self) -> String {
        format!(
            "{}-{}-{}-r{}-s{}",
            self.primitive_kind.label_stem(),
            self.feedback_mode.label_stem(),
            self.selection_mode.label_stem(),
            self.round_count,
            self.state_width
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VirtualNodeRecurrentRouter {
    pub d_model: usize,
    pub spec: RecurrentRouterSpec,
}

impl VirtualNodeRecurrentRouter {
    pub fn new(d_model: usize, spec: RecurrentRouterSpec) -> Result<Self, FractalError> {
        let contract = Self { d_model, spec };
        contract.validate()?;
        Ok(contract)
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.d_model == 0 {
            return Err(FractalError::InvalidConfig(
                "virtual_node_recurrent_router.d_model must be greater than zero".to_string(),
            ));
        }
        self.spec.validate()?;
        if self.spec.state_width > self.d_model {
            return Err(FractalError::InvalidConfig(format!(
                "virtual_node_recurrent_router.state_width {} must not exceed d_model {} in the minimal scaffold",
                self.spec.state_width, self.d_model
            )));
        }
        Ok(())
    }

    pub fn label(&self) -> String {
        format!("virtual-node-{}", self.spec.label_suffix())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentRouterExperimentVariantKind {
    OneShotDenseBaseline,
    RecurrentDenseVirtualNode,
    RecurrentDenseVirtualNodeWithExpertFeedback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentRouterExperimentVariantSpec {
    pub kind: RecurrentRouterExperimentVariantKind,
    pub label: String,
    pub router: Option<RecurrentRouterSpec>,
}

pub fn minimal_recurrent_router_experiment_matrix() -> Vec<RecurrentRouterExperimentVariantSpec> {
    vec![
        RecurrentRouterExperimentVariantSpec {
            kind: RecurrentRouterExperimentVariantKind::OneShotDenseBaseline,
            label: "dreegmor-one-shot-dense".to_string(),
            router: None,
        },
        RecurrentRouterExperimentVariantSpec {
            kind: RecurrentRouterExperimentVariantKind::RecurrentDenseVirtualNode,
            label: "dreegmor-recurrent-dense".to_string(),
            router: Some(RecurrentRouterSpec::minimal_dense_gru()),
        },
        RecurrentRouterExperimentVariantSpec {
            kind: RecurrentRouterExperimentVariantKind::RecurrentDenseVirtualNodeWithExpertFeedback,
            label: "dreegmor-recurrent-dense-feedback".to_string(),
            router: Some(RecurrentRouterSpec::minimal_dense_gru_with_expert_feedback()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        minimal_recurrent_router_experiment_matrix, RecurrentRouterFeedbackMode,
        RecurrentRouterSelectionMode, RecurrentRouterSpec, VirtualNodeRecurrentRouter,
        DEFAULT_RECURRENT_ROUTER_STATE_WIDTH,
    };

    #[test]
    fn minimal_dense_gru_spec_validates() {
        let spec = RecurrentRouterSpec::minimal_dense_gru();
        spec.validate().unwrap();
        assert_eq!(
            spec.feedback_mode,
            RecurrentRouterFeedbackMode::RouterStateOnly
        );
        assert_eq!(
            spec.selection_mode,
            RecurrentRouterSelectionMode::DenseSoftmax
        );
    }

    #[test]
    fn virtual_node_router_requires_state_width_not_exceed_model_width() {
        let error = VirtualNodeRecurrentRouter::new(32, RecurrentRouterSpec::minimal_dense_gru())
            .unwrap_err();
        assert!(error.to_string().contains("must not exceed d_model"));
    }

    #[test]
    fn virtual_node_router_label_stays_explicit() {
        let router = VirtualNodeRecurrentRouter::new(
            DEFAULT_RECURRENT_ROUTER_STATE_WIDTH,
            RecurrentRouterSpec::minimal_dense_gru_with_expert_feedback(),
        )
        .unwrap();
        assert_eq!(
            router.label(),
            "virtual-node-gru-virtual-node-expert-feedback-dense-softmax-r2-s64"
        );
    }

    #[test]
    fn minimal_recurrent_router_matrix_stays_narrow_and_ordered() {
        let matrix = minimal_recurrent_router_experiment_matrix();
        assert_eq!(matrix.len(), 3);
        assert!(matrix[0].router.is_none());
        assert_eq!(
            matrix[1].router.as_ref().unwrap().feedback_mode,
            RecurrentRouterFeedbackMode::RouterStateOnly
        );
        assert_eq!(
            matrix[2].router.as_ref().unwrap().feedback_mode,
            RecurrentRouterFeedbackMode::AggregatedExpertOutput
        );
    }
}
