use std::{
    cell::RefCell,
    collections::BTreeSet,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

use serde::{Deserialize, Serialize};

use crate::{error::FractalError, lifecycle::RunPhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticProbeKind {
    TrainStep,
    ForwardBoundary,
    ForwardPosition,
    RuleProjection,
    OutputProjection,
    LossBoundary,
    BackwardBoundary,
    OptimizerBoundary,
    CudaMemorySnapshot,
}

impl DiagnosticProbeKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrainStep => "train_step",
            Self::ForwardBoundary => "forward_boundary",
            Self::ForwardPosition => "forward_position",
            Self::RuleProjection => "rule_projection",
            Self::OutputProjection => "output_projection",
            Self::LossBoundary => "loss_boundary",
            Self::BackwardBoundary => "backward_boundary",
            Self::OptimizerBoundary => "optimizer_boundary",
            Self::CudaMemorySnapshot => "cuda_memory_snapshot",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProbeCadence {
    EveryStep,
    StepInterval { steps: usize },
    FirstNSteps { steps: usize },
}

impl ProbeCadence {
    pub fn validate(&self) -> Result<(), FractalError> {
        match self {
            Self::EveryStep => Ok(()),
            Self::StepInterval { steps } | Self::FirstNSteps { steps } if *steps == 0 => {
                Err(FractalError::InvalidConfig(
                    "diagnostic cadence steps must be greater than zero".into(),
                ))
            }
            Self::StepInterval { .. } | Self::FirstNSteps { .. } => Ok(()),
        }
    }

    pub fn matches_step(&self, step: usize) -> bool {
        match self {
            Self::EveryStep => true,
            Self::StepInterval { steps } => step.is_multiple_of(*steps),
            Self::FirstNSteps { steps } => step < *steps,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticProbeRequest {
    pub kind: DiagnosticProbeKind,
    pub cadence: ProbeCadence,
    #[serde(default)]
    pub position_interval: Option<usize>,
}

impl DiagnosticProbeRequest {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.cadence.validate()?;
        match (self.kind, self.position_interval) {
            (
                DiagnosticProbeKind::ForwardPosition
                | DiagnosticProbeKind::RuleProjection
                | DiagnosticProbeKind::OutputProjection,
                Some(0),
            ) => Err(FractalError::InvalidConfig(format!(
                "{} position_interval must be greater than zero",
                self.kind.as_str()
            ))),
            (
                DiagnosticProbeKind::ForwardPosition
                | DiagnosticProbeKind::RuleProjection
                | DiagnosticProbeKind::OutputProjection,
                None,
            ) => Err(FractalError::InvalidConfig(format!(
                "{} diagnostics require position_interval",
                self.kind.as_str()
            ))),
            (
                DiagnosticProbeKind::ForwardPosition
                | DiagnosticProbeKind::RuleProjection
                | DiagnosticProbeKind::OutputProjection,
                Some(_),
            ) => Ok(()),
            (_, Some(_)) => Err(FractalError::InvalidConfig(format!(
                "{} does not support position_interval",
                self.kind.as_str()
            ))),
            (_, None) => Ok(()),
        }
    }

    pub fn matches_step(&self, step: usize) -> bool {
        self.cadence.matches_step(step)
    }

    pub fn matches_position(&self, position: usize) -> bool {
        self.position_interval
            .is_some_and(|interval| position.is_multiple_of(interval))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredDiagnosticsOutput {
    #[default]
    Jsonl,
}

impl StructuredDiagnosticsOutput {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Jsonl => "jsonl",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticsPolicy {
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub probes: Vec<DiagnosticProbeRequest>,
    #[serde(default)]
    pub structured_output: StructuredDiagnosticsOutput,
}

impl Default for DiagnosticsPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

impl DiagnosticsPolicy {
    pub fn disabled() -> Self {
        Self {
            required: false,
            probes: Vec::new(),
            structured_output: StructuredDiagnosticsOutput::Jsonl,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.required && self.probes.is_empty() {
            return Err(FractalError::InvalidConfig(
                "required diagnostics must declare at least one probe".into(),
            ));
        }

        let mut seen = BTreeSet::new();
        for probe in &self.probes {
            probe.validate()?;
            if !seen.insert(probe.kind) {
                return Err(FractalError::InvalidConfig(format!(
                    "diagnostics probe {} may only be configured once",
                    probe.kind.as_str()
                )));
            }
        }

        Ok(())
    }

    pub fn validate_against_backend(
        &self,
        backend: &crate::registry::ComputeBackend,
    ) -> Result<(), FractalError> {
        self.validate()?;
        #[cfg(feature = "cuda")]
        let has_cuda_backend =
            matches!(backend, crate::registry::ComputeBackend::CudaCandle { .. });
        #[cfg(not(feature = "cuda"))]
        let has_cuda_backend = {
            let _ = backend;
            false
        };
        if self.required
            && self
                .probe(DiagnosticProbeKind::CudaMemorySnapshot)
                .is_some()
            && !has_cuda_backend
        {
            return Err(FractalError::InvalidConfig(
                "required cuda_memory_snapshot diagnostics require a CUDA execution backend".into(),
            ));
        }
        Ok(())
    }

    pub fn label(&self) -> String {
        if self.probes.is_empty() {
            return "disabled".to_owned();
        }
        let probes = self
            .probes
            .iter()
            .map(|probe| {
                let mut label = format!(
                    "{}:{}",
                    probe.kind.as_str(),
                    match &probe.cadence {
                        ProbeCadence::EveryStep => "every_step".to_owned(),
                        ProbeCadence::StepInterval { steps } => {
                            format!("step_interval({steps})")
                        }
                        ProbeCadence::FirstNSteps { steps } => {
                            format!("first_n_steps({steps})")
                        }
                    }
                );
                if let Some(position_interval) = probe.position_interval {
                    label.push_str(&format!(":position_interval({position_interval})"));
                }
                label
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "required={} structured_output={} probes=[{}]",
            self.required,
            self.structured_output.as_str(),
            probes
        )
    }

    pub fn probe(&self, kind: DiagnosticProbeKind) -> Option<&DiagnosticProbeRequest> {
        self.probes.iter().find(|probe| probe.kind == kind)
    }

    pub fn has_probes(&self) -> bool {
        !self.probes.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticBoundary {
    TrainStepStart,
    ForwardStart,
    ForwardComplete,
    LossStart,
    LossComplete,
    BackwardStart,
    BackwardComplete,
    OptimizerStepStart,
    OptimizerStepComplete,
}

impl DiagnosticBoundary {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TrainStepStart => "train_step_start",
            Self::ForwardStart => "forward_start",
            Self::ForwardComplete => "forward_complete",
            Self::LossStart => "loss_start",
            Self::LossComplete => "loss_complete",
            Self::BackwardStart => "backward_start",
            Self::BackwardComplete => "backward_complete",
            Self::OptimizerStepStart => "optimizer_step_start",
            Self::OptimizerStepComplete => "optimizer_step_complete",
        }
    }

    pub const fn required_completion(self) -> Option<Self> {
        match self {
            Self::ForwardStart => Some(Self::ForwardComplete),
            Self::LossStart => Some(Self::LossComplete),
            Self::BackwardStart => Some(Self::BackwardComplete),
            Self::OptimizerStepStart => Some(Self::OptimizerStepComplete),
            Self::TrainStepStart
            | Self::ForwardComplete
            | Self::LossComplete
            | Self::BackwardComplete
            | Self::OptimizerStepComplete => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainStepDiagnosticContext {
    pub step: usize,
    pub tokens_seen: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProjectionDiagnosticContext {
    pub step: usize,
    pub tokens_seen: usize,
    pub position: usize,
    pub sequence_length: usize,
    pub recursion_depth: usize,
    pub max_recursion_depth: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleProjectionKind {
    Gate,
    StateMix,
    InputMix,
}

impl RuleProjectionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::StateMix => "state_mix",
            Self::InputMix => "input_mix",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProjectionIdentity {
    pub rule_name: String,
    pub projection_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayoutOrigin {
    RuntimeObserved,
    LinearContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayoutTransform {
    Identity,
    UnsqueezedView,
    TransposedView,
    TransposedUnsqueezedView,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLayoutMetadata {
    pub origin: TensorLayoutOrigin,
    pub transform: TensorLayoutTransform,
    pub shape: Vec<usize>,
    pub strides: Option<Vec<usize>>,
    pub is_contiguous: Option<bool>,
}

impl TensorLayoutMetadata {
    pub fn observed(shape: Vec<usize>) -> Self {
        Self {
            origin: TensorLayoutOrigin::RuntimeObserved,
            transform: TensorLayoutTransform::Identity,
            shape,
            strides: None,
            is_contiguous: None,
        }
    }

    pub fn linear_contract(shape: Vec<usize>, transform: TensorLayoutTransform) -> Self {
        Self {
            origin: TensorLayoutOrigin::LinearContract,
            transform,
            shape,
            strides: None,
            is_contiguous: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearProjectionLayoutMetadata {
    pub stored_weight: TensorLayoutMetadata,
    pub forward_rhs: TensorLayoutMetadata,
    pub backward_input_grad_rhs: TensorLayoutMetadata,
}

impl LinearProjectionLayoutMetadata {
    pub fn burn_linear_contract(weight_shape: Vec<usize>) -> Self {
        let [d_input, d_output]: [usize; 2] = weight_shape
            .clone()
            .try_into()
            .expect("linear weight shape should be rank 2");
        Self {
            stored_weight: TensorLayoutMetadata::linear_contract(
                vec![d_input, d_output],
                TensorLayoutTransform::Identity,
            ),
            forward_rhs: TensorLayoutMetadata::linear_contract(
                vec![1, d_input, d_output],
                TensorLayoutTransform::UnsqueezedView,
            ),
            backward_input_grad_rhs: TensorLayoutMetadata::linear_contract(
                vec![d_output, d_input],
                TensorLayoutTransform::TransposedView,
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProjectionDiagnosticSpec {
    pub identity: RuleProjectionIdentity,
    pub input_layout: TensorLayoutMetadata,
    pub output_layout: TensorLayoutMetadata,
    pub linear_layout: Option<LinearProjectionLayoutMetadata>,
}

impl RuleProjectionDiagnosticSpec {
    pub fn linear(
        rule_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        weight_shape: Vec<usize>,
    ) -> Self {
        Self::linear_with_layout(
            rule_name,
            projection_name,
            input_shape,
            output_shape,
            LinearProjectionLayoutMetadata::burn_linear_contract(weight_shape),
        )
    }

    pub fn linear_with_layout(
        rule_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        linear_layout: LinearProjectionLayoutMetadata,
    ) -> Self {
        Self {
            identity: RuleProjectionIdentity {
                rule_name: rule_name.into(),
                projection_name: projection_name.into(),
            },
            input_layout: TensorLayoutMetadata::observed(input_shape),
            output_layout: TensorLayoutMetadata::observed(output_shape),
            linear_layout: Some(linear_layout),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProjectionDiagnosticContext {
    pub step: usize,
    pub tokens_seen: usize,
    pub position: usize,
    pub sequence_length: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProjectionIdentity {
    pub model_name: String,
    pub projection_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProjectionDiagnosticSpec {
    pub identity: OutputProjectionIdentity,
    pub input_layout: TensorLayoutMetadata,
    pub output_layout: TensorLayoutMetadata,
    pub linear_layout: Option<LinearProjectionLayoutMetadata>,
}

impl OutputProjectionDiagnosticSpec {
    pub fn linear(
        model_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        weight_shape: Vec<usize>,
    ) -> Self {
        Self::linear_with_layout(
            model_name,
            projection_name,
            input_shape,
            output_shape,
            LinearProjectionLayoutMetadata::burn_linear_contract(weight_shape),
        )
    }

    pub fn linear_with_layout(
        model_name: impl Into<String>,
        projection_name: impl Into<String>,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        linear_layout: LinearProjectionLayoutMetadata,
    ) -> Self {
        Self {
            identity: OutputProjectionIdentity {
                model_name: model_name.into(),
                projection_name: projection_name.into(),
            },
            input_layout: TensorLayoutMetadata::observed(input_shape),
            output_layout: TensorLayoutMetadata::observed(output_shape),
            linear_layout: Some(linear_layout),
        }
    }
}

pub trait ProjectionDiagnosticsSink {
    fn emit_rule_projection(
        &mut self,
        phase: RunPhase,
        context: RuleProjectionDiagnosticContext,
        projection: RuleProjectionKind,
        spec: RuleProjectionDiagnosticSpec,
    ) -> Result<(), FractalError>;

    fn emit_output_projection(
        &mut self,
        phase: RunPhase,
        context: OutputProjectionDiagnosticContext,
        spec: OutputProjectionDiagnosticSpec,
    ) -> Result<(), FractalError>;
}

pub trait RuleProjectionDiagnosticsSink: ProjectionDiagnosticsSink {}

impl<T: ProjectionDiagnosticsSink + ?Sized> RuleProjectionDiagnosticsSink for T {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaMemorySnapshot {
    pub used_mib: usize,
    pub free_mib: usize,
    pub total_mib: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardGraphBurden {
    pub batch_size: usize,
    pub sequence_length: usize,
    pub positions_processed: usize,
    pub rule_invocations: usize,
    pub max_recursion_depth_observed: usize,
    pub router_enabled: bool,
    pub forced_depth: Option<usize>,
    pub positions_early_exited: usize,
    pub positions_reached_depth_limit: usize,
    pub router_exit_elements: usize,
    pub router_continue_elements: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryMemoryDelta {
    pub boundary: DiagnosticBoundary,
    pub step: usize,
    pub tokens_seen: usize,
    pub correlation_id: u64,
    pub used_mib: usize,
    pub free_mib: usize,
    pub total_mib: usize,
    pub delta_used_mib: Option<i64>,
    pub delta_free_mib: Option<i64>,
    pub delta_total_mib: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagnosticIdentity {
    pub experiment_run_id: String,
    pub experiment_logical_name: Option<String>,
    pub species: String,
    pub variant_name: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DiagnosticEventKind {
    TrainStepStart {
        planned_steps: usize,
        batch_token_count: usize,
        input_shape: Vec<usize>,
        target_shape: Vec<usize>,
    },
    ForwardStart {
        input_shape: Vec<usize>,
    },
    ForwardPosition {
        position: usize,
        sequence_length: usize,
        input_shape: Vec<usize>,
        readout_shape: Vec<usize>,
    },
    RuleProjection {
        position: usize,
        sequence_length: usize,
        recursion_depth: usize,
        max_recursion_depth: usize,
        projection: RuleProjectionKind,
        spec: Box<RuleProjectionDiagnosticSpec>,
    },
    OutputProjection {
        position: usize,
        sequence_length: usize,
        spec: Box<OutputProjectionDiagnosticSpec>,
    },
    ForwardComplete {
        logits_shape: Vec<usize>,
        graph_burden: Box<ForwardGraphBurden>,
    },
    LossStart {
        target_shape: Vec<usize>,
    },
    LossComplete {
        loss_shape: Vec<usize>,
    },
    BackwardStart,
    BackwardComplete,
    OptimizerStepStart {
        scheduled_learning_rate: f64,
    },
    OptimizerStepComplete {
        scheduled_learning_rate: f64,
        completed_steps: usize,
        train_tokens_seen: usize,
    },
    CudaMemorySnapshot {
        boundary: DiagnosticBoundary,
        used_mib: usize,
        free_mib: usize,
        total_mib: usize,
    },
}

impl DiagnosticEventKind {
    pub fn probe_kind(&self) -> DiagnosticProbeKind {
        match self {
            Self::TrainStepStart { .. } => DiagnosticProbeKind::TrainStep,
            Self::ForwardStart { .. } | Self::ForwardComplete { .. } => {
                DiagnosticProbeKind::ForwardBoundary
            }
            Self::ForwardPosition { .. } => DiagnosticProbeKind::ForwardPosition,
            Self::RuleProjection { .. } => DiagnosticProbeKind::RuleProjection,
            Self::OutputProjection { .. } => DiagnosticProbeKind::OutputProjection,
            Self::LossStart { .. } | Self::LossComplete { .. } => DiagnosticProbeKind::LossBoundary,
            Self::BackwardStart | Self::BackwardComplete => DiagnosticProbeKind::BackwardBoundary,
            Self::OptimizerStepStart { .. } | Self::OptimizerStepComplete { .. } => {
                DiagnosticProbeKind::OptimizerBoundary
            }
            Self::CudaMemorySnapshot { .. } => DiagnosticProbeKind::CudaMemorySnapshot,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::TrainStepStart { .. } => "train_step_start",
            Self::ForwardStart { .. } => "forward_start",
            Self::ForwardPosition { .. } => "forward_position",
            Self::RuleProjection { .. } => "rule_projection",
            Self::OutputProjection { .. } => "output_projection",
            Self::ForwardComplete { .. } => "forward_complete",
            Self::LossStart { .. } => "loss_start",
            Self::LossComplete { .. } => "loss_complete",
            Self::BackwardStart => "backward_start",
            Self::BackwardComplete => "backward_complete",
            Self::OptimizerStepStart { .. } => "optimizer_step_start",
            Self::OptimizerStepComplete { .. } => "optimizer_step_complete",
            Self::CudaMemorySnapshot { .. } => "cuda_memory_snapshot",
        }
    }

    pub fn boundary(&self) -> Option<DiagnosticBoundary> {
        match self {
            Self::TrainStepStart { .. } => Some(DiagnosticBoundary::TrainStepStart),
            Self::ForwardStart { .. } => Some(DiagnosticBoundary::ForwardStart),
            Self::ForwardPosition { .. }
            | Self::RuleProjection { .. }
            | Self::OutputProjection { .. } => None,
            Self::ForwardComplete { .. } => Some(DiagnosticBoundary::ForwardComplete),
            Self::LossStart { .. } => Some(DiagnosticBoundary::LossStart),
            Self::LossComplete { .. } => Some(DiagnosticBoundary::LossComplete),
            Self::BackwardStart => Some(DiagnosticBoundary::BackwardStart),
            Self::BackwardComplete => Some(DiagnosticBoundary::BackwardComplete),
            Self::OptimizerStepStart { .. } => Some(DiagnosticBoundary::OptimizerStepStart),
            Self::OptimizerStepComplete { .. } => Some(DiagnosticBoundary::OptimizerStepComplete),
            Self::CudaMemorySnapshot { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticEvent {
    pub experiment_run_id: String,
    pub experiment_logical_name: Option<String>,
    pub species: String,
    pub variant_name: String,
    pub correlation_id: u64,
    pub phase: RunPhase,
    pub step: Option<usize>,
    pub tokens_seen: Option<usize>,
    pub event: DiagnosticEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticEventSummary {
    pub correlation_id: u64,
    pub phase: RunPhase,
    pub step: Option<usize>,
    pub tokens_seen: Option<usize>,
    pub event_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleProjectionDiagnosticEventSummary {
    pub correlation_id: u64,
    pub phase: RunPhase,
    pub step: Option<usize>,
    pub tokens_seen: Option<usize>,
    pub position: usize,
    pub sequence_length: usize,
    pub recursion_depth: usize,
    pub max_recursion_depth: usize,
    pub projection: RuleProjectionKind,
    pub identity: RuleProjectionIdentity,
    pub input_layout: TensorLayoutMetadata,
    pub output_layout: TensorLayoutMetadata,
    pub linear_layout: Option<LinearProjectionLayoutMetadata>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputProjectionDiagnosticEventSummary {
    pub correlation_id: u64,
    pub phase: RunPhase,
    pub step: Option<usize>,
    pub tokens_seen: Option<usize>,
    pub position: usize,
    pub sequence_length: usize,
    pub identity: OutputProjectionIdentity,
    pub input_layout: TensorLayoutMetadata,
    pub output_layout: TensorLayoutMetadata,
    pub linear_layout: Option<LinearProjectionLayoutMetadata>,
}

impl DiagnosticEvent {
    pub fn summary(&self) -> DiagnosticEventSummary {
        DiagnosticEventSummary {
            correlation_id: self.correlation_id,
            phase: self.phase,
            step: self.step,
            tokens_seen: self.tokens_seen,
            event_name: self.event.name().to_owned(),
        }
    }

    pub fn rule_projection_summary(&self) -> Option<RuleProjectionDiagnosticEventSummary> {
        let DiagnosticEventKind::RuleProjection {
            position,
            sequence_length,
            recursion_depth,
            max_recursion_depth,
            projection,
            spec,
        } = &self.event
        else {
            return None;
        };
        Some(RuleProjectionDiagnosticEventSummary {
            correlation_id: self.correlation_id,
            phase: self.phase,
            step: self.step,
            tokens_seen: self.tokens_seen,
            position: *position,
            sequence_length: *sequence_length,
            recursion_depth: *recursion_depth,
            max_recursion_depth: *max_recursion_depth,
            projection: *projection,
            identity: spec.identity.clone(),
            input_layout: spec.input_layout.clone(),
            output_layout: spec.output_layout.clone(),
            linear_layout: spec.linear_layout.clone(),
        })
    }

    pub fn output_projection_summary(&self) -> Option<OutputProjectionDiagnosticEventSummary> {
        let DiagnosticEventKind::OutputProjection {
            position,
            sequence_length,
            spec,
        } = &self.event
        else {
            return None;
        };
        Some(OutputProjectionDiagnosticEventSummary {
            correlation_id: self.correlation_id,
            phase: self.phase,
            step: self.step,
            tokens_seen: self.tokens_seen,
            position: *position,
            sequence_length: *sequence_length,
            identity: spec.identity.clone(),
            input_layout: spec.input_layout.clone(),
            output_layout: spec.output_layout.clone(),
            linear_layout: spec.linear_layout.clone(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticsRuntimeArtifact {
    pub policy: DiagnosticsPolicy,
    pub event_file: Option<String>,
    pub events: Vec<DiagnosticEvent>,
    pub emitted_probe_kinds: Vec<DiagnosticProbeKind>,
    pub missing_required_probe_kinds: Vec<DiagnosticProbeKind>,
    pub missing_required_boundary_completions: Vec<DiagnosticBoundary>,
    pub runtime_failure: Option<DiagnosticsRuntimeFailure>,
    pub diagnostics_incomplete: bool,
    pub boundary_memory_deltas: Vec<BoundaryMemoryDelta>,
    pub last_event: Option<DiagnosticEvent>,
    pub last_rule_projection_event: Option<RuleProjectionDiagnosticEventSummary>,
    pub last_output_projection_event: Option<OutputProjectionDiagnosticEventSummary>,
}

impl DiagnosticsRuntimeArtifact {
    pub(crate) fn from_policy(policy: DiagnosticsPolicy) -> Self {
        Self {
            policy,
            event_file: None,
            events: Vec::new(),
            emitted_probe_kinds: Vec::new(),
            missing_required_probe_kinds: Vec::new(),
            missing_required_boundary_completions: Vec::new(),
            runtime_failure: None,
            diagnostics_incomplete: false,
            boundary_memory_deltas: Vec::new(),
            last_event: None,
            last_rule_projection_event: None,
            last_output_projection_event: None,
        }
    }

    pub(crate) fn initialization_failure(
        policy: DiagnosticsPolicy,
        runtime_paths: Option<&DiagnosticsRuntimePaths>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            policy,
            event_file: runtime_paths.map(|paths| paths.event_file.display().to_string()),
            events: Vec::new(),
            emitted_probe_kinds: Vec::new(),
            missing_required_probe_kinds: Vec::new(),
            missing_required_boundary_completions: Vec::new(),
            runtime_failure: Some(DiagnosticsRuntimeFailure {
                kind: DiagnosticsRuntimeFailureKind::Initialization,
                probe_kind: None,
                boundary: None,
                message: message.into(),
            }),
            diagnostics_incomplete: true,
            boundary_memory_deltas: Vec::new(),
            last_event: None,
            last_rule_projection_event: None,
            last_output_projection_event: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticsRuntimeFailureKind {
    Initialization,
    EventPersistence,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticsRuntimeFailure {
    pub kind: DiagnosticsRuntimeFailureKind,
    pub probe_kind: Option<DiagnosticProbeKind>,
    pub boundary: Option<DiagnosticBoundary>,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DiagnosticsRuntimePaths {
    pub event_file: PathBuf,
}

#[derive(Debug)]
struct DiagnosticsRuntimePersistence {
    event_file: File,
    event_file_path: PathBuf,
}

impl DiagnosticsRuntimePersistence {
    fn new(paths: DiagnosticsRuntimePaths) -> Result<Self, FractalError> {
        if let Some(parent) = paths.event_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| FractalError::InvalidState(error.to_string()))?;
        }
        let event_file = File::create(&paths.event_file)
            .map_err(|error| FractalError::InvalidState(error.to_string()))?;
        Ok(Self {
            event_file,
            event_file_path: paths.event_file,
        })
    }

    fn event_file(&self) -> String {
        self.event_file_path.display().to_string()
    }

    fn persist_event(&mut self, event: &DiagnosticEvent) -> Result<(), FractalError> {
        maybe_fail_test_diagnostics_persistence()?;
        let serialized = serde_json::to_string(event)
            .map_err(|error| FractalError::InvalidState(error.to_string()))?;
        writeln!(self.event_file, "{serialized}")
            .map_err(|error| FractalError::InvalidState(error.to_string()))?;
        self.event_file
            .flush()
            .map_err(|error| FractalError::InvalidState(error.to_string()))?;
        self.event_file
            .sync_data()
            .map_err(|error| FractalError::InvalidState(error.to_string()))
    }
}

thread_local! {
    static LAST_DIAGNOSTICS_RUNTIME_ARTIFACT: RefCell<Option<DiagnosticsRuntimeArtifact>> = const {
        RefCell::new(None)
    };
}

#[cfg(test)]
thread_local! {
    static TEST_PERSISTENCE_FAILURE_AFTER_SUCCESSFUL_EVENTS: std::cell::Cell<Option<usize>> = const {
        std::cell::Cell::new(None)
    };
}

pub(crate) fn clear_last_diagnostics_runtime_artifact() {
    LAST_DIAGNOSTICS_RUNTIME_ARTIFACT.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

pub(crate) fn take_last_diagnostics_runtime_artifact() -> Option<DiagnosticsRuntimeArtifact> {
    LAST_DIAGNOSTICS_RUNTIME_ARTIFACT.with(|slot| slot.borrow_mut().take())
}

#[cfg(test)]
pub(crate) fn set_test_diagnostics_persistence_failure_after_successful_events(
    successful_events_before_failure: usize,
) {
    TEST_PERSISTENCE_FAILURE_AFTER_SUCCESSFUL_EVENTS.with(|slot| {
        slot.set(Some(successful_events_before_failure));
    });
}

#[cfg(test)]
pub(crate) fn clear_test_diagnostics_persistence_failure() {
    TEST_PERSISTENCE_FAILURE_AFTER_SUCCESSFUL_EVENTS.with(|slot| {
        slot.set(None);
    });
}

fn record_diagnostics_runtime_artifact(artifact: DiagnosticsRuntimeArtifact) {
    LAST_DIAGNOSTICS_RUNTIME_ARTIFACT.with(|slot| {
        *slot.borrow_mut() = Some(artifact);
    });
}

#[derive(Debug)]
pub struct DiagnosticsRecorder {
    policy: DiagnosticsPolicy,
    identity: DiagnosticIdentity,
    events: Vec<DiagnosticEvent>,
    next_correlation_id: u64,
    emitted_probe_kinds: BTreeSet<DiagnosticProbeKind>,
    missing_required_probe_kinds: BTreeSet<DiagnosticProbeKind>,
    missing_required_boundary_completions: BTreeSet<DiagnosticBoundary>,
    runtime_failure: Option<DiagnosticsRuntimeFailure>,
    persistence: Option<DiagnosticsRuntimePersistence>,
}

impl DiagnosticsRecorder {
    pub fn new(policy: DiagnosticsPolicy, identity: DiagnosticIdentity) -> Self {
        Self {
            policy,
            identity,
            events: Vec::new(),
            next_correlation_id: 1,
            emitted_probe_kinds: BTreeSet::new(),
            missing_required_probe_kinds: BTreeSet::new(),
            missing_required_boundary_completions: BTreeSet::new(),
            runtime_failure: None,
            persistence: None,
        }
    }

    pub(crate) fn new_with_runtime_paths(
        policy: DiagnosticsPolicy,
        identity: DiagnosticIdentity,
        runtime_paths: Option<DiagnosticsRuntimePaths>,
    ) -> Result<Self, FractalError> {
        let persistence = runtime_paths
            .map(DiagnosticsRuntimePersistence::new)
            .transpose()?;
        Self {
            policy,
            identity,
            events: Vec::new(),
            next_correlation_id: 1,
            emitted_probe_kinds: BTreeSet::new(),
            missing_required_probe_kinds: BTreeSet::new(),
            missing_required_boundary_completions: BTreeSet::new(),
            runtime_failure: None,
            persistence,
        }
        .with_recovery_snapshot()
    }

    pub fn policy(&self) -> &DiagnosticsPolicy {
        &self.policy
    }

    pub fn should_emit_step_probe(&self, kind: DiagnosticProbeKind, step: usize) -> bool {
        self.policy
            .probe(kind)
            .is_some_and(|probe| probe.matches_step(step))
    }

    pub fn should_emit_forward_position(&self, step: usize, position: usize) -> bool {
        self.should_emit_position_probe(DiagnosticProbeKind::ForwardPosition, step, position, None)
    }

    pub fn should_emit_rule_projection(
        &self,
        step: usize,
        position: usize,
        sequence_length: usize,
    ) -> bool {
        self.should_emit_position_probe(
            DiagnosticProbeKind::RuleProjection,
            step,
            position,
            Some(sequence_length),
        )
    }

    pub fn should_emit_output_projection(
        &self,
        step: usize,
        position: usize,
        sequence_length: usize,
    ) -> bool {
        self.should_emit_position_probe(
            DiagnosticProbeKind::OutputProjection,
            step,
            position,
            Some(sequence_length),
        )
    }

    pub fn emit_event(
        &mut self,
        phase: RunPhase,
        step: Option<usize>,
        tokens_seen: Option<usize>,
        event: DiagnosticEventKind,
    ) -> Result<(), FractalError> {
        let probe_kind = event.probe_kind();
        let Some(step) = step else {
            return Ok(());
        };
        if !self.should_emit_step_probe(probe_kind, step) {
            return Ok(());
        }
        self.push_event(phase, Some(step), tokens_seen, event)
    }

    pub fn emit_forward_position(
        &mut self,
        phase: RunPhase,
        context: TrainStepDiagnosticContext,
        position: usize,
        sequence_length: usize,
        input_shape: Vec<usize>,
        readout_shape: Vec<usize>,
    ) -> Result<(), FractalError> {
        if !self.should_emit_forward_position(context.step, position) {
            return Ok(());
        }
        self.push_event(
            phase,
            Some(context.step),
            Some(context.tokens_seen),
            DiagnosticEventKind::ForwardPosition {
                position,
                sequence_length,
                input_shape,
                readout_shape,
            },
        )
    }

    fn emit_rule_projection_inner(
        &mut self,
        phase: RunPhase,
        context: RuleProjectionDiagnosticContext,
        projection: RuleProjectionKind,
        spec: RuleProjectionDiagnosticSpec,
    ) -> Result<(), FractalError> {
        if !self.should_emit_rule_projection(
            context.step,
            context.position,
            context.sequence_length,
        ) {
            return Ok(());
        }
        self.push_event(
            phase,
            Some(context.step),
            Some(context.tokens_seen),
            DiagnosticEventKind::RuleProjection {
                position: context.position,
                sequence_length: context.sequence_length,
                recursion_depth: context.recursion_depth,
                max_recursion_depth: context.max_recursion_depth,
                projection,
                spec: Box::new(spec),
            },
        )
    }

    fn emit_output_projection_inner(
        &mut self,
        phase: RunPhase,
        context: OutputProjectionDiagnosticContext,
        spec: OutputProjectionDiagnosticSpec,
    ) -> Result<(), FractalError> {
        if !self.should_emit_output_projection(
            context.step,
            context.position,
            context.sequence_length,
        ) {
            return Ok(());
        }
        self.push_event(
            phase,
            Some(context.step),
            Some(context.tokens_seen),
            DiagnosticEventKind::OutputProjection {
                position: context.position,
                sequence_length: context.sequence_length,
                spec: Box::new(spec),
            },
        )
    }

    pub fn record_cuda_memory_snapshot(
        &mut self,
        phase: RunPhase,
        context: TrainStepDiagnosticContext,
        boundary: DiagnosticBoundary,
        snapshot: Option<CudaMemorySnapshot>,
    ) -> Result<(), FractalError> {
        if !self.should_emit_step_probe(DiagnosticProbeKind::CudaMemorySnapshot, context.step) {
            return Ok(());
        }
        let Some(snapshot) = snapshot else {
            if self.policy.required {
                self.missing_required_probe_kinds
                    .insert(DiagnosticProbeKind::CudaMemorySnapshot);
            }
            self.record_recovery_snapshot();
            return Ok(());
        };
        self.push_event(
            phase,
            Some(context.step),
            Some(context.tokens_seen),
            DiagnosticEventKind::CudaMemorySnapshot {
                boundary,
                used_mib: snapshot.used_mib,
                free_mib: snapshot.free_mib,
                total_mib: snapshot.total_mib,
            },
        )
    }

    pub fn artifact(&self) -> DiagnosticsRuntimeArtifact {
        DiagnosticsRuntimeArtifact {
            policy: self.policy.clone(),
            event_file: self
                .persistence
                .as_ref()
                .map(DiagnosticsRuntimePersistence::event_file),
            events: self.events.clone(),
            emitted_probe_kinds: self.emitted_probe_kinds.iter().copied().collect(),
            missing_required_probe_kinds: self
                .missing_required_probe_kinds
                .iter()
                .copied()
                .collect(),
            missing_required_boundary_completions: self
                .missing_required_boundary_completions
                .iter()
                .copied()
                .collect(),
            runtime_failure: self.runtime_failure.clone(),
            diagnostics_incomplete: !self.missing_required_probe_kinds.is_empty()
                || !self.missing_required_boundary_completions.is_empty()
                || self.runtime_failure.is_some(),
            boundary_memory_deltas: derive_boundary_memory_deltas(&self.events),
            last_event: self.events.last().cloned(),
            last_rule_projection_event: self
                .events
                .iter()
                .rev()
                .find_map(DiagnosticEvent::rule_projection_summary),
            last_output_projection_event: self
                .events
                .iter()
                .rev()
                .find_map(DiagnosticEvent::output_projection_summary),
        }
    }

    fn should_emit_position_probe(
        &self,
        kind: DiagnosticProbeKind,
        step: usize,
        position: usize,
        sequence_length: Option<usize>,
    ) -> bool {
        self.policy.probe(kind).is_some_and(|probe| {
            probe.matches_step(step)
                && (probe.matches_position(position)
                    || sequence_length.is_some_and(|length| position + 1 == length))
        })
    }

    fn push_event(
        &mut self,
        phase: RunPhase,
        step: Option<usize>,
        tokens_seen: Option<usize>,
        event: DiagnosticEventKind,
    ) -> Result<(), FractalError> {
        let probe_kind = event.probe_kind();
        let boundary = event.boundary();
        let record = DiagnosticEvent {
            experiment_run_id: self.identity.experiment_run_id.clone(),
            experiment_logical_name: self.identity.experiment_logical_name.clone(),
            species: self.identity.species.clone(),
            variant_name: self.identity.variant_name.clone(),
            correlation_id: self.next_correlation_id,
            phase,
            step,
            tokens_seen,
            event,
        };
        self.next_correlation_id += 1;
        if let Some(persistence) = self.persistence.as_mut() {
            if let Err(error) = persistence.persist_event(&record) {
                self.record_runtime_failure(DiagnosticsRuntimeFailure {
                    kind: DiagnosticsRuntimeFailureKind::EventPersistence,
                    probe_kind: Some(probe_kind),
                    boundary,
                    message: error.to_string(),
                });
                return Err(error);
            }
        }
        println!("{}", format_diagnostic_event(&record));
        self.emitted_probe_kinds.insert(probe_kind);
        self.track_required_boundary_completion(boundary);
        self.events.push(record);
        self.record_recovery_snapshot();
        Ok(())
    }

    fn track_required_boundary_completion(&mut self, boundary: Option<DiagnosticBoundary>) {
        if !self.policy.required {
            return;
        }
        let Some(boundary) = boundary else {
            return;
        };
        if let Some(required_completion) = boundary.required_completion() {
            self.missing_required_boundary_completions
                .insert(required_completion);
            return;
        }
        self.missing_required_boundary_completions.remove(&boundary);
    }

    fn record_recovery_snapshot(&self) {
        record_diagnostics_runtime_artifact(self.artifact());
    }

    fn record_runtime_failure(&mut self, failure: DiagnosticsRuntimeFailure) {
        if self.runtime_failure.is_none() {
            self.runtime_failure = Some(failure);
        }
        self.record_recovery_snapshot();
    }

    fn with_recovery_snapshot(self) -> Result<Self, FractalError> {
        self.record_recovery_snapshot();
        Ok(self)
    }
}

impl Drop for DiagnosticsRecorder {
    fn drop(&mut self) {
        self.record_recovery_snapshot();
    }
}

impl ProjectionDiagnosticsSink for DiagnosticsRecorder {
    fn emit_rule_projection(
        &mut self,
        phase: RunPhase,
        context: RuleProjectionDiagnosticContext,
        projection: RuleProjectionKind,
        spec: RuleProjectionDiagnosticSpec,
    ) -> Result<(), FractalError> {
        self.emit_rule_projection_inner(phase, context, projection, spec)
    }

    fn emit_output_projection(
        &mut self,
        phase: RunPhase,
        context: OutputProjectionDiagnosticContext,
        spec: OutputProjectionDiagnosticSpec,
    ) -> Result<(), FractalError> {
        self.emit_output_projection_inner(phase, context, spec)
    }
}

#[cfg(test)]
fn maybe_fail_test_diagnostics_persistence() -> Result<(), FractalError> {
    TEST_PERSISTENCE_FAILURE_AFTER_SUCCESSFUL_EVENTS.with(|slot| match slot.get() {
        None => Ok(()),
        Some(0) => Err(FractalError::InvalidState(
            "synthetic diagnostics persistence failure".into(),
        )),
        Some(remaining) => {
            slot.set(Some(remaining - 1));
            Ok(())
        }
    })
}

#[cfg(not(test))]
fn maybe_fail_test_diagnostics_persistence() -> Result<(), FractalError> {
    Ok(())
}

fn format_diagnostic_event(event: &DiagnosticEvent) -> String {
    let mut parts = vec![
        "[diagnostic]".to_owned(),
        format!("species={}", event.species),
        format!("phase={}", format_phase(event.phase)),
        format!("event={}", event.event.name()),
        format!("correlation_id={}", event.correlation_id),
    ];
    if let Some(step) = event.step {
        parts.push(format!("step={step}"));
    }
    if let Some(tokens_seen) = event.tokens_seen {
        parts.push(format!("tokens_seen={tokens_seen}"));
    }
    match &event.event {
        DiagnosticEventKind::TrainStepStart {
            planned_steps,
            batch_token_count,
            input_shape,
            target_shape,
        } => {
            parts.push(format!("planned_steps={planned_steps}"));
            parts.push(format!("batch_token_count={batch_token_count}"));
            parts.push(format!("input_shape={input_shape:?}"));
            parts.push(format!("target_shape={target_shape:?}"));
        }
        DiagnosticEventKind::ForwardStart { input_shape } => {
            parts.push(format!("input_shape={input_shape:?}"));
        }
        DiagnosticEventKind::ForwardPosition {
            position,
            sequence_length,
            input_shape,
            readout_shape,
        } => {
            parts.push(format!("position={position}/{sequence_length}"));
            parts.push(format!("input_shape={input_shape:?}"));
            parts.push(format!("readout_shape={readout_shape:?}"));
        }
        DiagnosticEventKind::RuleProjection {
            position,
            sequence_length,
            recursion_depth,
            max_recursion_depth,
            projection,
            spec,
        } => {
            parts.push(format!("position={position}/{sequence_length}"));
            parts.push(format!(
                "recursion_depth={recursion_depth}/{max_recursion_depth}"
            ));
            parts.push(format!(
                "projection_path={}.{}",
                spec.identity.rule_name, spec.identity.projection_name
            ));
            parts.push(format!("projection={}", projection.as_str()));
            parts.push(format!("input_shape={:?}", spec.input_layout.shape));
            parts.push(format!("output_shape={:?}", spec.output_layout.shape));
            if let Some(linear_layout) = &spec.linear_layout {
                parts.push(format!(
                    "weight_shape={:?}",
                    linear_layout.stored_weight.shape
                ));
                parts.push(format!(
                    "backward_rhs_shape={:?}",
                    linear_layout.backward_input_grad_rhs.shape
                ));
                parts.push(format!(
                    "backward_rhs_transform={:?}",
                    linear_layout.backward_input_grad_rhs.transform
                ));
            }
        }
        DiagnosticEventKind::OutputProjection {
            position,
            sequence_length,
            spec,
        } => {
            parts.push(format!("position={position}/{sequence_length}"));
            parts.push(format!(
                "projection_path={}.{}",
                spec.identity.model_name, spec.identity.projection_name
            ));
            parts.push(format!("input_shape={:?}", spec.input_layout.shape));
            parts.push(format!("output_shape={:?}", spec.output_layout.shape));
            if let Some(linear_layout) = &spec.linear_layout {
                parts.push(format!(
                    "weight_shape={:?}",
                    linear_layout.stored_weight.shape
                ));
                parts.push(format!(
                    "backward_rhs_shape={:?}",
                    linear_layout.backward_input_grad_rhs.shape
                ));
                parts.push(format!(
                    "backward_rhs_transform={:?}",
                    linear_layout.backward_input_grad_rhs.transform
                ));
            }
        }
        DiagnosticEventKind::ForwardComplete {
            logits_shape,
            graph_burden,
        } => {
            parts.push(format!("logits_shape={logits_shape:?}"));
            parts.push(format!(
                "rule_invocations={}",
                graph_burden.rule_invocations
            ));
            parts.push(format!(
                "positions_processed={}",
                graph_burden.positions_processed
            ));
            parts.push(format!(
                "positions_early_exited={}",
                graph_burden.positions_early_exited
            ));
            parts.push(format!(
                "positions_reached_depth_limit={}",
                graph_burden.positions_reached_depth_limit
            ));
        }
        DiagnosticEventKind::LossStart { target_shape } => {
            parts.push(format!("target_shape={target_shape:?}"));
        }
        DiagnosticEventKind::LossComplete { loss_shape } => {
            parts.push(format!("loss_shape={loss_shape:?}"));
        }
        DiagnosticEventKind::BackwardStart | DiagnosticEventKind::BackwardComplete => {}
        DiagnosticEventKind::OptimizerStepStart {
            scheduled_learning_rate,
        } => {
            parts.push(format!(
                "scheduled_learning_rate={scheduled_learning_rate:.8}"
            ));
        }
        DiagnosticEventKind::OptimizerStepComplete {
            scheduled_learning_rate,
            completed_steps,
            train_tokens_seen,
        } => {
            parts.push(format!(
                "scheduled_learning_rate={scheduled_learning_rate:.8}"
            ));
            parts.push(format!("completed_steps={completed_steps}"));
            parts.push(format!("train_tokens_seen={train_tokens_seen}"));
        }
        DiagnosticEventKind::CudaMemorySnapshot {
            boundary,
            used_mib,
            free_mib,
            total_mib,
        } => {
            parts.push(format!("boundary={}", boundary.as_str()));
            parts.push(format!("used_mib={used_mib}"));
            parts.push(format!("free_mib={free_mib}"));
            parts.push(format!("total_mib={total_mib}"));
        }
    }
    parts.join(" ")
}

fn format_phase(phase: RunPhase) -> &'static str {
    match phase {
        RunPhase::Train => "train",
        RunPhase::Stability => "stability",
        RunPhase::Perplexity => "perplexity",
        RunPhase::ArcSpeed => "arc_speed",
    }
}

fn derive_boundary_memory_deltas(events: &[DiagnosticEvent]) -> Vec<BoundaryMemoryDelta> {
    let mut deltas = Vec::new();
    let mut previous_snapshot: Option<(usize, usize, usize)> = None;
    for event in events {
        let DiagnosticEventKind::CudaMemorySnapshot {
            boundary,
            used_mib,
            free_mib,
            total_mib,
        } = event.event
        else {
            continue;
        };
        deltas.push(BoundaryMemoryDelta {
            boundary,
            step: event.step.unwrap_or_default(),
            tokens_seen: event.tokens_seen.unwrap_or_default(),
            correlation_id: event.correlation_id,
            used_mib,
            free_mib,
            total_mib,
            delta_used_mib: previous_snapshot.map(|(used, _, _)| used_mib as i64 - used as i64),
            delta_free_mib: previous_snapshot.map(|(_, free, _)| free_mib as i64 - free as i64),
            delta_total_mib: previous_snapshot.map(|(_, _, total)| total_mib as i64 - total as i64),
        });
        previous_snapshot = Some((used_mib, free_mib, total_mib));
    }
    deltas
}
