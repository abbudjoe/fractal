use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{error::FractalError, lifecycle::RunPhase};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticProbeKind {
    TrainStep,
    ForwardBoundary,
    ForwardPosition,
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
            (DiagnosticProbeKind::ForwardPosition, Some(0)) => Err(FractalError::InvalidConfig(
                "forward_position position_interval must be greater than zero".into(),
            )),
            (DiagnosticProbeKind::ForwardPosition, None) => Err(FractalError::InvalidConfig(
                "forward_position diagnostics require position_interval".into(),
            )),
            (DiagnosticProbeKind::ForwardPosition, Some(_)) => Ok(()),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainStepDiagnosticContext {
    pub step: usize,
    pub tokens_seen: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaMemorySnapshot {
    pub used_mib: usize,
    pub free_mib: usize,
    pub total_mib: usize,
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
    ForwardComplete {
        logits_shape: Vec<usize>,
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
            Self::ForwardPosition { .. } => None,
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
    pub phase: RunPhase,
    pub step: Option<usize>,
    pub tokens_seen: Option<usize>,
    pub event: DiagnosticEventKind,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DiagnosticsRuntimeArtifact {
    pub policy: DiagnosticsPolicy,
    pub events: Vec<DiagnosticEvent>,
    pub emitted_probe_kinds: Vec<DiagnosticProbeKind>,
    pub missing_required_probe_kinds: Vec<DiagnosticProbeKind>,
    pub diagnostics_incomplete: bool,
    pub last_event: Option<DiagnosticEvent>,
}

#[derive(Clone, Debug)]
pub struct DiagnosticsRecorder {
    policy: DiagnosticsPolicy,
    identity: DiagnosticIdentity,
    events: Vec<DiagnosticEvent>,
    emitted_probe_kinds: BTreeSet<DiagnosticProbeKind>,
    missing_required_probe_kinds: BTreeSet<DiagnosticProbeKind>,
}

impl DiagnosticsRecorder {
    pub fn new(policy: DiagnosticsPolicy, identity: DiagnosticIdentity) -> Self {
        Self {
            policy,
            identity,
            events: Vec::new(),
            emitted_probe_kinds: BTreeSet::new(),
            missing_required_probe_kinds: BTreeSet::new(),
        }
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
        self.policy
            .probe(DiagnosticProbeKind::ForwardPosition)
            .is_some_and(|probe| probe.matches_step(step) && probe.matches_position(position))
    }

    pub fn emit_event(
        &mut self,
        phase: RunPhase,
        step: Option<usize>,
        tokens_seen: Option<usize>,
        event: DiagnosticEventKind,
    ) {
        let probe_kind = event.probe_kind();
        let Some(step) = step else {
            return;
        };
        if !self.should_emit_step_probe(probe_kind, step) {
            return;
        }
        self.push_event(phase, Some(step), tokens_seen, event);
    }

    pub fn emit_forward_position(
        &mut self,
        phase: RunPhase,
        context: TrainStepDiagnosticContext,
        position: usize,
        sequence_length: usize,
        input_shape: Vec<usize>,
        readout_shape: Vec<usize>,
    ) {
        if !self.should_emit_forward_position(context.step, position) {
            return;
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
        );
    }

    pub fn record_cuda_memory_snapshot(
        &mut self,
        phase: RunPhase,
        context: TrainStepDiagnosticContext,
        boundary: DiagnosticBoundary,
        snapshot: Option<CudaMemorySnapshot>,
    ) {
        if !self.should_emit_step_probe(DiagnosticProbeKind::CudaMemorySnapshot, context.step) {
            return;
        }
        let Some(snapshot) = snapshot else {
            if self.policy.required {
                self.missing_required_probe_kinds
                    .insert(DiagnosticProbeKind::CudaMemorySnapshot);
            }
            return;
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
        );
    }

    pub fn artifact(&self) -> DiagnosticsRuntimeArtifact {
        DiagnosticsRuntimeArtifact {
            policy: self.policy.clone(),
            events: self.events.clone(),
            emitted_probe_kinds: self.emitted_probe_kinds.iter().copied().collect(),
            missing_required_probe_kinds: self
                .missing_required_probe_kinds
                .iter()
                .copied()
                .collect(),
            diagnostics_incomplete: !self.missing_required_probe_kinds.is_empty(),
            last_event: self.events.last().cloned(),
        }
    }

    fn push_event(
        &mut self,
        phase: RunPhase,
        step: Option<usize>,
        tokens_seen: Option<usize>,
        event: DiagnosticEventKind,
    ) {
        let probe_kind = event.probe_kind();
        let record = DiagnosticEvent {
            experiment_run_id: self.identity.experiment_run_id.clone(),
            experiment_logical_name: self.identity.experiment_logical_name.clone(),
            species: self.identity.species.clone(),
            variant_name: self.identity.variant_name.clone(),
            phase,
            step,
            tokens_seen,
            event,
        };
        println!("{}", format_diagnostic_event(&record));
        self.emitted_probe_kinds.insert(probe_kind);
        self.events.push(record);
    }
}

fn format_diagnostic_event(event: &DiagnosticEvent) -> String {
    let mut parts = vec![
        "[diagnostic]".to_owned(),
        format!("species={}", event.species),
        format!("phase={}", format_phase(event.phase)),
        format!("event={}", event.event.name()),
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
        DiagnosticEventKind::ForwardComplete { logits_shape } => {
            parts.push(format!("logits_shape={logits_shape:?}"));
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
