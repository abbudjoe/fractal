use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    diagnostics::{ProjectionDiagnosticsSink, RuleProjectionDiagnosticContext, RuleProjectionKind},
    error::FractalError,
    primitives::{gated_sigmoid, one_minus},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
    RunPhase,
};

#[derive(Module, Debug)]
pub struct P1Contractive<B: Backend> {
    pub g_proj: StructuredProjection<B>,
    pub w_h: StructuredProjection<B>,
    pub u: StructuredProjection<B>,
    hidden_dim: usize,
}

impl<B: Backend> P1Contractive<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        let projection = StructuredProjectionConfig::new(hidden_dim, hidden_dim)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Self {
            g_proj: projection.init(device),
            w_h: projection.init(device),
            u: projection.init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for P1Contractive<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.flat()?;
        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let mix = self.w_h.forward(state.clone()) + self.u.forward(x.clone());
        Ok(FractalState::Flat(g.clone() * mix + one_minus(g) * state))
    }

    fn apply_with_diagnostics(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
        mut diagnostics: Option<&mut dyn ProjectionDiagnosticsSink>,
        diagnostic_context: Option<RuleProjectionDiagnosticContext>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.flat()?;

        let gate_input_shape = x.dims().into_iter().collect::<Vec<_>>();
        let gate_projection = self.g_proj.forward(x.clone());
        if let (Some(recorder), Some(rule_context)) = (diagnostics.as_mut(), diagnostic_context) {
            recorder.emit_rule_projection(
                RunPhase::Train,
                rule_context,
                RuleProjectionKind::Gate,
                self.g_proj.diagnostic_spec(
                    self.name(),
                    "g_proj",
                    gate_input_shape,
                    gate_projection.dims().into_iter().collect(),
                ),
            )?;
        }
        let g = gated_sigmoid(gate_projection);

        let state_input_shape = state.dims().into_iter().collect::<Vec<_>>();
        let state_projection = self.w_h.forward(state.clone());
        if let (Some(recorder), Some(rule_context)) = (diagnostics.as_mut(), diagnostic_context) {
            recorder.emit_rule_projection(
                RunPhase::Train,
                rule_context,
                RuleProjectionKind::StateMix,
                self.w_h.diagnostic_spec(
                    self.name(),
                    "w_h",
                    state_input_shape,
                    state_projection.dims().into_iter().collect(),
                ),
            )?;
        }

        let input_mix_input_shape = x.dims().into_iter().collect::<Vec<_>>();
        let input_projection = self.u.forward(x.clone());
        if let (Some(recorder), Some(rule_context)) = (diagnostics.as_mut(), diagnostic_context) {
            recorder.emit_rule_projection(
                RunPhase::Train,
                rule_context,
                RuleProjectionKind::InputMix,
                self.u.diagnostic_spec(
                    self.name(),
                    "u",
                    input_mix_input_shape,
                    input_projection.dims().into_iter().collect(),
                ),
            )?;
        }
        let mix = state_projection + input_projection;
        Ok(FractalState::Flat(g.clone() * mix + one_minus(g) * state))
    }

    fn name(&self) -> &'static str {
        "p1_contractive"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
