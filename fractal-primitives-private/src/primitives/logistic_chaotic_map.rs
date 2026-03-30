use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::{gated_sigmoid, one_minus},
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::{contractive_linear, row_l2_norm};

const LOGISTIC_MIN_R: f64 = 3.6;
const LOGISTIC_MAX_R: f64 = 3.95;

#[derive(Module, Debug)]
pub struct LogisticChaoticMap<B: Backend> {
    pub r_proj: Linear<B>,
    pub g_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> LogisticChaoticMap<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            r_proj: contractive_linear(hidden_dim, hidden_dim, true, device),
            g_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for LogisticChaoticMap<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let bounded_state = gated_sigmoid(previous_state.clone());
        let r_t = gated_sigmoid(self.r_proj.forward(x.clone()))
            .mul_scalar(LOGISTIC_MAX_R - LOGISTIC_MIN_R)
            .add_scalar(LOGISTIC_MIN_R)
            .clamp(LOGISTIC_MIN_R, LOGISTIC_MAX_R);
        let g_t = gated_sigmoid(self.g_proj.forward(x.clone()));
        let next = r_t * bounded_state.clone() * one_minus(bounded_state) + g_t * x.clone();
        let alpha = gated_sigmoid(row_l2_norm(previous_state.clone()))
            .mul_scalar(0.08)
            .add_scalar(0.02)
            .repeat(&[1, self.hidden_dim]);
        let next = alpha.clone() * next + one_minus(alpha) * previous_state;

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "logistic_chaotic_map"
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
