use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::{complex_square, gated_sigmoid, one_minus},
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::norm_based_residual_alpha;

#[derive(Module, Debug)]
pub struct B1FractalGated<B: Backend> {
    pub g_proj: Linear<B>,
    pub c_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> B1FractalGated<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        let complex_dim = hidden_dim * 2;
        Self {
            g_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            c_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for B1FractalGated<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.complex()?;
        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let c_t = self.c_proj.forward(x.clone());
        let main_update = g.clone() * (complex_square(previous_state.clone()) + c_t)
            + one_minus(g) * previous_state.clone();
        let alpha = norm_based_residual_alpha(previous_state.clone(), self.hidden_dim * 2);
        let next = alpha.clone() * main_update + one_minus(alpha) * previous_state;

        Ok(FractalState::Complex(next))
    }

    fn name(&self) -> &'static str {
        "b1_fractal_gated"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Complex
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
