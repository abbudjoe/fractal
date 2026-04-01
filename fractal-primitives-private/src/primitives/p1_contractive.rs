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

#[derive(Module, Debug)]
pub struct P1Contractive<B: Backend> {
    pub g_proj: Linear<B>,
    pub w_h: Linear<B>,
    pub u: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> P1Contractive<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            g_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            w_h: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            u: LinearConfig::new(hidden_dim, hidden_dim).init(device),
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
