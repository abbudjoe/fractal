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
        let state = state.complex()?;
        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let c = self.c_proj.forward(x.clone());
        let base = complex_square(state.clone()) + c;
        Ok(FractalState::Complex(
            g.clone() * base + one_minus(g) * state,
        ))
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
