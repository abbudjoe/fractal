use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::{
    error::FractalError,
    primitives::{complex_square, gated_sigmoid_clamped},
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

#[derive(Module, Debug)]
pub struct P2Mandelbrot<B: Backend> {
    pub g_proj: Linear<B>,
    pub c_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> P2Mandelbrot<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        let complex_dim = hidden_dim * 2;
        Self {
            g_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            c_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for P2Mandelbrot<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.complex()?;
        let g = gated_sigmoid_clamped(self.g_proj.forward(x.clone()));
        let c = self.c_proj.forward(x.clone());
        Ok(FractalState::Complex(g * complex_square(state) + c))
    }

    fn name(&self) -> &'static str {
        "p2_mandelbrot"
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
