use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::{complex_square, gated_sigmoid},
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::{clamp_max_by_row, row_l2_norm};

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
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.complex()?;
        let clamp_val = gated_sigmoid(row_l2_norm(previous_state.clone()))
            .mul_scalar(-0.225)
            .add_scalar(0.9)
            .repeat(&[1, self.hidden_dim * 2]);
        let g = clamp_max_by_row(gated_sigmoid(self.g_proj.forward(x.clone())), clamp_val);
        let c_t = self.c_proj.forward(x.clone());
        let next = g * complex_square(previous_state) + c_t;

        Ok(FractalState::Complex(next))
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
