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

use crate::primitives::{clamp_symmetric_by_row, depth_fraction, row_l2_norm};

#[derive(Module, Debug)]
pub struct JuliaRecursiveEscape<B: Backend> {
    c_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> JuliaRecursiveEscape<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            c_proj: LinearConfig::new(hidden_dim, hidden_dim * 2).init(device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for JuliaRecursiveEscape<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.complex()?;
        let c_t = self.c_proj.forward(x.clone());
        let next = complex_square(previous_state) + c_t;
        let norm = row_l2_norm(next.clone());
        let escape_radius = gated_sigmoid(norm)
            .mul_scalar(1.0 - 0.2 * depth_fraction(context))
            .add_scalar(2.0)
            .repeat(&[1, self.hidden_dim * 2]);
        let bounded = clamp_symmetric_by_row(next, escape_radius);

        Ok(FractalState::Complex(bounded))
    }

    fn name(&self) -> &'static str {
        "julia_recursive_escape"
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
