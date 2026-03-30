use burn::{
    module::Module,
    nn::Linear,
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::{contractive_linear, row_l2_norm};

#[derive(Module, Debug)]
pub struct GeneralizedMobius<B: Backend> {
    pub a_proj: Linear<B>,
    pub b_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub d_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> GeneralizedMobius<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            a_proj: contractive_linear(hidden_dim, hidden_dim, true, device),
            b_proj: contractive_linear(hidden_dim, hidden_dim, true, device),
            c_proj: contractive_linear(hidden_dim, hidden_dim, true, device),
            d_proj: contractive_linear(hidden_dim, hidden_dim, true, device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for GeneralizedMobius<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.flat()?;
        let a = self.a_proj.forward(x.clone()).tanh();
        let b = self.b_proj.forward(x.clone()).tanh().mul_scalar(0.5);
        let c = self.c_proj.forward(x.clone()).tanh();
        let d = self
            .d_proj
            .forward(x.clone())
            .tanh()
            .mul_scalar(0.5)
            .add_scalar(1.0);
        let epsilon = row_l2_norm(state.clone())
            .mul_scalar(1e-5)
            .add_scalar(1e-6)
            .repeat(&[1, self.hidden_dim]);

        let numerator = a * state.clone() + b;
        let denominator = c * state + d + epsilon;
        let next = numerator * denominator.recip();

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "generalized_mobius"
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
