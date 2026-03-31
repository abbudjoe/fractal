use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use fractal_core::{
    error::FractalError,
    primitives::{gated_sigmoid, one_minus},
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

use crate::primitives::{clamp_symmetric_by_row, row_l2_norm};

#[derive(Module, Debug)]
pub struct P1FractalHybrid<B: Backend> {
    pub g_proj: Linear<B>,
    pub w_h: Linear<B>,
    pub u: Linear<B>,
    hidden_dim: usize,
    with_dynamic_lever: bool,
}

impl<B: Backend> P1FractalHybrid<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self::new_with_dynamic_lever(hidden_dim, false, device)
    }

    pub fn new_with_dynamic_lever(
        hidden_dim: usize,
        with_dynamic_lever: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            g_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            w_h: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            u: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
            with_dynamic_lever,
        }
    }

    pub fn with_dynamic_lever(&self) -> bool {
        self.with_dynamic_lever
    }
}

impl<B: Backend> FractalRule<B> for P1FractalHybrid<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let clamp = gated_sigmoid(row_l2_norm(previous_state.clone()))
            .mul_scalar(-0.225)
            .add_scalar(0.75)
            .repeat(&[1, self.hidden_dim]);
        let squared =
            clamp_symmetric_by_row(previous_state.clone() * previous_state.clone(), clamp);
        let main_update =
            self.w_h.forward(previous_state.clone()) + self.u.forward(x.clone()) + squared;
        Ok(FractalState::Flat(
            g.clone() * main_update + one_minus(g) * previous_state,
        ))
    }

    fn name(&self) -> &'static str {
        if self.with_dynamic_lever {
            "p1_fractal_hybrid_dyn-state-norm"
        } else {
            "p1_fractal_hybrid"
        }
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
