use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::{
    error::FractalError,
    primitives::{gated_sigmoid, one_minus},
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

#[derive(Module, Debug)]
pub struct B2StableHierarchical<B: Backend> {
    pub g_proj: Linear<B>,
    pub u: Linear<B>,
    pub w_h: Linear<B>,
    pub gamma_proj: Linear<B>,
    pub compressor: Linear<B>,
    hidden_dim: usize,
    levels: usize,
}

impl<B: Backend> B2StableHierarchical<B> {
    pub fn new(hidden_dim: usize, levels: usize, device: &B::Device) -> Self {
        Self {
            g_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            u: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            w_h: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            gamma_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            compressor: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
            levels,
        }
    }
}

impl<B: Backend> FractalRule<B> for B2StableHierarchical<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.hierarchical_checked(self.levels, self.hidden_dim)?;
        let [batch, levels, width] = state.dims();

        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let u_t = self.u.forward(x.clone());
        let gamma = gated_sigmoid(self.gamma_proj.forward(x.clone()));

        let mut next_levels: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let prev = state.clone().narrow(1, level, 1).reshape([batch, width]);
            let base = g.clone() * (self.w_h.forward(prev.clone()) + u_t.clone())
                + one_minus(g.clone()) * prev;
            let next = if level == 0 {
                base
            } else {
                base + gamma.clone() * self.compressor.forward(next_levels[level - 1].clone())
            };
            next_levels.push(next);
        }

        Ok(FractalState::Hierarchical(Tensor::stack::<3>(
            next_levels,
            1,
        )))
    }

    fn name(&self) -> &'static str {
        "b2_stable_hierarchical"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Hierarchical {
            levels: self.levels,
        }
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
