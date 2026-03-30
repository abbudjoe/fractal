use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::gated_sigmoid,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

#[derive(Module, Debug)]
pub struct P3Hierarchical<B: Backend> {
    pub u: Linear<B>,
    pub alpha_proj: Linear<B>,
    pub beta_proj: Linear<B>,
    pub gamma_proj: Linear<B>,
    pub compressor: Linear<B>,
    hidden_dim: usize,
    levels: usize,
}

impl<B: Backend> P3Hierarchical<B> {
    pub fn new(hidden_dim: usize, levels: usize, device: &B::Device) -> Self {
        Self {
            u: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            alpha_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            beta_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            gamma_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            compressor: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
            levels,
        }
    }
}

impl<B: Backend> FractalRule<B> for P3Hierarchical<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.hierarchical_checked(self.levels, self.hidden_dim)?;
        let [batch, levels, width] = state.dims();

        let u_t = self.u.forward(x.clone());
        let alpha = gated_sigmoid(self.alpha_proj.forward(x.clone()));
        let beta = gated_sigmoid(self.beta_proj.forward(x.clone()));
        let gamma = gated_sigmoid(self.gamma_proj.forward(x.clone()));

        let mut next_levels: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let prev = state.clone().narrow(1, level, 1).reshape([batch, width]);
            let mut next = alpha.clone() * prev + beta.clone() * u_t.clone();
            if level > 0 {
                let lower = next_levels[level - 1].clone();
                next = next + gamma.clone() * self.compressor.forward(lower);
            }
            next_levels.push(next);
        }

        Ok(FractalState::Hierarchical(Tensor::stack::<3>(
            next_levels,
            1,
        )))
    }

    fn name(&self) -> &'static str {
        "p3_hierarchical"
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
