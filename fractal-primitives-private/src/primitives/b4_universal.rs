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
pub struct B4Universal<B: Backend> {
    pub g_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub gamma_proj: Linear<B>,
    pub compressor: Linear<B>,
    hidden_dim: usize,
    levels: usize,
}

impl<B: Backend> B4Universal<B> {
    pub fn new(hidden_dim: usize, levels: usize, device: &B::Device) -> Self {
        let complex_dim = hidden_dim * 2;
        Self {
            g_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            c_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            gamma_proj: LinearConfig::new(hidden_dim, complex_dim).init(device),
            compressor: LinearConfig::new(complex_dim, complex_dim).init(device),
            hidden_dim,
            levels,
        }
    }
}

impl<B: Backend> FractalRule<B> for B4Universal<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.hierarchical_complex_checked(self.levels, self.hidden_dim)?;
        let [batch, levels, width] = state.dims();
        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let c_t = self.c_proj.forward(x.clone());
        let gamma = gated_sigmoid(self.gamma_proj.forward(x.clone()));

        let mut next_levels: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let previous_state = state.clone().narrow(1, level, 1).reshape([batch, width]);
            let mut base = g.clone() * (complex_square(previous_state.clone()) + c_t.clone())
                + one_minus(g.clone()) * previous_state.clone();
            if level > 0 {
                base =
                    base + gamma.clone() * self.compressor.forward(next_levels[level - 1].clone());
            }
            let alpha = norm_based_residual_alpha(previous_state.clone(), width);
            let next = alpha.clone() * base + one_minus(alpha) * previous_state;
            next_levels.push(next);
        }

        Ok(FractalState::HierarchicalComplex(Tensor::stack::<3>(
            next_levels,
            1,
        )))
    }

    fn name(&self) -> &'static str {
        "b4_universal"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::HierarchicalComplex {
            levels: self.levels,
        }
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
