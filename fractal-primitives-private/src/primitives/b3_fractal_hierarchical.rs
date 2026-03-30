use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::{complex_square, gated_sigmoid, gated_sigmoid_clamped},
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

#[derive(Module, Debug)]
pub struct B3FractalHierarchical<B: Backend> {
    pub g_proj: Linear<B>,
    pub c_proj: Linear<B>,
    pub gamma_proj: Linear<B>,
    pub compressor: Linear<B>,
    hidden_dim: usize,
    levels: usize,
}

impl<B: Backend> B3FractalHierarchical<B> {
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

impl<B: Backend> FractalRule<B> for B3FractalHierarchical<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.hierarchical_complex_checked(self.levels, self.hidden_dim)?;
        let [batch, levels, width] = state.dims();

        let g = gated_sigmoid_clamped(self.g_proj.forward(x.clone()));
        let c = self.c_proj.forward(x.clone());
        let gamma = gated_sigmoid(self.gamma_proj.forward(x.clone()));

        let mut next_levels: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let prev = state.clone().narrow(1, level, 1).reshape([batch, width]);
            let mut next = g.clone() * complex_square(prev) + c.clone();
            if level > 0 {
                next =
                    next + gamma.clone() * self.compressor.forward(next_levels[level - 1].clone());
            }
            next_levels.push(next);
        }

        Ok(FractalState::HierarchicalComplex(Tensor::stack::<3>(
            next_levels,
            1,
        )))
    }

    fn name(&self) -> &'static str {
        "b3_fractal_hierarchical"
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
