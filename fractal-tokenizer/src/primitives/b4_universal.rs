use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use fractal_core::{
    error::FractalError,
    primitives::{complex_square, gated_sigmoid, one_minus},
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

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
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.hierarchical_complex_checked(self.levels, self.hidden_dim)?;
        let [batch, levels, width] = state.dims();

        let g = gated_sigmoid(self.g_proj.forward(x.clone()));
        let c = self.c_proj.forward(x.clone());
        let gamma = gated_sigmoid(self.gamma_proj.forward(x.clone()));

        let mut next_levels: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for level in 0..levels {
            let prev = state.clone().narrow(1, level, 1).reshape([batch, width]);
            let base = g.clone() * (complex_square(prev.clone()) + c.clone())
                + one_minus(g.clone()) * prev;
            let next = if level == 0 {
                base
            } else {
                base + gamma.clone() * self.compressor.forward(next_levels[level - 1].clone())
            };
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
