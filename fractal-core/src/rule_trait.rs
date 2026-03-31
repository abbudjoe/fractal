// Naming Convention for all primitives and variants:
// [base]_[lever-description]_v[version]
// Examples: p1_fractal_hybrid_v1, b1_fractal_gated_dyn-residual-norm_v1

use burn::tensor::{backend::Backend, Tensor};

use crate::{
    error::FractalError,
    state::{FractalState, StateLayout},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyContext {
    pub depth: usize,
    pub max_depth: usize,
}

pub trait FractalRule<B: Backend>: Send + 'static {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError>;

    fn name(&self) -> &'static str;
    fn hidden_dim(&self) -> usize;
    fn state_layout(&self) -> StateLayout;
    fn clone_box(&self) -> Box<dyn FractalRule<B>>;
}

impl<B: Backend> Clone for Box<dyn FractalRule<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
