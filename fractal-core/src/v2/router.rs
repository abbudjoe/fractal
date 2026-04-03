use burn::{module::Module, tensor::backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractalRouterHeadShape {
    pub query_dim: usize,
    pub key_dim: usize,
    pub head_count: usize,
    pub beam_width: usize,
    pub top_leaf_reads: usize,
    pub allow_early_stop: bool,
}

pub trait FractalRouterHead<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> FractalRouterHeadShape;
}
