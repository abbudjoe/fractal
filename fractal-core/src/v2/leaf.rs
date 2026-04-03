use burn::{module::Module, tensor::backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafSummarizerShape {
    pub token_dim: usize,
    pub leaf_size: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub token_cache_key_dim: usize,
    pub token_cache_value_dim: usize,
}

pub trait LeafSummarizer<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> LeafSummarizerShape;
}
