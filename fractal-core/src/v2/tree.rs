use burn::{module::Module, tensor::backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeMergeCellShape {
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub scale_embedding_dim: usize,
}

pub trait TreeMergeCell<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> TreeMergeCellShape;
}
