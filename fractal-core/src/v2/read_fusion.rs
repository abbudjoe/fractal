use burn::{module::Module, tensor::backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadFusionShape {
    pub root_count: usize,
    pub root_readout_dim: usize,
    pub retrieved_value_dim: usize,
    pub fused_readout_dim: usize,
}

pub trait ReadFusion<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> ReadFusionShape;
}
