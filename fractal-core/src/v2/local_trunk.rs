use burn::{module::Module, tensor::backend::Backend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalTrunkShape {
    pub token_dim: usize,
    pub root_count: usize,
    pub root_state_dim: usize,
    pub root_readout_dim: usize,
    pub leaf_size: usize,
}

pub trait LocalTrunk<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> LocalTrunkShape;
}
