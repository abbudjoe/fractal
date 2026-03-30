use burn::{
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

use crate::primitives::{contractive_diag_param, entropy_regularized_router_probs};

const NUM_IFS_MAPS: usize = 4;
const IFS_SPECTRAL_RADIUS_LIMIT: f64 = 0.95;
const IFS_ROUTER_ENTROPY_MIX: f64 = 0.05;

#[derive(Module, Debug)]
pub struct Ifs<B: Backend> {
    pub router: Linear<B>,
    pub a_diag: Param<Tensor<B, 2>>,
    pub b_bias: Param<Tensor<B, 2>>,
    hidden_dim: usize,
}

impl<B: Backend> Ifs<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            router: LinearConfig::new(hidden_dim, NUM_IFS_MAPS).init(device),
            a_diag: contractive_diag_param(NUM_IFS_MAPS, hidden_dim, device),
            b_bias: contractive_diag_param(NUM_IFS_MAPS, hidden_dim, device),
            hidden_dim,
        }
    }
}

impl<B: Backend> FractalRule<B> for Ifs<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError> {
        let state = state.flat()?;
        let [batch, _] = state.dims();
        let probs = entropy_regularized_router_probs(
            self.router.forward(x.clone()),
            NUM_IFS_MAPS,
            IFS_ROUTER_ENTROPY_MIX,
        );
        let a_diag = self
            .a_diag
            .val()
            .clamp(-IFS_SPECTRAL_RADIUS_LIMIT, IFS_SPECTRAL_RADIUS_LIMIT);
        let b_bias = self.b_bias.val();
        let mut next = Tensor::<B, 2>::zeros([batch, self.hidden_dim], &state.device());

        for map_index in 0..NUM_IFS_MAPS {
            let a = a_diag
                .clone()
                .narrow(0, map_index, 1)
                .reshape([1, self.hidden_dim])
                .repeat(&[batch, 1]);
            let b = b_bias
                .clone()
                .narrow(0, map_index, 1)
                .reshape([1, self.hidden_dim])
                .repeat(&[batch, 1]);
            let weight = probs
                .clone()
                .narrow(1, map_index, 1)
                .repeat(&[1, self.hidden_dim]);
            let candidate = a * state.clone() + b;
            next = next + weight * candidate;
        }

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "ifs"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
