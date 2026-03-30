pub mod b2_stable_hierarchical;
pub mod generalized_mobius;
pub mod ifs;
pub mod logistic_chaotic_map;
pub mod p1_contractive;
pub mod p3_hierarchical;

use burn::{
    module::Initializer,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};

const CONTRACTIVE_INIT_BOUND: f64 = 0.1;
pub(crate) fn contractive_linear<B: Backend>(
    d_input: usize,
    d_output: usize,
    bias: bool,
    device: &B::Device,
) -> Linear<B> {
    LinearConfig::new(d_input, d_output)
        .with_bias(bias)
        .with_initializer(Initializer::Uniform {
            min: -CONTRACTIVE_INIT_BOUND,
            max: CONTRACTIVE_INIT_BOUND,
        })
        .init(device)
}

pub(crate) fn contractive_diag_param<B: Backend>(
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> burn::module::Param<Tensor<B, 2>> {
    Initializer::Uniform {
        min: -CONTRACTIVE_INIT_BOUND,
        max: CONTRACTIVE_INIT_BOUND,
    }
    .init([rows, cols], device)
}

pub(crate) fn clamped_contractive<B: Backend>(tensor: Tensor<B, 2>, limit: f64) -> Tensor<B, 2> {
    tensor.tanh().mul_scalar(limit)
}

pub(crate) fn entropy_regularized_router_probs<B: Backend>(
    logits: Tensor<B, 2>,
    num_choices: usize,
    entropy_mix: f64,
) -> Tensor<B, 2> {
    softmax(logits, 1)
        .mul_scalar(1.0 - entropy_mix)
        .add_scalar(entropy_mix / num_choices as f64)
}
