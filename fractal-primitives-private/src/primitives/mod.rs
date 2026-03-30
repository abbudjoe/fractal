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

pub(crate) fn router_probs<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2> {
    softmax(logits, 1)
}

pub(crate) fn row_l2_norm<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    (tensor.clone() * tensor)
        .sum_dim(1)
        .add_scalar(1e-12)
        .sqrt()
}
