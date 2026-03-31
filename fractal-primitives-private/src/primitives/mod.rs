pub mod b1_fractal_gated;
pub mod b2_stable_hierarchical;
pub mod b3_fractal_hierarchical;
pub mod b4_universal;
pub mod generalized_mobius;
pub mod ifs;
pub mod logistic_chaotic_map;
pub mod p1_contractive;
pub mod p1_fractal_hybrid;
pub mod p2_mandelbrot;
pub mod p3_hierarchical;

use burn::{
    module::Initializer,
    nn::{Linear, LinearConfig},
    tensor::{activation::softmax, backend::Backend, Tensor},
};
use fractal_core::{primitives::gated_sigmoid, rule_trait::ApplyContext};

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

pub(crate) fn depth_fraction(context: ApplyContext) -> f64 {
    context.depth as f64 / context.max_depth.max(1) as f64
}

pub(crate) fn norm_based_residual_alpha<B: Backend>(
    tensor: Tensor<B, 2>,
    width: usize,
) -> Tensor<B, 2> {
    gated_sigmoid(row_l2_norm(tensor))
        .mul_scalar(0.08)
        .add_scalar(0.02)
        .repeat(&[1, width])
}

pub(crate) fn clamp_symmetric_by_row<B: Backend>(
    tensor: Tensor<B, 2>,
    clamp: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let upper_mask = tensor.clone().greater(clamp.clone());
    let clamped = tensor.mask_where(upper_mask, clamp.clone());
    let lower = clamp.mul_scalar(-1.0);
    let lower_mask = clamped.clone().lower(lower.clone());
    clamped.mask_where(lower_mask, lower)
}
