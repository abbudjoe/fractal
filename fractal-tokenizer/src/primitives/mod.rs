mod b1_fractal_gated;
mod b3_fractal_hierarchical;
mod b4_universal;
mod p1_fractal_hybrid;
mod p2_mandelbrot;

use burn::tensor::{backend::Backend, Tensor};

pub use b1_fractal_gated::B1FractalGated;
pub use b3_fractal_hierarchical::B3FractalHierarchical;
pub use b4_universal::B4Universal;
pub use p1_fractal_hybrid::P1FractalHybrid;
pub use p2_mandelbrot::P2Mandelbrot;

pub(crate) fn row_l2_norm<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    (tensor.clone() * tensor)
        .sum_dim(1)
        .add_scalar(1e-12)
        .sqrt()
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
