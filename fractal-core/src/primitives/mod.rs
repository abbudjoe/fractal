use burn::tensor::{activation::sigmoid, backend::Backend, Tensor};

pub mod b1_fractal_gated;
pub mod b2_stable_hierarchical;
pub mod b3_fractal_hierarchical;
pub mod b4_universal;
pub mod p1_contractive;
pub mod p2_mandelbrot;
pub mod p3_hierarchical;

pub(crate) fn one_minus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.mul_scalar(-1.0).add_scalar(1.0)
}

pub(crate) fn gated_sigmoid<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    sigmoid(tensor)
}

pub(crate) fn gated_sigmoid_clamped<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    sigmoid(tensor).clamp(0.0, 0.9)
}

pub(crate) fn complex_square<B: Backend>(state: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, width] = state.dims();
    let half = width / 2;
    let real = state.clone().narrow(1, 0, half);
    let imag = state.narrow(1, half, half);

    let next_real = real.clone().square() - imag.clone().square();
    let next_imag = (real * imag).mul_scalar(2.0);

    Tensor::cat(vec![next_real, next_imag], 1).reshape([batch, width])
}
