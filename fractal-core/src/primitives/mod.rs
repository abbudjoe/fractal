use burn::tensor::{activation::sigmoid, backend::Backend, Tensor};

pub fn one_minus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.mul_scalar(-1.0).add_scalar(1.0)
}

pub fn gated_sigmoid<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    sigmoid(tensor)
}

pub fn gated_sigmoid_clamped<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    sigmoid(tensor).clamp(0.0, 0.9)
}

pub fn complex_square<B: Backend>(state: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, width] = state.dims();
    let half = width / 2;
    let real = state.clone().narrow(1, 0, half);
    let imag = state.narrow(1, half, half);

    let next_real = real.clone().square() - imag.clone().square();
    let next_imag = (real * imag).mul_scalar(2.0);

    Tensor::cat(vec![next_real, next_imag], 1).reshape([batch, width])
}
