use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, Bool, Tensor},
};

#[derive(Module, Debug)]
pub struct EarlyExitRouter<B: Backend> {
    pub projection: Linear<B>,
    threshold: f32,
}

impl<B: Backend> EarlyExitRouter<B> {
    pub fn new(readout_width: usize, threshold: f32, device: &B::Device) -> Self {
        Self {
            projection: LinearConfig::new(readout_width, 1).init(device),
            threshold,
        }
    }

    pub fn scores(&self, readout: Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch, _] = readout.dims();
        sigmoid(self.projection.forward(readout)).reshape([batch])
    }

    pub fn exit_mask(&self, readout: Tensor<B, 2>) -> Tensor<B, 1, Bool> {
        self.scores(readout).greater_equal_elem(self.threshold)
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}
