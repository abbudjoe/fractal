use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation::sigmoid, backend::Backend, ElementConversion, Tensor},
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

    pub fn should_exit(&self, readout: Tensor<B, 2>) -> bool {
        let scores = self.scores(readout);
        let [batch] = scores.dims();
        let mean_score = scores
            .sum()
            .div_scalar(batch as f32)
            .into_scalar()
            .elem::<f32>();
        mean_score >= self.threshold
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}
