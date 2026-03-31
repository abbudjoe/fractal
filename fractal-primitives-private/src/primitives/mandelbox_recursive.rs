use burn::{
    module::Module,
    nn::Linear,
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::{clamp_symmetric_by_row, depth_fraction};

const BASE_ESCAPE_RADIUS: f64 = 1.8;
const MIN_ESCAPE_RADIUS: f64 = 0.75;
const INPUT_DRIVE_SCALE: f64 = 0.1;
const RESIDUAL_MIX: f64 = 0.3;
const INNER_ITERATION_CAP: usize = 3;

#[derive(Module, Debug)]
pub struct MandelboxRecursiveDynEscapeRadius<B: Backend> {
    pub drive_proj: Linear<B>,
    hidden_dim: usize,
}

impl<B: Backend> MandelboxRecursiveDynEscapeRadius<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            drive_proj: burn::nn::LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
        }
    }

    fn escape_radius(
        &self,
        context: ApplyContext,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let depth_factor = depth_fraction(context);
        let radius = (BASE_ESCAPE_RADIUS - 0.45 * depth_factor).max(MIN_ESCAPE_RADIUS);
        Tensor::<B, 2>::ones([batch_size, self.hidden_dim], device).mul_scalar(radius)
    }

    fn mandelbox_step(
        &self,
        state: Tensor<B, 2>,
        escape_radius: Tensor<B, 2>,
        drive: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let folded = clamp_symmetric_by_row(state.clone(), escape_radius) * 2.0 - state;
        folded * 2.0 + drive
    }
}

impl<B: Backend> FractalRule<B> for MandelboxRecursiveDynEscapeRadius<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let [batch_size, _] = previous_state.dims();
        let drive = self
            .drive_proj
            .forward(x.clone())
            .tanh()
            .mul_scalar(INPUT_DRIVE_SCALE);
        let escape_radius = self.escape_radius(context, batch_size, &previous_state.device());
        let iterations = context.depth.clamp(1, INNER_ITERATION_CAP);

        let mut current = previous_state.clone() + drive.clone();
        for _ in 0..iterations {
            current = self.mandelbox_step(current, escape_radius.clone(), drive.clone());
        }

        let next = previous_state.clone().mul_scalar(1.0 - RESIDUAL_MIX)
            + current.mul_scalar(RESIDUAL_MIX);

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "mandelbox_recursive_dyn-escape-radius"
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
