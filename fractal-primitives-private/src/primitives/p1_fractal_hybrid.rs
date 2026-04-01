use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use fractal_core::{
    error::FractalError,
    primitives::{gated_sigmoid, one_minus},
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

use crate::primitives::{clamp_max_by_row, clamp_symmetric_by_row, depth_fraction, row_l2_norm};

#[derive(Module, Debug)]
pub(crate) struct P1FractalHybridCore<B: Backend> {
    pub(crate) g_proj: Linear<B>,
    pub(crate) w_h: Linear<B>,
    pub(crate) u: Linear<B>,
    pub(crate) hidden_dim: usize,
}

impl<B: Backend> P1FractalHybridCore<B> {
    fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            g_proj: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            w_h: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            u: LinearConfig::new(hidden_dim, hidden_dim).init(device),
            hidden_dim,
        }
    }

    fn clamp_val(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        gated_sigmoid(row_l2_norm(state))
            .mul_scalar(-0.225)
            .add_scalar(0.75)
            .repeat(&[1, self.hidden_dim])
    }

    fn squared_update(&self, state: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let clamp_val = self.clamp_val(state.clone());
        let squared = clamp_symmetric_by_row(state.clone() * state.clone(), clamp_val);
        self.w_h.forward(state) + self.u.forward(x) + squared
    }

    fn gated_update(
        &self,
        previous_state: Tensor<B, 2>,
        x: Tensor<B, 2>,
        gate: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let main_update = self.squared_update(previous_state.clone(), x);
        gate.clone() * main_update + one_minus(gate) * previous_state
    }

    fn contractive_inner(&self, previous_state: Tensor<B, 2>, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let inner_gate = gated_sigmoid(self.g_proj.forward(x.clone()));
        let mix = self.w_h.forward(previous_state.clone()) + self.u.forward(x);
        inner_gate.clone() * mix + one_minus(inner_gate) * previous_state
    }
}

#[derive(Module, Debug)]
pub struct P1FractalHybrid<B: Backend> {
    pub(crate) core: P1FractalHybridCore<B>,
}

impl<B: Backend> P1FractalHybrid<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            core: P1FractalHybridCore::new(hidden_dim, device),
        }
    }
}

impl<B: Backend> FractalRule<B> for P1FractalHybrid<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let g = gated_sigmoid(self.core.g_proj.forward(x.clone()));
        let next = self.core.gated_update(previous_state, x.clone(), g);

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "p1_fractal_hybrid"
    }

    fn hidden_dim(&self) -> usize {
        self.core.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}

#[derive(Module, Debug)]
pub struct P1FractalHybridComposite<B: Backend> {
    pub(crate) core: P1FractalHybridCore<B>,
}

impl<B: Backend> P1FractalHybridComposite<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            core: P1FractalHybridCore::new(hidden_dim, device),
        }
    }

    fn composite_depth(context: ApplyContext) -> usize {
        context.max_depth.max(2) / 2
    }
}

impl<B: Backend> FractalRule<B> for P1FractalHybridComposite<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let base_state = if context.depth == Self::composite_depth(context) {
            self.core
                .contractive_inner(previous_state.clone(), x.clone())
        } else {
            previous_state.clone()
        };
        let outer_gate = gated_sigmoid(self.core.g_proj.forward(x.clone()));
        let next = self.core.gated_update(base_state, x.clone(), outer_gate);

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "p1_fractal_hybrid_composite"
    }

    fn hidden_dim(&self) -> usize {
        self.core.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}

#[derive(Module, Debug)]
pub struct P1FractalHybridDynGate<B: Backend> {
    pub(crate) core: P1FractalHybridCore<B>,
}

impl<B: Backend> P1FractalHybridDynGate<B> {
    pub fn new(hidden_dim: usize, device: &B::Device) -> Self {
        Self {
            core: P1FractalHybridCore::new(hidden_dim, device),
        }
    }
}

impl<B: Backend> FractalRule<B> for P1FractalHybridDynGate<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let previous_state = state.flat()?;
        let gate_threshold = 0.95 - 0.25 * depth_fraction(context);
        let gate_cap = Tensor::<B, 2>::ones(previous_state.dims(), &previous_state.device())
            .mul_scalar(gate_threshold);
        let tuned_gate =
            clamp_max_by_row(gated_sigmoid(self.core.g_proj.forward(x.clone())), gate_cap);
        let next = self
            .core
            .gated_update(previous_state, x.clone(), tuned_gate);

        Ok(FractalState::Flat(next))
    }

    fn name(&self) -> &'static str {
        "p1_fractal_hybrid_dyn_gate"
    }

    fn hidden_dim(&self) -> usize {
        self.core.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(self.clone())
    }
}
