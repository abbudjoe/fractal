use burn::{
    module::{Ignored, Module, Param},
    nn::{
        transformer::{
            PositionWiseFeedForward, PositionWiseFeedForwardConfig, TransformerEncoder,
            TransformerEncoderConfig, TransformerEncoderInput,
        },
        Embedding, EmbeddingConfig, Initializer,
    },
    tensor::{activation::sigmoid, backend::Backend, Bool, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

use crate::{
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
    HybridAttentionLayerRole, HybridAttentionModelShape, HybridAttentionVariantSpec,
    ReferenceSsmFamily,
};

const DEFAULT_RUST_MAMBA3_INIT_MIN: f64 = -0.08;
const DEFAULT_RUST_MAMBA3_INIT_MAX: f64 = 0.08;
const DEFAULT_RUST_MAMBA3_NORM_EPS: f64 = 1.0e-5;
const PHASE1_INTERLEAVED_BLOCK_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RustMamba3RopeFraction {
    Half,
    Full,
}

impl RustMamba3RopeFraction {
    const fn numerator(self) -> usize {
        match self {
            Self::Half => 1,
            Self::Full => 2,
        }
    }

    const fn denominator(self) -> usize {
        2
    }

    const fn rotary_dim_divisor(self) -> usize {
        match self {
            Self::Half => 4,
            Self::Full => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RustMamba3BaselineConfig {
    pub d_model: usize,
    pub d_state: usize,
    pub expand: usize,
    pub headdim: usize,
    pub ngroups: usize,
    pub rope_fraction: RustMamba3RopeFraction,
    pub dt_min: f64,
    pub dt_max: f64,
    pub dt_init_floor: f64,
    pub a_floor: f64,
    pub is_outproj_norm: bool,
    pub is_mimo: bool,
    pub mimo_rank: usize,
    pub chunk_size: usize,
}

impl RustMamba3BaselineConfig {
    pub fn phase1_default(d_model: usize, head_count: usize) -> Result<Self, FractalError> {
        if d_model == 0 || head_count == 0 || !d_model.is_multiple_of(head_count) {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3.phase1_default requires d_model > 0 and divisibility by head_count, got d_model={d_model}, head_count={head_count}"
            )));
        }
        Ok(Self {
            d_model,
            d_state: 128,
            expand: 2,
            headdim: d_model / head_count,
            ngroups: 1,
            rope_fraction: RustMamba3RopeFraction::Half,
            dt_min: 1.0e-3,
            dt_max: 1.0e-1,
            dt_init_floor: 1.0e-4,
            a_floor: 1.0e-4,
            is_outproj_norm: false,
            is_mimo: true,
            mimo_rank: 4,
            chunk_size: 16,
        })
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.d_model == 0
            || self.d_state == 0
            || self.expand == 0
            || self.headdim == 0
            || self.ngroups == 0
            || self.chunk_size == 0
        {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3 dimensions, group counts, and chunk_size must be greater than zero"
                    .to_string(),
            ));
        }
        if !(self.dt_min.is_finite()
            && self.dt_max.is_finite()
            && self.dt_init_floor.is_finite()
            && self.a_floor.is_finite())
        {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3 continuous config values must be finite".to_string(),
            ));
        }
        if !(self.dt_min > 0.0
            && self.dt_max >= self.dt_min
            && self.dt_init_floor > 0.0
            && self.a_floor > 0.0)
        {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3 dt/a floors must be positive and dt_max must be at least dt_min"
                    .to_string(),
            ));
        }
        let derived = self.derived_shape()?;
        if self.ngroups > derived.nheads {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3.ngroups {} must not exceed nheads {}",
                self.ngroups, derived.nheads
            )));
        }
        if !derived.nheads.is_multiple_of(self.ngroups) {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3.ngroups {} must divide nheads {}",
                self.ngroups, derived.nheads
            )));
        }
        if self.is_mimo && self.mimo_rank < 2 {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3.is_mimo=true requires mimo_rank >= 2".to_string(),
            ));
        }
        if !self.is_mimo && self.mimo_rank != 1 {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3.is_mimo=false requires mimo_rank=1".to_string(),
            ));
        }
        Ok(())
    }

    pub fn derived_shape(&self) -> Result<RustMamba3DerivedShape, FractalError> {
        if !self.d_model.is_multiple_of(self.headdim) {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3.d_model {} must be divisible by headdim {}",
                self.d_model, self.headdim
            )));
        }
        let d_inner = self.expand * self.d_model;
        if !d_inner.is_multiple_of(self.headdim) {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3.d_inner {} must be divisible by headdim {}",
                d_inner, self.headdim
            )));
        }
        let nheads = d_inner / self.headdim;
        let rope_numerator = self.rope_fraction.numerator();
        let rope_denominator = self.rope_fraction.denominator();
        let mut split_tensor_size = (self.d_state * rope_numerator) / rope_denominator;
        if !split_tensor_size.is_multiple_of(2) {
            split_tensor_size -= 1;
        }
        let num_rope_angles = split_tensor_size / 2;
        if num_rope_angles == 0 {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3 rope fraction must yield at least one rotary angle pair".to_string(),
            ));
        }
        let mimo_rank = if self.is_mimo { self.mimo_rank } else { 1 };
        let d_in_proj =
            2 * d_inner + 2 * self.d_state * self.ngroups * mimo_rank + 3 * nheads + num_rope_angles;
        Ok(RustMamba3DerivedShape {
            d_inner,
            nheads,
            num_bc_heads: self.ngroups,
            split_tensor_size,
            num_rope_angles,
            rotary_dim_divisor: self.rope_fraction.rotary_dim_divisor(),
            d_in_proj,
            mimo_rank,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustMamba3DerivedShape {
    pub d_inner: usize,
    pub nheads: usize,
    pub num_bc_heads: usize,
    pub split_tensor_size: usize,
    pub num_rope_angles: usize,
    pub rotary_dim_divisor: usize,
    pub d_in_proj: usize,
    pub mimo_rank: usize,
}

#[derive(Module, Debug)]
pub struct SimpleRmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    eps: f64,
    width: usize,
}

impl<B: Backend> SimpleRmsNorm<B> {
    pub fn new(width: usize, eps: f64, device: &B::Device) -> Result<Self, FractalError> {
        if width == 0 {
            return Err(FractalError::InvalidConfig(
                "simple_rms_norm width must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            weight: Param::from_data(TensorData::new(vec![1.0f32; width], [width]), device),
            eps,
            width,
        })
    }

    pub fn forward3(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let denom = (input.clone() * input.clone())
            .sum_dim(2)
            .mul_scalar(1.0 / self.width as f64)
            .add_scalar(self.eps)
            .sqrt()
            .repeat(&[1, 1, self.width]);
        let weight = self.weight.val().reshape([1, 1, self.width]);
        input / denom * weight
    }

    pub fn forward4(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let denom = (input.clone() * input.clone())
            .sum_dim(3)
            .mul_scalar(1.0 / self.width as f64)
            .add_scalar(self.eps)
            .sqrt()
            .repeat(&[1, 1, 1, self.width]);
        let weight = self.weight.val().reshape([1, 1, 1, self.width]);
        input / denom * weight
    }

    pub fn forward4_with_gate(&self, input: Tensor<B, 4>, gate: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward4(input) * sigmoid(gate)
    }
}

#[derive(Module, Debug)]
pub struct RustMamba3Mixer<B: Backend> {
    config: Ignored<RustMamba3BaselineConfig>,
    derived: Ignored<RustMamba3DerivedShape>,
    in_proj: StructuredProjection<B>,
    dt_bias: Param<Tensor<B, 1>>,
    b_bias: Param<Tensor<B, 3>>,
    c_bias: Param<Tensor<B, 3>>,
    d_skip: Param<Tensor<B, 1>>,
    mimo_x: Option<Param<Tensor<B, 3>>>,
    mimo_z: Option<Param<Tensor<B, 3>>>,
    mimo_o: Option<Param<Tensor<B, 3>>>,
    b_norm: SimpleRmsNorm<B>,
    c_norm: SimpleRmsNorm<B>,
    out_proj_norm: Option<SimpleRmsNorm<B>>,
    out_proj: StructuredProjection<B>,
}

#[derive(Debug)]
struct RustMamba3StepProjection<B: Backend> {
    z: Tensor<B, 3>,
    x: Tensor<B, 3>,
    b: Tensor<B, 4>,
    c: Tensor<B, 4>,
    dt: Tensor<B, 2>,
    a: Tensor<B, 2>,
    trap: Tensor<B, 2>,
    angles: Tensor<B, 3>,
}

#[derive(Debug)]
struct RustMamba3StepOutput<B: Backend> {
    output: Tensor<B, 2>,
    next_state: Tensor<B, 5>,
}

impl<B: Backend> RustMamba3Mixer<B> {
    pub fn new(config: RustMamba3BaselineConfig, device: &B::Device) -> Result<Self, FractalError> {
        config.validate()?;
        let derived = config.derived_shape()?;
        let projection = StructuredProjectionConfig::new(config.d_model, derived.d_in_proj)
            .with_bias(false)
            .with_initializer(Initializer::Uniform {
                min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                max: DEFAULT_RUST_MAMBA3_INIT_MAX,
            })
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        let out_proj = StructuredProjectionConfig::new(derived.d_inner, config.d_model)
            .with_bias(false)
            .with_initializer(Initializer::Uniform {
                min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                max: DEFAULT_RUST_MAMBA3_INIT_MAX,
            })
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput);
        Ok(Self {
            dt_bias: Param::from_data(
                TensorData::new(
                    vec![dt_bias_init(config.dt_min, config.dt_max, config.dt_init_floor) as f32;
                        derived.nheads],
                    [derived.nheads],
                ),
                device,
            ),
            b_bias: Param::from_data(
                TensorData::new(
                    vec![1.0f32; derived.mimo_rank * derived.nheads * config.d_state],
                    [derived.mimo_rank, derived.nheads, config.d_state],
                ),
                device,
            ),
            c_bias: Param::from_data(
                TensorData::new(
                    vec![1.0f32; derived.mimo_rank * derived.nheads * config.d_state],
                    [derived.mimo_rank, derived.nheads, config.d_state],
                ),
                device,
            ),
            d_skip: Param::from_data(
                TensorData::new(vec![1.0f32; derived.nheads], [derived.nheads]),
                device,
            ),
            mimo_x: config.is_mimo.then(|| {
                Param::from_data(
                    TensorData::new(
                        vec![1.0f32 / derived.mimo_rank as f32;
                            derived.mimo_rank * derived.nheads * config.headdim],
                        [derived.mimo_rank, derived.nheads, config.headdim],
                    ),
                    device,
                )
            }),
            mimo_z: config.is_mimo.then(|| {
                Param::from_data(
                    TensorData::new(
                        vec![1.0f32; derived.mimo_rank * derived.nheads * config.headdim],
                        [derived.mimo_rank, derived.nheads, config.headdim],
                    ),
                    device,
                )
            }),
            mimo_o: config.is_mimo.then(|| {
                Param::from_data(
                    TensorData::new(
                        vec![1.0f32 / derived.mimo_rank as f32;
                            derived.mimo_rank * derived.nheads * config.headdim],
                        [derived.mimo_rank, derived.nheads, config.headdim],
                    ),
                    device,
                )
            }),
            b_norm: SimpleRmsNorm::new(config.d_state, DEFAULT_RUST_MAMBA3_NORM_EPS, device)?,
            c_norm: SimpleRmsNorm::new(config.d_state, DEFAULT_RUST_MAMBA3_NORM_EPS, device)?,
            out_proj_norm: config
                .is_outproj_norm
                .then(|| SimpleRmsNorm::new(derived.d_inner, DEFAULT_RUST_MAMBA3_NORM_EPS, device))
                .transpose()?,
            in_proj: projection.init(device),
            out_proj: out_proj.init(device),
            config: Ignored(config),
            derived: Ignored(derived),
        })
    }

    pub fn config(&self) -> &RustMamba3BaselineConfig {
        &self.config
    }

    pub fn derived_shape(&self) -> RustMamba3DerivedShape {
        *self.derived
    }

    pub fn forward_sequence(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len, width] = input.dims();
        if width != self.config.d_model {
            return Err(FractalError::Shape(format!(
                "rust_mamba3_mixer expected width {}, got {}",
                self.config.d_model, width
            )));
        }
        let device = input.device();
        let mut state = Tensor::<B, 5>::zeros(
            [
                batch_size,
                self.derived.mimo_rank,
                self.derived.nheads,
                self.config.headdim,
                self.config.d_state,
            ],
            &device,
        );
        let mut outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let x_t = input
                .clone()
                .slice([0..batch_size, position..position + 1, 0..self.config.d_model])
                .reshape([batch_size, self.config.d_model]);
            let step = self.step(x_t, state)?;
            outputs.push(step.output.reshape([batch_size, 1, self.config.d_model]));
            state = step.next_state;
        }
        Ok(Tensor::cat(outputs, 1))
    }

    fn step(
        &self,
        input: Tensor<B, 2>,
        previous_state: Tensor<B, 5>,
    ) -> Result<RustMamba3StepOutput<B>, FractalError> {
        let projections = self.project_step(input)?;
        let [batch_size, _, _] = projections.z.dims();
        let rank = self.derived.mimo_rank;
        let nheads = self.derived.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;

        let rotated_state = rotate_state_prefix_pairs_last_dim_5(
            previous_state,
            projections.angles.clone(),
            self.derived.split_tensor_size,
        );
        let x_ranked = self.expand_ranked_heads(projections.x.clone(), self.mimo_x.as_ref());
        let z_ranked = self.expand_ranked_heads(projections.z.clone(), self.mimo_z.as_ref());
        let decay = (projections.a.clone() * projections.dt.clone())
            .exp()
            .reshape([batch_size, 1, nheads, 1, 1]);
        let one_minus_decay = decay.clone().mul_scalar(-1.0).add_scalar(1.0);
        let b = projections
            .b
            .reshape([batch_size, rank, nheads, 1, d_state]);
        let candidate = x_ranked.clone().reshape([batch_size, rank, nheads, headdim, 1]) * b;
        let next_state = decay * rotated_state + one_minus_decay * candidate;

        let c = projections
            .c
            .reshape([batch_size, rank, nheads, 1, d_state]);
        let readout = (next_state.clone() * c)
            .sum_dim(4)
            .reshape([batch_size, rank, nheads, headdim]);
        let skip = x_ranked.clone()
            * self
                .d_skip
                .val()
                .reshape([1, 1, nheads, 1])
                .repeat(&[batch_size, rank, 1, headdim]);
        let trap = projections
            .trap
            .reshape([batch_size, 1, nheads, 1])
            .repeat(&[1, rank, 1, headdim]);
        let mixed_ranked = trap.clone() * readout + trap.mul_scalar(-1.0).add_scalar(1.0) * skip;
        let combined = self.combine_ranked_outputs(mixed_ranked);
        let z_combined = self.combine_ranked_outputs(z_ranked);
        let gated = if let Some(norm) = &self.out_proj_norm {
            norm.forward4_with_gate(
                combined
                    .clone()
                    .reshape([batch_size, 1, 1, self.derived.d_inner]),
                z_combined
                    .clone()
                    .reshape([batch_size, 1, 1, self.derived.d_inner]),
            )
            .reshape([batch_size, self.derived.d_inner])
        } else {
            combined * sigmoid(z_combined)
        };
        let output = self.out_proj.forward(gated);
        Ok(RustMamba3StepOutput { output, next_state })
    }

    fn project_step(&self, input: Tensor<B, 2>) -> Result<RustMamba3StepProjection<B>, FractalError> {
        let [batch_size, width] = input.dims();
        if width != self.config.d_model {
            return Err(FractalError::Shape(format!(
                "rust_mamba3.project_step expected width {}, got {}",
                self.config.d_model, width
            )));
        }
        let rank = self.derived.mimo_rank;
        let nheads = self.derived.nheads;
        let d_inner = self.derived.d_inner;
        let bc_width = self.config.d_state * self.config.ngroups * rank;
        let zx_bc_dt_a_trap_angles = self.in_proj.forward(input);
        let z_end = d_inner;
        let x_end = z_end + d_inner;
        let b_end = x_end + bc_width;
        let c_end = b_end + bc_width;
        let dt_end = c_end + nheads;
        let a_end = dt_end + nheads;
        let trap_end = a_end + nheads;

        let z = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, 0..z_end])
            .reshape([batch_size, nheads, self.config.headdim]);
        let x = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, z_end..x_end])
            .reshape([batch_size, nheads, self.config.headdim]);
        let raw_b = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, x_end..b_end])
            .reshape([batch_size, rank, self.config.ngroups, self.config.d_state]);
        let raw_c = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, b_end..c_end])
            .reshape([batch_size, rank, self.config.ngroups, self.config.d_state]);
        let dd_dt = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, c_end..dt_end])
            .reshape([batch_size, nheads]);
        let dd_a = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, dt_end..a_end])
            .reshape([batch_size, nheads]);
        let trap_proj = zx_bc_dt_a_trap_angles
            .clone()
            .slice([0..batch_size, a_end..trap_end])
            .reshape([batch_size, nheads]);
        let angle_proj = zx_bc_dt_a_trap_angles
            .slice([0..batch_size, trap_end..self.derived.d_in_proj])
            .reshape([batch_size, self.derived.num_rope_angles]);

        let dt = softplus(dd_dt + self.dt_bias.val().reshape([1, nheads]).repeat(&[batch_size, 1]));
        let a = softplus(dd_a).mul_scalar(-1.0).clamp(-1.0e9, -self.config.a_floor);
        let trap = sigmoid(trap_proj);
        let angles = angle_proj
            .reshape([batch_size, 1, self.derived.num_rope_angles])
            .repeat(&[1, nheads, 1]);
        let b = self.expand_grouped_bc(self.b_norm.forward4(raw_b))
            + self
                .b_bias
                .val()
                .reshape([1, rank, nheads, self.config.d_state])
                .repeat(&[batch_size, 1, 1, 1]);
        let c = self.expand_grouped_bc(self.c_norm.forward4(raw_c))
            + self
                .c_bias
                .val()
                .reshape([1, rank, nheads, self.config.d_state])
                .repeat(&[batch_size, 1, 1, 1]);

        Ok(RustMamba3StepProjection {
            z,
            x,
            b,
            c,
            dt,
            a,
            trap,
            angles,
        })
    }

    fn expand_grouped_bc(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, rank, groups, d_state] = tensor.dims();
        let heads_per_group = self.derived.nheads / groups;
        tensor
            .reshape([batch_size, rank, groups, 1, d_state])
            .repeat(&[1, 1, 1, heads_per_group, 1])
            .reshape([batch_size, rank, self.derived.nheads, d_state])
    }

    fn expand_ranked_heads(
        &self,
        tensor: Tensor<B, 3>,
        weights: Option<&Param<Tensor<B, 3>>>,
    ) -> Tensor<B, 4> {
        let [batch_size, nheads, headdim] = tensor.dims();
        match weights {
            Some(weights) => tensor.reshape([batch_size, 1, nheads, headdim]).repeat(&[
                1,
                self.derived.mimo_rank,
                1,
                1,
            ]) * weights
                .val()
                .reshape([1, self.derived.mimo_rank, nheads, headdim])
                .repeat(&[batch_size, 1, 1, 1]),
            None => tensor.reshape([batch_size, 1, nheads, headdim]),
        }
    }

    fn combine_ranked_outputs(&self, tensor: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, rank, nheads, headdim] = tensor.dims();
        let combined = if let Some(mimo_o) = &self.mimo_o {
            (tensor
                * mimo_o
                    .val()
                    .reshape([1, rank, nheads, headdim])
                    .repeat(&[batch_size, 1, 1, 1]))
            .sum_dim(1)
            .reshape([batch_size, nheads, headdim])
        } else {
            tensor
                .slice([0..batch_size, 0..1, 0..nheads, 0..headdim])
                .reshape([batch_size, nheads, headdim])
        };
        combined.reshape([batch_size, self.derived.d_inner])
    }
}

#[derive(Module, Debug)]
pub struct RustMamba3MixerBlock<B: Backend> {
    input_norm: SimpleRmsNorm<B>,
    output_norm: SimpleRmsNorm<B>,
    mixer: RustMamba3Mixer<B>,
    feedforward: PositionWiseFeedForward<B>,
    d_model: usize,
}

impl<B: Backend> RustMamba3MixerBlock<B> {
    pub fn new(
        shape: HybridAttentionModelShape,
        config: RustMamba3BaselineConfig,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        shape.validate()?;
        if shape.d_model != config.d_model {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3_mixer_block shape d_model {} must match config d_model {}",
                shape.d_model, config.d_model
            )));
        }
        Ok(Self {
            input_norm: SimpleRmsNorm::new(shape.d_model, DEFAULT_RUST_MAMBA3_NORM_EPS, device)?,
            output_norm: SimpleRmsNorm::new(shape.d_model, DEFAULT_RUST_MAMBA3_NORM_EPS, device)?,
            mixer: RustMamba3Mixer::new(config, device)?,
            feedforward: PositionWiseFeedForwardConfig::new(shape.d_model, shape.d_ff)
                .with_dropout(0.0)
                .with_initializer(Initializer::Uniform {
                    min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                    max: DEFAULT_RUST_MAMBA3_INIT_MAX,
                })
                .init(device),
            d_model: shape.d_model,
        })
    }

    pub fn mixer(&self) -> &RustMamba3Mixer<B> {
        &self.mixer
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
        let [_, _, width] = input.dims();
        if width != self.d_model {
            return Err(FractalError::Shape(format!(
                "rust_mamba3_mixer_block expected width {}, got {}",
                self.d_model, width
            )));
        }
        let normed = self.input_norm.forward3(input.clone());
        let mixed = self.mixer.forward_sequence(normed)?;
        let residual = input + mixed;
        let ff = self.feedforward.forward(self.output_norm.forward3(residual.clone()));
        Ok(residual + ff)
    }
}

#[derive(Module, Debug)]
pub struct RustMamba3ReferenceHybridAttentionModel<B: Backend> {
    embedding: Embedding<B>,
    attention_layers: Vec<TransformerEncoder<B>>,
    reference_layers: Vec<RustMamba3MixerBlock<B>>,
    final_norm: SimpleRmsNorm<B>,
    output: LanguageModelHead<B>,
    vocab_size: usize,
    d_model: usize,
    head_count: usize,
    local_window: usize,
    total_layers: usize,
    reference_family: Ignored<ReferenceSsmFamily>,
}

impl<B: Backend> RustMamba3ReferenceHybridAttentionModel<B> {
    pub fn new(
        shape: HybridAttentionModelShape,
        baseline: RustMamba3BaselineConfig,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        shape.validate()?;
        baseline.validate()?;
        if baseline.d_model != shape.d_model {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3_reference_hybrid shape d_model {} must match baseline d_model {}",
                shape.d_model, baseline.d_model
            )));
        }
        if shape.total_layers != PHASE1_INTERLEAVED_BLOCK_COUNT * 2 {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3_reference_hybrid phase-1 expects {} total layers, got {}",
                PHASE1_INTERLEAVED_BLOCK_COUNT * 2,
                shape.total_layers
            )));
        }
        let attention_layers = (0..PHASE1_INTERLEAVED_BLOCK_COUNT)
            .map(|_| {
                TransformerEncoderConfig::new(shape.d_model, shape.d_ff, shape.head_count, 1)
                    .with_dropout(0.0)
                    .with_norm_first(true)
                    .with_initializer(Initializer::Uniform {
                        min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                        max: DEFAULT_RUST_MAMBA3_INIT_MAX,
                    })
                    .init(device)
            })
            .collect();
        let mut reference_layers = Vec::with_capacity(PHASE1_INTERLEAVED_BLOCK_COUNT);
        for _ in 0..PHASE1_INTERLEAVED_BLOCK_COUNT {
            reference_layers.push(RustMamba3MixerBlock::new(shape, baseline.clone(), device)?);
        }
        Ok(Self {
            embedding: EmbeddingConfig::new(shape.vocab_size, shape.d_model)
                .with_initializer(Initializer::Uniform {
                    min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                    max: DEFAULT_RUST_MAMBA3_INIT_MAX,
                })
                .init(device),
            attention_layers,
            reference_layers,
            final_norm: SimpleRmsNorm::new(shape.d_model, DEFAULT_RUST_MAMBA3_NORM_EPS, device)?,
            output: LanguageModelHeadConfig::new(shape.d_model, shape.vocab_size)
                .with_bias(false)
                .with_initializer(Initializer::Uniform {
                    min: DEFAULT_RUST_MAMBA3_INIT_MIN,
                    max: DEFAULT_RUST_MAMBA3_INIT_MAX,
                })
                .init(device),
            vocab_size: shape.vocab_size,
            d_model: shape.d_model,
            head_count: shape.head_count,
            local_window: shape.local_window,
            total_layers: shape.total_layers,
            reference_family: Ignored(ReferenceSsmFamily::Mamba3RustV1),
        })
    }

    pub fn shape(&self) -> HybridAttentionModelShape {
        HybridAttentionModelShape {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            d_ff: self.d_model * 4,
            head_count: self.head_count,
            local_window: self.local_window,
            total_layers: self.total_layers,
        }
    }

    pub fn reference_family(&self) -> ReferenceSsmFamily {
        *self.reference_family
    }

    pub fn forward_logits(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let mut hidden = self.embedding.forward(input_ids);
        let mask =
            local_causal_mask::<B>(batch_size, seq_len, self.local_window, &hidden.device());
        for (attention, reference_ssm) in self
            .attention_layers
            .iter()
            .zip(self.reference_layers.iter())
        {
            hidden =
                attention.forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone()));
            hidden = reference_ssm.forward(hidden)?;
        }
        Ok(self.output.forward(self.final_norm.forward3(hidden)))
    }
}

pub fn build_rust_mamba3_reference_hybrid_attention_model<B: Backend>(
    vocab_size: usize,
    variant: &HybridAttentionVariantSpec,
    device: &B::Device,
) -> Result<RustMamba3ReferenceHybridAttentionModel<B>, FractalError> {
    variant.validate()?;
    if variant.reference_ssm_family != Some(ReferenceSsmFamily::Mamba3RustV1) {
        return Err(FractalError::InvalidConfig(format!(
            "reference variant {} must set reference_ssm_family=mamba3-rust-v1",
            variant.label
        )));
    }
    if variant
        .layer_schedule
        .iter()
        .any(|role| !matches!(role, HybridAttentionLayerRole::ExactAttention | HybridAttentionLayerRole::ReferenceSsm))
    {
        return Err(FractalError::InvalidConfig(format!(
            "reference variant {} must contain only exact-attention and reference-SSM layers",
            variant.label
        )));
    }

    let shape = HybridAttentionModelShape {
        vocab_size,
        d_model: variant.hidden_dim,
        d_ff: variant.hidden_dim * 4,
        head_count: variant.head_count,
        local_window: variant.local_window,
        total_layers: variant.total_layers(),
    };
    let baseline = RustMamba3BaselineConfig::phase1_default(variant.hidden_dim, variant.head_count)?;
    RustMamba3ReferenceHybridAttentionModel::new(shape, baseline, device)
}

fn local_causal_mask<B: Backend>(
    batch_size: usize,
    seq_len: usize,
    local_window: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mut data = Vec::with_capacity(batch_size * seq_len * seq_len);
    for _ in 0..batch_size {
        for query in 0..seq_len {
            let earliest_visible = query.saturating_sub(local_window.saturating_sub(1));
            for key in 0..seq_len {
                data.push(key >= earliest_visible && key <= query);
            }
        }
    }
    Tensor::from_data(
        TensorData::new(data, [batch_size, seq_len, seq_len]),
        device,
    )
}

fn softplus<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.exp().add_scalar(1.0).log()
}

fn dt_bias_init(dt_min: f64, dt_max: f64, dt_init_floor: f64) -> f64 {
    let dt = (dt_min * dt_max).sqrt().max(dt_init_floor);
    dt + (-(-dt).exp_m1()).ln()
}

fn rotate_state_prefix_pairs_last_dim_5<B: Backend>(
    tensor: Tensor<B, 5>,
    angles: Tensor<B, 3>,
    split_tensor_size: usize,
) -> Tensor<B, 5> {
    if split_tensor_size == 0 {
        return tensor;
    }
    let [batch_size, rank, heads, width, state_dim] = tensor.dims();
    let prefix_width = split_tensor_size.min(state_dim - (state_dim % 2));
    if prefix_width == 0 {
        return tensor;
    }
    let pair_count = prefix_width / 2;
    let prefix = tensor
        .clone()
        .slice([0..batch_size, 0..rank, 0..heads, 0..width, 0..prefix_width])
        .reshape([batch_size * rank * heads * width, pair_count, 2]);
    let first = prefix
        .clone()
        .slice([0..batch_size * rank * heads * width, 0..pair_count, 0..1])
        .reshape([batch_size * rank * heads * width, pair_count]);
    let second = prefix
        .slice([0..batch_size * rank * heads * width, 0..pair_count, 1..2])
        .reshape([batch_size * rank * heads * width, pair_count]);
    let angle_grid = angles
        .reshape([batch_size, 1, heads, 1, pair_count])
        .repeat(&[1, rank, 1, width, 1])
        .reshape([batch_size * rank * heads * width, pair_count]);
    let cos = angle_grid.clone().cos();
    let sin = angle_grid.sin();
    let rotated_first = first.clone() * cos.clone() - second.clone() * sin.clone();
    let rotated_second = first * sin + second * cos;
    let rotated_prefix = Tensor::cat(
        vec![
            rotated_first.reshape([batch_size * rank * heads * width, pair_count, 1]),
            rotated_second.reshape([batch_size * rank * heads * width, pair_count, 1]),
        ],
        2,
    )
    .reshape([batch_size, rank, heads, width, prefix_width]);
    if prefix_width == state_dim {
        rotated_prefix
    } else {
        Tensor::cat(
            vec![
                rotated_prefix,
                tensor.slice([0..batch_size, 0..rank, 0..heads, 0..width, prefix_width..state_dim]),
            ],
            4,
        )
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;
    use burn::tensor::{Int, Tensor};

    use super::{
        build_rust_mamba3_reference_hybrid_attention_model, rotate_state_prefix_pairs_last_dim_5,
        RustMamba3BaselineConfig, RustMamba3RopeFraction,
    };
    use crate::{phase1_hybrid_attention_baseline_matrix, ReferenceSsmFamily};

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn phase1_baseline_config_derives_official_style_shapes() {
        let config = RustMamba3BaselineConfig::phase1_default(128, 4).unwrap();
        let derived = config.derived_shape().unwrap();
        assert_eq!(derived.d_inner, 256);
        assert_eq!(derived.nheads, 8);
        assert_eq!(derived.num_bc_heads, 1);
        assert_eq!(derived.split_tensor_size, 64);
        assert_eq!(derived.num_rope_angles, 32);
    }

    #[test]
    fn rope_fraction_full_keeps_full_state_prefix() {
        let mut config = RustMamba3BaselineConfig::phase1_default(128, 4).unwrap();
        config.is_mimo = false;
        config.mimo_rank = 1;
        config.rope_fraction = RustMamba3RopeFraction::Full;
        let derived = config.derived_shape().unwrap();
        assert_eq!(derived.split_tensor_size, config.d_state);
    }

    #[test]
    fn reference_rust_mamba3_model_returns_logits() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        assert_eq!(
            matrix.reference_ssm_hybrid.reference_ssm_family,
            Some(ReferenceSsmFamily::Mamba3RustV1)
        );
        let model = build_rust_mamba3_reference_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.reference_ssm_hybrid,
            &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn rotary_prefix_rotation_changes_only_the_target_prefix() {
        let device = Default::default();
        let state = Tensor::<TestBackend, 5>::from_data(
            [[[[[1.0, 0.0, 2.0, 0.0]]]]],
            &device,
        );
        let angles =
            Tensor::<TestBackend, 3>::from_data([[[core::f32::consts::FRAC_PI_2]]], &device);
        let rotated = rotate_state_prefix_pairs_last_dim_5(state, angles, 2);
        let values = rotated.to_data().to_vec::<f32>().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-4);
        assert!((values[1] - 1.0).abs() < 1e-4);
        assert!((values[2] - 2.0).abs() < 1e-4);
        assert!((values[3] - 0.0).abs() < 1e-4);
    }
}
