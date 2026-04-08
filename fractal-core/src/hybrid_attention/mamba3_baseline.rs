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
pub const DEFAULT_RUST_MAMBA3_NORM_EPS: f64 = 1.0e-5;

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

    pub fn phase1_siso_default(d_model: usize, head_count: usize) -> Result<Self, FractalError> {
        let mut config = Self::phase1_default(d_model, head_count)?;
        config.is_mimo = false;
        config.mimo_rank = 1;
        Ok(config)
    }

    pub fn phase1_for_reference_family(
        reference_family: ReferenceSsmFamily,
        d_model: usize,
        head_count: usize,
    ) -> Result<Self, FractalError> {
        match reference_family {
            ReferenceSsmFamily::Mamba3ProxyV1 => Err(FractalError::InvalidConfig(
                "rust_mamba3 baseline config may not be derived for the proxy reference family"
                    .to_string(),
            )),
            ReferenceSsmFamily::Mamba3RustV1 => Self::phase1_default(d_model, head_count),
            ReferenceSsmFamily::Mamba3RustSisoV1 => Self::phase1_siso_default(d_model, head_count),
        }
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
        let d_in_proj = 2 * d_inner
            + 2 * self.d_state * self.ngroups * mimo_rank
            + 3 * nheads
            + num_rope_angles;
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

#[derive(Debug, Clone)]
struct RustMamba3RecurrentState<B: Backend> {
    angle_dt_state: Tensor<B, 3>,
    ssm_state: Tensor<B, 4>,
    k_state: Tensor<B, 4>,
    v_state: Tensor<B, 3>,
}

impl<B: Backend> RustMamba3RecurrentState<B> {
    fn zeros(
        batch_size: usize,
        derived: RustMamba3DerivedShape,
        config: &RustMamba3BaselineConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            angle_dt_state: Tensor::<B, 3>::zeros(
                [batch_size, derived.nheads, derived.num_rope_angles],
                device,
            ),
            ssm_state: Tensor::<B, 4>::zeros(
                [batch_size, derived.nheads, config.headdim, config.d_state],
                device,
            ),
            k_state: Tensor::<B, 4>::zeros(
                [
                    batch_size,
                    derived.mimo_rank,
                    derived.nheads,
                    config.d_state,
                ],
                device,
            ),
            v_state: Tensor::<B, 3>::zeros([batch_size, derived.nheads, config.headdim], device),
        }
    }
}

#[derive(Debug)]
struct RustMamba3StepOutput<B: Backend> {
    output: Tensor<B, 2>,
    next_state: RustMamba3RecurrentState<B>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
struct RustMamba3SequenceOutput<B: Backend> {
    outputs: Tensor<B, 3>,
    final_state: RustMamba3RecurrentState<B>,
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
                    vec![
                        dt_bias_init(config.dt_min, config.dt_max, config.dt_init_floor) as f32;
                        derived.nheads
                    ],
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
                        vec![
                            1.0f32 / derived.mimo_rank as f32;
                            derived.mimo_rank * derived.nheads * config.headdim
                        ],
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
                        vec![
                            1.0f32 / derived.mimo_rank as f32;
                            derived.mimo_rank * derived.nheads * config.headdim
                        ],
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

    fn allocate_recurrent_state(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> RustMamba3RecurrentState<B> {
        RustMamba3RecurrentState::zeros(batch_size, self.derived_shape(), self.config(), device)
    }

    pub fn forward_sequence(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
        Ok(self.scan_sequence(input)?.outputs)
    }

    fn scan_sequence(
        &self,
        input: Tensor<B, 3>,
    ) -> Result<RustMamba3SequenceOutput<B>, FractalError> {
        let [batch_size, _, width] = input.dims();
        if width != self.config.d_model {
            return Err(FractalError::Shape(format!(
                "rust_mamba3_mixer expected width {}, got {}",
                self.config.d_model, width
            )));
        }
        let device = input.device();
        let state = self.allocate_recurrent_state(batch_size, &device);
        self.scan_sequence_from_state(input, state)
    }

    fn scan_sequence_from_state(
        &self,
        input: Tensor<B, 3>,
        mut state: RustMamba3RecurrentState<B>,
    ) -> Result<RustMamba3SequenceOutput<B>, FractalError> {
        let [batch_size, seq_len, width] = input.dims();
        if width != self.config.d_model {
            return Err(FractalError::Shape(format!(
                "rust_mamba3_mixer expected width {}, got {}",
                self.config.d_model, width
            )));
        }
        let mut outputs = Vec::with_capacity(seq_len);
        for chunk_start in (0..seq_len).step_by(self.config.chunk_size) {
            let chunk_end = (chunk_start + self.config.chunk_size).min(seq_len);
            for position in chunk_start..chunk_end {
                let x_t = input
                    .clone()
                    .slice([
                        0..batch_size,
                        position..position + 1,
                        0..self.config.d_model,
                    ])
                    .reshape([batch_size, self.config.d_model]);
                let step = self.step(x_t, state)?;
                outputs.push(step.output.reshape([batch_size, 1, self.config.d_model]));
                state = step.next_state;
            }
        }
        Ok(RustMamba3SequenceOutput {
            outputs: Tensor::cat(outputs, 1),
            final_state: state,
        })
    }

    fn step(
        &self,
        input: Tensor<B, 2>,
        previous_state: RustMamba3RecurrentState<B>,
    ) -> Result<RustMamba3StepOutput<B>, FractalError> {
        let projections = self.project_step(input)?;
        let [batch_size, _, _] = projections.z.dims();
        let rank = self.derived.mimo_rank;
        let nheads = self.derived.nheads;
        let headdim = self.config.headdim;
        let d_state = self.config.d_state;
        let delta_angles = projections.angles.clone().tanh()
            * projections.dt.clone().reshape([batch_size, nheads, 1])
            * core::f64::consts::PI;
        let next_angle_state = previous_state.angle_dt_state.clone() + delta_angles;
        let rotated_b = rotate_state_prefix_pairs_last_dim_4(
            projections.b.clone(),
            next_angle_state.clone(),
            self.derived.split_tensor_size,
            !self.config.is_mimo,
        );
        let rotated_c = rotate_state_prefix_pairs_last_dim_4(
            projections.c.clone(),
            next_angle_state.clone(),
            self.derived.split_tensor_size,
            !self.config.is_mimo,
        );
        let x_ranked = self.expand_ranked_heads(projections.x.clone(), self.mimo_x.as_ref());
        let z_ranked = self.expand_ranked_heads(projections.z.clone(), self.mimo_z.as_ref());
        let alpha = (projections.a.clone() * projections.dt.clone())
            .exp()
            .reshape([batch_size, nheads, 1, 1]);
        let beta = projections.trap.clone().mul_scalar(-1.0).add_scalar(1.0)
            * projections.dt.clone()
            * alpha.clone().reshape([batch_size, nheads]);
        let gamma = projections.trap.clone() * projections.dt.clone();

        let x_bt_state = (x_ranked.clone()
            * gamma
                .clone()
                .reshape([batch_size, 1, nheads, 1])
                .repeat(&[1, rank, 1, headdim]))
        .reshape([batch_size, rank, nheads, headdim, 1])
            * rotated_b
                .clone()
                .reshape([batch_size, rank, nheads, 1, d_state]);
        let x_bt_prev = (self
            .expand_ranked_heads(previous_state.v_state.clone(), self.mimo_x.as_ref())
            * beta
                .reshape([batch_size, 1, nheads, 1])
                .repeat(&[1, rank, 1, headdim]))
        .reshape([batch_size, rank, nheads, headdim, 1])
            * previous_state
                .k_state
                .clone()
                .reshape([batch_size, rank, nheads, 1, d_state]);
        let next_ssm_state = previous_state.ssm_state.clone() * alpha
            + x_bt_state
                .sum_dim(1)
                .reshape([batch_size, nheads, headdim, d_state])
            + x_bt_prev
                .sum_dim(1)
                .reshape([batch_size, nheads, headdim, d_state]);

        let out_ranked = (next_ssm_state
            .clone()
            .reshape([batch_size, 1, nheads, headdim, d_state])
            * rotated_c
                .clone()
                .reshape([batch_size, rank, nheads, 1, d_state]))
        .sum_dim(4)
        .reshape([batch_size, rank, nheads, headdim])
            + x_ranked.clone()
                * self
                    .d_skip
                    .val()
                    .reshape([1, 1, nheads, 1])
                    .repeat(&[batch_size, rank, 1, headdim]);

        let next_v_state = projections.x.clone();
        let next_k_state = rotated_b;
        let gated_ranked = if let Some(norm) = &self.out_proj_norm {
            let combined = out_ranked
                .clone()
                .reshape([batch_size, rank, 1, self.derived.d_inner]);
            let z_combined = z_ranked
                .clone()
                .reshape([batch_size, rank, 1, self.derived.d_inner]);
            norm.forward4_with_gate(combined, z_combined)
                .reshape([batch_size, rank, nheads, headdim])
        } else {
            let z_silu = z_ranked.clone() * sigmoid(z_ranked);
            out_ranked * z_silu
        };
        let combined = self.combine_ranked_outputs(gated_ranked);
        let gated = combined.reshape([batch_size, self.derived.d_inner]);
        let output = self.out_proj.forward(gated);
        Ok(RustMamba3StepOutput {
            output,
            next_state: RustMamba3RecurrentState {
                angle_dt_state: next_angle_state,
                ssm_state: next_ssm_state,
                k_state: next_k_state,
                v_state: next_v_state,
            },
        })
    }

    fn project_step(
        &self,
        input: Tensor<B, 2>,
    ) -> Result<RustMamba3StepProjection<B>, FractalError> {
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

        let dt = softplus(
            dd_dt
                + self
                    .dt_bias
                    .val()
                    .reshape([1, nheads])
                    .repeat(&[batch_size, 1]),
        );
        let a = softplus(dd_a)
            .mul_scalar(-1.0)
            .clamp(-1.0e9, -self.config.a_floor);
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
            Some(weights) => {
                tensor.reshape([batch_size, 1, nheads, headdim]).repeat(&[
                    1,
                    self.derived.mimo_rank,
                    1,
                    1,
                ]) * weights
                    .val()
                    .reshape([1, self.derived.mimo_rank, nheads, headdim])
                    .repeat(&[batch_size, 1, 1, 1])
            }
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
        let ff = self
            .feedforward
            .forward(self.output_norm.forward3(residual.clone()));
        Ok(residual + ff)
    }
}

#[derive(Module, Debug)]
pub struct RustMamba3ReferenceHybridAttentionModel<B: Backend> {
    embedding: Embedding<B>,
    attention_layers: Vec<TransformerEncoder<B>>,
    reference_layers: Vec<RustMamba3MixerBlock<B>>,
    layer_schedule: Ignored<Vec<HybridAttentionLayerRole>>,
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
        reference_family: ReferenceSsmFamily,
        layer_schedule: &[HybridAttentionLayerRole],
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
        if shape.total_layers != layer_schedule.len() {
            return Err(FractalError::InvalidConfig(format!(
                "rust_mamba3_reference_hybrid expected total_layers {} to match schedule length {}",
                shape.total_layers,
                layer_schedule.len()
            )));
        }
        if layer_schedule.iter().any(|role| {
            !matches!(
                role,
                HybridAttentionLayerRole::ExactAttention | HybridAttentionLayerRole::ReferenceSsm
            )
        }) {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3_reference_hybrid schedule may contain only exact-attention and reference-SSM layers".to_string(),
            ));
        }
        let attention_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ExactAttention))
            .count();
        let reference_count = layer_schedule
            .iter()
            .filter(|role| matches!(role, HybridAttentionLayerRole::ReferenceSsm))
            .count();
        if attention_count == 0 || reference_count == 0 {
            return Err(FractalError::InvalidConfig(
                "rust_mamba3_reference_hybrid schedule must contain at least one exact-attention and one reference-SSM layer".to_string(),
            ));
        }
        let attention_layers = (0..attention_count)
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
        let mut reference_layers = Vec::with_capacity(reference_count);
        for _ in 0..reference_count {
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
            layer_schedule: Ignored(layer_schedule.to_vec()),
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
            reference_family: Ignored(reference_family),
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
        let mask = local_causal_mask::<B>(batch_size, seq_len, self.local_window, &hidden.device());
        let mut attention_index = 0;
        let mut reference_index = 0;
        for role in self.layer_schedule.iter() {
            match role {
                HybridAttentionLayerRole::ExactAttention => {
                    hidden = self.attention_layers[attention_index]
                        .forward(TransformerEncoderInput::new(hidden).mask_attn(mask.clone()));
                    attention_index += 1;
                }
                HybridAttentionLayerRole::ReferenceSsm => {
                    hidden = self.reference_layers[reference_index].forward(hidden)?;
                    reference_index += 1;
                }
                HybridAttentionLayerRole::Primitive => {
                    return Err(FractalError::InvalidConfig(
                        "rust_mamba3_reference_hybrid schedule contained an unexpected primitive layer at runtime".to_string(),
                    ));
                }
            }
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
    let reference_family = variant.reference_ssm_family.ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "reference variant {} must set reference_ssm_family",
            variant.label
        ))
    })?;
    if !matches!(
        reference_family,
        ReferenceSsmFamily::Mamba3RustV1 | ReferenceSsmFamily::Mamba3RustSisoV1
    ) {
        return Err(FractalError::InvalidConfig(format!(
            "reference variant {} must set reference_ssm_family to a Rust Mamba family, got {:?}",
            variant.label, reference_family
        )));
    }
    if variant.layer_schedule.iter().any(|role| {
        !matches!(
            role,
            HybridAttentionLayerRole::ExactAttention | HybridAttentionLayerRole::ReferenceSsm
        )
    }) {
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
    let baseline = RustMamba3BaselineConfig::phase1_for_reference_family(
        reference_family,
        variant.hidden_dim,
        variant.head_count,
    )?;
    RustMamba3ReferenceHybridAttentionModel::new(
        shape,
        baseline,
        reference_family,
        &variant.layer_schedule,
        device,
    )
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
                data.push(key < earliest_visible || key > query);
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

fn rotate_state_prefix_pairs_last_dim_4<B: Backend>(
    tensor: Tensor<B, 4>,
    angles: Tensor<B, 3>,
    split_tensor_size: usize,
    rotate_pairwise: bool,
) -> Tensor<B, 4> {
    if split_tensor_size == 0 {
        return tensor;
    }
    let [batch_size, rank, heads, state_dim] = tensor.dims();
    let prefix_width = split_tensor_size.min(state_dim - (state_dim % 2));
    if prefix_width == 0 {
        return tensor;
    }
    let pair_count = prefix_width / 2;
    let prefix = tensor
        .clone()
        .slice([0..batch_size, 0..rank, 0..heads, 0..prefix_width]);
    let angle_grid = angles
        .reshape([batch_size, 1, heads, pair_count])
        .repeat(&[1, rank, 1, 1])
        .reshape([batch_size, rank, heads, pair_count]);
    let cos = angle_grid.clone().cos();
    let sin = angle_grid.sin();
    let (rotated_first, rotated_second) = if rotate_pairwise {
        let paired = prefix.reshape([batch_size * rank * heads, pair_count, 2]);
        let first = paired
            .clone()
            .slice([0..batch_size * rank * heads, 0..pair_count, 0..1])
            .reshape([batch_size, rank, heads, pair_count]);
        let second = paired
            .slice([0..batch_size * rank * heads, 0..pair_count, 1..2])
            .reshape([batch_size, rank, heads, pair_count]);
        (
            first.clone() * cos.clone() - second.clone() * sin.clone(),
            first * sin + second * cos,
        )
    } else {
        let first = prefix
            .clone()
            .slice([0..batch_size, 0..rank, 0..heads, 0..pair_count]);
        let second = prefix.slice([0..batch_size, 0..rank, 0..heads, pair_count..prefix_width]);
        (
            first.clone() * cos.clone() - second.clone() * sin.clone(),
            first * sin + second * cos,
        )
    };
    let rotated_prefix = if rotate_pairwise {
        Tensor::cat(
            vec![
                rotated_first.reshape([batch_size, rank, heads, pair_count, 1]),
                rotated_second.reshape([batch_size, rank, heads, pair_count, 1]),
            ],
            4,
        )
        .reshape([batch_size, rank, heads, prefix_width])
    } else {
        Tensor::cat(vec![rotated_first, rotated_second], 3)
    };
    if prefix_width == state_dim {
        rotated_prefix
    } else {
        Tensor::cat(
            vec![
                rotated_prefix,
                tensor.slice([0..batch_size, 0..rank, 0..heads, prefix_width..state_dim]),
            ],
            3,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::PathBuf,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::{
        backend::Candle,
        module::{Module, Param},
        record::{BinFileRecorder, FullPrecisionSettings},
        tensor::{Int, Tensor, TensorData},
    };
    use serde::de::DeserializeOwned;
    use serde::{Deserialize, Serialize};

    use super::{
        build_rust_mamba3_reference_hybrid_attention_model, rotate_state_prefix_pairs_last_dim_4,
        RustMamba3BaselineConfig, RustMamba3DerivedShape, RustMamba3Mixer,
        RustMamba3RecurrentState, RustMamba3RopeFraction, SimpleRmsNorm,
        DEFAULT_RUST_MAMBA3_NORM_EPS,
    };
    use crate::{
        error::FractalError,
        language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
        phase1_hybrid_attention_baseline_matrix,
        projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionRecord},
        MetalTrainBackend, ReferenceSsmFamily,
    };

    type TestBackend = Candle<f32, i64>;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TensorFixture {
        shape: Vec<usize>,
        values: Vec<f32>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonReferenceInput {
        config: RustMamba3BaselineConfig,
        derived: RustMamba3DerivedShape,
        input: TensorFixture,
        angle_state: TensorFixture,
        in_proj_weight: TensorFixture,
        dt_bias: TensorFixture,
        b_bias: TensorFixture,
        c_bias: TensorFixture,
        b_norm_weight: TensorFixture,
        c_norm_weight: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonReferenceOutput {
        z: TensorFixture,
        x: TensorFixture,
        dt: TensorFixture,
        a: TensorFixture,
        trap: TensorFixture,
        angles: TensorFixture,
        b: TensorFixture,
        c: TensorFixture,
        next_angle_state: TensorFixture,
        rotated_b: TensorFixture,
        rotated_c: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonStepReferenceInput {
        mode: String,
        official_repo: Option<String>,
        config: RustMamba3BaselineConfig,
        derived: RustMamba3DerivedShape,
        input: TensorFixture,
        angle_state: TensorFixture,
        ssm_state: TensorFixture,
        k_state: TensorFixture,
        v_state: TensorFixture,
        in_proj_weight: TensorFixture,
        dt_bias: TensorFixture,
        b_bias: TensorFixture,
        c_bias: TensorFixture,
        b_norm_weight: TensorFixture,
        c_norm_weight: TensorFixture,
        d_skip: TensorFixture,
        mimo_x: Option<TensorFixture>,
        mimo_z: Option<TensorFixture>,
        mimo_o: Option<TensorFixture>,
        out_proj_weight: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonStepReferenceOutput {
        output: TensorFixture,
        next_angle_state: TensorFixture,
        next_ssm_state: TensorFixture,
        next_k_state: TensorFixture,
        next_v_state: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonSequenceReferenceInput {
        mode: String,
        official_repo: Option<String>,
        config: RustMamba3BaselineConfig,
        derived: RustMamba3DerivedShape,
        sequence_input: TensorFixture,
        angle_state: TensorFixture,
        ssm_state: TensorFixture,
        k_state: TensorFixture,
        v_state: TensorFixture,
        in_proj_weight: TensorFixture,
        dt_bias: TensorFixture,
        b_bias: TensorFixture,
        c_bias: TensorFixture,
        b_norm_weight: TensorFixture,
        c_norm_weight: TensorFixture,
        d_skip: TensorFixture,
        mimo_x: Option<TensorFixture>,
        mimo_z: Option<TensorFixture>,
        mimo_o: Option<TensorFixture>,
        out_proj_weight: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonSequenceReferenceOutput {
        outputs: TensorFixture,
        final_angle_state: TensorFixture,
        final_ssm_state: TensorFixture,
        final_k_state: TensorFixture,
        final_v_state: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonModelSmokeReferenceInput {
        mode: String,
        official_repo: Option<String>,
        config: RustMamba3BaselineConfig,
        derived: RustMamba3DerivedShape,
        sequence_input: TensorFixture,
        angle_state: TensorFixture,
        ssm_state: TensorFixture,
        k_state: TensorFixture,
        v_state: TensorFixture,
        in_proj_weight: TensorFixture,
        dt_bias: TensorFixture,
        b_bias: TensorFixture,
        c_bias: TensorFixture,
        b_norm_weight: TensorFixture,
        c_norm_weight: TensorFixture,
        d_skip: TensorFixture,
        mimo_x: Option<TensorFixture>,
        mimo_z: Option<TensorFixture>,
        mimo_o: Option<TensorFixture>,
        out_proj_weight: TensorFixture,
        final_norm_weight: TensorFixture,
        lm_head_weight: TensorFixture,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct PythonModelSmokeReferenceOutput {
        logits: TensorFixture,
        final_angle_state: TensorFixture,
        final_ssm_state: TensorFixture,
        final_k_state: TensorFixture,
        final_v_state: TensorFixture,
    }

    #[derive(Debug)]
    struct TinyRustMamba3SmokeModel<B: burn::tensor::backend::Backend> {
        mixer: RustMamba3Mixer<B>,
        final_norm: SimpleRmsNorm<B>,
        lm_head: LanguageModelHead<B>,
    }

    impl<B: burn::tensor::backend::Backend> TinyRustMamba3SmokeModel<B> {
        fn forward_logits(&self, input: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError> {
            let mixed = self.mixer.forward_sequence(input)?;
            Ok(self.lm_head.forward(self.final_norm.forward3(mixed)))
        }
    }

    fn tensor_fixture<const D: usize>(tensor: Tensor<TestBackend, D>) -> TensorFixture {
        TensorFixture {
            shape: tensor.dims().to_vec(),
            values: tensor.to_data().to_vec::<f32>().unwrap(),
        }
    }

    fn assert_fixture_close(actual: TensorFixture, expected: TensorFixture, label: &str, tol: f32) {
        assert_eq!(actual.shape, expected.shape, "{label} shape mismatch");
        assert_eq!(
            actual.values.len(),
            expected.values.len(),
            "{label} value length mismatch"
        );
        for (index, (lhs, rhs)) in actual.values.iter().zip(expected.values.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= tol,
                "{label}[{index}] mismatch: actual={lhs} expected={rhs} tol={tol}"
            );
        }
    }

    fn structured_projection_weight_fixture(
        record: StructuredProjectionRecord<TestBackend>,
    ) -> TensorFixture {
        let weight = match record.layout_policy {
            ProjectionLayoutPolicy::InputByOutput => record.weight.val(),
            ProjectionLayoutPolicy::OutputByInput => record.weight.val().transpose(),
        };
        tensor_fixture(weight)
    }

    fn projection_weight_fixture(projection: StructuredProjection<TestBackend>) -> TensorFixture {
        structured_projection_weight_fixture(Module::into_record(projection))
    }

    fn lm_head_weight_fixture(head: LanguageModelHead<TestBackend>) -> TensorFixture {
        structured_projection_weight_fixture(Module::into_record(head))
    }

    fn optional_param_fixture(
        tensor: &Option<Param<Tensor<TestBackend, 3>>>,
    ) -> Option<TensorFixture> {
        tensor.as_ref().map(|value| tensor_fixture(value.val()))
    }

    fn deterministic_values(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| offset + scale * (index as f32 + 1.0))
            .collect()
    }

    fn deterministic_test_mixer(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> RustMamba3Mixer<TestBackend> {
        let config = RustMamba3BaselineConfig {
            d_model: 8,
            d_state: 8,
            expand: 2,
            headdim: 4,
            ngroups: 1,
            rope_fraction: RustMamba3RopeFraction::Half,
            dt_min: 1.0e-3,
            dt_max: 1.0e-1,
            dt_init_floor: 1.0e-4,
            a_floor: 1.0e-4,
            is_outproj_norm: false,
            is_mimo: true,
            mimo_rank: 2,
            chunk_size: 4,
        };
        let mut mixer = RustMamba3Mixer::new(config, device).unwrap();

        let mut in_proj_record = Module::into_record(mixer.in_proj.clone());
        in_proj_record.weight = Param::from_data(
            TensorData::new(
                deterministic_values(
                    in_proj_record.weight.val().dims()[0] * in_proj_record.weight.val().dims()[1],
                    0.01,
                    -0.3,
                ),
                in_proj_record.weight.val().dims(),
            ),
            device,
        );
        mixer.in_proj = mixer.in_proj.clone().load_record(in_proj_record);

        mixer.dt_bias = Param::from_data(
            TensorData::new(
                deterministic_values(mixer.derived_shape().nheads, 0.02, -0.1),
                [mixer.derived_shape().nheads],
            ),
            device,
        );
        mixer.b_bias = Param::from_data(
            TensorData::new(
                deterministic_values(
                    mixer.derived_shape().mimo_rank
                        * mixer.derived_shape().nheads
                        * mixer.config().d_state,
                    0.01,
                    0.2,
                ),
                [
                    mixer.derived_shape().mimo_rank,
                    mixer.derived_shape().nheads,
                    mixer.config().d_state,
                ],
            ),
            device,
        );
        mixer.c_bias = Param::from_data(
            TensorData::new(
                deterministic_values(
                    mixer.derived_shape().mimo_rank
                        * mixer.derived_shape().nheads
                        * mixer.config().d_state,
                    -0.01,
                    0.15,
                ),
                [
                    mixer.derived_shape().mimo_rank,
                    mixer.derived_shape().nheads,
                    mixer.config().d_state,
                ],
            ),
            device,
        );
        mixer.b_norm.weight = Param::from_data(
            TensorData::new(
                deterministic_values(mixer.config().d_state, 0.03, 0.8),
                [mixer.config().d_state],
            ),
            device,
        );
        mixer.c_norm.weight = Param::from_data(
            TensorData::new(
                deterministic_values(mixer.config().d_state, -0.02, 1.1),
                [mixer.config().d_state],
            ),
            device,
        );
        mixer.d_skip = Param::from_data(
            TensorData::new(
                deterministic_values(mixer.derived_shape().nheads, 0.015, 0.5),
                [mixer.derived_shape().nheads],
            ),
            device,
        );
        mixer.mimo_x = Some(Param::from_data(
            TensorData::new(
                deterministic_values(
                    mixer.derived_shape().mimo_rank
                        * mixer.derived_shape().nheads
                        * mixer.config().headdim,
                    0.01,
                    0.25,
                ),
                [
                    mixer.derived_shape().mimo_rank,
                    mixer.derived_shape().nheads,
                    mixer.config().headdim,
                ],
            ),
            device,
        ));
        mixer.mimo_z = Some(Param::from_data(
            TensorData::new(
                deterministic_values(
                    mixer.derived_shape().mimo_rank
                        * mixer.derived_shape().nheads
                        * mixer.config().headdim,
                    -0.008,
                    0.9,
                ),
                [
                    mixer.derived_shape().mimo_rank,
                    mixer.derived_shape().nheads,
                    mixer.config().headdim,
                ],
            ),
            device,
        ));
        mixer.mimo_o = Some(Param::from_data(
            TensorData::new(
                deterministic_values(
                    mixer.derived_shape().mimo_rank
                        * mixer.derived_shape().nheads
                        * mixer.config().headdim,
                    0.007,
                    -0.15,
                ),
                [
                    mixer.derived_shape().mimo_rank,
                    mixer.derived_shape().nheads,
                    mixer.config().headdim,
                ],
            ),
            device,
        ));
        let mut out_proj_record = Module::into_record(mixer.out_proj.clone());
        out_proj_record.weight = Param::from_data(
            TensorData::new(
                deterministic_values(
                    out_proj_record.weight.val().dims()[0] * out_proj_record.weight.val().dims()[1],
                    0.009,
                    -0.05,
                ),
                out_proj_record.weight.val().dims(),
            ),
            device,
        );
        mixer.out_proj = mixer.out_proj.clone().load_record(out_proj_record);
        mixer
    }

    fn deterministic_recurrent_state(
        mixer: &RustMamba3Mixer<TestBackend>,
        batch_size: usize,
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> RustMamba3RecurrentState<TestBackend> {
        RustMamba3RecurrentState {
            angle_dt_state: Tensor::from_data(
                TensorData::new(
                    deterministic_values(
                        batch_size
                            * mixer.derived_shape().nheads
                            * mixer.derived_shape().num_rope_angles,
                        0.02,
                        -0.1,
                    ),
                    [
                        batch_size,
                        mixer.derived_shape().nheads,
                        mixer.derived_shape().num_rope_angles,
                    ],
                ),
                device,
            ),
            ssm_state: Tensor::from_data(
                TensorData::new(
                    deterministic_values(
                        batch_size
                            * mixer.derived_shape().nheads
                            * mixer.config().headdim
                            * mixer.config().d_state,
                        0.01,
                        -0.2,
                    ),
                    [
                        batch_size,
                        mixer.derived_shape().nheads,
                        mixer.config().headdim,
                        mixer.config().d_state,
                    ],
                ),
                device,
            ),
            k_state: Tensor::from_data(
                TensorData::new(
                    deterministic_values(
                        batch_size
                            * mixer.derived_shape().mimo_rank
                            * mixer.derived_shape().nheads
                            * mixer.config().d_state,
                        -0.015,
                        0.3,
                    ),
                    [
                        batch_size,
                        mixer.derived_shape().mimo_rank,
                        mixer.derived_shape().nheads,
                        mixer.config().d_state,
                    ],
                ),
                device,
            ),
            v_state: Tensor::from_data(
                TensorData::new(
                    deterministic_values(
                        batch_size * mixer.derived_shape().nheads * mixer.config().headdim,
                        0.025,
                        -0.35,
                    ),
                    [
                        batch_size,
                        mixer.derived_shape().nheads,
                        mixer.config().headdim,
                    ],
                ),
                device,
            ),
        }
    }

    fn deterministic_smoke_model(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> TinyRustMamba3SmokeModel<TestBackend> {
        let mixer = deterministic_test_mixer(device);
        let mut final_norm =
            SimpleRmsNorm::new(mixer.config().d_model, DEFAULT_RUST_MAMBA3_NORM_EPS, device)
                .unwrap();
        final_norm.weight = Param::from_data(
            TensorData::new(
                deterministic_values(mixer.config().d_model, 0.02, 0.75),
                [mixer.config().d_model],
            ),
            device,
        );

        let mut lm_head = LanguageModelHeadConfig::new(mixer.config().d_model, 11)
            .with_bias(false)
            .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
            .init::<TestBackend>(device);
        let mut lm_head_record = Module::into_record(lm_head.clone());
        lm_head_record.weight = Param::from_data(
            TensorData::new(
                deterministic_values(
                    lm_head_record.weight.val().dims()[0] * lm_head_record.weight.val().dims()[1],
                    -0.01,
                    0.2,
                ),
                lm_head_record.weight.val().dims(),
            ),
            device,
        );
        lm_head = lm_head.load_record(lm_head_record);

        TinyRustMamba3SmokeModel {
            mixer,
            final_norm,
            lm_head,
        }
    }

    fn temp_json_path(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("fractal-{label}-{stamp}.json"))
    }

    fn temp_model_stem(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("fractal-{label}-{stamp}"))
    }

    fn python_reference_script_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/mamba3_pytorch_reference.py")
    }

    fn repo_root_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn python_reference_launcher_path() -> PathBuf {
        repo_root_path().join("scripts/run_mamba3_python.sh")
    }

    fn resolve_python_reference_binary() -> String {
        let launcher = python_reference_launcher_path();
        if launcher.exists() {
            return launcher.display().to_string();
        }
        panic!(
            "missing Python reference launcher at {}; restore the repo-owned wrapper or set FRACTAL_MAMBA3_PYTHON explicitly",
            launcher.display()
        );
    }

    fn resolve_official_repo_checkout() -> String {
        if let Ok(repo) = env::var("FRACTAL_MAMBA3_OFFICIAL_REPO") {
            return repo;
        }
        let fallback = repo_root_path().join("third_party/state-spaces-mamba");
        if fallback.join("mamba_ssm/modules/mamba3.py").exists() {
            return fallback.display().to_string();
        }
        panic!(
            "set FRACTAL_MAMBA3_OFFICIAL_REPO or populate the repo-owned checkout at {}",
            fallback.display()
        );
    }

    fn run_python_reference<I: Serialize, O: DeserializeOwned>(python: &str, input: &I) -> O {
        let input_path = temp_json_path("mamba3-reference-input");
        let output_path = temp_json_path("mamba3-reference-output");
        fs::write(&input_path, serde_json::to_vec(input).unwrap()).unwrap();
        let status = Command::new(python)
            .arg(python_reference_script_path())
            .arg(&input_path)
            .arg(&output_path)
            .status()
            .expect("python reference script should launch");
        assert!(status.success(), "python reference script failed");
        let output = serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
        output
    }

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
    fn phase1_siso_baseline_config_disables_mimo_ranking() {
        let config = RustMamba3BaselineConfig::phase1_siso_default(128, 4).unwrap();
        let derived = config.derived_shape().unwrap();
        assert!(!config.is_mimo);
        assert_eq!(config.mimo_rank, 1);
        assert_eq!(derived.mimo_rank, 1);
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
    fn reference_rust_siso_mamba3_model_returns_logits() {
        let device = Default::default();
        let mut variant = phase1_hybrid_attention_baseline_matrix().reference_ssm_hybrid;
        variant.label = "reference-ssm-hybrid-rust-siso".to_string();
        variant.reference_ssm_family = Some(ReferenceSsmFamily::Mamba3RustSisoV1);
        let model = build_rust_mamba3_reference_hybrid_attention_model::<TestBackend>(
            257, &variant, &device,
        )
        .unwrap();
        assert_eq!(
            model.reference_family(),
            ReferenceSsmFamily::Mamba3RustSisoV1
        );
        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [2, 8, 257]);
    }

    #[test]
    fn reference_rust_mamba3_model_round_trips_through_recording_without_logit_drift() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_rust_mamba3_reference_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.reference_ssm_hybrid,
            &device,
        )
        .unwrap();
        let input = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                vec![1i64, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8, 1, 3, 5, 7],
                [2, 8],
            ),
            &device,
        );
        let expected_logits = model.forward_logits(input.clone()).unwrap();

        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let stem = temp_model_stem("rust-mamba3-roundtrip");
        model.clone().save_file(stem.clone(), &recorder).unwrap();
        let restored = model.load_file(stem.clone(), &recorder, &device).unwrap();
        let restored_logits = restored.forward_logits(input).unwrap();
        restored_logits.into_data().assert_approx_eq::<f32>(
            &expected_logits.into_data(),
            burn::tensor::Tolerance::default(),
        );

        let _ = fs::remove_file(stem.with_extension("bin"));
    }

    #[test]
    fn rust_mamba3_mixer_allocates_official_style_recurrent_state() {
        let device = Default::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_rust_mamba3_reference_hybrid_attention_model::<TestBackend>(
            257,
            &matrix.reference_ssm_hybrid,
            &device,
        )
        .unwrap();
        let state = model.reference_layers[0]
            .mixer()
            .allocate_recurrent_state(2, &device);
        assert_eq!(state.angle_dt_state.dims(), [2, 8, 32]);
        assert_eq!(state.ssm_state.dims(), [2, 8, 32, 128]);
        assert_eq!(state.k_state.dims(), [2, 4, 8, 128]);
        assert_eq!(state.v_state.dims(), [2, 8, 32]);
    }

    #[test]
    fn rust_mamba3_step_loop_matches_sequence_scan_from_same_initial_state() {
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let seq_len = 5;
        let sequence_input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                deterministic_values(batch_size * seq_len * mixer.config().d_model, 0.015, -0.12),
                [batch_size, seq_len, mixer.config().d_model],
            ),
            &device,
        );
        let initial_state = deterministic_recurrent_state(&mixer, batch_size, &device);
        let scanned = mixer
            .scan_sequence_from_state(sequence_input.clone(), initial_state.clone())
            .unwrap();

        let mut manual_state = initial_state;
        let mut manual_outputs = Vec::with_capacity(seq_len);
        for position in 0..seq_len {
            let input_t = sequence_input
                .clone()
                .slice([
                    0..batch_size,
                    position..position + 1,
                    0..mixer.config().d_model,
                ])
                .reshape([batch_size, mixer.config().d_model]);
            let step = mixer.step(input_t, manual_state).unwrap();
            manual_outputs.push(step.output.reshape([batch_size, 1, mixer.config().d_model]));
            manual_state = step.next_state;
        }
        let manual_outputs = Tensor::cat(manual_outputs, 1);

        assert_fixture_close(
            tensor_fixture(scanned.outputs),
            tensor_fixture(manual_outputs),
            "step_vs_sequence_outputs",
            1.0e-5,
        );
        assert_fixture_close(
            tensor_fixture(scanned.final_state.angle_dt_state),
            tensor_fixture(manual_state.angle_dt_state),
            "step_vs_sequence_angle_state",
            1.0e-5,
        );
        assert_fixture_close(
            tensor_fixture(scanned.final_state.ssm_state),
            tensor_fixture(manual_state.ssm_state),
            "step_vs_sequence_ssm_state",
            1.0e-5,
        );
        assert_fixture_close(
            tensor_fixture(scanned.final_state.k_state),
            tensor_fixture(manual_state.k_state),
            "step_vs_sequence_k_state",
            1.0e-5,
        );
        assert_fixture_close(
            tensor_fixture(scanned.final_state.v_state),
            tensor_fixture(manual_state.v_state),
            "step_vs_sequence_v_state",
            1.0e-5,
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn reference_rust_mamba3_model_runs_on_metal_backend() {
        let device = <MetalTrainBackend as burn::tensor::backend::Backend>::Device::default();
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let model = build_rust_mamba3_reference_hybrid_attention_model::<MetalTrainBackend>(
            257,
            &matrix.reference_ssm_hybrid,
            &device,
        )
        .unwrap();
        let input = Tensor::<MetalTrainBackend, 2, Int>::zeros([1, 8], &device);
        assert_eq!(model.forward_logits(input).unwrap().dims(), [1, 8, 257]);
    }

    #[test]
    fn rotary_prefix_rotation_changes_only_the_target_prefix() {
        let device = Default::default();
        let state = Tensor::<TestBackend, 4>::from_data([[[[1.0, 0.0, 2.0, 0.0]]]], &device);
        let angles =
            Tensor::<TestBackend, 3>::from_data([[[core::f32::consts::FRAC_PI_2]]], &device);
        let rotated = rotate_state_prefix_pairs_last_dim_4(state, angles, 2, true);
        let values = rotated.to_data().to_vec::<f32>().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-4);
        assert!((values[1] - 1.0).abs() < 1e-4);
        assert!((values[2] - 2.0).abs() < 1e-4);
        assert!((values[3] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn rust_mamba3_pre_kernel_contract_matches_pytorch_reference() {
        let python = resolve_python_reference_binary();
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                deterministic_values(batch_size * mixer.config().d_model, 0.05, -0.25),
                [batch_size, mixer.config().d_model],
            ),
            &device,
        );
        let angle_state = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                deterministic_values(
                    batch_size
                        * mixer.derived_shape().nheads
                        * mixer.derived_shape().num_rope_angles,
                    0.04,
                    -0.2,
                ),
                [
                    batch_size,
                    mixer.derived_shape().nheads,
                    mixer.derived_shape().num_rope_angles,
                ],
            ),
            &device,
        );

        let projections = mixer.project_step(input.clone()).unwrap();
        let next_angle_state = angle_state.clone()
            + projections.angles.clone().tanh()
                * projections
                    .dt
                    .clone()
                    .reshape([batch_size, mixer.derived_shape().nheads, 1])
                * core::f64::consts::PI;
        let rotated_b = rotate_state_prefix_pairs_last_dim_4(
            projections.b.clone(),
            next_angle_state.clone(),
            mixer.derived_shape().split_tensor_size,
            !mixer.config().is_mimo,
        );
        let rotated_c = rotate_state_prefix_pairs_last_dim_4(
            projections.c.clone(),
            next_angle_state.clone(),
            mixer.derived_shape().split_tensor_size,
            !mixer.config().is_mimo,
        );

        let bundle = PythonReferenceInput {
            config: mixer.config().clone(),
            derived: mixer.derived_shape(),
            input: tensor_fixture(input),
            angle_state: tensor_fixture(angle_state),
            in_proj_weight: projection_weight_fixture(mixer.in_proj.clone()),
            dt_bias: tensor_fixture(mixer.dt_bias.val()),
            b_bias: tensor_fixture(mixer.b_bias.val()),
            c_bias: tensor_fixture(mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(mixer.c_norm.weight.val()),
        };
        let reference: PythonReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(tensor_fixture(projections.z), reference.z, "z", 1.0e-4);
        assert_fixture_close(tensor_fixture(projections.x), reference.x, "x", 1.0e-4);
        assert_fixture_close(tensor_fixture(projections.dt), reference.dt, "dt", 1.0e-4);
        assert_fixture_close(tensor_fixture(projections.a), reference.a, "a", 1.0e-4);
        assert_fixture_close(
            tensor_fixture(projections.trap),
            reference.trap,
            "trap",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(projections.angles),
            reference.angles,
            "angles",
            1.0e-4,
        );
        assert_fixture_close(tensor_fixture(projections.b), reference.b, "b", 1.0e-4);
        assert_fixture_close(tensor_fixture(projections.c), reference.c, "c", 1.0e-4);
        assert_fixture_close(
            tensor_fixture(next_angle_state),
            reference.next_angle_state,
            "next_angle_state",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(rotated_b),
            reference.rotated_b,
            "rotated_b",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(rotated_c),
            reference.rotated_c,
            "rotated_c",
            1.0e-4,
        );
    }

    #[test]
    fn rust_mamba3_step_matches_pytorch_reference() {
        let python = resolve_python_reference_binary();
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                deterministic_values(batch_size * mixer.config().d_model, 0.04, -0.3),
                [batch_size, mixer.config().d_model],
            ),
            &device,
        );
        let previous_state = deterministic_recurrent_state(&mixer, batch_size, &device);
        let step = mixer.step(input.clone(), previous_state.clone()).unwrap();
        let bundle = PythonStepReferenceInput {
            mode: "step".to_string(),
            official_repo: None,
            config: mixer.config().clone(),
            derived: mixer.derived_shape(),
            input: tensor_fixture(input),
            angle_state: tensor_fixture(previous_state.angle_dt_state),
            ssm_state: tensor_fixture(previous_state.ssm_state),
            k_state: tensor_fixture(previous_state.k_state),
            v_state: tensor_fixture(previous_state.v_state),
            in_proj_weight: projection_weight_fixture(mixer.in_proj.clone()),
            dt_bias: tensor_fixture(mixer.dt_bias.val()),
            b_bias: tensor_fixture(mixer.b_bias.val()),
            c_bias: tensor_fixture(mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(mixer.c_norm.weight.val()),
            d_skip: tensor_fixture(mixer.d_skip.val()),
            mimo_x: optional_param_fixture(&mixer.mimo_x),
            mimo_z: optional_param_fixture(&mixer.mimo_z),
            mimo_o: optional_param_fixture(&mixer.mimo_o),
            out_proj_weight: projection_weight_fixture(mixer.out_proj.clone()),
        };
        let reference: PythonStepReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(
            tensor_fixture(step.output),
            reference.output,
            "output",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.angle_dt_state),
            reference.next_angle_state,
            "next_angle_state",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.ssm_state),
            reference.next_ssm_state,
            "next_ssm_state",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.k_state),
            reference.next_k_state,
            "next_k_state",
            1.0e-4,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.v_state),
            reference.next_v_state,
            "next_v_state",
            1.0e-4,
        );
    }

    #[test]
    fn rust_mamba3_short_sequence_matches_pytorch_reference() {
        let python = resolve_python_reference_binary();
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let seq_len = 4;
        let zero_state = mixer.allocate_recurrent_state(batch_size, &device);
        let sequence_input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                deterministic_values(batch_size * seq_len * mixer.config().d_model, 0.02, -0.4),
                [batch_size, seq_len, mixer.config().d_model],
            ),
            &device,
        );
        let sequence = mixer.scan_sequence(sequence_input.clone()).unwrap();
        let bundle = PythonSequenceReferenceInput {
            mode: "sequence".to_string(),
            official_repo: None,
            config: mixer.config().clone(),
            derived: mixer.derived_shape(),
            sequence_input: tensor_fixture(sequence_input),
            angle_state: tensor_fixture(zero_state.angle_dt_state),
            ssm_state: tensor_fixture(zero_state.ssm_state),
            k_state: tensor_fixture(zero_state.k_state),
            v_state: tensor_fixture(zero_state.v_state),
            in_proj_weight: projection_weight_fixture(mixer.in_proj.clone()),
            dt_bias: tensor_fixture(mixer.dt_bias.val()),
            b_bias: tensor_fixture(mixer.b_bias.val()),
            c_bias: tensor_fixture(mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(mixer.c_norm.weight.val()),
            d_skip: tensor_fixture(mixer.d_skip.val()),
            mimo_x: optional_param_fixture(&mixer.mimo_x),
            mimo_z: optional_param_fixture(&mixer.mimo_z),
            mimo_o: optional_param_fixture(&mixer.mimo_o),
            out_proj_weight: projection_weight_fixture(mixer.out_proj.clone()),
        };
        let reference: PythonSequenceReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(
            tensor_fixture(sequence.outputs),
            reference.outputs,
            "sequence_outputs",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.angle_dt_state),
            reference.final_angle_state,
            "final_angle_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.ssm_state),
            reference.final_ssm_state,
            "final_ssm_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.k_state),
            reference.final_k_state,
            "final_k_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.v_state),
            reference.final_v_state,
            "final_v_state",
            1.0e-3,
        );
    }

    #[test]
    #[ignore = "requires official Mamba3 checkout and validates official module wiring under reference-kernel shims"]
    fn rust_mamba3_step_matches_official_module_wiring_under_reference_kernel_shims() {
        let python = resolve_python_reference_binary();
        let official_repo = resolve_official_repo_checkout();
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::new(
                deterministic_values(batch_size * mixer.config().d_model, 0.04, -0.3),
                [batch_size, mixer.config().d_model],
            ),
            &device,
        );
        let previous_state = deterministic_recurrent_state(&mixer, batch_size, &device);
        let step = mixer.step(input.clone(), previous_state.clone()).unwrap();
        let bundle = PythonStepReferenceInput {
            mode: "official-module-wiring-step".to_string(),
            official_repo: Some(official_repo),
            config: mixer.config().clone(),
            derived: mixer.derived_shape(),
            input: tensor_fixture(input),
            angle_state: tensor_fixture(previous_state.angle_dt_state),
            ssm_state: tensor_fixture(previous_state.ssm_state),
            k_state: tensor_fixture(previous_state.k_state),
            v_state: tensor_fixture(previous_state.v_state),
            in_proj_weight: projection_weight_fixture(mixer.in_proj.clone()),
            dt_bias: tensor_fixture(mixer.dt_bias.val()),
            b_bias: tensor_fixture(mixer.b_bias.val()),
            c_bias: tensor_fixture(mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(mixer.c_norm.weight.val()),
            d_skip: tensor_fixture(mixer.d_skip.val()),
            mimo_x: optional_param_fixture(&mixer.mimo_x),
            mimo_z: optional_param_fixture(&mixer.mimo_z),
            mimo_o: optional_param_fixture(&mixer.mimo_o),
            out_proj_weight: projection_weight_fixture(mixer.out_proj.clone()),
        };
        let reference: PythonStepReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(
            tensor_fixture(step.output),
            reference.output,
            "official_module_wiring_output",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.angle_dt_state),
            reference.next_angle_state,
            "official_module_wiring_next_angle_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.ssm_state),
            reference.next_ssm_state,
            "official_module_wiring_next_ssm_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.k_state),
            reference.next_k_state,
            "official_module_wiring_next_k_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(step.next_state.v_state),
            reference.next_v_state,
            "official_module_wiring_next_v_state",
            1.0e-3,
        );
    }

    #[test]
    #[ignore = "requires official Mamba3 checkout and validates official module step-loop wiring under reference-kernel shims"]
    fn rust_mamba3_short_sequence_matches_official_module_step_loop_under_reference_kernel_shims() {
        let python = resolve_python_reference_binary();
        let official_repo = resolve_official_repo_checkout();
        let device = Default::default();
        let mixer = deterministic_test_mixer(&device);
        let batch_size = 2;
        let seq_len = 4;
        let zero_state = mixer.allocate_recurrent_state(batch_size, &device);
        let sequence_input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                deterministic_values(batch_size * seq_len * mixer.config().d_model, 0.02, -0.4),
                [batch_size, seq_len, mixer.config().d_model],
            ),
            &device,
        );
        let sequence = mixer.scan_sequence(sequence_input.clone()).unwrap();
        let bundle = PythonSequenceReferenceInput {
            mode: "official-module-wiring-sequence".to_string(),
            official_repo: Some(official_repo),
            config: mixer.config().clone(),
            derived: mixer.derived_shape(),
            sequence_input: tensor_fixture(sequence_input),
            angle_state: tensor_fixture(zero_state.angle_dt_state),
            ssm_state: tensor_fixture(zero_state.ssm_state),
            k_state: tensor_fixture(zero_state.k_state),
            v_state: tensor_fixture(zero_state.v_state),
            in_proj_weight: projection_weight_fixture(mixer.in_proj.clone()),
            dt_bias: tensor_fixture(mixer.dt_bias.val()),
            b_bias: tensor_fixture(mixer.b_bias.val()),
            c_bias: tensor_fixture(mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(mixer.c_norm.weight.val()),
            d_skip: tensor_fixture(mixer.d_skip.val()),
            mimo_x: optional_param_fixture(&mixer.mimo_x),
            mimo_z: optional_param_fixture(&mixer.mimo_z),
            mimo_o: optional_param_fixture(&mixer.mimo_o),
            out_proj_weight: projection_weight_fixture(mixer.out_proj.clone()),
        };
        let reference: PythonSequenceReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(
            tensor_fixture(sequence.outputs),
            reference.outputs,
            "official_module_wiring_sequence_outputs",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.angle_dt_state),
            reference.final_angle_state,
            "official_module_wiring_final_angle_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.ssm_state),
            reference.final_ssm_state,
            "official_module_wiring_final_ssm_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.k_state),
            reference.final_k_state,
            "official_module_wiring_final_k_state",
            1.0e-3,
        );
        assert_fixture_close(
            tensor_fixture(sequence.final_state.v_state),
            reference.final_v_state,
            "official_module_wiring_final_v_state",
            1.0e-3,
        );
    }

    #[test]
    fn rust_mamba3_model_smoke_logits_match_pytorch_reference() {
        let python = resolve_python_reference_binary();
        let device = Default::default();
        let model = deterministic_smoke_model(&device);
        let batch_size = 2;
        let seq_len = 3;
        let zero_state = model.mixer.allocate_recurrent_state(batch_size, &device);
        let sequence_input = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                deterministic_values(
                    batch_size * seq_len * model.mixer.config().d_model,
                    0.03,
                    -0.2,
                ),
                [batch_size, seq_len, model.mixer.config().d_model],
            ),
            &device,
        );
        let logits = model.forward_logits(sequence_input.clone()).unwrap();
        let bundle = PythonModelSmokeReferenceInput {
            mode: "model-smoke".to_string(),
            official_repo: None,
            config: model.mixer.config().clone(),
            derived: model.mixer.derived_shape(),
            sequence_input: tensor_fixture(sequence_input),
            angle_state: tensor_fixture(zero_state.angle_dt_state),
            ssm_state: tensor_fixture(zero_state.ssm_state),
            k_state: tensor_fixture(zero_state.k_state),
            v_state: tensor_fixture(zero_state.v_state),
            in_proj_weight: projection_weight_fixture(model.mixer.in_proj.clone()),
            dt_bias: tensor_fixture(model.mixer.dt_bias.val()),
            b_bias: tensor_fixture(model.mixer.b_bias.val()),
            c_bias: tensor_fixture(model.mixer.c_bias.val()),
            b_norm_weight: tensor_fixture(model.mixer.b_norm.weight.val()),
            c_norm_weight: tensor_fixture(model.mixer.c_norm.weight.val()),
            d_skip: tensor_fixture(model.mixer.d_skip.val()),
            mimo_x: optional_param_fixture(&model.mixer.mimo_x),
            mimo_z: optional_param_fixture(&model.mixer.mimo_z),
            mimo_o: optional_param_fixture(&model.mixer.mimo_o),
            out_proj_weight: projection_weight_fixture(model.mixer.out_proj.clone()),
            final_norm_weight: tensor_fixture(model.final_norm.weight.val()),
            lm_head_weight: lm_head_weight_fixture(model.lm_head.clone()),
        };
        let reference: PythonModelSmokeReferenceOutput = run_python_reference(&python, &bundle);
        assert_fixture_close(tensor_fixture(logits), reference.logits, "logits", 1.0e-4);
    }
}
