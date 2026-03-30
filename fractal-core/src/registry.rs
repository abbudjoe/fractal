use std::{
    collections::HashSet,
    fmt::{Display, Formatter},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use burn::{
    backend::{
        candle::CandleDevice,
        wgpu::{self, WgpuDevice},
        Autodiff, Candle, Wgpu as BurnWgpu,
    },
    module::{AutodiffModule, Module, ModuleDisplay, ModuleVisitor, ParamId},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Bool, Element, ElementConversion, Int, Tensor, TensorData},
};
use burn_mlx::Mlx;
pub use burn_mlx::MlxDevice;

use crate::{
    data_generator::{
        DatasetSplit, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    lifecycle::TournamentConfig,
    model::FractalModel,
    rule_trait::FractalRule,
};

pub type CpuBackend = Candle<f32, i64>;
pub type CpuTrainBackend = Autodiff<CpuBackend>;
pub type MetalBackend = BurnWgpu<f32, i32>;
pub type MetalTrainBackend = Autodiff<MetalBackend>;
pub type MlxBackend = Mlx;
pub type MlxTrainBackend = Autodiff<MlxBackend>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeBackend {
    CpuCandle,
    #[cfg(feature = "cuda")]
    CudaCandle {
        device_index: usize,
    },
    MetalWgpu {
        device: WgpuDevice,
    },
    Mlx {
        device: MlxDevice,
    },
}

impl ComputeBackend {
    pub fn default_for_current_platform() -> Self {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Self::mlx_default()
        } else if cfg!(target_os = "macos") {
            Self::metal_default()
        } else {
            Self::CpuCandle
        }
    }

    pub fn metal_default() -> Self {
        Self::MetalWgpu {
            device: WgpuDevice::DefaultDevice,
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn cuda_default() -> Self {
        Self::CudaCandle { device_index: 0 }
    }

    pub fn mlx_default() -> Self {
        Self::Mlx {
            device: MlxDevice::Gpu,
        }
    }

    pub fn is_supported_on_current_platform(&self) -> bool {
        match self {
            Self::CpuCandle => true,
            #[cfg(feature = "cuda")]
            Self::CudaCandle { .. } => cfg!(not(target_os = "macos")),
            Self::MetalWgpu { .. } => cfg!(target_os = "macos"),
            Self::Mlx { .. } => cfg!(all(target_os = "macos", target_arch = "aarch64")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SpeciesId {
    P1Contractive,
    P2Mandelbrot,
    P3Hierarchical,
    B1FractalGated,
    B2StableHierarchical,
    B3FractalHierarchical,
    B4Universal,
}

impl SpeciesId {
    pub const ALL: [Self; 7] = [
        Self::P1Contractive,
        Self::P2Mandelbrot,
        Self::P3Hierarchical,
        Self::B1FractalGated,
        Self::B2StableHierarchical,
        Self::B3FractalHierarchical,
        Self::B4Universal,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1Contractive => "p1_contractive",
            Self::P2Mandelbrot => "p2_mandelbrot",
            Self::P3Hierarchical => "p3_hierarchical",
            Self::B1FractalGated => "b1_fractal_gated",
            Self::B2StableHierarchical => "b2_stable_hierarchical",
            Self::B3FractalHierarchical => "b3_fractal_hierarchical",
            Self::B4Universal => "b4_universal",
        }
    }
}

impl Display for SpeciesId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).as_str())
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesRunContext {
    pub index: usize,
    pub config: TournamentConfig,
    pub generator: Arc<SimpleHierarchicalGenerator>,
}

type CpuRunner = fn(SpeciesRunContext) -> Result<SpeciesRawMetrics, FractalError>;
#[cfg(feature = "cuda")]
type CudaRunner = fn(SpeciesRunContext, CandleDevice) -> Result<SpeciesRawMetrics, FractalError>;
type MetalRunner = fn(SpeciesRunContext, WgpuDevice) -> Result<SpeciesRawMetrics, FractalError>;
type MlxRunner = fn(SpeciesRunContext, MlxDevice) -> Result<SpeciesRawMetrics, FractalError>;

#[derive(Clone, Copy)]
pub struct SpeciesDefinition {
    pub id: SpeciesId,
    cpu_runner: CpuRunner,
    #[cfg(feature = "cuda")]
    cuda_runner: CudaRunner,
    metal_runner: MetalRunner,
    mlx_runner: MlxRunner,
}

impl SpeciesDefinition {
    #[cfg(not(feature = "cuda"))]
    pub const fn new(
        id: SpeciesId,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
        mlx_runner: MlxRunner,
    ) -> Self {
        Self {
            id,
            cpu_runner,
            metal_runner,
            mlx_runner,
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn new(
        id: SpeciesId,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
        mlx_runner: MlxRunner,
        cuda_runner: CudaRunner,
    ) -> Self {
        Self {
            id,
            cpu_runner,
            cuda_runner,
            metal_runner,
            mlx_runner,
        }
    }

    pub fn run(
        &self,
        context: SpeciesRunContext,
        backend: &ComputeBackend,
    ) -> Result<SpeciesRawMetrics, FractalError> {
        match backend {
            ComputeBackend::CpuCandle => (self.cpu_runner)(context),
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaCandle { device_index } => {
                (self.cuda_runner)(context, cuda_device(*device_index))
            }
            ComputeBackend::MetalWgpu { device } => (self.metal_runner)(context, device.clone()),
            ComputeBackend::Mlx { device } => (self.mlx_runner)(context, *device),
        }
    }
}

pub fn cpu_device() -> CandleDevice {
    CandleDevice::Cpu
}

#[cfg(feature = "cuda")]
pub fn cuda_device(index: usize) -> CandleDevice {
    CandleDevice::cuda(index)
}

pub fn run_species_with_factory<B, R, F>(
    species: SpeciesId,
    context: SpeciesRunContext,
    device: B::Device,
    factory: F,
) -> Result<SpeciesRawMetrics, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B>
        + Module<B>
        + AutodiffModule<B>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
    F: FnOnce(&TournamentConfig, &B::Device) -> R,
{
    B::seed(context.config.seed.wrapping_add(context.index as u64 * 101));

    let rule = factory(&context.config, &device);
    run_species(species, context, device, rule)
}

pub fn initialize_metal_runtime(device: &WgpuDevice) {
    static INITIALIZED_DEVICES: OnceLock<Mutex<HashSet<WgpuDevice>>> = OnceLock::new();

    let devices = INITIALIZED_DEVICES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut initialized = match devices.lock() {
        Ok(initialized) => initialized,
        Err(poisoned) => poisoned.into_inner(),
    };
    if initialized.contains(device) {
        return;
    }

    wgpu::init_setup::<wgpu::Metal>(device, Default::default());
    initialized.insert(device.clone());
}

fn run_species<B, R>(
    species: SpeciesId,
    context: SpeciesRunContext,
    device: B::Device,
    rule: R,
) -> Result<SpeciesRawMetrics, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B>
        + Module<B>
        + AutodiffModule<B>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend> + ModuleDisplay,
{
    let config = context.config;
    let generator = context.generator;
    let mut model = FractalModel::new(
        config.vocab_size,
        config.dim,
        config.max_recursion_depth,
        config.router_threshold,
        PAD_TOKEN,
        rule,
        &device,
    );
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN]))
        .init(&device);
    let mut optimizer = AdamConfig::new().init();

    for step in 0..config.train_steps_per_species {
        let family = if step % 2 == 0 {
            TaskFamily::RecursiveSentence
        } else {
            TaskFamily::ArcGrid
        };
        let batch = generator.batch_for::<B>(
            family,
            DatasetSplit::Train,
            step,
            config.batch_size,
            &device,
        )?;
        let loss = model.loss(&batch, &criterion, None, true)?;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
    }

    let stability_batch = generator.batch_for::<B>(
        TaskFamily::RecursiveSentence,
        DatasetSplit::Eval,
        0,
        config.batch_size,
        &device,
    )?;
    let stability_loss = model.loss(&stability_batch, &criterion, Some(20), false)?;
    let stability_grads = GradientsParams::from_grads(stability_loss.backward(), &model);
    let grad_norm_depth_20 = gradient_l2_norm(&model, &stability_grads);

    let sentence_batches = generator.eval_batches_for::<B>(
        TaskFamily::RecursiveSentence,
        config.batch_size,
        config.eval_batches_per_family,
        &device,
    )?;
    let arc_batches = generator.eval_batches_for::<B>(
        TaskFamily::ArcGrid,
        config.batch_size,
        config.eval_batches_per_family,
        &device,
    )?;

    let long_context_perplexity = evaluate_perplexity(&model, &criterion, &sentence_batches)?;
    let (arc_accuracy, tokens_per_sec) = evaluate_accuracy_and_speed(&model, &arc_batches)?;

    Ok(SpeciesRawMetrics {
        species,
        grad_norm_depth_20,
        long_context_perplexity,
        arc_accuracy,
        tokens_per_sec,
    })
}

fn evaluate_perplexity<B, R>(
    model: &FractalModel<B, R>,
    criterion: &CrossEntropyLoss<B>,
    batches: &[TokenBatch<B>],
) -> Result<f64, FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + Clone + std::fmt::Debug,
{
    let mut total_loss = 0.0f64;
    for batch in batches {
        let loss = model.loss(batch, criterion, None, true)?;
        total_loss += loss.into_scalar().elem::<f64>();
    }
    let mean_loss = total_loss / batches.len() as f64;
    Ok(mean_loss.exp())
}

fn evaluate_accuracy_and_speed<B, R>(
    model: &FractalModel<B, R>,
    batches: &[TokenBatch<B>],
) -> Result<(f64, f64), FractalError>
where
    B: AutodiffBackend,
    R: FractalRule<B> + Module<B> + Clone + std::fmt::Debug,
{
    let mut correct = 0usize;
    let mut total = 0usize;
    let start = Instant::now();

    for batch in batches {
        let logits = model.forward_tokens(batch.input_ids.clone())?;
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let flat_logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let flat_targets = batch.target_ids.clone().reshape([batch_size * seq_len]);

        let logits_data = tensor_data_to_vec::<f32>(flat_logits.into_data(), "logits")?;
        let targets_data = tensor_data_to_vec::<B::IntElem>(flat_targets.into_data(), "targets")?;

        for (row_index, target) in targets_data.iter().enumerate() {
            let target = (*target).elem::<i64>();
            if target == PAD_TOKEN as i64 {
                continue;
            }
            let row_start = row_index * vocab_size;
            let row = &logits_data[row_start..row_start + vocab_size];
            let prediction = row
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index as i64)
                .unwrap_or_default();

            if prediction == target {
                correct += 1;
            }
            total += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let accuracy = if total == 0 {
        0.0
    } else {
        correct as f64 / total as f64
    };
    let tokens_per_sec = total as f64 / elapsed;

    Ok((accuracy, tokens_per_sec))
}

fn tensor_data_to_vec<E: Element>(
    data: TensorData,
    label: &'static str,
) -> Result<Vec<E>, FractalError> {
    data.to_vec::<E>()
        .map_err(|err| FractalError::InvalidState(format!("failed to extract {label}: {err:?}")))
}

fn gradient_l2_norm<M, B>(module: &M, grads: &GradientsParams) -> f64
where
    M: Module<B>,
    B: AutodiffBackend,
{
    struct Collector<'a, B: AutodiffBackend> {
        grads: &'a GradientsParams,
        sum_sq: f64,
        _marker: std::marker::PhantomData<B>,
    }

    impl<'a, B: AutodiffBackend> ModuleVisitor<B> for Collector<'a, B> {
        fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
            if let Some(grad) = self.grads.get::<B::InnerBackend, D>(id) {
                let value = (grad.clone() * grad).sum().into_scalar().elem::<f64>();
                self.sum_sq += value;
            }
        }

        fn visit_int<const D: usize>(&mut self, _id: ParamId, _tensor: &Tensor<B, D, Int>) {}

        fn visit_bool<const D: usize>(&mut self, _id: ParamId, _tensor: &Tensor<B, D, Bool>) {}
    }

    let mut collector = Collector::<B> {
        grads,
        sum_sq: 0.0,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut collector);
    collector.sum_sq.sqrt()
}
