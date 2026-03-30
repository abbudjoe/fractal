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
        Autodiff, Candle, Metal as BurnMetal,
    },
    module::{AutodiffModule, Module, ModuleDisplay, ModuleVisitor, Param},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::AutodiffBackend, Bool, Element, ElementConversion, Int, Tensor, TensorData},
};

use crate::{
    data_generator::{
        DatasetSplit, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    lifecycle::TournamentConfig,
    model::FractalModel,
    primitives::{
        b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
        b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal,
        p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot,
        p3_hierarchical::P3Hierarchical,
    },
    rule_trait::FractalRule,
};

pub type CpuBackend = Candle<f32, i64>;
pub type CpuTrainBackend = Autodiff<CpuBackend>;
pub type MetalBackend = BurnMetal<f32, i64>;
pub type MetalTrainBackend = Autodiff<MetalBackend>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeBackend {
    CpuCandle,
    MetalWgpu { device: WgpuDevice },
}

impl ComputeBackend {
    pub fn default_for_current_platform() -> Self {
        if cfg!(target_os = "macos") {
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

    pub fn is_supported_on_current_platform(&self) -> bool {
        match self {
            Self::CpuCandle => true,
            Self::MetalWgpu { .. } => cfg!(target_os = "macos"),
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
type MetalRunner = fn(SpeciesRunContext, WgpuDevice) -> Result<SpeciesRawMetrics, FractalError>;

#[derive(Clone, Copy)]
pub struct SpeciesDefinition {
    pub id: SpeciesId,
    cpu_runner: CpuRunner,
    metal_runner: MetalRunner,
}

impl SpeciesDefinition {
    pub const fn new(id: SpeciesId, cpu_runner: CpuRunner, metal_runner: MetalRunner) -> Self {
        Self {
            id,
            cpu_runner,
            metal_runner,
        }
    }

    pub fn run(
        &self,
        context: SpeciesRunContext,
        backend: &ComputeBackend,
    ) -> Result<SpeciesRawMetrics, FractalError> {
        match backend {
            ComputeBackend::CpuCandle => (self.cpu_runner)(context),
            ComputeBackend::MetalWgpu { device } => (self.metal_runner)(context, device.clone()),
        }
    }
}

macro_rules! define_flat_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $species:ident, $rule:ident) => {
        fn $cpu_fn(context: SpeciesRunContext) -> Result<SpeciesRawMetrics, FractalError> {
            let device = CandleDevice::Cpu;
            run_species_with_factory::<CpuTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, device),
            )
        }

        fn $metal_fn(
            context: SpeciesRunContext,
            device: WgpuDevice,
        ) -> Result<SpeciesRawMetrics, FractalError> {
            initialize_metal_runtime(&device);
            run_species_with_factory::<MetalTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, device),
            )
        }
    };
}

macro_rules! define_hierarchical_species_runner {
    ($cpu_fn:ident, $metal_fn:ident, $species:ident, $rule:ident) => {
        fn $cpu_fn(context: SpeciesRunContext) -> Result<SpeciesRawMetrics, FractalError> {
            let device = CandleDevice::Cpu;
            run_species_with_factory::<CpuTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, config.levels, device),
            )
        }

        fn $metal_fn(
            context: SpeciesRunContext,
            device: WgpuDevice,
        ) -> Result<SpeciesRawMetrics, FractalError> {
            initialize_metal_runtime(&device);
            run_species_with_factory::<MetalTrainBackend, _, _>(
                SpeciesId::$species,
                context,
                device,
                |config, device| $rule::new(config.dim, config.levels, device),
            )
        }
    };
}

define_flat_species_runner!(run_p1_cpu, run_p1_metal, P1Contractive, P1Contractive);
define_flat_species_runner!(run_p2_cpu, run_p2_metal, P2Mandelbrot, P2Mandelbrot);
define_hierarchical_species_runner!(run_p3_cpu, run_p3_metal, P3Hierarchical, P3Hierarchical);
define_flat_species_runner!(run_b1_cpu, run_b1_metal, B1FractalGated, B1FractalGated);
define_hierarchical_species_runner!(
    run_b2_cpu,
    run_b2_metal,
    B2StableHierarchical,
    B2StableHierarchical
);
define_hierarchical_species_runner!(
    run_b3_cpu,
    run_b3_metal,
    B3FractalHierarchical,
    B3FractalHierarchical
);
define_hierarchical_species_runner!(run_b4_cpu, run_b4_metal, B4Universal, B4Universal);

pub const SPECIES_REGISTRY: [SpeciesDefinition; 7] = [
    SpeciesDefinition::new(SpeciesId::P1Contractive, run_p1_cpu, run_p1_metal),
    SpeciesDefinition::new(SpeciesId::P2Mandelbrot, run_p2_cpu, run_p2_metal),
    SpeciesDefinition::new(SpeciesId::P3Hierarchical, run_p3_cpu, run_p3_metal),
    SpeciesDefinition::new(SpeciesId::B1FractalGated, run_b1_cpu, run_b1_metal),
    SpeciesDefinition::new(SpeciesId::B2StableHierarchical, run_b2_cpu, run_b2_metal),
    SpeciesDefinition::new(SpeciesId::B3FractalHierarchical, run_b3_cpu, run_b3_metal),
    SpeciesDefinition::new(SpeciesId::B4Universal, run_b4_cpu, run_b4_metal),
];

pub fn species_registry() -> &'static [SpeciesDefinition] {
    &SPECIES_REGISTRY
}

fn run_species_with_factory<B, R, F>(
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
    B::seed(
        &device,
        context.config.seed.wrapping_add(context.index as u64 * 101),
    );

    let rule = factory(&context.config, &device);
    run_species(species, context, device, rule)
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
        let targets_data = tensor_data_to_vec::<i64>(flat_targets.into_data(), "targets")?;

        for (row_index, target) in targets_data.iter().enumerate() {
            if *target == PAD_TOKEN as i64 {
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

            if prediction == *target {
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
        .map_err(|err| FractalError::InvalidState(format!("failed to extract {label}: {err}")))
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
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
                let value = grad.square().sum().into_scalar().elem::<f64>();
                self.sum_sq += value;
            }
        }

        fn visit_int<const D: usize>(&mut self, _param: &Param<Tensor<B, D, Int>>) {}

        fn visit_bool<const D: usize>(&mut self, _param: &Param<Tensor<B, D, Bool>>) {}
    }

    let mut collector = Collector::<B> {
        grads,
        sum_sq: 0.0,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut collector);
    collector.sum_sq.sqrt()
}

fn initialize_metal_runtime(device: &WgpuDevice) {
    static INITIALIZED_DEVICES: OnceLock<Mutex<HashSet<WgpuDevice>>> = OnceLock::new();

    let devices = INITIALIZED_DEVICES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut initialized = match devices.lock() {
        Ok(initialized) => initialized,
        Err(poisoned) => poisoned.into_inner(),
    };
    if initialized.contains(device) {
        return;
    }

    wgpu::init_setup::<wgpu::graphics::Metal>(device, Default::default());
    initialized.insert(device.clone());
}
