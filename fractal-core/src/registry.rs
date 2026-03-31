use std::{
    collections::HashSet,
    fmt::{Display, Formatter},
    str::FromStr,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use burn::{
    backend::{
        candle::CandleDevice,
        wgpu::{self, graphics::Metal, WgpuDevice},
        Autodiff, Candle, Wgpu as BurnWgpu,
    },
    module::{AutodiffModule, Module, ModuleDisplay, ModuleVisitor},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Bool, ElementConversion, Int, Tensor},
};

use crate::{
    data_generator::{SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN},
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

    #[cfg(feature = "cuda")]
    pub const fn cuda_default() -> Self {
        Self::CudaCandle { device_index: 0 }
    }

    pub fn is_supported_on_current_platform(&self) -> bool {
        match self {
            Self::CpuCandle => true,
            #[cfg(feature = "cuda")]
            Self::CudaCandle { .. } => cfg!(not(target_os = "macos")),
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
    P3Hierarchical,
    B2StableHierarchical,
    B1FractalGated,
    P1FractalHybrid,
    P1FractalHybridComposite,
    P1FractalHybridDynGate,
    P2Mandelbrot,
    B3FractalHierarchical,
    B4Universal,
    Ifs,
    GeneralizedMobius,
    LogisticChaoticMap,
    JuliaRecursiveEscape,
}

impl SpeciesId {
    pub const ALL: [Self; 14] = [
        Self::P1Contractive,
        Self::P3Hierarchical,
        Self::B2StableHierarchical,
        Self::B1FractalGated,
        Self::P1FractalHybrid,
        Self::P1FractalHybridComposite,
        Self::P1FractalHybridDynGate,
        Self::P2Mandelbrot,
        Self::B3FractalHierarchical,
        Self::B4Universal,
        Self::Ifs,
        Self::GeneralizedMobius,
        Self::LogisticChaoticMap,
        Self::JuliaRecursiveEscape,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::P1Contractive => "p1_contractive",
            Self::P3Hierarchical => "p3_hierarchical",
            Self::B2StableHierarchical => "b2_stable_hierarchical",
            Self::B1FractalGated => "b1_fractal_gated",
            Self::P1FractalHybrid => "p1_fractal_hybrid",
            Self::P1FractalHybridComposite => "p1_fractal_hybrid_composite",
            Self::P1FractalHybridDynGate => "p1_fractal_hybrid_dyn_gate",
            Self::P2Mandelbrot => "p2_mandelbrot",
            Self::B3FractalHierarchical => "b3_fractal_hierarchical",
            Self::B4Universal => "b4_universal",
            Self::Ifs => "ifs",
            Self::GeneralizedMobius => "generalized_mobius",
            Self::LogisticChaoticMap => "logistic_chaotic_map",
            Self::JuliaRecursiveEscape => "julia_recursive_escape",
        }
    }
}

impl Display for SpeciesId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).as_str())
    }
}

impl FromStr for SpeciesId {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "p1_contractive" => Ok(Self::P1Contractive),
            "p3_hierarchical" => Ok(Self::P3Hierarchical),
            "b2_stable_hierarchical" => Ok(Self::B2StableHierarchical),
            "b1_fractal_gated" => Ok(Self::B1FractalGated),
            "p1_fractal_hybrid" => Ok(Self::P1FractalHybrid),
            "p1_fractal_hybrid_composite" => Ok(Self::P1FractalHybridComposite),
            "p1_fractal_hybrid_dyn_gate" => Ok(Self::P1FractalHybridDynGate),
            "p2_mandelbrot" => Ok(Self::P2Mandelbrot),
            "b3_fractal_hierarchical" => Ok(Self::B3FractalHierarchical),
            "b4_universal" => Ok(Self::B4Universal),
            "ifs" => Ok(Self::Ifs),
            "generalized_mobius" => Ok(Self::GeneralizedMobius),
            "logistic_chaotic_map" => Ok(Self::LogisticChaoticMap),
            "julia_recursive_escape" => Ok(Self::JuliaRecursiveEscape),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrimitiveVariantName(&'static str);

impl PrimitiveVariantName {
    pub const fn new_unchecked(name: &'static str) -> Self {
        Self(name)
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }

    pub fn validate(self) -> Result<(), FractalError> {
        if is_valid_primitive_variant_name(self.0) {
            Ok(())
        } else {
            Err(FractalError::InvalidConfig(format!(
                "primitive variant name must match [base]_[lever-description]_v[version]: {}",
                self.0
            )))
        }
    }
}

impl Display for PrimitiveVariantName {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

pub fn is_valid_primitive_variant_name(name: &str) -> bool {
    let mut parts = name.split('_').peekable();
    let mut count = 0usize;

    while let Some(part) = parts.next() {
        count += 1;
        if part.is_empty() {
            return false;
        }
        if parts.peek().is_none() {
            return is_valid_variant_version(part) && count >= 3;
        }
        if !part
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
        {
            return false;
        }
    }

    false
}

fn is_valid_variant_version(part: &str) -> bool {
    let Some(version) = part.strip_prefix('v') else {
        return false;
    };
    !version.is_empty()
        && !version.starts_with('0')
        && version.chars().all(|ch| ch.is_ascii_digit())
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

#[derive(Clone, Copy)]
pub struct SpeciesDefinition {
    pub id: SpeciesId,
    pub variant_name: PrimitiveVariantName,
    cpu_runner: CpuRunner,
    #[cfg(feature = "cuda")]
    cuda_runner: CudaRunner,
    metal_runner: MetalRunner,
}

impl SpeciesDefinition {
    #[cfg(not(feature = "cuda"))]
    pub const fn new(
        id: SpeciesId,
        variant_name: PrimitiveVariantName,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
    ) -> Self {
        Self {
            id,
            variant_name,
            cpu_runner,
            metal_runner,
        }
    }

    #[cfg(feature = "cuda")]
    pub const fn new(
        id: SpeciesId,
        variant_name: PrimitiveVariantName,
        cpu_runner: CpuRunner,
        metal_runner: MetalRunner,
        cuda_runner: CudaRunner,
    ) -> Self {
        Self {
            id,
            variant_name,
            cpu_runner,
            cuda_runner,
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
            #[cfg(feature = "cuda")]
            ComputeBackend::CudaCandle { device_index } => {
                (self.cuda_runner)(context, cuda_device(*device_index))
            }
            ComputeBackend::MetalWgpu { device } => (self.metal_runner)(context, device.clone()),
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
    B::seed(
        &device,
        context.config.seed.wrapping_add(context.index as u64 * 101),
    );

    let batches = prepare_batches_for_run::<B>(&context.generator, &context.config, &device)?;
    let rule = factory(&context.config, &device);
    run_species_with_batches(species, context.config, device, rule, batches)
}

pub fn run_species_with_factory_candle<R, F>(
    species: SpeciesId,
    context: SpeciesRunContext,
    device: CandleDevice,
    factory: F,
) -> Result<SpeciesRawMetrics, FractalError>
where
    R: FractalRule<CpuTrainBackend>
        + Module<CpuTrainBackend>
        + AutodiffModule<CpuTrainBackend>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<CpuTrainBackend>>::InnerModule:
        Module<<CpuTrainBackend as AutodiffBackend>::InnerBackend> + ModuleDisplay,
    F: FnOnce(&TournamentConfig, &CandleDevice) -> R,
{
    CpuTrainBackend::seed(
        &device,
        context.config.seed.wrapping_add(context.index as u64 * 101),
    );

    let batches = prepare_candle_batches_for_run(&context.generator, &context.config, &device)?;
    let rule = factory(&context.config, &device);
    run_species_with_batches(species, context.config, device, rule, batches)
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

    wgpu::init_setup::<Metal>(device, Default::default());
    initialized.insert(device.clone());
}

struct RunBatches<B: AutodiffBackend> {
    train_sentence: Vec<TokenBatch<B>>,
    train_arc: Vec<TokenBatch<B>>,
    eval_sentence: Vec<TokenBatch<B>>,
    eval_arc: Vec<TokenBatch<B>>,
}

fn run_species_with_batches<B, R>(
    species: SpeciesId,
    config: TournamentConfig,
    device: B::Device,
    rule: R,
    batches: RunBatches<B>,
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

    log_species_phase_start(
        species,
        "train",
        &[
            format!("steps={}", config.train_steps_per_species),
            format!("train_batch={}", config.train_batch_size),
        ],
    );
    let train_started = Instant::now();
    for step in 0..config.train_steps_per_species {
        let train_batches = if step % 2 == 0 {
            &batches.train_sentence
        } else {
            &batches.train_arc
        };
        let batch = &train_batches[step % train_batches.len()];
        let loss = model.loss(batch, &criterion, None, true)?;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        let completed_step = step + 1;
        if should_log_training_checkpoint(completed_step, config.train_steps_per_species) {
            log_species_phase_progress(
                species,
                "train",
                completed_step,
                config.train_steps_per_species,
                train_started.elapsed(),
            );
        }
    }
    log_species_phase_done(species, "train", train_started.elapsed());

    log_species_phase_start(
        species,
        "stability",
        &[format!("depth={}", config.stability_depth)],
    );
    let stability_started = Instant::now();
    let stability_loss = model.loss(
        &batches.eval_sentence[0],
        &criterion,
        Some(config.stability_depth),
        false,
    )?;
    let stability_grads = GradientsParams::from_grads(stability_loss.backward(), &model);
    let grad_norm_depth_20 = gradient_l2_norm(&model, &stability_grads);
    log_species_phase_done(species, "stability", stability_started.elapsed());

    log_species_phase_start(
        species,
        "perplexity",
        &[format!("batches={}", batches.eval_sentence.len())],
    );
    let perplexity_started = Instant::now();
    let long_context_perplexity = evaluate_perplexity(&model, &criterion, &batches.eval_sentence)?;
    log_species_phase_done(species, "perplexity", perplexity_started.elapsed());

    log_species_phase_start(
        species,
        "arc_speed",
        &[format!("batches={}", batches.eval_arc.len())],
    );
    let accuracy_started = Instant::now();
    let (arc_accuracy, tokens_per_sec) = evaluate_accuracy_and_speed(&model, &batches.eval_arc)?;
    log_species_phase_done(species, "arc_speed", accuracy_started.elapsed());

    Ok(SpeciesRawMetrics {
        species,
        grad_norm_depth_20,
        long_context_perplexity,
        arc_accuracy,
        tokens_per_sec,
    })
}

fn prepare_batches_for_run<B: AutodiffBackend>(
    generator: &SimpleHierarchicalGenerator,
    config: &TournamentConfig,
    device: &B::Device,
) -> Result<RunBatches<B>, FractalError> {
    Ok(RunBatches {
        train_sentence: generator.train_batches_for::<B>(
            TaskFamily::RecursiveSentence,
            config.train_batch_size,
            device,
        )?,
        train_arc: generator.train_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.train_batch_size,
            device,
        )?,
        eval_sentence: generator.eval_batches_for::<B>(
            TaskFamily::RecursiveSentence,
            config.eval_batch_size,
            config.eval_batches_per_family,
            device,
        )?,
        eval_arc: generator.eval_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.eval_batch_size,
            config.eval_batches_per_family,
            device,
        )?,
    })
}

fn prepare_candle_batches_for_run(
    generator: &SimpleHierarchicalGenerator,
    config: &TournamentConfig,
    device: &CandleDevice,
) -> Result<RunBatches<CpuTrainBackend>, FractalError> {
    let staging_device = CandleDevice::Cpu;
    let move_batches = |batches: Vec<TokenBatch<CpuTrainBackend>>| {
        batches
            .into_iter()
            .map(|batch| batch.to_device(device))
            .collect::<Vec<_>>()
    };

    Ok(RunBatches {
        train_sentence: move_batches(generator.train_batches_for::<CpuTrainBackend>(
            TaskFamily::RecursiveSentence,
            config.train_batch_size,
            &staging_device,
        )?),
        train_arc: move_batches(generator.train_batches_for::<CpuTrainBackend>(
            TaskFamily::ArcGrid,
            config.train_batch_size,
            &staging_device,
        )?),
        eval_sentence: move_batches(generator.eval_batches_for::<CpuTrainBackend>(
            TaskFamily::RecursiveSentence,
            config.eval_batch_size,
            config.eval_batches_per_family,
            &staging_device,
        )?),
        eval_arc: move_batches(generator.eval_batches_for::<CpuTrainBackend>(
            TaskFamily::ArcGrid,
            config.eval_batch_size,
            config.eval_batches_per_family,
            &staging_device,
        )?),
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
        let predictions = flat_logits.argmax(1).reshape([batch_size * seq_len]);
        let valid_mask = flat_targets
            .clone()
            .equal_elem((PAD_TOKEN as i64).elem::<B::IntElem>())
            .bool_not();
        let correct_mask = predictions.equal(flat_targets).bool_and(valid_mask.clone());

        correct += correct_mask.int().sum().into_scalar().elem::<i64>() as usize;
        total += valid_mask.int().sum().into_scalar().elem::<i64>() as usize;
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

const TRAIN_PROGRESS_TARGET_EVENTS: usize = 4;

pub(crate) fn should_log_training_checkpoint(completed_step: usize, total_steps: usize) -> bool {
    if total_steps == 0 || completed_step == 0 {
        return false;
    }

    if completed_step == total_steps {
        return total_steps > 0 && completed_step == total_steps;
    }

    completed_step.is_multiple_of(training_progress_interval(total_steps))
}

pub(crate) fn training_progress_interval(total_steps: usize) -> usize {
    total_steps.max(1).div_ceil(TRAIN_PROGRESS_TARGET_EVENTS)
}

fn log_species_phase_start(species: SpeciesId, phase: &str, details: &[String]) {
    let suffix = if details.is_empty() {
        String::new()
    } else {
        format!(" {}", details.join(" "))
    };
    println!("[phase:start] {species} {phase}{suffix}");
}

fn log_species_phase_progress(
    species: SpeciesId,
    phase: &str,
    completed: usize,
    total: usize,
    elapsed: Duration,
) {
    println!(
        "[phase:progress] {species} {phase} {completed}/{total} elapsed={:.1}s",
        elapsed.as_secs_f64()
    );
}

fn log_species_phase_done(species: SpeciesId, phase: &str, elapsed: Duration) {
    println!(
        "[phase:done] {species} {phase} elapsed={:.1}s",
        elapsed.as_secs_f64()
    );
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
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
                let value = (grad.clone() * grad).sum().into_scalar().elem::<f64>();
                self.sum_sq += value;
            }
        }

        fn visit_int<const D: usize>(&mut self, _param: &burn::module::Param<Tensor<B, D, Int>>) {}

        fn visit_bool<const D: usize>(&mut self, _param: &burn::module::Param<Tensor<B, D, Bool>>) {
        }
    }

    let mut collector = Collector::<B> {
        grads,
        sum_sq: 0.0,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut collector);
    collector.sum_sq.sqrt()
}
