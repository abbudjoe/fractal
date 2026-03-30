use std::{sync::Arc, thread, time::Instant};

use burn::{
    backend::{candle::CandleDevice, Autodiff, Candle},
    module::{AutodiffModule, Module, ModuleDisplay, ModuleVisitor, Param},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Bool, Element, ElementConversion, Int, Tensor, TensorData,
    },
};

use crate::{
    data_generator::{
        DatasetSplit, GeneratorConfig, SimpleHierarchicalGenerator, TaskFamily, TokenBatch,
        MIN_VOCAB_SIZE, PAD_TOKEN,
    },
    error::FractalError,
    fitness::{aggregate_results, RankedSpeciesResult, SpeciesRawMetrics},
    model::FractalModel,
    primitives::{
        b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
        b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal,
        p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot,
        p3_hierarchical::P3Hierarchical,
    },
    rule_trait::FractalRule,
};

pub type CandleBackend = Candle<f32, i64>;
pub type TrainBackend = Autodiff<CandleBackend>;

#[derive(Clone, Debug)]
pub struct TournamentConfig {
    pub dim: usize,
    pub levels: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub max_recursion_depth: usize,
    pub router_threshold: f32,
    pub batch_size: usize,
    pub train_steps_per_species: usize,
    pub eval_batches_per_family: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 8,
            max_recursion_depth: 1,
            router_threshold: 1.1,
            batch_size: 1,
            train_steps_per_species: 1,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
        }
    }
}

impl TournamentConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.dim == 0 {
            return Err(FractalError::InvalidConfig(
                "dim must be greater than zero".into(),
            ));
        }
        if self.levels < 2 {
            return Err(FractalError::InvalidConfig(
                "levels must be at least 2 for hierarchical species".into(),
            ));
        }
        if self.vocab_size < MIN_VOCAB_SIZE {
            return Err(FractalError::InvalidConfig(format!(
                "vocab_size must be at least {MIN_VOCAB_SIZE}"
            )));
        }
        if self.max_seq_len == 0 {
            return Err(FractalError::InvalidConfig(
                "max_seq_len must be greater than zero".into(),
            ));
        }
        if self.max_recursion_depth == 0 {
            return Err(FractalError::InvalidConfig(
                "max_recursion_depth must be greater than zero".into(),
            ));
        }
        if self.batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "batch_size must be greater than zero".into(),
            ));
        }
        if self.eval_batches_per_family == 0 {
            return Err(FractalError::InvalidConfig(
                "eval_batches_per_family must be greater than zero".into(),
            ));
        }
        if self.learning_rate <= 0.0 {
            return Err(FractalError::InvalidConfig(
                "learning_rate must be greater than zero".into(),
            ));
        }

        Ok(())
    }

    pub fn pressure_test() -> Self {
        Self {
            dim: 128,
            levels: 4,
            vocab_size: 64,
            max_seq_len: 128,
            max_recursion_depth: 20,
            router_threshold: 0.90,
            batch_size: 16,
            train_steps_per_species: 50,
            eval_batches_per_family: 8,
            learning_rate: 1e-3,
            seed: 42,
        }
    }

    pub fn fast_test() -> Self {
        Self {
            dim: 4,
            levels: 2,
            vocab_size: 64,
            max_seq_len: 8,
            max_recursion_depth: 1,
            router_threshold: 1.1,
            batch_size: 1,
            train_steps_per_species: 0,
            eval_batches_per_family: 1,
            learning_rate: 1e-3,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Tournament {
    pub config: TournamentConfig,
    generator: Arc<SimpleHierarchicalGenerator>,
}

impl Tournament {
    pub fn new(config: TournamentConfig) -> Result<Self, FractalError> {
        config.validate()?;
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family: 96,
            eval_examples_per_family: 32,
            seed: config.seed,
        })?;

        Ok(Self {
            config,
            generator: Arc::new(generator),
        })
    }

    pub fn run_generation(&self) -> Result<Vec<RankedSpeciesResult>, FractalError> {
        let mut handles = Vec::new();
        let dim = self.config.dim;
        let levels = self.config.levels;

        handles.push(spawn_species(
            0,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| P1Contractive::new(dim, device),
        ));
        handles.push(spawn_species(
            1,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| P2Mandelbrot::new(dim, device),
        ));
        handles.push(spawn_species(
            2,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| P3Hierarchical::new(dim, levels, device),
        ));
        handles.push(spawn_species(
            3,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| B1FractalGated::new(dim, device),
        ));
        handles.push(spawn_species(
            4,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| B2StableHierarchical::new(dim, levels, device),
        ));
        handles.push(spawn_species(
            5,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| B3FractalHierarchical::new(dim, levels, device),
        ));
        handles.push(spawn_species(
            6,
            self.config.clone(),
            Arc::clone(&self.generator),
            move |device| B4Universal::new(dim, levels, device),
        ));

        let mut metrics = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| FractalError::InvalidState("species worker panicked".into()))??;
            metrics.push(result);
        }

        Ok(aggregate_results(metrics))
    }
}

fn spawn_species<R, F>(
    index: usize,
    config: TournamentConfig,
    generator: Arc<SimpleHierarchicalGenerator>,
    factory: F,
) -> thread::JoinHandle<Result<SpeciesRawMetrics, FractalError>>
where
    R: FractalRule<TrainBackend>
        + Module<TrainBackend>
        + AutodiffModule<TrainBackend>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<TrainBackend>>::InnerModule: Module<CandleBackend> + ModuleDisplay,
    F: FnOnce(&CandleDevice) -> R + Send + 'static,
{
    thread::spawn(move || {
        let device = CandleDevice::Cpu;
        TrainBackend::seed(&device, config.seed.wrapping_add(index as u64 * 101));
        let rule = factory(&device);
        run_species(config, generator, device, rule)
    })
}

fn run_species<R>(
    config: TournamentConfig,
    generator: Arc<SimpleHierarchicalGenerator>,
    device: CandleDevice,
    rule: R,
) -> Result<SpeciesRawMetrics, FractalError>
where
    R: FractalRule<TrainBackend>
        + Module<TrainBackend>
        + AutodiffModule<TrainBackend>
        + ModuleDisplay
        + Clone
        + Send
        + std::fmt::Debug,
    <R as AutodiffModule<TrainBackend>>::InnerModule: Module<CandleBackend> + ModuleDisplay,
{
    let species = rule.name().to_string();
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
        let batch = generator.batch_for::<TrainBackend>(
            family,
            DatasetSplit::Train,
            step,
            config.batch_size,
            &device,
        );
        let loss = model.loss(&batch, &criterion, None, true)?;
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
    }

    let stability_batch = generator.batch_for::<TrainBackend>(
        TaskFamily::RecursiveSentence,
        DatasetSplit::Eval,
        0,
        config.batch_size,
        &device,
    );
    let stability_loss = model.loss(&stability_batch, &criterion, Some(20), false)?;
    let stability_grads = GradientsParams::from_grads(stability_loss.backward(), &model);
    let grad_norm_depth_20 = gradient_l2_norm(&model, &stability_grads);

    let sentence_batches = generator.eval_batches_for::<TrainBackend>(
        TaskFamily::RecursiveSentence,
        config.batch_size,
        config.eval_batches_per_family,
        &device,
    );
    let arc_batches = generator.eval_batches_for::<TrainBackend>(
        TaskFamily::ArcGrid,
        config.batch_size,
        config.eval_batches_per_family,
        &device,
    );

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

fn evaluate_perplexity<R>(
    model: &FractalModel<TrainBackend, R>,
    criterion: &burn::nn::loss::CrossEntropyLoss<TrainBackend>,
    batches: &[TokenBatch<TrainBackend>],
) -> Result<f64, FractalError>
where
    R: FractalRule<TrainBackend> + Module<TrainBackend> + Clone + std::fmt::Debug,
{
    let mut total_loss = 0.0f64;
    for batch in batches {
        let loss = model.loss(batch, criterion, None, true)?;
        total_loss += loss.into_scalar() as f64;
    }
    let mean_loss = total_loss / batches.len() as f64;
    Ok(mean_loss.exp())
}

fn evaluate_accuracy_and_speed<R>(
    model: &FractalModel<TrainBackend, R>,
    batches: &[TokenBatch<TrainBackend>],
) -> Result<(f64, f64), FractalError>
where
    R: FractalRule<TrainBackend> + Module<TrainBackend> + Clone + std::fmt::Debug,
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
