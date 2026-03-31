use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};

#[cfg(feature = "cuda")]
use burn::backend::candle::CandleDevice;
use burn::{
    backend::{wgpu::WgpuDevice, Candle},
    module::{Module, Param},
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use crate::{
    data_generator::{
        DatasetSplit, GeneratorConfig, GeneratorDepthConfig, SimpleHierarchicalGenerator,
        TaskFamily, MIN_SEQUENCE_LEN, MIN_VOCAB_SIZE, PAD_TOKEN,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    lifecycle::{
        Tournament, TournamentConfig, TournamentPreset, TournamentProgressEvent, TournamentSequence,
    },
    model::FractalModel,
    primitives::complex_square,
    registry::{ComputeBackend, ExecutionMode, SpeciesDefinition, SpeciesId, SpeciesRunContext},
    router::EarlyExitRouter,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

type TestBackend = Candle<f32, i64>;
static APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);
static CONCURRENT_SPECIES: AtomicUsize = AtomicUsize::new(0);
static MAX_CONCURRENT_SPECIES: AtomicUsize = AtomicUsize::new(0);

#[test]
fn complex_square_matches_hand_computed_values() {
    let device = Default::default();
    let tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], [1, 4]),
        &device,
    );
    let squared = complex_square(tensor).into_data().to_vec::<f32>().unwrap();

    assert_eq!(squared, vec![-12.0, -16.0, 16.0, 30.0]);
}

#[test]
fn router_exit_mask_is_per_sample() {
    let device = Default::default();
    let mut router = EarlyExitRouter::<TestBackend>::new(1, 0.6, &device);
    router.projection.weight = Param::from_data(TensorData::from([[1.0f32]]), &device);
    router.projection.bias = Some(Param::from_data(TensorData::from([0.0f32]), &device));

    let readout = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0f32, 0.0, -1.0], [3, 1]),
        &device,
    );
    let exit_mask = router
        .exit_mask(readout.clone())
        .into_data()
        .to_vec::<bool>()
        .unwrap();

    assert_eq!(readout.dims(), [3, 1]);
    assert_eq!(exit_mask, vec![true, false, false]);
}

#[test]
fn tournament_returns_one_result_per_registered_species() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let results = tournament.run_generation(&test_species_registry()).unwrap();

    assert_eq!(results.len(), SpeciesId::ALL.len());
}

#[test]
fn tournament_parallel_mode_returns_one_result_per_registered_species() {
    let tournament =
        Tournament::new(TournamentConfig::fast_test().with_execution_mode(ExecutionMode::Parallel))
            .unwrap();
    let results = tournament.run_generation(&test_species_registry()).unwrap();

    assert_eq!(results.len(), SpeciesId::ALL.len());
}

#[test]
fn first_run_sequence_stages_are_ordered_for_safe_ramp_up() {
    assert_eq!(
        TournamentSequence::FirstRun.stages(),
        &[
            TournamentPreset::FastTest,
            TournamentPreset::ResearchMedium,
            TournamentPreset::PressureTest,
        ]
    );
}

#[test]
fn research_medium_preset_targets_single_gpu_sequential_run() {
    let config = TournamentPreset::ResearchMedium.config();

    assert_eq!(config.dim, 16);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 32);
    assert_eq!(config.max_recursion_depth, 4);
    assert_eq!(config.stability_depth, 4);
    assert_eq!(config.train_batch_size, 2);
    assert_eq!(config.eval_batch_size, 2);
    assert_eq!(config.train_steps_per_species, 5);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.execution_mode, ExecutionMode::Sequential);
    assert_eq!(config.parallelism, 4);
}

#[test]
fn challenger_lane_preset_targets_midweight_single_gpu_run() {
    let config = TournamentPreset::ChallengerLane.config();

    assert_eq!(config.dim, 96);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 64);
    assert_eq!(config.max_recursion_depth, 6);
    assert_eq!(config.stability_depth, 6);
    assert_eq!(config.train_batch_size, 8);
    assert_eq!(config.eval_batch_size, 4);
    assert_eq!(config.train_steps_per_species, 12);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.execution_mode, ExecutionMode::Sequential);
    assert_eq!(config.parallelism, 4);
}

#[test]
fn minimal_proving_ground_preset_targets_reintroduced_squaring_family() {
    let config = TournamentPreset::MinimalProvingGround.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 96);
    assert_eq!(config.max_recursion_depth, 8);
    assert_eq!(config.stability_depth, 8);
    assert_eq!(config.train_batch_size, 8);
    assert_eq!(config.eval_batch_size, 8);
    assert_eq!(config.train_steps_per_species, 30);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.execution_mode, ExecutionMode::Sequential);
}

#[test]
fn bullpen_polish_preset_targets_top_candidates_with_harder_recursion() {
    let config = TournamentPreset::BullpenPolish.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 96);
    assert_eq!(config.max_recursion_depth, 8);
    assert_eq!(config.stability_depth, 8);
    assert_eq!(config.train_batch_size, 16);
    assert_eq!(config.eval_batch_size, 8);
    assert_eq!(config.train_steps_per_species, 24);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.generator_depth_config.sentence_eval_max_depth, 10);
}

#[test]
fn candidate_stress_preset_targets_single_species_full_stress_run() {
    let config = TournamentPreset::CandidateStress.config();

    assert_eq!(config.dim, 192);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 128);
    assert_eq!(config.max_recursion_depth, 16);
    assert_eq!(config.stability_depth, 20);
    assert_eq!(config.train_batch_size, 8);
    assert_eq!(config.eval_batch_size, 4);
    assert_eq!(config.train_steps_per_species, 120);
    assert_eq!(config.eval_batches_per_family, 4);
    assert_eq!(config.generator_depth_config.sentence_eval_max_depth, 12);
}

#[test]
fn generation_four_preset_uses_pressure_test_shape() {
    let config = TournamentPreset::GenerationFour.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.levels, 4);
    assert_eq!(config.max_seq_len, 128);
    assert_eq!(config.max_recursion_depth, 20);
    assert_eq!(config.stability_depth, 20);
    assert_eq!(config.train_batch_size, 16);
    assert_eq!(config.eval_batch_size, 8);
    assert_eq!(config.train_steps_per_species, 50);
    assert_eq!(config.eval_batches_per_family, 8);
    assert_eq!(config.execution_mode, ExecutionMode::Sequential);
    assert_eq!(config.parallelism, 4);

    #[cfg(feature = "cuda")]
    assert!(matches!(
        config.execution_backend,
        ComputeBackend::CudaCandle { .. }
    ));
}

#[test]
fn bullpen_polish_only_applies_documented_temporary_ifs_override() {
    let config = TournamentPreset::BullpenPolish.config();
    let ifs = config.effective_for_species(SpeciesId::Ifs);
    let mobius = config.effective_for_species(SpeciesId::GeneralizedMobius);

    assert_eq!(ifs.train_batch_size, 8);
    assert_eq!(ifs.eval_batch_size, 4);
    assert_eq!(ifs.train_steps_per_species, 16);
    assert_eq!(mobius.train_batch_size, config.train_batch_size);
    assert_eq!(mobius.eval_batch_size, config.eval_batch_size);
    assert_eq!(
        mobius.train_steps_per_species,
        config.train_steps_per_species
    );
}

#[test]
fn default_config_uses_a_supported_backend() {
    let config = TournamentConfig::default();

    assert!(config.execution_backend.is_supported_on_current_platform());
    Tournament::new(config.clone()).unwrap();
    #[cfg(target_os = "macos")]
    assert!(matches!(
        config.execution_backend,
        ComputeBackend::MetalWgpu { .. }
    ));
    #[cfg(not(target_os = "macos"))]
    assert!(matches!(
        config.execution_backend,
        ComputeBackend::CpuCandle
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_backend_respects_current_platform_support_contract() {
    let backend = ComputeBackend::cuda_default();

    #[cfg(target_os = "macos")]
    assert!(!backend.is_supported_on_current_platform());
    #[cfg(not(target_os = "macos"))]
    assert!(backend.is_supported_on_current_platform());
}

#[test]
fn tournament_presets_never_clip_eval_examples() {
    for preset in [
        TournamentPreset::FastTest,
        TournamentPreset::ResearchMedium,
        TournamentPreset::ChallengerLane,
        TournamentPreset::MinimalProvingGround,
        TournamentPreset::BullpenPolish,
        TournamentPreset::PressureTest,
        TournamentPreset::CandidateStress,
    ] {
        let config = preset.config();
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family: 8,
            eval_examples_per_family: 8,
            seed: config.seed,
            depth_config: config.generator_depth_config,
        })
        .unwrap();

        assert!(
            generator.max_tokens_for(TaskFamily::RecursiveSentence, DatasetSplit::Eval)
                <= config.max_seq_len
        );
        assert!(
            generator.max_tokens_for(TaskFamily::ArcGrid, DatasetSplit::Eval) <= config.max_seq_len
        );
    }
}

#[test]
fn tournament_rejects_invalid_workspace_config() {
    let error = Tournament::new(TournamentConfig {
        levels: 1,
        ..TournamentConfig::fast_test()
    })
    .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn tournament_rejects_zero_parallelism() {
    let error = Tournament::new(TournamentConfig {
        parallelism: 0,
        ..TournamentConfig::fast_test()
    })
    .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn tournament_streams_species_events_in_sequential_mode() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let reporter_events = Arc::clone(&events);
    let reporter = Arc::new(move |event: TournamentProgressEvent| {
        reporter_events.lock().unwrap().push(event);
    });

    let results = tournament
        .run_generation_with_reporter(&test_species_registry(), Some(reporter))
        .unwrap();

    assert_eq!(results.len(), SpeciesId::ALL.len());

    let events = events.lock().unwrap();
    let started = events
        .iter()
        .filter(|event| matches!(event, TournamentProgressEvent::SpeciesStarted(_)))
        .count();
    let completed = events
        .iter()
        .filter(|event| matches!(event, TournamentProgressEvent::SpeciesCompleted(_)))
        .count();

    assert_eq!(started, SpeciesId::ALL.len());
    assert_eq!(completed, SpeciesId::ALL.len());
}

#[test]
fn tournament_parallel_mode_respects_parallelism_cap() {
    CONCURRENT_SPECIES.store(0, Ordering::SeqCst);
    MAX_CONCURRENT_SPECIES.store(0, Ordering::SeqCst);
    let tournament = Tournament::new(
        TournamentConfig::fast_test()
            .with_execution_mode(ExecutionMode::Parallel)
            .with_parallelism(2),
    )
    .unwrap();

    let results = tournament
        .run_generation(&parallelism_test_species_registry())
        .unwrap();

    assert_eq!(results.len(), SpeciesId::ALL.len());
    assert!(MAX_CONCURRENT_SPECIES.load(Ordering::SeqCst) <= 2);
    assert!(MAX_CONCURRENT_SPECIES.load(Ordering::SeqCst) >= 2);
}

#[test]
fn generator_rejects_vocab_that_cannot_encode_reserved_tokens() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: MIN_VOCAB_SIZE - 1,
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn generator_rejects_sequence_lengths_that_clip_recursive_tasks() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        max_seq_len: MIN_SEQUENCE_LEN - 1,
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn generator_rejects_invalid_depth_ranges() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        depth_config: GeneratorDepthConfig {
            sentence_train_max_depth: 4,
            sentence_eval_min_depth: 6,
            sentence_eval_max_depth: 5,
            ..GeneratorDepthConfig::default()
        },
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[derive(Module, Debug)]
struct CountingRule<B: Backend> {
    hidden_dim: usize,
    _marker: core::marker::PhantomData<B>,
}

#[derive(Module, Debug)]
struct AddInputRule<B: Backend> {
    hidden_dim: usize,
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend> AddInputRule<B> {
    fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> FractalRule<B> for AddInputRule<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        Ok(FractalState::Flat(state.flat()? + x.clone()))
    }

    fn name(&self) -> &'static str {
        "add_input_rule"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(Self::new(self.hidden_dim))
    }
}

impl<B: Backend> CountingRule<B> {
    fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> FractalRule<B> for CountingRule<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        _x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        APPLY_COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(state.clone())
    }

    fn name(&self) -> &'static str {
        "counting_rule"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(Self::new(self.hidden_dim))
    }
}

#[test]
fn recurrent_model_uses_only_rule_apply_for_state_transitions() {
    let device = Default::default();
    APPLY_COUNTER.store(0, Ordering::SeqCst);
    let rule = CountingRule::<TestBackend>::new(8);
    let model = FractalModel::new(64, 8, 3, 1.1, PAD_TOKEN, rule, &device);
    let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 4], &device);

    let _ = model.forward_tokens(input_ids).unwrap();

    assert_eq!(APPLY_COUNTER.load(Ordering::SeqCst), 12);
}

#[test]
fn recurrent_model_routes_recursion_per_sample() {
    let device = Default::default();
    let rule = AddInputRule::<TestBackend>::new(1);
    let mut model = FractalModel::new(3, 1, 4, 0.65, PAD_TOKEN, rule, &device);
    model.embedding.weight =
        Param::from_data(TensorData::new(vec![0.0f32, 1.0, 0.3], [3, 1]), &device);
    model.router.projection.weight = Param::from_data(TensorData::from([[1.0f32]]), &device);
    model.router.projection.bias = Some(Param::from_data(TensorData::from([0.0f32]), &device));
    model.output.weight = Param::from_data(TensorData::from([[1.0f32, 1.0, 1.0]]), &device);
    model.output.bias = Some(Param::from_data(
        TensorData::from([0.0f32, 0.0, 0.0]),
        &device,
    ));

    let input_ids =
        Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![1i64, 2], [2, 1]), &device);
    let logits = model.forward_tokens(input_ids).unwrap();
    let values = logits.into_data().to_vec::<f32>().unwrap();

    assert!((values[0] - 1.0).abs() < 1e-5);
    assert!((values[3] - 0.9).abs() < 1e-5);
}

#[test]
fn state_batch_mask_where_expands_sample_mask_across_hidden_width() {
    let device = Default::default();
    let current = FractalState::Flat(Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![1.0f32, 1.0, 1.0, 1.0], [2, 2]),
        &device,
    ));
    let next = FractalState::Flat(Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![9.0f32, 9.0, 8.0, 8.0], [2, 2]),
        &device,
    ));
    let mask = Tensor::<TestBackend, 1, burn::tensor::Bool>::from_bool(
        TensorData::new(vec![true, false], [2]),
        &device,
    );

    let merged = current.batch_mask_where(mask, next).unwrap();
    let values = merged.flat().unwrap().into_data().to_vec::<f32>().unwrap();

    assert_eq!(values, vec![9.0, 9.0, 1.0, 1.0]);
}

fn test_species_registry() -> Vec<SpeciesDefinition> {
    SpeciesId::ALL
        .iter()
        .copied()
        .map(test_species_definition)
        .collect()
}

#[cfg(not(feature = "cuda"))]
fn test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(id, indexed_stub_species_runner, stub_species_runner_metal)
}

#[cfg(feature = "cuda")]
fn test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(
        id,
        indexed_stub_species_runner,
        stub_species_runner_metal,
        stub_species_runner_cuda,
    )
}

fn indexed_stub_species_runner(
    context: SpeciesRunContext,
) -> Result<SpeciesRawMetrics, FractalError> {
    Ok(SpeciesRawMetrics {
        species: SpeciesId::ALL[context.index],
        grad_norm_depth_20: 1.0,
        long_context_perplexity: 10.0,
        arc_accuracy: 0.5,
        tokens_per_sec: 100.0,
    })
}

fn parallelism_stub_species_runner(
    context: SpeciesRunContext,
) -> Result<SpeciesRawMetrics, FractalError> {
    let current = CONCURRENT_SPECIES.fetch_add(1, Ordering::SeqCst) + 1;
    MAX_CONCURRENT_SPECIES.fetch_max(current, Ordering::SeqCst);
    thread::sleep(Duration::from_millis(40));
    CONCURRENT_SPECIES.fetch_sub(1, Ordering::SeqCst);
    indexed_stub_species_runner(context)
}

fn stub_species_runner_metal(
    context: SpeciesRunContext,
    _device: WgpuDevice,
) -> Result<SpeciesRawMetrics, FractalError> {
    indexed_stub_species_runner(context)
}

#[cfg(feature = "cuda")]
fn stub_species_runner_cuda(
    context: SpeciesRunContext,
    _device: CandleDevice,
) -> Result<SpeciesRawMetrics, FractalError> {
    indexed_stub_species_runner(context)
}

fn parallelism_stub_species_runner_metal(
    context: SpeciesRunContext,
    _device: WgpuDevice,
) -> Result<SpeciesRawMetrics, FractalError> {
    parallelism_stub_species_runner(context)
}

#[cfg(feature = "cuda")]
fn parallelism_stub_species_runner_cuda(
    context: SpeciesRunContext,
    _device: CandleDevice,
) -> Result<SpeciesRawMetrics, FractalError> {
    parallelism_stub_species_runner(context)
}

fn parallelism_test_species_registry() -> Vec<SpeciesDefinition> {
    SpeciesId::ALL
        .iter()
        .copied()
        .map(parallelism_test_species_definition)
        .collect()
}

#[cfg(not(feature = "cuda"))]
fn parallelism_test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(
        id,
        parallelism_stub_species_runner,
        parallelism_stub_species_runner_metal,
    )
}

#[cfg(feature = "cuda")]
fn parallelism_test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(
        id,
        parallelism_stub_species_runner,
        parallelism_stub_species_runner_metal,
        parallelism_stub_species_runner_cuda,
    )
}
