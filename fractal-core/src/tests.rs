use std::sync::atomic::{AtomicUsize, Ordering};

use burn::{
    backend::Candle,
    module::Module,
    module::Param,
    tensor::{backend::Backend, Int, Tensor, TensorData},
};

use crate::{
    data_generator::{
        DatasetSplit, GeneratorConfig, SimpleHierarchicalGenerator, TaskFamily, MIN_SEQUENCE_LEN,
        MIN_VOCAB_SIZE, PAD_TOKEN,
    },
    fitness::{aggregate_results, SpeciesRawMetrics},
    lifecycle::{Tournament, TournamentConfig, TournamentPreset, TournamentSequence},
    model::FractalModel,
    primitives::{
        b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
        b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal, complex_square,
        p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot,
        p3_hierarchical::P3Hierarchical,
    },
    registry::{species_registry, ComputeBackend, ExecutionMode, SpeciesId},
    router::EarlyExitRouter,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

type TestBackend = Candle<f32, i64>;
static APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);

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
fn primitives_preserve_declared_layout_and_shape() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::zeros([2, 8], &device);

    let primitives: Vec<Box<dyn FractalRule<TestBackend>>> = vec![
        Box::new(P1Contractive::new(8, &device)),
        Box::new(P2Mandelbrot::new(8, &device)),
        Box::new(P3Hierarchical::new(8, 3, &device)),
        Box::new(B1FractalGated::new(8, &device)),
        Box::new(B2StableHierarchical::new(8, 3, &device)),
        Box::new(B3FractalHierarchical::new(8, 3, &device)),
        Box::new(B4Universal::new(8, 3, &device)),
    ];

    for primitive in primitives {
        let state =
            FractalState::zeros(primitive.state_layout(), 2, primitive.hidden_dim(), &device)
                .unwrap();
        let next = primitive.apply(&state, &x).unwrap();
        assert_eq!(primitive.state_layout(), next.layout());
    }
}

#[test]
fn clone_box_preserves_metadata() {
    let device = Default::default();
    let primitive: Box<dyn FractalRule<TestBackend>> = Box::new(B4Universal::new(8, 3, &device));
    let clone = primitive.clone_box();

    assert_eq!(primitive.name(), clone.name());
    assert_eq!(primitive.hidden_dim(), clone.hidden_dim());
    assert_eq!(primitive.state_layout(), clone.state_layout());
}

#[test]
fn hierarchical_rules_update_all_levels_without_shape_collapse() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::zeros([2, 8], &device);
    let rule = B2StableHierarchical::new(8, 4, &device);
    let state =
        FractalState::zeros(StateLayout::Hierarchical { levels: 4 }, 2, 8, &device).unwrap();
    let next = rule.apply(&state, &x).unwrap();

    match next {
        FractalState::Hierarchical(tensor) => assert_eq!(tensor.dims(), [2, 4, 8]),
        _ => panic!("expected hierarchical state"),
    }
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
fn tournament_returns_exactly_seven_results() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let results = tournament.run_generation().unwrap();

    assert_eq!(results.len(), 7);
    assert_eq!(results[0].rank, 1);
}

#[test]
fn species_registry_lists_every_species_once() {
    let ids = species_registry()
        .iter()
        .map(|species| species.id)
        .collect::<Vec<_>>();

    assert_eq!(ids, SpeciesId::ALL.to_vec());
}

#[test]
fn tournament_parallel_mode_returns_exactly_seven_results() {
    let tournament =
        Tournament::new(TournamentConfig::fast_test().with_execution_mode(ExecutionMode::Parallel))
            .unwrap();
    let results = tournament.run_generation().unwrap();

    assert_eq!(results.len(), 7);
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
    assert_eq!(config.batch_size, 2);
    assert_eq!(config.train_steps_per_species, 5);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.execution_mode, ExecutionMode::Sequential);
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

#[test]
fn tournament_presets_never_clip_eval_examples() {
    for preset in [
        TournamentPreset::FastTest,
        TournamentPreset::ResearchMedium,
        TournamentPreset::PressureTest,
    ] {
        let config = preset.config();
        let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
            vocab_size: config.vocab_size,
            max_seq_len: config.max_seq_len,
            train_examples_per_family: 8,
            eval_examples_per_family: 8,
            seed: config.seed,
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
fn aggregate_results_penalizes_non_finite_perplexity() {
    let ranked = aggregate_results(vec![
        SpeciesRawMetrics {
            species: SpeciesId::P1Contractive,
            grad_norm_depth_20: 1.0,
            long_context_perplexity: f64::NAN,
            arc_accuracy: 0.0,
            tokens_per_sec: 100.0,
        },
        SpeciesRawMetrics {
            species: SpeciesId::P2Mandelbrot,
            grad_norm_depth_20: 1.0,
            long_context_perplexity: 20.0,
            arc_accuracy: 0.0,
            tokens_per_sec: 100.0,
        },
    ]);

    assert_eq!(ranked[0].species, SpeciesId::P2Mandelbrot);
    assert_eq!(ranked[1].species, SpeciesId::P1Contractive);
}

#[test]
fn tournament_rejects_invalid_workspace_config() {
    let error = Tournament::new(TournamentConfig {
        levels: 1,
        ..TournamentConfig::fast_test()
    })
    .unwrap_err();

    assert!(matches!(
        error,
        crate::error::FractalError::InvalidConfig(_)
    ));
}

#[test]
fn generator_rejects_vocab_that_cannot_encode_reserved_tokens() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: MIN_VOCAB_SIZE - 1,
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(
        error,
        crate::error::FractalError::InvalidConfig(_)
    ));
}

#[test]
fn generator_rejects_sequence_lengths_that_clip_recursive_tasks() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        max_seq_len: MIN_SEQUENCE_LEN - 1,
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(
        error,
        crate::error::FractalError::InvalidConfig(_)
    ));
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
    ) -> Result<FractalState<B>, crate::error::FractalError> {
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
    ) -> Result<FractalState<B>, crate::error::FractalError> {
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
    let input_ids = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([2, 4], &device);

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
