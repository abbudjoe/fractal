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
    nn,
    optim::GradientsParams,
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
        ArtifactPolicy, BudgetSpec, ComparisonContract, DecisionIntent, ExecutionBackend,
        ExecutionTarget, ExecutionTargetKind, ExperimentId, ExperimentQuestion,
        ExperimentSpecTemplate, LaneIntent, LearningRateScheduleSpec, OptimizerKind, OptimizerSpec,
        RunExecutionOutcome, RunOutcomeClass, RunQualityOutcome, RuntimeSurfaceSpec, Tournament,
        TournamentConfig, TournamentPreset, TournamentProgressEvent, TournamentSequence,
    },
    model::FractalModel,
    primitives::complex_square,
    registry::{
        clip_gradients_global_norm, gradient_l2_norm, is_valid_primitive_variant_name,
        should_log_training_checkpoint, training_progress_interval, ComputeBackend,
        CpuTrainBackend, ExecutionMode, PrimitiveVariantName, SpeciesDefinition, SpeciesId,
        SpeciesRunContext,
    },
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
fn proving_ground_baseline_preset_targets_bounded_first_mandelbox_run() {
    let config = TournamentPreset::ProvingGroundBaseline.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.levels, 3);
    assert_eq!(config.max_seq_len, 64);
    assert_eq!(config.max_recursion_depth, 6);
    assert_eq!(config.stability_depth, 6);
    assert_eq!(config.train_batch_size, 16);
    assert_eq!(config.eval_batch_size, 16);
    assert_eq!(config.train_steps_per_species, 5);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.learning_rate, 5e-4);
    assert_eq!(config.optimizer, OptimizerSpec::legacy_adam(5e-4));
}

#[test]
fn default_and_preset_optimizer_contracts_remain_legacy_for_current_tournaments() {
    let default_config = TournamentConfig::default();
    assert_eq!(
        default_config.optimizer,
        OptimizerSpec::legacy_adam(default_config.learning_rate)
    );
    assert_eq!(default_config.optimizer.kind, OptimizerKind::Adam);
    assert_eq!(
        default_config.optimizer.schedule,
        LearningRateScheduleSpec::constant()
    );

    let fast_test = TournamentPreset::FastTest.config();
    assert_eq!(
        fast_test.optimizer,
        OptimizerSpec::legacy_adam(fast_test.learning_rate)
    );
    assert_eq!(fast_test.optimizer.kind, OptimizerKind::Adam);
}

#[test]
fn stage0_optimizer_contract_is_adamw_with_warmup_cosine_schedule() {
    let optimizer = OptimizerSpec::stage0_adamw();

    assert_eq!(optimizer.kind, OptimizerKind::AdamW);
    assert_eq!(optimizer.peak_learning_rate, 2e-4);
    assert_eq!(optimizer.beta_1, 0.9);
    assert_eq!(optimizer.beta_2, 0.95);
    assert_eq!(optimizer.epsilon, 1e-8);
    assert_eq!(optimizer.weight_decay, 0.05);
    assert_eq!(optimizer.gradient_clip_norm, Some(1.0));
    assert_eq!(
        optimizer.schedule,
        LearningRateScheduleSpec::warmup_cosine(0.02, 0.1)
    );
}

#[test]
fn warmup_cosine_schedule_progresses_from_zero_to_floor() {
    let schedule = LearningRateScheduleSpec::warmup_cosine(0.02, 0.1);
    let peak = 2e-4;

    assert_eq!(schedule.learning_rate_at_tokens(peak, 0, 100), 0.0);
    assert!((schedule.learning_rate_at_tokens(peak, 1, 100) - 1e-4).abs() < 1e-10);
    assert!((schedule.learning_rate_at_tokens(peak, 2, 100) - peak).abs() < 1e-10);
    assert!((schedule.learning_rate_at_tokens(peak, 100, 100) - peak * 0.1).abs() < 1e-10);
}

#[test]
fn minimal_baseline_preset_matches_minimal_proving_ground_shape() {
    let config = TournamentPreset::MinimalBaseline.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.max_recursion_depth, 8);
    assert_eq!(config.stability_depth, 8);
    assert_eq!(config.train_steps_per_species, 30);
    assert_eq!(config.eval_batches_per_family, 2);
}

#[test]
fn minimal_stress_lane_preset_matches_minimal_baseline_shape() {
    let config = TournamentPreset::MinimalStressLane.config();

    assert_eq!(config.dim, 128);
    assert_eq!(config.max_recursion_depth, 8);
    assert_eq!(config.stability_depth, 8);
    assert_eq!(config.train_steps_per_species, 30);
    assert_eq!(config.eval_batches_per_family, 2);
}

#[test]
fn eval_budget_defaults_to_eval_batches_per_family_when_unset() {
    let config = TournamentPreset::MinimalStressLane.config();

    assert_eq!(config.perplexity_eval_batches, None);
    assert_eq!(config.arc_eval_batches, None);
    assert_eq!(config.effective_perplexity_eval_batches(), 2);
    assert_eq!(config.effective_arc_eval_batches(), 2);
}

#[test]
fn eval_budget_can_be_split_explicitly() {
    let mut config = TournamentPreset::MinimalStressLane.config();
    config.perplexity_eval_batches = Some(1);
    config.arc_eval_batches = Some(3);

    assert_eq!(config.effective_perplexity_eval_batches(), 1);
    assert_eq!(config.effective_arc_eval_batches(), 3);
}

#[test]
fn zero_explicit_eval_budget_is_rejected() {
    let mut config = TournamentPreset::MinimalStressLane.config();
    config.perplexity_eval_batches = Some(0);

    let error = Tournament::new(config).unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn medium_stress_preset_targets_midweight_single_species_run() {
    let config = TournamentPreset::MediumStress.config();

    assert_eq!(config.dim, 192);
    assert_eq!(config.max_seq_len, 128);
    assert_eq!(config.max_recursion_depth, 12);
    assert_eq!(config.stability_depth, 12);
    assert_eq!(config.train_steps_per_species, 80);
    assert_eq!(config.eval_batches_per_family, 2);
}

#[test]
fn full_medium_stress_matches_medium_stress_shape() {
    let config = TournamentPreset::FullMediumStress.config();

    assert_eq!(config.dim, 192);
    assert_eq!(config.max_seq_len, 128);
    assert_eq!(config.max_recursion_depth, 12);
    assert_eq!(config.stability_depth, 12);
    assert_eq!(config.train_steps_per_species, 80);
    assert_eq!(config.eval_batches_per_family, 2);
}

#[test]
fn intermediate_stress_preset_targets_hybrid_decider_lane() {
    let config = TournamentPreset::IntermediateStress.config();

    assert_eq!(config.dim, 160);
    assert_eq!(config.max_seq_len, 96);
    assert_eq!(config.max_recursion_depth, 10);
    assert_eq!(config.stability_depth, 10);
    assert_eq!(config.train_steps_per_species, 48);
    assert_eq!(config.eval_batches_per_family, 2);
    assert_eq!(config.generator_depth_config.sentence_eval_max_depth, 10);
}

#[test]
fn lighter_intermediate_stress_halves_medium_batch_sizes() {
    let config = TournamentPreset::LighterIntermediateStress.config();

    assert_eq!(config.dim, 160);
    assert_eq!(config.max_seq_len, 96);
    assert_eq!(config.max_recursion_depth, 10);
    assert_eq!(config.stability_depth, 10);
    assert_eq!(config.train_batch_size, 4);
    assert_eq!(config.eval_batch_size, 2);
    assert_eq!(config.train_steps_per_species, 48);
}

#[test]
fn training_progress_interval_targets_quarterly_phase_checkpoints() {
    assert_eq!(training_progress_interval(1), 1);
    assert_eq!(training_progress_interval(4), 1);
    assert_eq!(training_progress_interval(5), 2);
    assert_eq!(training_progress_interval(48), 12);
    assert_eq!(training_progress_interval(80), 20);
}

#[test]
fn training_progress_checkpoint_logs_quarterly_and_final_steps() {
    let checkpoints = (1..=10)
        .filter(|step| should_log_training_checkpoint(*step, 10))
        .collect::<Vec<_>>();

    assert_eq!(checkpoints, vec![3, 6, 9, 10]);
    assert!(!should_log_training_checkpoint(0, 10));
    assert!(!should_log_training_checkpoint(1, 0));
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
        TournamentPreset::MinimalBaseline,
        TournamentPreset::MinimalProvingGround,
        TournamentPreset::ProvingGroundBaseline,
        TournamentPreset::IntermediateStress,
        TournamentPreset::MediumStress,
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
fn tournament_artifacts_capture_manifests_and_phase_timings() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let artifact = tournament
        .run_generation_artifacts(&test_species_registry(), None)
        .unwrap();

    assert_eq!(artifact.species.len(), SpeciesId::ALL.len());
    assert_eq!(artifact.outcome_class(), RunOutcomeClass::Success);

    let first = &artifact.species[0];
    assert_eq!(first.manifest.variant_name.as_str(), "p1_contractive_v1");
    assert!(first.manifest.timeout_budget.is_none());
    assert!(first.manifest.experiment.is_none());
    assert!(!first.phase_timings.is_empty());
    assert_eq!(first.outcome_class(), RunOutcomeClass::Success);
}

#[test]
fn tournament_artifacts_capture_resolved_experiment_spec() {
    let config = TournamentPreset::FastTest
        .config()
        .with_experiment(test_experiment_template(
            TournamentPreset::FastTest.config(),
        ));
    let tournament = Tournament::new(config).unwrap();
    let artifact = tournament
        .run_generation_artifacts(&[test_species_definition(SpeciesId::P1Contractive)], None)
        .unwrap();

    let experiment = artifact.species[0].manifest.experiment.as_ref().unwrap();
    assert_eq!(experiment.variant.species, SpeciesId::P1Contractive);
    assert_eq!(
        experiment.variant.variant_name.as_str(),
        "p1_contractive_v1"
    );
    assert_eq!(experiment.comparison.label(), "authoritative same-preset");
    assert_eq!(experiment.runtime.label(), "conservative-defaults");
    assert_eq!(experiment.execution.backend, ExecutionBackend::Cpu);
}

#[test]
fn tournament_rejects_experiment_templates_that_disagree_with_config() {
    let mut config = TournamentPreset::FastTest.config();
    let mut experiment = test_experiment_template(config.clone());
    experiment.budget.train_batch_size = config.train_batch_size + 1;
    config.experiment = Some(experiment);

    let error = Tournament::new(config).unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
}

#[test]
fn tournament_artifacts_classify_numeric_failure_without_losing_metrics() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let artifact = tournament
        .run_generation_artifacts(
            &[numeric_failure_species_definition(
                SpeciesId::P1FractalHybrid,
            )],
            None,
        )
        .unwrap();

    assert_eq!(artifact.species.len(), 1);
    let record = &artifact.species[0];
    assert_eq!(record.outcome_class(), RunOutcomeClass::NumericFailure);
    assert!(matches!(
        record.quality_outcome,
        crate::lifecycle::RunQualityOutcome::NumericFailure
    ));
    assert!(record.metrics.is_some());
}

#[test]
fn run_outcome_class_combines_execution_and_quality_contracts() {
    assert_eq!(
        RunOutcomeClass::from_components(RunExecutionOutcome::Success, RunQualityOutcome::Clean),
        RunOutcomeClass::Success
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::Success,
            RunQualityOutcome::NumericFailure
        ),
        RunOutcomeClass::NumericFailure
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::Success,
            RunQualityOutcome::LowSignal
        ),
        RunOutcomeClass::LowSignal
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::Success,
            RunQualityOutcome::RuntimeCost
        ),
        RunOutcomeClass::RuntimeCost
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::TrainTimeout,
            RunQualityOutcome::Clean
        ),
        RunOutcomeClass::TrainTimeout
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::EvalConstrained,
            RunQualityOutcome::Clean
        ),
        RunOutcomeClass::EvalConstrained
    );
    assert_eq!(
        RunOutcomeClass::from_components(
            RunExecutionOutcome::InfraFailure,
            RunQualityOutcome::Clean
        ),
        RunOutcomeClass::InfraFailure
    );
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
fn primitive_variant_naming_convention_accepts_versioned_lowercase_variants() {
    assert!(is_valid_primitive_variant_name("p1_fractal_hybrid_v1"));
    assert!(is_valid_primitive_variant_name(
        "b1_fractal_gated_dyn-residual-norm_v1"
    ));
    assert!(is_valid_primitive_variant_name(
        "generalized_mobius_dyn-jitter-norm_v2"
    ));
    assert!(is_valid_primitive_variant_name(
        "mandelbox_recursive_dyn-escape-radius_v1"
    ));
}

#[test]
fn primitive_variant_naming_convention_rejects_invalid_shapes() {
    assert!(!is_valid_primitive_variant_name("ifs_v1"));
    assert!(!is_valid_primitive_variant_name("IFS_dyn-radius-depth_v1"));
    assert!(!is_valid_primitive_variant_name("p1_fractal_hybrid"));
    assert!(!is_valid_primitive_variant_name("p1__hybrid_v1"));
}

#[test]
fn tournament_rejects_species_with_invalid_variant_names() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let error = tournament
        .run_generation(&[invalid_species_definition(
            SpeciesId::P1Contractive,
            PrimitiveVariantName::new_unchecked("invalid"),
        )])
        .unwrap_err();

    assert!(matches!(error, FractalError::InvalidConfig(_)));
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
fn global_gradient_clipping_scales_the_combined_norm() {
    let device = Default::default();
    let linear = nn::LinearConfig::new(4, 3).with_bias(true).init(&device);
    let input = Tensor::<CpuTrainBackend, 2>::from_floats(
        [[0.25f32, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25]],
        &device,
    );

    let grads = linear.forward(input).backward();
    let mut grads = GradientsParams::from_grads(grads, &linear);
    let before = gradient_l2_norm(&linear, &grads);
    assert!(before > 0.0);

    clip_gradients_global_norm(&linear, &mut grads, before * 0.5);

    let after = gradient_l2_norm(&linear, &grads);
    assert!(after <= before * 0.5 + 1e-6);
    assert!(after > 0.0);
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
    SpeciesDefinition::new(
        id,
        test_variant_name(id),
        indexed_stub_species_runner,
        stub_species_runner_metal,
    )
}

#[cfg(feature = "cuda")]
fn test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(
        id,
        test_variant_name(id),
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

fn numeric_failure_species_runner(
    context: SpeciesRunContext,
) -> Result<SpeciesRawMetrics, FractalError> {
    Ok(SpeciesRawMetrics {
        species: SpeciesId::ALL[context.index],
        grad_norm_depth_20: f64::NAN,
        long_context_perplexity: f64::NAN,
        arc_accuracy: 0.0,
        tokens_per_sec: 0.0,
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
        test_variant_name(id),
        parallelism_stub_species_runner,
        parallelism_stub_species_runner_metal,
    )
}

#[cfg(feature = "cuda")]
fn parallelism_test_species_definition(id: SpeciesId) -> SpeciesDefinition {
    SpeciesDefinition::new(
        id,
        test_variant_name(id),
        parallelism_stub_species_runner,
        parallelism_stub_species_runner_metal,
        parallelism_stub_species_runner_cuda,
    )
}

fn invalid_species_definition(
    id: SpeciesId,
    variant_name: PrimitiveVariantName,
) -> SpeciesDefinition {
    #[cfg(feature = "cuda")]
    {
        SpeciesDefinition::new(
            id,
            variant_name,
            indexed_stub_species_runner,
            stub_species_runner_metal,
            stub_species_runner_cuda,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        SpeciesDefinition::new(
            id,
            variant_name,
            indexed_stub_species_runner,
            stub_species_runner_metal,
        )
    }
}

fn numeric_failure_species_definition(id: SpeciesId) -> SpeciesDefinition {
    let variant_name = test_variant_name(id);
    #[cfg(feature = "cuda")]
    {
        SpeciesDefinition::new(
            id,
            variant_name,
            numeric_failure_species_runner,
            stub_species_runner_metal,
            stub_species_runner_cuda,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        SpeciesDefinition::new(
            id,
            variant_name,
            numeric_failure_species_runner,
            stub_species_runner_metal,
        )
    }
}

fn test_experiment_template(config: TournamentConfig) -> ExperimentSpecTemplate {
    ExperimentSpecTemplate {
        experiment_id: ExperimentId {
            logical_name: "fast-test-control".to_owned(),
            run_id: "run-123".to_owned(),
            branch: Some("codex/exp-spec-core".to_owned()),
            commit_sha: Some("abc123".to_owned()),
            created_at_unix_ms: 123,
        },
        question: ExperimentQuestion {
            summary: "evaluate fast-test control plane".to_owned(),
            lane_intent: LaneIntent::Benchmark,
            decision_intent: DecisionIntent::Benchmark,
        },
        budget: BudgetSpec::from_config(TournamentPreset::FastTest, &config),
        optimizer: config.optimizer.clone(),
        runtime: RuntimeSurfaceSpec::default(),
        comparison: ComparisonContract::authoritative_same_preset(),
        execution: ExecutionTarget {
            kind: ExecutionTargetKind::Local,
            backend: ExecutionBackend::from_compute_backend(&config.execution_backend),
            execution_mode: config.execution_mode,
            pod_id: None,
            wrapper_timeout_seconds: None,
        },
        artifacts: ArtifactPolicy::default(),
    }
}

fn test_variant_name(id: SpeciesId) -> PrimitiveVariantName {
    let name = match id {
        SpeciesId::P1Contractive => "p1_contractive_v1",
        SpeciesId::P3Hierarchical => "p3_hierarchical_v1",
        SpeciesId::B2StableHierarchical => "b2_stable_hierarchical_v1",
        SpeciesId::B1FractalGated => "b1_fractal_gated_dyn-residual-norm_v1",
        SpeciesId::P1FractalHybrid => "p1_fractal_hybrid_v1",
        SpeciesId::P1FractalHybridComposite => "p1_fractal_hybrid_composite_v1",
        SpeciesId::P1FractalHybridDynGate => "p1_fractal_hybrid_dyn-gate_v1",
        SpeciesId::P2Mandelbrot => "p2_mandelbrot_dyn-gate-norm_v1",
        SpeciesId::B3FractalHierarchical => "b3_fractal_hierarchical_dyn-radius-depth_v1",
        SpeciesId::B4Universal => "b4_universal_dyn-residual-norm_v1",
        SpeciesId::Ifs => "ifs_dyn-radius-depth_v1",
        SpeciesId::GeneralizedMobius => "generalized_mobius_dyn-jitter-norm_v2",
        SpeciesId::LogisticChaoticMap => "logistic_chaotic_map_v1",
        SpeciesId::JuliaRecursiveEscape => "julia_recursive_escape_v1",
        SpeciesId::MandelboxRecursive => "mandelbox_recursive_dyn-escape-radius_v1",
    };
    PrimitiveVariantName::new_unchecked(name)
}
