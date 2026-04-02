use std::{
    env, fs,
    panic::AssertUnwindSafe,
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
        TaskFamily, TokenBatch, MIN_SEQUENCE_LEN, MIN_VOCAB_SIZE, PAD_TOKEN,
    },
    diagnostics::{
        clear_last_diagnostics_runtime_artifact, take_last_diagnostics_runtime_artifact,
        DiagnosticEventKind, DiagnosticProbeKind, DiagnosticProbeRequest, DiagnosticsPolicy,
        DiagnosticsRecorder, ProbeCadence, TrainStepDiagnosticContext,
    },
    error::FractalError,
    fitness::SpeciesRawMetrics,
    lifecycle::{
        ArtifactPolicy, BridgePackagingSpec, BridgeSplitPolicy, BridgeSubstrateMode, BudgetSpec,
        ComparisonContract, DecisionIntent, ExecutionBackend, ExecutionTarget, ExecutionTargetKind,
        ExperimentId, ExperimentQuestion, ExperimentSpecTemplate, LaneIntent, LaunchPolicySpec,
        LearningRateScheduleSpec, ModelContractSpec, NumericPrecisionKind, OptimizerKind,
        OptimizerSpec, QuantizationPolicy, QuantizedPrecisionKind, RunExecutionOutcome,
        RunOutcomeClass, RunQualityOutcome, RuntimeSurfaceSpec, TextCorpusFormat,
        TextCorpusSourceSpec, TextCorpusSplitSpec, TokenizerArtifactSpec, Tournament,
        TournamentConfig, TournamentPreset, TournamentProgressEvent, TournamentSequence,
        TrainingInputSpec, WeightExportFormat, WeightExportPhase, WeightExportPolicy,
    },
    model::FractalModel,
    primitives::complex_square,
    registry::{
        clip_gradients_global_norm, export_weight_phase, gradient_l2_norm,
        is_valid_primitive_variant_name, load_weight_export_metadata, read_weight_export_metadata,
        resolve_precision_profile, resolve_weight_export_paths, run_species_with_factory_candle,
        should_log_training_checkpoint, take_last_species_run_artifact, training_progress_interval,
        ComputeBackend, CpuTrainBackend, ExecutionMode, PrimitiveVariantName, SpeciesDefinition,
        SpeciesId, SpeciesRunContext,
    },
    router::EarlyExitRouter,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};

type TestBackend = Candle<f32, i64>;
static APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);
static CONCURRENT_SPECIES: AtomicUsize = AtomicUsize::new(0);
static MAX_CONCURRENT_SPECIES: AtomicUsize = AtomicUsize::new(0);
static FAILING_APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);
static PANICKING_APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);
static EXPORT_ENV_MUTEX: Mutex<()> = Mutex::new(());

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
fn default_launch_policy_contracts_remain_legacy_for_current_tournaments() {
    let default_config = TournamentConfig::default();
    assert_eq!(
        default_config.launch_policy,
        LaunchPolicySpec::legacy_default()
    );

    let fast_test = TournamentPreset::FastTest.config();
    assert_eq!(fast_test.launch_policy, LaunchPolicySpec::legacy_default());
    assert_eq!(
        RuntimeSurfaceSpec::default().launch_policy,
        LaunchPolicySpec::legacy_default()
    );
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
fn stage0_launch_policy_contract_matches_stage0_training_plan() {
    let launch_policy = LaunchPolicySpec::stage0_default();

    assert_eq!(launch_policy.precision.compute, NumericPrecisionKind::Bf16);
    assert_eq!(
        launch_policy.precision.optimizer_state,
        NumericPrecisionKind::Fp32
    );
    assert_eq!(
        launch_policy.precision.reduction,
        NumericPrecisionKind::Fp32
    );
    assert!(launch_policy.precision.tf32_enabled);

    assert_eq!(launch_policy.checkpoint.interval_tokens, Some(10_000_000));
    assert!(launch_policy.checkpoint.keep_latest);
    assert!(launch_policy.checkpoint.keep_best);
    assert!(launch_policy.checkpoint.keep_final);
    assert!(launch_policy.checkpoint.keep_previous);

    assert_eq!(
        launch_policy.eval_cadence.perplexity_interval_tokens,
        Some(10_000_000)
    );
    assert_eq!(
        launch_policy.eval_cadence.stability_interval_tokens,
        Some(10_000_000)
    );
    assert_eq!(
        launch_policy.eval_cadence.arc_interval_tokens,
        Some(20_000_000)
    );
    assert_eq!(
        launch_policy.eval_cadence.systems_speed_interval_tokens,
        Some(20_000_000)
    );
    assert!(launch_policy.eval_cadence.final_full_eval);

    assert!(launch_policy.resume.resume_on_interrupt);
    assert!(launch_policy.resume.restart_on_corruption);
    assert!(launch_policy.resume.restart_on_contract_ambiguity);
    assert_eq!(
        launch_policy.diagnostics,
        crate::DiagnosticsPolicy::disabled()
    );
    assert_eq!(
        launch_policy.weight_export,
        WeightExportPolicy::legacy_default()
    );
}

#[test]
fn stage1_weight_export_contract_is_burn_bin_best_and_final_required() {
    let policy = WeightExportPolicy::stage1_default();

    assert_eq!(policy.format, WeightExportFormat::BurnBin);
    assert_eq!(
        policy.phases,
        vec![WeightExportPhase::Best, WeightExportPhase::Final]
    );
    assert!(policy.required);
    assert!(policy.validate().is_ok());
}

#[test]
fn weight_export_policy_rejects_unsupported_format_and_phase() {
    let unsupported_format_policy = WeightExportPolicy {
        format: WeightExportFormat::SafeTensors,
        phases: vec![WeightExportPhase::Best],
        required: true,
    };
    assert!(unsupported_format_policy
        .validate_against_backend(&ComputeBackend::CpuCandle)
        .is_err());

    let unsupported_phase_policy = WeightExportPolicy {
        format: WeightExportFormat::BurnBin,
        phases: vec![WeightExportPhase::Latest],
        required: true,
    };
    assert!(unsupported_phase_policy
        .validate_against_backend(&ComputeBackend::CpuCandle)
        .is_err());
}

#[test]
fn weight_export_contract_validates_against_config_and_rejects_mismatch() {
    let mut config = TournamentPreset::FastTest.config();
    config.launch_policy.precision = LaunchPolicySpec::stage0_default().precision;
    let contract = crate::lifecycle::WeightExportContract {
        experiment_logical_name: "stage1-export".to_owned(),
        experiment_run_id: "run-123".to_owned(),
        experiment_branch: Some("codex/stage0-launch".to_owned()),
        experiment_commit_sha: "abc123".to_owned(),
        species: SpeciesId::P1Contractive.as_str().to_owned(),
        variant_name: "p1_contractive_v1".to_owned(),
        model: ModelContractSpec::recursive_kernel_v1(config.dim, config.max_recursion_depth),
        vocab_size: config.vocab_size,
        precision: config.launch_policy.precision.clone(),
        format: WeightExportFormat::BurnBin,
    };

    assert!(contract.validate_against_config(&config).is_ok());

    let mut mismatched = contract.clone();
    mismatched.vocab_size += 1;
    assert!(mismatched.validate_against_config(&config).is_err());
}

#[test]
fn weight_export_artifact_validate_against_config_rejects_unsupported_format() {
    let config = TournamentPreset::FastTest.config();
    let artifact = crate::lifecycle::WeightExportArtifact {
        format: WeightExportFormat::SafeTensors,
        phase: WeightExportPhase::Best,
        path: "/tmp/weights".to_owned(),
        metadata_path: "/tmp/metadata.json".to_owned(),
        required: true,
        contract: crate::lifecycle::WeightExportContract {
            experiment_logical_name: "stage1-export".to_owned(),
            experiment_run_id: "run-123".to_owned(),
            experiment_branch: Some("codex/stage0-launch".to_owned()),
            experiment_commit_sha: "abc123".to_owned(),
            species: SpeciesId::P1Contractive.as_str().to_owned(),
            variant_name: "p1_contractive_v1".to_owned(),
            model: ModelContractSpec::recursive_kernel_v1(config.dim, config.max_recursion_depth),
            vocab_size: config.vocab_size,
            precision: config.launch_policy.precision.clone(),
            format: WeightExportFormat::SafeTensors,
        },
    };

    assert!(artifact.validate_against_config(&config).is_err());
}

#[test]
fn burn_bin_weight_export_writes_metadata_and_loads_back() {
    let _env_lock = EXPORT_ENV_MUTEX.lock().unwrap();
    let config = TournamentPreset::FastTest.config();
    let device = Default::default();
    let rule = AddInputRule::<TestBackend>::new(config.dim);
    let mut model = FractalModel::new(
        config.vocab_size,
        config.dim,
        config.max_recursion_depth,
        config.router_threshold,
        PAD_TOKEN,
        rule,
        &device,
    );
    model.output.weight = Param::from_data(
        TensorData::new(
            vec![0.5f32; config.vocab_size * config.dim],
            [config.dim, config.vocab_size],
        ),
        &device,
    );
    let mut template = test_experiment_template(config.clone());
    template.runtime.launch_policy.weight_export = WeightExportPolicy::stage1_default();
    let experiment = template.resolve_variant(
        SpeciesId::P1Contractive,
        PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
    );
    let manifest = crate::RunManifest {
        variant_name: PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
        timeout_budget: None,
        config: config.clone(),
        experiment: Some(experiment),
    };
    let stage = crate::SpeciesRunStage {
        species: SpeciesId::P1Contractive,
        ordinal: 1,
        total: 1,
    };
    let temp_root =
        std::env::temp_dir().join(format!("fractal-weight-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_root);
    std::env::set_var("FRACTAL_RUN_EXPORT_DIR", &temp_root);
    let paths = resolve_weight_export_paths(&stage, &manifest);

    let artifact = export_weight_phase(
        stage,
        &manifest,
        &template.runtime.launch_policy.weight_export,
        WeightExportPhase::Best,
        &model,
        &paths,
        template.runtime.launch_policy.weight_export.required,
    )
    .expect("burn-bin export succeeds");
    let metadata = read_weight_export_metadata(std::path::Path::new(&artifact.metadata_path))
        .expect("export metadata loads");

    assert_eq!(metadata, artifact);
    assert!(metadata.validate_against_config(&config).is_ok());

    let probe_input = Tensor::<TestBackend, 2, Int>::from_data(
        TensorData::new(vec![1i64, 2i64], [1, 2]),
        &device,
    );
    let expected_logits = model
        .clone()
        .forward_tokens(probe_input.clone())
        .expect("expected logits");

    let restored_model = load_weight_export_metadata(
        std::path::Path::new(&artifact.metadata_path),
        FractalModel::new(
            config.vocab_size,
            config.dim,
            config.max_recursion_depth,
            config.router_threshold,
            PAD_TOKEN,
            AddInputRule::<TestBackend>::new(config.dim),
            &device,
        ),
        &config,
        &device,
    )
    .expect("exported weights load back");

    let actual_logits = restored_model
        .forward_tokens(probe_input)
        .expect("loaded logits");
    assert_eq!(
        actual_logits.into_data().to_vec::<f32>().unwrap(),
        expected_logits.into_data().to_vec::<f32>().unwrap()
    );

    std::env::remove_var("FRACTAL_RUN_EXPORT_DIR");
    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn best_weight_export_refreshes_when_improved_again() {
    let _env_lock = EXPORT_ENV_MUTEX.lock().unwrap();
    let config = TournamentPreset::FastTest.config();
    let device = Default::default();
    let rule = AddInputRule::<TestBackend>::new(config.dim);
    let mut model = FractalModel::new(
        config.vocab_size,
        config.dim,
        config.max_recursion_depth,
        config.router_threshold,
        PAD_TOKEN,
        rule,
        &device,
    );
    model.output.weight = Param::from_data(
        TensorData::new(
            vec![0.25f32; config.vocab_size * config.dim],
            [config.vocab_size, config.dim],
        ),
        &device,
    );
    let mut template = test_experiment_template(config.clone());
    template.runtime.launch_policy.weight_export = WeightExportPolicy::stage1_default();
    let experiment = template.resolve_variant(
        SpeciesId::P1Contractive,
        PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
    );
    let manifest = crate::RunManifest {
        variant_name: PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
        timeout_budget: None,
        config: config.clone(),
        experiment: Some(experiment),
    };
    let stage = crate::SpeciesRunStage {
        species: SpeciesId::P1Contractive,
        ordinal: 1,
        total: 1,
    };
    let temp_root = std::env::temp_dir().join(format!(
        "fractal-weight-export-refresh-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_root);
    std::env::set_var("FRACTAL_RUN_EXPORT_DIR", &temp_root);
    let paths = resolve_weight_export_paths(&stage, &manifest);

    let first_artifact = export_weight_phase(
        stage.clone(),
        &manifest,
        &template.runtime.launch_policy.weight_export,
        WeightExportPhase::Best,
        &model,
        &paths,
        template.runtime.launch_policy.weight_export.required,
    )
    .expect("first best export succeeds");
    let first_metadata = std::fs::metadata(&first_artifact.metadata_path)
        .expect("first best export metadata exists")
        .modified()
        .expect("first best export metadata timestamp");

    model.output.weight = Param::from_data(
        TensorData::new(
            vec![0.75f32; config.vocab_size * config.dim],
            [config.dim, config.vocab_size],
        ),
        &device,
    );
    std::thread::sleep(Duration::from_millis(10));
    let second_artifact = export_weight_phase(
        stage,
        &manifest,
        &template.runtime.launch_policy.weight_export,
        WeightExportPhase::Best,
        &model,
        &paths,
        template.runtime.launch_policy.weight_export.required,
    )
    .expect("refreshed best export succeeds");
    let second_metadata = std::fs::metadata(&second_artifact.metadata_path)
        .expect("refreshed best export metadata exists")
        .modified()
        .expect("refreshed best export metadata timestamp");

    assert_eq!(first_artifact.path, second_artifact.path);
    assert!(second_metadata > first_metadata);

    std::env::remove_var("FRACTAL_RUN_EXPORT_DIR");
    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn external_weight_export_roots_include_run_identity() {
    let _env_lock = EXPORT_ENV_MUTEX.lock().unwrap();
    let config = TournamentPreset::FastTest.config();
    let device = Default::default();
    let rule = AddInputRule::<TestBackend>::new(config.dim);
    let model = FractalModel::new(
        config.vocab_size,
        config.dim,
        config.max_recursion_depth,
        config.router_threshold,
        PAD_TOKEN,
        rule,
        &device,
    );
    let mut template = test_experiment_template(config.clone());
    template.runtime.launch_policy.weight_export = WeightExportPolicy::stage1_default();
    let experiment_a = template.resolve_variant(
        SpeciesId::P1Contractive,
        PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
    );
    let mut template_b = test_experiment_template(config.clone());
    template_b.experiment_id.logical_name = "fast-test-control-b".to_owned();
    template_b.experiment_id.run_id = "run-456".to_owned();
    template_b.runtime.launch_policy.weight_export = WeightExportPolicy::stage1_default();
    let experiment_b = template_b.resolve_variant(
        SpeciesId::P1Contractive,
        PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
    );
    let manifest_a = crate::RunManifest {
        variant_name: PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
        timeout_budget: None,
        config: config.clone(),
        experiment: Some(experiment_a),
    };
    let manifest_b = crate::RunManifest {
        variant_name: PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
        timeout_budget: None,
        config: config.clone(),
        experiment: Some(experiment_b),
    };
    let stage = crate::SpeciesRunStage {
        species: SpeciesId::P1Contractive,
        ordinal: 1,
        total: 1,
    };
    let temp_root = std::env::temp_dir().join(format!(
        "fractal-weight-export-collision-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&temp_root);
    std::env::set_var("FRACTAL_RUN_EXPORT_DIR", &temp_root);

    let artifact_a = export_weight_phase(
        stage.clone(),
        &manifest_a,
        &template.runtime.launch_policy.weight_export,
        WeightExportPhase::Best,
        &model,
        &resolve_weight_export_paths(&stage, &manifest_a),
        template.runtime.launch_policy.weight_export.required,
    )
    .expect("first export succeeds");
    let artifact_b = export_weight_phase(
        stage.clone(),
        &manifest_b,
        &template_b.runtime.launch_policy.weight_export,
        WeightExportPhase::Best,
        &model,
        &resolve_weight_export_paths(&stage, &manifest_b),
        template_b.runtime.launch_policy.weight_export.required,
    )
    .expect("second export succeeds");

    assert_ne!(artifact_a.path, artifact_b.path);
    assert!(artifact_a.path.contains("fast-test-control"));
    assert!(artifact_a.path.contains("run-123"));
    assert!(artifact_b.path.contains("fast-test-control-b"));
    assert!(artifact_b.path.contains("run-456"));

    std::env::remove_var("FRACTAL_RUN_EXPORT_DIR");
    let _ = std::fs::remove_dir_all(&temp_root);
}

#[test]
fn launch_policy_rejects_invalid_diagnostics_policy() {
    let mut launch_policy = LaunchPolicySpec::stage0_default();
    launch_policy.diagnostics = DiagnosticsPolicy {
        required: true,
        probes: Vec::new(),
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    };
    assert!(launch_policy.validate().is_err());

    launch_policy.diagnostics = DiagnosticsPolicy {
        required: false,
        probes: vec![DiagnosticProbeRequest {
            kind: DiagnosticProbeKind::ForwardPosition,
            cadence: ProbeCadence::StepInterval { steps: 1 },
            position_interval: None,
        }],
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    };
    assert!(launch_policy.validate().is_err());

    launch_policy.diagnostics = DiagnosticsPolicy {
        required: false,
        probes: vec![DiagnosticProbeRequest {
            kind: DiagnosticProbeKind::TrainStep,
            cadence: ProbeCadence::StepInterval { steps: 0 },
            position_interval: None,
        }],
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    };
    assert!(launch_policy.validate().is_err());
}

#[test]
fn launch_policy_rejects_required_cuda_memory_diagnostics_without_cuda_backend() {
    let mut launch_policy = LaunchPolicySpec::stage0_default();
    launch_policy.diagnostics = DiagnosticsPolicy {
        required: true,
        probes: vec![DiagnosticProbeRequest {
            kind: DiagnosticProbeKind::CudaMemorySnapshot,
            cadence: ProbeCadence::EveryStep,
            position_interval: None,
        }],
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    };

    assert!(launch_policy
        .validate_against_backend(&ComputeBackend::CpuCandle)
        .is_err());
}

#[test]
fn precision_policy_allows_tf32_with_fp32_compute_contract() {
    let policy = crate::lifecycle::PrecisionPolicy {
        compute: NumericPrecisionKind::Fp32,
        optimizer_state: NumericPrecisionKind::Fp32,
        reduction: NumericPrecisionKind::Fp32,
        tf32_enabled: true,
        quantization: QuantizationPolicy::disabled(),
    };

    assert!(policy.validate().is_ok());
}

#[test]
fn eval_cadence_rejects_unsupported_partial_final_eval_contract() {
    let policy = crate::lifecycle::EvalCadencePolicy {
        perplexity_interval_tokens: Some(1),
        stability_interval_tokens: Some(1),
        arc_interval_tokens: Some(1),
        systems_speed_interval_tokens: Some(1),
        final_full_eval: false,
    };

    assert!(policy.validate().is_err());
}

#[test]
fn precision_profile_resolves_stage0_bf16_for_candle_backends() {
    let policy = crate::PrecisionPolicy::stage0_default();

    assert!(resolve_precision_profile(&ComputeBackend::CpuCandle, &policy).is_err());

    #[cfg(feature = "cuda")]
    assert_eq!(
        resolve_precision_profile(&ComputeBackend::cuda_default(), &policy).unwrap(),
        ResolvedExecutablePrecisionProfile::CandleBf16
    );
}

#[test]
fn precision_profile_rejects_unimplemented_metal_bf16_runtime() {
    let policy = crate::PrecisionPolicy::stage0_default();
    let result = resolve_precision_profile(&ComputeBackend::metal_default(), &policy);
    assert!(result.is_err());
}

#[test]
fn precision_profile_rejects_quantized_profiles_without_runtime_support() {
    let policy = crate::PrecisionPolicy {
        compute: NumericPrecisionKind::Fp32,
        optimizer_state: NumericPrecisionKind::Fp32,
        reduction: NumericPrecisionKind::Fp32,
        tf32_enabled: false,
        quantization: QuantizationPolicy {
            weights: Some(QuantizedPrecisionKind::Int4),
            activations: Some(QuantizedPrecisionKind::Bit1),
        },
    };

    let result = resolve_precision_profile(&ComputeBackend::CpuCandle, &policy);
    assert!(result.is_err());
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

#[derive(Module, Debug)]
struct FailingRule<B: Backend> {
    hidden_dim: usize,
    fail_after_apply_count: usize,
    _marker: core::marker::PhantomData<B>,
}

#[derive(Module, Debug)]
struct PanickingRule<B: Backend> {
    hidden_dim: usize,
    panic_after_apply_count: usize,
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

impl<B: Backend> FailingRule<B> {
    fn new(hidden_dim: usize, fail_after_apply_count: usize) -> Self {
        Self {
            hidden_dim,
            fail_after_apply_count,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> PanickingRule<B> {
    fn new(hidden_dim: usize, panic_after_apply_count: usize) -> Self {
        Self {
            hidden_dim,
            panic_after_apply_count,
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

impl<B: Backend> FractalRule<B> for FailingRule<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        _x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let applied = FAILING_APPLY_COUNTER.fetch_add(1, Ordering::SeqCst);
        if applied >= self.fail_after_apply_count {
            return Err(FractalError::InvalidState(
                "synthetic forward failure".to_owned(),
            ));
        }
        Ok(state.clone())
    }

    fn name(&self) -> &'static str {
        "failing_rule"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(Self::new(self.hidden_dim, self.fail_after_apply_count))
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

impl<B: Backend> FractalRule<B> for PanickingRule<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        _x: &Tensor<B, 2>,
        _context: ApplyContext,
    ) -> Result<FractalState<B>, FractalError> {
        let applied = PANICKING_APPLY_COUNTER.fetch_add(1, Ordering::SeqCst);
        assert!(
            applied < self.panic_after_apply_count,
            "synthetic panic during forward"
        );
        Ok(state.clone())
    }

    fn name(&self) -> &'static str {
        "panicking_rule"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(Self::new(self.hidden_dim, self.panic_after_apply_count))
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
fn diagnostics_recorder_marks_required_cuda_probe_missing_when_snapshot_unavailable() {
    let mut recorder = DiagnosticsRecorder::new(
        DiagnosticsPolicy {
            required: true,
            probes: vec![DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::CudaMemorySnapshot,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            }],
            structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
        },
        crate::DiagnosticIdentity {
            experiment_run_id: "run-123".to_owned(),
            experiment_logical_name: Some("fast-test-control".to_owned()),
            species: SpeciesId::P1Contractive.as_str().to_owned(),
            variant_name: "p1_contractive_v1".to_owned(),
        },
    );

    recorder
        .record_cuda_memory_snapshot(
            crate::RunPhase::Train,
            TrainStepDiagnosticContext {
                step: 0,
                tokens_seen: 0,
            },
            crate::DiagnosticBoundary::ForwardStart,
            None,
        )
        .unwrap();

    let artifact = recorder.artifact();
    assert!(artifact.diagnostics_incomplete);
    assert_eq!(
        artifact.missing_required_probe_kinds,
        vec![DiagnosticProbeKind::CudaMemorySnapshot]
    );
}

#[test]
fn model_loss_seam_owns_forward_and_loss_diagnostics() {
    let device = Default::default();
    let rule = AddInputRule::<CpuTrainBackend>::new(4);
    let model = FractalModel::new(64, 4, 1, 1.1, PAD_TOKEN, rule, &device);
    let criterion = nn::loss::CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![PAD_TOKEN]))
        .init(&device);
    let batch = TokenBatch {
        input_ids: Tensor::<CpuTrainBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 4], [1, 4]),
            &device,
        ),
        target_ids: Tensor::<CpuTrainBackend, 2, Int>::from_data(
            TensorData::new(vec![2i64, 3, 4, PAD_TOKEN as i64], [1, 4]),
            &device,
        ),
        token_count: 4,
        family: TaskFamily::RecursiveSentence,
    };
    let mut recorder = DiagnosticsRecorder::new(
        DiagnosticsPolicy {
            required: false,
            probes: vec![
                DiagnosticProbeRequest {
                    kind: DiagnosticProbeKind::ForwardBoundary,
                    cadence: ProbeCadence::EveryStep,
                    position_interval: None,
                },
                DiagnosticProbeRequest {
                    kind: DiagnosticProbeKind::LossBoundary,
                    cadence: ProbeCadence::EveryStep,
                    position_interval: None,
                },
                DiagnosticProbeRequest {
                    kind: DiagnosticProbeKind::ForwardPosition,
                    cadence: ProbeCadence::EveryStep,
                    position_interval: Some(1),
                },
            ],
            structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
        },
        crate::DiagnosticIdentity {
            experiment_run_id: "run-123".to_owned(),
            experiment_logical_name: Some("fast-test-control".to_owned()),
            species: SpeciesId::P1Contractive.as_str().to_owned(),
            variant_name: "p1_contractive_v1".to_owned(),
        },
    );

    model
        .loss_with_diagnostics(
            &batch,
            &criterion,
            None,
            true,
            Some(&mut recorder),
            Some(TrainStepDiagnosticContext {
                step: 0,
                tokens_seen: 0,
            }),
        )
        .expect("loss seam should succeed");

    let event_names = recorder
        .artifact()
        .events
        .iter()
        .map(|event| event.event.name())
        .collect::<Vec<_>>();
    assert_eq!(
        event_names,
        vec![
            "forward_start",
            "forward_position",
            "forward_position",
            "forward_position",
            "forward_position",
            "forward_complete",
            "loss_start",
            "loss_complete",
        ]
    );
}

#[test]
fn diagnostics_recorder_marks_required_boundary_completion_missing_when_forward_truncates() {
    let mut recorder = DiagnosticsRecorder::new(
        DiagnosticsPolicy {
            required: true,
            probes: vec![DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::ForwardBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            }],
            structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
        },
        crate::DiagnosticIdentity {
            experiment_run_id: "run-123".to_owned(),
            experiment_logical_name: Some("fast-test-control".to_owned()),
            species: SpeciesId::P1Contractive.as_str().to_owned(),
            variant_name: "p1_contractive_v1".to_owned(),
        },
    );

    recorder
        .emit_event(
            crate::RunPhase::Train,
            Some(0),
            Some(0),
            DiagnosticEventKind::ForwardStart {
                input_shape: vec![1, 16],
            },
        )
        .unwrap();

    let artifact = recorder.artifact();
    assert!(artifact.diagnostics_incomplete);
    assert_eq!(
        artifact.missing_required_boundary_completions,
        vec![crate::DiagnosticBoundary::ForwardComplete]
    );
}

#[test]
fn training_runtime_emits_typed_diagnostics_for_first_step() {
    let _ = take_last_species_run_artifact();
    let mut config = TournamentPreset::FastTest.config();
    config.launch_policy.diagnostics = test_training_diagnostics_policy();
    let context = test_training_run_context(config.clone());

    let _metrics = run_species_with_factory_candle::<CountingRule<CpuTrainBackend>, _>(
        SpeciesId::P1Contractive,
        context,
        Default::default(),
        |config, _device| CountingRule::new(config.dim),
    )
    .expect("fast-test training run should succeed");

    let artifact =
        take_last_species_run_artifact().expect("successful training run should record artifact");
    let event_names = artifact
        .training_runtime
        .diagnostics
        .events
        .iter()
        .map(|event| event.event.name())
        .collect::<Vec<_>>();

    assert!(event_names.contains(&"train_step_start"));
    assert!(event_names.contains(&"forward_start"));
    assert!(event_names.contains(&"forward_position"));
    assert!(event_names.contains(&"forward_complete"));
    assert!(event_names.contains(&"loss_start"));
    assert!(event_names.contains(&"loss_complete"));
    assert!(event_names.contains(&"backward_start"));
    assert!(event_names.contains(&"backward_complete"));
    assert!(event_names.contains(&"optimizer_step_start"));
    assert!(event_names.contains(&"optimizer_step_complete"));
    assert!(artifact
        .training_runtime
        .diagnostics
        .missing_required_probe_kinds
        .is_empty());
    assert!(artifact
        .training_runtime
        .diagnostics
        .missing_required_boundary_completions
        .is_empty());
}

#[test]
fn failed_forward_records_last_successful_diagnostic_boundary() {
    let _ = take_last_species_run_artifact();
    FAILING_APPLY_COUNTER.store(0, Ordering::SeqCst);
    let mut config = TournamentPreset::FastTest.config();
    config.launch_policy.diagnostics = test_training_diagnostics_policy();
    let context = test_training_run_context(config.clone());

    let error = run_species_with_factory_candle::<FailingRule<CpuTrainBackend>, _>(
        SpeciesId::P1Contractive,
        context,
        Default::default(),
        |config, _device| FailingRule::new(config.dim, 1),
    )
    .expect_err("synthetic forward failure should bubble out");
    assert!(error.to_string().contains("synthetic forward failure"));

    let artifact =
        take_last_species_run_artifact().expect("failed training run should record artifact");
    assert!(matches!(
        artifact.execution_outcome,
        RunExecutionOutcome::InfraFailure
    ));
    assert!(artifact
        .error
        .as_deref()
        .expect("failure message should be recorded")
        .contains("synthetic forward failure"));
    let last_event = artifact
        .training_runtime
        .diagnostics
        .last_event
        .expect("last diagnostic event should be captured");
    assert_eq!(last_event.event.name(), "forward_position");
    assert!(matches!(
        last_event.event,
        DiagnosticEventKind::ForwardPosition { .. }
    ));
}

#[test]
fn runtime_persists_structured_diagnostics_before_panic_unwind() {
    let _export_lock = EXPORT_ENV_MUTEX.lock().unwrap();
    let temp_root =
        env::temp_dir().join(format!("fractal-diagnostics-unwind-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_root);
    env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &temp_root);
    let _ = take_last_species_run_artifact();
    clear_last_diagnostics_runtime_artifact();
    PANICKING_APPLY_COUNTER.store(0, Ordering::SeqCst);

    let mut config = TournamentPreset::FastTest.config();
    config.launch_policy.diagnostics = DiagnosticsPolicy {
        required: true,
        probes: vec![
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::ForwardBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::ForwardPosition,
                cadence: ProbeCadence::FirstNSteps { steps: 1 },
                position_interval: Some(1),
            },
        ],
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    };
    let context = test_training_run_context(config.clone());

    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        run_species_with_factory_candle::<PanickingRule<CpuTrainBackend>, _>(
            SpeciesId::P1Contractive,
            context,
            Default::default(),
            |config, _device| PanickingRule::new(config.dim, 1),
        )
    }));
    assert!(panic_result.is_err());

    let diagnostics =
        take_last_diagnostics_runtime_artifact().expect("panic unwind should preserve diagnostics");
    assert!(diagnostics.diagnostics_incomplete);
    assert_eq!(
        diagnostics.missing_required_boundary_completions,
        vec![crate::DiagnosticBoundary::ForwardComplete]
    );
    let event_file = diagnostics
        .event_file
        .expect("runtime diagnostics should record an event file");
    let event_stream = fs::read_to_string(&event_file).expect("event file should be readable");
    assert!(event_stream.contains("\"kind\":\"forward_start\""));
    assert!(event_stream.contains("\"kind\":\"forward_position\""));

    env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
    clear_last_diagnostics_runtime_artifact();
    let _ = fs::remove_dir_all(&temp_root);
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
        model: ModelContractSpec::recursive_kernel_v1(config.dim, config.max_recursion_depth),
        training_input: TrainingInputSpec::synthetic(),
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

fn test_training_diagnostics_policy() -> DiagnosticsPolicy {
    DiagnosticsPolicy {
        required: false,
        probes: vec![
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::TrainStep,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::ForwardBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::ForwardPosition,
                cadence: ProbeCadence::FirstNSteps { steps: 1 },
                position_interval: Some(1),
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::LossBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::BackwardBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
            DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::OptimizerBoundary,
                cadence: ProbeCadence::EveryStep,
                position_interval: None,
            },
        ],
        structured_output: crate::StructuredDiagnosticsOutput::Jsonl,
    }
}

fn test_training_run_context(config: TournamentConfig) -> SpeciesRunContext {
    let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        train_examples_per_family: (config.train_batch_size * 8).max(96),
        eval_examples_per_family: (config.eval_batch_size
            * config
                .effective_perplexity_eval_batches()
                .max(config.effective_arc_eval_batches()))
        .max(32),
        seed: config.seed,
        depth_config: config.generator_depth_config,
    })
    .expect("test generator should build");
    let species = SpeciesId::P1Contractive;
    let variant_name = test_variant_name(species);
    let experiment =
        test_experiment_template(config.clone()).resolve_variant(species, variant_name);

    SpeciesRunContext {
        index: 0,
        config,
        generator: Arc::new(generator),
        variant_name,
        experiment: Some(experiment),
    }
}

#[test]
fn tokenizer_backed_training_input_rejects_tokenizer_model_mismatch() {
    let config = TournamentPreset::FastTest.config();
    let mut template = test_experiment_template(config.clone());
    template.training_input = TrainingInputSpec::tokenizer_backed_text(
        "fineweb-stage0-smoke",
        TokenizerArtifactSpec {
            artifact_id: "mismatch-vocab".to_owned(),
            artifact_path: Some("inline://mismatch-vocab".to_owned()),
            vocab_size: config.vocab_size + 1,
            pad_token_id: PAD_TOKEN,
        },
        TextCorpusSourceSpec {
            train: TextCorpusSplitSpec {
                path: "/tmp/fineweb-train.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
            eval: TextCorpusSplitSpec {
                path: "/tmp/fineweb-eval.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
        },
    )
    .with_bridge_packaging(BridgePackagingSpec {
        vocab_artifact_path: "/tmp/fineweb-bridge-vocab.json".to_owned(),
        dim: config.dim,
        levels: config.levels,
        max_depth: config.max_recursion_depth,
        seed: config.seed,
        split_policy: BridgeSplitPolicy::Balanced,
        substrate_mode: BridgeSubstrateMode::RawBytes,
        chunk_max_tokens: config.max_seq_len,
        chunk_max_bytes: config.max_seq_len,
    });

    let error = template.validate_against_config(&config).unwrap_err();
    let error_text = error.to_string();
    assert!(error_text.contains("tokenizer vocab_size"));
    assert!(error_text.contains("must match model vocab_size"));

    let mut pad_mismatch_config = config.clone();
    pad_mismatch_config.vocab_size = 32_000;
    let mut pad_mismatch_template = test_experiment_template(pad_mismatch_config.clone());
    pad_mismatch_template.training_input = TrainingInputSpec::tokenizer_backed_text(
        "fineweb-stage0-smoke",
        TokenizerArtifactSpec {
            artifact_id: "mismatch-pad".to_owned(),
            artifact_path: Some("inline://mismatch-pad".to_owned()),
            vocab_size: pad_mismatch_config.vocab_size,
            pad_token_id: 1,
        },
        TextCorpusSourceSpec {
            train: TextCorpusSplitSpec {
                path: "/tmp/fineweb-train.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
            eval: TextCorpusSplitSpec {
                path: "/tmp/fineweb-eval.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
        },
    )
    .with_bridge_packaging(BridgePackagingSpec {
        vocab_artifact_path: "/tmp/fineweb-bridge-vocab.json".to_owned(),
        dim: pad_mismatch_config.dim,
        levels: pad_mismatch_config.levels,
        max_depth: pad_mismatch_config.max_recursion_depth,
        seed: pad_mismatch_config.seed,
        split_policy: BridgeSplitPolicy::Balanced,
        substrate_mode: BridgeSubstrateMode::RawBytes,
        chunk_max_tokens: pad_mismatch_config.max_seq_len,
        chunk_max_bytes: pad_mismatch_config.max_seq_len,
    });

    let error = pad_mismatch_template
        .validate_against_config(&pad_mismatch_config)
        .unwrap_err();
    let error_text = error.to_string();
    assert!(error_text.contains("tokenizer pad_token_id"));
    assert!(error_text.contains("must match model pad_token_id"));
}

#[test]
fn experiment_template_rejects_launch_policy_mismatch() {
    let config = TournamentPreset::FastTest.config();
    let mut template = test_experiment_template(config.clone());
    template.runtime.launch_policy = LaunchPolicySpec::stage0_default();

    let error = template.validate_against_config(&config).unwrap_err();
    assert!(error.to_string().contains("launch policy"));
}

#[test]
fn experiment_template_allows_independent_bridge_packaging_contract() {
    let config = TournamentPreset::FastTest.config();
    let mut template = test_experiment_template(config.clone());
    template.training_input = TrainingInputSpec::tokenizer_backed_text(
        "fineweb-stage0-smoke",
        TokenizerArtifactSpec {
            artifact_id: "openlm-research/open_llama_3b_v2".to_owned(),
            artifact_path: Some("inline://open-llama".to_owned()),
            vocab_size: config.vocab_size,
            pad_token_id: PAD_TOKEN,
        },
        TextCorpusSourceSpec {
            train: TextCorpusSplitSpec {
                path: "/tmp/fineweb-train.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
            eval: TextCorpusSplitSpec {
                path: "/tmp/fineweb-eval.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
        },
    )
    .with_bridge_packaging(BridgePackagingSpec {
        vocab_artifact_path: "/tmp/fineweb-bridge-vocab.json".to_owned(),
        dim: 64,
        levels: 3,
        max_depth: 6,
        seed: 7,
        split_policy: BridgeSplitPolicy::Balanced,
        substrate_mode: BridgeSubstrateMode::RawBytes,
        chunk_max_tokens: config.max_seq_len,
        chunk_max_bytes: config.max_seq_len,
    });

    template
        .validate_against_config(&config)
        .expect("bridge packaging should be independently configurable");
}

#[test]
fn experiment_template_rejects_bridge_chunk_limit_above_model_context() {
    let config = TournamentPreset::FastTest.config();
    let mut template = test_experiment_template(config.clone());
    template.training_input = TrainingInputSpec::tokenizer_backed_text(
        "fineweb-stage0-smoke",
        TokenizerArtifactSpec {
            artifact_id: "openlm-research/open_llama_3b_v2".to_owned(),
            artifact_path: Some("inline://open-llama".to_owned()),
            vocab_size: config.vocab_size,
            pad_token_id: PAD_TOKEN,
        },
        TextCorpusSourceSpec {
            train: TextCorpusSplitSpec {
                path: "/tmp/fineweb-train.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
            eval: TextCorpusSplitSpec {
                path: "/tmp/fineweb-eval.jsonl".to_owned(),
                format: TextCorpusFormat::JsonlText {
                    text_field: "text".to_owned(),
                },
                max_documents: None,
            },
        },
    )
    .with_bridge_packaging(BridgePackagingSpec {
        vocab_artifact_path: "/tmp/fineweb-bridge-vocab.json".to_owned(),
        dim: 64,
        levels: 3,
        max_depth: 6,
        seed: 7,
        split_policy: BridgeSplitPolicy::Balanced,
        substrate_mode: BridgeSubstrateMode::RawBytes,
        chunk_max_tokens: config.max_seq_len + 1,
        chunk_max_bytes: config.max_seq_len + 1,
    });

    let error = template.validate_against_config(&config).unwrap_err();
    assert!(error.to_string().contains("chunk_max_tokens"));
    assert!(error.to_string().contains("max_seq_len"));
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
