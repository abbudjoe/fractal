use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use fractal::{
    aggregate_results,
    error::FractalError,
    lifecycle::{
        ArtifactPolicy, BenchmarkMode, BudgetSpec, ComparisonContract, DecisionIntent,
        ExecutionBackend, ExecutionTarget, ExecutionTargetKind, ExperimentId, ExperimentQuestion,
        ExperimentSpecTemplate, LaneIntent, ModelContractSpec, OptimizerSpec, RunExecutionOutcome,
        RuntimeSurfaceSpec, SpeciesCompletion, SpeciesRunStage, Tournament, TournamentConfig,
        TournamentPreset, TournamentProgressEvent, TournamentReporter, TournamentSequence,
        TrainingInputMode, TrainingInputSpec,
    },
    materialize_bridge_vocab_artifact, persist_run_artifacts, primitive_tracker_reminder_lines,
    registry::{
        resolve_precision_profile, CandleBf16TrainBackend, CandleF32TrainBackend, ComputeBackend,
        ExecutionMode, MetalF32TrainBackend, ResolvedExecutablePrecisionProfile, SpeciesId,
    },
    run_tokenizer_backed_species_from_experiment, species_registry_for_lane,
    species_registry_for_species, TournamentLane, TournamentRunReport, TournamentRunReportParts,
};
#[cfg(feature = "cuda")]
use fractal_core::registry::cuda_device;
use fractal_core::registry::{
    cpu_device, initialize_metal_runtime, take_last_species_run_artifact,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn main() -> Result<(), FractalError> {
    match parse_command(std::env::args().skip(1))? {
        CliCommand::Help => {
            print_usage();
            Ok(())
        }
        CliCommand::PrepareStage0Assets(options) => prepare_stage0_assets(&options),
        CliCommand::Run(options) => run_options(&options),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CliCommand {
    Help,
    PrepareStage0Assets(RunOptions),
    Run(RunOptions),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunSelection {
    Preset(TournamentPreset),
    Sequence(TournamentSequence),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendOverride {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    Metal,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RunOptions {
    selection: Option<RunSelection>,
    lane: Option<TournamentLane>,
    species: Option<SpeciesId>,
    seed: Option<u64>,
    execution_mode: Option<ExecutionMode>,
    parallelism: Option<usize>,
    backend: Option<BackendOverride>,
    perplexity_eval_batches: Option<usize>,
    arc_eval_batches: Option<usize>,
    benchmark_mode: Option<BenchmarkMode>,
    manifest_path: Option<PathBuf>,
    prepare_stage0_assets: bool,
    logical_name: Option<String>,
    question_summary: Option<String>,
    comparison_override: Option<ComparisonContract>,
}

impl RunOptions {
    fn ensure_manifest_isolated(&self) -> Result<(), FractalError> {
        let mut normalized = self.clone();
        normalized.backend = None;
        normalized.prepare_stage0_assets = false;
        if normalized != Self::default() {
            return Err(invalid_argument(
                "--experiment-manifest cannot be combined with other run-shaping flags".to_owned(),
            ));
        }
        Ok(())
    }

    fn ensure_no_manifest_source(&self) -> Result<(), FractalError> {
        if self.manifest_path.is_some() {
            return Err(invalid_argument(
                "--experiment-manifest must be the only run-shaping input".to_owned(),
            ));
        }
        Ok(())
    }

    fn selection(&self) -> RunSelection {
        self.selection.unwrap_or_else(|| {
            let preset = if self.species.is_some() {
                TournamentPreset::CandidateStress
            } else {
                self.lane.unwrap_or(TournamentLane::All).default_preset()
            };
            RunSelection::Preset(preset)
        })
    }

    fn set_selection(&mut self, selection: RunSelection) -> Result<(), FractalError> {
        if self.selection.replace(selection).is_some() {
            return Err(invalid_argument(
                "choose either --preset or --sequence, not both".to_owned(),
            ));
        }
        Ok(())
    }

    fn set_lane(&mut self, lane: TournamentLane) -> Result<(), FractalError> {
        if self.species.is_some() {
            return Err(invalid_argument(
                "choose either --lane or --species, not both".to_owned(),
            ));
        }
        if self.lane.replace(lane).is_some() {
            return Err(invalid_argument(
                "choose one --lane value for a run".to_owned(),
            ));
        }
        Ok(())
    }

    fn set_species(&mut self, species: SpeciesId) -> Result<(), FractalError> {
        if self.lane.is_some() {
            return Err(invalid_argument(
                "choose either --lane or --species, not both".to_owned(),
            ));
        }
        if self.species.replace(species).is_some() {
            return Err(invalid_argument(
                "choose one --species value for a run".to_owned(),
            ));
        }
        Ok(())
    }

    fn lane(&self) -> TournamentLane {
        self.lane.unwrap_or(TournamentLane::All)
    }

    fn species(&self) -> Option<SpeciesId> {
        self.species
    }

    fn config_for(&self, preset: TournamentPreset) -> Result<TournamentConfig, FractalError> {
        let mut config = preset.config();
        if let Some(seed) = self.seed {
            config.seed = seed;
        }
        if let Some(execution_mode) = self.execution_mode {
            config.execution_mode = execution_mode;
        }
        if let Some(parallelism) = self.parallelism {
            config.parallelism = parallelism;
        }
        if let Some(backend) = self.backend {
            config.execution_backend = backend.into();
        }
        if let Some(perplexity_eval_batches) = self.perplexity_eval_batches {
            config.perplexity_eval_batches = Some(perplexity_eval_batches);
        }
        if let Some(arc_eval_batches) = self.arc_eval_batches {
            config.arc_eval_batches = Some(arc_eval_batches);
        }
        if let Some(path) = self.manifest_path.as_deref() {
            if let Some(manifest) = load_manifest_v2(path)? {
                manifest.experiment.config.apply(&mut config)?;
                if let Some(optimizer) = manifest.experiment.optimizer {
                    config = config.with_optimizer(optimizer);
                }
                if let Some(runtime) = manifest.experiment.runtime {
                    config = config.with_launch_policy(runtime.launch_policy);
                }
            }
        }
        Ok(config)
    }

    fn comparison_for(&self, fallback: ComparisonContract) -> ComparisonContract {
        self.comparison_override.clone().unwrap_or(fallback)
    }

    fn runtime_surface_spec(&self) -> RuntimeSurfaceSpec {
        let mut runtime = RuntimeSurfaceSpec::default();
        if let Some(benchmark_mode) = self.benchmark_mode {
            runtime.benchmark_mode = benchmark_mode;
        }
        runtime
    }

    fn merge_manifest(self, path: &Path) -> Result<Self, FractalError> {
        let explicit_backend = self.backend;
        let mut loaded = load_manifest_run_options(path)?;
        if let Some(explicit_backend) = explicit_backend {
            if let Some(manifest_backend) = loaded.backend {
                if manifest_backend != explicit_backend {
                    return Err(invalid_argument(format!(
                        "--backend {} conflicts with experiment manifest backend {}",
                        backend_override_name(explicit_backend),
                        backend_override_name(manifest_backend),
                    )));
                }
            }
            loaded.backend = Some(explicit_backend);
        }
        Ok(loaded)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExperimentManifestFile {
    logical_name: String,
    #[serde(default)]
    question_summary: Option<String>,
    preset: String,
    #[serde(default)]
    lane: Option<String>,
    #[serde(default)]
    species: Option<String>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    execution_mode: Option<String>,
    #[serde(default)]
    parallelism: Option<usize>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    perplexity_eval_batches: Option<usize>,
    #[serde(default)]
    arc_eval_batches: Option<usize>,
    #[serde(default)]
    comparison: Option<String>,
    #[serde(default)]
    benchmark_mode: Option<String>,
    #[serde(default)]
    expected_branch: Option<String>,
    #[serde(default)]
    expected_commit_sha: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ManifestQuestionOverrides {
    lane_intent: Option<LaneIntent>,
    decision_intent: Option<DecisionIntent>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ManifestConfigOverrides {
    seed: Option<u64>,
    dim: Option<usize>,
    levels: Option<usize>,
    vocab_size: Option<usize>,
    max_seq_len: Option<usize>,
    max_recursion_depth: Option<usize>,
    stability_depth: Option<usize>,
    router_threshold: Option<f32>,
    train_batch_size: Option<usize>,
    eval_batch_size: Option<usize>,
    train_steps_per_species: Option<usize>,
    train_token_budget: Option<usize>,
    eval_batches_per_family: Option<usize>,
    perplexity_eval_batches: Option<usize>,
    arc_eval_batches: Option<usize>,
    execution_mode: Option<String>,
    parallelism: Option<usize>,
    backend: Option<String>,
    timeout_seconds: Option<f64>,
}

impl ManifestConfigOverrides {
    fn apply(&self, config: &mut TournamentConfig) -> Result<(), FractalError> {
        self.apply_with_backend_policy(config, true)
    }

    fn apply_without_backend(&self, config: &mut TournamentConfig) -> Result<(), FractalError> {
        self.apply_with_backend_policy(config, false)
    }

    fn apply_with_backend_policy(
        &self,
        config: &mut TournamentConfig,
        parse_backend_override: bool,
    ) -> Result<(), FractalError> {
        if let Some(seed) = self.seed {
            config.seed = seed;
        }
        apply_positive_override(&mut config.dim, self.dim, "dim")?;
        apply_positive_override(&mut config.levels, self.levels, "levels")?;
        apply_positive_override(&mut config.vocab_size, self.vocab_size, "vocab_size")?;
        apply_positive_override(&mut config.max_seq_len, self.max_seq_len, "max_seq_len")?;
        apply_positive_override(
            &mut config.max_recursion_depth,
            self.max_recursion_depth,
            "max_recursion_depth",
        )?;
        apply_positive_override(
            &mut config.stability_depth,
            self.stability_depth,
            "stability_depth",
        )?;
        if let Some(router_threshold) = self.router_threshold {
            config.router_threshold = router_threshold;
        }
        apply_positive_override(
            &mut config.train_batch_size,
            self.train_batch_size,
            "train_batch_size",
        )?;
        apply_positive_override(
            &mut config.eval_batch_size,
            self.eval_batch_size,
            "eval_batch_size",
        )?;
        apply_positive_override(
            &mut config.train_steps_per_species,
            self.train_steps_per_species,
            "train_steps_per_species",
        )?;
        if let Some(train_token_budget) = self.train_token_budget {
            if train_token_budget == 0 {
                return Err(invalid_argument(
                    "manifest v2 train_token_budget must be greater than zero".to_owned(),
                ));
            }
            config.train_token_budget = Some(train_token_budget);
        }
        apply_positive_override(
            &mut config.eval_batches_per_family,
            self.eval_batches_per_family,
            "eval_batches_per_family",
        )?;
        if let Some(perplexity_eval_batches) = self.perplexity_eval_batches {
            if perplexity_eval_batches == 0 {
                return Err(invalid_argument(
                    "manifest v2 perplexity_eval_batches must be greater than zero".to_owned(),
                ));
            }
            config.perplexity_eval_batches = Some(perplexity_eval_batches);
        }
        if let Some(arc_eval_batches) = self.arc_eval_batches {
            if arc_eval_batches == 0 {
                return Err(invalid_argument(
                    "manifest v2 arc_eval_batches must be greater than zero".to_owned(),
                ));
            }
            config.arc_eval_batches = Some(arc_eval_batches);
        }
        if let Some(execution_mode) = self.execution_mode.as_deref() {
            config.execution_mode = parse_execution_mode(execution_mode)?;
        }
        if let Some(parallelism) = self.parallelism {
            if parallelism == 0 {
                return Err(invalid_argument(
                    "manifest v2 parallelism must be greater than zero".to_owned(),
                ));
            }
            config.parallelism = parallelism;
        }
        if parse_backend_override {
            if let Some(backend) = self.backend.as_deref() {
                config.execution_backend = parse_backend(backend)?.into();
            }
        }
        if let Some(timeout_seconds) = self.timeout_seconds {
            if timeout_seconds <= 0.0 {
                return Err(invalid_argument(
                    "manifest v2 timeout_seconds must be greater than zero".to_owned(),
                ));
            }
            config.run_timeout = Some(std::time::Duration::from_secs_f64(timeout_seconds));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct ExperimentManifestV2Spec {
    question: ManifestQuestionOverrides,
    config: ManifestConfigOverrides,
    model: Option<ModelContractSpec>,
    training_input: Option<TrainingInputSpec>,
    optimizer: Option<OptimizerSpec>,
    runtime: Option<RuntimeSurfaceSpec>,
    comparison: Option<ComparisonContract>,
    artifacts: Option<ArtifactPolicy>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ExperimentManifestFileV2 {
    logical_name: String,
    #[serde(default)]
    question_summary: Option<String>,
    preset: String,
    #[serde(default)]
    lane: Option<String>,
    #[serde(default)]
    species: Option<String>,
    #[serde(default)]
    expected_branch: Option<String>,
    #[serde(default)]
    expected_commit_sha: Option<String>,
    experiment: ExperimentManifestV2Spec,
}

#[derive(Clone, Debug)]
enum LoadedManifest {
    Legacy(Box<ExperimentManifestFile>),
    V2(Box<ExperimentManifestFileV2>),
}

impl ExperimentManifestFile {
    fn into_run_options(self, path: &Path) -> Result<RunOptions, FractalError> {
        if self.logical_name.trim().is_empty() {
            return Err(invalid_argument(format!(
                "experiment manifest {} must define a non-empty logical_name",
                path.display()
            )));
        }
        if self.lane.is_some() && self.species.is_some() {
            return Err(invalid_argument(format!(
                "experiment manifest {} must choose either lane or species, not both",
                path.display()
            )));
        }

        validate_manifest_identity(path, &self.expected_branch, &self.expected_commit_sha)?;

        Ok(RunOptions {
            selection: Some(RunSelection::Preset(parse_preset(&self.preset)?)),
            lane: self.lane.as_deref().map(parse_lane).transpose()?,
            species: self.species.as_deref().map(parse_species).transpose()?,
            seed: self.seed,
            execution_mode: self
                .execution_mode
                .as_deref()
                .map(parse_execution_mode)
                .transpose()?,
            parallelism: self.parallelism,
            backend: self.backend.as_deref().map(parse_backend).transpose()?,
            perplexity_eval_batches: self.perplexity_eval_batches,
            arc_eval_batches: self.arc_eval_batches,
            benchmark_mode: Some(parse_benchmark_mode(
                self.benchmark_mode.as_deref().unwrap_or("leaderboard"),
            )?),
            manifest_path: Some(path.to_path_buf()),
            prepare_stage0_assets: false,
            logical_name: Some(self.logical_name),
            question_summary: self.question_summary,
            comparison_override: Some(parse_comparison_name(
                self.comparison
                    .as_deref()
                    .unwrap_or("authoritative_same_preset"),
            )?),
        })
    }
}

impl ExperimentManifestFileV2 {
    fn into_run_options(self, path: &Path) -> Result<RunOptions, FractalError> {
        if self.logical_name.trim().is_empty() {
            return Err(invalid_argument(format!(
                "experiment manifest {} must define a non-empty logical_name",
                path.display()
            )));
        }
        if self.lane.is_some() && self.species.is_some() {
            return Err(invalid_argument(format!(
                "experiment manifest {} must choose either lane or species, not both",
                path.display()
            )));
        }

        validate_manifest_identity(path, &self.expected_branch, &self.expected_commit_sha)?;

        Ok(RunOptions {
            selection: Some(RunSelection::Preset(parse_preset(&self.preset)?)),
            lane: self.lane.as_deref().map(parse_lane).transpose()?,
            species: self.species.as_deref().map(parse_species).transpose()?,
            seed: self.experiment.config.seed,
            execution_mode: self
                .experiment
                .config
                .execution_mode
                .as_deref()
                .map(parse_execution_mode)
                .transpose()?,
            parallelism: self.experiment.config.parallelism,
            backend: self
                .experiment
                .config
                .backend
                .as_deref()
                .map(parse_backend)
                .transpose()?,
            perplexity_eval_batches: self.experiment.config.perplexity_eval_batches,
            arc_eval_batches: self.experiment.config.arc_eval_batches,
            benchmark_mode: Some(
                self.experiment
                    .runtime
                    .as_ref()
                    .map(|runtime| runtime.benchmark_mode)
                    .unwrap_or(BenchmarkMode::Leaderboard),
            ),
            manifest_path: Some(path.to_path_buf()),
            prepare_stage0_assets: false,
            logical_name: Some(self.logical_name),
            question_summary: self.question_summary,
            comparison_override: Some(
                self.experiment
                    .comparison
                    .unwrap_or_else(ComparisonContract::authoritative_same_preset),
            ),
        })
    }
}

impl LoadedManifest {
    fn into_run_options(self, path: &Path) -> Result<RunOptions, FractalError> {
        match self {
            Self::Legacy(manifest) => (*manifest).into_run_options(path),
            Self::V2(manifest) => (*manifest).into_run_options(path),
        }
    }
}

fn load_manifest_file(path: &Path) -> Result<LoadedManifest, FractalError> {
    let content = fs::read_to_string(path)
        .map_err(|error| invalid_argument(format!("failed to read {}: {error}", path.display())))?;
    let value: Value = serde_json::from_str(&content).map_err(|error| {
        invalid_argument(format!(
            "failed to parse experiment manifest {}: {error}",
            path.display()
        ))
    })?;

    if value.get("experiment").is_some() {
        let manifest: ExperimentManifestFileV2 =
            serde_json::from_value(value).map_err(|error| {
                invalid_argument(format!(
                    "failed to parse experiment manifest v2 {}: {error}",
                    path.display()
                ))
            })?;
        Ok(LoadedManifest::V2(Box::new(manifest)))
    } else {
        let manifest: ExperimentManifestFile = serde_json::from_value(value).map_err(|error| {
            invalid_argument(format!(
                "failed to parse experiment manifest {}: {error}",
                path.display()
            ))
        })?;
        Ok(LoadedManifest::Legacy(Box::new(manifest)))
    }
}

fn load_manifest_v2(path: &Path) -> Result<Option<ExperimentManifestFileV2>, FractalError> {
    match load_manifest_file(path)? {
        LoadedManifest::Legacy(_) => Ok(None),
        LoadedManifest::V2(manifest) => Ok(Some(*manifest)),
    }
}

fn load_manifest_run_options(path: &Path) -> Result<RunOptions, FractalError> {
    load_manifest_file(path)?.into_run_options(path)
}

fn validate_manifest_identity(
    path: &Path,
    expected_branch: &Option<String>,
    expected_commit_sha: &Option<String>,
) -> Result<(), FractalError> {
    if let Some(expected_branch) = expected_branch {
        let current_branch = detect_git_ref("FRACTAL_BRANCH", &["rev-parse", "--abbrev-ref", "HEAD"])
            .ok_or_else(|| {
                invalid_argument(format!(
                    "experiment manifest {} requires branch {}, but the current branch could not be detected",
                    path.display(),
                    expected_branch
                ))
            })?;
        if current_branch != *expected_branch {
            return Err(invalid_argument(format!(
                "experiment manifest {} requires branch {}, found {}",
                path.display(),
                expected_branch,
                current_branch
            )));
        }
    }

    if let Some(expected_commit_sha) = expected_commit_sha {
        let current_commit = detect_git_ref("FRACTAL_COMMIT_SHA", &["rev-parse", "HEAD"])
            .ok_or_else(|| {
                invalid_argument(format!(
                    "experiment manifest {} requires commit {}, but the current commit could not be detected",
                    path.display(),
                    expected_commit_sha
                ))
            })?;
        if current_commit != *expected_commit_sha {
            return Err(invalid_argument(format!(
                "experiment manifest {} requires commit {}, found {}",
                path.display(),
                expected_commit_sha,
                current_commit
            )));
        }
    }

    Ok(())
}

impl From<BackendOverride> for ComputeBackend {
    fn from(value: BackendOverride) -> Self {
        match value {
            BackendOverride::Cpu => ComputeBackend::CpuCandle,
            #[cfg(feature = "cuda")]
            BackendOverride::Cuda => ComputeBackend::cuda_default(),
            BackendOverride::Metal => ComputeBackend::metal_default(),
        }
    }
}

fn parse_command<I>(args: I) -> Result<CliCommand, FractalError>
where
    I: IntoIterator<Item = String>,
{
    let collected_args = args.into_iter().collect::<Vec<_>>();
    let prepare_mode = collected_args
        .iter()
        .any(|arg| arg == "--prepare-stage0-assets");
    let mut options = RunOptions::default();
    let mut args = collected_args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--help" {
            return Ok(CliCommand::Help);
        }
        parse_arg(&mut options, &mut args, arg, prepare_mode)?;
    }
    if prepare_mode {
        options.prepare_stage0_assets = true;
        if options.manifest_path.is_none() {
            return Err(invalid_argument(
                "--prepare-stage0-assets requires --experiment-manifest".to_owned(),
            ));
        }
        return Ok(CliCommand::PrepareStage0Assets(options));
    }
    Ok(CliCommand::Run(options))
}

fn parse_arg<I>(
    options: &mut RunOptions,
    args: &mut I,
    arg: String,
    prepare_mode: bool,
) -> Result<(), FractalError>
where
    I: Iterator<Item = String>,
{
    match arg.as_str() {
        "--experiment-manifest" => {
            options.ensure_manifest_isolated()?;
            let value = next_value(args, "--experiment-manifest")?;
            if prepare_mode {
                options.manifest_path = Some(PathBuf::from(value));
            } else {
                *options = options.clone().merge_manifest(Path::new(&value))?;
            }
            Ok(())
        }
        "--prepare-stage0-assets" => {
            options.prepare_stage0_assets = true;
            Ok(())
        }
        "--preset" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--preset")?;
            options.set_selection(RunSelection::Preset(parse_preset(&value)?))?;
            Ok(())
        }
        "--sequence" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--sequence")?;
            options.set_selection(RunSelection::Sequence(parse_sequence(&value)?))?;
            Ok(())
        }
        "--lane" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--lane")?;
            options.set_lane(parse_lane(&value)?)?;
            Ok(())
        }
        "--species" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--species")?;
            options.set_species(parse_species(&value)?)?;
            Ok(())
        }
        "--seed" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--seed")?;
            options.seed = Some(parse_seed(&value)?);
            Ok(())
        }
        "--mode" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--mode")?;
            options.execution_mode = Some(parse_execution_mode(&value)?);
            Ok(())
        }
        "--parallelism" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--parallelism")?;
            options.parallelism = Some(parse_parallelism(&value)?);
            Ok(())
        }
        "--backend" => {
            let value = next_value(args, "--backend")?;
            let backend = parse_backend(&value)?;
            if options.manifest_path.is_some() {
                if let Some(current_backend) = options.backend {
                    if current_backend != backend {
                        return Err(invalid_argument(format!(
                            "--backend {} conflicts with experiment manifest backend {}",
                            backend_override_name(backend),
                            backend_override_name(current_backend),
                        )));
                    }
                }
                options.backend = Some(backend);
                return Ok(());
            }
            options.ensure_no_manifest_source()?;
            options.backend = Some(backend);
            Ok(())
        }
        "--perplexity-eval-batches" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--perplexity-eval-batches")?;
            options.perplexity_eval_batches =
                Some(parse_positive_usize(&value, "--perplexity-eval-batches")?);
            Ok(())
        }
        "--arc-eval-batches" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--arc-eval-batches")?;
            options.arc_eval_batches = Some(parse_positive_usize(&value, "--arc-eval-batches")?);
            Ok(())
        }
        "--benchmark-mode" => {
            options.ensure_no_manifest_source()?;
            let value = next_value(args, "--benchmark-mode")?;
            options.benchmark_mode = Some(parse_benchmark_mode(&value)?);
            Ok(())
        }
        _ => Err(invalid_argument(format!("unknown argument: {arg}"))),
    }
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<String, FractalError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| invalid_argument(format!("missing value for {flag}")))
}

fn parse_preset(value: &str) -> Result<TournamentPreset, FractalError> {
    match value {
        "default" => Ok(TournamentPreset::Default),
        "fast-test" => Ok(TournamentPreset::FastTest),
        "research-medium" => Ok(TournamentPreset::ResearchMedium),
        "challenger-lane" => Ok(TournamentPreset::ChallengerLane),
        "minimal-baseline" => Ok(TournamentPreset::MinimalBaseline),
        "minimal-stress-lane" | "minimal_stress_lane" => Ok(TournamentPreset::MinimalStressLane),
        "minimal-proving-ground" => Ok(TournamentPreset::MinimalProvingGround),
        "proving-ground-baseline" | "proving_ground_baseline" => {
            Ok(TournamentPreset::ProvingGroundBaseline)
        }
        "bullpen-polish" => Ok(TournamentPreset::BullpenPolish),
        "lighter-intermediate-stress" | "lighter_intermediate_stress" => {
            Ok(TournamentPreset::LighterIntermediateStress)
        }
        "intermediate-stress" => Ok(TournamentPreset::IntermediateStress),
        "full-medium-stress" | "full_medium_stress" => Ok(TournamentPreset::FullMediumStress),
        "medium-stress" => Ok(TournamentPreset::MediumStress),
        "pressure-test" => Ok(TournamentPreset::PressureTest),
        "candidate-stress" => Ok(TournamentPreset::CandidateStress),
        "generation-four" => Ok(TournamentPreset::GenerationFour),
        _ => Err(invalid_argument(format!("unknown preset: {value}"))),
    }
}

fn parse_sequence(value: &str) -> Result<TournamentSequence, FractalError> {
    match value {
        "first-run" => Ok(TournamentSequence::FirstRun),
        _ => Err(invalid_argument(format!("unknown sequence: {value}"))),
    }
}

fn parse_seed(value: &str) -> Result<u64, FractalError> {
    value
        .parse::<u64>()
        .map_err(|_| invalid_argument(format!("invalid seed: {value}")))
}

fn parse_lane(value: &str) -> Result<TournamentLane, FractalError> {
    match value {
        "all" => Ok(TournamentLane::All),
        "baseline" => Ok(TournamentLane::Baseline),
        "challenger" | "bullpen" => Ok(TournamentLane::Challenger),
        "proving-ground" | "squaring" => Ok(TournamentLane::ProvingGround),
        "leader" => Ok(TournamentLane::Leader),
        _ => Err(invalid_argument(format!("unknown lane: {value}"))),
    }
}

fn parse_species(value: &str) -> Result<SpeciesId, FractalError> {
    value
        .parse::<SpeciesId>()
        .map_err(|_| invalid_argument(format!("unknown species: {value}")))
}

fn parse_execution_mode(value: &str) -> Result<ExecutionMode, FractalError> {
    match value {
        "sequential" => Ok(ExecutionMode::Sequential),
        "parallel" => Ok(ExecutionMode::Parallel),
        _ => Err(invalid_argument(format!("unknown execution mode: {value}"))),
    }
}

fn parse_parallelism(value: &str) -> Result<usize, FractalError> {
    parse_positive_usize(value, "--parallelism")
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, FractalError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| invalid_argument(format!("invalid value for {flag}: {value}")))?;
    if parsed == 0 {
        return Err(invalid_argument(format!(
            "{flag} must be greater than zero"
        )));
    }
    Ok(parsed)
}

fn apply_positive_override(
    slot: &mut usize,
    value: Option<usize>,
    field_name: &str,
) -> Result<(), FractalError> {
    if let Some(value) = value {
        if value == 0 {
            return Err(invalid_argument(format!(
                "manifest v2 {field_name} must be greater than zero"
            )));
        }
        *slot = value;
    }
    Ok(())
}

fn parse_backend(value: &str) -> Result<BackendOverride, FractalError> {
    match value {
        "cpu" => Ok(BackendOverride::Cpu),
        #[cfg(feature = "cuda")]
        "cuda" => Ok(BackendOverride::Cuda),
        "metal" => Ok(BackendOverride::Metal),
        _ => Err(invalid_argument(format!("unknown backend: {value}"))),
    }
}

fn parse_benchmark_mode(value: &str) -> Result<BenchmarkMode, FractalError> {
    match value {
        "leaderboard" => Ok(BenchmarkMode::Leaderboard),
        "systems-speed" | "systems_speed" => Ok(BenchmarkMode::SystemsSpeed),
        _ => Err(invalid_argument(format!("unknown benchmark mode: {value}"))),
    }
}

fn parse_comparison_name(value: &str) -> Result<ComparisonContract, FractalError> {
    match value {
        "authoritative_same_preset" | "authoritative same-preset" => {
            Ok(ComparisonContract::authoritative_same_preset())
        }
        "advisory_mixed_preset" | "advisory mixed-preset" => {
            Ok(ComparisonContract::advisory_mixed_preset())
        }
        "advisory_same_preset" | "advisory same-preset" => Ok(ComparisonContract {
            authority: fractal::ComparisonAuthority::Advisory,
            requires_same_preset: true,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: false,
            requires_same_backend: true,
        }),
        "authoritative_mixed_preset" | "authoritative mixed-preset" => Ok(ComparisonContract {
            authority: fractal::ComparisonAuthority::Authoritative,
            requires_same_preset: false,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: true,
            requires_same_backend: true,
        }),
        _ => Err(invalid_argument(format!(
            "unknown comparison contract: {value}"
        ))),
    }
}

fn run_options(options: &RunOptions) -> Result<(), FractalError> {
    match options.selection() {
        RunSelection::Preset(preset) => run_preset(
            options,
            preset,
            options.comparison_for(ComparisonContract::authoritative_same_preset()),
        ),
        RunSelection::Sequence(sequence) => {
            if options.manifest_path.is_some() {
                return Err(invalid_argument(
                    "experiment manifests cannot use sequence mode".to_owned(),
                ));
            }
            run_sequence(options, sequence)
        }
    }
}

fn prepare_stage0_assets(options: &RunOptions) -> Result<(), FractalError> {
    let manifest_path = options.manifest_path.as_deref().ok_or_else(|| {
        invalid_argument("--prepare-stage0-assets requires --experiment-manifest".to_owned())
    })?;
    let manifest = load_manifest_v2(manifest_path)?.ok_or_else(|| {
        invalid_argument("--prepare-stage0-assets requires a manifest v2 experiment".to_owned())
    })?;
    validate_manifest_identity(
        manifest_path,
        &manifest.expected_branch,
        &manifest.expected_commit_sha,
    )?;
    let preset = match options.selection() {
        RunSelection::Preset(preset) => preset,
        RunSelection::Sequence(_) => {
            return Err(invalid_argument(
                "Stage 0 asset preparation does not support sequence manifests".to_owned(),
            ));
        }
    };
    let mut config = preset.config();
    manifest
        .experiment
        .config
        .apply_without_backend(&mut config)?;
    if let Some(optimizer) = manifest.experiment.optimizer.clone() {
        config = config.with_optimizer(optimizer);
    }
    if let Some(runtime) = manifest.experiment.runtime.clone() {
        config = config.with_launch_policy(runtime.launch_policy);
    }
    let lane = options.lane();
    let species = options.species();
    let comparison = options.comparison_for(ComparisonContract::authoritative_same_preset());
    let template = build_experiment_template(
        preset,
        lane,
        species,
        &comparison,
        &config,
        options,
        Some(&manifest),
    );
    template.validate_against_config(&config)?;
    let output_path = materialize_bridge_vocab_artifact(&template.training_input, &config)?;
    println!(
        "Materialized Stage 0 bridge vocab artifact at {}",
        output_path.display()
    );
    let frozen_manifest_path = materialize_frozen_manifest_v2(manifest_path, &manifest)?;
    println!(
        "Materialized frozen Stage 0 manifest at {}",
        frozen_manifest_path.display()
    );
    Ok(())
}

fn current_git_commit_sha() -> Result<String, FractalError> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|error| {
            invalid_argument(format!("failed to resolve current git commit sha: {error}"))
        })?;
    if !output.status.success() {
        return Err(invalid_argument(format!(
            "failed to resolve current git commit sha: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let sha = String::from_utf8(output.stdout).map_err(|error| {
        invalid_argument(format!(
            "git rev-parse returned non-utf8 output for current commit sha: {error}"
        ))
    })?;
    let sha = sha.trim().to_owned();
    if sha.is_empty() {
        return Err(invalid_argument(
            "git rev-parse returned an empty current commit sha".to_owned(),
        ));
    }
    Ok(sha)
}

fn resolve_frozen_manifest_output_path(manifest: &ExperimentManifestFileV2) -> PathBuf {
    if let Some(root) = std::env::var_os("FRACTAL_RUN_MANIFEST_DIR") {
        PathBuf::from(root).join(format!("{}.frozen.json", manifest.logical_name))
    } else {
        std::env::temp_dir()
            .join("fractal-prepared-manifests")
            .join(format!("{}.frozen.json", manifest.logical_name))
    }
}

fn materialize_frozen_manifest_v2(
    source_path: &Path,
    manifest: &ExperimentManifestFileV2,
) -> Result<PathBuf, FractalError> {
    let mut frozen = manifest.clone();
    frozen.expected_commit_sha = Some(current_git_commit_sha()?);
    let output_path = resolve_frozen_manifest_output_path(&frozen);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            invalid_argument(format!(
                "failed to create frozen manifest directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&frozen).map_err(|error| {
            invalid_argument(format!(
                "failed to serialize frozen manifest for {}: {error}",
                source_path.display()
            ))
        })?,
    )
    .map_err(|error| {
        invalid_argument(format!(
            "failed to write frozen manifest for {} to {}: {error}",
            source_path.display(),
            output_path.display()
        ))
    })?;
    Ok(output_path)
}

fn run_sequence(options: &RunOptions, sequence: TournamentSequence) -> Result<(), FractalError> {
    for (index, preset) in sequence.stages().iter().copied().enumerate() {
        if index > 0 {
            println!();
        }
        run_preset(options, preset, ComparisonContract::advisory_mixed_preset())?;
    }
    Ok(())
}

fn run_preset(
    options: &RunOptions,
    preset: TournamentPreset,
    comparison: ComparisonContract,
) -> Result<(), FractalError> {
    let lane = options.lane();
    let species = options.species();
    let manifest_v2 = options
        .manifest_path
        .as_deref()
        .map(load_manifest_v2)
        .transpose()?
        .flatten();
    let base_config = options.config_for(preset)?;
    let experiment_template = build_experiment_template(
        preset,
        lane,
        species,
        &comparison,
        &base_config,
        options,
        manifest_v2.as_ref(),
    );
    let config = base_config
        .clone()
        .with_experiment(experiment_template.clone());
    print_header(preset, lane, species, &config, &comparison);
    if experiment_template.training_input.mode == TrainingInputMode::TokenizerBackedText {
        return run_tokenizer_backed_preset(preset, lane, species, comparison, config);
    }
    let tournament = Tournament::new(config.clone())?;
    let reporter: Arc<dyn TournamentReporter> = Arc::new(StdoutProgressReporter);
    let species = if let Some(species) = species {
        species_registry_for_species(species)
    } else {
        species_registry_for_lane(lane)
    };
    let artifact = tournament.run_generation_artifacts(&species, Some(reporter))?;
    let results = aggregate_results(
        artifact
            .species
            .iter()
            .filter_map(|record| record.metrics.clone())
            .collect::<Vec<_>>(),
    );
    let report = TournamentRunReport::new(TournamentRunReportParts {
        preset,
        lane,
        comparison,
        config,
        species,
        results,
        artifact,
        bridge_stats: BTreeMap::new(),
    });
    emit_persisted_report(&report)
}

fn run_tokenizer_backed_preset(
    preset: TournamentPreset,
    lane: TournamentLane,
    species: Option<SpeciesId>,
    comparison: ComparisonContract,
    config: TournamentConfig,
) -> Result<(), FractalError> {
    if config.execution_mode != ExecutionMode::Sequential || config.parallelism != 1 {
        return Err(invalid_argument(
            "tokenizer-backed Stage 0 currently requires sequential execution with parallelism=1"
                .to_owned(),
        ));
    }

    let species = species.ok_or_else(|| {
        invalid_argument(
            "tokenizer-backed Stage 0 manifests must target a single --species run".to_owned(),
        )
    })?;
    let definitions = species_registry_for_species(species);
    let definition = definitions.first().copied().ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "no species definition registered for tokenizer-backed run {}",
            species
        ))
    })?;
    let experiment = config
        .resolved_experiment(species, definition.variant_name)
        .ok_or_else(|| {
            FractalError::InvalidConfig(
                "tokenizer-backed Stage 0 requires an experiment spec".into(),
            )
        })?;
    let precision_profile =
        resolve_precision_profile(&config.execution_backend, &config.launch_policy.precision)?;
    let (metrics, bridge_stats) = match (&config.execution_backend, precision_profile) {
        (ComputeBackend::CpuCandle, ResolvedExecutablePrecisionProfile::CandleF32) => {
            let (metrics, bridge_stats, _artifact) =
                run_tokenizer_backed_species_from_experiment::<CandleF32TrainBackend>(
                    species,
                    definition.variant_name,
                    config.clone(),
                    experiment.clone(),
                    cpu_device(),
                )?;
            (metrics, bridge_stats)
        }
        (ComputeBackend::CpuCandle, ResolvedExecutablePrecisionProfile::CandleBf16) => {
            let (metrics, bridge_stats, _artifact) =
                run_tokenizer_backed_species_from_experiment::<CandleBf16TrainBackend>(
                    species,
                    definition.variant_name,
                    config.clone(),
                    experiment.clone(),
                    cpu_device(),
                )?;
            (metrics, bridge_stats)
        }
        #[cfg(feature = "cuda")]
        (
            ComputeBackend::CudaCandle { device_index },
            ResolvedExecutablePrecisionProfile::CandleF32,
        ) => {
            let (metrics, bridge_stats, _artifact) =
                run_tokenizer_backed_species_from_experiment::<CandleF32TrainBackend>(
                    species,
                    definition.variant_name,
                    config.clone(),
                    experiment.clone(),
                    cuda_device(*device_index),
                )?;
            (metrics, bridge_stats)
        }
        #[cfg(feature = "cuda")]
        (
            ComputeBackend::CudaCandle { device_index },
            ResolvedExecutablePrecisionProfile::CandleBf16,
        ) => {
            let (metrics, bridge_stats, _artifact) =
                run_tokenizer_backed_species_from_experiment::<CandleBf16TrainBackend>(
                    species,
                    definition.variant_name,
                    config.clone(),
                    experiment.clone(),
                    cuda_device(*device_index),
                )?;
            (metrics, bridge_stats)
        }
        (ComputeBackend::MetalWgpu { device }, ResolvedExecutablePrecisionProfile::MetalF32) => {
            initialize_metal_runtime(device);
            let (metrics, bridge_stats, _artifact) =
                run_tokenizer_backed_species_from_experiment::<MetalF32TrainBackend>(
                    species,
                    definition.variant_name,
                    config.clone(),
                    experiment.clone(),
                    device.clone(),
                )?;
            (metrics, bridge_stats)
        }
        _ => {
            return Err(FractalError::InvalidConfig(format!(
                "resolved precision profile {} is not executable for backend {:?}",
                precision_profile.label(),
                config.execution_backend
            )))
        }
    };

    let artifact = take_last_species_run_artifact().ok_or_else(|| {
        FractalError::InvalidState(
            "tokenizer-backed Stage 0 run completed without recording a species artifact".into(),
        )
    })?;
    let report = TournamentRunReport::new(TournamentRunReportParts {
        preset,
        lane,
        comparison,
        config: config.clone(),
        species: definitions,
        results: aggregate_results(vec![metrics]),
        artifact: fractal::TournamentRunArtifact {
            config,
            species: vec![artifact],
        },
        bridge_stats: BTreeMap::from([(species, bridge_stats)]),
    });
    emit_persisted_report(&report)
}

fn build_experiment_template(
    preset: TournamentPreset,
    lane: TournamentLane,
    species: Option<SpeciesId>,
    comparison: &ComparisonContract,
    config: &TournamentConfig,
    options: &RunOptions,
    manifest_v2: Option<&ExperimentManifestFileV2>,
) -> ExperimentSpecTemplate {
    let run_id = std::env::var("FRACTAL_RUN_ID").unwrap_or_else(|_| {
        format!(
            "{}-{}-{}",
            current_unix_ms(),
            preset.name(),
            species
                .map(|species| species.as_str().to_owned())
                .unwrap_or_else(|| lane.name().to_owned())
        )
    });
    let scope = species
        .map(|species| format!("species={species}"))
        .unwrap_or_else(|| format!("lane={}", lane.name()));
    let logical_name = options
        .logical_name
        .clone()
        .unwrap_or_else(|| format!("{}-{}", preset.name(), scope.replace('=', "-")));
    let question_summary = options
        .question_summary
        .clone()
        .unwrap_or_else(|| format!("evaluate {scope} on {}", preset.name()));

    let mut runtime = manifest_v2
        .and_then(|manifest| manifest.experiment.runtime.clone())
        .unwrap_or_else(|| options.runtime_surface_spec());
    runtime.launch_policy = config.launch_policy.clone();
    let model = manifest_v2
        .and_then(|manifest| manifest.experiment.model.clone())
        .unwrap_or_else(|| {
            ModelContractSpec::recursive_kernel_v1(config.dim, config.max_recursion_depth)
        });

    ExperimentSpecTemplate {
        experiment_id: ExperimentId {
            logical_name,
            run_id,
            branch: detect_git_ref("FRACTAL_BRANCH", &["rev-parse", "--abbrev-ref", "HEAD"]),
            commit_sha: detect_git_ref("FRACTAL_COMMIT_SHA", &["rev-parse", "HEAD"]),
            created_at_unix_ms: current_unix_ms(),
        },
        question: ExperimentQuestion {
            summary: question_summary,
            lane_intent: manifest_v2
                .and_then(|manifest| manifest.experiment.question.lane_intent)
                .unwrap_or_else(|| lane_intent_for_preset(preset)),
            decision_intent: manifest_v2
                .and_then(|manifest| manifest.experiment.question.decision_intent)
                .unwrap_or_else(|| decision_intent_for_contract(comparison)),
        },
        budget: BudgetSpec::from_config(preset, config),
        optimizer: config.optimizer.clone(),
        model,
        training_input: manifest_v2
            .and_then(|manifest| manifest.experiment.training_input.clone())
            .unwrap_or_else(TrainingInputSpec::synthetic),
        runtime,
        comparison: comparison.clone(),
        execution: ExecutionTarget {
            kind: if std::env::var_os("FRACTAL_RUN_POD_ID").is_some() {
                ExecutionTargetKind::RunPod
            } else {
                ExecutionTargetKind::Local
            },
            backend: ExecutionBackend::from_compute_backend(&config.execution_backend),
            execution_mode: config.execution_mode,
            pod_id: std::env::var("FRACTAL_RUN_POD_ID").ok(),
            wrapper_timeout_seconds: std::env::var("FRACTAL_WRAPPER_TIMEOUT_SECONDS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok()),
        },
        artifacts: manifest_v2
            .and_then(|manifest| manifest.experiment.artifacts.clone())
            .unwrap_or_default(),
    }
}

fn lane_intent_for_preset(preset: TournamentPreset) -> LaneIntent {
    match preset {
        TournamentPreset::FastTest
        | TournamentPreset::Default
        | TournamentPreset::ResearchMedium
        | TournamentPreset::ChallengerLane => LaneIntent::Benchmark,
        TournamentPreset::MinimalProvingGround
        | TournamentPreset::ProvingGroundBaseline
        | TournamentPreset::BullpenPolish => LaneIntent::Bullpen,
        TournamentPreset::MinimalBaseline
        | TournamentPreset::MinimalStressLane
        | TournamentPreset::LighterIntermediateStress
        | TournamentPreset::IntermediateStress
        | TournamentPreset::CandidateStress => LaneIntent::Validation,
        TournamentPreset::FullMediumStress
        | TournamentPreset::MediumStress
        | TournamentPreset::PressureTest
        | TournamentPreset::GenerationFour => LaneIntent::Winner,
    }
}

fn decision_intent_for_contract(comparison: &ComparisonContract) -> DecisionIntent {
    if comparison.is_authoritative_same_preset() {
        DecisionIntent::Promote
    } else {
        DecisionIntent::Benchmark
    }
}

fn current_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn detect_git_ref(env_var: &str, args: &[&str]) -> Option<String> {
    if let Ok(value) = std::env::var(env_var) {
        if !value.is_empty() {
            return Some(value);
        }
    }

    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn print_header(
    preset: TournamentPreset,
    lane: TournamentLane,
    species: Option<SpeciesId>,
    config: &TournamentConfig,
    comparison: &ComparisonContract,
) {
    println!("== {} ==", preset.name());
    let scope = species
        .map(|species| format!("species={species}"))
        .unwrap_or_else(|| format!("lane={}", lane.name()));
    println!(
        "{} backend={} mode={} parallelism={} seed={} dim={} levels={} seq={} depth={} stability_depth={} train_batch={} eval_batch={} train_steps={} eval_batches={} perplexity_eval_batches={} arc_eval_batches={}",
        scope,
        backend_name(&config.execution_backend),
        execution_mode_name(config.execution_mode),
        config.parallelism,
        config.seed,
        config.dim,
        config.levels,
        config.max_seq_len,
        config.max_recursion_depth,
        config.stability_depth,
        config.train_batch_size,
        config.eval_batch_size,
        config.train_steps_per_species,
        config.eval_batches_per_family,
        config.effective_perplexity_eval_batches(),
        config.effective_arc_eval_batches(),
    );
    println!(
        "comparison={} runtime={}",
        comparison.label(),
        config
            .experiment
            .as_ref()
            .map(|experiment| experiment.runtime.label())
            .unwrap_or_else(|| RuntimeSurfaceSpec::default().label())
    );
    if !config.species_overrides.is_empty() {
        println!(
            "temporary_overrides={}",
            config
                .species_overrides
                .iter()
                .map(format_species_override)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!("rank  species                  stability  perplexity  arc_acc  tok/s   fitness");
}

fn format_species_override(
    override_config: &fractal_core::lifecycle::SpeciesPresetOverride,
) -> String {
    let mut fields = Vec::new();
    if let Some(train_batch_size) = override_config.train_batch_size {
        fields.push(format!("train_batch={train_batch_size}"));
    }
    if let Some(eval_batch_size) = override_config.eval_batch_size {
        fields.push(format!("eval_batch={eval_batch_size}"));
    }
    if let Some(train_steps_per_species) = override_config.train_steps_per_species {
        fields.push(format!("train_steps={train_steps_per_species}"));
    }
    if let Some(max_recursion_depth) = override_config.max_recursion_depth {
        fields.push(format!("depth={max_recursion_depth}"));
    }
    if let Some(stability_depth) = override_config.stability_depth {
        fields.push(format!("stability_depth={stability_depth}"));
    }

    format!("{}({})", override_config.species, fields.join(" "))
}

struct StdoutProgressReporter;

impl TournamentReporter for StdoutProgressReporter {
    fn on_event(&self, event: TournamentProgressEvent) {
        match event {
            TournamentProgressEvent::SpeciesStarted(stage) => print_species_started(&stage),
            TournamentProgressEvent::SpeciesCompleted(completion) => {
                print_species_completed(&completion)
            }
        }
    }
}

fn print_species_started(stage: &SpeciesRunStage) {
    println!(
        "[start {}/{}] {}",
        stage.ordinal, stage.total, stage.species
    );
}

fn print_species_completed(completion: &SpeciesCompletion) {
    println!(
        "[done  {}/{}] {} elapsed={:.1}s stability={:.2} perplexity={:.2} arc={:.2} tok/s={:.0}",
        completion.stage.ordinal,
        completion.stage.total,
        completion.stage.species,
        completion.elapsed.as_secs_f64(),
        completion.metrics.grad_norm_depth_20,
        completion.metrics.long_context_perplexity,
        completion.metrics.arc_accuracy,
        completion.metrics.tokens_per_sec,
    );
}

fn print_results(report: &TournamentRunReport) {
    for result in &report.results {
        println!(
            "{:<5} {:<24} {:<10.2} {:<11.2} {:<8.2} {:<7.0} {:.2}",
            result.rank,
            result.species,
            result.stability_score,
            result.long_context_perplexity,
            result.arc_accuracy,
            result.tokens_per_sec,
            result.fitness
        );
    }
}

fn print_failures(report: &TournamentRunReport) {
    for record in &report.artifact.species {
        if matches!(record.execution_outcome, RunExecutionOutcome::Success) {
            continue;
        }
        println!(
            "failure {:<24} outcome={} error={}",
            report.variant_name_for(record.stage.species),
            outcome_label(record.outcome_class()),
            record.error.as_deref().unwrap_or("unknown error")
        );
    }
}

fn print_primitive_tracker_reminder(report: &TournamentRunReport) {
    for line in primitive_tracker_reminder_lines(report) {
        println!("{line}");
    }
}

fn emit_persisted_report(report: &TournamentRunReport) -> Result<(), FractalError> {
    let persisted = persist_run_artifacts(report)?;
    print_results(report);
    print_failures(report);
    print_primitive_tracker_reminder(report);
    println!(
        "artifacts={} manifest={}",
        persisted.artifact_path.display(),
        persisted.manifest_path.display()
    );
    ensure_successful_execution(report)
}

fn ensure_successful_execution(report: &TournamentRunReport) -> Result<(), FractalError> {
    if let Some(record) = report
        .artifact
        .species
        .iter()
        .find(|record| !matches!(record.execution_outcome, RunExecutionOutcome::Success))
    {
        return Err(FractalError::InvalidState(format!(
            "species {} failed with {:?}: {}",
            record.stage.species,
            record.outcome_class(),
            record.error.as_deref().unwrap_or("unknown error")
        )));
    }
    Ok(())
}

fn outcome_label(outcome: fractal::RunOutcomeClass) -> &'static str {
    match outcome {
        fractal::RunOutcomeClass::Success => "success",
        fractal::RunOutcomeClass::TrainTimeout => "train-timeout",
        fractal::RunOutcomeClass::EvalConstrained => "eval-constrained",
        fractal::RunOutcomeClass::NumericFailure => "numeric-failure",
        fractal::RunOutcomeClass::LowSignal => "low-signal",
        fractal::RunOutcomeClass::RuntimeCost => "runtime-cost",
        fractal::RunOutcomeClass::InfraFailure => "infra-failure",
    }
}

fn backend_name(backend: &ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::CpuCandle => "cpu",
        #[cfg(feature = "cuda")]
        ComputeBackend::CudaCandle { .. } => "cuda",
        ComputeBackend::MetalWgpu { .. } => "metal",
    }
}

fn backend_override_name(backend: BackendOverride) -> &'static str {
    match backend {
        BackendOverride::Cpu => "cpu",
        #[cfg(feature = "cuda")]
        BackendOverride::Cuda => "cuda",
        BackendOverride::Metal => "metal",
    }
}

fn execution_mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Sequential => "sequential",
        ExecutionMode::Parallel => "parallel",
    }
}

fn invalid_argument(message: String) -> FractalError {
    FractalError::InvalidConfig(message)
}

fn print_usage() {
    println!("Usage: cargo run --example tournament -- [options]");
    println!();
    println!("Options:");
    println!(
        "  --preset <default|fast-test|research-medium|challenger-lane|minimal-baseline|minimal-stress-lane|minimal-proving-ground|proving-ground-baseline|bullpen-polish|lighter-intermediate-stress|intermediate-stress|full-medium-stress|medium-stress|pressure-test|candidate-stress|generation-four>"
    );
    println!("  --sequence <first-run>");
    println!("  --lane <all|baseline|challenger|bullpen|proving-ground|squaring|leader>");
    println!("  --species <species-id>");
    println!("  --seed <u64>");
    println!("  --mode <sequential|parallel>");
    println!("  --parallelism <usize>");
    println!("  --perplexity-eval-batches <usize>");
    println!("  --arc-eval-batches <usize>");
    println!("  --benchmark-mode <leaderboard|systems-speed>");
    println!("  --experiment-manifest <path>");
    println!("  --prepare-stage0-assets");
    #[cfg(feature = "cuda")]
    println!("  --backend <cpu|cuda|metal>");
    #[cfg(not(feature = "cuda"))]
    println!("  --backend <cpu|metal>");
    println!("  --help");
    println!();
    println!("Examples:");
    println!("  cargo run --example tournament -- --preset fast-test");
    println!("  cargo run --release --example tournament -- --preset minimal-baseline");
    println!("  cargo run --release --example tournament -- --preset minimal-stress-lane");
    println!("  cargo run --release --example tournament -- --preset lighter-intermediate-stress");
    println!("  cargo run --release --example tournament -- --preset intermediate-stress");
    println!("  cargo run --release --example tournament -- --preset full-medium-stress");
    println!("  cargo run --release --example tournament -- --preset medium-stress");
    println!(
        "  cargo run --release --example tournament -- --preset proving-ground-baseline --species mandelbox_recursive"
    );
    println!("  cargo run --release --example tournament -- --lane baseline");
    println!("  cargo run --release --example tournament -- --lane bullpen");
    println!("  cargo run --release --example tournament -- --lane proving-ground");
    println!("  cargo run --release --example tournament -- --species generalized_mobius");
    println!("  cargo run --release --example tournament -- --preset research-medium --mode parallel --parallelism 4");
    println!("  cargo run --release --example tournament -- --prepare-stage0-assets --experiment-manifest experiments/stage0/canary/seed42-p1_contractive.json");
    #[cfg(feature = "cuda")]
    println!(
        "  cargo run --release --features cuda --example tournament -- --lane leader --mode sequential"
    );
    println!("  cargo run --release --example tournament -- --sequence first-run");
}

#[cfg(test)]
mod tests {
    use super::*;
    use fractal::TrainingInputMode;
    use fractal_tokenizer::{FaceoffTokenizer, FaceoffVocab, TokenizerConfig};
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Mutex,
    };

    static MANIFEST_ENV_MUTEX: Mutex<()> = Mutex::new(());

    struct TestEnvGuard {
        root: PathBuf,
    }

    impl TestEnvGuard {
        fn new(prefix: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "{prefix}-{}-{}",
                std::process::id(),
                current_unix_ms()
            ));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            Self { root }
        }

        fn path(&self) -> &Path {
            &self.root
        }
    }

    impl Drop for TestEnvGuard {
        fn drop(&mut self) {
            std::env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
            std::env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn write_jsonl_corpus(path: &Path, documents: &[&str]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let body = documents
            .iter()
            .map(|document| serde_json::json!({ "text": document }).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, format!("{body}\n")).unwrap();
    }

    fn sentencepiece_testdata_model() -> PathBuf {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("cargo metadata should run");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata should be valid json");
        let packages = metadata["packages"]
            .as_array()
            .expect("cargo metadata should include packages");
        let manifest_path = packages
            .iter()
            .find(|package| package["name"] == "sentencepiece")
            .and_then(|package| package["manifest_path"].as_str())
            .expect("sentencepiece package should be present in cargo metadata");
        PathBuf::from(manifest_path)
            .parent()
            .expect("sentencepiece manifest should have parent")
            .join("testdata")
            .join("toy.model")
    }

    fn build_file_backed_sentencepiece_tokenizer(root: &Path) -> PathBuf {
        let path = root.join("tokenizer.model");
        fs::copy(sentencepiece_testdata_model(), &path).unwrap();
        path
    }

    fn build_bridge_vocab_artifact(
        root: &Path,
        dim: usize,
        levels: usize,
        max_depth: usize,
        seed: u64,
        documents: &[&str],
    ) -> PathBuf {
        let path = root.join("bridge-vocab.json");
        let tokenizer = FaceoffTokenizer::new(TokenizerConfig {
            dim,
            levels,
            max_depth,
            seed,
            split_policy: fractal_tokenizer::SplitPolicy::Balanced,
            substrate_mode: fractal_tokenizer::TokenizerSubstrateMode::RawBytes,
        });
        let device = fractal_core::registry::cpu_device();
        let vocab = tokenizer
            .induce_vocab_from_texts::<fractal_core::CpuTrainBackend>(documents, &device)
            .expect("test bridge vocab should build");
        vocab
            .save_to_file(&path)
            .expect("test bridge vocab should persist");
        path
    }

    #[test]
    fn parse_command_uses_default_preset_when_no_args() {
        let command = parse_command(Vec::<String>::new()).unwrap();

        assert_eq!(command, CliCommand::Run(RunOptions::default()));
    }

    #[test]
    fn parse_command_supports_first_run_sequence_with_overrides() {
        let command = parse_command(vec![
            "--sequence".to_owned(),
            "first-run".to_owned(),
            "--lane".to_owned(),
            "baseline".to_owned(),
            "--seed".to_owned(),
            "43".to_owned(),
            "--mode".to_owned(),
            "parallel".to_owned(),
            "--parallelism".to_owned(),
            "3".to_owned(),
            "--backend".to_owned(),
            "cpu".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Sequence(TournamentSequence::FirstRun)),
                lane: Some(TournamentLane::Baseline),
                species: None,
                seed: Some(43),
                execution_mode: Some(ExecutionMode::Parallel),
                parallelism: Some(3),
                backend: Some(BackendOverride::Cpu),
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_proving_ground_baseline_preset() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "proving-ground-baseline".to_owned(),
            "--species".to_owned(),
            "mandelbox_recursive".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(
                    TournamentPreset::ProvingGroundBaseline
                )),
                lane: None,
                species: Some(SpeciesId::MandelboxRecursive),
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn parse_command_accepts_cuda_backend_override() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "fast-test".to_owned(),
            "--backend".to_owned(),
            "cuda".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FastTest)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: Some(BackendOverride::Cuda),
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn parse_command_rejects_cuda_backend_without_feature() {
        let error = parse_command(vec![
            "--preset".to_owned(),
            "fast-test".to_owned(),
            "--backend".to_owned(),
            "cuda".to_owned(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
    }

    #[test]
    fn parse_command_rejects_preset_and_sequence_together() {
        let error = parse_command(vec![
            "--preset".to_owned(),
            "fast-test".to_owned(),
            "--sequence".to_owned(),
            "first-run".to_owned(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
    }

    #[test]
    fn parse_command_accepts_generation_four_preset() {
        let command =
            parse_command(vec!["--preset".to_owned(), "generation-four".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::GenerationFour)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_minimal_proving_ground_preset() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "minimal-proving-ground".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::MinimalProvingGround)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_minimal_baseline_preset() {
        let command =
            parse_command(vec!["--preset".to_owned(), "minimal-baseline".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::MinimalBaseline)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_medium_stress_preset() {
        let command =
            parse_command(vec!["--preset".to_owned(), "medium-stress".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::MediumStress)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_intermediate_stress_preset() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "intermediate-stress".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::IntermediateStress)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_challenger_lane_preset() {
        let command =
            parse_command(vec!["--preset".to_owned(), "challenger-lane".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::ChallengerLane)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_lane_only_and_uses_lane_default_preset() {
        let command = parse_command(vec!["--lane".to_owned(), "challenger".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: None,
                lane: Some(TournamentLane::Challenger),
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_proving_ground_lane_alias() {
        let command = parse_command(vec!["--lane".to_owned(), "squaring".to_owned()]).unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: None,
                lane: Some(TournamentLane::ProvingGround),
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_species_only_and_uses_species_default_preset() {
        let command = parse_command(vec![
            "--species".to_owned(),
            "generalized_mobius".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: None,
                lane: None,
                species: Some(SpeciesId::GeneralizedMobius),
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_rejects_lane_and_species_together() {
        let error = parse_command(vec![
            "--lane".to_owned(),
            "bullpen".to_owned(),
            "--species".to_owned(),
            "ifs".to_owned(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
    }

    #[test]
    fn parse_command_rejects_zero_parallelism() {
        let error = parse_command(vec![
            "--preset".to_owned(),
            "fast-test".to_owned(),
            "--parallelism".to_owned(),
            "0".to_owned(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
    }

    #[test]
    fn parse_command_accepts_explicit_eval_budget_overrides() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "minimal-stress-lane".to_owned(),
            "--perplexity-eval-batches".to_owned(),
            "1".to_owned(),
            "--arc-eval-batches".to_owned(),
            "3".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::MinimalStressLane)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: Some(1),
                arc_eval_batches: Some(3),
                benchmark_mode: None,
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_accepts_benchmark_mode_override() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "full-medium-stress".to_owned(),
            "--benchmark-mode".to_owned(),
            "systems-speed".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FullMediumStress)),
                lane: None,
                species: None,
                seed: None,
                execution_mode: None,
                parallelism: None,
                backend: None,
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: Some(BenchmarkMode::SystemsSpeed),
                manifest_path: None,
                prepare_stage0_assets: false,
                logical_name: None,
                question_summary: None,
                comparison_override: None,
            })
        );
    }

    #[test]
    fn parse_command_loads_experiment_manifest() {
        let root =
            std::env::temp_dir().join(format!("fractal-exp-manifest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("winner-bakeoff-s42.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "winner-bakeoff-s42-p1-fractal-hybrid",
                "question_summary": "rerun the frozen winner bakeoff row for p1_fractal_hybrid_v1",
                "preset": "full-medium-stress",
                "species": "p1_fractal_hybrid",
                "seed": 42,
                "backend": "cpu",
                "execution_mode": "sequential",
                "parallelism": 1,
                "comparison": "authoritative_same_preset",
                "benchmark_mode": "leaderboard"
            }))
            .unwrap(),
        )
        .unwrap();

        let command = parse_command(vec![
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FullMediumStress)),
                lane: None,
                species: Some(SpeciesId::P1FractalHybrid),
                seed: Some(42),
                execution_mode: Some(ExecutionMode::Sequential),
                parallelism: Some(1),
                backend: Some(BackendOverride::Cpu),
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: Some(BenchmarkMode::Leaderboard),
                manifest_path: Some(manifest_path.clone()),
                prepare_stage0_assets: false,
                logical_name: Some("winner-bakeoff-s42-p1-fractal-hybrid".to_owned()),
                question_summary: Some(
                    "rerun the frozen winner bakeoff row for p1_fractal_hybrid_v1".to_owned(),
                ),
                comparison_override: Some(ComparisonContract::authoritative_same_preset()),
            })
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_command_rejects_manifest_mixed_with_flags() {
        let root =
            std::env::temp_dir().join(format!("fractal-exp-manifest-mixed-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("winner-bakeoff-s43.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "winner-bakeoff-s43-p1-contractive",
                "preset": "full-medium-stress",
                "species": "p1_contractive"
            }))
            .unwrap(),
        )
        .unwrap();

        let error = parse_command(vec![
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
            "--seed".to_owned(),
            "43".to_owned(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_command_allows_backend_before_experiment_manifest() {
        let root = std::env::temp_dir().join(format!(
            "fractal-exp-manifest-backend-prefix-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("winner-bakeoff-s44.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "winner-bakeoff-s44-p1-contractive",
                "preset": "full-medium-stress",
                "species": "p1_contractive",
                "seed": 44,
                "backend": "cpu"
            }))
            .unwrap(),
        )
        .unwrap();

        let command = parse_command(vec![
            "--backend".to_owned(),
            "cpu".to_owned(),
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FullMediumStress)),
                lane: None,
                species: Some(SpeciesId::P1Contractive),
                seed: Some(44),
                execution_mode: None,
                parallelism: None,
                backend: Some(BackendOverride::Cpu),
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: Some(BenchmarkMode::Leaderboard),
                manifest_path: Some(manifest_path.clone()),
                prepare_stage0_assets: false,
                logical_name: Some("winner-bakeoff-s44-p1-contractive".to_owned()),
                question_summary: None,
                comparison_override: Some(ComparisonContract::authoritative_same_preset()),
            })
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_command_allows_backend_after_experiment_manifest() {
        let root = std::env::temp_dir().join(format!(
            "fractal-exp-manifest-backend-suffix-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("winner-bakeoff-s45.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "winner-bakeoff-s45-p1-contractive",
                "preset": "full-medium-stress",
                "species": "p1_contractive",
                "seed": 45,
                "backend": "cpu"
            }))
            .unwrap(),
        )
        .unwrap();

        let command = parse_command(vec![
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
            "--backend".to_owned(),
            "cpu".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FullMediumStress)),
                lane: None,
                species: Some(SpeciesId::P1Contractive),
                seed: Some(45),
                execution_mode: None,
                parallelism: None,
                backend: Some(BackendOverride::Cpu),
                perplexity_eval_batches: None,
                arc_eval_batches: None,
                benchmark_mode: Some(BenchmarkMode::Leaderboard),
                manifest_path: Some(manifest_path.clone()),
                prepare_stage0_assets: false,
                logical_name: Some("winner-bakeoff-s45-p1-contractive".to_owned()),
                question_summary: None,
                comparison_override: Some(ComparisonContract::authoritative_same_preset()),
            })
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_command_rejects_conflicting_backend_for_experiment_manifest() {
        let root = std::env::temp_dir().join(format!(
            "fractal-exp-manifest-backend-conflict-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("winner-bakeoff-s46.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "winner-bakeoff-s46-p1-contractive",
                "preset": "full-medium-stress",
                "species": "p1_contractive",
                "seed": 46,
                "backend": "cpu"
            }))
            .unwrap(),
        )
        .unwrap();

        let error = parse_command(vec![
            "--backend".to_owned(),
            "metal".to_owned(),
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .unwrap_err();

        assert!(matches!(error, FractalError::InvalidConfig(_)));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn parse_command_loads_experiment_manifest_v2_with_typed_stage0_contract() {
        let root =
            std::env::temp_dir().join(format!("fractal-exp-manifest-v2-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("stage0-canary-s42-p1-contractive.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "stage0-canary-s42-p1-contractive",
                "question_summary": "prove the Stage 0 canary launch path end-to-end for p1_contractive_v1",
                "preset": "fast-test",
                "species": "p1_contractive",
                "experiment": {
                    "question": {
                        "lane_intent": "benchmark",
                        "decision_intent": "benchmark"
                    },
                    "model": {
                        "architecture": "recursive-kernel-v1",
                        "hidden_dim": 1024,
                        "max_recursion_depth": 16,
                        "router_enabled": true
                    },
                    "config": {
                        "seed": 42,
                        "dim": 1024,
                        "levels": 6,
                        "vocab_size": 32000,
                        "max_seq_len": 2048,
                        "train_batch_size": 2,
                        "eval_batch_size": 2,
                        "train_steps_per_species": 8,
                        "eval_batches_per_family": 1,
                        "perplexity_eval_batches": 1,
                        "arc_eval_batches": 1,
                        "execution_mode": "sequential",
                        "parallelism": 1,
                        "backend": "cpu"
                    },
                    "training_input": {
                        "mode": "tokenizer-backed-text",
                        "corpus_name": "fineweb-stage0",
                        "corpus_source": {
                            "train": {
                                "path": "/tmp/fineweb-train.jsonl",
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            },
                            "eval": {
                                "path": "/tmp/fineweb-eval.jsonl",
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            }
                        },
                        "tokenizer": {
                            "artifact_id": "openlm-research/open_llama_3b_v2",
                            "artifact_path": "/tmp/tokenizer.model",
                            "vocab_size": 32000,
                            "pad_token_id": 0
                        },
                        "bridge": {
                            "enabled": true,
                            "observational_only": true
                        },
                        "bridge_packaging": {
                            "vocab_artifact_path": "/tmp/fineweb-bridge-vocab.json",
                            "dim": 1024,
                            "levels": 6,
                            "max_depth": 16,
                            "seed": 42,
                            "split_policy": "balanced",
                            "substrate_mode": "raw-bytes",
                            "chunk_max_tokens": 2048,
                            "chunk_max_bytes": 2048
                        },
                        "arc_source": {
                            "mode": "synthetic-canonical"
                        }
                    },
                    "optimizer": {
                        "kind": "adamw",
                        "peak_learning_rate": 0.0002,
                        "beta_1": 0.9,
                        "beta_2": 0.95,
                        "epsilon": 1e-8,
                        "weight_decay": 0.05,
                        "gradient_clip_norm": 1.0,
                        "schedule": {
                            "kind": "warmup-cosine",
                            "warmup_fraction": 0.02,
                            "decay_floor_fraction": 0.1
                        }
                    },
                    "runtime": {
                        "eval_backend_policy": "shared-backend",
                        "batching_policy": "padded",
                        "execution_policy": "simple-loop",
                        "buffer_reuse_policy": "disabled",
                        "benchmark_mode": "leaderboard",
                        "backend_policy": "active-execution-backend",
                        "launch_policy": {
                            "precision": {
                                "compute": "fp32",
                                "optimizer_state": "fp32",
                                "reduction": "fp32",
                                "tf32_enabled": true
                            },
                            "checkpoint": {
                                "interval_tokens": 10000000,
                                "keep_latest": true,
                                "keep_best": true,
                                "keep_final": true,
                                "keep_previous": true
                            },
                            "eval_cadence": {
                                "perplexity_interval_tokens": 10000000,
                                "stability_interval_tokens": 10000000,
                                "arc_interval_tokens": 20000000,
                                "systems_speed_interval_tokens": 20000000,
                                "final_full_eval": true
                            },
                            "resume": {
                                "resume_on_interrupt": true,
                                "restart_on_corruption": true,
                                "restart_on_contract_ambiguity": true
                            }
                        }
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "artifacts": {
                        "manifest_required": true,
                        "structured_artifact_required": true,
                        "final_log_required": true,
                        "tracker_ready_output_required": true
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let command = parse_command(vec![
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .unwrap();

        let options = match command {
            CliCommand::Run(options) => options,
            other => panic!("expected run command, got {other:?}"),
        };

        assert_eq!(options.species, Some(SpeciesId::P1Contractive));
        assert_eq!(options.seed, Some(42));
        assert_eq!(options.backend, Some(BackendOverride::Cpu));
        assert_eq!(options.parallelism, Some(1));
        assert_eq!(
            options.comparison_override,
            Some(ComparisonContract::authoritative_same_preset())
        );

        let config = options.config_for(TournamentPreset::FastTest).unwrap();
        assert_eq!(config.dim, 1024);
        assert_eq!(config.levels, 6);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.optimizer, OptimizerSpec::stage0_adamw());
        let mut expected_launch_policy = fractal::LaunchPolicySpec::stage0_default();
        expected_launch_policy.precision.compute = fractal::NumericPrecisionKind::Fp32;
        assert_eq!(config.launch_policy, expected_launch_policy);

        let manifest_v2 = load_manifest_v2(&manifest_path).unwrap().unwrap();
        let template = build_experiment_template(
            TournamentPreset::FastTest,
            TournamentLane::All,
            Some(SpeciesId::P1Contractive),
            &ComparisonContract::authoritative_same_preset(),
            &config,
            &options,
            Some(&manifest_v2),
        );

        assert_eq!(
            template.training_input.mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(
            template.training_input.corpus_name.as_deref(),
            Some("fineweb-stage0")
        );
        assert_eq!(template.model.architecture.as_str(), "recursive-kernel-v1");
        assert_eq!(
            template
                .training_input
                .bridge_packaging
                .as_ref()
                .map(|packaging| packaging.vocab_artifact_path.as_str()),
            Some("/tmp/fineweb-bridge-vocab.json")
        );
        assert_eq!(template.runtime.launch_policy, expected_launch_policy);
        assert_eq!(template.question.lane_intent, LaneIntent::Benchmark);
        assert_eq!(template.question.decision_intent, DecisionIntent::Benchmark);
        assert_eq!(template.artifacts, ArtifactPolicy::default());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn checked_in_stage0_canary_manifest_loads_as_v2_contract() {
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("experiments/stage0/canary/seed42-p1_contractive.json");

        let manifest = load_manifest_v2(&manifest_path)
            .expect("checked-in canary manifest should parse")
            .expect("checked-in canary manifest should use manifest v2");

        assert_eq!(manifest.logical_name, "stage0-canary-s42-p1-contractive");
        assert_eq!(manifest.preset, "full-medium-stress");
        assert_eq!(manifest.species.as_deref(), Some("p1_contractive"));
        assert_eq!(
            manifest.expected_branch.as_deref(),
            Some("codex/stage0-launch")
        );
        assert_eq!(
            manifest.experiment.config.train_token_budget,
            Some(10_000_000)
        );
        assert_eq!(
            manifest
                .experiment
                .model
                .as_ref()
                .map(|model| model.architecture.as_str()),
            Some("recursive-kernel-v1")
        );
        assert_eq!(
            manifest
                .experiment
                .training_input
                .as_ref()
                .and_then(|training_input| training_input.corpus_name.as_deref()),
            Some("fineweb-stage0-canary")
        );
        assert_eq!(
            manifest
                .experiment
                .training_input
                .as_ref()
                .and_then(|training_input| training_input.bridge_packaging.as_ref())
                .map(|packaging| packaging.vocab_artifact_path.as_str()),
            Some("experiments/stage0/assets/open_llama_3b_v2/fineweb-stage0-canary-bridge-vocab.json")
        );
    }

    #[test]
    fn prepare_stage0_assets_materializes_bridge_vocab_from_manifest_v2() {
        let _env_lock = MANIFEST_ENV_MUTEX.lock().unwrap();
        let guard = TestEnvGuard::new("fractal-stage0-prepare");
        let manifest_output_dir = guard.path().join("prepared-manifests");
        let train_path = guard.path().join("train.jsonl");
        let eval_path = guard.path().join("eval.jsonl");
        let tokenizer_path = build_file_backed_sentencepiece_tokenizer(guard.path());
        let bridge_vocab_path = guard.path().join("bridge-vocab.json");
        let manifest_path = guard.path().join("stage0-canary-prepare.json");

        write_jsonl_corpus(
            &train_path,
            &["I saw a girl with a telescope.", "alpha beta gamma delta"],
        );
        write_jsonl_corpus(&eval_path, &["I saw a girl with a telescope."]);

        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "stage0-canary-prepare-s42-p1-contractive",
                "question_summary": "materialize frozen Stage 0 assets from the manifest control plane",
                "preset": "fast-test",
                "species": "p1_contractive",
                "experiment": {
                    "question": {
                        "lane_intent": "benchmark",
                        "decision_intent": "benchmark"
                    },
                    "model": {
                        "architecture": "recursive-kernel-v1",
                        "hidden_dim": 8,
                        "max_recursion_depth": 2,
                        "router_enabled": true
                    },
                    "config": {
                        "seed": 42,
                        "dim": 8,
                        "levels": 2,
                        "vocab_size": 1000,
                        "max_seq_len": 64,
                        "max_recursion_depth": 2,
                        "stability_depth": 1,
                        "train_batch_size": 1,
                        "eval_batch_size": 1,
                        "train_steps_per_species": 1,
                        "train_token_budget": 1,
                        "eval_batches_per_family": 1,
                        "perplexity_eval_batches": 1,
                        "arc_eval_batches": 1,
                        "execution_mode": "sequential",
                        "parallelism": 1,
                        "backend": "cpu"
                    },
                    "training_input": {
                        "mode": "tokenizer-backed-text",
                        "corpus_name": "fineweb-stage0-prepare",
                        "corpus_source": {
                            "train": {
                                "path": train_path,
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            },
                            "eval": {
                                "path": eval_path,
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            }
                        },
                        "tokenizer": {
                            "artifact_id": "openlm-research/open_llama_3b_v2",
                            "artifact_path": tokenizer_path,
                            "vocab_size": 1000,
                            "pad_token_id": 0
                        },
                        "bridge": {
                            "enabled": true,
                            "observational_only": true
                        },
                        "bridge_packaging": {
                            "vocab_artifact_path": bridge_vocab_path,
                            "dim": 8,
                            "levels": 2,
                            "max_depth": 2,
                            "seed": 42,
                            "split_policy": "balanced",
                            "substrate_mode": "raw-bytes",
                            "chunk_max_tokens": 64,
                            "chunk_max_bytes": 64
                        },
                        "arc_source": {
                            "mode": "synthetic-canonical"
                        }
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        std::env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_output_dir);

        let command = parse_command(vec![
            "--prepare-stage0-assets".to_owned(),
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .expect("prepare command should parse");
        let options = match command {
            CliCommand::PrepareStage0Assets(options) => options,
            other => panic!("expected prepare command, got {other:?}"),
        };

        prepare_stage0_assets(&options).expect("asset preparation should succeed");

        assert!(bridge_vocab_path.is_file());
        let vocab = FaceoffVocab::load_from_file(&bridge_vocab_path)
            .expect("prepared bridge vocab should load");
        assert!(!vocab.entries().is_empty());
        let frozen_manifest_path =
            manifest_output_dir.join("stage0-canary-prepare-s42-p1-contractive.frozen.json");
        assert!(frozen_manifest_path.is_file());
        let frozen_manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&frozen_manifest_path).unwrap()).unwrap();
        assert_eq!(
            frozen_manifest["expected_commit_sha"].as_str(),
            Some(
                current_git_commit_sha()
                    .expect("current git commit should resolve")
                    .as_str()
            )
        );
    }

    #[test]
    fn run_options_executes_tokenizer_backed_manifest_v2_and_persists_artifacts() {
        let _env_lock = MANIFEST_ENV_MUTEX.lock().unwrap();
        let guard = TestEnvGuard::new("fractal-stage0-manifest-run");
        let artifact_dir = guard.path().join("artifacts");
        let manifest_dir = guard.path().join("manifests");
        let train_path = guard.path().join("train.jsonl");
        let eval_path = guard.path().join("eval.jsonl");
        let tokenizer_path = build_file_backed_sentencepiece_tokenizer(guard.path());
        let bridge_vocab_path = build_bridge_vocab_artifact(
            guard.path(),
            8,
            2,
            2,
            42,
            &["I saw a girl with a telescope.", "alpha beta gamma delta"],
        );
        let manifest_path = guard
            .path()
            .join("stage0-canary-smoke-s42-p1-contractive.json");

        write_jsonl_corpus(
            &train_path,
            &["I saw a girl with a telescope.", "alpha beta gamma delta"],
        );
        write_jsonl_corpus(&eval_path, &["I saw a girl with a telescope."]);

        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "logical_name": "stage0-canary-smoke-s42-p1-contractive",
                "question_summary": "exercise tokenizer-backed Stage 0 manifest runner end-to-end",
                "preset": "fast-test",
                "species": "p1_contractive",
                "experiment": {
                    "question": {
                        "lane_intent": "benchmark",
                        "decision_intent": "benchmark"
                    },
                    "model": {
                        "architecture": "recursive-kernel-v1",
                        "hidden_dim": 8,
                        "max_recursion_depth": 2,
                        "router_enabled": true
                    },
                    "config": {
                        "seed": 42,
                        "dim": 8,
                        "levels": 2,
                        "vocab_size": 1000,
                        "max_seq_len": 64,
                        "max_recursion_depth": 2,
                        "stability_depth": 1,
                        "train_batch_size": 1,
                        "eval_batch_size": 1,
                        "train_steps_per_species": 1,
                        "train_token_budget": 1,
                        "eval_batches_per_family": 1,
                        "perplexity_eval_batches": 1,
                        "arc_eval_batches": 1,
                        "execution_mode": "sequential",
                        "parallelism": 1,
                        "backend": "cpu"
                    },
                    "training_input": {
                        "mode": "tokenizer-backed-text",
                        "corpus_name": "fineweb-stage0-smoke",
                        "corpus_source": {
                            "train": {
                                "path": train_path,
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            },
                            "eval": {
                                "path": eval_path,
                                "format": {
                                    "format": "jsonl-text",
                                    "text_field": "text"
                                }
                            }
                        },
                        "tokenizer": {
                            "artifact_id": "openlm-research/open_llama_3b_v2",
                            "artifact_path": tokenizer_path,
                            "vocab_size": 1000,
                            "pad_token_id": 0
                        },
                        "bridge": {
                            "enabled": true,
                            "observational_only": true
                        },
                        "bridge_packaging": {
                            "vocab_artifact_path": bridge_vocab_path,
                            "dim": 8,
                            "levels": 2,
                            "max_depth": 2,
                            "seed": 42,
                            "split_policy": "balanced",
                            "substrate_mode": "raw-bytes",
                            "chunk_max_tokens": 64,
                            "chunk_max_bytes": 64
                        },
                        "arc_source": {
                            "mode": "synthetic-canonical"
                        }
                    },
                    "optimizer": {
                        "kind": "adamw",
                        "peak_learning_rate": 0.0002,
                        "beta_1": 0.9,
                        "beta_2": 0.95,
                        "epsilon": 1e-8,
                        "weight_decay": 0.05,
                        "gradient_clip_norm": 1.0,
                        "schedule": {
                            "kind": "warmup-cosine",
                            "warmup_fraction": 0.02,
                            "decay_floor_fraction": 0.1
                        }
                    },
                    "runtime": {
                        "eval_backend_policy": "shared-backend",
                        "batching_policy": "padded",
                        "execution_policy": "simple-loop",
                        "buffer_reuse_policy": "disabled",
                        "benchmark_mode": "leaderboard",
                        "backend_policy": "active-execution-backend",
                        "launch_policy": {
                            "precision": {
                                "compute": "fp32",
                                "optimizer_state": "fp32",
                                "reduction": "fp32",
                                "tf32_enabled": true
                            },
                            "checkpoint": {
                                "interval_tokens": 1,
                                "keep_latest": true,
                                "keep_best": true,
                                "keep_final": true,
                                "keep_previous": true
                            },
                            "eval_cadence": {
                                "perplexity_interval_tokens": 1,
                                "stability_interval_tokens": 1,
                                "arc_interval_tokens": 1,
                                "systems_speed_interval_tokens": 1,
                                "final_full_eval": true
                            },
                            "resume": {
                                "resume_on_interrupt": true,
                                "restart_on_corruption": true,
                                "restart_on_contract_ambiguity": true
                            }
                        }
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "artifacts": {
                        "manifest_required": true,
                        "structured_artifact_required": true,
                        "final_log_required": true,
                        "tracker_ready_output_required": true
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();

        std::env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        std::env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let command = parse_command(vec![
            "--experiment-manifest".to_owned(),
            manifest_path.display().to_string(),
        ])
        .unwrap();
        let options = match command {
            CliCommand::Run(options) => options,
            other => panic!("expected run command, got {other:?}"),
        };

        run_options(&options).expect("tokenizer-backed manifest run should succeed");

        let persisted_manifest_path = manifest_dir.join("tournament-run-manifest.json");
        let persisted_artifact_path = artifact_dir.join("tournament-run-artifact.json");
        assert!(persisted_manifest_path.exists());
        assert!(persisted_artifact_path.exists());

        let persisted_manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&persisted_manifest_path).unwrap()).unwrap();
        let persisted_artifact: serde_json::Value =
            serde_json::from_slice(&fs::read(&persisted_artifact_path).unwrap()).unwrap();

        assert_eq!(
            persisted_manifest["experiments"][0]["training_input"]["mode"],
            serde_json::Value::String("tokenizer-backed-text".to_owned())
        );
        assert_eq!(
            persisted_manifest["experiments"][0]["training_input"]["corpus_source"]["train"]
                ["path"],
            serde_json::Value::String(train_path.display().to_string())
        );
        assert_eq!(
            persisted_manifest["experiments"][0]["model"]["architecture"],
            serde_json::Value::String("recursive-kernel-v1".to_owned())
        );
        assert_eq!(
            persisted_artifact["results"][0]["execution_outcome"],
            serde_json::Value::String("success".to_owned())
        );
        assert_eq!(
            persisted_artifact["results"][0]["quality_outcome"],
            serde_json::Value::String("low-signal".to_owned())
        );
        assert_eq!(
            persisted_artifact["manifest"]["config"]["train_token_budget"],
            serde_json::Value::Number(1usize.into())
        );
        assert_eq!(
            persisted_artifact["results"][0]["tokenizer_bridge"]["training_input_mode"],
            serde_json::Value::String("tokenizer-backed-text".to_owned())
        );
        assert_eq!(
            persisted_artifact["results"][0]["tokenizer_bridge"]["bridge_vocab_artifact_path"],
            serde_json::Value::String(bridge_vocab_path.display().to_string())
        );
        let runtime = &persisted_artifact["results"][0]["training_runtime"];
        assert_eq!(
            runtime["target_train_tokens"],
            serde_json::Value::Number(1usize.into())
        );
        assert_eq!(
            runtime["completed_steps"],
            serde_json::Value::Number(1usize.into())
        );
        assert_eq!(
            runtime["resumed_from_checkpoint"],
            serde_json::Value::Bool(false)
        );
        assert!(
            runtime["train_tokens_seen"]
                .as_u64()
                .expect("train token count should be recorded")
                >= 1
        );
        assert!(
            runtime["checkpoints"]
                .as_array()
                .expect("checkpoint list should be present")
                .len()
                >= 3
        );
        assert!(
            runtime["interim_evaluations"]
                .as_array()
                .expect("interim eval list should be present")
                .len()
                >= 1
        );

        run_options(&options).expect("rerunning manifest should resume cleanly");

        let resumed_artifact: serde_json::Value =
            serde_json::from_slice(&fs::read(&persisted_artifact_path).unwrap()).unwrap();
        assert_eq!(
            resumed_artifact["results"][0]["training_runtime"]["resumed_from_checkpoint"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            persisted_artifact["results"][0]["tokenizer_bridge"]["corpus_name"],
            serde_json::Value::String("fineweb-stage0-smoke".to_owned())
        );
        assert_eq!(
            persisted_artifact["results"][0]["experiment"]["training_input"]["tokenizer"]
                ["artifact_id"],
            serde_json::Value::String("openlm-research/open_llama_3b_v2".to_owned())
        );
    }
}
