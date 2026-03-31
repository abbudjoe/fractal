use std::sync::Arc;

use fractal::{
    aggregate_results,
    error::FractalError,
    lifecycle::{
        RunExecutionOutcome, SpeciesCompletion, SpeciesRunStage, Tournament, TournamentConfig,
        TournamentPreset, TournamentProgressEvent, TournamentReporter, TournamentSequence,
    },
    persist_run_artifacts, primitive_tracker_reminder_lines,
    registry::{ComputeBackend, ExecutionMode, SpeciesId},
    species_registry_for_lane, species_registry_for_species, ComparisonAuthority, TournamentLane,
    TournamentRunReport,
};

fn main() -> Result<(), FractalError> {
    match parse_command(std::env::args().skip(1))? {
        CliCommand::Help => {
            print_usage();
            Ok(())
        }
        CliCommand::Run(options) => run_options(&options),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CliCommand {
    Help,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunOptions {
    selection: Option<RunSelection>,
    lane: Option<TournamentLane>,
    species: Option<SpeciesId>,
    seed: Option<u64>,
    execution_mode: Option<ExecutionMode>,
    parallelism: Option<usize>,
    backend: Option<BackendOverride>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            selection: None,
            lane: None,
            species: None,
            seed: None,
            execution_mode: None,
            parallelism: None,
            backend: None,
        }
    }
}

impl RunOptions {
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

    fn config_for(&self, preset: TournamentPreset) -> TournamentConfig {
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
        config
    }
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
    let mut options = RunOptions::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--help" {
            return Ok(CliCommand::Help);
        }
        parse_arg(&mut options, &mut args, arg)?;
    }
    Ok(CliCommand::Run(options))
}

fn parse_arg<I>(options: &mut RunOptions, args: &mut I, arg: String) -> Result<(), FractalError>
where
    I: Iterator<Item = String>,
{
    match arg.as_str() {
        "--preset" => {
            let value = next_value(args, "--preset")?;
            options.set_selection(RunSelection::Preset(parse_preset(&value)?))?;
            Ok(())
        }
        "--sequence" => {
            let value = next_value(args, "--sequence")?;
            options.set_selection(RunSelection::Sequence(parse_sequence(&value)?))?;
            Ok(())
        }
        "--lane" => {
            let value = next_value(args, "--lane")?;
            options.set_lane(parse_lane(&value)?)?;
            Ok(())
        }
        "--species" => {
            let value = next_value(args, "--species")?;
            options.set_species(parse_species(&value)?)?;
            Ok(())
        }
        "--seed" => {
            let value = next_value(args, "--seed")?;
            options.seed = Some(parse_seed(&value)?);
            Ok(())
        }
        "--mode" => {
            let value = next_value(args, "--mode")?;
            options.execution_mode = Some(parse_execution_mode(&value)?);
            Ok(())
        }
        "--parallelism" => {
            let value = next_value(args, "--parallelism")?;
            options.parallelism = Some(parse_parallelism(&value)?);
            Ok(())
        }
        "--backend" => {
            let value = next_value(args, "--backend")?;
            options.backend = Some(parse_backend(&value)?);
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
    let parallelism = value
        .parse::<usize>()
        .map_err(|_| invalid_argument(format!("invalid parallelism: {value}")))?;
    if parallelism == 0 {
        return Err(invalid_argument(
            "parallelism must be greater than zero".to_owned(),
        ));
    }
    Ok(parallelism)
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

fn run_options(options: &RunOptions) -> Result<(), FractalError> {
    match options.selection() {
        RunSelection::Preset(preset) => run_preset(
            options,
            preset,
            ComparisonAuthority::AuthoritativeSamePreset,
        ),
        RunSelection::Sequence(sequence) => run_sequence(options, sequence),
    }
}

fn run_sequence(options: &RunOptions, sequence: TournamentSequence) -> Result<(), FractalError> {
    for (index, preset) in sequence.stages().iter().copied().enumerate() {
        if index > 0 {
            println!();
        }
        run_preset(options, preset, ComparisonAuthority::AdvisoryMixedPreset)?;
    }
    Ok(())
}

fn run_preset(
    options: &RunOptions,
    preset: TournamentPreset,
    comparison_authority: ComparisonAuthority,
) -> Result<(), FractalError> {
    let config = options.config_for(preset);
    let lane = options.lane();
    let species = options.species();
    print_header(preset, lane, species, &config, comparison_authority);
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
    let report = TournamentRunReport::new(
        preset,
        lane,
        comparison_authority,
        config,
        species,
        results,
        artifact,
    );
    let persisted = persist_run_artifacts(&report)?;
    print_results(&report);
    print_failures(&report);
    print_primitive_tracker_reminder(&report);
    println!(
        "artifacts={} manifest={}",
        persisted.artifact_path.display(),
        persisted.manifest_path.display()
    );
    ensure_successful_execution(&report)?;
    Ok(())
}

fn print_header(
    preset: TournamentPreset,
    lane: TournamentLane,
    species: Option<SpeciesId>,
    config: &TournamentConfig,
    comparison_authority: ComparisonAuthority,
) {
    println!("== {} ==", preset.name());
    let scope = species
        .map(|species| format!("species={species}"))
        .unwrap_or_else(|| format!("lane={}", lane.name()));
    println!(
        "{} backend={} mode={} parallelism={} seed={} dim={} levels={} seq={} depth={} stability_depth={} train_batch={} eval_batch={} train_steps={} eval_batches={}",
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
    );
    println!("comparison={}", comparison_authority.label());
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
    #[cfg(feature = "cuda")]
    println!(
        "  cargo run --release --features cuda --example tournament -- --lane leader --mode sequential"
    );
    println!("  cargo run --release --example tournament -- --sequence first-run");
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
