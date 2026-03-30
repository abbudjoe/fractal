use fractal::{
    aggregate_results,
    error::FractalError,
    lifecycle::{Tournament, TournamentConfig, TournamentPreset, TournamentSequence},
    registry::{ComputeBackend, ExecutionMode},
    species_registry,
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
    Metal,
    Mlx,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunOptions {
    selection: Option<RunSelection>,
    seed: Option<u64>,
    execution_mode: Option<ExecutionMode>,
    backend: Option<BackendOverride>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            selection: None,
            seed: None,
            execution_mode: None,
            backend: None,
        }
    }
}

impl RunOptions {
    fn selection(&self) -> RunSelection {
        self.selection
            .unwrap_or(RunSelection::Preset(TournamentPreset::Default))
    }

    fn set_selection(&mut self, selection: RunSelection) -> Result<(), FractalError> {
        if self.selection.replace(selection).is_some() {
            return Err(invalid_argument(
                "choose either --preset or --sequence, not both".to_owned(),
            ));
        }
        Ok(())
    }

    fn config_for(&self, preset: TournamentPreset) -> TournamentConfig {
        let mut config = preset.config();
        if let Some(seed) = self.seed {
            config.seed = seed;
        }
        if let Some(execution_mode) = self.execution_mode {
            config.execution_mode = execution_mode;
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
            BackendOverride::Metal => ComputeBackend::metal_default(),
            BackendOverride::Mlx => ComputeBackend::mlx_default(),
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
        "pressure-test" => Ok(TournamentPreset::PressureTest),
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

fn parse_execution_mode(value: &str) -> Result<ExecutionMode, FractalError> {
    match value {
        "sequential" => Ok(ExecutionMode::Sequential),
        "parallel" => Ok(ExecutionMode::Parallel),
        _ => Err(invalid_argument(format!("unknown execution mode: {value}"))),
    }
}

fn parse_backend(value: &str) -> Result<BackendOverride, FractalError> {
    match value {
        "cpu" => Ok(BackendOverride::Cpu),
        "metal" => Ok(BackendOverride::Metal),
        "mlx" => Ok(BackendOverride::Mlx),
        _ => Err(invalid_argument(format!("unknown backend: {value}"))),
    }
}

fn run_options(options: &RunOptions) -> Result<(), FractalError> {
    match options.selection() {
        RunSelection::Preset(preset) => run_preset(options, preset),
        RunSelection::Sequence(sequence) => run_sequence(options, sequence),
    }
}

fn run_sequence(options: &RunOptions, sequence: TournamentSequence) -> Result<(), FractalError> {
    for (index, preset) in sequence.stages().iter().copied().enumerate() {
        if index > 0 {
            println!();
        }
        run_preset(options, preset)?;
    }
    Ok(())
}

fn run_preset(options: &RunOptions, preset: TournamentPreset) -> Result<(), FractalError> {
    let config = options.config_for(preset);
    print_header(preset, &config);
    let tournament = Tournament::new(config)?;
    let results = aggregate_results(tournament.run_generation(species_registry())?);
    print_results(results);
    Ok(())
}

fn print_header(preset: TournamentPreset, config: &TournamentConfig) {
    println!("== {} ==", preset.name());
    println!(
        "backend={} mode={} seed={} dim={} levels={} seq={} depth={} batch={} train_steps={} eval_batches={}",
        backend_name(&config.execution_backend),
        execution_mode_name(config.execution_mode),
        config.seed,
        config.dim,
        config.levels,
        config.max_seq_len,
        config.max_recursion_depth,
        config.batch_size,
        config.train_steps_per_species,
        config.eval_batches_per_family,
    );
    println!("rank  species                  stability  perplexity  arc_acc  tok/s   fitness");
}

fn print_results(results: Vec<fractal::RankedSpeciesResult>) {
    for result in results {
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

fn backend_name(backend: &ComputeBackend) -> &'static str {
    match backend {
        ComputeBackend::CpuCandle => "cpu",
        ComputeBackend::MetalWgpu { .. } => "metal",
        ComputeBackend::Mlx { .. } => "mlx",
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
    println!("  --preset <default|fast-test|research-medium|pressure-test>");
    println!("  --sequence <first-run>");
    println!("  --seed <u64>");
    println!("  --mode <sequential|parallel>");
    println!("  --backend <cpu|metal|mlx>");
    println!("  --help");
    println!();
    println!("Examples:");
    println!("  cargo run --example tournament -- --preset fast-test");
    println!("  cargo run --release --example tournament -- --preset research-medium");
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
            "--seed".to_owned(),
            "43".to_owned(),
            "--mode".to_owned(),
            "parallel".to_owned(),
            "--backend".to_owned(),
            "cpu".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Sequence(TournamentSequence::FirstRun)),
                seed: Some(43),
                execution_mode: Some(ExecutionMode::Parallel),
                backend: Some(BackendOverride::Cpu),
            })
        );
    }

    #[test]
    fn parse_command_accepts_mlx_backend_override() {
        let command = parse_command(vec![
            "--preset".to_owned(),
            "fast-test".to_owned(),
            "--backend".to_owned(),
            "mlx".to_owned(),
        ])
        .unwrap();

        assert_eq!(
            command,
            CliCommand::Run(RunOptions {
                selection: Some(RunSelection::Preset(TournamentPreset::FastTest)),
                seed: None,
                execution_mode: None,
                backend: Some(BackendOverride::Mlx),
            })
        );
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
}
