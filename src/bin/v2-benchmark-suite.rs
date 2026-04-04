use std::fmt::Write as _;

use burn::backend::Candle;
use fractal_eval_private::{
    load_baseline_v2_checkpoint_model, run_baseline_v2_benchmark_suite,
    run_v2_benchmark_suite_for_model, V2BenchmarkConfig, V2BenchmarkReport, V2BenchmarkSurface,
    V2CheckpointSelection, DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS,
};
use std::path::PathBuf;

type BenchmarkBackend = Candle<f32, i64>;

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-benchmark-suite: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let device = <BenchmarkBackend as burn::tensor::backend::Backend>::Device::default();
    let report = run_benchmark(&args, &device)?;
    let rendered = render_report(&report, args.output)?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    sequence_lengths: Vec<usize>,
    leaf_size_override: Option<usize>,
    iterations: usize,
    warmup_iterations: usize,
    output: OutputFormat,
    checkpoint_report: Option<PathBuf>,
    checkpoint_kind: V2CheckpointSelection,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut sequence_lengths = DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS.to_vec();
        let mut leaf_size_override = None;
        let mut iterations = 3usize;
        let mut warmup_iterations = 1usize;
        let mut output = OutputFormat::Table;
        let mut checkpoint_report = None;
        let mut checkpoint_kind = V2CheckpointSelection::Best;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--lengths" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--lengths requires a comma-separated value".to_owned())?;
                    sequence_lengths = parse_lengths(&value)?;
                }
                "--leaf-size" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--leaf-size requires a value".to_owned())?;
                    leaf_size_override = Some(parse_positive_usize("--leaf-size", &value)?);
                }
                "--iterations" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--iterations requires a value".to_owned())?;
                    iterations = parse_positive_usize("--iterations", &value)?;
                }
                "--warmup" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--warmup requires a value".to_owned())?;
                    warmup_iterations = parse_usize("--warmup", &value)?;
                }
                "--output" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--output requires a value".to_owned())?;
                    output = OutputFormat::parse(&value)?;
                }
                "--checkpoint-report" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--checkpoint-report requires a value".to_owned())?;
                    checkpoint_report = Some(PathBuf::from(value));
                }
                "--checkpoint-kind" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--checkpoint-kind requires a value".to_owned())?;
                    checkpoint_kind = parse_checkpoint_kind(&value)?;
                }
                "--help" | "-h" => {
                    show_help = true;
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        if show_help {
            println!("{}", usage());
            std::process::exit(0);
        }

        Ok(Self {
            sequence_lengths,
            leaf_size_override,
            iterations,
            warmup_iterations,
            output,
            checkpoint_report,
            checkpoint_kind,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Table,
    Json,
}

impl OutputFormat {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "table" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            _ => Err(format!("unknown output format: {value}")),
        }
    }
}

fn parse_lengths(value: &str) -> Result<Vec<usize>, String> {
    let lengths = value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_positive_usize("--lengths", part.trim()))
        .collect::<Result<Vec<_>, _>>()?;

    if lengths.is_empty() {
        return Err("--lengths must include at least one positive integer".to_owned());
    }

    Ok(lengths)
}

fn parse_positive_usize(flag: &str, value: &str) -> Result<usize, String> {
    let parsed = parse_usize(flag, value)?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }

    Ok(parsed)
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("{flag} must be an integer: {error}"))
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-benchmark-suite -- [--lengths <n1,n2,...>] [--leaf-size <n>] [--iterations <n>] [--warmup <n>] [--output <table|json>] [--checkpoint-report <report.json>] [--checkpoint-kind <best|final>]"
    );
    let _ = writeln!(
        output,
        "Defaults: --lengths 256,512,1024,2048,4096,8192 --leaf-size 16 --iterations 3 --warmup 1 --output table --checkpoint-kind best"
    );
    output
}

fn run_benchmark(
    args: &CliArgs,
    device: &<BenchmarkBackend as burn::tensor::backend::Backend>::Device,
) -> Result<V2BenchmarkReport, String> {
    match &args.checkpoint_report {
        Some(report_path) => {
            if args.leaf_size_override.is_some() {
                return Err(
                    "--leaf-size cannot be combined with --checkpoint-report; trained checkpoints already carry a validated leaf_size".to_owned(),
                );
            }
            let loaded = load_baseline_v2_checkpoint_model::<BenchmarkBackend>(
                report_path,
                args.checkpoint_kind,
                device,
            )
            .map_err(|error| format!("failed to load trained checkpoint model: {error}"))?;
            let config = V2BenchmarkConfig {
                sequence_lengths: args.sequence_lengths.clone(),
                leaf_size: loaded.report.config.model.leaf_size,
                iterations: args.iterations,
                warmup_iterations: args.warmup_iterations,
            };
            run_v2_benchmark_suite_for_model(
                &loaded.model,
                config,
                format!("baseline_v2_smoke_checkpoint_{}", loaded.selection.label()),
                format!(
                    "trained smoke checkpoint loaded from {} using the {} checkpoint artifact; RSS metrics are sampled from getrusage(RUSAGE_SELF) after warmup and are useful for trend detection, not precise kernel-level attribution",
                    loaded.report_path.display(),
                    loaded.selection.label()
                ),
                device,
            )
            .map_err(|error| format!("failed to run v2 benchmark suite: {error}"))
        }
        None => run_baseline_v2_benchmark_suite::<BenchmarkBackend>(
            V2BenchmarkConfig {
                sequence_lengths: args.sequence_lengths.clone(),
                leaf_size: args.leaf_size_override.unwrap_or(16),
                iterations: args.iterations,
                warmup_iterations: args.warmup_iterations,
            },
            device,
        )
        .map_err(|error| format!("failed to run v2 benchmark suite: {error}")),
    }
}

fn parse_checkpoint_kind(value: &str) -> Result<V2CheckpointSelection, String> {
    match value {
        "best" => Ok(V2CheckpointSelection::Best),
        "final" => Ok(V2CheckpointSelection::Final),
        _ => Err(format!(
            "unknown checkpoint kind: {value} (expected best or final)"
        )),
    }
}

fn render_report(report: &V2BenchmarkReport, output: OutputFormat) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report)),
        OutputFormat::Json => serde_json::to_string_pretty(report)
            .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &V2BenchmarkReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Benchmark and Observability Suite");
    let _ = writeln!(output, "model: {}", report.model);
    let _ = writeln!(output, "leaf_size: {}", report.config.leaf_size);
    let _ = writeln!(output, "note: {}", report.note);

    let mut current_length = None;
    for entry in &report.entries {
        if current_length != Some(entry.sequence_length) {
            current_length = Some(entry.sequence_length);
            let _ = writeln!(output);
            let _ = writeln!(output, "sequence_length={}", entry.sequence_length);
        }
        let _ = writeln!(
            output,
            "  {:<12} mean_ms={:>8.3} total_ms={:>8.3} tok/s={:>10.2} rss_delta_mb={:>8.2} sparsity={:>5.3} collapse={:>5.3} exact={:>5.3} distance={:>7.2} depth={} leaves={} agreement={:>5.3} dead_nodes={} leaf_bins={}",
            surface_label(entry.surface),
            entry.mean_wall_time_ms,
            entry.total_wall_time_ms,
            entry.tokens_per_sec,
            entry.peak_rss_delta_bytes as f64 / (1024.0 * 1024.0),
            entry.observability.routing_sparsity,
            entry.observability.root_collapse_mean_pairwise_cosine_similarity,
            entry.observability.exact_read_usage,
            entry.observability.mean_retrieval_distance,
            entry.observability.tree_depth_reached,
            entry.observability.level0_leaf_count,
            entry.observability.head_agreement_rate,
            entry.observability.has_dead_or_unused_tree_nodes,
            entry.observability.selected_leaf_usage.len(),
        );
    }

    output
}

fn surface_label(surface: V2BenchmarkSurface) -> &'static str {
    match surface {
        V2BenchmarkSurface::TokenAppend => "token_append",
        V2BenchmarkSurface::LeafSealing => "leaf_sealing",
        V2BenchmarkSurface::TreeUpdate => "tree_update",
        V2BenchmarkSurface::Routing => "routing",
        V2BenchmarkSurface::ExactLeafRead => "exact_read",
        V2BenchmarkSurface::ForwardPass => "forward_pass",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_match_phase_10_contract() {
        let args = CliArgs::parse(std::iter::empty()).unwrap();

        assert_eq!(
            args.sequence_lengths,
            DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS.to_vec()
        );
        assert_eq!(args.leaf_size_override, None);
        assert_eq!(args.iterations, 3);
        assert_eq!(args.warmup_iterations, 1);
        assert_eq!(args.output, OutputFormat::Table);
        assert_eq!(args.checkpoint_report, None);
        assert_eq!(args.checkpoint_kind, V2CheckpointSelection::Best);
    }

    #[test]
    fn cli_parses_lengths_iterations_and_json_output() {
        let args = CliArgs::parse(
            [
                "--lengths",
                "32,64,128",
                "--leaf-size",
                "32",
                "--iterations",
                "5",
                "--warmup",
                "2",
                "--output",
                "json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.sequence_lengths, vec![32, 64, 128]);
        assert_eq!(args.leaf_size_override, Some(32));
        assert_eq!(args.iterations, 5);
        assert_eq!(args.warmup_iterations, 2);
        assert_eq!(args.output, OutputFormat::Json);
    }

    #[test]
    fn cli_parses_checkpoint_source() {
        let args = CliArgs::parse(
            [
                "--checkpoint-report",
                "/tmp/report.json",
                "--checkpoint-kind",
                "final",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(
            args.checkpoint_report,
            Some(PathBuf::from("/tmp/report.json"))
        );
        assert_eq!(args.checkpoint_kind, V2CheckpointSelection::Final);
    }

    #[test]
    fn run_benchmark_rejects_leaf_size_override_for_checkpoint_models() {
        let args = CliArgs {
            sequence_lengths: vec![32],
            leaf_size_override: Some(32),
            iterations: 1,
            warmup_iterations: 0,
            output: OutputFormat::Table,
            checkpoint_report: Some(PathBuf::from("/tmp/report.json")),
            checkpoint_kind: V2CheckpointSelection::Best,
        };

        let error = run_benchmark(&args, &Default::default()).unwrap_err();

        assert!(error.contains("--leaf-size cannot be combined with --checkpoint-report"));
    }
}
