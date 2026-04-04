use std::fmt::Write as _;

use burn::backend::Candle;
use fractal_eval_private::{
    run_baseline_v2_benchmark_suite, V2BenchmarkConfig, V2BenchmarkReport, V2BenchmarkSurface,
    DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS,
};

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
    let report = run_baseline_v2_benchmark_suite::<BenchmarkBackend>(
        V2BenchmarkConfig {
            sequence_lengths: args.sequence_lengths.clone(),
            iterations: args.iterations,
            warmup_iterations: args.warmup_iterations,
        },
        &device,
    )
    .map_err(|error| format!("failed to run v2 benchmark suite: {error}"))?;
    let rendered = render_report(&report, args.output)?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    sequence_lengths: Vec<usize>,
    iterations: usize,
    warmup_iterations: usize,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut sequence_lengths = DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS.to_vec();
        let mut iterations = 3usize;
        let mut warmup_iterations = 1usize;
        let mut output = OutputFormat::Table;
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
            iterations,
            warmup_iterations,
            output,
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
        "Usage: cargo run --bin v2-benchmark-suite -- [--lengths <n1,n2,...>] [--iterations <n>] [--warmup <n>] [--output <table|json>]"
    );
    let _ = writeln!(
        output,
        "Defaults: --lengths 256,512,1024,2048,4096,8192 --iterations 3 --warmup 1 --output table"
    );
    output
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
            "  {:<12} mean_ms={:>8.3} total_ms={:>8.3} tok/s={:>10.2} rss_delta_mb={:>8.2} sparsity={:>5.3} collapse={:>5.3} exact={:>5.3} distance={:>7.2} depth={} leaves={} agreement={:>5.3}",
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
        assert_eq!(args.iterations, 3);
        assert_eq!(args.warmup_iterations, 1);
        assert_eq!(args.output, OutputFormat::Table);
    }

    #[test]
    fn cli_parses_lengths_iterations_and_json_output() {
        let args = CliArgs::parse(
            [
                "--lengths",
                "32,64,128",
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
        assert_eq!(args.iterations, 5);
        assert_eq!(args.warmup_iterations, 2);
        assert_eq!(args.output, OutputFormat::Json);
    }
}
