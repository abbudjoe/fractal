use std::fmt::Write as _;

use burn::backend::Candle;
use fractal_eval_private::{
    build_baseline_v2_synthetic_model, default_v2_synthetic_probe_suites,
    run_v2_synthetic_probe_suites, BaselineV2SyntheticModelConfig, SyntheticProbeKind,
    SyntheticProbeReport, SyntheticProbeSuite,
};
use serde::Serialize;

type ProbeBackend = Candle<f32, i64>;

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-synthetic-probe: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let device = <ProbeBackend as burn::tensor::backend::Backend>::Device::default();
    let model = build_baseline_v2_synthetic_model::<ProbeBackend>(
        BaselineV2SyntheticModelConfig::default(),
        &device,
    )
    .map_err(|error| format!("failed to build baseline v2 model: {error}"))?;
    let suites = filter_suites(default_v2_synthetic_probe_suites(), args.suite)?;
    let report = run_v2_synthetic_probe_suites(&model, &suites, &device)
        .map_err(|error| format!("failed to run v2 synthetic probes: {error}"))?;
    let rendered = render_report(&report, args.output)?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    suite: SuiteSelection,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut suite = SuiteSelection::All;
        let mut output = OutputFormat::Table;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--suite" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--suite requires a value".to_owned())?;
                    suite = SuiteSelection::parse(&value)?;
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

        Ok(Self { suite, output })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuiteSelection {
    All,
    Kind(SyntheticProbeKind),
}

impl SuiteSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "copy" => Ok(Self::Kind(SyntheticProbeKind::Copy)),
            "associative-recall" | "associative_recall" | "assoc" => {
                Ok(Self::Kind(SyntheticProbeKind::AssociativeRecall))
            }
            "induction" => Ok(Self::Kind(SyntheticProbeKind::Induction)),
            "noisy-retrieval" | "noisy_retrieval" => {
                Ok(Self::Kind(SyntheticProbeKind::NoisyRetrieval))
            }
            "far-token-comparison" | "far_token_comparison" | "compare" => {
                Ok(Self::Kind(SyntheticProbeKind::FarTokenComparison))
            }
            _ => Err(format!("unknown suite selection: {value}")),
        }
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

#[derive(Debug, Serialize)]
struct RenderedReport<'a> {
    model: &'static str,
    note: &'static str,
    report: &'a SyntheticProbeReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-synthetic-probe -- [--suite <all|copy|associative-recall|induction|noisy-retrieval|far-token-comparison>] [--output <table|json>]"
    );
    let _ = writeln!(output, "Defaults: --suite all --output table");
    output
}

fn filter_suites(
    suites: Vec<SyntheticProbeSuite>,
    selection: SuiteSelection,
) -> Result<Vec<SyntheticProbeSuite>, String> {
    let filtered = match selection {
        SuiteSelection::All => suites,
        SuiteSelection::Kind(kind) => suites
            .into_iter()
            .filter(|suite| suite.kind == kind)
            .collect(),
    };
    if filtered.is_empty() {
        return Err("no suites matched the requested selection".to_owned());
    }

    Ok(filtered)
}

fn render_report(report: &SyntheticProbeReport, output: OutputFormat) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report)),
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedReport {
            model: "baseline_v2_random_init_cpu_candle",
            note: "untrained random baseline; use for live execution sanity checks, not architecture quality claims",
            report,
        })
        .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &SyntheticProbeReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Synthetic Probe Live Run");
    let _ = writeln!(output, "model: baseline_v2 (random init, CPU Candle)");
    let _ = writeln!(
        output,
        "note: this is an untrained live execution run, so accuracy is expected to be noisy"
    );

    for suite in &report.suites {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "{} (samples={})",
            suite_label(suite.kind),
            suite.sample_count
        );
        for mode_report in &suite.mode_reports {
            let metrics = mode_report.metrics;
            let _ = writeln!(
                output,
                "  {:<18} accuracy={:.3} target_logit={:.3} loss={:.3}",
                mode_label(mode_report.mode),
                metrics.accuracy,
                metrics.mean_target_logit,
                metrics.mean_loss
            );
        }
    }

    output
}

fn suite_label(kind: SyntheticProbeKind) -> &'static str {
    match kind {
        SyntheticProbeKind::Copy => "Copy",
        SyntheticProbeKind::AssociativeRecall => "AssociativeRecall",
        SyntheticProbeKind::Induction => "Induction",
        SyntheticProbeKind::NoisyRetrieval => "NoisyRetrieval",
        SyntheticProbeKind::FarTokenComparison => "FarTokenComparison",
    }
}

fn mode_label(mode: fractal_eval_private::SyntheticProbeMode) -> &'static str {
    match mode {
        fractal_eval_private::SyntheticProbeMode::NoMemory => "NoMemory",
        fractal_eval_private::SyntheticProbeMode::TreeOnly => "TreeOnly",
        fractal_eval_private::SyntheticProbeMode::TreePlusExactRead => "TreePlusExactRead",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fractal_eval_private::{
        SyntheticProbeMetrics, SyntheticProbeMode, SyntheticProbeModeReport,
    };

    #[test]
    fn cli_defaults_to_all_suites_and_table_output() {
        let args = CliArgs::parse(std::iter::empty()).unwrap();

        assert_eq!(args.suite, SuiteSelection::All);
        assert_eq!(args.output, OutputFormat::Table);
    }

    #[test]
    fn cli_parses_specific_suite_and_json_output() {
        let args = CliArgs::parse(
            ["--suite", "copy", "--output", "json"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.suite, SuiteSelection::Kind(SyntheticProbeKind::Copy));
        assert_eq!(args.output, OutputFormat::Json);
    }

    #[test]
    fn filter_suites_keeps_only_requested_kind() {
        let suites = filter_suites(
            default_v2_synthetic_probe_suites(),
            SuiteSelection::Kind(SyntheticProbeKind::Induction),
        )
        .unwrap();

        assert_eq!(suites.len(), 1);
        assert_eq!(suites[0].kind, SyntheticProbeKind::Induction);
    }

    #[test]
    fn json_output_serializes_report_metadata_and_suite_kind() {
        let report = SyntheticProbeReport {
            suites: vec![fractal_eval_private::SyntheticProbeSuiteReport {
                kind: SyntheticProbeKind::Copy,
                sample_count: 1,
                mode_reports: vec![SyntheticProbeModeReport {
                    mode: SyntheticProbeMode::NoMemory,
                    metrics: SyntheticProbeMetrics {
                        accuracy: 0.0,
                        mean_target_logit: 0.0,
                        mean_loss: 1.0,
                    },
                    sample_results: Vec::new(),
                }],
            }],
        };

        let rendered = render_report(&report, OutputFormat::Json).unwrap();

        assert!(rendered.contains("\"model\": \"baseline_v2_random_init_cpu_candle\""));
        assert!(rendered.contains("\"kind\": \"copy\""));
    }
}
