use std::fmt::Write as _;

use burn::backend::Candle;
use fractal_eval_private::{
    append_v2_results_ledger_entry, build_baseline_v2_synthetic_model,
    default_v2_synthetic_probe_suites, load_baseline_v2_checkpoint_model,
    resolve_requested_v2_results_ledger_path, run_v2_synthetic_probe_suites_with_modes,
    BaselineV2SyntheticModelConfig, SyntheticProbeKind, SyntheticProbeMode, SyntheticProbeReport,
    SyntheticProbeSuite, V2CheckpointSelection, V2ResultsLedgerEntry,
};
use serde::Serialize;
use std::path::PathBuf;

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
    let loaded_model = load_model(&args, &device)?;
    let suites = filter_suites(default_v2_synthetic_probe_suites(), args.suite)?;
    let execution_modes = if args.oracle {
        &SyntheticProbeMode::ALL_WITH_ORACLE[..]
    } else {
        &SyntheticProbeMode::ALL[..]
    };
    let report = run_v2_synthetic_probe_suites_with_modes(
        &loaded_model.model,
        &suites,
        execution_modes,
        &device,
    )
    .map_err(|error| format!("failed to run v2 synthetic probes: {error}"))?;
    let rendered = render_report(
        &report,
        args.output,
        &loaded_model.model_label,
        &render_note(&loaded_model.note, args.oracle),
    )?;
    maybe_append_ledger_entry(
        resolve_requested_v2_results_ledger_path(
            env!("CARGO_MANIFEST_DIR"),
            args.ledger_path.as_deref(),
        )
        .map_err(|error| format!("failed to resolve v2 results ledger path: {error}"))?,
        &report,
        &loaded_model.model_label,
        &render_note(&loaded_model.note, args.oracle),
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    suite: SuiteSelection,
    output: OutputFormat,
    oracle: bool,
    checkpoint_report: Option<PathBuf>,
    checkpoint_kind_override: Option<V2CheckpointSelection>,
    ledger_path: Option<String>,
    run_label: Option<String>,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut suite = SuiteSelection::All;
        let mut output = OutputFormat::Table;
        let mut oracle = false;
        let mut checkpoint_report = None;
        let mut checkpoint_kind_override = None;
        let mut ledger_path = None;
        let mut run_label = None;
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
                "--oracle" => {
                    oracle = true;
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
                    checkpoint_kind_override = Some(parse_checkpoint_kind(&value)?);
                }
                "--ledger-path" => {
                    ledger_path = Some(
                        iter.next()
                            .ok_or_else(|| "--ledger-path requires a value".to_owned())?,
                    );
                }
                "--run-label" => {
                    run_label = Some(
                        iter.next()
                            .ok_or_else(|| "--run-label requires a value".to_owned())?,
                    );
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

        if checkpoint_report.is_none() && checkpoint_kind_override.is_some() {
            return Err("--checkpoint-kind requires --checkpoint-report".to_owned());
        }

        Ok(Self {
            suite,
            output,
            oracle,
            checkpoint_report,
            checkpoint_kind_override,
            ledger_path,
            run_label,
        })
    }

    fn checkpoint_kind(&self) -> V2CheckpointSelection {
        self.checkpoint_kind_override
            .unwrap_or(V2CheckpointSelection::Best)
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
    model: &'a str,
    note: &'a str,
    report: &'a SyntheticProbeReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-synthetic-probe -- [--suite <all|copy|associative-recall|induction|noisy-retrieval|far-token-comparison>] [--output <table|json>] [--oracle] [--checkpoint-report <report.json>] [--checkpoint-kind <best|final>] [--ledger-path <default|path>] [--run-label <label>]"
    );
    let _ = writeln!(
        output,
        "Defaults: --suite all --output table --checkpoint-kind best"
    );
    output
}

struct LoadedModel {
    model: fractal_eval_private::BaselineV2SyntheticModel<ProbeBackend>,
    model_label: String,
    note: String,
}

fn load_model(
    args: &CliArgs,
    device: &<ProbeBackend as burn::tensor::backend::Backend>::Device,
) -> Result<LoadedModel, String> {
    match &args.checkpoint_report {
        Some(report_path) => {
            let loaded = load_baseline_v2_checkpoint_model::<ProbeBackend>(
                report_path,
                args.checkpoint_kind(),
                device,
            )
            .map_err(|error| format!("failed to load trained checkpoint model: {error}"))?;
            Ok(LoadedModel {
                model: loaded.model,
                model_label: format!("baseline_v2_smoke_checkpoint_{}", loaded.selection.label()),
                note: format!(
                    "trained smoke checkpoint loaded from {} using the {} checkpoint artifact",
                    loaded.report_path.display(),
                    loaded.selection.label()
                ),
            })
        }
        None => {
            let model = build_baseline_v2_synthetic_model::<ProbeBackend>(
                BaselineV2SyntheticModelConfig::default(),
                device,
            )
            .map_err(|error| format!("failed to build baseline v2 model: {error}"))?;
            Ok(LoadedModel {
                model,
                model_label: "baseline_v2_random_init_cpu_candle".to_string(),
                note: "untrained random baseline; use for live execution sanity checks, not architecture quality claims".to_string(),
            })
        }
    }
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

fn render_report(
    report: &SyntheticProbeReport,
    output: OutputFormat,
    model_label: &str,
    note: &str,
) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report, model_label, note)),
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedReport {
            model: model_label,
            note,
            report,
        })
        .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_note(note: &str, oracle: bool) -> String {
    if oracle {
        format!("{note}; oracle routing forced to the evidence leaf")
    } else {
        note.to_string()
    }
}

fn render_table(report: &SyntheticProbeReport, model_label: &str, note: &str) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Synthetic Probe Live Run");
    let _ = writeln!(output, "model: {model_label}");
    let _ = writeln!(output, "note: {note}");

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

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &SyntheticProbeReport,
    model_label: &str,
    note: &str,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V2ResultsLedgerEntry::synthetic_probe(
        model_label,
        note,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build v2 results ledger entry: {error}"))?;
    append_v2_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append v2 results ledger entry: {error}"))
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
        fractal_eval_private::SyntheticProbeMode::SummariesOnly => "SummariesOnly",
        fractal_eval_private::SyntheticProbeMode::TreeOnly => "TreeOnly",
        fractal_eval_private::SyntheticProbeMode::TreePlusExactRead => "TreePlusExactRead",
        fractal_eval_private::SyntheticProbeMode::OracleTreeOnly => "OracleTreeOnly",
        fractal_eval_private::SyntheticProbeMode::OracleTreePlusExactRead => {
            "OracleTreePlusExactRead"
        }
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
        assert!(!args.oracle);
        assert_eq!(args.checkpoint_report, None);
        assert_eq!(args.checkpoint_kind_override, None);
        assert_eq!(args.ledger_path, None);
        assert_eq!(args.run_label, None);
        assert_eq!(args.checkpoint_kind(), V2CheckpointSelection::Best);
    }

    #[test]
    fn cli_parses_specific_suite_json_output_and_checkpoint_source() {
        let args = CliArgs::parse(
            [
                "--suite",
                "copy",
                "--output",
                "json",
                "--oracle",
                "--checkpoint-report",
                "/tmp/report.json",
                "--checkpoint-kind",
                "final",
                "--ledger-path",
                "default",
                "--run-label",
                "learned-probe",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.suite, SuiteSelection::Kind(SyntheticProbeKind::Copy));
        assert_eq!(args.output, OutputFormat::Json);
        assert!(args.oracle);
        assert_eq!(
            args.checkpoint_report,
            Some(PathBuf::from("/tmp/report.json"))
        );
        assert_eq!(
            args.checkpoint_kind_override,
            Some(V2CheckpointSelection::Final)
        );
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("learned-probe"));
        assert_eq!(args.checkpoint_kind(), V2CheckpointSelection::Final);
    }

    #[test]
    fn cli_rejects_checkpoint_kind_without_checkpoint_report() {
        let error = CliArgs::parse(
            ["--checkpoint-kind", "final"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap_err();

        assert!(error.contains("--checkpoint-kind requires --checkpoint-report"));
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

        let rendered = render_report(
            &report,
            OutputFormat::Json,
            "baseline_v2_random_init_cpu_candle",
            "untrained random baseline; use for live execution sanity checks, not architecture quality claims",
        )
        .unwrap();

        assert!(rendered.contains("\"model\": \"baseline_v2_random_init_cpu_candle\""));
        assert!(rendered.contains("\"kind\": \"copy\""));
    }
}
