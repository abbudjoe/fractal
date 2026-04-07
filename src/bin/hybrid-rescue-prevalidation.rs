use std::{fmt::Write as _, path::PathBuf};

use burn::backend::Candle;
use fractal_eval_private::{
    append_hybrid_results_ledger_entry, build_baseline_hybrid_rescue_model,
    default_hybrid_rescue_prevalidation_suites, resolve_requested_hybrid_results_ledger_path,
    run_hybrid_rescue_prevalidation_with_modes, BaselineHybridRescueModelConfig,
    HybridRescuePrevalidationReport, HybridRescueProbeMode, HybridRescueSuiteKind,
    HybridResultsLedgerEntry,
};
use serde::Serialize;

type ProbeBackend = Candle<f32, i64>;

fn main() {
    if let Err(error) = run() {
        eprintln!("hybrid-rescue-prevalidation: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let device = <ProbeBackend as burn::tensor::backend::Backend>::Device::default();
    let model = build_baseline_hybrid_rescue_model::<ProbeBackend>(
        BaselineHybridRescueModelConfig::default(),
        &device,
    )
    .map_err(|error| format!("failed to build hybrid rescue baseline model: {error}"))?;
    let suites = filter_suites(
        default_hybrid_rescue_prevalidation_suites().map_err(|error| {
            format!("failed to build default hybrid prevalidation suites: {error}")
        })?,
        args.suite,
    )?;
    let report = run_hybrid_rescue_prevalidation_with_modes(
        &model,
        &suites,
        &HybridRescueProbeMode::INITIAL_FOUR,
        &device,
    )
    .map_err(|error| format!("failed to run hybrid rescue prevalidation: {error}"))?;
    let model_label = "hybrid_rescue_prevalidation_random_init_cpu_candle";
    let note = "single-root phase-1 rescue matrix over fixed suites; random-init live sanity check";
    let rendered = render_report(&report, args.output, model_label, note)?;
    maybe_append_ledger_entry(
        resolve_requested_hybrid_results_ledger_path(
            env!("CARGO_MANIFEST_DIR"),
            args.ledger_path.as_deref(),
        )
        .map_err(|error| format!("failed to resolve hybrid results ledger path: {error}"))?,
        &report,
        model_label,
        note,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    suite: SuiteSelection,
    output: OutputFormat,
    ledger_path: Option<String>,
    run_label: Option<String>,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut suite = SuiteSelection::All;
        let mut output = OutputFormat::Table;
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
                "--help" | "-h" => show_help = true,
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        if show_help {
            println!("{}", usage());
            std::process::exit(0);
        }

        Ok(Self {
            suite,
            output,
            ledger_path,
            run_label,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SuiteSelection {
    All,
    Kind(HybridRescueSuiteKind),
}

impl SuiteSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "mqar" => Ok(Self::Kind(HybridRescueSuiteKind::Mqar)),
            "copy" => Ok(Self::Kind(HybridRescueSuiteKind::Copy)),
            "induction" => Ok(Self::Kind(HybridRescueSuiteKind::Induction)),
            "retrieval-heavy" | "retrieval_heavy" => {
                Ok(Self::Kind(HybridRescueSuiteKind::RetrievalHeavy))
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
    report: &'a HybridRescuePrevalidationReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin hybrid-rescue-prevalidation -- [--suite <all|mqar|copy|induction|retrieval-heavy>] [--output <table|json>] [--ledger-path <default|path>] [--run-label <label>]"
    );
    let _ = writeln!(output, "Defaults: --suite all --output table");
    output
}

fn filter_suites(
    suites: Vec<fractal_eval_private::HybridRescueProbeSuite>,
    selection: SuiteSelection,
) -> Result<Vec<fractal_eval_private::HybridRescueProbeSuite>, String> {
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
    report: &HybridRescuePrevalidationReport,
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

fn render_table(report: &HybridRescuePrevalidationReport, model_label: &str, note: &str) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Hybrid Rescue Prevalidation");
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
                "  {:<34} accuracy={:.3} rank={:.2} target_logit={:.3} loss={:.3} local_mass={:.3} remote_mass={:.3} span_recall={:.3} token_recall={:.3}",
                mode_label(mode_report.mode),
                metrics.accuracy,
                metrics.mean_target_rank,
                metrics.mean_target_logit,
                metrics.mean_loss,
                metrics.mean_local_attention_mass,
                metrics.mean_remote_attention_mass,
                metrics.evidence_span_recall_rate,
                metrics.mean_evidence_token_recall,
            );
        }
    }

    output
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &HybridRescuePrevalidationReport,
    model_label: &str,
    note: &str,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = HybridResultsLedgerEntry::rescue_prevalidation_probe(
        model_label,
        note,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build hybrid results ledger entry: {error}"))?;
    append_hybrid_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append hybrid results ledger entry: {error}"))
}

fn suite_label(kind: HybridRescueSuiteKind) -> &'static str {
    match kind {
        HybridRescueSuiteKind::Mqar => "MQAR",
        HybridRescueSuiteKind::Copy => "Copy",
        HybridRescueSuiteKind::Induction => "Induction",
        HybridRescueSuiteKind::RetrievalHeavy => "RetrievalHeavy",
    }
}

fn mode_label(mode: HybridRescueProbeMode) -> &'static str {
    match mode {
        HybridRescueProbeMode::LocalOnly => "LocalOnly",
        HybridRescueProbeMode::RoutedRemote => "RoutedRemote",
        HybridRescueProbeMode::OracleRemote => "OracleRemote",
        HybridRescueProbeMode::OracleRemoteWithOracleExactTokenSubset => {
            "OracleRemoteWithOracleExactTokenSubset"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_to_all_suites_and_table_output() {
        let args = CliArgs::parse(std::iter::empty()).unwrap();

        assert_eq!(args.suite, SuiteSelection::All);
        assert_eq!(args.output, OutputFormat::Table);
        assert_eq!(args.ledger_path, None);
        assert_eq!(args.run_label, None);
    }

    #[test]
    fn cli_parses_specific_suite_and_ledger_args() {
        let args = CliArgs::parse(
            [
                "--suite",
                "mqar",
                "--output",
                "json",
                "--ledger-path",
                "default",
                "--run-label",
                "phase1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(
            args.suite,
            SuiteSelection::Kind(HybridRescueSuiteKind::Mqar)
        );
        assert_eq!(args.output, OutputFormat::Json);
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("phase1"));
    }
}
