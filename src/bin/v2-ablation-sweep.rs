use std::fmt::Write as _;

use burn::backend::Candle;
use fractal_eval_private::{
    append_v2_results_ledger_entry, default_v2_synthetic_probe_suites,
    resolve_requested_v2_results_ledger_path, run_required_v2_ablation_sweep, SyntheticProbeKind,
    SyntheticProbeMode, SyntheticProbeSuite, V2AblationConfig, V2AblationReport,
    V2ResultsLedgerEntry, V2RootTopology,
};
use std::path::PathBuf;

type AblationBackend = Candle<f32, i64>;

const ABLATION_MODEL_LABEL: &str = "baseline_v2_required_ablation_cpu_candle";

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-ablation-sweep: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let device = <AblationBackend as burn::tensor::backend::Backend>::Device::default();
    let suites = filter_suites(default_v2_synthetic_probe_suites(), args.suite)?;
    let report = run_required_v2_ablation_sweep::<AblationBackend>(
        V2AblationConfig::default(),
        &suites,
        &device,
    )
    .map_err(|error| format!("failed to run v2 ablation sweep: {error}"))?;
    let rendered = render_report(&report, args.output)?;
    maybe_append_ledger_entry(
        resolve_requested_v2_results_ledger_path(
            env!("CARGO_MANIFEST_DIR"),
            args.ledger_path.as_deref(),
        )
        .map_err(|error| format!("failed to resolve v2 results ledger path: {error}"))?,
        &report,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    suite: SuiteSelection,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut suite = SuiteSelection::All;
        let mut ledger_path = None;
        let mut run_label = None;
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

        Ok(Self {
            suite,
            ledger_path,
            run_label,
            output,
        })
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

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-ablation-sweep -- [--suite <all|copy|associative-recall|induction|noisy-retrieval|far-token-comparison>] [--ledger-path <default|path>] [--run-label <label>] [--output <table|json>]"
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

fn render_report(report: &V2AblationReport, output: OutputFormat) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report)),
        OutputFormat::Json => serde_json::to_string_pretty(report)
            .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &V2AblationReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Required Ablation Sweep");
    let _ = writeln!(output, "note: {}", report.note);

    for case in &report.cases {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "{} roots={} state_budget={} readout_budget={} per_root_state={} per_root_readout={}",
            topology_label(case.topology),
            case.model_config.root_count,
            case.model_config.total_root_state_dim,
            case.model_config.total_root_readout_dim,
            case.model_config.root_state_dim(),
            case.model_config.root_readout_dim(),
        );
        for suite in &case.synthetic.suites {
            let _ = writeln!(output, "  {}", suite_label(suite.kind));
            for mode_report in &suite.mode_reports {
                let metrics = mode_report.metrics;
                let _ = writeln!(
                    output,
                    "    {:<18} accuracy={:.3} target_logit={:.3} loss={:.3}",
                    mode_label(mode_report.mode),
                    metrics.accuracy,
                    metrics.mean_target_logit,
                    metrics.mean_loss
                );
            }
        }
    }

    output
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &V2AblationReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V2ResultsLedgerEntry::ablation_sweep(
        ABLATION_MODEL_LABEL,
        report.note.as_str(),
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build v2 results ledger entry: {error}"))?;
    append_v2_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append v2 results ledger entry: {error}"))
}

fn topology_label(topology: V2RootTopology) -> &'static str {
    match topology {
        V2RootTopology::SingleRoot => "single_root",
        V2RootTopology::MultiRoot => "multi_root",
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

fn mode_label(mode: SyntheticProbeMode) -> &'static str {
    match mode {
        SyntheticProbeMode::NoMemory => "NoMemory",
        SyntheticProbeMode::SummariesOnly => "SummariesOnly",
        SyntheticProbeMode::TreeOnly => "TreeOnly",
        SyntheticProbeMode::TreePlusExactRead => "TreePlusExactRead",
        SyntheticProbeMode::OracleTreeOnly => "OracleTreeOnly",
        SyntheticProbeMode::OracleTreePlusExactRead => "OracleTreePlusExactRead",
        SyntheticProbeMode::OracleTreePlusOracleExactRead => "OracleTreePlusOracleExactRead",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_to_all_suites_and_table_output() {
        let args = CliArgs::parse(std::iter::empty()).unwrap();

        assert_eq!(args.suite, SuiteSelection::All);
        assert_eq!(args.ledger_path, None);
        assert_eq!(args.run_label, None);
        assert_eq!(args.output, OutputFormat::Table);
    }

    #[test]
    fn cli_parses_specific_suite_and_json_output() {
        let args = CliArgs::parse(
            [
                "--suite",
                "copy",
                "--ledger-path",
                "default",
                "--run-label",
                "ablation-sweep",
                "--output",
                "json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.suite, SuiteSelection::Kind(SyntheticProbeKind::Copy));
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("ablation-sweep"));
        assert_eq!(args.output, OutputFormat::Json);
    }
}
