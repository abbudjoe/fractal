use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::CpuTrainBackend;
use fractal_eval_private::{
    append_v2_results_ledger_entry, default_v2_synthetic_probe_suites,
    filter_synthetic_probe_suites, mode_eval_summary_by_kind,
    resolve_requested_v2_results_ledger_path, run_baseline_v2_supervised_synthetic_train,
    SyntheticProbeKind, SyntheticProbeMode, SyntheticProbeReport, V2CheckpointKind,
    V2ResultsLedgerEntry, V2SupervisedSyntheticTrainConfig, V2SupervisedSyntheticTrainReport,
    V2_SUPERVISED_SYNTHETIC_LEAF_SIZE,
};
use serde::Serialize;

const MODEL_LABEL: &str = "baseline_v2_supervised_synthetic_cpu_candle";
const NOTE: &str = "supervised synthetic retrieval training over held-out probe samples";

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-supervised-synthetic-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let suites = filter_requested_suites(default_v2_synthetic_probe_suites(), args.suite)?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let mut config = V2SupervisedSyntheticTrainConfig::new(output_dir, suites);
    config.training_mode = args.training_mode;
    config.train_steps = args.steps;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    if let Some(root_count) = args.root_count {
        config.model = config
            .model
            .with_root_count_preserving_total_budget(root_count);
    }
    let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
    let result = run_baseline_v2_supervised_synthetic_train::<CpuTrainBackend>(config, &device)
        .map_err(|error| format!("failed to run supervised synthetic training: {error}"))?;
    let rendered = render_report(&result.report, args.output)?;
    maybe_append_ledger_entry(
        resolve_requested_v2_results_ledger_path(&repo_root, args.ledger_path.as_deref())
            .map_err(|error| format!("failed to resolve v2 results ledger path: {error}"))?,
        &result.report,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    suite: SuiteSelection,
    output_dir: Option<PathBuf>,
    steps: usize,
    eval_holdout_every: usize,
    learning_rate: f64,
    root_count: Option<usize>,
    training_mode: SyntheticProbeMode,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut suite = SuiteSelection::All;
        let mut output_dir = None;
        let mut steps = fractal_eval_private::DEFAULT_V2_SUPERVISED_SYNTHETIC_STEPS;
        let mut eval_holdout_every =
            fractal_eval_private::DEFAULT_V2_SUPERVISED_SYNTHETIC_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = fractal_eval_private::DEFAULT_V2_SUPERVISED_SYNTHETIC_LEARNING_RATE;
        let mut root_count = None;
        let mut training_mode = SyntheticProbeMode::TreePlusExactRead;
        let mut ledger_path = None;
        let mut run_label = None;
        let mut output = OutputFormat::Table;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--suite" => {
                    suite = SuiteSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--suite requires a value".to_owned())?,
                    )?;
                }
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        iter.next()
                            .ok_or_else(|| "--output-dir requires a value".to_owned())?,
                    ));
                }
                "--steps" => {
                    steps = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--steps requires a value".to_owned())?,
                        "--steps",
                    )?;
                }
                "--eval-holdout-every" => {
                    eval_holdout_every = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--eval-holdout-every requires a value".to_owned())?,
                        "--eval-holdout-every",
                    )?;
                }
                "--learning-rate" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--learning-rate requires a value".to_owned())?;
                    learning_rate = value.parse::<f64>().map_err(|error| {
                        format!("invalid --learning-rate value '{value}': {error}")
                    })?;
                }
                "--root-count" => {
                    root_count = Some(parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--root-count requires a value".to_owned())?,
                        "--root-count",
                    )?);
                }
                "--training-mode" => {
                    training_mode = parse_training_mode(
                        &iter
                            .next()
                            .ok_or_else(|| "--training-mode requires a value".to_owned())?,
                    )?;
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
                "--output" => {
                    output = OutputFormat::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--output requires a value".to_owned())?,
                    )?;
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
            output_dir,
            steps,
            eval_holdout_every,
            learning_rate,
            root_count,
            training_mode,
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

#[derive(Debug, Serialize)]
struct RenderedReport<'a> {
    model: &'static str,
    note: &'static str,
    report: &'a V2SupervisedSyntheticTrainReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-supervised-synthetic-train -- [--suite <all|copy|associative-recall|induction|noisy-retrieval|far-token-comparison>] [--output-dir <path>] [--steps <usize>] [--eval-holdout-every <usize>] [--learning-rate <float>] [--root-count <usize>] [--training-mode <tree-only|tree-plus-exact-read|oracle-tree-only|oracle-tree-plus-exact-read|oracle-tree-plus-oracle-exact-read>] [--ledger-path <default|path>] [--run-label <label>] [--output <table|json>]"
    );
    let _ = writeln!(
        output,
        "Defaults: suite=all output-dir=artifacts/v2-supervised-synthetic/<timestamp> steps={} eval-holdout-every={} root_count=2 leaf_size={} training-mode=tree-plus-exact-read output=table",
        fractal_eval_private::DEFAULT_V2_SUPERVISED_SYNTHETIC_STEPS,
        fractal_eval_private::DEFAULT_V2_SUPERVISED_SYNTHETIC_EVAL_HOLDOUT_EVERY,
        V2_SUPERVISED_SYNTHETIC_LEAF_SIZE
    );
    output
}

fn filter_requested_suites(
    suites: Vec<fractal_eval_private::SyntheticProbeSuite>,
    selection: SuiteSelection,
) -> Result<Vec<fractal_eval_private::SyntheticProbeSuite>, String> {
    let filtered = match selection {
        SuiteSelection::All => suites,
        SuiteSelection::Kind(kind) => filter_synthetic_probe_suites(suites, &[kind]),
    };
    if filtered.is_empty() {
        return Err("no suites matched the requested selection".to_owned());
    }
    Ok(filtered)
}

fn parse_positive_usize(value: String, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_training_mode(value: &str) -> Result<SyntheticProbeMode, String> {
    match value {
        "tree-only" | "tree_only" => Ok(SyntheticProbeMode::TreeOnly),
        "tree-plus-exact-read" | "tree_plus_exact_read" => {
            Ok(SyntheticProbeMode::TreePlusExactRead)
        }
        "oracle-tree-only" | "oracle_tree_only" => Ok(SyntheticProbeMode::OracleTreeOnly),
        "oracle-tree-plus-exact-read" | "oracle_tree_plus_exact_read" => {
            Ok(SyntheticProbeMode::OracleTreePlusExactRead)
        }
        "oracle-tree-plus-oracle-exact-read" | "oracle_tree_plus_oracle_exact_read" => {
            Ok(SyntheticProbeMode::OracleTreePlusOracleExactRead)
        }
        _ => Err(format!("unknown training mode: {value}")),
    }
}

fn render_report(
    report: &V2SupervisedSyntheticTrainReport,
    output: OutputFormat,
) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report)),
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedReport {
            model: MODEL_LABEL,
            note: NOTE,
            report,
        })
        .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &V2SupervisedSyntheticTrainReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Supervised Synthetic Training");
    let _ = writeln!(output, "model: {MODEL_LABEL}");
    let _ = writeln!(output, "note: {NOTE}");
    let _ = writeln!(
        output,
        "config: suites={} steps={} eval_holdout_every={} lr={} root_count={} leaf_size={} training_mode={}",
        report.config.suites.len(),
        report.config.train_steps,
        report.config.eval_holdout_every,
        report.config.learning_rate,
        report.config.model.root_count,
        report.config.model.leaf_size,
        mode_label(report.config.training_mode),
    );
    let _ = writeln!(
        output,
        "model_contract: total_root_state_dim={} total_root_readout_dim={} per_root_state_dim={} per_root_readout_dim={}",
        report.config.model.total_root_state_dim,
        report.config.model.total_root_readout_dim,
        report.config.model.root_state_dim(),
        report.config.model.root_readout_dim(),
    );
    let _ = writeln!(
        output,
        "split: train_samples={} eval_samples={}",
        report.split.train_sample_count, report.split.eval_sample_count
    );
    let _ = writeln!(
        output,
        "initial_eval: accuracy={:.3} target_logit={:.3} loss={:.3}",
        report.initial_eval_metrics.accuracy,
        report.initial_eval_metrics.mean_target_logit,
        report.initial_eval_metrics.mean_loss
    );
    let _ = writeln!(
        output,
        "final_eval:   accuracy={:.3} target_logit={:.3} loss={:.3}",
        report.final_eval_metrics.accuracy,
        report.final_eval_metrics.mean_target_logit,
        report.final_eval_metrics.mean_loss
    );
    let _ = writeln!(
        output,
        "best_eval:    accuracy={:.3} target_logit={:.3} loss={:.3} source={}",
        report.best_eval_metrics.accuracy,
        report.best_eval_metrics.mean_target_logit,
        report.best_eval_metrics.mean_loss,
        checkpoint_kind_label(report.best_checkpoint_kind)
    );
    if let Some(last_step) = report.train_steps.last() {
        let _ = writeln!(
            output,
            "last_train_step: step={} loss={:.4} seen_samples={}",
            last_step.step, last_step.train_loss, last_step.seen_samples
        );
    }

    let final_train_by_kind =
        mode_eval_summary_by_kind(&report.final_train_probe, report.config.training_mode)
            .unwrap_or_default();
    let final_eval_by_kind =
        mode_eval_summary_by_kind(&report.final_eval_probe, report.config.training_mode)
            .unwrap_or_default();
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Final Train/Eval By Kind ({})",
        mode_label(report.config.training_mode)
    );
    for suite in &report.config.suites {
        let train = final_train_by_kind.get(&suite.kind);
        let eval = final_eval_by_kind.get(&suite.kind);
        let _ = writeln!(
            output,
            "  {:<20} train_acc={:.3} eval_acc={:.3} train_loss={:.3} eval_loss={:.3}",
            suite_label(suite.kind),
            train.map(|metrics| metrics.accuracy).unwrap_or(0.0),
            eval.map(|metrics| metrics.accuracy).unwrap_or(0.0),
            train.map(|metrics| metrics.mean_loss).unwrap_or(0.0),
            eval.map(|metrics| metrics.mean_loss).unwrap_or(0.0),
        );
    }

    render_suite_modes(
        &mut output,
        "Final Eval Probe Modes",
        &report.final_eval_probe,
    );

    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "checkpoint_dir: {}",
        report.checkpoint.directory.display()
    );
    let _ = writeln!(
        output,
        "final_model_path: {}",
        report.checkpoint.final_model_path.display()
    );
    let _ = writeln!(
        output,
        "best_model_path: {}",
        report.checkpoint.best_model_path.display()
    );
    let _ = writeln!(
        output,
        "report_path: {}",
        report.checkpoint.report_path.display()
    );
    output
}

fn render_suite_modes(output: &mut String, title: &str, report: &SyntheticProbeReport) {
    let _ = writeln!(output);
    let _ = writeln!(output, "{title}");
    for suite in &report.suites {
        let _ = writeln!(
            output,
            "  {} (samples={})",
            suite_label(suite.kind),
            suite.sample_count
        );
        for mode_report in &suite.mode_reports {
            let _ = writeln!(
                output,
                "    {:<28} acc={:.3} logit={:.3} loss={:.3}",
                mode_label(mode_report.mode),
                mode_report.metrics.accuracy,
                mode_report.metrics.mean_target_logit,
                mode_report.metrics.mean_loss
            );
        }
    }
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &V2SupervisedSyntheticTrainReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V2ResultsLedgerEntry::supervised_synthetic_train(
        MODEL_LABEL,
        NOTE,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build v2 results ledger entry: {error}"))?;
    append_v2_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append v2 results ledger entry: {error}"))
}

fn checkpoint_kind_label(kind: V2CheckpointKind) -> &'static str {
    match kind {
        V2CheckpointKind::InitialEval => "initial-eval",
        V2CheckpointKind::FinalEval => "final-eval",
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

fn default_output_dir(repo_root: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    repo_root
        .join("artifacts")
        .join("v2-supervised-synthetic")
        .join(format!("{timestamp}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parser_accepts_training_flags() {
        let args = CliArgs::parse(
            [
                "--suite",
                "copy",
                "--steps",
                "32",
                "--eval-holdout-every",
                "2",
                "--learning-rate",
                "0.01",
                "--root-count",
                "1",
                "--training-mode",
                "oracle-tree-plus-oracle-exact-read",
                "--ledger-path",
                "default",
                "--run-label",
                "supervised-check",
                "--output",
                "json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.suite, SuiteSelection::Kind(SyntheticProbeKind::Copy));
        assert_eq!(args.steps, 32);
        assert_eq!(args.eval_holdout_every, 2);
        assert_eq!(args.learning_rate, 0.01);
        assert_eq!(args.root_count, Some(1));
        assert_eq!(
            args.training_mode,
            SyntheticProbeMode::OracleTreePlusOracleExactRead
        );
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("supervised-check"));
        assert_eq!(args.output, OutputFormat::Json);
    }

    #[test]
    fn filter_requested_suites_keeps_only_requested_kind() {
        let suites = filter_requested_suites(
            default_v2_synthetic_probe_suites(),
            SuiteSelection::Kind(SyntheticProbeKind::Induction),
        )
        .unwrap();

        assert_eq!(suites.len(), 1);
        assert_eq!(suites[0].kind, SyntheticProbeKind::Induction);
    }
}
