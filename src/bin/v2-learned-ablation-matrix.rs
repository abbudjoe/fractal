use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::backend::Candle;
use fractal_core::CpuTrainBackend;
use fractal_eval_private::{
    append_v2_results_ledger_entry, default_v2_smoke_corpus_paths, filter_synthetic_probe_suites,
    resolve_requested_v2_results_ledger_path, run_required_v2_learned_ablation_matrix,
    v2_synthetic_probe_suites_for_leaf_size, SyntheticProbeKind, V2CheckpointSelection,
    V2LearnedAblationConfig, V2LearnedAblationReport, V2ResultsLedgerEntry, V2RootTopology,
    V2SmokeTrainConfig,
};
use serde::Serialize;

type EvalBackend = Candle<f32, i64>;

const MODEL_LABEL: &str = "baseline_v2_learned_ablation_cpu_candle";
const NOTE: &str =
    "train single-root and multi-root checkpoints, then fan out the required learned ablation matrix";

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-learned-ablation-matrix: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_paths = if args.corpus_paths.is_empty() {
        default_v2_smoke_corpus_paths(&repo_root)
            .map_err(|error| format!("failed to resolve default v2 smoke corpus: {error}"))?
    } else {
        args.corpus_paths.clone()
    };
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let mut smoke = V2SmokeTrainConfig::new(corpus_paths, output_dir);
    smoke.seq_len = args.seq_len;
    smoke.window_stride = args.window_stride.unwrap_or(args.seq_len);
    smoke.batch_size = args.batch_size;
    smoke.train_steps = args.steps;
    smoke.eval_batches = args.eval_batches;
    smoke.eval_holdout_every = args.eval_holdout_every;
    smoke.learning_rate = args.learning_rate;
    smoke.model = smoke
        .model
        .with_root_count_preserving_total_budget(args.multi_root_count);
    if let Some(leaf_size) = args.leaf_size {
        smoke.model = smoke.model.with_leaf_size(leaf_size);
    }
    let suites = filter_requested_suites(
        v2_synthetic_probe_suites_for_leaf_size(smoke.model.leaf_size)
            .map_err(|error| format!("failed to build synthetic probe suites: {error}"))?,
        args.suite,
    )?;

    let config = V2LearnedAblationConfig {
        smoke,
        checkpoint_selection: V2CheckpointSelection::Final,
        suites,
    };
    let train_device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
    let eval_device = <EvalBackend as burn::tensor::backend::Backend>::Device::default();
    let report = run_required_v2_learned_ablation_matrix::<CpuTrainBackend, EvalBackend>(
        config,
        &train_device,
        &eval_device,
    )
    .map_err(|error| format!("failed to run learned ablation matrix: {error}"))?;
    let rendered = render_report(&report, args.output)?;
    maybe_append_ledger_entry(
        resolve_requested_v2_results_ledger_path(&repo_root, args.ledger_path.as_deref())
            .map_err(|error| format!("failed to resolve v2 results ledger path: {error}"))?,
        &report,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    corpus_paths: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    suite: SuiteSelection,
    seq_len: usize,
    window_stride: Option<usize>,
    batch_size: usize,
    steps: usize,
    eval_batches: usize,
    eval_holdout_every: usize,
    learning_rate: f64,
    multi_root_count: usize,
    leaf_size: Option<usize>,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut corpus_paths = Vec::new();
        let mut output_dir = None;
        let mut suite = SuiteSelection::All;
        let mut seq_len = fractal_eval_private::DEFAULT_V2_SMOKE_SEQ_LEN;
        let mut window_stride = None;
        let mut batch_size = fractal_eval_private::DEFAULT_V2_SMOKE_BATCH_SIZE;
        let mut steps = fractal_eval_private::DEFAULT_V2_SMOKE_TRAIN_STEPS;
        let mut eval_batches = fractal_eval_private::DEFAULT_V2_SMOKE_EVAL_BATCHES;
        let mut eval_holdout_every = fractal_eval_private::DEFAULT_V2_SMOKE_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = fractal_eval_private::DEFAULT_V2_SMOKE_LEARNING_RATE;
        let mut multi_root_count = 2usize;
        let mut leaf_size = None;
        let mut ledger_path = None;
        let mut run_label = None;
        let mut output = OutputFormat::Table;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--corpus-path" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--corpus-path requires a value".to_owned())?;
                    corpus_paths.push(PathBuf::from(value));
                }
                "--output-dir" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--output-dir requires a value".to_owned())?;
                    output_dir = Some(PathBuf::from(value));
                }
                "--suite" => {
                    suite = SuiteSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--suite requires a value".to_owned())?,
                    )?;
                }
                "--seq-len" => {
                    seq_len = parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--seq-len requires a value".to_owned())?,
                        "--seq-len",
                    )?;
                }
                "--window-stride" => {
                    window_stride = Some(parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--window-stride requires a value".to_owned())?,
                        "--window-stride",
                    )?);
                }
                "--batch-size" => {
                    batch_size = parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--batch-size requires a value".to_owned())?,
                        "--batch-size",
                    )?;
                }
                "--steps" => {
                    steps = parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--steps requires a value".to_owned())?,
                        "--steps",
                    )?;
                }
                "--eval-batches" => {
                    eval_batches = parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--eval-batches requires a value".to_owned())?,
                        "--eval-batches",
                    )?;
                }
                "--eval-holdout-every" => {
                    eval_holdout_every = parse_positive_usize(
                        &iter
                            .next()
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
                "--multi-root-count" => {
                    multi_root_count = parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--multi-root-count requires a value".to_owned())?,
                        "--multi-root-count",
                    )?;
                }
                "--leaf-size" => {
                    leaf_size = Some(parse_positive_usize(
                        &iter
                            .next()
                            .ok_or_else(|| "--leaf-size requires a value".to_owned())?,
                        "--leaf-size",
                    )?);
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
            corpus_paths,
            output_dir,
            suite,
            seq_len,
            window_stride,
            batch_size,
            steps,
            eval_batches,
            eval_holdout_every,
            learning_rate,
            multi_root_count,
            leaf_size,
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
    report: &'a V2LearnedAblationReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-learned-ablation-matrix -- [--corpus-path <path>]... [--output-dir <path>] [--suite <all|copy|associative-recall|induction|noisy-retrieval|far-token-comparison>] [--seq-len <n>] [--window-stride <n>] [--batch-size <n>] [--steps <n>] [--eval-batches <n>] [--eval-holdout-every <n>] [--learning-rate <f>] [--multi-root-count <n>] [--leaf-size <n>] [--ledger-path <default|path>] [--run-label <label>] [--output <table|json>]"
    );
    let _ = writeln!(
        output,
        "Defaults: suite=all steps={} eval_batches={} multi_root_count=2 output=table",
        fractal_eval_private::DEFAULT_V2_SMOKE_TRAIN_STEPS,
        fractal_eval_private::DEFAULT_V2_SMOKE_EVAL_BATCHES,
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

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn render_report(report: &V2LearnedAblationReport, output: OutputFormat) -> Result<String, String> {
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

fn render_table(report: &V2LearnedAblationReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Learned Ablation Matrix");
    let _ = writeln!(output, "model: {MODEL_LABEL}");
    let _ = writeln!(output, "note: {}", report.note);

    for topology_run in &report.topology_runs {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "{} roots={} total_state={} total_readout={} best_eval_loss={:.4} evaluated_eval_loss={:.4} evaluated_checkpoint={}",
            topology_label(topology_run.topology),
            topology_run.model_config.root_count,
            topology_run.model_config.total_root_state_dim,
            topology_run.model_config.total_root_readout_dim,
            topology_run.smoke.best_eval.mean_loss,
            topology_run.evaluated_eval.mean_loss,
            checkpoint_kind_label(topology_run.evaluated_checkpoint_kind),
        );
    }

    let _ = writeln!(output);
    let _ = writeln!(output, "Ordered Checklist Steps");
    for step in &report.ordered_steps {
        let _ = writeln!(
            output,
            "{}. {} [{} / {}]",
            step.step.step_number(),
            step.step.label(),
            topology_label(step.topology),
            mode_label(step.mode),
        );
        for suite in &step.suite_reports {
            let _ = writeln!(
                output,
                "  {:<20} acc={:.3} logit={:.3} loss={:.3}",
                suite_label(suite.kind),
                suite.metrics.accuracy,
                suite.metrics.mean_target_logit,
                suite.metrics.mean_loss
            );
        }
    }

    output
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &V2LearnedAblationReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V2ResultsLedgerEntry::learned_ablation_matrix(
        MODEL_LABEL,
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
        fractal_eval_private::SyntheticProbeMode::OracleTreePlusOracleExactRead => {
            "OracleTreePlusOracleExactRead"
        }
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

fn checkpoint_kind_label(kind: fractal_eval_private::V2CheckpointKind) -> &'static str {
    match kind {
        fractal_eval_private::V2CheckpointKind::InitialEval => "initial-eval",
        fractal_eval_private::V2CheckpointKind::FinalEval => "final-eval",
    }
}

fn default_output_dir(repo_root: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    repo_root
        .join("artifacts")
        .join("v2-learned-ablation")
        .join(format!("{timestamp}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_matrix_training_flags() {
        let args = CliArgs::parse(
            [
                "--suite",
                "copy",
                "--steps",
                "32",
                "--eval-batches",
                "4",
                "--multi-root-count",
                "4",
                "--leaf-size",
                "32",
                "--ledger-path",
                "default",
                "--run-label",
                "learned-matrix",
                "--output",
                "json",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.suite, SuiteSelection::Kind(SyntheticProbeKind::Copy));
        assert_eq!(args.steps, 32);
        assert_eq!(args.eval_batches, 4);
        assert_eq!(args.multi_root_count, 4);
        assert_eq!(args.leaf_size, Some(32));
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("learned-matrix"));
        assert_eq!(args.output, OutputFormat::Json);
    }
}
