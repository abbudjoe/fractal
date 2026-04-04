use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::CpuTrainBackend;
use fractal_eval_private::{
    append_v2_results_ledger_entry, default_v2_smoke_corpus_paths,
    resolve_requested_v2_results_ledger_path, run_baseline_v2_smoke_train, V2ResultsLedgerEntry,
    V2SmokeCheckpointKind, V2SmokeTrainConfig,
};
use serde::Serialize;

const SMOKE_MODEL_LABEL: &str = "baseline_v2_byte_level_smoke_cpu_candle";
const SMOKE_NOTE: &str = "small byte-level v2 smoke training run on real local text";

fn main() {
    if let Err(error) = run() {
        eprintln!("v2-smoke-train: {error}");
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
    let mut config = V2SmokeTrainConfig::new(corpus_paths, output_dir);
    config.seq_len = args.seq_len;
    config.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.batch_size = args.batch_size;
    config.train_steps = args.steps;
    config.eval_batches = args.eval_batches;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    if let Some(root_count) = args.root_count {
        config.model = config
            .model
            .with_root_count_preserving_total_budget(root_count);
    }
    if let Some(leaf_size) = args.leaf_size {
        config.model = config.model.with_leaf_size(leaf_size);
    }

    let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
    let result = run_baseline_v2_smoke_train::<CpuTrainBackend>(config, &device)
        .map_err(|error| format!("failed to run v2 smoke training: {error}"))?;
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
    corpus_paths: Vec<PathBuf>,
    output_dir: Option<PathBuf>,
    seq_len: usize,
    window_stride: Option<usize>,
    batch_size: usize,
    steps: usize,
    eval_batches: usize,
    eval_holdout_every: usize,
    learning_rate: f64,
    root_count: Option<usize>,
    leaf_size: Option<usize>,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut corpus_paths = Vec::new();
        let mut output_dir = None;
        let mut seq_len = fractal_eval_private::DEFAULT_V2_SMOKE_SEQ_LEN;
        let mut window_stride = None;
        let mut batch_size = fractal_eval_private::DEFAULT_V2_SMOKE_BATCH_SIZE;
        let mut steps = fractal_eval_private::DEFAULT_V2_SMOKE_TRAIN_STEPS;
        let mut eval_batches = fractal_eval_private::DEFAULT_V2_SMOKE_EVAL_BATCHES;
        let mut eval_holdout_every = fractal_eval_private::DEFAULT_V2_SMOKE_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = fractal_eval_private::DEFAULT_V2_SMOKE_LEARNING_RATE;
        let mut root_count = None;
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
                "--seq-len" => {
                    seq_len = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--seq-len requires a value".to_owned())?,
                        "--seq-len",
                    )?;
                }
                "--window-stride" => {
                    window_stride = Some(parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--window-stride requires a value".to_owned())?,
                        "--window-stride",
                    )?);
                }
                "--batch-size" => {
                    batch_size = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--batch-size requires a value".to_owned())?,
                        "--batch-size",
                    )?;
                }
                "--steps" => {
                    steps = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--steps requires a value".to_owned())?,
                        "--steps",
                    )?;
                }
                "--eval-batches" => {
                    eval_batches = parse_positive_usize(
                        iter.next()
                            .ok_or_else(|| "--eval-batches requires a value".to_owned())?,
                        "--eval-batches",
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
                "--leaf-size" => {
                    leaf_size = Some(parse_positive_usize(
                        iter.next()
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
            seq_len,
            window_stride,
            batch_size,
            steps,
            eval_batches,
            eval_holdout_every,
            learning_rate,
            root_count,
            leaf_size,
            ledger_path,
            run_label,
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

#[derive(Debug, Serialize)]
struct RenderedReport<'a> {
    model: &'static str,
    note: &'static str,
    report: &'a fractal_eval_private::V2SmokeTrainReport,
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &fractal_eval_private::V2SmokeTrainReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V2ResultsLedgerEntry::smoke_train(
        SMOKE_MODEL_LABEL,
        SMOKE_NOTE,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build v2 results ledger entry: {error}"))?;
    append_v2_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append v2 results ledger entry: {error}"))
}

fn render_report(
    report: &fractal_eval_private::V2SmokeTrainReport,
    output: OutputFormat,
) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report)),
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedReport {
            model: SMOKE_MODEL_LABEL,
            note: SMOKE_NOTE,
            report,
        })
        .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &fractal_eval_private::V2SmokeTrainReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "V2 Smoke Training");
    let _ = writeln!(
        output,
        "model: baseline_v2 byte-level smoke (CPU Candle autodiff)"
    );
    let _ = writeln!(
        output,
        "corpus: files={} total_bytes={} train_sequences={} eval_sequences={}",
        report.corpus.files.len(),
        report.corpus.total_bytes,
        report.corpus.train_sequences,
        report.corpus.eval_sequences
    );
    let _ = writeln!(
        output,
        "config: seq_len={} stride={} batch_size={} steps={} lr={} leaf_size={}",
        report.config.seq_len,
        report.config.window_stride,
        report.config.batch_size,
        report.config.train_steps,
        report.config.learning_rate,
        report.config.model.leaf_size
    );
    let _ = writeln!(
        output,
        "model_contract: root_count={} total_root_state_dim={} total_root_readout_dim={} per_root_state_dim={} per_root_readout_dim={}",
        report.config.model.root_count,
        report.config.model.total_root_state_dim,
        report.config.model.total_root_readout_dim,
        report.config.model.root_state_dim(),
        report.config.model.root_readout_dim()
    );
    let _ = writeln!(
        output,
        "initial_eval: loss={:.4} ppl={:.4} batches={}",
        report.initial_eval.mean_loss,
        report.initial_eval.perplexity,
        report.initial_eval.batch_count
    );
    let _ = writeln!(
        output,
        "final_eval:   loss={:.4} ppl={:.4} batches={}",
        report.final_eval.mean_loss, report.final_eval.perplexity, report.final_eval.batch_count
    );
    let _ = writeln!(
        output,
        "best_eval:    loss={:.4} ppl={:.4} source={}",
        report.best_eval.mean_loss,
        report.best_eval.perplexity,
        best_checkpoint_kind_label(report.best_checkpoint_kind)
    );
    if let Some(last_step) = report.train_steps.last() {
        let _ = writeln!(
            output,
            "last_train_step: step={} loss={:.4} ppl={:.4} seen_tokens={}",
            last_step.step, last_step.train_loss, last_step.train_perplexity, last_step.seen_tokens
        );
    }
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
        "final_optimizer_path: {}",
        report.checkpoint.final_optimizer_path.display()
    );
    let _ = writeln!(
        output,
        "best_optimizer_path: {}",
        report.checkpoint.best_optimizer_path.display()
    );
    let _ = writeln!(
        output,
        "report_path: {}",
        report.checkpoint.report_path.display()
    );
    output
}

fn best_checkpoint_kind_label(kind: V2SmokeCheckpointKind) -> &'static str {
    match kind {
        V2SmokeCheckpointKind::InitialEval => "initial-eval",
        V2SmokeCheckpointKind::FinalEval => "final-eval",
    }
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin v2-smoke-train -- [--corpus-path <path>]... [--output-dir <path>] [--seq-len <usize>] [--window-stride <usize>] [--batch-size <usize>] [--steps <usize>] [--eval-batches <usize>] [--eval-holdout-every <usize>] [--learning-rate <float>] [--root-count <usize>] [--leaf-size <usize>] [--ledger-path <default|path>] [--run-label <label>] [--output <table|json>]"
    );
    let _ = writeln!(
        output,
        "Defaults: corpus=repo markdown specs/policies, output-dir=artifacts/v2-smoke-train/<timestamp>, seq-len={}, batch-size={}, steps={}, output=table",
        fractal_eval_private::DEFAULT_V2_SMOKE_SEQ_LEN,
        fractal_eval_private::DEFAULT_V2_SMOKE_BATCH_SIZE,
        fractal_eval_private::DEFAULT_V2_SMOKE_TRAIN_STEPS
    );
    output
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

fn default_output_dir(repo_root: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    repo_root
        .join("artifacts")
        .join("v2-smoke-train")
        .join(format!("{timestamp}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parser_accepts_training_flags() {
        let args = CliArgs::parse(
            [
                "--seq-len",
                "32",
                "--batch-size",
                "2",
                "--steps",
                "4",
                "--root-count",
                "1",
                "--output",
                "json",
                "--leaf-size",
                "32",
                "--ledger-path",
                "default",
                "--run-label",
                "smoke-baseline",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();

        assert_eq!(args.seq_len, 32);
        assert_eq!(args.batch_size, 2);
        assert_eq!(args.steps, 4);
        assert_eq!(args.root_count, Some(1));
        assert_eq!(args.leaf_size, Some(32));
        assert_eq!(args.ledger_path.as_deref(), Some("default"));
        assert_eq!(args.run_label.as_deref(), Some("smoke-baseline"));
        assert_eq!(args.output, OutputFormat::Json);
    }
}
