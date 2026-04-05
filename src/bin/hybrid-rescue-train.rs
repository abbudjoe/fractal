use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::tensor::backend::AutodiffBackend;
use fractal_core::CpuTrainBackend;
use fractal_eval_private::{
    append_hybrid_results_ledger_entry, default_hybrid_rescue_prevalidation_suites,
    resolve_requested_hybrid_results_ledger_path, run_baseline_hybrid_rescue_frozen_train,
    HybridRescueFrozenEvalModeSet, HybridRescueFrozenTrainConfig, HybridRescueFrozenTrainReport,
    HybridRescueProbeMode, HybridRescueSuiteKind, HybridResultsLedgerEntry,
};
use serde::Serialize;

const NOTE: &str = "phase-1 frozen-backbone rescue training over fixed hybrid probe suites";

fn main() {
    if let Err(error) = run() {
        eprintln!("hybrid-rescue-train: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let suites = filter_suites(
        default_hybrid_rescue_prevalidation_suites()
            .map_err(|error| format!("failed to build default hybrid rescue suites: {error}"))?,
        args.suite,
    )?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let mut config = HybridRescueFrozenTrainConfig::new(output_dir, suites);
    config.train_steps = args.steps;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    config.training_mode = args.training_mode;
    config.eval_mode_set = args.eval_mode_set;
    config.include_train_probe_reports = args.include_train_probes;

    let model_label = args.backend.model_label();
    let report = match args.backend {
        BackendSelection::Cpu => run_with_backend::<CpuTrainBackend>(config)?,
        BackendSelection::Metal => {
            return Err(
                "metal backend is not yet supported for hybrid rescue training: the shared v2 state path still allocates zero-sized tensors that WGPU/Metal rejects. Use --backend cpu until the state/control plane is de-zeroed."
                    .to_string(),
            )
        }
    };
    let rendered = render_report(&report, args.output, model_label)?;
    maybe_append_ledger_entry(
        resolve_requested_hybrid_results_ledger_path(&repo_root, args.ledger_path.as_deref())
            .map_err(|error| format!("failed to resolve hybrid results ledger path: {error}"))?,
        &report,
        model_label,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    backend: BackendSelection,
    suite: SuiteSelection,
    output_dir: Option<PathBuf>,
    steps: usize,
    eval_holdout_every: usize,
    learning_rate: f64,
    training_mode: HybridRescueProbeMode,
    eval_mode_set: HybridRescueFrozenEvalModeSet,
    include_train_probes: bool,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut backend = BackendSelection::Cpu;
        let mut suite = SuiteSelection::All;
        let mut output_dir = None;
        let mut steps = fractal_eval_private::DEFAULT_HYBRID_RESCUE_FROZEN_STEPS;
        let mut eval_holdout_every =
            fractal_eval_private::DEFAULT_HYBRID_RESCUE_FROZEN_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = fractal_eval_private::DEFAULT_HYBRID_RESCUE_FROZEN_LEARNING_RATE;
        let mut training_mode = HybridRescueProbeMode::OracleRemoteWithOracleExactTokenSubset;
        let mut eval_mode_set = HybridRescueFrozenEvalModeSet::TrainingVsLocal;
        let mut include_train_probes = false;
        let mut ledger_path = None;
        let mut run_label = None;
        let mut output = OutputFormat::Table;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--backend" => {
                    backend = BackendSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--backend requires a value".to_owned())?,
                    )?;
                }
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
                "--training-mode" => {
                    training_mode = parse_training_mode(
                        &iter
                            .next()
                            .ok_or_else(|| "--training-mode requires a value".to_owned())?,
                    )?;
                }
                "--eval-mode-set" => {
                    eval_mode_set = parse_eval_mode_set(
                        &iter
                            .next()
                            .ok_or_else(|| "--eval-mode-set requires a value".to_owned())?,
                    )?;
                }
                "--include-train-probes" => include_train_probes = true,
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
            backend,
            suite,
            output_dir,
            steps,
            eval_holdout_every,
            learning_rate,
            training_mode,
            eval_mode_set,
            include_train_probes,
            ledger_path,
            run_label,
            output,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendSelection {
    Cpu,
    Metal,
}

impl BackendSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            _ => Err(format!("unknown backend selection: {value}")),
        }
    }

    fn model_label(self) -> &'static str {
        match self {
            Self::Cpu => "hybrid_rescue_frozen_backbone_cpu_candle",
            Self::Metal => "hybrid_rescue_frozen_backbone_metal_wgpu",
        }
    }
}

#[derive(Debug, Serialize)]
struct RenderedReport<'a> {
    model: &'static str,
    note: &'static str,
    report: &'a HybridRescueFrozenTrainReport,
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin hybrid-rescue-train -- [--backend <cpu|metal>] [--suite <all|mqar|copy|induction|retrieval-heavy>] [--output-dir <path>] [--steps <usize>] [--eval-holdout-every <usize>] [--learning-rate <float>] [--training-mode <routed-remote|oracle-remote|oracle-remote-with-oracle-exact-token-subset>] [--eval-mode-set <training-only|training-vs-local|initial-four>] [--include-train-probes] [--ledger-path <default|path>] [--run-label <label>] [--output <table|json>]"
    );
    let _ = writeln!(
        output,
        "Defaults: backend=cpu suite=all output-dir=artifacts/hybrid-rescue-train/<timestamp> steps={} eval-holdout-every={} training-mode=oracle-remote-with-oracle-exact-token-subset eval-mode-set=training-vs-local output=table",
        fractal_eval_private::DEFAULT_HYBRID_RESCUE_FROZEN_STEPS,
        fractal_eval_private::DEFAULT_HYBRID_RESCUE_FROZEN_EVAL_HOLDOUT_EVERY,
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

fn parse_training_mode(value: &str) -> Result<HybridRescueProbeMode, String> {
    match value {
        "routed-remote" | "routed_remote" => Ok(HybridRescueProbeMode::RoutedRemote),
        "oracle-remote" | "oracle_remote" => Ok(HybridRescueProbeMode::OracleRemote),
        "oracle-remote-with-oracle-exact-token-subset"
        | "oracle_remote_with_oracle_exact_token_subset" => {
            Ok(HybridRescueProbeMode::OracleRemoteWithOracleExactTokenSubset)
        }
        "local-only" | "local_only" => Err(
            "local-only is not a valid frozen rescue training mode; use a remote retrieval mode"
                .to_string(),
        ),
        _ => Err(format!("unknown training mode: {value}")),
    }
}

fn parse_eval_mode_set(value: &str) -> Result<HybridRescueFrozenEvalModeSet, String> {
    match value {
        "training-only" | "training_only" => Ok(HybridRescueFrozenEvalModeSet::TrainingOnly),
        "training-vs-local" | "training_vs_local" => {
            Ok(HybridRescueFrozenEvalModeSet::TrainingVsLocal)
        }
        "initial-four" | "initial_four" => Ok(HybridRescueFrozenEvalModeSet::InitialFour),
        _ => Err(format!("unknown eval mode set: {value}")),
    }
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
    report: &HybridRescueFrozenTrainReport,
    output: OutputFormat,
    model_label: &'static str,
) -> Result<String, String> {
    match output {
        OutputFormat::Table => Ok(render_table(report, model_label)),
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedReport {
            model: model_label,
            note: NOTE,
            report,
        })
        .map_err(|error| format!("failed to serialize json report: {error}")),
    }
}

fn render_table(report: &HybridRescueFrozenTrainReport, model_label: &'static str) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Hybrid Rescue Frozen Training");
    let _ = writeln!(output, "model: {model_label}");
    let _ = writeln!(output, "note: {NOTE}");
    let _ = writeln!(
        output,
        "config: backend={} suites={} steps={} eval_holdout_every={} lr={} training_mode={} eval_mode_set={} include_train_probes={} leaf_size={} routed_spans=8 local_window=256 total_budget=384",
        backend_label(model_label),
        report.config.suites.len(),
        report.config.train_steps,
        report.config.eval_holdout_every,
        report.config.learning_rate,
        mode_label(report.config.training_mode),
        eval_mode_set_label(report.config.eval_mode_set),
        report.config.include_train_probe_reports,
        report.config.model.backbone.leaf_size,
    );
    let _ = writeln!(
        output,
        "split: train_samples={} eval_samples={}",
        report.split.train_sample_count, report.split.eval_sample_count
    );
    let _ = writeln!(
        output,
        "initial_eval: accuracy={:.3} rank={:.1} target_logit={:.3} loss={:.3} span_recall={:.3} token_recall={:.3}",
        report.initial_eval_metrics.accuracy,
        report.initial_eval_metrics.mean_target_rank,
        report.initial_eval_metrics.mean_target_logit,
        report.initial_eval_metrics.mean_loss,
        report.initial_eval_metrics.evidence_span_recall_rate,
        report.initial_eval_metrics.mean_evidence_token_recall,
    );
    let _ = writeln!(
        output,
        "final_eval:   accuracy={:.3} rank={:.1} target_logit={:.3} loss={:.3} span_recall={:.3} token_recall={:.3}",
        report.final_eval_metrics.accuracy,
        report.final_eval_metrics.mean_target_rank,
        report.final_eval_metrics.mean_target_logit,
        report.final_eval_metrics.mean_loss,
        report.final_eval_metrics.evidence_span_recall_rate,
        report.final_eval_metrics.mean_evidence_token_recall,
    );
    let _ = writeln!(
        output,
        "best_eval:    accuracy={:.3} rank={:.1} target_logit={:.3} loss={:.3} source={}",
        report.best_eval_metrics.accuracy,
        report.best_eval_metrics.mean_target_rank,
        report.best_eval_metrics.mean_target_logit,
        report.best_eval_metrics.mean_loss,
        checkpoint_kind_label(report.best_checkpoint_kind),
    );
    if let Some(last_step) = report.train_steps.last() {
        let _ = writeln!(
            output,
            "last_train_step: step={} loss={:.4} seen_samples={}",
            last_step.step, last_step.train_loss, last_step.seen_samples
        );
    }

    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Eval By Suite ({})",
        mode_label(report.config.training_mode)
    );
    for suite in &report.final_eval_probe.suites {
        let initial_mode = suite_mode_metrics(
            &report.initial_eval_probe,
            suite.kind,
            report.config.training_mode,
        );
        let final_mode = suite_mode_metrics(
            &report.final_eval_probe,
            suite.kind,
            report.config.training_mode,
        );
        let _ = writeln!(
            output,
            "  {:<16} init_acc={:.3} final_acc={:.3} init_rank={:.1} final_rank={:.1} init_loss={:.3} final_loss={:.3}",
            suite_label(suite.kind),
            initial_mode.map(|metrics| metrics.accuracy).unwrap_or(0.0),
            final_mode.map(|metrics| metrics.accuracy).unwrap_or(0.0),
            initial_mode.map(|metrics| metrics.mean_target_rank).unwrap_or(0.0),
            final_mode.map(|metrics| metrics.mean_target_rank).unwrap_or(0.0),
            initial_mode.map(|metrics| metrics.mean_loss).unwrap_or(0.0),
            final_mode.map(|metrics| metrics.mean_loss).unwrap_or(0.0),
        );
    }

    if let Some(report) = &report.initial_train_probe {
        render_modes(&mut output, "Initial Train Modes", report);
    }
    render_modes(
        &mut output,
        "Initial Eval Modes",
        &report.initial_eval_probe,
    );
    if let Some(report) = &report.final_train_probe {
        render_modes(&mut output, "Final Train Modes", report);
    }
    render_modes(&mut output, "Final Eval Modes", &report.final_eval_probe);

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

fn render_modes(
    output: &mut String,
    title: &str,
    report: &fractal_eval_private::HybridRescuePrevalidationReport,
) {
    let _ = writeln!(output);
    let _ = writeln!(output, "{title}");
    for suite in &report.suites {
        let _ = writeln!(output, "  {}", suite_label(suite.kind));
        for mode in HybridRescueProbeMode::INITIAL_FOUR {
            if let Some(metrics) = suite_mode_metrics(report, suite.kind, mode) {
                let _ = writeln!(
                    output,
                    "    {:<36} acc={:.3} rank={:.1} loss={:.3} remote_mass={:.3} span_recall={:.3} token_recall={:.3}",
                    mode_label(mode),
                    metrics.accuracy,
                    metrics.mean_target_rank,
                    metrics.mean_loss,
                    metrics.mean_remote_attention_mass,
                    metrics.evidence_span_recall_rate,
                    metrics.mean_evidence_token_recall,
                );
            }
        }
    }
}

fn suite_mode_metrics(
    report: &fractal_eval_private::HybridRescuePrevalidationReport,
    kind: HybridRescueSuiteKind,
    mode: HybridRescueProbeMode,
) -> Option<fractal_eval_private::HybridRescueMetrics> {
    report
        .suites
        .iter()
        .find(|suite| suite.kind == kind)
        .and_then(|suite| {
            suite
                .mode_reports
                .iter()
                .find(|candidate| candidate.mode == mode)
        })
        .map(|report| report.metrics)
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &HybridRescueFrozenTrainReport,
    model_label: &'static str,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = HybridResultsLedgerEntry::rescue_frozen_train(
        model_label,
        NOTE,
        report,
        run_label.map(ToOwned::to_owned),
    )
    .map_err(|error| format!("failed to build hybrid frozen-train ledger entry: {error}"))?;
    append_hybrid_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append hybrid frozen-train ledger entry: {error}"))
}

fn backend_label(model_label: &str) -> &'static str {
    if model_label.contains("metal") {
        "metal"
    } else {
        "cpu"
    }
}

fn default_output_dir(repo_root: &Path) -> PathBuf {
    repo_root
        .join("artifacts")
        .join("hybrid-rescue-train")
        .join(timestamp())
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
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

fn eval_mode_set_label(mode_set: HybridRescueFrozenEvalModeSet) -> &'static str {
    match mode_set {
        HybridRescueFrozenEvalModeSet::TrainingOnly => "training-only",
        HybridRescueFrozenEvalModeSet::TrainingVsLocal => "training-vs-local",
        HybridRescueFrozenEvalModeSet::InitialFour => "initial-four",
    }
}

fn checkpoint_kind_label(kind: fractal_eval_private::V2CheckpointKind) -> &'static str {
    match kind {
        fractal_eval_private::V2CheckpointKind::InitialEval => "initial-eval",
        fractal_eval_private::V2CheckpointKind::FinalEval => "final-eval",
    }
}

fn run_with_backend<B: AutodiffBackend>(
    config: HybridRescueFrozenTrainConfig,
) -> Result<HybridRescueFrozenTrainReport, String> {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    run_baseline_hybrid_rescue_frozen_train::<B>(config, &device)
        .map(|result| result.report)
        .map_err(|error| format!("failed to run frozen hybrid rescue training: {error}"))
}
