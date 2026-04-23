use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::{
    initialize_metal_runtime, CpuTrainBackend, MetalTrainBackend, MiniMoeBackendKind,
    MiniMoeSurfaceSpec,
};
use fractal_eval_private::{
    default_v3a_fineweb_stage0_canary_corpus_source, run_mini_moe_smoke_train,
    ByteLevelSmokeCorpusSource, MiniMoeSmokeTrainConfig, MiniMoeSmokeTrainReport,
    DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES, DEFAULT_V3A_SMOKE_LEARNING_RATE,
    DEFAULT_V3A_SMOKE_SEED, DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS,
    DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};
use serde::Serialize;

const EXPERIMENT_TITLE: &str = "DREEGMOR Mini-MoE Experiment";
const EXPERIMENT_SLUG: &str = "dreegmor-mini-moe-experiment";

fn main() {
    if let Err(error) = run() {
        eprintln!("{EXPERIMENT_SLUG}: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_source = default_v3a_fineweb_stage0_canary_corpus_source(&repo_root)
        .map_err(|error| format!("failed to resolve default mini-moe corpus: {error}"))?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let summaries = run_selected_variants(&args, corpus_source, &output_dir)?;
    print!("{}", render_report(&summaries, args.output)?);
    Ok(())
}

fn run_selected_variants(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
) -> Result<Vec<ExperimentVariantSummary>, String> {
    match args.backend {
        BackendSelection::Cpu => {
            let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
            run_selected_variants_with_backend::<CpuTrainBackend>(
                args,
                corpus_source,
                output_dir,
                &device,
            )
        }
        BackendSelection::Metal => {
            let device = <MetalTrainBackend as burn::tensor::backend::Backend>::Device::default();
            initialize_metal_runtime(&device);
            run_selected_variants_with_backend::<MetalTrainBackend>(
                args,
                corpus_source,
                output_dir,
                &device,
            )
        }
    }
}

fn run_selected_variants_with_backend<B>(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Result<Vec<ExperimentVariantSummary>, String>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    let mut variants = Vec::new();

    if args.variant.includes_reference() {
        let report = run_mini_moe_smoke_train::<B>(
            mini_moe_config(
                corpus_source.clone(),
                isolated_variant_dir(output_dir, "reference"),
                MiniMoeSurfaceSpec::phase1_reference_default(),
                args,
            )?,
            device,
        )
        .map_err(|error| format!("failed to run reference mini-moe: {error}"))?;
        variants.push(ExperimentVariantSummary::from_report(
            "Mini-MoE Reference",
            report,
        ));
    }

    if args.variant.includes_recurrent() {
        let report = run_mini_moe_smoke_train::<B>(
            mini_moe_config(
                corpus_source,
                isolated_variant_dir(output_dir, "recurrent"),
                MiniMoeSurfaceSpec::phase1_recurrent_default(),
                args,
            )?,
            device,
        )
        .map_err(|error| format!("failed to run recurrent mini-moe: {error}"))?;
        variants.push(ExperimentVariantSummary::from_report(
            "Mini-MoE Recurrent",
            report,
        ));
    }

    Ok(variants)
}

fn mini_moe_config(
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: PathBuf,
    surface: MiniMoeSurfaceSpec,
    args: &CliArgs,
) -> Result<MiniMoeSmokeTrainConfig, String> {
    let mut config = MiniMoeSmokeTrainConfig::new(corpus_source, output_dir, surface)
        .map_err(|error| format!("failed to build mini-moe config: {error}"))?;
    config.manifest.data.seq_len = args.seq_len;
    config.manifest.data.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.manifest.train.steps = args.steps;
    config.manifest.train.batch_size = args.batch_size;
    config.manifest.train.learning_rate = args.learning_rate;
    config.manifest.train.model_seed = args.seed;
    config.manifest.train.data_seed = args.data_seed;
    config.manifest.eval.eval_batches = args.eval_batches;
    config.manifest.eval.full_eval_pass = args.full_eval_pass;
    config.manifest.backend.backend = args.backend.backend_kind();
    if args.variant.is_single() {
        config.manifest.benchmark_policy = fractal_core::BenchmarkPolicy::Benchmark;
        config.manifest.isolation_mode = fractal_core::ExecutionIsolationMode::IsolatedProcess;
    }
    Ok(config)
}

fn default_output_dir(repo_root: &Path) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    repo_root
        .join("artifacts")
        .join(EXPERIMENT_SLUG)
        .join(timestamp.to_string())
}

fn isolated_variant_dir(root: &Path, label: &str) -> PathBuf {
    root.join(label)
}

fn render_report(
    summaries: &[ExperimentVariantSummary],
    format: OutputFormat,
) -> Result<String, String> {
    match format {
        OutputFormat::Table => render_table(summaries),
        OutputFormat::Json => serde_json::to_string_pretty(summaries)
            .map_err(|error| format!("failed to serialize experiment summary: {error}")),
    }
}

fn render_table(summaries: &[ExperimentVariantSummary]) -> Result<String, String> {
    let mut out = String::new();
    writeln!(&mut out, "{EXPERIMENT_TITLE}").map_err(|error| error.to_string())?;
    writeln!(&mut out, "variants: {}", summaries.len()).map_err(|error| error.to_string())?;
    for summary in summaries {
        writeln!(&mut out).map_err(|error| error.to_string())?;
        writeln!(&mut out, "{} [{}]", summary.display_name, summary.backend)
            .map_err(|error| error.to_string())?;
        writeln!(
            &mut out,
            "  loss: {:.4} -> {:.4}",
            summary.initial_loss, summary.final_loss
        )
        .map_err(|error| error.to_string())?;
        writeln!(
            &mut out,
            "  throughput: train {:.2} tok/s | overall {:.2} tok/s",
            summary.train_tokens_per_second.unwrap_or(0.0),
            summary.overall_tokens_per_second.unwrap_or(0.0),
        )
        .map_err(|error| error.to_string())?;
        writeln!(
            &mut out,
            "  memory: {} {:.2} MB",
            summary
                .process_memory_metric
                .as_deref()
                .unwrap_or("unknown_metric"),
            summary.peak_process_memory_mb.unwrap_or(0.0),
        )
        .map_err(|error| error.to_string())?;
        writeln!(
            &mut out,
            "  routing: sampled_tokens={} active_experts={} round_count={}",
            summary.sampled_tokens, summary.active_expert_count, summary.round_count,
        )
        .map_err(|error| error.to_string())?;
        writeln!(&mut out, "  report: {}", summary.report_path.display())
            .map_err(|error| error.to_string())?;
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ExperimentVariantSummary {
    display_name: String,
    backend: String,
    model_label: String,
    initial_loss: f64,
    final_loss: f64,
    perplexity: f64,
    train_tokens_per_second: Option<f64>,
    overall_tokens_per_second: Option<f64>,
    process_memory_metric: Option<String>,
    peak_process_memory_mb: Option<f64>,
    sampled_tokens: usize,
    active_expert_count: usize,
    round_count: usize,
    report_path: PathBuf,
}

impl ExperimentVariantSummary {
    fn from_report(display_name: &str, report: MiniMoeSmokeTrainReport) -> Self {
        Self {
            display_name: display_name.to_string(),
            backend: backend_label(report.config.manifest.backend.backend).to_string(),
            model_label: report.model_label,
            initial_loss: report.initial_eval.mean_loss,
            final_loss: report.final_eval.mean_loss,
            perplexity: report.final_eval.perplexity,
            train_tokens_per_second: report.artifact.system_metrics.train_tokens_per_second,
            overall_tokens_per_second: report.artifact.system_metrics.overall_tokens_per_second,
            process_memory_metric: report.artifact.system_metrics.process_memory_metric,
            peak_process_memory_mb: report.artifact.system_metrics.peak_process_memory_mb,
            sampled_tokens: report.artifact.summary.routing.sampled_tokens,
            active_expert_count: report.artifact.summary.routing.active_expert_count,
            round_count: report.artifact.summary.routing.round_count,
            report_path: report.report_path,
        }
    }
}

fn backend_label(backend: MiniMoeBackendKind) -> &'static str {
    match backend {
        MiniMoeBackendKind::Cpu => "cpu",
        MiniMoeBackendKind::Metal => "metal",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Table,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendSelection {
    Cpu,
    Metal,
}

impl BackendSelection {
    fn backend_kind(self) -> MiniMoeBackendKind {
        match self {
            Self::Cpu => MiniMoeBackendKind::Cpu,
            Self::Metal => MiniMoeBackendKind::Metal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariantSelection {
    All,
    Reference,
    Recurrent,
}

impl VariantSelection {
    fn includes_reference(self) -> bool {
        matches!(self, Self::All | Self::Reference)
    }

    fn includes_recurrent(self) -> bool {
        matches!(self, Self::All | Self::Recurrent)
    }

    fn is_single(self) -> bool {
        !matches!(self, Self::All)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    variant: VariantSelection,
    backend: BackendSelection,
    steps: usize,
    eval_batches: usize,
    seq_len: usize,
    window_stride: Option<usize>,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
    data_seed: Option<u64>,
    full_eval_pass: bool,
    output: OutputFormat,
    output_dir: Option<PathBuf>,
}

impl CliArgs {
    fn parse<I>(args: I) -> Result<Self, String>
    where
        I: IntoIterator<Item = String>,
    {
        let mut cli = Self {
            variant: VariantSelection::All,
            backend: BackendSelection::Cpu,
            steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE),
            batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
            learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: DEFAULT_V3A_SMOKE_SEED,
            data_seed: None,
            full_eval_pass: false,
            output: OutputFormat::Table,
            output_dir: None,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--variant" => {
                    cli.variant = parse_variant(&next_arg(&mut args, "--variant")?)?;
                }
                "--backend" => {
                    cli.backend = parse_backend(&next_arg(&mut args, "--backend")?)?;
                }
                "--steps" => {
                    cli.steps = parse_usize(&next_arg(&mut args, "--steps")?, "--steps")?;
                }
                "--eval-batches" => {
                    cli.eval_batches =
                        parse_usize(&next_arg(&mut args, "--eval-batches")?, "--eval-batches")?;
                }
                "--seq-len" => {
                    cli.seq_len = parse_usize(&next_arg(&mut args, "--seq-len")?, "--seq-len")?;
                }
                "--window-stride" => {
                    cli.window_stride = Some(parse_usize(
                        &next_arg(&mut args, "--window-stride")?,
                        "--window-stride",
                    )?);
                }
                "--batch-size" => {
                    cli.batch_size =
                        parse_usize(&next_arg(&mut args, "--batch-size")?, "--batch-size")?;
                }
                "--learning-rate" => {
                    cli.learning_rate =
                        parse_f64(&next_arg(&mut args, "--learning-rate")?, "--learning-rate")?;
                }
                "--seed" => {
                    cli.seed = parse_u64(&next_arg(&mut args, "--seed")?, "--seed")?;
                }
                "--data-seed" => {
                    cli.data_seed = Some(parse_u64(
                        &next_arg(&mut args, "--data-seed")?,
                        "--data-seed",
                    )?);
                }
                "--full-eval-pass" => {
                    cli.full_eval_pass = true;
                }
                "--output" => {
                    cli.output = parse_output_format(&next_arg(&mut args, "--output")?)?;
                }
                "--output-dir" => {
                    cli.output_dir = Some(PathBuf::from(next_arg(&mut args, "--output-dir")?));
                }
                "--help" | "-h" => {
                    return Err(help_text());
                }
                other => {
                    return Err(format!(
                        "unrecognized argument `{other}`\n\n{}",
                        help_text()
                    ));
                }
            }
        }
        Ok(cli)
    }
}

fn next_arg<I>(args: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_variant(value: &str) -> Result<VariantSelection, String> {
    match value {
        "all" => Ok(VariantSelection::All),
        "reference" => Ok(VariantSelection::Reference),
        "recurrent" => Ok(VariantSelection::Recurrent),
        other => Err(format!(
            "invalid --variant `{other}`; expected one of: all, reference, recurrent"
        )),
    }
}

fn parse_backend(value: &str) -> Result<BackendSelection, String> {
    match value {
        "cpu" => Ok(BackendSelection::Cpu),
        "metal" => Ok(BackendSelection::Metal),
        other => Err(format!(
            "invalid --backend `{other}`; expected one of: cpu, metal"
        )),
    }
}

fn parse_output_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "table" => Ok(OutputFormat::Table),
        "json" => Ok(OutputFormat::Json),
        other => Err(format!(
            "invalid --output `{other}`; expected one of: table, json"
        )),
    }
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag} `{value}`: {error}"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid value for {flag} `{value}`: {error}"))
}

fn parse_f64(value: &str, flag: &str) -> Result<f64, String> {
    value
        .parse::<f64>()
        .map_err(|error| format!("invalid value for {flag} `{value}`: {error}"))
}

fn help_text() -> String {
    format!(
        "{EXPERIMENT_TITLE}\n\
         Usage: cargo run --bin {EXPERIMENT_SLUG} -- [options]\n\n\
         Options:\n\
           --variant <all|reference|recurrent>\n\
           --backend <cpu|metal>\n\
           --steps <usize>\n\
           --eval-batches <usize>\n\
           --seq-len <usize>\n\
           --window-stride <usize>\n\
           --batch-size <usize>\n\
           --learning-rate <f64>\n\
           --seed <u64>\n\
           --data-seed <u64>\n\
           --full-eval-pass\n\
           --output <table|json>\n\
           --output-dir <path>\n"
    )
}
