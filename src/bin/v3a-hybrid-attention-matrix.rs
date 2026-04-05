use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::{phase1_hybrid_attention_baseline_matrix, CpuTrainBackend};
use fractal_eval_private::{
    default_v2_smoke_corpus_paths, run_attention_only_hybrid_attention_smoke_train,
    run_primitive_hybrid_attention_smoke_train, run_reference_ssm_hybrid_attention_smoke_train,
    HybridAttentionMatrixVariantOutcome, HybridAttentionSmokeTrainConfig,
    DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE, DEFAULT_V3A_SMOKE_SEED,
    DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};
use serde::Serialize;

fn main() {
    if let Err(error) = run() {
        eprintln!("v3a-hybrid-attention-matrix: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_paths = if args.corpus_paths.is_empty() {
        default_v2_smoke_corpus_paths(&repo_root)
            .map_err(|error| format!("failed to resolve default smoke corpus: {error}"))?
    } else {
        args.corpus_paths.clone()
    };
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let matrix = phase1_hybrid_attention_baseline_matrix();
    let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
    let mut variants = Vec::new();
    if args.variant.includes_attention_only() {
        variants.push(
            run_attention_only_hybrid_attention_smoke_train::<CpuTrainBackend>(
                smoke_config(
                    corpus_paths.clone(),
                    output_dir.join("attention-only"),
                    matrix.attention_only,
                    &args,
                ),
                &device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run attention-only baseline: {error}"))?,
        );
    }
    if args.variant.includes_reference_ssm_hybrid() {
        variants.push(
            run_reference_ssm_hybrid_attention_smoke_train::<CpuTrainBackend>(
                smoke_config(
                    corpus_paths.clone(),
                    output_dir.join("reference-ssm-hybrid"),
                    matrix.reference_ssm_hybrid,
                    &args,
                ),
                &device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run reference-ssm-hybrid baseline: {error}"))?,
        );
    }
    if args.variant.includes_primitive_hybrid() {
        variants.push(
            run_primitive_hybrid_attention_smoke_train::<CpuTrainBackend>(
                smoke_config(
                    corpus_paths,
                    output_dir.join("primitive-hybrid"),
                    matrix.primitive_hybrid,
                    &args,
                ),
                &device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run primitive-hybrid baseline: {error}"))?,
        );
    }

    let rendered = render_report(
        &RenderedMatrixReport {
            note: "Path 1 baseline matrix: attention-only, reference-ssm-hybrid, and primitive-hybrid are executable. The reference lane now uses the faithful Rust Mamba-3-style baseline block instead of the old proxy lane.",
            variants,
        },
        args.output,
    )?;
    print!("{rendered}");
    Ok(())
}

fn smoke_config(
    corpus_paths: Vec<PathBuf>,
    output_dir: PathBuf,
    variant: fractal_core::HybridAttentionVariantSpec,
    args: &CliArgs,
) -> HybridAttentionSmokeTrainConfig {
    let mut config = HybridAttentionSmokeTrainConfig::new(corpus_paths, output_dir, variant);
    config.seq_len = args.seq_len;
    config.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.batch_size = args.batch_size;
    config.train_steps = args.steps;
    config.eval_batches = args.eval_batches;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    config.seed = args.seed;
    config
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
    seed: u64,
    variant: VariantSelection,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut corpus_paths = Vec::new();
        let mut output_dir = None;
        let mut seq_len = DEFAULT_V3A_SMOKE_SEQ_LEN;
        let mut window_stride = Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE);
        let mut batch_size = DEFAULT_V3A_SMOKE_BATCH_SIZE;
        let mut steps = DEFAULT_V3A_SMOKE_TRAIN_STEPS;
        let mut eval_batches = DEFAULT_V3A_SMOKE_EVAL_BATCHES;
        let mut eval_holdout_every = DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = DEFAULT_V3A_SMOKE_LEARNING_RATE;
        let mut seed = DEFAULT_V3A_SMOKE_SEED;
        let mut variant = VariantSelection::All;
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
                "--seed" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--seed requires a value".to_owned())?;
                    seed = value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid --seed value '{value}': {error}"))?;
                }
                "--variant" => {
                    variant = VariantSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--variant requires a value".to_owned())?,
                    )?;
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
            seed,
            variant,
            output,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariantSelection {
    All,
    AttentionOnly,
    ReferenceSsmHybrid,
    PrimitiveHybrid,
}

impl VariantSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "attention-only" => Ok(Self::AttentionOnly),
            "reference-ssm-hybrid" => Ok(Self::ReferenceSsmHybrid),
            "primitive-hybrid" => Ok(Self::PrimitiveHybrid),
            _ => Err(format!("unknown variant selection: {value}")),
        }
    }

    fn includes_attention_only(self) -> bool {
        matches!(self, Self::All | Self::AttentionOnly)
    }

    fn includes_reference_ssm_hybrid(self) -> bool {
        matches!(self, Self::All | Self::ReferenceSsmHybrid)
    }

    fn includes_primitive_hybrid(self) -> bool {
        matches!(self, Self::All | Self::PrimitiveHybrid)
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
struct RenderedMatrixReport<'a> {
    note: &'a str,
    variants: Vec<HybridAttentionMatrixVariantOutcome>,
}

fn render_report(
    report: &RenderedMatrixReport<'_>,
    format: OutputFormat,
) -> Result<String, String> {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(report)
            .map(|rendered| format!("{rendered}\n"))
            .map_err(|error| format!("failed to serialize matrix report: {error}")),
        OutputFormat::Table => Ok(render_table(report)),
    }
}

fn render_table(report: &RenderedMatrixReport<'_>) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Path 1 Hybrid Attention Matrix");
    let _ = writeln!(output, "note: {}", report.note);
    for variant in &report.variants {
        match variant {
            HybridAttentionMatrixVariantOutcome::Executed(report) => {
                let _ = writeln!(
                    output,
                    "- {} ({})",
                    report.config.variant.label, report.model_label
                );
                let _ = writeln!(
                    output,
                    "  initial_loss={:.4} final_loss={:.4} initial_ppl={:.2} final_ppl={:.2} steps={}",
                    report.initial_eval.mean_loss,
                    report.final_eval.mean_loss,
                    report.initial_eval.perplexity,
                    report.final_eval.perplexity,
                    report.train_steps.len(),
                );
                let _ = writeln!(
                    output,
                    "  seed={} report={}",
                    report.config.seed,
                    report.report_path.display()
                );
            }
            HybridAttentionMatrixVariantOutcome::RequiredMissing { label, reason, .. } => {
                let _ = writeln!(output, "- {} (required-missing)", label);
                let _ = writeln!(output, "  reason={reason}");
            }
        }
    }
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
        .as_secs();
    repo_root
        .join("artifacts")
        .join("v3a-hybrid-attention-matrix")
        .join(timestamp.to_string())
}

fn usage() -> String {
    format!(
        concat!(
            "Usage: cargo run --bin v3a-hybrid-attention-matrix -- [options]\n\n",
            "Options:\n",
            "  --corpus-path <path>         Add a byte-level corpus file (repeatable)\n",
            "  --output-dir <path>          Directory for per-variant reports\n",
            "  --seq-len <n>                Training sequence length (default: {seq_len})\n",
            "  --window-stride <n>          Sliding window stride (default: {stride})\n",
            "  --batch-size <n>             Batch size (default: {batch_size})\n",
            "  --steps <n>                  Training steps per implemented variant (default: {steps})\n",
            "  --eval-batches <n>           Eval batches (default: {eval_batches})\n",
            "  --eval-holdout-every <n>     Hold out every nth sequence for eval (default: {holdout})\n",
            "  --learning-rate <value>      Learning rate (default: {lr})\n",
            "  --seed <n>                   Random seed for model initialization (default: {seed})\n",
            "  --variant <name>             One of: all, attention-only, reference-ssm-hybrid, primitive-hybrid (default: all)\n",
            "  --output <table|json>        Output format (default: table)\n",
            "  -h, --help                   Show this help\n",
        ),
        seq_len = DEFAULT_V3A_SMOKE_SEQ_LEN,
        stride = DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
        batch_size = DEFAULT_V3A_SMOKE_BATCH_SIZE,
        steps = DEFAULT_V3A_SMOKE_TRAIN_STEPS,
        eval_batches = DEFAULT_V3A_SMOKE_EVAL_BATCHES,
        holdout = DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
        lr = DEFAULT_V3A_SMOKE_LEARNING_RATE,
        seed = DEFAULT_V3A_SMOKE_SEED,
    )
}
