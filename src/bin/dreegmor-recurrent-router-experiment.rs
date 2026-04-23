use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::{
    goe_over_attention_only_variant_with_controller, initialize_metal_runtime,
    phase1_hybrid_attention_baseline_matrix, recurrent_goe_over_attention_only_variant,
    CpuTrainBackend, GraphOfExpertsControllerSpec, MetalTrainBackend,
};
use fractal_eval_private::{
    byte_level_smoke_corpus_stats_from_source, default_v3a_fineweb_stage0_canary_corpus_source,
    run_attention_only_goe_smoke_train, run_attention_only_hybrid_attention_smoke_train,
    run_attention_only_recurrent_goe_smoke_train, ByteLevelSmokeCorpusSource,
    GraphOfExpertsRoutingSummary, GraphOfExpertsSmokeTrainConfig, HybridAttentionExecutionBackend,
    HybridAttentionSmokeTrainConfig, RecurrentGraphOfExpertsSmokeTrainConfig,
    DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE, DEFAULT_V3A_SMOKE_SEED,
    DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};
use serde::Serialize;

const EXPERIMENT_TITLE: &str = "DREEGMOR Recurrent Router Experiment";
const EXPERIMENT_SLUG: &str = "dreegmor-recurrent-router-experiment";

fn main() {
    if let Err(error) = run() {
        eprintln!("{EXPERIMENT_SLUG}: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_source = resolve_corpus_source(&args, &repo_root)?;
    let args = apply_full_pass_overrides(args, &corpus_source)?;
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
    let matrix = phase1_hybrid_attention_baseline_matrix();
    let one_shot_variant = goe_over_attention_only_variant_with_controller(
        GraphOfExpertsControllerSpec::routed_no_graph_mix(),
    );
    let recurrent_variant = recurrent_goe_over_attention_only_variant();
    let mut variants = Vec::new();

    if args.variant.includes_attention_only() {
        let report = run_attention_only_hybrid_attention_smoke_train::<B>(
            hybrid_smoke_config(
                corpus_source.clone(),
                isolated_variant_dir(output_dir, "a"),
                matrix.attention_only,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run A baseline: {error}"))?;
        variants.push(ExperimentVariantSummary::baseline(
            "A",
            report.config.execution_backend,
            report.config.variant.label.clone(),
            report.model_label,
            report.note,
            report.config.seed,
            report.config.data_seed,
            report.initial_eval,
            report.final_eval,
            report.runtime,
            report.train_steps.len(),
            report.report_path,
        ));
    }

    if args.variant.includes_one_shot_dense() {
        let label = one_shot_variant.label.clone();
        let report = run_attention_only_goe_smoke_train::<B>(
            goe_smoke_config(
                corpus_source.clone(),
                isolated_variant_dir(output_dir, &label),
                one_shot_variant,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run one-shot DREEGMOR(A): {error}"))?;
        variants.push(ExperimentVariantSummary::routed(
            "DREEGMOR(A)",
            report.config.execution_backend,
            report.config.variant.label.clone(),
            report.model_label,
            report.note,
            report.config.seed,
            report.config.data_seed,
            report.initial_eval,
            report.final_eval,
            report.runtime,
            report.train_steps.len(),
            report.report_path,
            report.routing,
        ));
    }

    if args.variant.includes_recurrent_dense() {
        let label = recurrent_variant.label.clone();
        let report = run_attention_only_recurrent_goe_smoke_train::<B>(
            recurrent_smoke_config(
                corpus_source,
                isolated_variant_dir(output_dir, &label),
                recurrent_variant,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run recurrent DREEGMOR(A): {error}"))?;
        variants.push(ExperimentVariantSummary::routed(
            "DREEGMOR-Recurrent(A)",
            report.config.execution_backend,
            report.config.variant.label.clone(),
            report.model_label,
            report.note,
            report.config.seed,
            report.config.data_seed,
            report.initial_eval,
            report.final_eval,
            report.runtime,
            report.train_steps.len(),
            report.report_path,
            report.routing,
        ));
    }

    Ok(variants)
}

fn hybrid_smoke_config(
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: PathBuf,
    variant: fractal_core::HybridAttentionVariantSpec,
    args: &CliArgs,
) -> HybridAttentionSmokeTrainConfig {
    let mut config = HybridAttentionSmokeTrainConfig::new(corpus_source, output_dir, variant);
    config.execution_backend = args.backend.execution_backend();
    config.seq_len = args.seq_len;
    config.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.batch_size = args.batch_size;
    config.train_steps = args.steps;
    config.eval_batches = args.eval_batches;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    config.seed = args.seed;
    config.data_seed = args.data_seed;
    config
}

fn goe_smoke_config(
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: PathBuf,
    variant: fractal_core::GraphOfExpertsVariantSpec,
    args: &CliArgs,
) -> GraphOfExpertsSmokeTrainConfig {
    let mut config = GraphOfExpertsSmokeTrainConfig::new(corpus_source, output_dir, variant);
    config.execution_backend = args.backend.execution_backend();
    config.seq_len = args.seq_len;
    config.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.batch_size = args.batch_size;
    config.train_steps = args.steps;
    config.eval_batches = args.eval_batches;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    config.seed = args.seed;
    config.data_seed = args.data_seed;
    config
}

fn recurrent_smoke_config(
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: PathBuf,
    variant: fractal_core::AttentionOnlyRecurrentGraphOfExpertsVariantSpec,
    args: &CliArgs,
) -> RecurrentGraphOfExpertsSmokeTrainConfig {
    let mut config =
        RecurrentGraphOfExpertsSmokeTrainConfig::new(corpus_source, output_dir, variant);
    config.execution_backend = args.backend.execution_backend();
    config.seq_len = args.seq_len;
    config.window_stride = args.window_stride.unwrap_or(args.seq_len);
    config.batch_size = args.batch_size;
    config.train_steps = args.steps;
    config.eval_batches = args.eval_batches;
    config.eval_holdout_every = args.eval_holdout_every;
    config.learning_rate = args.learning_rate;
    config.seed = args.seed;
    config.data_seed = args.data_seed;
    config
}

fn resolve_corpus_source(
    args: &CliArgs,
    repo_root: &Path,
) -> Result<ByteLevelSmokeCorpusSource, String> {
    if let (Some(train_path), Some(eval_path)) = (&args.jsonl_train_path, &args.jsonl_eval_path) {
        let corpus_name = args.corpus_name.clone().unwrap_or_else(|| {
            train_path
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                .unwrap_or("dreegmor-recurrent-jsonl-split")
                .to_owned()
        });
        Ok(ByteLevelSmokeCorpusSource::jsonl_text_splits(
            corpus_name,
            train_path.clone(),
            eval_path.clone(),
            args.corpus_text_field.clone(),
        ))
    } else if args.corpus_paths.is_empty() {
        default_v3a_fineweb_stage0_canary_corpus_source(repo_root)
            .map_err(|error| format!("failed to resolve default v3a FineWeb smoke corpus: {error}"))
    } else {
        Ok(ByteLevelSmokeCorpusSource::raw_files(
            args.corpus_paths.clone(),
        ))
    }
}

fn apply_full_pass_overrides(
    mut args: CliArgs,
    corpus_source: &ByteLevelSmokeCorpusSource,
) -> Result<CliArgs, String> {
    if !(args.full_train_pass || args.full_eval_pass) {
        return Ok(args);
    }
    let stats = byte_level_smoke_corpus_stats_from_source(
        corpus_source,
        args.seq_len,
        args.window_stride.unwrap_or(args.seq_len),
        args.eval_holdout_every,
    )
    .map_err(|error| format!("failed to derive corpus stats for full-pass run: {error}"))?;
    if args.full_train_pass {
        args.steps = stats.train_sequences.div_ceil(args.batch_size);
    }
    if args.full_eval_pass {
        args.eval_batches = stats.eval_sequences.div_ceil(args.batch_size);
    }
    Ok(args)
}

fn isolated_variant_dir(root: &Path, label: &str) -> PathBuf {
    root.join(label.replace('/', "_"))
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    corpus_paths: Vec<PathBuf>,
    jsonl_train_path: Option<PathBuf>,
    jsonl_eval_path: Option<PathBuf>,
    corpus_name: Option<String>,
    corpus_text_field: String,
    output_dir: Option<PathBuf>,
    seq_len: usize,
    window_stride: Option<usize>,
    batch_size: usize,
    steps: usize,
    eval_batches: usize,
    full_train_pass: bool,
    full_eval_pass: bool,
    eval_holdout_every: usize,
    learning_rate: f64,
    backend: BackendSelection,
    seed: u64,
    data_seed: Option<u64>,
    variant: VariantSelection,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut corpus_paths = Vec::new();
        let mut jsonl_train_path = None;
        let mut jsonl_eval_path = None;
        let mut corpus_name = None;
        let mut corpus_text_field = "text".to_owned();
        let mut output_dir = None;
        let mut seq_len = DEFAULT_V3A_SMOKE_SEQ_LEN;
        let mut window_stride = Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE);
        let mut batch_size = DEFAULT_V3A_SMOKE_BATCH_SIZE;
        let mut steps = DEFAULT_V3A_SMOKE_TRAIN_STEPS;
        let mut eval_batches = DEFAULT_V3A_SMOKE_EVAL_BATCHES;
        let mut full_train_pass = false;
        let mut full_eval_pass = false;
        let mut eval_holdout_every = DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY;
        let mut learning_rate = DEFAULT_V3A_SMOKE_LEARNING_RATE;
        let mut backend = BackendSelection::Cpu;
        let mut seed = DEFAULT_V3A_SMOKE_SEED;
        let mut data_seed = None;
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
                "--jsonl-train-path" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--jsonl-train-path requires a value".to_owned())?;
                    jsonl_train_path = Some(PathBuf::from(value));
                }
                "--jsonl-eval-path" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--jsonl-eval-path requires a value".to_owned())?;
                    jsonl_eval_path = Some(PathBuf::from(value));
                }
                "--corpus-name" => {
                    corpus_name = Some(
                        iter.next()
                            .ok_or_else(|| "--corpus-name requires a value".to_owned())?,
                    );
                }
                "--corpus-text-field" => {
                    corpus_text_field = iter
                        .next()
                        .ok_or_else(|| "--corpus-text-field requires a value".to_owned())?;
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
                "--full-train-pass" => full_train_pass = true,
                "--full-eval-pass" => full_eval_pass = true,
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
                "--backend" => {
                    backend = BackendSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--backend requires a value".to_owned())?,
                    )?;
                }
                "--seed" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--seed requires a value".to_owned())?;
                    seed = value
                        .parse::<u64>()
                        .map_err(|error| format!("invalid --seed value '{value}': {error}"))?;
                }
                "--data-seed" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--data-seed requires a value".to_owned())?;
                    data_seed = Some(value.parse::<u64>().map_err(|error| {
                        format!("invalid --data-seed value '{value}': {error}")
                    })?);
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

        if (!corpus_paths.is_empty()) && (jsonl_train_path.is_some() || jsonl_eval_path.is_some()) {
            return Err(
                "--corpus-path may not be combined with --jsonl-train-path/--jsonl-eval-path"
                    .to_owned(),
            );
        }
        if jsonl_train_path.is_some() ^ jsonl_eval_path.is_some() {
            return Err(
                "--jsonl-train-path and --jsonl-eval-path must be provided together".to_owned(),
            );
        }
        if full_train_pass && steps != DEFAULT_V3A_SMOKE_TRAIN_STEPS {
            return Err("--full-train-pass may not be combined with explicit --steps".to_owned());
        }
        if full_eval_pass && eval_batches != DEFAULT_V3A_SMOKE_EVAL_BATCHES {
            return Err(
                "--full-eval-pass may not be combined with explicit --eval-batches".to_owned(),
            );
        }

        Ok(Self {
            corpus_paths,
            jsonl_train_path,
            jsonl_eval_path,
            corpus_name,
            corpus_text_field,
            output_dir,
            seq_len,
            window_stride,
            batch_size,
            steps,
            eval_batches,
            full_train_pass,
            full_eval_pass,
            eval_holdout_every,
            learning_rate,
            backend,
            seed,
            data_seed,
            variant,
            output,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariantSelection {
    All,
    AttentionOnly,
    OneShotDense,
    RecurrentDense,
}

impl VariantSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "a" | "attention-only" => Ok(Self::AttentionOnly),
            "one-shot-a" | "dreegmor-a" | "one-shot-dense" => Ok(Self::OneShotDense),
            "recurrent-a" | "dreegmor-recurrent-a" | "recurrent-dense" => Ok(Self::RecurrentDense),
            _ => Err(format!("unknown variant selection: {value}")),
        }
    }

    fn includes_attention_only(self) -> bool {
        matches!(self, Self::All | Self::AttentionOnly)
    }

    fn includes_one_shot_dense(self) -> bool {
        matches!(self, Self::All | Self::OneShotDense)
    }

    fn includes_recurrent_dense(self) -> bool {
        matches!(self, Self::All | Self::RecurrentDense)
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

    fn execution_backend(self) -> HybridAttentionExecutionBackend {
        match self {
            Self::Cpu => HybridAttentionExecutionBackend::Cpu,
            Self::Metal => HybridAttentionExecutionBackend::Metal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct ExperimentVariantSummary {
    label: String,
    execution_backend: HybridAttentionExecutionBackend,
    variant_label: String,
    model_label: String,
    note: String,
    seed: u64,
    data_seed: Option<u64>,
    initial_loss: f64,
    final_loss: f64,
    initial_perplexity: f64,
    final_perplexity: f64,
    train_tokens_per_second: f64,
    overall_tokens_per_second: f64,
    process_memory_metric: fractal_eval_private::ProcessMemoryMetricKind,
    peak_process_memory_delta_bytes: u64,
    train_step_count: usize,
    report_path: PathBuf,
    routing: Option<GraphOfExpertsRoutingSummary>,
}

impl ExperimentVariantSummary {
    fn baseline(
        label: &str,
        execution_backend: HybridAttentionExecutionBackend,
        variant_label: String,
        model_label: String,
        note: String,
        seed: u64,
        data_seed: Option<u64>,
        initial_eval: fractal_eval_private::V2SmokeEvalMetrics,
        final_eval: fractal_eval_private::V2SmokeEvalMetrics,
        runtime: fractal_eval_private::HybridAttentionRuntimeMetrics,
        train_step_count: usize,
        report_path: PathBuf,
    ) -> Self {
        Self {
            label: label.to_string(),
            execution_backend,
            variant_label,
            model_label,
            note,
            seed,
            data_seed,
            initial_loss: initial_eval.mean_loss,
            final_loss: final_eval.mean_loss,
            initial_perplexity: initial_eval.perplexity,
            final_perplexity: final_eval.perplexity,
            train_tokens_per_second: runtime.train_tokens_per_second,
            overall_tokens_per_second: runtime.overall_tokens_per_second,
            process_memory_metric: runtime.process_memory_metric,
            peak_process_memory_delta_bytes: runtime.peak_process_memory_delta_bytes,
            train_step_count,
            report_path,
            routing: None,
        }
    }

    fn routed(
        label: &str,
        execution_backend: HybridAttentionExecutionBackend,
        variant_label: String,
        model_label: String,
        note: String,
        seed: u64,
        data_seed: Option<u64>,
        initial_eval: fractal_eval_private::V2SmokeEvalMetrics,
        final_eval: fractal_eval_private::V2SmokeEvalMetrics,
        runtime: fractal_eval_private::HybridAttentionRuntimeMetrics,
        train_step_count: usize,
        report_path: PathBuf,
        routing: GraphOfExpertsRoutingSummary,
    ) -> Self {
        let mut summary = Self::baseline(
            label,
            execution_backend,
            variant_label,
            model_label,
            note,
            seed,
            data_seed,
            initial_eval,
            final_eval,
            runtime,
            train_step_count,
            report_path,
        );
        summary.routing = Some(routing);
        summary
    }
}

fn render_report(
    summaries: &[ExperimentVariantSummary],
    output: OutputFormat,
) -> Result<String, String> {
    match output {
        OutputFormat::Json => serde_json::to_string_pretty(summaries)
            .map(|rendered| format!("{rendered}\n"))
            .map_err(|error| format!("failed to serialize recurrent router report: {error}")),
        OutputFormat::Table => Ok(render_table(summaries)),
    }
}

fn render_table(summaries: &[ExperimentVariantSummary]) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "{EXPERIMENT_TITLE}");
    let _ = writeln!(
        output,
        "note: recurrent-router lane over frozen A. variants run sequentially and write to isolated subdirectories."
    );
    for summary in summaries {
        let _ = writeln!(output, "- {} ({})", summary.label, summary.model_label);
        let _ = writeln!(
            output,
            "  variant={} backend={} seed={} data_seed={}",
            summary.variant_label,
            summary.execution_backend.as_str(),
            summary.seed,
            format_data_seed(summary.data_seed)
        );
        let _ = writeln!(
            output,
            "  initial_loss={:.4} final_loss={:.4} initial_ppl={:.2} final_ppl={:.2} steps={}",
            summary.initial_loss,
            summary.final_loss,
            summary.initial_perplexity,
            summary.final_perplexity,
            summary.train_step_count,
        );
        let _ = writeln!(
            output,
            "  train_tok_s={:.2} overall_tok_s={:.2} process_mem_delta_mb={:.2} metric={}",
            summary.train_tokens_per_second,
            summary.overall_tokens_per_second,
            summary.peak_process_memory_delta_bytes as f64 / (1024.0 * 1024.0),
            summary.process_memory_metric.as_str(),
        );
        if let Some(routing) = &summary.routing {
            let _ = writeln!(
                output,
                "  routing_initial=[{:.3}, {:.3}] routing_final=[{:.3}, {:.3}] winners=[{}, {}] ties={} active_channels={} rounds={}",
                routing.mean_pre_graph_channel_weights[0],
                routing.mean_pre_graph_channel_weights[1],
                routing.mean_channel_weights[0],
                routing.mean_channel_weights[1],
                routing.winner_counts[0],
                routing.winner_counts[1],
                routing.tied_token_count,
                routing.active_channel_count,
                routing.round_count,
            );
            let _ = writeln!(
                output,
                "  routing_entropy_bits={:.3} winner_margin={:.3} final_adjustment_l1={:.3} per_round_l1={:?} edge_mix={:.3} sampled_tokens={}",
                routing.mean_route_entropy_bits,
                routing.mean_winner_margin,
                routing.mean_graph_adjustment_l1,
                routing.mean_round_adjustment_l1,
                routing.edge_mix_fraction,
                routing.sampled_tokens,
            );
        }
        let _ = writeln!(output, "  report={}", summary.report_path.display());
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
        .join(EXPERIMENT_SLUG)
        .join(timestamp.to_string())
}

fn usage() -> String {
    concat!(
        "Usage: cargo run --bin dreegmor-recurrent-router-experiment -- [options]\n\n",
        "Options:\n",
        "  --corpus-path <path>              Add a raw text corpus file\n",
        "  --jsonl-train-path <path>         Train JSONL split path\n",
        "  --jsonl-eval-path <path>          Eval JSONL split path\n",
        "  --corpus-name <name>              Corpus name for JSONL split input\n",
        "  --corpus-text-field <field>       JSONL text field (default: text)\n",
        "  --output-dir <path>               Artifact output directory root\n",
        "  --seq-len <n>                     Sequence length (default: 32)\n",
        "  --window-stride <n>               Window stride (default: seq-len)\n",
        "  --batch-size <n>                  Batch size (default: 1)\n",
        "  --steps <n>                       Train steps per variant\n",
        "  --eval-batches <n>                Eval batches per pass\n",
        "  --full-train-pass                 Train over the full derived train split\n",
        "  --full-eval-pass                  Evaluate over the full derived eval split\n",
        "  --eval-holdout-every <n>          Holdout cadence (default: 10)\n",
        "  --learning-rate <lr>              Learning rate (default: 1e-3)\n",
        "  --backend <cpu|metal>             Execution backend (default: cpu)\n",
        "  --seed <n>                        Model/init RNG seed (default: 42)\n",
        "  --data-seed <n>                   Optional train-order shuffle seed (default: fixed order)\n",
        "  --variant <all|a|one-shot-a|recurrent-a>\n",
        "                                    Variant selection (default: all)\n",
        "  --output <table|json>             Output format (default: table)\n",
        "  --help                            Show this help\n",
    )
    .to_string()
}

fn format_data_seed(data_seed: Option<u64>) -> String {
    match data_seed {
        Some(seed) => seed.to_string(),
        None => "fixed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendSelection, CliArgs, OutputFormat, VariantSelection};

    #[test]
    fn defaults_stay_cpu_smoke_friendly() {
        let args = CliArgs::parse(std::iter::empty()).unwrap();
        assert_eq!(args.batch_size, 1);
        assert_eq!(args.backend, BackendSelection::Cpu);
        assert_eq!(args.variant, VariantSelection::All);
        assert_eq!(args.output, OutputFormat::Table);
    }

    #[test]
    fn parses_metal_backend() {
        let args =
            CliArgs::parse(["--backend".to_string(), "metal".to_string()].into_iter()).unwrap();
        assert_eq!(args.backend, BackendSelection::Metal);
    }
}
