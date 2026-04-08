use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use fractal_core::{
    goe_over_attention_only_variant_with_controller,
    goe_over_reference_ssm_variant_with_controller, initialize_metal_runtime,
    phase1_hybrid_attention_baseline_matrix, CpuTrainBackend, GraphOfExpertsControllerSpec,
    GraphOfExpertsRoutingMode, GraphOfExpertsTopology, MetalTrainBackend,
};
use fractal_eval_private::{
    append_goe_results_ledger_entry, byte_level_smoke_corpus_stats_from_source,
    default_v3a_fineweb_stage0_canary_corpus_source, resolve_requested_goe_results_ledger_path,
    run_attention_only_goe_smoke_train, run_attention_only_hybrid_attention_smoke_train,
    run_reference_ssm_goe_smoke_train, run_reference_ssm_hybrid_attention_smoke_train,
    ByteLevelSmokeCorpusSource, GoeResultsLedgerEntry, GraphOfExpertsExperimentReport,
    GraphOfExpertsExperimentVariantSummary, GraphOfExpertsSmokeTrainConfig,
    HybridAttentionExecutionBackend, HybridAttentionSmokeTrainConfig, DEFAULT_V3A_SMOKE_BATCH_SIZE,
    DEFAULT_V3A_SMOKE_EVAL_BATCHES, DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
    DEFAULT_V3A_SMOKE_LEARNING_RATE, DEFAULT_V3A_SMOKE_SEED, DEFAULT_V3A_SMOKE_SEQ_LEN,
    DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};
use serde::Serialize;

const EXPERIMENT_TITLE: &str = "DREEGMOR A/A+M Experiment";
const EXPERIMENT_SLUG: &str = "dreegmor-a-am-experiment";
const LEDGER_MODEL_NAME: &str = "dreegmor_a_am_experiment";

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
    let controller = selected_goe_controller(&args)?;
    let note = experiment_note(args.backend.as_str(), &args.variant, &controller);
    let variants = run_selected_variants(&args, corpus_source, &output_dir, controller)?;
    let report = GraphOfExpertsExperimentReport {
        note: note.clone(),
        variants,
    };
    let rendered = render_report(&report, args.output)?;
    maybe_append_ledger_entry(
        resolve_requested_goe_results_ledger_path(&repo_root, args.ledger_path.as_deref())
            .map_err(|error| format!("failed to resolve GoE results ledger path: {error}"))?,
        &report,
        args.run_label.as_deref(),
    )?;
    print!("{rendered}");
    Ok(())
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
                .unwrap_or("goe-jsonl-split")
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

fn selected_goe_controller(args: &CliArgs) -> Result<GraphOfExpertsControllerSpec, String> {
    let controller = GraphOfExpertsControllerSpec {
        routing_mode: args.goe_routing_mode.routing_mode(),
        topology: args.goe_graph_topology.topology(),
        channel_count: fractal_core::GOE_CHANNEL_COUNT,
    };
    controller
        .validate()
        .map_err(|error| format!("invalid GoE controller selection: {error}"))?;
    Ok(controller)
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

fn run_selected_variants(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
    controller: GraphOfExpertsControllerSpec,
) -> Result<Vec<GraphOfExpertsExperimentVariantSummary>, String> {
    match args.backend {
        BackendSelection::Cpu => {
            let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
            run_selected_variants_with_backend::<CpuTrainBackend>(
                args,
                corpus_source,
                output_dir,
                controller,
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
                controller,
                &device,
            )
        }
    }
}

fn run_selected_variants_with_backend<B>(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
    controller: GraphOfExpertsControllerSpec,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Result<Vec<GraphOfExpertsExperimentVariantSummary>, String>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    let matrix = phase1_hybrid_attention_baseline_matrix();
    let goe_a_variant = goe_over_attention_only_variant_with_controller(controller.clone());
    let goe_am_variant = goe_over_reference_ssm_variant_with_controller(controller);
    let mut variants = Vec::new();

    if args.variant.includes_attention_only() {
        let report = run_attention_only_hybrid_attention_smoke_train::<B>(
            hybrid_smoke_config(
                corpus_source.clone(),
                output_dir.join("a"),
                matrix.attention_only,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run A baseline: {error}"))?;
        variants.push(GraphOfExpertsExperimentVariantSummary::attention_only_baseline(&report));
    }
    if args.variant.includes_reference_ssm() {
        let report = run_reference_ssm_hybrid_attention_smoke_train::<B>(
            hybrid_smoke_config(
                corpus_source.clone(),
                output_dir.join("a-plus-m"),
                matrix.reference_ssm_hybrid,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run A + M baseline: {error}"))?;
        variants.push(GraphOfExpertsExperimentVariantSummary::reference_ssm_baseline(&report));
    }
    if args.variant.includes_goe_attention_only() {
        let report = run_attention_only_goe_smoke_train::<B>(
            goe_smoke_config(
                corpus_source.clone(),
                output_dir.join(&goe_a_variant.label),
                goe_a_variant,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run DREEGMOR(A): {error}"))?;
        variants.push(GraphOfExpertsExperimentVariantSummary::goe_attention_only(
            &report,
        ));
    }
    if args.variant.includes_goe_reference_ssm() {
        let report = run_reference_ssm_goe_smoke_train::<B>(
            goe_smoke_config(
                corpus_source,
                output_dir.join(&goe_am_variant.label),
                goe_am_variant,
                args,
            ),
            device,
        )
        .map_err(|error| format!("failed to run DREEGMOR(A + M): {error}"))?;
        variants.push(GraphOfExpertsExperimentVariantSummary::goe_reference_ssm(
            &report,
        ));
    }

    Ok(variants)
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    backend: BackendSelection,
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
    seed: u64,
    data_seed: Option<u64>,
    variant: VariantSelection,
    goe_routing_mode: GoeRoutingModeSelection,
    goe_graph_topology: GoeGraphTopologySelection,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut backend = BackendSelection::Cpu;
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
        let mut seed = DEFAULT_V3A_SMOKE_SEED;
        let mut data_seed = None;
        let mut variant = VariantSelection::All;
        let mut goe_routing_mode = GoeRoutingModeSelection::TokenLocalRouter;
        let mut goe_graph_topology = GoeGraphTopologySelection::TwoNodeLine;
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
                "--goe-routing-mode" => {
                    goe_routing_mode = GoeRoutingModeSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--goe-routing-mode requires a value".to_owned())?,
                    )?;
                }
                "--goe-graph-topology" => {
                    goe_graph_topology = GoeGraphTopologySelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--goe-graph-topology requires a value".to_owned())?,
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
            backend,
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
            seed,
            data_seed,
            variant,
            goe_routing_mode,
            goe_graph_topology,
            ledger_path,
            run_label,
            output,
        })
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

    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }

    fn execution_backend(self) -> HybridAttentionExecutionBackend {
        match self {
            Self::Cpu => HybridAttentionExecutionBackend::Cpu,
            Self::Metal => HybridAttentionExecutionBackend::Metal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoeRoutingModeSelection {
    UniformAverage,
    TokenLocalRouter,
}

impl GoeRoutingModeSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "uniform-average" => Ok(Self::UniformAverage),
            "token-local-router" | "routed" => Ok(Self::TokenLocalRouter),
            _ => Err(format!("unknown GoE routing mode: {value}")),
        }
    }

    fn routing_mode(self) -> GraphOfExpertsRoutingMode {
        match self {
            Self::UniformAverage => GraphOfExpertsRoutingMode::UniformAverage,
            Self::TokenLocalRouter => GraphOfExpertsRoutingMode::TokenLocalRouter,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoeGraphTopologySelection {
    None,
    TwoNodeLine,
}

impl GoeGraphTopologySelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "two-node-line" | "line" => Ok(Self::TwoNodeLine),
            _ => Err(format!("unknown GoE graph topology: {value}")),
        }
    }

    fn topology(self) -> GraphOfExpertsTopology {
        match self {
            Self::None => GraphOfExpertsTopology::None,
            Self::TwoNodeLine => GraphOfExpertsTopology::TwoNodeLine,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariantSelection {
    All,
    AttentionOnly,
    ReferenceSsm,
    GoeAttentionOnly,
    GoeReferenceSsm,
}

impl VariantSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "a" | "attention-only" => Ok(Self::AttentionOnly),
            "a-plus-m" | "reference-ssm-hybrid" => Ok(Self::ReferenceSsm),
            "goe-a" | "dreegmor-a" => Ok(Self::GoeAttentionOnly),
            "goe-a-plus-m" | "goe-reference-ssm" | "dreegmor-a-plus-m" => Ok(Self::GoeReferenceSsm),
            _ => Err(format!("unknown variant selection: {value}")),
        }
    }

    fn includes_attention_only(self) -> bool {
        matches!(self, Self::All | Self::AttentionOnly)
    }

    fn includes_reference_ssm(self) -> bool {
        matches!(self, Self::All | Self::ReferenceSsm)
    }

    fn includes_goe_attention_only(self) -> bool {
        matches!(self, Self::All | Self::GoeAttentionOnly)
    }

    fn includes_goe_reference_ssm(self) -> bool {
        matches!(self, Self::All | Self::GoeReferenceSsm)
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
struct RenderedExperimentReport<'a> {
    note: &'a str,
    variants: &'a [GraphOfExpertsExperimentVariantSummary],
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &GraphOfExpertsExperimentReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = GoeResultsLedgerEntry::aam_experiment_run(
        LEDGER_MODEL_NAME,
        &report.note,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build GoE results ledger entry: {error}"))?;
    append_goe_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append GoE results ledger entry: {error}"))
}

fn render_report(
    report: &GraphOfExpertsExperimentReport,
    format: OutputFormat,
) -> Result<String, String> {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(&RenderedExperimentReport {
            note: &report.note,
            variants: &report.variants,
        })
        .map(|rendered| format!("{rendered}\n"))
        .map_err(|error| format!("failed to serialize experiment report: {error}")),
        OutputFormat::Table => Ok(render_table(report)),
    }
}

fn render_table(report: &GraphOfExpertsExperimentReport) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "{EXPERIMENT_TITLE}");
    let _ = writeln!(output, "note: {}", report.note);
    for variant in &report.variants {
        let _ = writeln!(output, "- {} ({})", variant.label, variant.model_label);
        let _ = writeln!(
            output,
            "  variant={} backend={} seed={}",
            variant.variant_label,
            variant.execution_backend.as_str(),
            variant.seed,
        );
        let _ = writeln!(
            output,
            "  initial_loss={:.4} final_loss={:.4} initial_ppl={:.2} final_ppl={:.2} steps={}",
            variant.initial_eval.mean_loss,
            variant.final_eval.mean_loss,
            variant.initial_eval.perplexity,
            variant.final_eval.perplexity,
            variant.train_step_count,
        );
        let _ = writeln!(
            output,
            "  train_tok_s={:.2} overall_tok_s={:.2} process_mem_delta_mb={:.2} metric={}",
            variant.runtime.train_tokens_per_second,
            variant.runtime.overall_tokens_per_second,
            variant.runtime.peak_process_memory_delta_bytes as f64 / (1024.0 * 1024.0),
            variant.runtime.process_memory_metric.as_str(),
        );
        if let Some(routing) = &variant.routing {
            let _ = writeln!(
                output,
                "  routing_pre=[{:.3}, {:.3}] routing_post=[{:.3}, {:.3}] winners=[{}, {}] ties={} active_channels={}",
                routing.mean_pre_graph_channel_weights[0],
                routing.mean_pre_graph_channel_weights[1],
                routing.mean_channel_weights[0],
                routing.mean_channel_weights[1],
                routing.winner_counts[0],
                routing.winner_counts[1],
                routing.tied_token_count,
                routing.active_channel_count,
            );
            let _ = writeln!(
                output,
                "  routing_entropy_bits={:.3} winner_margin={:.3} graph_adjustment_l1={:.3} edge_mix={:.3} sampled_tokens={}",
                routing.mean_route_entropy_bits,
                routing.mean_winner_margin,
                routing.mean_graph_adjustment_l1,
                routing.edge_mix_fraction,
                routing.sampled_tokens,
            );
        }
        let _ = writeln!(output, "  report={}", variant.report_path.display());
    }
    output
}

fn experiment_note(
    backend: &str,
    variant: &VariantSelection,
    controller: &GraphOfExpertsControllerSpec,
) -> String {
    let mut note = format!(
        "backend={backend}. Exploratory DREEGMOR A/A+M matrix on the shared byte-level smoke lane. This surface is separate from the Path 1 proving line, excludes P2 entirely, uses no external memory sidecar, and holds the frozen A / A + M backbones fixed while adding a dense two-channel controller only for the DREEGMOR variants."
    );
    if variant.includes_goe_attention_only() || variant.includes_goe_reference_ssm() {
        note = format!("{note} dreegmor_structure={}.", controller.label_suffix());
    }
    note
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
        "Usage: cargo run --bin dreegmor-a-am-experiment -- [options]\n\n",
        "Options:\n",
        "  --backend <cpu|metal>             Execution backend (default: cpu)\n",
        "  --corpus-path <path>              Add a raw text corpus file\n",
        "  --jsonl-train-path <path>         Train JSONL split path\n",
        "  --jsonl-eval-path <path>          Eval JSONL split path\n",
        "  --corpus-name <name>              Corpus name for JSONL split input\n",
        "  --corpus-text-field <field>       JSONL text field (default: text)\n",
        "  --output-dir <path>               Artifact output directory\n",
        "  --seq-len <n>                     Sequence length (default: 32)\n",
        "  --window-stride <n>               Window stride (default: seq-len)\n",
        "  --batch-size <n>                  Batch size (default: 1)\n",
        "  --steps <n>                       Train steps per variant\n",
        "  --eval-batches <n>                Eval batches per pass\n",
        "  --full-train-pass                 Train over the full derived train split\n",
        "  --full-eval-pass                  Evaluate over the full derived eval split\n",
        "  --eval-holdout-every <n>          Holdout cadence (default: 10)\n",
        "  --learning-rate <lr>              Learning rate (default: 1e-3)\n",
        "  --seed <n>                        Model/init RNG seed (default: 42)\n",
        "  --data-seed <n>                   Optional train-order shuffle seed (default: fixed order)\n",
        "  --variant <all|a|a-plus-m|dreegmor-a|dreegmor-a-plus-m>\n",
        "                                    Variant selection (default: all)\n",
        "  --goe-routing-mode <uniform-average|token-local-router>\n",
        "                                    GoE routing structure (default: token-local-router)\n",
        "  --goe-graph-topology <none|two-node-line>\n",
        "                                    Graph smoothing structure (default: two-node-line)\n",
        "  --ledger-path <default|path>      Append a structured GoE ledger entry\n",
        "  --run-label <label>               Optional label stored with the ledger entry\n",
        "  --output <table|json>             Output format (default: table)\n",
        "  --help                            Show this help\n",
    )
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        apply_full_pass_overrides, experiment_note, selected_goe_controller, CliArgs,
        GoeGraphTopologySelection, GoeRoutingModeSelection, OutputFormat, VariantSelection,
    };
    use fractal_core::{
        GraphOfExpertsControllerSpec, GraphOfExpertsRoutingMode, GraphOfExpertsTopology,
    };
    use fractal_eval_private::{
        byte_level_smoke_corpus_stats_from_source, default_v3a_fineweb_stage0_canary_corpus_source,
        DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
        DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE,
        DEFAULT_V3A_SMOKE_SEED, DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS,
        DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
    };
    use std::path::PathBuf;

    #[test]
    fn full_pass_overrides_match_checked_in_canary_counts() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let corpus = default_v3a_fineweb_stage0_canary_corpus_source(&repo_root).unwrap();
        let stats = byte_level_smoke_corpus_stats_from_source(
            &corpus,
            DEFAULT_V3A_SMOKE_SEQ_LEN,
            DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
            DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
        )
        .unwrap();
        let args = CliArgs {
            backend: super::BackendSelection::Cpu,
            corpus_paths: vec![],
            jsonl_train_path: None,
            jsonl_eval_path: None,
            corpus_name: None,
            corpus_text_field: "text".to_string(),
            output_dir: None,
            seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE),
            batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
            steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            full_train_pass: true,
            full_eval_pass: true,
            eval_holdout_every: DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: DEFAULT_V3A_SMOKE_SEED,
            data_seed: None,
            variant: VariantSelection::All,
            goe_routing_mode: GoeRoutingModeSelection::TokenLocalRouter,
            goe_graph_topology: GoeGraphTopologySelection::TwoNodeLine,
            ledger_path: None,
            run_label: None,
            output: OutputFormat::Table,
        };
        let overridden = apply_full_pass_overrides(args, &corpus).unwrap();
        assert_eq!(
            overridden.steps,
            stats.train_sequences.div_ceil(DEFAULT_V3A_SMOKE_BATCH_SIZE)
        );
        assert_eq!(
            overridden.eval_batches,
            stats.eval_sequences.div_ceil(DEFAULT_V3A_SMOKE_BATCH_SIZE)
        );
    }

    #[test]
    fn selected_goe_controller_respects_structure_flags() {
        let args = CliArgs {
            backend: super::BackendSelection::Cpu,
            corpus_paths: vec![],
            jsonl_train_path: None,
            jsonl_eval_path: None,
            corpus_name: None,
            corpus_text_field: "text".to_string(),
            output_dir: None,
            seq_len: DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE),
            batch_size: DEFAULT_V3A_SMOKE_BATCH_SIZE,
            steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            full_train_pass: false,
            full_eval_pass: false,
            eval_holdout_every: DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: DEFAULT_V3A_SMOKE_SEED,
            data_seed: None,
            variant: VariantSelection::All,
            goe_routing_mode: GoeRoutingModeSelection::UniformAverage,
            goe_graph_topology: GoeGraphTopologySelection::None,
            ledger_path: None,
            run_label: None,
            output: OutputFormat::Table,
        };
        let controller = selected_goe_controller(&args).unwrap();
        assert_eq!(
            controller.routing_mode,
            GraphOfExpertsRoutingMode::UniformAverage
        );
        assert_eq!(controller.topology, GraphOfExpertsTopology::None);
    }

    #[test]
    fn experiment_note_omits_dreegmor_structure_for_baseline_only_runs() {
        let note = experiment_note(
            "cpu",
            &VariantSelection::AttentionOnly,
            &GraphOfExpertsControllerSpec::two_channel_line(),
        );
        assert!(!note.contains("dreegmor_structure="));
    }
}
