use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "cuda")]
use fractal_core::cuda_device;
use fractal_core::{
    initialize_metal_runtime, phase1_hybrid_attention_baseline_matrix,
    phase1_p20_candidate_variant, phase1_p21_candidate_variant, phase1_p22_candidate_variant,
    phase1_p23_candidate_variant, phase1_p2_candidate_variant,
    phase1_p2_interface_candidate_variant, CpuTrainBackend, HybridAttentionVariantSpec,
    MetalTrainBackend, PrimitiveHybridNormMode, PrimitiveHybridPrimitive,
    PrimitiveHybridReadoutMode, PrimitiveHybridResidualMode,
};
use fractal_eval_private::{
    append_v3a_results_ledger_entry, byte_level_smoke_corpus_stats_from_source,
    default_v3a_fineweb_stage0_canary_corpus_source,
    default_v3a_fineweb_stage0_local_bench_9row_v1_corpus_source,
    resolve_requested_v3a_results_ledger_path, run_attention_only_hybrid_attention_smoke_train,
    run_primitive_hybrid_attention_smoke_train, run_reference_ssm_hybrid_attention_smoke_train,
    ByteLevelSmokeCorpusSource, HybridAttentionCudaDeviceMemoryMetrics,
    HybridAttentionExecutionBackend, HybridAttentionMatrixLedgerReport,
    HybridAttentionMatrixVariantOutcome, HybridAttentionSmokeTrainConfig, V3aResultsLedgerEntry,
    DEFAULT_V3A_SMOKE_BATCH_SIZE, DEFAULT_V3A_SMOKE_EVAL_BATCHES,
    DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V3A_SMOKE_LEARNING_RATE, DEFAULT_V3A_SMOKE_SEED,
    DEFAULT_V3A_SMOKE_SEQ_LEN, DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
};
use serde::{Deserialize, Serialize};

fn main() {
    if let Err(error) = run() {
        eprintln!("v3a-hybrid-attention-matrix: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let args = apply_benchmark_profile_overrides(args)?;
    if matches!(args.primitive_profile, PrimitiveProfile::P1)
        && !matches!(
            args.primitive_residual_profile,
            PrimitiveResidualProfile::Plain
        )
    {
        return Err(
            "--primitive-residual-profile may not be changed when --primitive-profile p1"
                .to_owned(),
        );
    }
    if matches!(args.primitive_profile, PrimitiveProfile::P1)
        && !matches!(
            args.primitive_readout_profile,
            PrimitiveReadoutProfile::Direct
        )
    {
        return Err(
            "--primitive-readout-profile may not be changed when --primitive-profile p1".to_owned(),
        );
    }
    if matches!(args.primitive_profile, PrimitiveProfile::P1)
        && !matches!(
            args.primitive_norm_profile,
            PrimitiveNormProfile::PreNormOnly
        )
    {
        return Err(
            "--primitive-norm-profile may not be changed when --primitive-profile p1".to_owned(),
        );
    }
    if matches!(args.primitive_profile, PrimitiveProfile::P1)
        && !matches!(
            args.primitive_wrapper_profile,
            PrimitiveWrapperProfile::Standard
        )
    {
        return Err(
            "--primitive-wrapper-profile may not be changed when --primitive-profile p1".to_owned(),
        );
    }
    let base_note = primitive_lane_note(
        args.primitive_profile,
        args.primitive_residual_profile,
        args.primitive_readout_profile,
        args.primitive_norm_profile,
        args.primitive_wrapper_profile,
    );
    let benchmark_prefix = args
        .benchmark_profile
        .map(|profile| format!("benchmark={}. {} ", profile.as_str(), profile.note()))
        .unwrap_or_default();
    let matrix_note = if args.isolate_variants && args.variant.includes_multiple_variants() {
        format!(
            "{benchmark_prefix}backend={}. {base_note} Variants are executed in isolated child processes for fairer throughput and memory comparison.",
            args.backend.as_str()
        )
    } else {
        format!(
            "{benchmark_prefix}backend={}. {base_note}",
            args.backend.as_str()
        )
    };
    let corpus_source = resolve_corpus_source(&args, &repo_root)?;
    let args = apply_full_pass_overrides(args, &corpus_source)?;
    let output_dir = args
        .output_dir
        .clone()
        .unwrap_or_else(|| default_output_dir(&repo_root));
    let matrix = phase1_hybrid_attention_baseline_matrix();
    let primitive_variant = selected_primitive_variant(
        &args.primitive_profile,
        &args.primitive_residual_profile,
        &args.primitive_readout_profile,
        &args.primitive_norm_profile,
        &args.primitive_wrapper_profile,
        &matrix,
    );
    let variants = if args.isolate_variants && args.variant.includes_multiple_variants() {
        run_selected_variants_isolated(&args, &repo_root, &output_dir, &matrix, &primitive_variant)?
    } else {
        run_selected_variants_in_process(
            &args,
            corpus_source,
            &output_dir,
            &matrix,
            &primitive_variant,
        )?
    };

    let ledger_report = HybridAttentionMatrixLedgerReport {
        requested_variant: args.variant.as_str().to_owned(),
        note: matrix_note.clone(),
        variants: variants.clone(),
    };
    let rendered = render_report(
        &RenderedMatrixReport {
            note: &matrix_note,
            variants,
        },
        args.output,
    )?;
    maybe_append_ledger_entry(
        resolve_requested_v3a_results_ledger_path(&repo_root, args.ledger_path.as_deref())
            .map_err(|error| format!("failed to resolve v3a results ledger path: {error}"))?,
        &ledger_report,
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
                .unwrap_or("v3a-jsonl-split")
                .to_owned()
        });
        Ok(ByteLevelSmokeCorpusSource::jsonl_text_splits(
            corpus_name,
            train_path.clone(),
            eval_path.clone(),
            args.corpus_text_field.clone(),
        ))
    } else if args.corpus_paths.is_empty() {
        match args.benchmark_profile {
            Some(BenchmarkProfile::CudaFaithfulSmallV1) => {
                default_v3a_fineweb_stage0_local_bench_9row_v1_corpus_source(repo_root).map_err(
                    |error| {
                        format!(
                            "failed to resolve default v3a FineWeb faithful-small corpus: {error}"
                        )
                    },
                )
            }
            None => default_v3a_fineweb_stage0_canary_corpus_source(repo_root).map_err(|error| {
                format!("failed to resolve default v3a FineWeb smoke corpus: {error}")
            }),
        }
    } else {
        Ok(ByteLevelSmokeCorpusSource::raw_files(
            args.corpus_paths.clone(),
        ))
    }
}

fn apply_benchmark_profile_overrides(mut args: CliArgs) -> Result<CliArgs, String> {
    let Some(profile) = args.benchmark_profile else {
        return Ok(args);
    };
    match profile {
        BenchmarkProfile::CudaFaithfulSmallV1 => {
            if !args.corpus_paths.is_empty()
                || args.jsonl_train_path.is_some()
                || args.jsonl_eval_path.is_some()
            {
                return Err(
                    "--benchmark-profile cuda-faithful-small-v1 may not be combined with explicit corpus path flags"
                        .to_owned(),
                );
            }
            if args.steps != DEFAULT_V3A_SMOKE_TRAIN_STEPS && !args.full_train_pass {
                return Err(
                    "--benchmark-profile cuda-faithful-small-v1 may not be combined with explicit --steps"
                        .to_owned(),
                );
            }
            if args.eval_batches != DEFAULT_V3A_SMOKE_EVAL_BATCHES && !args.full_eval_pass {
                return Err(
                    "--benchmark-profile cuda-faithful-small-v1 may not be combined with explicit --eval-batches"
                        .to_owned(),
                );
            }
            args.full_train_pass = true;
            args.full_eval_pass = true;
            args.batch_size = DEFAULT_V3A_SMOKE_BATCH_SIZE;
            args.seq_len = DEFAULT_V3A_SMOKE_SEQ_LEN;
            args.window_stride = Some(DEFAULT_V3A_SMOKE_WINDOW_STRIDE);
            args.eval_holdout_every = DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY;
        }
    }
    Ok(args)
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

fn smoke_config(
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: PathBuf,
    variant: fractal_core::HybridAttentionVariantSpec,
    args: &CliArgs,
) -> HybridAttentionSmokeTrainConfig {
    let mut config = HybridAttentionSmokeTrainConfig::new(corpus_source, output_dir, variant);
    config.benchmark_name = args
        .benchmark_profile
        .map(|profile| profile.as_str().to_owned());
    config.execution_backend = args.backend.execution_backend();
    config.cuda_device_index = match args.backend {
        BackendSelection::Cuda => Some(args.cuda_device),
        BackendSelection::Cpu | BackendSelection::Metal => None,
    };
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

fn primitive_lane_note(
    primitive_profile: PrimitiveProfile,
    primitive_residual_profile: PrimitiveResidualProfile,
    primitive_readout_profile: PrimitiveReadoutProfile,
    primitive_norm_profile: PrimitiveNormProfile,
    primitive_wrapper_profile: PrimitiveWrapperProfile,
) -> &'static str {
    let interface_is_default =
        matches!(primitive_residual_profile, PrimitiveResidualProfile::Plain)
            && matches!(primitive_readout_profile, PrimitiveReadoutProfile::Direct)
            && matches!(primitive_norm_profile, PrimitiveNormProfile::PreNormOnly)
            && matches!(primitive_wrapper_profile, PrimitiveWrapperProfile::Standard);
    match primitive_profile {
        PrimitiveProfile::P1 => {
            "Path 1 baseline matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane uses the historical P1 contractive primitive on the shared tracked surface."
        }
        PrimitiveProfile::P20 => {
            if interface_is_default {
                "Path 1 primitive-quality matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the base-width direct-state corner of the width/readout sweep with the incumbent wrapper held fixed on the shared tracked surface."
            } else {
                "Path 1 contender sweep: attention-only and reference-ssm-hybrid remain frozen, while the primitive lane tests the base-width direct-state family under a non-incumbent interface configuration on the shared tracked surface."
            }
        }
        PrimitiveProfile::P2 => {
            if !matches!(primitive_wrapper_profile, PrimitiveWrapperProfile::Standard) {
                match primitive_wrapper_profile {
                    PrimitiveWrapperProfile::Standard => unreachable!(),
                    PrimitiveWrapperProfile::MambaRms => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with a Rust Mamba-style RMS wrapper on the same schedule, budget, and ledger surface."
                    }
                }
            } else if !matches!(primitive_norm_profile, PrimitiveNormProfile::PreNormOnly) {
                match primitive_norm_profile {
                    PrimitiveNormProfile::PreNormOnly => unreachable!(),
                    PrimitiveNormProfile::PostReadoutNorm => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with a post-readout normalization wrapper on the same schedule, budget, and ledger surface."
                    }
                    PrimitiveNormProfile::ResidualRenorm => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with a residual-side renormalization wrapper on the same schedule, budget, and ledger surface."
                    }
                }
            } else if !matches!(primitive_residual_profile, PrimitiveResidualProfile::Plain) {
                match primitive_residual_profile {
                    PrimitiveResidualProfile::Plain => unreachable!(),
                    PrimitiveResidualProfile::Scaled => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with the scaled residual interface on the same schedule, budget, and ledger surface."
                    }
                    PrimitiveResidualProfile::Gated => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with the gated residual interface on the same schedule, budget, and ledger surface."
                    }
                }
            } else {
                match primitive_readout_profile {
                    PrimitiveReadoutProfile::Direct => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender on the same schedule, budget, and ledger surface."
                    }
                    PrimitiveReadoutProfile::Projected => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with a projected readout handoff on the same schedule, budget, and ledger surface."
                    }
                    PrimitiveReadoutProfile::ProjectedNorm => {
                        "Path 1 proving matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the first P2 contender with a projected-plus-normalized readout handoff on the same schedule, budget, and ledger surface."
                    }
                }
            }
        }
        PrimitiveProfile::P21 => {
            if interface_is_default {
                "Path 1 primitive-quality matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in P2.1 with wider latent state only and the incumbent wrapper held fixed on the shared tracked surface."
            } else {
                "Path 1 contender sweep: attention-only and reference-ssm-hybrid remain frozen, while the primitive lane tests the wide-latent direct-state family under a non-incumbent interface configuration on the shared tracked surface."
            }
        }
        PrimitiveProfile::P23 => {
            if interface_is_default {
                "Path 1 primitive-quality matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in P2.3 with richer blended carry dynamics and the incumbent wrapper held fixed on the shared tracked surface."
            } else {
                "Path 1 contender sweep: attention-only and reference-ssm-hybrid remain frozen, while the primitive lane tests the blended-carry state-dynamics family under a non-incumbent interface configuration on the shared tracked surface."
            }
        }
        PrimitiveProfile::P22 => {
            if interface_is_default {
                "Path 1 primitive-quality matrix: attention-only and reference-ssm-hybrid remain fixed, while the primitive lane swaps in the wide-latent explicit-readout corner of the width/readout sweep with the incumbent wrapper held fixed on the shared tracked surface."
            } else {
                "Path 1 contender sweep: attention-only and reference-ssm-hybrid remain frozen, while the primitive lane tests the wide-latent explicit-readout family under a non-incumbent interface configuration on the shared tracked surface."
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkProfile {
    CudaFaithfulSmallV1,
}

impl BenchmarkProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cuda-faithful-small-v1" => Ok(Self::CudaFaithfulSmallV1),
            _ => Err(format!("unknown benchmark profile: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::CudaFaithfulSmallV1 => "cuda-faithful-small-v1",
        }
    }

    fn note(self) -> &'static str {
        match self {
            Self::CudaFaithfulSmallV1 => {
                "Benchmark profile pins the larger frozen 9-row FineWeb slice with full-train/full-eval pass semantics as the CUDA-faithful small proving surface."
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct CliArgs {
    benchmark_profile: Option<BenchmarkProfile>,
    backend: BackendSelection,
    cuda_device: usize,
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
    variant: VariantSelection,
    primitive_profile: PrimitiveProfile,
    primitive_residual_profile: PrimitiveResidualProfile,
    primitive_readout_profile: PrimitiveReadoutProfile,
    primitive_norm_profile: PrimitiveNormProfile,
    primitive_wrapper_profile: PrimitiveWrapperProfile,
    isolate_variants: bool,
    ledger_path: Option<String>,
    run_label: Option<String>,
    output: OutputFormat,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut benchmark_profile = None;
        let mut backend = BackendSelection::Cpu;
        let mut cuda_device = 0usize;
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
        let mut variant = VariantSelection::All;
        let mut primitive_profile = PrimitiveProfile::P1;
        let mut primitive_residual_profile = PrimitiveResidualProfile::Plain;
        let mut primitive_readout_profile = PrimitiveReadoutProfile::Direct;
        let mut primitive_norm_profile = PrimitiveNormProfile::PreNormOnly;
        let mut primitive_wrapper_profile = PrimitiveWrapperProfile::Standard;
        let mut isolate_variants = true;
        let mut ledger_path = None;
        let mut run_label = None;
        let mut output = OutputFormat::Table;
        let mut show_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--benchmark-profile" => {
                    benchmark_profile =
                        Some(BenchmarkProfile::parse(&iter.next().ok_or_else(|| {
                            "--benchmark-profile requires a value".to_owned()
                        })?)?);
                }
                "--backend" => {
                    backend = BackendSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--backend requires a value".to_owned())?,
                    )?;
                }
                "--cuda-device" => {
                    cuda_device = parse_usize(
                        iter.next()
                            .ok_or_else(|| "--cuda-device requires a value".to_owned())?,
                        "--cuda-device",
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
                "--variant" => {
                    variant = VariantSelection::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--variant requires a value".to_owned())?,
                    )?;
                }
                "--primitive-profile" => {
                    primitive_profile = PrimitiveProfile::parse(
                        &iter
                            .next()
                            .ok_or_else(|| "--primitive-profile requires a value".to_owned())?,
                    )?;
                }
                "--primitive-residual-profile" => {
                    primitive_residual_profile =
                        PrimitiveResidualProfile::parse(&iter.next().ok_or_else(|| {
                            "--primitive-residual-profile requires a value".to_owned()
                        })?)?;
                }
                "--primitive-readout-profile" => {
                    primitive_readout_profile =
                        PrimitiveReadoutProfile::parse(&iter.next().ok_or_else(|| {
                            "--primitive-readout-profile requires a value".to_owned()
                        })?)?;
                }
                "--primitive-norm-profile" => {
                    primitive_norm_profile =
                        PrimitiveNormProfile::parse(&iter.next().ok_or_else(|| {
                            "--primitive-norm-profile requires a value".to_owned()
                        })?)?;
                }
                "--primitive-wrapper-profile" => {
                    primitive_wrapper_profile =
                        PrimitiveWrapperProfile::parse(&iter.next().ok_or_else(|| {
                            "--primitive-wrapper-profile requires a value".to_owned()
                        })?)?;
                }
                "--shared-process" => isolate_variants = false,
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
            benchmark_profile,
            backend,
            cuda_device,
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
            variant,
            primitive_profile,
            primitive_residual_profile,
            primitive_readout_profile,
            primitive_norm_profile,
            primitive_wrapper_profile,
            isolate_variants,
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
    Cuda,
}

impl BackendSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(format!("unknown backend selection: {value}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    fn execution_backend(self) -> HybridAttentionExecutionBackend {
        match self {
            Self::Cpu => HybridAttentionExecutionBackend::Cpu,
            Self::Metal => HybridAttentionExecutionBackend::Metal,
            Self::Cuda => HybridAttentionExecutionBackend::Cuda,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveProfile {
    P1,
    P20,
    P2,
    P23,
    P21,
    P22,
}

impl PrimitiveProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "p1" => Ok(Self::P1),
            "p2-0" | "p2-base-direct" => Ok(Self::P20),
            "p2" => Ok(Self::P2),
            "p2-3" => Ok(Self::P23),
            "p2-1" => Ok(Self::P21),
            "p2-2" => Ok(Self::P22),
            _ => Err(format!("unknown primitive profile: {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::P1 => "p1",
            Self::P20 => "p2-0",
            Self::P2 => "p2",
            Self::P23 => "p2-3",
            Self::P21 => "p2-1",
            Self::P22 => "p2-2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveResidualProfile {
    Plain,
    Scaled,
    Gated,
}

impl PrimitiveResidualProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "plain" => Ok(Self::Plain),
            "scaled" => Ok(Self::Scaled),
            "gated" => Ok(Self::Gated),
            _ => Err(format!("unknown primitive residual profile: {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Scaled => "scaled",
            Self::Gated => "gated",
        }
    }

    fn residual_mode(&self) -> PrimitiveHybridResidualMode {
        match self {
            Self::Plain => PrimitiveHybridResidualMode::PlainAdd,
            Self::Scaled => PrimitiveHybridResidualMode::ScaledAdd,
            Self::Gated => PrimitiveHybridResidualMode::GatedAdd,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveReadoutProfile {
    Direct,
    Projected,
    ProjectedNorm,
}

impl PrimitiveReadoutProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "direct" => Ok(Self::Direct),
            "projected" => Ok(Self::Projected),
            "projected-norm" => Ok(Self::ProjectedNorm),
            _ => Err(format!("unknown primitive readout profile: {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Projected => "projected",
            Self::ProjectedNorm => "projected-norm",
        }
    }

    fn readout_mode(&self) -> PrimitiveHybridReadoutMode {
        match self {
            Self::Direct => PrimitiveHybridReadoutMode::Direct,
            Self::Projected => PrimitiveHybridReadoutMode::Projected,
            Self::ProjectedNorm => PrimitiveHybridReadoutMode::ProjectedNorm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveNormProfile {
    PreNormOnly,
    PostReadoutNorm,
    ResidualRenorm,
}

impl PrimitiveNormProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "pre-norm-only" => Ok(Self::PreNormOnly),
            "post-readout-norm" => Ok(Self::PostReadoutNorm),
            "residual-renorm" => Ok(Self::ResidualRenorm),
            _ => Err(format!("unknown primitive norm profile: {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::PreNormOnly => "pre-norm-only",
            Self::PostReadoutNorm => "post-readout-norm",
            Self::ResidualRenorm => "residual-renorm",
        }
    }

    fn norm_mode(&self) -> PrimitiveHybridNormMode {
        match self {
            Self::PreNormOnly => PrimitiveHybridNormMode::PreNormOnly,
            Self::PostReadoutNorm => PrimitiveHybridNormMode::PostReadoutNorm,
            Self::ResidualRenorm => PrimitiveHybridNormMode::ResidualSideRenorm,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveWrapperProfile {
    Standard,
    MambaRms,
}

impl PrimitiveWrapperProfile {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "standard" => Ok(Self::Standard),
            "mamba-rms" => Ok(Self::MambaRms),
            _ => Err(format!("unknown primitive wrapper profile: {value}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::MambaRms => "mamba-rms",
        }
    }

    fn wrapper_mode(&self) -> fractal_core::PrimitiveHybridWrapperSymmetryMode {
        match self {
            Self::Standard => fractal_core::PrimitiveHybridWrapperSymmetryMode::Standard,
            Self::MambaRms => fractal_core::PrimitiveHybridWrapperSymmetryMode::MambaRms,
        }
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

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::AttentionOnly => "attention-only",
            Self::ReferenceSsmHybrid => "reference-ssm-hybrid",
            Self::PrimitiveHybrid => "primitive-hybrid",
        }
    }

    fn includes_multiple_variants(self) -> bool {
        matches!(self, Self::All)
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

#[derive(Debug, Serialize, Deserialize)]
struct RenderedMatrixReport<'a> {
    note: &'a str,
    variants: Vec<HybridAttentionMatrixVariantOutcome>,
}

#[derive(Debug, Deserialize)]
struct OwnedRenderedMatrixReport {
    variants: Vec<HybridAttentionMatrixVariantOutcome>,
}

fn maybe_append_ledger_entry(
    ledger_path: Option<PathBuf>,
    report: &HybridAttentionMatrixLedgerReport,
    run_label: Option<&str>,
) -> Result<(), String> {
    let Some(ledger_path) = ledger_path else {
        return Ok(());
    };
    let entry = V3aResultsLedgerEntry::path1_matrix_run(
        "v3a_hybrid_attention_matrix",
        &report.note,
        report,
        run_label.map(str::to_owned),
    )
    .map_err(|error| format!("failed to build v3a results ledger entry: {error}"))?;
    append_v3a_results_ledger_entry(&ledger_path, &entry)
        .map_err(|error| format!("failed to append v3a results ledger entry: {error}"))
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
                    "  backend={} implementation={}",
                    report.config.execution_backend.as_str(),
                    report.implementation_kind.as_str(),
                );
                if let Some(benchmark_name) = &report.config.benchmark_name {
                    let _ = writeln!(output, "  benchmark={benchmark_name}");
                }
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
                    "  train_tok_s={:.2} overall_tok_s={:.2} mem_metric={} mem_delta_mb={:.2}",
                    report.runtime.train_tokens_per_second,
                    report.runtime.overall_tokens_per_second,
                    report.runtime.process_memory_metric.as_str(),
                    report.runtime.peak_process_memory_delta_bytes as f64 / (1024.0 * 1024.0),
                );
                if let Some(cuda_memory) = &report.runtime.cuda_device_memory {
                    write_cuda_memory_line(&mut output, cuda_memory);
                }
                let _ = writeln!(
                    output,
                    "  total_ms={:.1} train_ms={:.1} eval_ms={:.1}+{:.1}",
                    report.runtime.total_wall_time_ms,
                    report.runtime.train_wall_time_ms,
                    report.runtime.initial_eval_wall_time_ms,
                    report.runtime.final_eval_wall_time_ms,
                );
                let _ = writeln!(
                    output,
                    "  seed={} report={}",
                    report.config.seed,
                    report.report_path.display()
                );
            }
            HybridAttentionMatrixVariantOutcome::Skipped { label, reason, .. } => {
                let _ = writeln!(output, "- {} (skipped)", label);
                let _ = writeln!(output, "  reason={reason}");
            }
            HybridAttentionMatrixVariantOutcome::RequiredMissing { label, reason, .. } => {
                let _ = writeln!(output, "- {} (required-missing)", label);
                let _ = writeln!(output, "  reason={reason}");
            }
        }
    }
    output
}

fn write_cuda_memory_line(
    output: &mut String,
    cuda_memory: &HybridAttentionCudaDeviceMemoryMetrics,
) {
    let _ = writeln!(
        output,
        "  cuda_device={} cuda_mem_metric={} cuda_peak_mb={:.2} cuda_delta_mb={:.2}",
        cuda_memory.device_index,
        cuda_memory.memory_metric.as_str(),
        cuda_memory.peak_used_bytes as f64 / (1024.0 * 1024.0),
        cuda_memory.peak_used_delta_bytes as f64 / (1024.0 * 1024.0),
    );
}

fn parse_positive_usize(value: String, flag: &str) -> Result<usize, String> {
    let parsed = parse_usize(value, flag)?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_usize(value: String, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))
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

fn selected_primitive_variant(
    primitive_profile: &PrimitiveProfile,
    primitive_residual_profile: &PrimitiveResidualProfile,
    primitive_readout_profile: &PrimitiveReadoutProfile,
    primitive_norm_profile: &PrimitiveNormProfile,
    primitive_wrapper_profile: &PrimitiveWrapperProfile,
    matrix: &fractal_core::HybridAttentionBaselineMatrix,
) -> HybridAttentionVariantSpec {
    let primitive_kind = match primitive_profile {
        PrimitiveProfile::P1 => PrimitiveHybridPrimitive::P1Contractive,
        PrimitiveProfile::P20 => PrimitiveHybridPrimitive::P20RotaryStateOutput,
        PrimitiveProfile::P2 => PrimitiveHybridPrimitive::P2RotaryReadout,
        PrimitiveProfile::P23 => PrimitiveHybridPrimitive::P23RotaryCarryBlendReadout,
        PrimitiveProfile::P21 => PrimitiveHybridPrimitive::P21WideLatent,
        PrimitiveProfile::P22 => PrimitiveHybridPrimitive::P22WideLatentReadout,
    };
    let interface_is_default =
        matches!(primitive_residual_profile, PrimitiveResidualProfile::Plain)
            && matches!(primitive_readout_profile, PrimitiveReadoutProfile::Direct)
            && matches!(primitive_norm_profile, PrimitiveNormProfile::PreNormOnly)
            && matches!(primitive_wrapper_profile, PrimitiveWrapperProfile::Standard);
    match primitive_profile {
        PrimitiveProfile::P1 => matrix.primitive_hybrid.clone(),
        PrimitiveProfile::P20 => {
            if interface_is_default {
                phase1_p20_candidate_variant()
            } else {
                phase1_p2_interface_candidate_variant(
                    primitive_kind,
                    primitive_residual_profile.residual_mode(),
                    primitive_readout_profile.readout_mode(),
                    primitive_norm_profile.norm_mode(),
                    primitive_wrapper_profile.wrapper_mode(),
                )
            }
        }
        PrimitiveProfile::P2 => {
            if interface_is_default {
                phase1_p2_candidate_variant()
            } else {
                phase1_p2_interface_candidate_variant(
                    primitive_kind,
                    primitive_residual_profile.residual_mode(),
                    primitive_readout_profile.readout_mode(),
                    primitive_norm_profile.norm_mode(),
                    primitive_wrapper_profile.wrapper_mode(),
                )
            }
        }
        PrimitiveProfile::P23 => {
            if interface_is_default {
                phase1_p23_candidate_variant()
            } else {
                phase1_p2_interface_candidate_variant(
                    primitive_kind,
                    primitive_residual_profile.residual_mode(),
                    primitive_readout_profile.readout_mode(),
                    primitive_norm_profile.norm_mode(),
                    primitive_wrapper_profile.wrapper_mode(),
                )
            }
        }
        PrimitiveProfile::P21 => {
            if interface_is_default {
                phase1_p21_candidate_variant()
            } else {
                phase1_p2_interface_candidate_variant(
                    primitive_kind,
                    primitive_residual_profile.residual_mode(),
                    primitive_readout_profile.readout_mode(),
                    primitive_norm_profile.norm_mode(),
                    primitive_wrapper_profile.wrapper_mode(),
                )
            }
        }
        PrimitiveProfile::P22 => {
            if interface_is_default {
                phase1_p22_candidate_variant()
            } else {
                phase1_p2_interface_candidate_variant(
                    primitive_kind,
                    primitive_residual_profile.residual_mode(),
                    primitive_readout_profile.readout_mode(),
                    primitive_norm_profile.norm_mode(),
                    primitive_wrapper_profile.wrapper_mode(),
                )
            }
        }
    }
}

fn run_selected_variants_in_process(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
    matrix: &fractal_core::HybridAttentionBaselineMatrix,
    primitive_variant: &HybridAttentionVariantSpec,
) -> Result<Vec<HybridAttentionMatrixVariantOutcome>, String> {
    match args.backend {
        BackendSelection::Cpu => {
            let device = <CpuTrainBackend as burn::tensor::backend::Backend>::Device::default();
            run_selected_variants_with_backend::<CpuTrainBackend>(
                args,
                corpus_source,
                output_dir,
                matrix,
                primitive_variant,
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
                matrix,
                primitive_variant,
                &device,
            )
        }
        BackendSelection::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let device = cuda_device(args.cuda_device);
                run_selected_variants_with_backend::<fractal_core::CandleF32TrainBackend>(
                    args,
                    corpus_source,
                    output_dir,
                    matrix,
                    primitive_variant,
                    &device,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = corpus_source;
                let _ = output_dir;
                let _ = matrix;
                let _ = primitive_variant;
                Err(
                    "cuda backend requested, but v3a-hybrid-attention-matrix was not built with --features cuda"
                        .to_owned(),
                )
            }
        }
    }
}

fn run_selected_variants_with_backend<B>(
    args: &CliArgs,
    corpus_source: ByteLevelSmokeCorpusSource,
    output_dir: &Path,
    matrix: &fractal_core::HybridAttentionBaselineMatrix,
    primitive_variant: &HybridAttentionVariantSpec,
    device: &<B as burn::tensor::backend::Backend>::Device,
) -> Result<Vec<HybridAttentionMatrixVariantOutcome>, String>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    let mut variants = Vec::new();
    if args.variant.includes_attention_only() {
        variants.push(
            run_attention_only_hybrid_attention_smoke_train::<B>(
                smoke_config(
                    corpus_source.clone(),
                    output_dir.join("attention-only"),
                    matrix.attention_only.clone(),
                    args,
                ),
                device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run attention-only baseline: {error}"))?,
        );
    } else {
        variants.push(HybridAttentionMatrixVariantOutcome::Skipped {
            label: matrix.attention_only.label.to_owned(),
            kind: matrix.attention_only.kind,
            reason: "not requested by --variant selection".to_owned(),
        });
    }
    if args.variant.includes_reference_ssm_hybrid() {
        variants.push(
            run_reference_ssm_hybrid_attention_smoke_train::<B>(
                smoke_config(
                    corpus_source.clone(),
                    output_dir.join("reference-ssm-hybrid"),
                    matrix.reference_ssm_hybrid.clone(),
                    args,
                ),
                device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run reference-ssm-hybrid baseline: {error}"))?,
        );
    } else {
        variants.push(HybridAttentionMatrixVariantOutcome::Skipped {
            label: matrix.reference_ssm_hybrid.label.to_owned(),
            kind: matrix.reference_ssm_hybrid.kind,
            reason: "not requested by --variant selection".to_owned(),
        });
    }
    if args.variant.includes_primitive_hybrid() {
        variants.push(
            run_primitive_hybrid_attention_smoke_train::<B>(
                smoke_config(
                    corpus_source,
                    output_dir.join("primitive-hybrid"),
                    primitive_variant.clone(),
                    args,
                ),
                device,
            )
            .map(|report| HybridAttentionMatrixVariantOutcome::Executed(Box::new(report)))
            .map_err(|error| format!("failed to run primitive-hybrid baseline: {error}"))?,
        );
    } else {
        variants.push(HybridAttentionMatrixVariantOutcome::Skipped {
            label: primitive_variant.label.to_owned(),
            kind: primitive_variant.kind,
            reason: "not requested by --variant selection".to_owned(),
        });
    }
    Ok(variants)
}

fn run_selected_variants_isolated(
    args: &CliArgs,
    repo_root: &Path,
    output_dir: &Path,
    matrix: &fractal_core::HybridAttentionBaselineMatrix,
    primitive_variant: &HybridAttentionVariantSpec,
) -> Result<Vec<HybridAttentionMatrixVariantOutcome>, String> {
    Ok(vec![
        run_isolated_variant(
            repo_root,
            output_dir,
            args,
            VariantSelection::AttentionOnly,
            &matrix.attention_only.label,
        )?,
        run_isolated_variant(
            repo_root,
            output_dir,
            args,
            VariantSelection::ReferenceSsmHybrid,
            &matrix.reference_ssm_hybrid.label,
        )?,
        run_isolated_variant(
            repo_root,
            output_dir,
            args,
            VariantSelection::PrimitiveHybrid,
            &primitive_variant.label,
        )?,
    ])
}

fn run_isolated_variant(
    repo_root: &Path,
    output_dir: &Path,
    args: &CliArgs,
    variant: VariantSelection,
    expected_label: &str,
) -> Result<HybridAttentionMatrixVariantOutcome, String> {
    let mut command = Command::new(std::env::current_exe().map_err(|error| {
        format!("failed to resolve current v3a matrix executable for isolation: {error}")
    })?);
    command
        .args(
            args.benchmark_profile
                .map(|profile| {
                    vec![
                        "--benchmark-profile".to_owned(),
                        profile.as_str().to_owned(),
                    ]
                })
                .unwrap_or_default(),
        )
        .arg("--backend")
        .arg(args.backend.as_str())
        .arg("--cuda-device")
        .arg(args.cuda_device.to_string())
        .arg("--variant")
        .arg(variant.as_str())
        .arg("--primitive-profile")
        .arg(args.primitive_profile.as_str())
        .arg("--primitive-residual-profile")
        .arg(args.primitive_residual_profile.as_str())
        .arg("--primitive-readout-profile")
        .arg(args.primitive_readout_profile.as_str())
        .arg("--primitive-norm-profile")
        .arg(args.primitive_norm_profile.as_str())
        .arg("--primitive-wrapper-profile")
        .arg(args.primitive_wrapper_profile.as_str())
        .arg("--eval-holdout-every")
        .arg(args.eval_holdout_every.to_string())
        .arg("--batch-size")
        .arg(args.batch_size.to_string())
        .arg("--seq-len")
        .arg(args.seq_len.to_string())
        .arg("--window-stride")
        .arg(args.window_stride.unwrap_or(args.seq_len).to_string())
        .arg("--learning-rate")
        .arg(args.learning_rate.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--output")
        .arg("json")
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--shared-process");

    if args.full_train_pass {
        command.arg("--full-train-pass");
    } else {
        command.arg("--steps").arg(args.steps.to_string());
    }
    if args.full_eval_pass {
        command.arg("--full-eval-pass");
    } else {
        command
            .arg("--eval-batches")
            .arg(args.eval_batches.to_string());
    }

    for corpus_path in &args.corpus_paths {
        command.arg("--corpus-path").arg(corpus_path);
    }
    if let Some(train_path) = &args.jsonl_train_path {
        command.arg("--jsonl-train-path").arg(train_path);
    }
    if let Some(eval_path) = &args.jsonl_eval_path {
        command.arg("--jsonl-eval-path").arg(eval_path);
    }
    if let Some(corpus_name) = &args.corpus_name {
        command.arg("--corpus-name").arg(corpus_name);
    }
    if args.corpus_text_field != "text" {
        command
            .arg("--corpus-text-field")
            .arg(&args.corpus_text_field);
    }

    command.current_dir(repo_root);
    let output = command.output().map_err(|error| {
        format!("failed to launch isolated {expected_label} child run: {error}")
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "isolated {expected_label} child run failed with status {}: {}",
            output.status,
            stderr.trim()
        ));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|error| {
        format!("isolated {expected_label} child output was not valid UTF-8: {error}")
    })?;
    let rendered: OwnedRenderedMatrixReport = serde_json::from_str(&stdout).map_err(|error| {
        format!("failed to parse isolated {expected_label} child JSON output: {error}")
    })?;
    rendered
        .variants
        .into_iter()
        .find(|variant_outcome| match variant_outcome {
            HybridAttentionMatrixVariantOutcome::Executed(report) => {
                report.config.variant.label == expected_label
            }
            HybridAttentionMatrixVariantOutcome::Skipped { label, .. } => label == expected_label,
            HybridAttentionMatrixVariantOutcome::RequiredMissing { label, .. } => {
                label == expected_label
            }
        })
        .ok_or_else(|| {
            format!("isolated {expected_label} child output did not contain the expected variant")
        })
}

fn usage() -> String {
    format!(
        concat!(
            "Usage: cargo run --bin v3a-hybrid-attention-matrix -- [options]\n\n",
            "Options:\n",
            "  --benchmark-profile <name>   One of: cuda-faithful-small-v1 (pins the larger frozen 9-row FineWeb full-pass CUDA-faithful surface)\n",
            "  --backend <cpu|metal|cuda>  Execution backend for this run (default: cpu)\n",
            "  --cuda-device <n>           CUDA device index when --backend cuda (default: 0)\n",
            "  --corpus-path <path>         Override the default frozen FineWeb canary corpus with raw byte-level files (repeatable)\n",
            "  --jsonl-train-path <path>    Use an explicit JSONL text train split instead of the default canary corpus\n",
            "  --jsonl-eval-path <path>     Use an explicit JSONL text eval split instead of the default canary corpus\n",
            "  --corpus-name <name>         Logical corpus name recorded with explicit JSONL text splits\n",
            "  --corpus-text-field <field>  JSONL text field name for explicit split corpora (default: text)\n",
            "  --output-dir <path>          Directory for per-variant reports\n",
            "  --seq-len <n>                Training sequence length (default: {seq_len})\n",
            "  --window-stride <n>          Sliding window stride (default: {stride})\n",
            "  --batch-size <n>             Batch size (default: {batch_size})\n",
            "  --steps <n>                  Training steps per implemented variant (default: {steps})\n",
            "  --eval-batches <n>           Eval batches (default: {eval_batches})\n",
            "  --full-train-pass            Derive --steps from the full train split for the selected corpus\n",
            "  --full-eval-pass             Derive --eval-batches from the full eval split for the selected corpus\n",
            "  --eval-holdout-every <n>     Hold out every nth sequence for eval when using raw file corpora (default: {holdout})\n",
            "  --learning-rate <value>      Learning rate (default: {lr})\n",
            "  --seed <n>                   Random seed for model initialization (default: {seed})\n",
            "  --variant <name>             One of: all, attention-only, reference-ssm-hybrid, primitive-hybrid (default: all)\n",
            "  --primitive-profile <name>   One of: p1, p2-0, p2, p2-1, p2-2, p2-3 (default: p1)\n",
            "  --primitive-residual-profile One of: plain, scaled, gated (default: plain; P2-family contenders only)\n",
            "  --primitive-readout-profile  One of: direct, projected, projected-norm (default: direct; P2-family contenders only)\n",
            "  --primitive-norm-profile     One of: pre-norm-only, post-readout-norm, residual-renorm (default: pre-norm-only; P2-family contenders only)\n",
            "  --primitive-wrapper-profile  One of: standard, mamba-rms (default: standard; P2-family contenders only)\n",
            "  --shared-process             Disable sequential per-variant child-process isolation\n",
            "  --ledger-path <default|path> Append a structured v3a ledger entry for this run\n",
            "  --run-label <label>          Optional label stored with the v3a ledger entry\n",
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

#[cfg(test)]
mod tests {
    use super::{
        apply_benchmark_profile_overrides, apply_full_pass_overrides, resolve_corpus_source,
        BenchmarkProfile, CliArgs, OutputFormat, PrimitiveNormProfile, PrimitiveProfile,
        PrimitiveReadoutProfile, PrimitiveResidualProfile, PrimitiveWrapperProfile,
        VariantSelection,
    };
    use fractal_eval_private::byte_level_smoke_corpus_stats_from_source;
    use std::path::PathBuf;

    fn base_args() -> CliArgs {
        CliArgs {
            benchmark_profile: None,
            backend: super::BackendSelection::Cpu,
            cuda_device: 0,
            corpus_paths: Vec::new(),
            jsonl_train_path: None,
            jsonl_eval_path: None,
            corpus_name: None,
            corpus_text_field: "text".to_owned(),
            output_dir: None,
            seq_len: super::DEFAULT_V3A_SMOKE_SEQ_LEN,
            window_stride: Some(super::DEFAULT_V3A_SMOKE_WINDOW_STRIDE),
            batch_size: super::DEFAULT_V3A_SMOKE_BATCH_SIZE,
            steps: super::DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: super::DEFAULT_V3A_SMOKE_EVAL_BATCHES,
            full_train_pass: false,
            full_eval_pass: false,
            eval_holdout_every: super::DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: super::DEFAULT_V3A_SMOKE_LEARNING_RATE,
            seed: super::DEFAULT_V3A_SMOKE_SEED,
            variant: VariantSelection::AttentionOnly,
            primitive_profile: PrimitiveProfile::P1,
            primitive_residual_profile: PrimitiveResidualProfile::Plain,
            primitive_readout_profile: PrimitiveReadoutProfile::Direct,
            primitive_norm_profile: PrimitiveNormProfile::PreNormOnly,
            primitive_wrapper_profile: PrimitiveWrapperProfile::Standard,
            isolate_variants: true,
            ledger_path: None,
            run_label: None,
            output: OutputFormat::Table,
        }
    }

    #[test]
    fn cuda_faithful_small_profile_resolves_full_pass_budget() {
        let args = CliArgs {
            benchmark_profile: Some(BenchmarkProfile::CudaFaithfulSmallV1),
            ..base_args()
        };
        let args = apply_benchmark_profile_overrides(args).expect("profile overrides should apply");
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let corpus =
            resolve_corpus_source(&args, &repo_root).expect("profile corpus should resolve");
        let stats = byte_level_smoke_corpus_stats_from_source(
            &corpus,
            args.seq_len,
            args.window_stride.unwrap_or(args.seq_len),
            args.eval_holdout_every,
        )
        .expect("corpus stats should derive");
        let args =
            apply_full_pass_overrides(args, &corpus).expect("full pass overrides should derive");

        assert!(args.full_train_pass);
        assert!(args.full_eval_pass);
        assert_eq!(stats.train_sequences, 961);
        assert_eq!(stats.eval_sequences, 94);
        assert_eq!(args.steps, stats.train_sequences.div_ceil(args.batch_size));
        assert_eq!(
            args.eval_batches,
            stats.eval_sequences.div_ceil(args.batch_size)
        );
    }

    #[test]
    fn cuda_faithful_small_profile_rejects_explicit_jsonl_paths() {
        let args = CliArgs {
            benchmark_profile: Some(BenchmarkProfile::CudaFaithfulSmallV1),
            jsonl_train_path: Some(PathBuf::from("/tmp/train.jsonl")),
            jsonl_eval_path: Some(PathBuf::from("/tmp/eval.jsonl")),
            ..base_args()
        };
        let error = apply_benchmark_profile_overrides(args)
            .expect_err("profile should reject explicit corpus overrides");
        assert!(error.contains("may not be combined with explicit corpus path flags"));
    }
}
