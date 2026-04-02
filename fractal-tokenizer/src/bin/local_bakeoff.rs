use burn::backend::Candle;
use fractal_primitives_private::{
    B2StableHierarchical, GeneralizedMobius, Ifs, JuliaRecursiveEscape, LogisticChaoticMap,
    MandelboxRecursiveDynEscapeRadius, P1Contractive, P1FractalHybridComposite, P3Hierarchical,
};
use fractal_tokenizer::{
    build_recursive_overlay, p1_dynamic_lever_factory, pack_overlay_documents_in_batches,
    revived_primitive_factories, EncodedDocument, FaceoffChunkLimits, FaceoffEmissionPolicy,
    FaceoffEncodingOptions, FaceoffFallbackMode, FaceoffIdentityMode, FaceoffLocalCacheMode,
    FaceoffTokenizer, FaceoffVocab, FaceoffVocabConfig, HuggingFaceNativeTokenizer,
    ModelFacingBatch, ModelFacingDocument, MotifReusePolicy, NativeCollationSpec,
    NativeCompatibilityAdapter, OverlayBatchPackingStrategy, OverlayDictionaryScope,
    OverlaySharingPolicy, PrimitiveFactory, PrototypeGranularityMode, RecursiveOverlayConfig,
    RecursiveOverlayMode, SplitPolicy, TokenizerConfig, TokenizerSubstrateMode,
};
use reqwest::blocking::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    env,
    error::Error,
    fs,
    fs::File,
    io::{BufWriter, Write},
    ops::RangeInclusive,
    path::{Path, PathBuf},
    thread,
    time::Duration,
    time::Instant,
};

type Backend = Candle<f32, i64>;

const DEFAULT_OUTPUT_DIR: &str =
    "/Users/joseph/fractal-tokenizer-checkout/fractal-tokenizer/benchmarks/.local";
const DEFAULT_FAWX_ROOT: &str = "/Users/joseph/fawx";
const DEFAULT_HOME_STATE_ROOT: &str = "/Users/joseph/.fawx";
const DEFAULT_CORPUS_LIMIT: usize = 120;
const DEFAULT_HYBRID_CORPUS_LIMIT: usize = 240;
const DEFAULT_LOG_WINDOW_LINES: usize = 120;
const DEFAULT_JSON_WINDOW_LINES: usize = 100;
const DEFAULT_PAD_MULTIPLE: usize = 8;
const DEFAULT_CHUNK_LIMIT_TOKENS: usize = 8;
const DEFAULT_CHUNK_LIMIT_BYTES: usize = 4096;
const DEFAULT_HF_DATASETS_ENDPOINT: &str = "https://datasets-server.huggingface.co";
const LOG_BUCKET_TARGET: usize = 36;
const JSONL_BUCKET_TARGET: usize = 24;
const CODE_BUCKET_TARGET: usize = 36;
const DOCS_BUCKET_TARGET: usize = 24;
const OVERSAMPLE_FACTOR: usize = 2;
const WINDOW_SPLIT_GROUP_SIZE: usize = 3;
const HELD_OUT_NONLOG_CAUTION_RATIO: f64 = 20.0;
const HELD_OUT_NONLOG_CAUTION_REUSE: usize = 2;
const EXTERNAL_CHAR_LIMIT: usize = 12_000;
const EXTERNAL_PROSE_MIN_CHARS: usize = 700;
const EXTERNAL_CODE_MIN_CHARS: usize = 400;
const EXTERNAL_MULTILINGUAL_MIN_CHARS: usize = 700;
const DEFAULT_OVERLAY_PACK_DOCS: usize = 16;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum SourceFamily {
    LocalFawx,
    ExternalHf,
}

impl SourceFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::LocalFawx => "local_fawx",
            Self::ExternalHf => "external_hf",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorpusSourceMode {
    Local,
    Hybrid,
}

impl CorpusSourceMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum CorpusSplit {
    Induction,
    Evaluation,
}

#[derive(Clone, Debug, Serialize)]
struct CorpusDocument {
    id: String,
    source_family: SourceFamily,
    split: CorpusSplit,
    bucket: String,
    source_path: String,
    start_line: usize,
    end_line: usize,
    byte_len: usize,
    char_len: usize,
    text: String,
}

#[derive(Clone, Debug, Serialize)]
struct FractalMetrics {
    input_bytes: usize,
    input_chars: usize,
    frontier_token_count: usize,
    chunk_count: usize,
    avg_chars_per_frontier_token: f64,
    motif_reuse_count: usize,
    fallback_motif_hits: usize,
    fallback_exact_motif_hits: usize,
    fallback_prototype_hits: usize,
    fallback_literal_hits: usize,
    fallback_shape_hits: usize,
    fallback_unknown_motifs: usize,
    fallback_recursed_to_children: usize,
    fallback_local_cache_hits: usize,
    fallback_local_cache_stores: usize,
    fallback_lexical_fallback_tokens: usize,
    fallback_byte_fallback_tokens: usize,
    roundtrip_ok: bool,
    chunk_utf8_ok: bool,
    collation_ok: bool,
    wall_time_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ModelMetrics {
    model_label: String,
    tokenizer_json_path: String,
    status: String,
    native_token_count: usize,
    avg_chars_per_native_token: f64,
    compression_ratio_vs_native: f64,
    native_chunk_count: usize,
    retokenize_ms: f64,
    collate_ms: f64,
    collation_ok: bool,
}

#[derive(Clone, Debug, Serialize)]
struct BakeoffRecord {
    corpus: CorpusDocument,
    fractal: FractalMetrics,
    overlay: Option<OverlayMetrics>,
    models: BTreeMap<String, ModelMetrics>,
}

#[derive(Clone, Debug)]
struct CorpusCandidate {
    corpus: CorpusDocument,
    split_group_key: String,
}

#[derive(Clone, Debug)]
struct ExternalDatasetPlan {
    bucket: &'static str,
    docs: usize,
    slices: Vec<ExternalDatasetSlice>,
    min_chars: usize,
}

#[derive(Clone, Debug)]
struct ExternalDatasetSlice {
    dataset: &'static str,
    config_selector: ExternalConfigSelector,
    split: &'static str,
    text_field: &'static str,
    source_key_fields: &'static [&'static str],
}

#[derive(Clone, Debug)]
enum ExternalConfigSelector {
    Fixed(&'static str),
    LatestSuffix(&'static str),
}

#[derive(Clone, Debug)]
struct ResolvedExternalSlice {
    dataset: &'static str,
    config: String,
    split: &'static str,
    text_field: &'static str,
    source_key_fields: &'static [&'static str],
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetSplitsResponse {
    splits: Option<Vec<DatasetSplitRef>>,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetSplitRef {
    config: String,
    split: String,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetSizeEnvelope {
    size: DatasetSizePayload,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetSizePayload {
    splits: Vec<DatasetSizeSplit>,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetSizeSplit {
    config: String,
    split: String,
    num_rows: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetRowsResponse {
    rows: Vec<DatasetRowsEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct DatasetRowsEntry {
    row_idx: usize,
    row: serde_json::Value,
}

struct HfDatasetsClient {
    endpoint: String,
    client: Client,
}

#[derive(Clone, Debug)]
struct DocumentWork {
    record: BakeoffRecord,
    model_facing: ModelFacingDocument,
}

#[derive(Clone)]
struct ModelTokenizerSource {
    label: String,
    tokenizer_json_path: PathBuf,
    tokenizer: Option<HuggingFaceNativeTokenizer>,
    status: String,
}

#[derive(Clone, Debug)]
struct PrimitiveCandidate {
    name: &'static str,
    factory: PrimitiveFactory<Backend>,
}

#[derive(Clone, Copy, Debug)]
struct FractalEncodingModes {
    fallback_mode: FaceoffFallbackMode,
    identity_mode: FaceoffIdentityMode,
    prototype_granularity: PrototypeGranularityMode,
    split_policy: SplitPolicy,
    substrate_mode: TokenizerSubstrateMode,
    local_cache_mode: FaceoffLocalCacheMode,
}

#[derive(Clone, Debug)]
struct PrimitiveBakeoffRun {
    primitive: String,
    results: Vec<BakeoffRecord>,
    vocab: FaceoffVocab,
}

#[derive(Clone, Debug, Serialize)]
struct OverlayTimingMetrics {
    discovery_ms: f64,
    allocated_pack_ms: f64,
    allocated_materialize_ms: f64,
    client_overhead_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct OverlayMetrics {
    status: String,
    base_tokenizer_label: String,
    base_tokenizer_path: String,
    mode: String,
    canonical_token_count: usize,
    base_slice_symbol_count: usize,
    macro_ref_symbol_count: usize,
    macro_definition_symbol_count: usize,
    overlay_symbol_count: usize,
    compression_ratio_vs_canonical: f64,
    macro_count: usize,
    macro_ref_count: usize,
    repeated_token_mass_saved: usize,
    batch_local_transport_symbols: f64,
    batch_local_transport_ratio: f64,
    batch_local_allocated_definition_symbols: f64,
    batch_local_definition_overhead_rate: f64,
    sequential_pack_transport_symbols: f64,
    sequential_pack_transport_ratio: f64,
    sequential_pack_allocated_definition_symbols: f64,
    sequential_pack_definition_overhead_rate: f64,
    structure_aware_pack_transport_symbols: f64,
    structure_aware_pack_transport_ratio: f64,
    structure_aware_pack_allocated_definition_symbols: f64,
    structure_aware_pack_definition_overhead_rate: f64,
    timing: OverlayTimingMetrics,
    exact_ok: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverlayOfflineBenchmarkVerdict {
    Inactive,
    Fail,
    Pass,
    Strong,
}

impl OverlayOfflineBenchmarkVerdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inactive => "inactive",
            Self::Fail => "fail",
            Self::Pass => "pass",
            Self::Strong => "strong",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverlayBenchmarkBucketRole {
    PrimaryWin,
    NeutralControl,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OverlaySummaryMode {
    Off,
    LocalLineMacro,
    LocalRecordMacro,
}

impl OverlaySummaryMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::LocalLineMacro => "local-line-macro",
            Self::LocalRecordMacro => "local-record-macro",
        }
    }

    fn recursive_mode(self) -> RecursiveOverlayMode {
        match self {
            Self::Off => RecursiveOverlayMode::Off,
            Self::LocalLineMacro => RecursiveOverlayMode::LocalLineMacro,
            Self::LocalRecordMacro => RecursiveOverlayMode::LocalRecordMacro,
        }
    }
}

#[derive(Clone, Debug)]
struct PrimitiveRunDigest {
    primitive: String,
    fallback_mode: FaceoffFallbackMode,
    verdict: BakeoffVerdict,
    byte_fallback_docs: usize,
    exact_motif_hit_docs: usize,
    prototype_hit_docs: usize,
    local_cache_hit_docs: usize,
    lexical_only_docs: usize,
    logs_repetition_heavy_ratio: f64,
    logs_operational_mixed_ratio: f64,
    jsonl_signals_ratio: f64,
    code_rust_ratio: f64,
    code_swift_ratio: f64,
    docs_spec_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BakeoffVerdict {
    Red,
    Yellow,
    Green,
}

impl BakeoffVerdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Red => "RED",
            Self::Yellow => "YELLOW",
            Self::Green => "GREEN",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct BucketSummary {
    bucket: String,
    doc_count: usize,
    median_best_ratio: f64,
    median_motif_reuse: f64,
    byte_fallback_docs: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct FamilySummary {
    source_family: SourceFamily,
    doc_count: usize,
    median_best_ratio: f64,
    median_motif_reuse: f64,
    byte_fallback_docs: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerdictSummary {
    verdict: BakeoffVerdict,
    roundtrip_failures: usize,
    chunk_utf8_failures: usize,
    collation_failures: usize,
    byte_fallback_docs: usize,
    suspicious_nonlog_overcollapse_docs: usize,
    weak_log_buckets: usize,
    external_structural_hit_docs: usize,
    external_code_buckets_below_parity: usize,
    reasons: Vec<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.output_dir)?;

    let corpus = build_corpus(&args)?;
    write_jsonl(
        args.output_dir.join(format!(
            "{}_bakeoff_corpus.jsonl",
            args.corpus_source.as_str()
        )),
        corpus.iter().map(|candidate| &candidate.corpus),
    )?;

    let model_sources = discover_model_tokenizers(&args)?;
    let primitives = selected_primitive_candidates(&args)?;
    let mut runs = Vec::with_capacity(primitives.len());

    for primitive in primitives {
        let run = run_primitive_bakeoff(&corpus, &model_sources, &args, primitive)?;
        write_jsonl(
            args.output_dir.join(format!(
                "{}_{}_{}_results.jsonl",
                args.corpus_source.as_str(),
                run.primitive,
                args.fallback_mode.as_str()
            )),
            run.results.iter(),
        )?;
        print_summary(&run.primitive, &run.results, &run.vocab, &args);
        print_review_list(&run.primitive, &run.results, &args);
        runs.push(run);
    }

    if runs.len() > 1 {
        print_field_summary(&runs, args.fallback_mode);
    }

    Ok(())
}

struct Args {
    output_dir: PathBuf,
    corpus_limit: usize,
    corpus_source: CorpusSourceMode,
    fawx_root: PathBuf,
    home_state_root: PathBuf,
    hf_datasets_endpoint: String,
    max_review_count: usize,
    selected_primitives: Vec<String>,
    all_primitives: bool,
    fallback_mode: FaceoffFallbackMode,
    identity_mode: FaceoffIdentityMode,
    prototype_granularity: PrototypeGranularityMode,
    split_policy: SplitPolicy,
    substrate_mode: TokenizerSubstrateMode,
    local_cache_mode: FaceoffLocalCacheMode,
    overlay_mode: OverlaySummaryMode,
    overlay_base_tokenizer: String,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        Self::parse_from(env::args().skip(1))
    }

    fn parse_from<I, S>(iter: I) -> Result<Self, Box<dyn Error>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
        let mut corpus_limit = DEFAULT_CORPUS_LIMIT;
        let mut corpus_limit_explicit = false;
        let mut corpus_source = CorpusSourceMode::Local;
        let mut fawx_root = PathBuf::from(DEFAULT_FAWX_ROOT);
        let mut home_state_root = PathBuf::from(DEFAULT_HOME_STATE_ROOT);
        let mut hf_datasets_endpoint = DEFAULT_HF_DATASETS_ENDPOINT.to_string();
        let mut max_review_count = 10usize;
        let mut selected_primitives = Vec::new();
        let mut all_primitives = false;
        let mut fallback_mode = FaceoffFallbackMode::Full;
        let mut identity_mode = FaceoffIdentityMode::Legacy;
        let mut prototype_granularity = PrototypeGranularityMode::Coarse;
        let mut split_policy = SplitPolicy::BoundaryAware;
        let mut substrate_mode = TokenizerSubstrateMode::RawBytes;
        let mut local_cache_mode = FaceoffLocalCacheMode::Off;
        let mut overlay_mode = OverlaySummaryMode::Off;
        let mut overlay_base_tokenizer = "qwen25".to_string();

        let mut args = iter.into_iter().map(Into::into);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--output-dir" => {
                    output_dir = PathBuf::from(args.next().ok_or("--output-dir requires a value")?);
                }
                "--corpus-limit" => {
                    corpus_limit_explicit = true;
                    corpus_limit = args
                        .next()
                        .ok_or("--corpus-limit requires a value")?
                        .parse()?;
                }
                "--corpus-source" => {
                    corpus_source = parse_corpus_source(
                        &args.next().ok_or("--corpus-source requires a value")?,
                    )?;
                }
                "--fawx-root" => {
                    fawx_root = PathBuf::from(args.next().ok_or("--fawx-root requires a value")?);
                }
                "--home-state-root" => {
                    home_state_root =
                        PathBuf::from(args.next().ok_or("--home-state-root requires a value")?);
                }
                "--hf-datasets-endpoint" => {
                    hf_datasets_endpoint = args
                        .next()
                        .ok_or("--hf-datasets-endpoint requires a value")?;
                }
                "--max-review-count" => {
                    max_review_count = args
                        .next()
                        .ok_or("--max-review-count requires a value")?
                        .parse()?;
                }
                "--primitive" => {
                    selected_primitives.push(args.next().ok_or("--primitive requires a value")?);
                }
                "--all-primitives" => {
                    all_primitives = true;
                }
                "--fallback-mode" => {
                    fallback_mode = parse_fallback_mode(
                        &args.next().ok_or("--fallback-mode requires a value")?,
                    )?;
                }
                "--identity-mode" => {
                    identity_mode = parse_identity_mode(
                        &args.next().ok_or("--identity-mode requires a value")?,
                    )?;
                }
                "--prototype-granularity" => {
                    prototype_granularity = parse_prototype_granularity(
                        &args
                            .next()
                            .ok_or("--prototype-granularity requires a value")?,
                    )?;
                }
                "--split-policy" => {
                    split_policy =
                        parse_split_policy(&args.next().ok_or("--split-policy requires a value")?)?;
                }
                "--substrate" => {
                    substrate_mode =
                        parse_substrate_mode(&args.next().ok_or("--substrate requires a value")?)?;
                }
                "--local-cache" => {
                    local_cache_mode = parse_local_cache_mode(
                        &args.next().ok_or("--local-cache requires a value")?,
                    )?;
                }
                "--overlay-mode" => {
                    overlay_mode =
                        parse_overlay_mode(&args.next().ok_or("--overlay-mode requires a value")?)?;
                }
                "--overlay-base-tokenizer" => {
                    overlay_base_tokenizer = args
                        .next()
                        .ok_or("--overlay-base-tokenizer requires a value")?;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(format!("unrecognized argument `{other}`").into());
                }
            }
        }

        if corpus_source == CorpusSourceMode::Hybrid && !corpus_limit_explicit {
            corpus_limit = DEFAULT_HYBRID_CORPUS_LIMIT;
        }

        Ok(Self {
            output_dir,
            corpus_limit,
            corpus_source,
            fawx_root,
            home_state_root,
            hf_datasets_endpoint,
            max_review_count,
            selected_primitives,
            all_primitives,
            fallback_mode,
            identity_mode,
            prototype_granularity,
            split_policy,
            substrate_mode,
            local_cache_mode,
            overlay_mode,
            overlay_base_tokenizer,
        })
    }
}

fn print_help() {
    eprintln!(
        "Usage: cargo run -p fractal-tokenizer --bin local_bakeoff -- [--output-dir DIR] [--corpus-limit N] [--corpus-source local|hybrid] [--fawx-root DIR] [--home-state-root DIR] [--hf-datasets-endpoint URL] [--max-review-count N] [--primitive NAME] [--all-primitives] [--fallback-mode full|motif-only] [--identity-mode legacy|prototype-primary] [--prototype-granularity coarse|adaptive] [--split-policy balanced|boundary-aware|syntax-aware] [--substrate raw|lexical] [--local-cache off|exact] [--overlay-mode off|local-line-macro|local-record-macro] [--overlay-base-tokenizer LABEL]"
    );
}

fn parse_corpus_source(value: &str) -> Result<CorpusSourceMode, Box<dyn Error>> {
    match value {
        "local" => Ok(CorpusSourceMode::Local),
        "hybrid" => Ok(CorpusSourceMode::Hybrid),
        other => {
            Err(format!("unknown corpus source `{other}`; expected one of: local, hybrid").into())
        }
    }
}

fn parse_fallback_mode(value: &str) -> Result<FaceoffFallbackMode, Box<dyn Error>> {
    match value {
        "full" => Ok(FaceoffFallbackMode::Full),
        "motif-only" => Ok(FaceoffFallbackMode::MotifOnly),
        other => Err(
            format!("unknown fallback mode `{other}`; expected one of: full, motif-only").into(),
        ),
    }
}

fn parse_identity_mode(value: &str) -> Result<FaceoffIdentityMode, Box<dyn Error>> {
    match value {
        "legacy" => Ok(FaceoffIdentityMode::Legacy),
        "prototype-primary" => Ok(FaceoffIdentityMode::PrototypePrimary),
        other => Err(format!(
            "unknown identity mode `{other}`; expected one of: legacy, prototype-primary"
        )
        .into()),
    }
}

fn parse_prototype_granularity(value: &str) -> Result<PrototypeGranularityMode, Box<dyn Error>> {
    match value {
        "coarse" => Ok(PrototypeGranularityMode::Coarse),
        "adaptive" => Ok(PrototypeGranularityMode::Adaptive),
        other => Err(format!(
            "unknown prototype granularity `{other}`; expected one of: coarse, adaptive"
        )
        .into()),
    }
}

fn parse_substrate_mode(value: &str) -> Result<TokenizerSubstrateMode, Box<dyn Error>> {
    match value {
        "raw" => Ok(TokenizerSubstrateMode::RawBytes),
        "lexical" => Ok(TokenizerSubstrateMode::LexicalAtoms),
        other => {
            Err(format!("unknown substrate mode `{other}`; expected one of: raw, lexical").into())
        }
    }
}

fn parse_split_policy(value: &str) -> Result<SplitPolicy, Box<dyn Error>> {
    match value {
        "balanced" => Ok(SplitPolicy::Balanced),
        "boundary-aware" => Ok(SplitPolicy::BoundaryAware),
        "syntax-aware" => Ok(SplitPolicy::SyntaxAware),
        other => Err(format!(
            "unknown split policy `{other}`; expected one of: balanced, boundary-aware, syntax-aware"
        )
        .into()),
    }
}

fn parse_local_cache_mode(value: &str) -> Result<FaceoffLocalCacheMode, Box<dyn Error>> {
    match value {
        "off" => Ok(FaceoffLocalCacheMode::Off),
        "exact" => Ok(FaceoffLocalCacheMode::ExactSpan),
        other => {
            Err(format!("unknown local cache mode `{other}`; expected one of: off, exact").into())
        }
    }
}

fn parse_overlay_mode(value: &str) -> Result<OverlaySummaryMode, Box<dyn Error>> {
    match value {
        "off" => Ok(OverlaySummaryMode::Off),
        "local-line-macro" => Ok(OverlaySummaryMode::LocalLineMacro),
        "local-record-macro" => Ok(OverlaySummaryMode::LocalRecordMacro),
        other => Err(format!(
            "unknown overlay mode `{other}`; expected one of: off, local-line-macro, local-record-macro"
        )
        .into()),
    }
}

fn selected_primitive_candidates(args: &Args) -> Result<Vec<PrimitiveCandidate>, Box<dyn Error>> {
    let available = available_primitive_candidates();
    if args.all_primitives {
        return Ok(available);
    }

    if args.selected_primitives.is_empty() {
        return Ok(vec![PrimitiveCandidate {
            name: "p1_fractal_hybrid_dyn-state-norm_v2",
            factory: p1_dynamic_lever_factory::<Backend>(),
        }]);
    }

    let mut selected = Vec::new();
    for name in &args.selected_primitives {
        let candidate = available
            .iter()
            .find(|candidate| candidate.name == name)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "unknown primitive `{name}`; available: {}",
                    available
                        .iter()
                        .map(|candidate| candidate.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        if !selected
            .iter()
            .any(|existing: &PrimitiveCandidate| existing.name == candidate.name)
        {
            selected.push(candidate);
        }
    }

    Ok(selected)
}

fn available_primitive_candidates() -> Vec<PrimitiveCandidate> {
    let mut out = revived_primitive_factories::<Backend>()
        .into_iter()
        .map(|factory| PrimitiveCandidate {
            name: factory.name,
            factory,
        })
        .collect::<Vec<_>>();
    out.extend(shared_base_primitive_candidates());
    out.push(PrimitiveCandidate {
        name: "p1_fractal_hybrid_dyn-state-norm_v2",
        factory: p1_dynamic_lever_factory::<Backend>(),
    });
    out
}

fn shared_base_primitive_candidates() -> Vec<PrimitiveCandidate> {
    vec![
        PrimitiveCandidate {
            name: "p1_contractive_v1",
            factory: PrimitiveFactory::new(
                "p1_contractive_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(P1Contractive::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "p1_fractal_hybrid_composite_v1",
            factory: PrimitiveFactory::new(
                "p1_fractal_hybrid_composite_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(P1FractalHybridComposite::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "logistic_chaotic_map_v1",
            factory: PrimitiveFactory::new(
                "logistic_chaotic_map_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(LogisticChaoticMap::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "p3_hierarchical_v1",
            factory: PrimitiveFactory::new(
                "p3_hierarchical_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(P3Hierarchical::new(config.dim, config.levels, device)),
            ),
        },
        PrimitiveCandidate {
            name: "b2_stable_hierarchical_v1",
            factory: PrimitiveFactory::new(
                "b2_stable_hierarchical_v1",
                MotifReusePolicy::Off,
                |config, device| {
                    Box::new(B2StableHierarchical::new(config.dim, config.levels, device))
                },
            ),
        },
        PrimitiveCandidate {
            name: "ifs_dyn-radius-depth_v1",
            factory: PrimitiveFactory::new(
                "ifs_dyn-radius-depth_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(Ifs::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "generalized_mobius_dyn-jitter-norm_v2",
            factory: PrimitiveFactory::new(
                "generalized_mobius_dyn-jitter-norm_v2",
                MotifReusePolicy::Off,
                |config, device| Box::new(GeneralizedMobius::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "julia_recursive_escape_v1",
            factory: PrimitiveFactory::new(
                "julia_recursive_escape_v1",
                MotifReusePolicy::Off,
                |config, device| Box::new(JuliaRecursiveEscape::new(config.dim, device)),
            ),
        },
        PrimitiveCandidate {
            name: "mandelbox_recursive_dyn-escape-radius_v1",
            factory: PrimitiveFactory::new(
                "mandelbox_recursive_dyn-escape-radius_v1",
                MotifReusePolicy::Off,
                |config, device| {
                    Box::new(MandelboxRecursiveDynEscapeRadius::new(config.dim, device))
                },
            ),
        },
    ]
}

fn build_corpus(args: &Args) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    match args.corpus_source {
        CorpusSourceMode::Local => build_local_corpus(args, args.corpus_limit),
        CorpusSourceMode::Hybrid => build_hybrid_corpus(args),
    }
}

fn build_local_corpus(args: &Args, target: usize) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut deduped = dedupe_candidates(build_local_candidate_pool(args)?, target)?;
    assign_balanced_splits(&mut deduped);
    let induction_docs = deduped
        .iter()
        .filter(|candidate| candidate.corpus.split == CorpusSplit::Induction)
        .count();
    let evaluation_docs = deduped
        .iter()
        .filter(|candidate| candidate.corpus.split == CorpusSplit::Evaluation)
        .count();
    if induction_docs == 0 || evaluation_docs == 0 {
        return Err(format!(
            "local bakeoff requires both induction and evaluation documents (induction={induction_docs}, evaluation={evaluation_docs})"
        )
        .into());
    }
    Ok(deduped)
}

fn build_hybrid_corpus(args: &Args) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let local_target = args.corpus_limit / 2;
    let external_target = args.corpus_limit - local_target;
    let local = dedupe_candidates(build_local_candidate_pool(args)?, local_target)?;
    let external = collect_external_candidates(args, external_target)?;

    let mut combined = Vec::with_capacity(local.len() + external.len());
    combined.extend(local);
    combined.extend(external);
    assign_balanced_splits(&mut combined);

    let local_docs = combined
        .iter()
        .filter(|candidate| candidate.corpus.source_family == SourceFamily::LocalFawx)
        .count();
    let external_docs = combined
        .iter()
        .filter(|candidate| candidate.corpus.source_family == SourceFamily::ExternalHf)
        .count();
    let induction_docs = combined
        .iter()
        .filter(|candidate| candidate.corpus.split == CorpusSplit::Induction)
        .count();
    let evaluation_docs = combined
        .iter()
        .filter(|candidate| candidate.corpus.split == CorpusSplit::Evaluation)
        .count();

    if local_docs == 0 || external_docs == 0 {
        return Err(format!(
            "hybrid bakeoff requires both local and external documents (local={local_docs}, external={external_docs})"
        )
        .into());
    }
    if induction_docs == 0 || evaluation_docs == 0 {
        return Err(format!(
            "hybrid bakeoff requires both induction and evaluation documents (induction={induction_docs}, evaluation={evaluation_docs})"
        )
        .into());
    }

    Ok(combined)
}

fn build_local_candidate_pool(args: &Args) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut seen = HashSet::<u64>::new();
    let buckets = vec![
        collect_log_candidates(&args.home_state_root)?,
        collect_jsonl_candidates(&args.home_state_root)?,
        collect_code_candidates(&args.fawx_root)?,
        collect_markdown_candidates(&args.fawx_root)?,
    ];

    let mut candidates = Vec::new();
    round_robin_extend(
        buckets,
        &mut candidates,
        (LOG_BUCKET_TARGET + JSONL_BUCKET_TARGET + CODE_BUCKET_TARGET + DOCS_BUCKET_TARGET)
            .saturating_mul(OVERSAMPLE_FACTOR),
    );

    let mut deduped = Vec::new();
    for candidate in candidates {
        let fingerprint = fnv1a64(&candidate.corpus.text);
        if seen.insert(fingerprint) {
            deduped.push(candidate);
        }
    }
    Ok(deduped)
}

fn dedupe_candidates(
    candidates: Vec<CorpusCandidate>,
    target: usize,
) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut seen = HashSet::<u64>::new();
    let mut deduped = Vec::new();
    for candidate in candidates {
        let fingerprint = fnv1a64(&candidate.corpus.text);
        if seen.insert(fingerprint) {
            deduped.push(candidate);
        }
        if deduped.len() >= target {
            break;
        }
    }

    if deduped.len() < target {
        return Err(format!(
            "only built {} unique documents, wanted {}",
            deduped.len(),
            target
        )
        .into());
    }

    Ok(deduped)
}

fn assign_balanced_splits(candidates: &mut [CorpusCandidate]) {
    let mut per_bucket = BTreeMap::<(SourceFamily, String), BTreeMap<String, Vec<usize>>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        per_bucket
            .entry((
                candidate.corpus.source_family,
                candidate.corpus.bucket.clone(),
            ))
            .or_default()
            .entry(candidate.split_group_key.clone())
            .or_default()
            .push(index);
    }

    for source_groups in per_bucket.into_values() {
        let mut grouped = source_groups.into_iter().collect::<Vec<_>>();
        grouped.sort_by(|left, right| {
            right
                .1
                .len()
                .cmp(&left.1.len())
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut induction_docs = 0usize;
        let mut evaluation_docs = 0usize;
        for (_, indices) in grouped {
            let split = if induction_docs <= evaluation_docs {
                induction_docs += indices.len();
                CorpusSplit::Induction
            } else {
                evaluation_docs += indices.len();
                CorpusSplit::Evaluation
            };

            for index in indices {
                candidates[index].corpus.split = split;
            }
        }
    }
}

fn collect_external_candidates(
    args: &Args,
    target: usize,
) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let client = HfDatasetsClient::new(&args.hf_datasets_endpoint)?;
    let plans = external_dataset_plans(target);
    let mut buckets = Vec::new();

    for plan in plans {
        let mut per_slice = Vec::new();
        for slice in &plan.slices {
            match fetch_external_slice_candidates(
                &client,
                slice,
                plan.bucket,
                plan.docs,
                plan.min_chars,
            ) {
                Ok(candidates) => per_slice.push(candidates),
                Err(error) => eprintln!(
                    "EXTERNAL_SLICE_SKIP bucket={} dataset={} reason={}",
                    plan.bucket, slice.dataset, error
                ),
            }
        }

        if per_slice.is_empty() {
            return Err(format!("no external slices resolved for bucket {}", plan.bucket).into());
        }

        let mut bucket_docs = Vec::new();
        round_robin_extend(
            per_slice,
            &mut bucket_docs,
            plan.docs.saturating_mul(OVERSAMPLE_FACTOR),
        );
        buckets.push(bucket_docs);
    }

    dedupe_candidates(
        {
            let mut candidates = Vec::new();
            round_robin_extend(
                buckets,
                &mut candidates,
                target.saturating_mul(OVERSAMPLE_FACTOR),
            );
            candidates
        },
        target,
    )
}

fn external_dataset_plans(target: usize) -> Vec<ExternalDatasetPlan> {
    let counts = distribute_count(target, 4);
    vec![
        ExternalDatasetPlan {
            bucket: "external.prose.web",
            docs: counts[0],
            slices: vec![ExternalDatasetSlice {
                dataset: "HuggingFaceFW/fineweb-edu",
                config_selector: ExternalConfigSelector::Fixed("default"),
                split: "train",
                text_field: "text",
                source_key_fields: &["url", "id"],
            }],
            min_chars: EXTERNAL_PROSE_MIN_CHARS,
        },
        ExternalDatasetPlan {
            bucket: "external.code.python",
            docs: counts[1],
            slices: vec![ExternalDatasetSlice {
                dataset: "codeparrot/github-code-clean",
                config_selector: ExternalConfigSelector::Fixed("Python-all"),
                split: "train",
                text_field: "code",
                source_key_fields: &["repo_name", "path"],
            }],
            min_chars: EXTERNAL_CODE_MIN_CHARS,
        },
        ExternalDatasetPlan {
            bucket: "external.code.js_ts",
            docs: counts[2],
            slices: vec![
                ExternalDatasetSlice {
                    dataset: "codeparrot/github-code-clean",
                    config_selector: ExternalConfigSelector::Fixed("JavaScript-all"),
                    split: "train",
                    text_field: "code",
                    source_key_fields: &["repo_name", "path"],
                },
                ExternalDatasetSlice {
                    dataset: "codeparrot/github-code-clean",
                    config_selector: ExternalConfigSelector::Fixed("TypeScript-all"),
                    split: "train",
                    text_field: "code",
                    source_key_fields: &["repo_name", "path"],
                },
            ],
            min_chars: EXTERNAL_CODE_MIN_CHARS,
        },
        ExternalDatasetPlan {
            bucket: "external.multilingual",
            docs: counts[3],
            slices: vec![
                ExternalDatasetSlice {
                    dataset: "wikimedia/wikipedia",
                    config_selector: ExternalConfigSelector::LatestSuffix(".es"),
                    split: "train",
                    text_field: "text",
                    source_key_fields: &["url", "title"],
                },
                ExternalDatasetSlice {
                    dataset: "wikimedia/wikipedia",
                    config_selector: ExternalConfigSelector::LatestSuffix(".ja"),
                    split: "train",
                    text_field: "text",
                    source_key_fields: &["url", "title"],
                },
                ExternalDatasetSlice {
                    dataset: "wikimedia/wikipedia",
                    config_selector: ExternalConfigSelector::LatestSuffix(".ar"),
                    split: "train",
                    text_field: "text",
                    source_key_fields: &["url", "title"],
                },
            ],
            min_chars: EXTERNAL_MULTILINGUAL_MIN_CHARS,
        },
    ]
}

fn fetch_external_slice_candidates(
    client: &HfDatasetsClient,
    slice: &ExternalDatasetSlice,
    bucket: &str,
    docs: usize,
    min_chars: usize,
) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let resolved = client.resolve_slice(slice)?;
    let num_rows = client.fetch_num_rows(&resolved)?;
    let wanted = docs.saturating_mul(OVERSAMPLE_FACTOR).max(docs);
    let page_len = wanted.clamp(1, 25);
    let page_starts = evenly_spaced_page_starts(num_rows, page_len, wanted.div_ceil(page_len));
    let mut candidates = Vec::new();

    for start in page_starts {
        let entries = client.fetch_rows(&resolved, start, page_len)?;
        for entry in entries {
            if let Some(text) = extract_string_field(&entry.row, resolved.text_field) {
                let normalized = normalize_external_text(&text, EXTERNAL_CHAR_LIMIT);
                if normalized.chars().count() < min_chars {
                    continue;
                }
                let source_key = external_source_key(&entry.row, resolved.source_key_fields)
                    .unwrap_or_else(|| format!("row-{}", entry.row_idx));
                let source_path = format!(
                    "hf://datasets/{}/{}#split={}&row={}",
                    resolved.dataset, resolved.config, resolved.split, entry.row_idx
                );
                candidates.push(CorpusCandidate {
                    corpus: build_corpus_document(
                        format!(
                            "{}-{:016x}-{:08}",
                            bucket.replace('.', "-"),
                            fnv1a64(&source_key),
                            entry.row_idx
                        ),
                        bucket.to_string(),
                        SourceFamily::ExternalHf,
                        CorpusSplit::Induction,
                        source_path,
                        1..=line_count(&normalized),
                        normalized,
                    ),
                    split_group_key: format!(
                        "{}:{}:{}",
                        resolved.dataset, resolved.config, source_key
                    ),
                });
            }
            if candidates.len() >= wanted {
                break;
            }
        }
        if candidates.len() >= wanted {
            break;
        }
    }

    if candidates.len() < docs {
        return Err(format!(
            "only collected {} external docs for bucket {} from {}:{}; needed {}",
            candidates.len(),
            bucket,
            resolved.dataset,
            resolved.config,
            docs
        )
        .into());
    }

    Ok(candidates)
}

fn distribute_count(total: usize, buckets: usize) -> Vec<usize> {
    let base = total / buckets.max(1);
    let remainder = total % buckets.max(1);
    (0..buckets)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

fn evenly_spaced_offsets(total_rows: usize, wanted: usize) -> Vec<usize> {
    if total_rows == 0 || wanted == 0 {
        return Vec::new();
    }
    if wanted >= total_rows {
        return (0..total_rows).collect();
    }

    let stride = total_rows as f64 / wanted as f64;
    let mut seen = HashSet::new();
    let mut offsets = Vec::with_capacity(wanted);
    for index in 0..wanted {
        let offset = ((index as f64 + 0.5) * stride).floor() as usize;
        let bounded = offset.min(total_rows.saturating_sub(1));
        if seen.insert(bounded) {
            offsets.push(bounded);
        }
    }
    offsets
}

fn evenly_spaced_page_starts(total_rows: usize, page_len: usize, pages: usize) -> Vec<usize> {
    if total_rows == 0 || pages == 0 {
        return Vec::new();
    }
    if total_rows <= page_len {
        return vec![0];
    }
    evenly_spaced_offsets(total_rows - page_len + 1, pages)
}

fn normalize_external_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }

    let mut out = String::new();
    for ch in text.chars().take(limit) {
        out.push(ch);
    }
    out
}

fn extract_string_field(row: &serde_json::Value, field: &str) -> Option<String> {
    row.get(field)
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn external_source_key(row: &serde_json::Value, fields: &[&str]) -> Option<String> {
    let parts = fields
        .iter()
        .filter_map(|field| extract_string_field(row, field))
        .collect::<Vec<_>>();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("::"))
    }
}

impl HfDatasetsClient {
    fn new(endpoint: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client: Client::builder().build()?,
        })
    }

    fn resolve_slice(
        &self,
        slice: &ExternalDatasetSlice,
    ) -> Result<ResolvedExternalSlice, Box<dyn Error>> {
        let splits = self.fetch_splits(slice.dataset)?;
        let config =
            select_config_for_slice(slice.dataset, &slice.config_selector, slice.split, &splits)?;

        Ok(ResolvedExternalSlice {
            dataset: slice.dataset,
            config,
            split: slice.split,
            text_field: slice.text_field,
            source_key_fields: slice.source_key_fields,
        })
    }

    fn fetch_splits(&self, dataset: &str) -> Result<Vec<DatasetSplitRef>, Box<dyn Error>> {
        let response =
            self.get_json::<DatasetSplitsResponse>("/splits", &[("dataset", dataset)])?;
        Ok(response.splits.unwrap_or_default())
    }

    fn fetch_num_rows(&self, slice: &ResolvedExternalSlice) -> Result<usize, Box<dyn Error>> {
        let response = self.get_json::<DatasetSizeEnvelope>(
            "/size",
            &[
                ("dataset", slice.dataset),
                ("config", slice.config.as_str()),
                ("split", slice.split),
            ],
        )?;
        response
            .size
            .splits
            .into_iter()
            .find(|candidate| candidate.config == slice.config && candidate.split == slice.split)
            .map(|candidate| candidate.num_rows)
            .ok_or_else(|| {
                format!(
                    "size endpoint missing num_rows for dataset={} config={} split={}",
                    slice.dataset, slice.config, slice.split
                )
                .into()
            })
    }

    fn fetch_rows(
        &self,
        slice: &ResolvedExternalSlice,
        offset: usize,
        length: usize,
    ) -> Result<Vec<DatasetRowsEntry>, Box<dyn Error>> {
        let offset_string = offset.to_string();
        let length_string = length.to_string();
        let response = self.get_json::<DatasetRowsResponse>(
            "/rows",
            &[
                ("dataset", slice.dataset),
                ("config", slice.config.as_str()),
                ("split", slice.split),
                ("offset", offset_string.as_str()),
                ("length", length_string.as_str()),
            ],
        )?;
        if response.rows.is_empty() {
            return Err(format!(
                "rows endpoint returned no rows for dataset={} config={} split={} offset={offset} length={length}",
                slice.dataset, slice.config, slice.split
            )
            .into());
        }
        Ok(response.rows)
    }

    fn get_json<T>(&self, path: &str, query: &[(&str, &str)]) -> Result<T, Box<dyn Error>>
    where
        T: DeserializeOwned,
    {
        let url = format!("{}{}", self.endpoint, path);
        let mut last_error = None;
        for attempt in 0..5 {
            match self.client.get(&url).query(query).send() {
                Ok(response) => {
                    let status = response.status();
                    let retryable_status = status.is_server_error()
                        || status == reqwest::StatusCode::TOO_MANY_REQUESTS;
                    if retryable_status && attempt < 4 {
                        thread::sleep(Duration::from_millis(500 * (attempt + 1) as u64));
                        continue;
                    }
                    let response = response.error_for_status()?;
                    return Ok(response.json::<T>()?);
                }
                Err(error) => {
                    let retryable = error.is_timeout() || error.is_connect() || error.is_request();
                    last_error = Some(error);
                    if retryable && attempt < 4 {
                        thread::sleep(Duration::from_millis(500 * (attempt + 1) as u64));
                        continue;
                    }
                    break;
                }
            }
        }

        Err(last_error
            .map(|error| error.into())
            .unwrap_or_else(|| "dataset request failed without an error".into()))
    }
}

fn select_config_for_slice(
    dataset: &str,
    selector: &ExternalConfigSelector,
    split: &str,
    splits: &[DatasetSplitRef],
) -> Result<String, Box<dyn Error>> {
    match selector {
        ExternalConfigSelector::Fixed(config) => splits
            .iter()
            .find(|candidate| candidate.config == *config && candidate.split == split)
            .map(|_| (*config).to_string())
            .ok_or_else(|| {
                format!("dataset {dataset} missing config={config} split={split}").into()
            }),
        ExternalConfigSelector::LatestSuffix(suffix) => splits
            .iter()
            .filter(|candidate| candidate.split == split && candidate.config.ends_with(suffix))
            .map(|candidate| candidate.config.clone())
            .max()
            .ok_or_else(|| {
                format!("dataset {dataset} missing config suffix {suffix} for split {split}").into()
            }),
    }
}

fn collect_log_candidates(home_state_root: &Path) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut buckets = Vec::new();

    let server_log = home_state_root.join("server.log");
    if let Ok(text) = fs::read_to_string(&server_log) {
        buckets.push(line_window_documents(
            &text,
            &server_log,
            "logs.repetition_heavy",
            DEFAULT_LOG_WINDOW_LINES,
            "logs-server",
            18,
        ));
    }

    let mut rotated = Vec::new();
    for path in sorted_log_files(&home_state_root.join("logs"))?
        .into_iter()
        .take(6)
    {
        if let Ok(text) = fs::read_to_string(&path) {
            rotated.push(line_window_documents(
                &text,
                &path,
                "logs.operational_mixed",
                DEFAULT_LOG_WINDOW_LINES,
                "logs-rotated",
                3,
            ));
        }
    }
    if !rotated.is_empty() {
        buckets.extend(rotated);
    }

    let mut out = Vec::new();
    round_robin_extend(
        buckets,
        &mut out,
        LOG_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR),
    );
    Ok(out)
}

fn collect_jsonl_candidates(
    home_state_root: &Path,
) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut buckets = Vec::new();

    let journal_path = home_state_root.join("journal.jsonl");
    if let Ok(text) = fs::read_to_string(&journal_path) {
        buckets.push(vec![whole_file_candidate(
            &text,
            &journal_path,
            "jsonl.journal",
            "journal-0001",
        )]);
    }

    let signals_path = home_state_root.join("signals").join("headless.jsonl");
    if let Ok(text) = fs::read_to_string(&signals_path) {
        buckets.push(line_window_documents(
            &text,
            &signals_path,
            "jsonl.signals",
            DEFAULT_JSON_WINDOW_LINES,
            "signals",
            23,
        ));
    }

    let mut out = Vec::new();
    round_robin_extend(
        buckets,
        &mut out,
        JSONL_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR),
    );
    Ok(out)
}

fn collect_code_candidates(fawx_root: &Path) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut buckets = Vec::new();

    let mut rust_candidates = Vec::new();
    for path in sorted_files_with_extension(&fawx_root.join("engine"), "rs")?
        .into_iter()
        .take(CODE_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR))
    {
        if let Ok(text) = fs::read_to_string(&path) {
            rust_candidates.push(prefix_file_document(&text, &path, "code.rust", 12_000));
        }
    }
    if !rust_candidates.is_empty() {
        buckets.push(rust_candidates);
    }

    let mut swift_candidates = Vec::new();
    for path in sorted_files_with_extension(&fawx_root.join("app"), "swift")?
        .into_iter()
        .take((CODE_BUCKET_TARGET / 2).saturating_mul(OVERSAMPLE_FACTOR))
    {
        if let Ok(text) = fs::read_to_string(&path) {
            swift_candidates.push(prefix_file_document(&text, &path, "code.swift", 12_000));
        }
    }
    if !swift_candidates.is_empty() {
        buckets.push(swift_candidates);
    }

    let mut out = Vec::new();
    round_robin_extend(
        buckets,
        &mut out,
        CODE_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR),
    );
    Ok(out)
}

fn collect_markdown_candidates(fawx_root: &Path) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut candidates = Vec::new();

    for path in sorted_files_with_extension(&fawx_root.join("docs"), "md")?
        .into_iter()
        .take(DOCS_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR))
    {
        if let Ok(text) = fs::read_to_string(&path) {
            candidates.push(prefix_file_document(&text, &path, "docs.spec", 12_000));
        }
    }

    let mut out = Vec::new();
    round_robin_extend(
        vec![candidates],
        &mut out,
        DOCS_BUCKET_TARGET.saturating_mul(OVERSAMPLE_FACTOR),
    );
    Ok(out)
}

fn sorted_files_with_extension(
    root: &Path,
    extension: &str,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = Vec::new();
    if root.exists() {
        collect_files_recursively(root, &mut files, extension)?;
    }
    files.sort();
    Ok(files)
}

fn sorted_log_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = Vec::new();
    if root.exists() {
        collect_files_recursively(root, &mut files, "log")?;
    }
    files.sort();
    Ok(files)
}

fn collect_files_recursively(
    root: &Path,
    out: &mut Vec<PathBuf>,
    extension: &str,
) -> Result<(), Box<dyn Error>> {
    if !root.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        if path.is_dir() {
            if should_skip_directory(name) {
                continue;
            }
            collect_files_recursively(&path, out, extension)?;
        } else if path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }

    Ok(())
}

fn should_skip_directory(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "build"
            | ".derived"
            | ".derived-ios"
            | ".derived-macos"
            | ".openclaw-worktrees"
    ) || name.starts_with('.')
}

fn line_window_documents(
    text: &str,
    source_path: &Path,
    bucket: &str,
    window_lines: usize,
    prefix: &str,
    max_windows: usize,
) -> Vec<CorpusCandidate> {
    let lines = split_inclusive_lines(text);
    if lines.is_empty() {
        return Vec::new();
    }

    let mut docs = Vec::new();
    let mut start = 0usize;
    let mut window_index = 1usize;
    while start < lines.len() && docs.len() < max_windows {
        let end = (start + window_lines).min(lines.len());
        let slice = lines[start..end].join("");
        let split_group = format!(
            "{}#group-{:04}",
            source_path.display(),
            (window_index - 1) / WINDOW_SPLIT_GROUP_SIZE
        );
        docs.push(CorpusCandidate {
            corpus: build_corpus_document(
                format!(
                    "{}-{}-{:016x}-{:04}",
                    prefix,
                    source_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("source"),
                    fnv1a64(&source_path.to_string_lossy()),
                    window_index
                ),
                bucket.to_string(),
                SourceFamily::LocalFawx,
                CorpusSplit::Induction,
                source_path.to_string_lossy().to_string(),
                (start + 1)..=end,
                slice,
            ),
            split_group_key: split_group,
        });
        start = end;
        window_index += 1;
    }

    docs
}

fn whole_file_candidate(
    text: &str,
    source_path: &Path,
    bucket: &str,
    suffix: &str,
) -> CorpusCandidate {
    CorpusCandidate {
        corpus: build_corpus_document(
            format!(
                "{}-{}-{:016x}",
                source_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("source"),
                suffix,
                fnv1a64(&source_path.to_string_lossy())
            ),
            bucket.to_string(),
            SourceFamily::LocalFawx,
            CorpusSplit::Induction,
            source_path.to_string_lossy().to_string(),
            1..=line_count(text),
            text.to_string(),
        ),
        split_group_key: source_path.to_string_lossy().to_string(),
    }
}

fn prefix_file_document(
    text: &str,
    source_path: &Path,
    bucket: &str,
    char_limit: usize,
) -> CorpusCandidate {
    let (slice, end_line) = prefix_lines_to_char_limit(text, char_limit);
    CorpusCandidate {
        corpus: build_corpus_document(
            format!(
                "{}-prefix-{:016x}",
                source_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("source"),
                fnv1a64(&source_path.to_string_lossy())
            ),
            bucket.to_string(),
            SourceFamily::LocalFawx,
            CorpusSplit::Induction,
            source_path.to_string_lossy().to_string(),
            1..=end_line,
            slice,
        ),
        split_group_key: source_path.to_string_lossy().to_string(),
    }
}

fn build_corpus_document(
    id: String,
    bucket: String,
    source_family: SourceFamily,
    split: CorpusSplit,
    source_path: String,
    line_range: RangeInclusive<usize>,
    text: String,
) -> CorpusDocument {
    let (start_line, end_line) = (*line_range.start(), *line_range.end());
    let byte_len = text.len();
    let char_len = text.chars().count();
    CorpusDocument {
        id,
        bucket,
        source_family,
        split,
        source_path,
        start_line,
        end_line,
        byte_len,
        char_len,
        text,
    }
}

fn split_inclusive_lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

fn line_count(text: &str) -> usize {
    split_inclusive_lines(text).len().max(1)
}

fn prefix_lines_to_char_limit(text: &str, limit: usize) -> (String, usize) {
    if text.chars().count() <= limit {
        return (text.to_string(), line_count(text));
    }

    let mut kept = String::new();
    let mut line_count = 0usize;
    let mut char_count = 0usize;

    for line in split_inclusive_lines(text) {
        let line_chars = line.chars().count();
        if !kept.is_empty() && char_count + line_chars > limit {
            break;
        }
        kept.push_str(line);
        char_count += line_chars;
        line_count += 1;
        if char_count >= limit {
            break;
        }
    }

    if kept.is_empty() {
        if let Some(first_line) = split_inclusive_lines(text).into_iter().next() {
            kept.push_str(first_line);
            line_count = 1;
        }
    }

    (kept, line_count.max(1))
}

fn round_robin_extend(
    mut per_file: Vec<Vec<CorpusCandidate>>,
    out: &mut Vec<CorpusCandidate>,
    target: usize,
) {
    let mut index = 0usize;
    loop {
        let mut progress = false;
        for docs in per_file.iter_mut() {
            if out.len() >= target {
                return;
            }
            if let Some(candidate) = docs.get(index).cloned() {
                out.push(candidate);
                progress = true;
            }
        }
        if !progress {
            break;
        }
        index += 1;
    }
}

fn run_primitive_bakeoff(
    corpus: &[CorpusCandidate],
    model_sources: &[ModelTokenizerSource],
    args: &Args,
    primitive: PrimitiveCandidate,
) -> Result<PrimitiveBakeoffRun, Box<dyn Error>> {
    let primitive_name = primitive.name.to_string();
    let modes = FractalEncodingModes {
        fallback_mode: args.fallback_mode,
        identity_mode: args.identity_mode,
        prototype_granularity: args.prototype_granularity,
        split_policy: args.split_policy,
        substrate_mode: args.substrate_mode,
        local_cache_mode: args.local_cache_mode,
    };
    let (works, vocab) = build_fractal_documents(corpus, primitive, modes)?;
    let model_results = run_model_bakeoff(&works, model_sources)?;
    let mut results = merge_results(works, model_results);
    attach_overlay_shadow(&mut results, model_sources, args)?;
    Ok(PrimitiveBakeoffRun {
        primitive: primitive_name,
        results,
        vocab,
    })
}

fn build_fractal_documents(
    corpus: &[CorpusCandidate],
    primitive: PrimitiveCandidate,
    modes: FractalEncodingModes,
) -> Result<(Vec<DocumentWork>, FaceoffVocab), Box<dyn Error>> {
    let device = Default::default();
    let tokenizer = FaceoffTokenizer::new(TokenizerConfig {
        split_policy: modes.split_policy,
        substrate_mode: modes.substrate_mode,
        ..TokenizerConfig::default()
    });
    let texts = corpus
        .iter()
        .filter(|candidate| candidate.corpus.split == CorpusSplit::Induction)
        .map(|candidate| candidate.corpus.text.as_str())
        .collect::<Vec<_>>();
    if texts.is_empty() {
        return Err("local bakeoff needs at least one induction document".into());
    }
    let vocab = tokenizer.induce_vocab_from_texts_for_factory_with_config::<Backend>(
        &texts,
        &device,
        primitive.factory.clone(),
        FaceoffVocabConfig {
            identity_mode: modes.identity_mode,
            prototype_granularity: modes.prototype_granularity,
            ..FaceoffVocabConfig::default()
        },
    )?;
    let limits = FaceoffChunkLimits::new(DEFAULT_CHUNK_LIMIT_TOKENS, DEFAULT_CHUNK_LIMIT_BYTES);
    let mut works = Vec::with_capacity(corpus.len());

    for candidate in corpus {
        let started = Instant::now();
        let encoded = tokenizer.encode_text_with_factory::<Backend>(
            &candidate.corpus.text,
            &vocab,
            &device,
            primitive.factory.clone(),
            FaceoffEncodingOptions {
                policy: FaceoffEmissionPolicy::NoveltyAware,
                fallback_mode: modes.fallback_mode,
                local_cache_mode: modes.local_cache_mode,
            },
        )?;
        let wall_time_ms = started.elapsed().as_secs_f64() * 1000.0;
        let model_facing = ModelFacingDocument::from_encoded(encoded.clone(), limits)?;
        let chunk_utf8_ok = model_facing
            .chunked()
            .chunks
            .iter()
            .all(|chunk| std::str::from_utf8(&chunk.payload).is_ok());
        let roundtrip_ok = model_facing.reconstruct()? == candidate.corpus.text;
        let motif_reuse_count = encoded_cross_depth_motif_reuse_count(&encoded);

        let fractal = FractalMetrics {
            input_bytes: candidate.corpus.byte_len,
            input_chars: candidate.corpus.char_len,
            frontier_token_count: encoded.tokens.len(),
            chunk_count: model_facing.chunk_count(),
            avg_chars_per_frontier_token: candidate.corpus.char_len as f64
                / encoded.tokens.len().max(1) as f64,
            motif_reuse_count,
            fallback_motif_hits: encoded.fallback.motif_hits,
            fallback_exact_motif_hits: encoded.fallback.exact_motif_hits,
            fallback_prototype_hits: encoded.fallback.prototype_hits,
            fallback_literal_hits: encoded.fallback.literal_hits,
            fallback_shape_hits: encoded.fallback.shape_hits,
            fallback_unknown_motifs: encoded.fallback.unknown_motifs,
            fallback_recursed_to_children: encoded.fallback.recursed_to_children,
            fallback_local_cache_hits: encoded.fallback.local_cache_hits,
            fallback_local_cache_stores: encoded.fallback.local_cache_stores,
            fallback_lexical_fallback_tokens: encoded.fallback.lexical_fallback_tokens,
            fallback_byte_fallback_tokens: encoded.fallback.byte_fallback_tokens,
            roundtrip_ok,
            chunk_utf8_ok,
            collation_ok: false,
            wall_time_ms,
        };

        works.push(DocumentWork {
            record: BakeoffRecord {
                corpus: candidate.corpus.clone(),
                fractal,
                overlay: None,
                models: BTreeMap::new(),
            },
            model_facing,
        });
    }

    Ok((works, vocab))
}

fn discover_model_tokenizers(_args: &Args) -> Result<Vec<ModelTokenizerSource>, Box<dyn Error>> {
    let mut sources = Vec::new();
    for (label, env_var, fallback) in [
        (
            "llama31",
            "HF_LLAMA31_TOKENIZER_JSON",
            "/Users/joseph/hf-tokenizers/llama31/tokenizer.json",
        ),
        (
            "mistral7",
            "HF_MISTRAL7_TOKENIZER_JSON",
            "/Users/joseph/hf-tokenizers/mistral7/tokenizer.json",
        ),
        (
            "mixtral8x7b",
            "HF_MIXTRAL8X7B_TOKENIZER_JSON",
            "/Users/joseph/hf-tokenizers/mixtral8x7b/tokenizer.json",
        ),
        (
            "qwen25",
            "HF_QWEN25_TOKENIZER_JSON",
            "/Users/joseph/hf-tokenizers/qwen25/tokenizer.json",
        ),
        (
            "phi3mini",
            "HF_PHI3MINI_TOKENIZER_JSON",
            "/Users/joseph/hf-tokenizers/phi3mini/tokenizer.json",
        ),
    ] {
        let path = env::var_os(env_var)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(fallback));
        let tokenizer = if path.exists() {
            Some(HuggingFaceNativeTokenizer::from_file(&path)?)
        } else {
            None
        };
        let status = if tokenizer.is_some() {
            "ok".to_string()
        } else {
            "skipped".to_string()
        };
        sources.push(ModelTokenizerSource {
            label: label.to_string(),
            tokenizer_json_path: path,
            tokenizer,
            status,
        });
    }
    Ok(sources)
}

fn run_model_bakeoff(
    works: &[DocumentWork],
    model_sources: &[ModelTokenizerSource],
) -> Result<Vec<BTreeMap<String, ModelMetrics>>, Box<dyn Error>> {
    let adapter = NativeCompatibilityAdapter;
    let batch = ModelFacingBatch::from(
        works
            .iter()
            .map(|work| work.model_facing.clone())
            .collect::<Vec<_>>(),
    );
    let mut per_doc = vec![BTreeMap::new(); works.len()];

    for source in model_sources {
        let metrics = if let Some(tokenizer) = &source.tokenizer {
            let retokenize_started = Instant::now();
            let native = adapter.retokenize_batch(&batch, tokenizer)?;
            let retokenize_ms = retokenize_started.elapsed().as_secs_f64() * 1000.0;

            let pad_id = tokenizer
                .tokenizer()
                .get_padding()
                .map(|padding| padding.pad_id)
                .unwrap_or(0);
            let spec = NativeCollationSpec::try_new(pad_id, Some(DEFAULT_PAD_MULTIPLE))?;
            let collate_started = Instant::now();
            let collated = native.collate(&spec)?;
            let collate_ms = collate_started.elapsed().as_secs_f64() * 1000.0;
            let collation_ok = collated.len() == batch.len()
                && collated
                    .iter()
                    .zip(batch.iter())
                    .all(|(collated_doc, original)| {
                        collated_doc.input_len == original.input_len()
                            && collated_doc.frontier_token_count == original.frontier_token_count()
                            && collated_doc.chunk_count() == original.chunk_count()
                    });

            native
                .iter()
                .enumerate()
                .map(|(index, native_doc)| {
                    let frontier = works[index].record.fractal.frontier_token_count.max(1);
                    ModelMetrics {
                        model_label: source.label.clone(),
                        tokenizer_json_path: source.tokenizer_json_path.display().to_string(),
                        status: "ok".to_string(),
                        native_token_count: native_doc.native_token_count(),
                        avg_chars_per_native_token: works[index].record.corpus.char_len as f64
                            / native_doc.native_token_count().max(1) as f64,
                        compression_ratio_vs_native: native_doc.native_token_count() as f64
                            / frontier as f64,
                        native_chunk_count: native_doc.chunk_count(),
                        retokenize_ms,
                        collate_ms,
                        collation_ok,
                    }
                })
                .collect::<Vec<_>>()
        } else {
            works
                .iter()
                .map(|_| ModelMetrics {
                    model_label: source.label.clone(),
                    tokenizer_json_path: source.tokenizer_json_path.display().to_string(),
                    status: source.status.clone(),
                    native_token_count: 0,
                    avg_chars_per_native_token: 0.0,
                    compression_ratio_vs_native: 0.0,
                    native_chunk_count: 0,
                    retokenize_ms: 0.0,
                    collate_ms: 0.0,
                    collation_ok: false,
                })
                .collect::<Vec<_>>()
        };

        for (index, metric) in metrics.into_iter().enumerate() {
            per_doc[index].insert(source.label.clone(), metric);
        }
    }

    Ok(per_doc)
}

fn merge_results(
    works: Vec<DocumentWork>,
    model_results: Vec<BTreeMap<String, ModelMetrics>>,
) -> Vec<BakeoffRecord> {
    works
        .into_iter()
        .enumerate()
        .map(|(index, mut work)| {
            work.record.models = model_results[index].clone();
            let collation_ok = work
                .record
                .models
                .values()
                .filter(|metric| metric.status == "ok")
                .all(|metric| metric.collation_ok);
            work.record.fractal.collation_ok = collation_ok;
            work.record
        })
        .collect()
}

fn overlay_placeholder_metrics(
    status: &str,
    base_tokenizer_label: String,
    base_tokenizer_path: String,
    mode: String,
) -> OverlayMetrics {
    OverlayMetrics {
        status: status.to_string(),
        base_tokenizer_label,
        base_tokenizer_path,
        mode,
        canonical_token_count: 0,
        base_slice_symbol_count: 0,
        macro_ref_symbol_count: 0,
        macro_definition_symbol_count: 0,
        overlay_symbol_count: 0,
        compression_ratio_vs_canonical: 0.0,
        macro_count: 0,
        macro_ref_count: 0,
        repeated_token_mass_saved: 0,
        batch_local_transport_symbols: 0.0,
        batch_local_transport_ratio: 0.0,
        batch_local_allocated_definition_symbols: 0.0,
        batch_local_definition_overhead_rate: 0.0,
        sequential_pack_transport_symbols: 0.0,
        sequential_pack_transport_ratio: 0.0,
        sequential_pack_allocated_definition_symbols: 0.0,
        sequential_pack_definition_overhead_rate: 0.0,
        structure_aware_pack_transport_symbols: 0.0,
        structure_aware_pack_transport_ratio: 0.0,
        structure_aware_pack_allocated_definition_symbols: 0.0,
        structure_aware_pack_definition_overhead_rate: 0.0,
        timing: OverlayTimingMetrics {
            discovery_ms: 0.0,
            allocated_pack_ms: 0.0,
            allocated_materialize_ms: 0.0,
            client_overhead_ms: 0.0,
        },
        exact_ok: true,
    }
}

fn overlay_bucket_role(bucket: &str) -> OverlayBenchmarkBucketRole {
    match bucket {
        "jsonl.signals" | "logs.operational_mixed" => OverlayBenchmarkBucketRole::PrimaryWin,
        "docs.spec"
        | "external.prose.web"
        | "external.code.python"
        | "external.code.js_ts"
        | "external.multilingual" => OverlayBenchmarkBucketRole::NeutralControl,
        _ => OverlayBenchmarkBucketRole::Other,
    }
}

fn attach_overlay_shadow(
    results: &mut [BakeoffRecord],
    model_sources: &[ModelTokenizerSource],
    args: &Args,
) -> Result<(), Box<dyn Error>> {
    if args.overlay_mode == OverlaySummaryMode::Off {
        for record in results {
            record.overlay = Some(overlay_placeholder_metrics(
                "off",
                args.overlay_base_tokenizer.clone(),
                String::new(),
                "off".to_string(),
            ));
        }
        return Ok(());
    }

    let Some(source) = select_overlay_base_tokenizer(model_sources, &args.overlay_base_tokenizer)?
    else {
        for record in results {
            record.overlay = Some(overlay_placeholder_metrics(
                "skipped",
                args.overlay_base_tokenizer.clone(),
                String::new(),
                args.overlay_mode.as_str().to_string(),
            ));
        }
        return Ok(());
    };

    let Some(tokenizer) = &source.tokenizer else {
        for record in results {
            record.overlay = Some(overlay_placeholder_metrics(
                "skipped",
                source.label.clone(),
                source.tokenizer_json_path.display().to_string(),
                args.overlay_mode.as_str().to_string(),
            ));
        }
        return Ok(());
    };

    let config = RecursiveOverlayConfig::default();
    let mut overlays = Vec::with_capacity(results.len());
    let mut discovery_ms = Vec::with_capacity(results.len());
    for record in results.iter() {
        let started = Instant::now();
        let canonical = tokenizer.tokenize_with_byte_offsets(&record.corpus.text)?;
        let overlay = build_recursive_overlay(
            &record.corpus.text,
            canonical,
            args.overlay_mode.recursive_mode(),
            &config,
        );
        discovery_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        overlays.push(overlay);
    }

    let sharing_policy = OverlaySharingPolicy::default();
    let batch_pack_started = Instant::now();
    let batch_local_pack = pack_overlay_documents_in_batches(
        OverlayDictionaryScope::BatchLocal,
        &overlays,
        &sharing_policy,
        overlays.len().max(1),
        OverlayBatchPackingStrategy::Sequential,
    );
    let batch_local_pack_ms = batch_pack_started.elapsed().as_secs_f64() * 1000.0;
    let batch_materialize_started = Instant::now();
    let batch_materialize_exact_ok = batch_local_pack.expanded_token_ids_by_document().is_ok();
    let batch_materialize_ms = batch_materialize_started.elapsed().as_secs_f64() * 1000.0;
    let batch_local_views = batch_local_pack.document_views.clone();
    let batch_local_exact_ok = batch_local_pack.exact_ok();
    let sequential_pack = pack_overlay_documents_in_batches(
        OverlayDictionaryScope::BatchLocal,
        &overlays,
        &sharing_policy,
        DEFAULT_OVERLAY_PACK_DOCS,
        OverlayBatchPackingStrategy::Sequential,
    );
    let sequential_pack_exact_ok = sequential_pack.exact_ok();
    let structure_aware_pack = pack_overlay_documents_in_batches(
        OverlayDictionaryScope::BatchLocal,
        &overlays,
        &sharing_policy,
        DEFAULT_OVERLAY_PACK_DOCS,
        OverlayBatchPackingStrategy::StructureAware,
    );
    let structure_aware_pack_exact_ok = structure_aware_pack.exact_ok();
    let total_canonical_tokens = overlays
        .iter()
        .map(|overlay| overlay.canonical.token_ids.len())
        .sum::<usize>()
        .max(1) as f64;

    for (
        ((((record, overlay), batch_view), sequential_view), structure_aware_view),
        discovery_ms,
    ) in results
        .iter_mut()
        .zip(overlays.into_iter())
        .zip(batch_local_views.into_iter())
        .zip(sequential_pack.document_views.into_iter())
        .zip(structure_aware_pack.document_views.into_iter())
        .zip(discovery_ms.into_iter())
    {
        let canonical_token_count = overlay.canonical.token_ids.len();
        let allocation_weight = canonical_token_count as f64 / total_canonical_tokens;
        let allocated_pack_ms = batch_local_pack_ms * allocation_weight;
        let allocated_materialize_ms = batch_materialize_ms * allocation_weight;
        record.overlay = Some(OverlayMetrics {
            status: "ok".to_string(),
            base_tokenizer_label: source.label.clone(),
            base_tokenizer_path: source.tokenizer_json_path.display().to_string(),
            mode: match overlay.mode {
                fractal_tokenizer::OverlayDocumentMode::Passthrough => "passthrough".to_string(),
                fractal_tokenizer::OverlayDocumentMode::LocalMacro => "local-macro".to_string(),
            },
            canonical_token_count,
            base_slice_symbol_count: overlay.base_slice_symbol_count(),
            macro_ref_symbol_count: overlay.macro_ref_symbol_count(),
            macro_definition_symbol_count: overlay.macro_definition_symbol_count(),
            overlay_symbol_count: overlay.overlay_symbol_count(),
            compression_ratio_vs_canonical: overlay.compression_ratio_vs_canonical(),
            macro_count: overlay.macros.len(),
            macro_ref_count: overlay.macro_ref_count(),
            repeated_token_mass_saved: overlay.repeated_token_mass_saved(),
            batch_local_transport_symbols: batch_view.transport_symbols(),
            batch_local_transport_ratio: batch_view.transport_ratio(),
            batch_local_allocated_definition_symbols: batch_view.allocated_macro_definition_symbols,
            batch_local_definition_overhead_rate: batch_view.definition_overhead_rate(),
            sequential_pack_transport_symbols: sequential_view.transport_symbols(),
            sequential_pack_transport_ratio: sequential_view.transport_ratio(),
            sequential_pack_allocated_definition_symbols: sequential_view
                .allocated_macro_definition_symbols,
            sequential_pack_definition_overhead_rate: sequential_view.definition_overhead_rate(),
            structure_aware_pack_transport_symbols: structure_aware_view.transport_symbols(),
            structure_aware_pack_transport_ratio: structure_aware_view.transport_ratio(),
            structure_aware_pack_allocated_definition_symbols: structure_aware_view
                .allocated_macro_definition_symbols,
            structure_aware_pack_definition_overhead_rate: structure_aware_view
                .definition_overhead_rate(),
            timing: OverlayTimingMetrics {
                discovery_ms,
                allocated_pack_ms,
                allocated_materialize_ms,
                client_overhead_ms: discovery_ms + allocated_pack_ms + allocated_materialize_ms,
            },
            exact_ok: overlay.exact_ok()
                && batch_local_exact_ok
                && batch_materialize_exact_ok
                && sequential_pack_exact_ok
                && structure_aware_pack_exact_ok,
        });
    }

    Ok(())
}

fn select_overlay_base_tokenizer<'a>(
    model_sources: &'a [ModelTokenizerSource],
    preferred_label: &str,
) -> Result<Option<&'a ModelTokenizerSource>, Box<dyn Error>> {
    if preferred_label.is_empty() {
        return Ok(model_sources
            .iter()
            .find(|source| source.tokenizer.is_some()));
    }

    if let Some(source) = model_sources
        .iter()
        .find(|source| source.label == preferred_label)
    {
        if source.tokenizer.is_some() {
            return Ok(Some(source));
        }
        return Ok(model_sources
            .iter()
            .find(|candidate| candidate.tokenizer.is_some()));
    }

    Err(format!(
        "unknown overlay base tokenizer `{preferred_label}`; available: {}",
        model_sources
            .iter()
            .map(|source| source.label.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
    .into())
}

fn print_summary(primitive: &str, results: &[BakeoffRecord], vocab: &FaceoffVocab, args: &Args) {
    let total_docs = results.len();
    let induction_docs = results
        .iter()
        .filter(|record| record.corpus.split == CorpusSplit::Induction)
        .count();
    let held_out = held_out_records(results);
    let total_chars: usize = results.iter().map(|record| record.corpus.char_len).sum();
    let total_bytes: usize = results.iter().map(|record| record.corpus.byte_len).sum();
    let total_frontier: usize = results
        .iter()
        .map(|record| record.fractal.frontier_token_count)
        .sum();
    let held_out_frontier: usize = held_out
        .iter()
        .map(|record| record.fractal.frontier_token_count)
        .sum();
    let exact_motif_hit_docs = held_out
        .iter()
        .filter(|record| record.fractal.fallback_exact_motif_hits > 0)
        .count();
    let prototype_hit_docs = held_out
        .iter()
        .filter(|record| record.fractal.fallback_prototype_hits > 0)
        .count();
    let local_cache_hit_docs = held_out
        .iter()
        .filter(|record| record.fractal.fallback_local_cache_hits > 0)
        .count();
    let lexical_only_docs = held_out
        .iter()
        .filter(|record| {
            record.fractal.fallback_exact_motif_hits == 0
                && record.fractal.fallback_prototype_hits == 0
                && record.fractal.fallback_literal_hits == 0
                && record.fractal.fallback_shape_hits == 0
                && record.fractal.fallback_local_cache_hits == 0
                && record.fractal.fallback_lexical_fallback_tokens > 0
        })
        .count();

    println!("PRIMITIVE_START name={primitive}");
    println!("BAKEOFF_PRIMITIVE={primitive}");
    println!("BAKEOFF_CORPUS_SOURCE={}", args.corpus_source.as_str());
    println!("BAKEOFF_FALLBACK_MODE={}", args.fallback_mode.as_str());
    println!("BAKEOFF_IDENTITY_MODE={}", args.identity_mode.as_str());
    println!(
        "BAKEOFF_PROTOTYPE_GRANULARITY={}",
        match args.prototype_granularity {
            PrototypeGranularityMode::Coarse => "coarse",
            PrototypeGranularityMode::Adaptive => "adaptive",
        }
    );
    println!(
        "BAKEOFF_SUBSTRATE={}",
        match args.substrate_mode {
            TokenizerSubstrateMode::RawBytes => "raw",
            TokenizerSubstrateMode::LexicalAtoms => "lexical",
        }
    );
    println!("BAKEOFF_LOCAL_CACHE={}", args.local_cache_mode.as_str());
    println!("BAKEOFF_DOCUMENTS={total_docs}");
    println!("BAKEOFF_INDUCTION_DOCUMENTS={induction_docs}");
    println!("BAKEOFF_EVALUATION_DOCUMENTS={}", held_out.len());
    println!("BAKEOFF_CORPUS_BYTES={total_bytes}");
    println!("BAKEOFF_CORPUS_CHARS={total_chars}");
    println!("BAKEOFF_FRONTIER_TOKENS={total_frontier}");
    println!("BAKEOFF_EVALUATION_FRONTIER_TOKENS={held_out_frontier}");
    println!("BAKEOFF_VOCAB_MOTIFS={}", vocab.motif_count());
    println!("BAKEOFF_OUTPUT_DIR={}", args.output_dir.display());
    println!(
        "BAKEOFF_SPLIT_POLICY={}",
        match args.split_policy {
            SplitPolicy::Balanced => "balanced",
            SplitPolicy::BoundaryAware => "boundary_aware",
            SplitPolicy::SyntaxAware => "syntax_aware",
        }
    );
    println!("BAKEOFF_VERDICT_SCOPE=evaluation");
    println!(
        "BAKEOFF_DIAGNOSTIC split=evaluation exact_motif_hit_docs={} prototype_hit_docs={} local_cache_hit_docs={} lexical_only_docs={}",
        exact_motif_hit_docs, prototype_hit_docs, local_cache_hit_docs, lexical_only_docs
    );
    print_overlay_summary(results, args);

    let verdict = summarize_verdict(results);
    println!(
        "BAKEOFF_HARD_GATES split=evaluation roundtrip_failures={} chunk_utf8_failures={} collation_failures={} byte_fallback_docs={}",
        verdict.roundtrip_failures,
        verdict.chunk_utf8_failures,
        verdict.collation_failures,
        verdict.byte_fallback_docs
    );
    println!(
        "BAKEOFF_HEURISTICS split=evaluation suspicious_nonlog_overcollapse_docs={} weak_log_buckets={}",
        verdict.suspicious_nonlog_overcollapse_docs,
        verdict.weak_log_buckets
    );
    println!(
        "BAKEOFF_HYBRID_HEURISTICS split=evaluation external_structural_hit_docs={} external_code_buckets_below_parity={}",
        verdict.external_structural_hit_docs,
        verdict.external_code_buckets_below_parity
    );

    for family in summarize_families(results) {
        println!(
            "FAMILY_SUMMARY split=evaluation source_family={} docs={} median_best_ratio={:.2} median_motif_reuse={:.2} byte_fallback_docs={}",
            family.source_family.as_str(),
            family.doc_count,
            family.median_best_ratio,
            family.median_motif_reuse,
            family.byte_fallback_docs
        );
    }

    for bucket in summarize_buckets(results) {
        println!(
            "BUCKET_SUMMARY split=evaluation bucket={} docs={} median_best_ratio={:.2} median_motif_reuse={:.2} byte_fallback_docs={}",
            bucket.bucket,
            bucket.doc_count,
            bucket.median_best_ratio,
            bucket.median_motif_reuse,
            bucket.byte_fallback_docs
        );
    }

    let mut model_totals: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for record in &held_out {
        for (label, metric) in &record.models {
            let entry = model_totals.entry(label.as_str()).or_insert((0, 0, 0));
            if metric.status == "ok" {
                entry.0 += metric.native_token_count;
                entry.1 += record.corpus.char_len;
                entry.2 += 1;
            }
        }
    }

    for (label, (native_tokens, chars, count)) in model_totals {
        println!(
            "MODEL_SUMMARY split=evaluation label={label} docs={count} native_tokens={native_tokens} avg_chars_per_native_token={:.2}",
            chars as f64 / native_tokens.max(1) as f64
        );
    }

    println!("BAKEOFF_VERDICT={}", verdict.verdict.as_str());
    for reason in verdict.reasons {
        println!("BAKEOFF_VERDICT_REASON={reason}");
    }
    println!("PRIMITIVE_END name={primitive}");
}

#[derive(Clone, Debug)]
struct OverlayOfflineBucketSummary {
    bucket: String,
    role: OverlayBenchmarkBucketRole,
    doc_count: usize,
    activation_rate: f64,
    median_transport_ratio: f64,
    median_definition_overhead_rate: f64,
    median_client_overhead_ms: f64,
    p95_client_overhead_ms: f64,
}

fn build_overlay_offline_bucket_summary(
    bucket: String,
    overlays: &[&OverlayMetrics],
) -> OverlayOfflineBucketSummary {
    let mut transport_ratios = overlays
        .iter()
        .map(|overlay| overlay.batch_local_transport_ratio)
        .collect::<Vec<_>>();
    let mut definition_overhead = overlays
        .iter()
        .map(|overlay| overlay.batch_local_definition_overhead_rate)
        .collect::<Vec<_>>();
    let mut client_overhead_ms = overlays
        .iter()
        .map(|overlay| overlay.timing.client_overhead_ms)
        .collect::<Vec<_>>();
    transport_ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
    definition_overhead.sort_by(|left, right| left.partial_cmp(right).unwrap());
    client_overhead_ms.sort_by(|left, right| left.partial_cmp(right).unwrap());
    OverlayOfflineBucketSummary {
        role: overlay_bucket_role(&bucket),
        bucket,
        doc_count: overlays.len(),
        activation_rate: overlays
            .iter()
            .filter(|overlay| overlay.mode == "local-macro" && overlay.macro_ref_count > 0)
            .count() as f64
            / overlays.len().max(1) as f64,
        median_transport_ratio: median_sorted(&transport_ratios),
        median_definition_overhead_rate: median_sorted(&definition_overhead),
        median_client_overhead_ms: median_sorted(&client_overhead_ms),
        p95_client_overhead_ms: percentile_sorted(&client_overhead_ms, 0.95),
    }
}

fn overlay_offline_benchmark_verdict(
    bucket_summaries: &[OverlayOfflineBucketSummary],
    exact_failures: usize,
    overall_batch_transport_ratio: f64,
    overall_batch_definition_overhead_rate: f64,
) -> OverlayOfflineBenchmarkVerdict {
    if bucket_summaries.is_empty() {
        return OverlayOfflineBenchmarkVerdict::Inactive;
    }
    if exact_failures > 0 {
        return OverlayOfflineBenchmarkVerdict::Fail;
    }

    let primary_ok = bucket_summaries.iter().all(|summary| {
        if summary.role != OverlayBenchmarkBucketRole::PrimaryWin {
            return true;
        }
        match summary.bucket.as_str() {
            "jsonl.signals" => summary.median_transport_ratio >= 2.0,
            "logs.operational_mixed" => summary.median_transport_ratio >= 1.5,
            _ => true,
        }
    });
    let controls_ok = bucket_summaries.iter().all(|summary| {
        summary.role != OverlayBenchmarkBucketRole::NeutralControl
            || summary.median_transport_ratio >= 0.98
    });
    if !(primary_ok && controls_ok) {
        return OverlayOfflineBenchmarkVerdict::Fail;
    }
    if overall_batch_transport_ratio >= 1.40 && overall_batch_definition_overhead_rate <= 0.10 {
        return OverlayOfflineBenchmarkVerdict::Strong;
    }
    OverlayOfflineBenchmarkVerdict::Pass
}

fn print_overlay_summary(results: &[BakeoffRecord], args: &Args) {
    let held_out = held_out_records(results);
    let overlay_records = held_out
        .iter()
        .filter_map(|record| {
            record
                .overlay
                .as_ref()
                .map(|overlay| (&record.corpus.bucket, overlay))
        })
        .collect::<Vec<_>>();
    let ok_records = overlay_records
        .iter()
        .filter(|(_, overlay)| overlay.status == "ok")
        .collect::<Vec<_>>();

    println!("OVERLAY_MODE={}", args.overlay_mode.as_str());
    println!("OVERLAY_BASE_TOKENIZER={}", args.overlay_base_tokenizer);

    if ok_records.is_empty() {
        let skipped = overlay_records
            .iter()
            .filter(|(_, overlay)| overlay.status == "skipped")
            .count();
        println!(
            "OVERLAY_DIAGNOSTIC split=evaluation status=inactive ok_docs=0 skipped_docs={} activation_docs=0 macro_hit_docs=0 exact_failures=0",
            skipped
        );
        println!(
            "OVERLAY_OFFLINE_BENCHMARK split=evaluation verdict={} ok_docs=0 primary_win_buckets=0 neutral_control_buckets=0 exact_failures=0 overall_batch_transport_ratio=0.00 overall_batch_definition_overhead_rate=0.00 median_client_overhead_ms=0.00 p95_client_overhead_ms=0.00",
            OverlayOfflineBenchmarkVerdict::Inactive.as_str()
        );
        return;
    }

    let activation_docs = ok_records
        .iter()
        .filter(|(_, overlay)| overlay.mode == "local-macro" && overlay.macro_ref_count > 0)
        .count();
    let exact_failures = ok_records
        .iter()
        .filter(|(_, overlay)| !overlay.exact_ok)
        .count();
    let macro_hit_docs = ok_records
        .iter()
        .filter(|(_, overlay)| overlay.macro_ref_count > 0)
        .count();
    let total_canonical_tokens = ok_records
        .iter()
        .map(|(_, overlay)| overlay.canonical_token_count)
        .sum::<usize>();
    let total_transport_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.overlay_symbol_count)
        .sum::<usize>();
    let total_base_slice_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.base_slice_symbol_count)
        .sum::<usize>();
    let total_macro_ref_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.macro_ref_symbol_count)
        .sum::<usize>();
    let total_macro_definition_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.macro_definition_symbol_count)
        .sum::<usize>();
    println!(
        "OVERLAY_DIAGNOSTIC split=evaluation status=ok ok_docs={} skipped_docs=0 activation_docs={} macro_hit_docs={} exact_failures={}",
        ok_records.len(),
        activation_docs,
        macro_hit_docs,
        exact_failures
    );
    println!(
        "OVERLAY_TRANSPORT_SUMMARY split=evaluation scope=document_local docs={} canonical_tokens={} transport_symbols={} transport_ratio={:.2} base_slice_symbols={} macro_ref_symbols={} macro_definition_symbols={} definition_overhead_rate={:.2}",
        ok_records.len(),
        total_canonical_tokens,
        total_transport_symbols,
        total_canonical_tokens as f64 / total_transport_symbols.max(1) as f64,
        total_base_slice_symbols,
        total_macro_ref_symbols,
        total_macro_definition_symbols,
        total_macro_definition_symbols as f64 / total_transport_symbols.max(1) as f64
    );
    let total_batch_transport_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.batch_local_transport_symbols)
        .sum::<f64>();
    let total_batch_definition_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.batch_local_allocated_definition_symbols)
        .sum::<f64>();
    let overall_batch_transport_ratio =
        total_canonical_tokens as f64 / total_batch_transport_symbols.max(1.0);
    let overall_batch_definition_overhead_rate =
        total_batch_definition_symbols / total_batch_transport_symbols.max(1.0);
    println!(
        "OVERLAY_TRANSPORT_SUMMARY split=evaluation scope=batch_local docs={} canonical_tokens={} transport_symbols={:.2} transport_ratio={:.2} base_slice_symbols={} macro_ref_symbols={} macro_definition_symbols={:.2} definition_overhead_rate={:.2}",
        ok_records.len(),
        total_canonical_tokens,
        total_batch_transport_symbols,
        overall_batch_transport_ratio,
        total_base_slice_symbols,
        total_macro_ref_symbols,
        total_batch_definition_symbols,
        overall_batch_definition_overhead_rate
    );
    let total_sequential_pack_transport_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.sequential_pack_transport_symbols)
        .sum::<f64>();
    let total_sequential_pack_definition_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.sequential_pack_allocated_definition_symbols)
        .sum::<f64>();
    println!(
        "OVERLAY_PACKING_SUMMARY split=evaluation strategy=sequential max_pack_docs={} docs={} canonical_tokens={} transport_symbols={:.2} transport_ratio={:.2} base_slice_symbols={} macro_ref_symbols={} macro_definition_symbols={:.2} definition_overhead_rate={:.2}",
        DEFAULT_OVERLAY_PACK_DOCS,
        ok_records.len(),
        total_canonical_tokens,
        total_sequential_pack_transport_symbols,
        total_canonical_tokens as f64 / total_sequential_pack_transport_symbols.max(1.0),
        total_base_slice_symbols,
        total_macro_ref_symbols,
        total_sequential_pack_definition_symbols,
        total_sequential_pack_definition_symbols / total_sequential_pack_transport_symbols.max(1.0)
    );
    let total_structure_aware_pack_transport_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.structure_aware_pack_transport_symbols)
        .sum::<f64>();
    let total_structure_aware_pack_definition_symbols = ok_records
        .iter()
        .map(|(_, overlay)| overlay.structure_aware_pack_allocated_definition_symbols)
        .sum::<f64>();
    println!(
        "OVERLAY_PACKING_SUMMARY split=evaluation strategy=structure_aware max_pack_docs={} docs={} canonical_tokens={} transport_symbols={:.2} transport_ratio={:.2} base_slice_symbols={} macro_ref_symbols={} macro_definition_symbols={:.2} definition_overhead_rate={:.2}",
        DEFAULT_OVERLAY_PACK_DOCS,
        ok_records.len(),
        total_canonical_tokens,
        total_structure_aware_pack_transport_symbols,
        total_canonical_tokens as f64 / total_structure_aware_pack_transport_symbols.max(1.0),
        total_base_slice_symbols,
        total_macro_ref_symbols,
        total_structure_aware_pack_definition_symbols,
        total_structure_aware_pack_definition_symbols
            / total_structure_aware_pack_transport_symbols.max(1.0)
    );

    let mut per_bucket = BTreeMap::<String, Vec<&OverlayMetrics>>::new();
    for (bucket, overlay) in &ok_records {
        per_bucket
            .entry((*bucket).clone())
            .or_default()
            .push(*overlay);
    }

    let bucket_summaries = per_bucket
        .iter()
        .map(|(bucket, overlays)| build_overlay_offline_bucket_summary(bucket.clone(), overlays))
        .collect::<Vec<_>>();
    let offline_verdict = overlay_offline_benchmark_verdict(
        &bucket_summaries,
        exact_failures,
        overall_batch_transport_ratio,
        overall_batch_definition_overhead_rate,
    );
    let mut client_overheads = ok_records
        .iter()
        .map(|(_, overlay)| overlay.timing.client_overhead_ms)
        .collect::<Vec<_>>();
    client_overheads.sort_by(|left, right| left.partial_cmp(right).unwrap());
    println!(
        "OVERLAY_OFFLINE_BENCHMARK split=evaluation verdict={} ok_docs={} primary_win_buckets={} neutral_control_buckets={} exact_failures={} overall_batch_transport_ratio={:.2} overall_batch_definition_overhead_rate={:.2} median_client_overhead_ms={:.2} p95_client_overhead_ms={:.2}",
        offline_verdict.as_str(),
        ok_records.len(),
        bucket_summaries
            .iter()
            .filter(|summary| summary.role == OverlayBenchmarkBucketRole::PrimaryWin)
            .count(),
        bucket_summaries
            .iter()
            .filter(|summary| summary.role == OverlayBenchmarkBucketRole::NeutralControl)
            .count(),
        exact_failures,
        overall_batch_transport_ratio,
        overall_batch_definition_overhead_rate,
        median_sorted(&client_overheads),
        percentile_sorted(&client_overheads, 0.95)
    );

    for summary in &bucket_summaries {
        println!(
            "OVERLAY_OFFLINE_BUCKET split=evaluation role={} bucket={} docs={} activation_rate={:.2} median_transport_ratio={:.2} median_definition_overhead_rate={:.2} median_client_overhead_ms={:.2} p95_client_overhead_ms={:.2}",
            match summary.role {
                OverlayBenchmarkBucketRole::PrimaryWin => "primary_win",
                OverlayBenchmarkBucketRole::NeutralControl => "neutral_control",
                OverlayBenchmarkBucketRole::Other => "other",
            },
            summary.bucket,
            summary.doc_count,
            summary.activation_rate,
            summary.median_transport_ratio,
            summary.median_definition_overhead_rate,
            summary.median_client_overhead_ms,
            summary.p95_client_overhead_ms,
        );
    }

    for (bucket, overlays) in per_bucket {
        let mut ratios = overlays
            .iter()
            .map(|overlay| overlay.compression_ratio_vs_canonical)
            .collect::<Vec<_>>();
        let mut saved = overlays
            .iter()
            .map(|overlay| overlay.repeated_token_mass_saved as f64)
            .collect::<Vec<_>>();
        let mut definition_overhead = overlays
            .iter()
            .map(|overlay| {
                overlay.macro_definition_symbol_count as f64
                    / overlay.overlay_symbol_count.max(1) as f64
            })
            .collect::<Vec<_>>();
        let mut batch_ratios = overlays
            .iter()
            .map(|overlay| overlay.batch_local_transport_ratio)
            .collect::<Vec<_>>();
        let mut batch_definition_overhead = overlays
            .iter()
            .map(|overlay| overlay.batch_local_definition_overhead_rate)
            .collect::<Vec<_>>();
        let mut sequential_pack_ratios = overlays
            .iter()
            .map(|overlay| overlay.sequential_pack_transport_ratio)
            .collect::<Vec<_>>();
        let mut sequential_pack_definition_overhead = overlays
            .iter()
            .map(|overlay| overlay.sequential_pack_definition_overhead_rate)
            .collect::<Vec<_>>();
        let mut structure_aware_pack_ratios = overlays
            .iter()
            .map(|overlay| overlay.structure_aware_pack_transport_ratio)
            .collect::<Vec<_>>();
        let mut structure_aware_pack_definition_overhead = overlays
            .iter()
            .map(|overlay| overlay.structure_aware_pack_definition_overhead_rate)
            .collect::<Vec<_>>();
        ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
        saved.sort_by(|left, right| left.partial_cmp(right).unwrap());
        definition_overhead.sort_by(|left, right| left.partial_cmp(right).unwrap());
        batch_ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
        batch_definition_overhead.sort_by(|left, right| left.partial_cmp(right).unwrap());
        sequential_pack_ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
        sequential_pack_definition_overhead.sort_by(|left, right| left.partial_cmp(right).unwrap());
        structure_aware_pack_ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
        structure_aware_pack_definition_overhead
            .sort_by(|left, right| left.partial_cmp(right).unwrap());
        let activation_rate = overlays
            .iter()
            .filter(|overlay| overlay.mode == "local-macro" && overlay.macro_ref_count > 0)
            .count() as f64
            / overlays.len().max(1) as f64;
        println!(
            "OVERLAY_BUCKET_SUMMARY split=evaluation scope=document_local bucket={} docs={} median_ratio={:.2} activation_rate={:.2} median_saved_tokens={:.2} median_definition_overhead_rate={:.2}",
            bucket,
            overlays.len(),
            median_sorted(&ratios),
            activation_rate,
            median_sorted(&saved),
            median_sorted(&definition_overhead)
        );
        println!(
            "OVERLAY_BUCKET_SUMMARY split=evaluation scope=batch_local bucket={} docs={} median_ratio={:.2} activation_rate={:.2} median_saved_tokens={:.2} median_definition_overhead_rate={:.2}",
            bucket,
            overlays.len(),
            median_sorted(&batch_ratios),
            activation_rate,
            median_sorted(&saved),
            median_sorted(&batch_definition_overhead)
        );
        println!(
            "OVERLAY_PACKING_BUCKET_SUMMARY split=evaluation strategy=sequential max_pack_docs={} bucket={} docs={} median_ratio={:.2} activation_rate={:.2} median_saved_tokens={:.2} median_definition_overhead_rate={:.2}",
            DEFAULT_OVERLAY_PACK_DOCS,
            bucket,
            overlays.len(),
            median_sorted(&sequential_pack_ratios),
            activation_rate,
            median_sorted(&saved),
            median_sorted(&sequential_pack_definition_overhead)
        );
        println!(
            "OVERLAY_PACKING_BUCKET_SUMMARY split=evaluation strategy=structure_aware max_pack_docs={} bucket={} docs={} median_ratio={:.2} activation_rate={:.2} median_saved_tokens={:.2} median_definition_overhead_rate={:.2}",
            DEFAULT_OVERLAY_PACK_DOCS,
            bucket,
            overlays.len(),
            median_sorted(&structure_aware_pack_ratios),
            activation_rate,
            median_sorted(&saved),
            median_sorted(&structure_aware_pack_definition_overhead)
        );
    }
}

fn print_review_list(primitive: &str, results: &[BakeoffRecord], args: &Args) {
    let held_out = held_out_records(results);
    let mut review = held_out
        .iter()
        .map(|record| {
            let best_ratio = record
                .models
                .values()
                .filter(|metric| metric.status == "ok")
                .map(|metric| metric.compression_ratio_vs_native)
                .fold(f64::INFINITY, f64::min);
            (
                best_ratio,
                record.corpus.id.as_str(),
                record.corpus.bucket.as_str(),
                record.fractal.motif_reuse_count,
                record.fractal.fallback_exact_motif_hits,
                record.fractal.fallback_prototype_hits,
                record.fractal.fallback_literal_hits,
                record.fractal.fallback_shape_hits,
                record.fractal.fallback_lexical_fallback_tokens,
                record.fractal.roundtrip_ok,
                record.fractal.chunk_utf8_ok,
                record.fractal.collation_ok,
            )
        })
        .collect::<Vec<_>>();
    review.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "BAKEOFF_REVIEW_SET primitive={primitive} split=evaluation count={}",
        args.max_review_count.min(review.len())
    );
    for (
        ratio,
        id,
        bucket,
        motif_reuse,
        exact_hits,
        prototype_hits,
        literal_hits,
        shape_hits,
        lexical_tokens,
        roundtrip_ok,
        chunk_utf8_ok,
        collation_ok,
    ) in review.into_iter().take(args.max_review_count)
    {
        println!(
            "review primitive={primitive} fallback_mode={} id={id} bucket={bucket} best_ratio={ratio:.2} motif_reuse={motif_reuse} exact_hits={exact_hits} prototype_hits={prototype_hits} literal_hits={literal_hits} shape_hits={shape_hits} lexical_tokens={lexical_tokens} roundtrip={roundtrip_ok} chunk_utf8={chunk_utf8_ok} collation={collation_ok}",
            args.fallback_mode.as_str()
        );
    }
}

fn print_field_summary(runs: &[PrimitiveBakeoffRun], fallback_mode: FaceoffFallbackMode) {
    let mut digests = runs
        .iter()
        .map(|run| {
            let verdict = summarize_verdict(&run.results);
            PrimitiveRunDigest {
                primitive: run.primitive.clone(),
                fallback_mode,
                verdict: verdict.verdict,
                byte_fallback_docs: verdict.byte_fallback_docs,
                exact_motif_hit_docs: held_out_records(&run.results)
                    .iter()
                    .filter(|record| record.fractal.fallback_exact_motif_hits > 0)
                    .count(),
                prototype_hit_docs: held_out_records(&run.results)
                    .iter()
                    .filter(|record| record.fractal.fallback_prototype_hits > 0)
                    .count(),
                local_cache_hit_docs: held_out_records(&run.results)
                    .iter()
                    .filter(|record| record.fractal.fallback_local_cache_hits > 0)
                    .count(),
                lexical_only_docs: held_out_records(&run.results)
                    .iter()
                    .filter(|record| {
                        record.fractal.fallback_exact_motif_hits == 0
                            && record.fractal.fallback_prototype_hits == 0
                            && record.fractal.fallback_literal_hits == 0
                            && record.fractal.fallback_shape_hits == 0
                            && record.fractal.fallback_local_cache_hits == 0
                            && record.fractal.fallback_lexical_fallback_tokens > 0
                    })
                    .count(),
                logs_repetition_heavy_ratio: bucket_ratio(&run.results, "logs.repetition_heavy"),
                logs_operational_mixed_ratio: bucket_ratio(&run.results, "logs.operational_mixed"),
                jsonl_signals_ratio: bucket_ratio(&run.results, "jsonl.signals"),
                code_rust_ratio: bucket_ratio(&run.results, "code.rust"),
                code_swift_ratio: bucket_ratio(&run.results, "code.swift"),
                docs_spec_ratio: bucket_ratio(&run.results, "docs.spec"),
            }
        })
        .collect::<Vec<_>>();

    digests.sort_by(|left, right| {
        right
            .jsonl_signals_ratio
            .partial_cmp(&left.jsonl_signals_ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .logs_repetition_heavy_ratio
                    .partial_cmp(&left.logs_repetition_heavy_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.primitive.cmp(&right.primitive))
    });

    println!("FIELD_FALLBACK_MODE={}", fallback_mode.as_str());
    println!("FIELD_SUMMARY_COUNT={}", digests.len());
    for digest in digests {
        println!(
            "FIELD_SUMMARY primitive={} fallback_mode={} verdict={} byte_fallback_docs={} exact_motif_hit_docs={} prototype_hit_docs={} local_cache_hit_docs={} lexical_only_docs={} logs.repetition_heavy={:.2} logs.operational_mixed={:.2} jsonl.signals={:.2} code.rust={:.2} code.swift={:.2} docs.spec={:.2}",
            digest.primitive,
            digest.fallback_mode.as_str(),
            digest.verdict.as_str(),
            digest.byte_fallback_docs,
            digest.exact_motif_hit_docs,
            digest.prototype_hit_docs,
            digest.local_cache_hit_docs,
            digest.lexical_only_docs,
            digest.logs_repetition_heavy_ratio,
            digest.logs_operational_mixed_ratio,
            digest.jsonl_signals_ratio,
            digest.code_rust_ratio,
            digest.code_swift_ratio,
            digest.docs_spec_ratio
        );
    }
}

fn held_out_records(results: &[BakeoffRecord]) -> Vec<BakeoffRecord> {
    results
        .iter()
        .filter(|record| record.corpus.split == CorpusSplit::Evaluation)
        .cloned()
        .collect()
}

fn summarize_buckets(results: &[BakeoffRecord]) -> Vec<BucketSummary> {
    let results = held_out_records(results);
    let mut per_bucket = BTreeMap::<String, Vec<&BakeoffRecord>>::new();
    for record in &results {
        per_bucket
            .entry(record.corpus.bucket.clone())
            .or_default()
            .push(record);
    }

    per_bucket
        .into_iter()
        .map(|(bucket, records)| {
            let mut ratios = records
                .iter()
                .filter_map(|record| best_ratio(record))
                .collect::<Vec<_>>();
            let mut reuses = records
                .iter()
                .map(|record| record.fractal.motif_reuse_count as f64)
                .collect::<Vec<_>>();
            ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
            reuses.sort_by(|left, right| left.partial_cmp(right).unwrap());

            BucketSummary {
                bucket,
                doc_count: records.len(),
                median_best_ratio: median_sorted(&ratios),
                median_motif_reuse: median_sorted(&reuses),
                byte_fallback_docs: records
                    .iter()
                    .filter(|record| record.fractal.fallback_byte_fallback_tokens > 0)
                    .count(),
            }
        })
        .collect()
}

fn summarize_families(results: &[BakeoffRecord]) -> Vec<FamilySummary> {
    let results = held_out_records(results);
    let mut per_family = BTreeMap::<SourceFamily, Vec<&BakeoffRecord>>::new();
    for record in &results {
        per_family
            .entry(record.corpus.source_family)
            .or_default()
            .push(record);
    }

    per_family
        .into_iter()
        .map(|(source_family, records)| {
            let mut ratios = records
                .iter()
                .filter_map(|record| best_ratio(record))
                .collect::<Vec<_>>();
            let mut reuses = records
                .iter()
                .map(|record| record.fractal.motif_reuse_count as f64)
                .collect::<Vec<_>>();
            ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
            reuses.sort_by(|left, right| left.partial_cmp(right).unwrap());

            FamilySummary {
                source_family,
                doc_count: records.len(),
                median_best_ratio: median_sorted(&ratios),
                median_motif_reuse: median_sorted(&reuses),
                byte_fallback_docs: records
                    .iter()
                    .filter(|record| record.fractal.fallback_byte_fallback_tokens > 0)
                    .count(),
            }
        })
        .collect()
}

fn summarize_verdict(results: &[BakeoffRecord]) -> VerdictSummary {
    let results = held_out_records(results);
    if results.is_empty() {
        return VerdictSummary {
            verdict: BakeoffVerdict::Red,
            roundtrip_failures: 0,
            chunk_utf8_failures: 0,
            collation_failures: 0,
            byte_fallback_docs: 0,
            suspicious_nonlog_overcollapse_docs: 0,
            weak_log_buckets: 0,
            external_structural_hit_docs: 0,
            external_code_buckets_below_parity: 0,
            reasons: vec!["evaluation_docs=0".to_string()],
        };
    }

    let roundtrip_failures = results
        .iter()
        .filter(|record| !record.fractal.roundtrip_ok)
        .count();
    let chunk_utf8_failures = results
        .iter()
        .filter(|record| !record.fractal.chunk_utf8_ok)
        .count();
    let collation_failures = results
        .iter()
        .filter(|record| !record.fractal.collation_ok)
        .count();
    let byte_fallback_docs = results
        .iter()
        .filter(|record| record.fractal.fallback_byte_fallback_tokens > 0)
        .count();
    let external_structural_hit_docs = results
        .iter()
        .filter(|record| {
            record.corpus.source_family == SourceFamily::ExternalHf
                && (record.fractal.fallback_exact_motif_hits > 0
                    || record.fractal.fallback_prototype_hits > 0)
        })
        .count();
    let suspicious_nonlog_overcollapse_docs = results
        .iter()
        .filter(|record| {
            !record.corpus.bucket.starts_with("logs.")
                && (record.fractal.motif_reuse_count > HELD_OUT_NONLOG_CAUTION_REUSE
                    || best_ratio(record)
                        .map(|ratio| ratio > HELD_OUT_NONLOG_CAUTION_RATIO)
                        .unwrap_or(false))
        })
        .count();

    let bucket_summaries = summarize_buckets(&results);
    let has_external = results
        .iter()
        .any(|record| record.corpus.source_family == SourceFamily::ExternalHf);
    let weak_log_buckets = bucket_summaries
        .iter()
        .filter(|bucket| bucket.bucket == "logs.repetition_heavy" && bucket.median_best_ratio < 5.0)
        .count();
    let external_code_buckets_below_parity = bucket_summaries
        .iter()
        .filter(|bucket| {
            matches!(
                bucket.bucket.as_str(),
                "external.code.python" | "external.code.js_ts"
            ) && bucket.median_best_ratio < 1.0
        })
        .count();

    let mut reasons = Vec::new();
    let verdict = if roundtrip_failures > 0 || chunk_utf8_failures > 0 || collation_failures > 0 {
        if roundtrip_failures > 0 {
            reasons.push(format!("roundtrip_failures={roundtrip_failures}"));
        }
        if chunk_utf8_failures > 0 {
            reasons.push(format!("chunk_utf8_failures={chunk_utf8_failures}"));
        }
        if collation_failures > 0 {
            reasons.push(format!("collation_failures={collation_failures}"));
        }
        BakeoffVerdict::Red
    } else if byte_fallback_docs > 0
        || suspicious_nonlog_overcollapse_docs > 0
        || weak_log_buckets > 0
        || (has_external && external_structural_hit_docs == 0)
        || (has_external && external_code_buckets_below_parity > 0)
    {
        if byte_fallback_docs > 0 {
            reasons.push(format!("byte_fallback_docs={byte_fallback_docs}"));
        }
        if suspicious_nonlog_overcollapse_docs > 0 {
            reasons.push(format!(
                "suspicious_nonlog_overcollapse_docs={suspicious_nonlog_overcollapse_docs}"
            ));
        }
        if weak_log_buckets > 0 {
            reasons.push(format!("weak_log_buckets={weak_log_buckets}"));
        }
        if has_external && external_structural_hit_docs == 0 {
            reasons.push("external_structural_hit_docs=0".to_string());
        }
        if has_external && external_code_buckets_below_parity > 0 {
            reasons.push(format!(
                "external_code_buckets_below_parity={external_code_buckets_below_parity}"
            ));
        }
        BakeoffVerdict::Yellow
    } else {
        reasons.push("hard_gates_clear".to_string());
        reasons.push("log_compression_is_strong".to_string());
        reasons.push("non_log_buckets_remain_restrained".to_string());
        BakeoffVerdict::Green
    };

    VerdictSummary {
        verdict,
        roundtrip_failures,
        chunk_utf8_failures,
        collation_failures,
        byte_fallback_docs,
        suspicious_nonlog_overcollapse_docs,
        weak_log_buckets,
        external_structural_hit_docs,
        external_code_buckets_below_parity,
        reasons,
    }
}

fn best_ratio(record: &BakeoffRecord) -> Option<f64> {
    record
        .models
        .values()
        .filter(|metric| metric.status == "ok")
        .map(|metric| metric.compression_ratio_vs_native)
        .min_by(|left, right| left.partial_cmp(right).unwrap())
}

fn bucket_ratio(results: &[BakeoffRecord], bucket: &str) -> f64 {
    summarize_buckets(results)
        .into_iter()
        .find(|summary| summary.bucket == bucket)
        .map(|summary| summary.median_best_ratio)
        .unwrap_or(0.0)
}

fn median_sorted(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let index = ((values.len() - 1) as f64 * clamped).ceil() as usize;
    values[index.min(values.len() - 1)]
}

fn write_jsonl<'a, I, T>(path: PathBuf, items: I) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = &'a T>,
    T: Serialize + 'a,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for item in items {
        serde_json::to_writer(&mut writer, item)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn encoded_cross_depth_motif_reuse_count(document: &EncodedDocument) -> usize {
    let mut motif_hits = BTreeMap::<String, HashSet<usize>>::new();
    for token in &document.tokens {
        if let fractal_tokenizer::EncodedTokenKind::Motif { digest } = &token.kind {
            motif_hits
                .entry(digest.clone())
                .or_default()
                .insert(token.depth);
        }
    }

    motif_hits
        .into_iter()
        .filter(|(_, depths)| depths.len() > 1)
        .count()
}

fn fnv1a64(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            output_dir: PathBuf::from("/tmp/out"),
            corpus_limit: DEFAULT_CORPUS_LIMIT,
            corpus_source: CorpusSourceMode::Local,
            fawx_root: PathBuf::from(DEFAULT_FAWX_ROOT),
            home_state_root: PathBuf::from(DEFAULT_HOME_STATE_ROOT),
            hf_datasets_endpoint: DEFAULT_HF_DATASETS_ENDPOINT.to_string(),
            max_review_count: 10,
            selected_primitives: Vec::new(),
            all_primitives: false,
            fallback_mode: FaceoffFallbackMode::Full,
            identity_mode: FaceoffIdentityMode::Legacy,
            prototype_granularity: PrototypeGranularityMode::Coarse,
            split_policy: SplitPolicy::BoundaryAware,
            substrate_mode: TokenizerSubstrateMode::RawBytes,
            local_cache_mode: FaceoffLocalCacheMode::Off,
            overlay_mode: OverlaySummaryMode::Off,
            overlay_base_tokenizer: "qwen25".to_string(),
        }
    }

    #[test]
    fn selected_primitives_default_to_dynamic_p1() {
        let args = base_args();
        let selected = selected_primitive_candidates(&args).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].name, "p1_fractal_hybrid_dyn-state-norm_v2");
    }

    #[test]
    fn parse_fallback_mode_defaults_to_full() {
        assert_eq!(base_args().fallback_mode, FaceoffFallbackMode::Full);
    }

    #[test]
    fn parse_corpus_source_defaults_to_local() {
        assert_eq!(base_args().corpus_source, CorpusSourceMode::Local);
    }

    #[test]
    fn parse_corpus_source_accepts_hybrid() {
        assert_eq!(
            parse_corpus_source("hybrid").unwrap(),
            CorpusSourceMode::Hybrid
        );
    }

    #[test]
    fn parse_corpus_source_rejects_unknown_value() {
        let error = parse_corpus_source("totally-fake")
            .expect_err("unknown corpus source should error")
            .to_string();
        assert!(error.contains("unknown corpus source `totally-fake`"));
    }

    #[test]
    fn parse_from_uses_hybrid_default_corpus_limit() {
        let args = Args::parse_from(["--corpus-source", "hybrid"]).unwrap();
        assert_eq!(args.corpus_source, CorpusSourceMode::Hybrid);
        assert_eq!(args.corpus_limit, DEFAULT_HYBRID_CORPUS_LIMIT);
    }

    #[test]
    fn parse_from_keeps_explicit_corpus_limit_in_hybrid_mode() {
        let args = Args::parse_from(["--corpus-source", "hybrid", "--corpus-limit", "64"]).unwrap();
        assert_eq!(args.corpus_limit, 64);
    }

    #[test]
    fn parse_fallback_mode_accepts_motif_only() {
        assert_eq!(
            parse_fallback_mode("motif-only").unwrap(),
            FaceoffFallbackMode::MotifOnly
        );
    }

    #[test]
    fn parse_fallback_mode_rejects_unknown_value() {
        let error = parse_fallback_mode("totally-fake")
            .expect_err("unknown fallback mode should error")
            .to_string();
        assert!(error.contains("unknown fallback mode `totally-fake`"));
    }

    #[test]
    fn parse_identity_mode_defaults_to_legacy() {
        assert_eq!(base_args().identity_mode, FaceoffIdentityMode::Legacy);
    }

    #[test]
    fn parse_identity_mode_accepts_prototype_primary() {
        assert_eq!(
            parse_identity_mode("prototype-primary").unwrap(),
            FaceoffIdentityMode::PrototypePrimary
        );
    }

    #[test]
    fn parse_identity_mode_rejects_unknown_value() {
        let error = parse_identity_mode("totally-fake")
            .expect_err("unknown identity mode should error")
            .to_string();
        assert!(error.contains("unknown identity mode `totally-fake`"));
    }

    #[test]
    fn parse_prototype_granularity_defaults_to_coarse() {
        assert_eq!(
            base_args().prototype_granularity,
            PrototypeGranularityMode::Coarse
        );
    }

    #[test]
    fn parse_prototype_granularity_accepts_adaptive() {
        assert_eq!(
            parse_prototype_granularity("adaptive").unwrap(),
            PrototypeGranularityMode::Adaptive
        );
    }

    #[test]
    fn parse_prototype_granularity_rejects_unknown_value() {
        let error = parse_prototype_granularity("totally-fake")
            .expect_err("unknown prototype granularity should error")
            .to_string();
        assert!(error.contains("unknown prototype granularity `totally-fake`"));
    }

    #[test]
    fn parse_substrate_defaults_to_raw() {
        assert_eq!(base_args().substrate_mode, TokenizerSubstrateMode::RawBytes);
    }

    #[test]
    fn parse_substrate_accepts_lexical() {
        assert_eq!(
            parse_substrate_mode("lexical").unwrap(),
            TokenizerSubstrateMode::LexicalAtoms
        );
    }

    #[test]
    fn parse_substrate_rejects_unknown_value() {
        let error = parse_substrate_mode("totally-fake")
            .expect_err("unknown substrate mode should error")
            .to_string();
        assert!(error.contains("unknown substrate mode `totally-fake`"));
    }

    #[test]
    fn parse_split_policy_defaults_to_boundary_aware() {
        assert_eq!(base_args().split_policy, SplitPolicy::BoundaryAware);
    }

    #[test]
    fn parse_split_policy_accepts_syntax_aware() {
        assert_eq!(
            parse_split_policy("syntax-aware").unwrap(),
            SplitPolicy::SyntaxAware
        );
    }

    #[test]
    fn parse_split_policy_rejects_unknown_value() {
        let error = parse_split_policy("totally-fake")
            .expect_err("unknown split policy should error")
            .to_string();
        assert!(error.contains("unknown split policy `totally-fake`"));
    }

    #[test]
    fn parse_local_cache_defaults_to_off() {
        assert_eq!(base_args().local_cache_mode, FaceoffLocalCacheMode::Off);
    }

    #[test]
    fn parse_overlay_mode_defaults_to_off() {
        assert_eq!(base_args().overlay_mode, OverlaySummaryMode::Off);
    }

    #[test]
    fn parse_overlay_mode_accepts_local_line_macro() {
        assert_eq!(
            parse_overlay_mode("local-line-macro").unwrap(),
            OverlaySummaryMode::LocalLineMacro
        );
    }

    #[test]
    fn parse_overlay_mode_accepts_local_record_macro() {
        assert_eq!(
            parse_overlay_mode("local-record-macro").unwrap(),
            OverlaySummaryMode::LocalRecordMacro
        );
    }

    #[test]
    fn parse_overlay_mode_rejects_unknown_value() {
        let error = parse_overlay_mode("totally-fake")
            .expect_err("unknown overlay mode should error")
            .to_string();
        assert!(error.contains("unknown overlay mode `totally-fake`"));
    }

    #[test]
    fn parse_local_cache_accepts_exact() {
        assert_eq!(
            parse_local_cache_mode("exact").unwrap(),
            FaceoffLocalCacheMode::ExactSpan
        );
    }

    #[test]
    fn parse_local_cache_rejects_unknown_value() {
        let error = parse_local_cache_mode("totally-fake")
            .expect_err("unknown local cache mode should error")
            .to_string();
        assert!(error.contains("unknown local cache mode `totally-fake`"));
    }

    #[test]
    fn selected_primitives_all_returns_revived_field_plus_dynamic() {
        let mut args = base_args();
        args.all_primitives = true;
        let selected = selected_primitive_candidates(&args).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.name)
                .collect::<Vec<_>>(),
            vec![
                "b1_fractal_gated_v1",
                "p1_fractal_hybrid_v1",
                "p2_mandelbrot_v1",
                "b3_fractal_hierarchical_v1",
                "b4_universal_v1",
                "p1_contractive_v1",
                "p1_fractal_hybrid_composite_v1",
                "logistic_chaotic_map_v1",
                "p3_hierarchical_v1",
                "b2_stable_hierarchical_v1",
                "ifs_dyn-radius-depth_v1",
                "generalized_mobius_dyn-jitter-norm_v2",
                "julia_recursive_escape_v1",
                "mandelbox_recursive_dyn-escape-radius_v1",
                "p1_fractal_hybrid_dyn-state-norm_v2",
            ]
        );
    }

    #[test]
    fn selected_primitives_reject_unknown_names() {
        let mut args = base_args();
        args.selected_primitives = vec!["totally_fake_v1".to_string()];
        let error = selected_primitive_candidates(&args)
            .expect_err("unknown primitive should error")
            .to_string();
        assert!(error.contains("unknown primitive `totally_fake_v1`"));
    }

    #[test]
    fn split_inclusive_lines_preserves_trailing_segment() {
        let text = "alpha\nbeta\ngamma";
        let lines = split_inclusive_lines(text);
        assert_eq!(lines, vec!["alpha\n", "beta\n", "gamma"]);
        assert_eq!(line_count(text), 3);
    }

    #[test]
    fn prefix_lines_to_char_limit_is_deterministic() {
        let text = "aaaa\nbbbb\ncccc\ndddd\n";
        let first = prefix_lines_to_char_limit(text, 7);
        let second = prefix_lines_to_char_limit(text, 7);
        assert_eq!(first, second);
        assert_eq!(first.0, "aaaa\n");
        assert_eq!(first.1, 1);
    }

    #[test]
    fn round_robin_extend_is_stable() {
        let make = |label: &str, idx: usize| CorpusCandidate {
            corpus: build_corpus_document(
                format!("{label}-{idx}"),
                label.to_string(),
                SourceFamily::LocalFawx,
                CorpusSplit::Induction,
                "/tmp/source".to_string(),
                1..=1,
                format!("{label}-{idx}"),
            ),
            split_group_key: "/tmp/source".to_string(),
        };

        let mut out = Vec::new();
        round_robin_extend(
            vec![
                vec![make("a", 1), make("a", 2)],
                vec![make("b", 1), make("b", 2)],
            ],
            &mut out,
            3,
        );

        assert_eq!(
            out.into_iter()
                .map(|candidate| candidate.corpus.id)
                .collect::<Vec<_>>(),
            vec!["a-1", "b-1", "a-2"]
        );
    }

    #[test]
    fn line_window_documents_is_bounded_and_ordered() {
        let text = (1..=7)
            .map(|line| format!("line-{line}\n"))
            .collect::<String>();
        let docs = line_window_documents(
            &text,
            Path::new("/tmp/server.log"),
            "logs.repetition_heavy",
            2,
            "logs-server",
            2,
        );

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].corpus.start_line, 1);
        assert_eq!(docs[0].corpus.end_line, 2);
        assert_eq!(docs[1].corpus.start_line, 3);
        assert_eq!(docs[1].corpus.end_line, 4);
        assert!(docs[0].corpus.text.starts_with("line-1"));
        assert!(docs[1].corpus.text.starts_with("line-3"));
    }

    #[test]
    fn assign_balanced_splits_keeps_same_source_together() {
        let mut candidates = vec![
            CorpusCandidate {
                corpus: build_corpus_document(
                    "a-1".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/a.rs".to_string(),
                    1..=10,
                    "fn a() {}\n".to_string(),
                ),
                split_group_key: "/tmp/a.rs".to_string(),
            },
            CorpusCandidate {
                corpus: build_corpus_document(
                    "a-2".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/a.rs".to_string(),
                    11..=20,
                    "fn b() {}\n".to_string(),
                ),
                split_group_key: "/tmp/a.rs".to_string(),
            },
            CorpusCandidate {
                corpus: build_corpus_document(
                    "b-1".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/b.rs".to_string(),
                    1..=10,
                    "fn c() {}\n".to_string(),
                ),
                split_group_key: "/tmp/b.rs".to_string(),
            },
        ];

        assign_balanced_splits(&mut candidates);

        assert_eq!(candidates[0].corpus.split, candidates[1].corpus.split);
        assert_ne!(candidates[0].corpus.split, candidates[2].corpus.split);
    }

    #[test]
    fn assign_balanced_splits_balances_window_groups() {
        let mut candidates = (0..8)
            .map(|index| CorpusCandidate {
                corpus: build_corpus_document(
                    format!("logs-{index}"),
                    "logs.repetition_heavy".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/server.log".to_string(),
                    (index + 1)..=(index + 1),
                    format!("line-{index}\n"),
                ),
                split_group_key: format!("/tmp/server.log#group-{}", index / 2),
            })
            .collect::<Vec<_>>();

        assign_balanced_splits(&mut candidates);

        let induction = candidates
            .iter()
            .filter(|candidate| candidate.corpus.split == CorpusSplit::Induction)
            .count();
        let evaluation = candidates.len() - induction;

        assert_eq!(induction, 4);
        assert_eq!(evaluation, 4);
        assert_eq!(candidates[0].corpus.split, candidates[1].corpus.split);
        assert_eq!(candidates[2].corpus.split, candidates[3].corpus.split);
    }

    #[test]
    fn assign_balanced_splits_keeps_families_balanced_independently() {
        let mut candidates = vec![
            CorpusCandidate {
                corpus: build_corpus_document(
                    "local-1".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/local-a.rs".to_string(),
                    1..=1,
                    "fn a() {}\n".to_string(),
                ),
                split_group_key: "local-a".to_string(),
            },
            CorpusCandidate {
                corpus: build_corpus_document(
                    "local-2".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::LocalFawx,
                    CorpusSplit::Induction,
                    "/tmp/local-b.rs".to_string(),
                    1..=1,
                    "fn b() {}\n".to_string(),
                ),
                split_group_key: "local-b".to_string(),
            },
            CorpusCandidate {
                corpus: build_corpus_document(
                    "external-1".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::ExternalHf,
                    CorpusSplit::Induction,
                    "hf://datasets/demo/a".to_string(),
                    1..=1,
                    "fn c() {}\n".to_string(),
                ),
                split_group_key: "external-a".to_string(),
            },
            CorpusCandidate {
                corpus: build_corpus_document(
                    "external-2".to_string(),
                    "code.rust".to_string(),
                    SourceFamily::ExternalHf,
                    CorpusSplit::Induction,
                    "hf://datasets/demo/b".to_string(),
                    1..=1,
                    "fn d() {}\n".to_string(),
                ),
                split_group_key: "external-b".to_string(),
            },
        ];

        assign_balanced_splits(&mut candidates);

        let local_induction = candidates
            .iter()
            .filter(|candidate| {
                candidate.corpus.source_family == SourceFamily::LocalFawx
                    && candidate.corpus.split == CorpusSplit::Induction
            })
            .count();
        let external_induction = candidates
            .iter()
            .filter(|candidate| {
                candidate.corpus.source_family == SourceFamily::ExternalHf
                    && candidate.corpus.split == CorpusSplit::Induction
            })
            .count();
        assert_eq!(local_induction, 1);
        assert_eq!(external_induction, 1);
    }

    #[test]
    fn distribute_count_spreads_remainder_to_early_buckets() {
        assert_eq!(distribute_count(10, 4), vec![3, 3, 2, 2]);
    }

    #[test]
    fn evenly_spaced_offsets_are_unique_and_bounded() {
        let offsets = evenly_spaced_offsets(1000, 7);
        assert_eq!(offsets.len(), 7);
        assert!(offsets.windows(2).all(|window| window[0] < window[1]));
        assert!(offsets.iter().all(|offset| *offset < 1000));
    }

    #[test]
    fn select_config_for_slice_picks_latest_suffix_match() {
        let splits = vec![
            DatasetSplitRef {
                config: "20231101.es".to_string(),
                split: "train".to_string(),
            },
            DatasetSplitRef {
                config: "20240101.es".to_string(),
                split: "train".to_string(),
            },
            DatasetSplitRef {
                config: "20240101.ja".to_string(),
                split: "train".to_string(),
            },
        ];

        let config = select_config_for_slice(
            "wikimedia/wikipedia",
            &ExternalConfigSelector::LatestSuffix(".es"),
            "train",
            &splits,
        )
        .unwrap();

        assert_eq!(config, "20240101.es");
    }

    #[test]
    fn external_source_key_prefers_requested_fields() {
        let row = serde_json::json!({
            "repo_name": "octocat/demo",
            "path": "src/main.py",
            "code": "print('hi')",
        });

        assert_eq!(
            external_source_key(&row, &["repo_name", "path"]).as_deref(),
            Some("octocat/demo::src/main.py")
        );
    }

    #[derive(Clone, Copy)]
    struct FakeRecordMetrics {
        ratio: f64,
        motif_reuse_count: usize,
        roundtrip_ok: bool,
        chunk_utf8_ok: bool,
        collation_ok: bool,
        byte_fallback_tokens: usize,
        exact_hits: usize,
        prototype_hits: usize,
    }

    fn fake_record(
        bucket: &str,
        split: CorpusSplit,
        source_family: SourceFamily,
        metrics: FakeRecordMetrics,
    ) -> BakeoffRecord {
        let mut models = BTreeMap::new();
        models.insert(
            "llama31".to_string(),
            ModelMetrics {
                model_label: "llama31".to_string(),
                tokenizer_json_path: "/tmp/tokenizer.json".to_string(),
                status: "ok".to_string(),
                native_token_count: 100,
                avg_chars_per_native_token: 4.0,
                compression_ratio_vs_native: metrics.ratio,
                native_chunk_count: 1,
                retokenize_ms: 1.0,
                collate_ms: 1.0,
                collation_ok: metrics.collation_ok,
            },
        );

        BakeoffRecord {
            corpus: CorpusDocument {
                id: format!("{bucket}-doc"),
                bucket: bucket.to_string(),
                source_family,
                split,
                source_path: "/tmp/source".to_string(),
                start_line: 1,
                end_line: 1,
                byte_len: 100,
                char_len: 100,
                text: "sample".to_string(),
            },
            fractal: FractalMetrics {
                input_bytes: 100,
                input_chars: 100,
                frontier_token_count: 10,
                chunk_count: 1,
                avg_chars_per_frontier_token: 10.0,
                motif_reuse_count: metrics.motif_reuse_count,
                fallback_motif_hits: 0,
                fallback_exact_motif_hits: metrics.exact_hits,
                fallback_prototype_hits: metrics.prototype_hits,
                fallback_literal_hits: 0,
                fallback_shape_hits: 0,
                fallback_unknown_motifs: 0,
                fallback_recursed_to_children: 0,
                fallback_local_cache_hits: 0,
                fallback_local_cache_stores: 0,
                fallback_lexical_fallback_tokens: 0,
                fallback_byte_fallback_tokens: metrics.byte_fallback_tokens,
                roundtrip_ok: metrics.roundtrip_ok,
                chunk_utf8_ok: metrics.chunk_utf8_ok,
                collation_ok: metrics.collation_ok,
                wall_time_ms: 1.0,
            },
            overlay: None,
            models,
        }
    }

    fn fake_overlay_metrics(
        batch_local_transport_ratio: f64,
        batch_local_definition_overhead_rate: f64,
        client_overhead_ms: f64,
        macro_ref_count: usize,
    ) -> OverlayMetrics {
        OverlayMetrics {
            status: "ok".to_string(),
            base_tokenizer_label: "qwen25".to_string(),
            base_tokenizer_path: "/tmp/tokenizer.json".to_string(),
            mode: "local-macro".to_string(),
            canonical_token_count: 100,
            base_slice_symbol_count: 60,
            macro_ref_symbol_count: 10,
            macro_definition_symbol_count: 15,
            overlay_symbol_count: 85,
            compression_ratio_vs_canonical: 1.18,
            macro_count: 2,
            macro_ref_count,
            repeated_token_mass_saved: 15,
            batch_local_transport_symbols: 100.0 / batch_local_transport_ratio,
            batch_local_transport_ratio,
            batch_local_allocated_definition_symbols: batch_local_definition_overhead_rate
                * (100.0 / batch_local_transport_ratio),
            batch_local_definition_overhead_rate,
            sequential_pack_transport_symbols: 0.0,
            sequential_pack_transport_ratio: 0.0,
            sequential_pack_allocated_definition_symbols: 0.0,
            sequential_pack_definition_overhead_rate: 0.0,
            structure_aware_pack_transport_symbols: 0.0,
            structure_aware_pack_transport_ratio: 0.0,
            structure_aware_pack_allocated_definition_symbols: 0.0,
            structure_aware_pack_definition_overhead_rate: 0.0,
            timing: OverlayTimingMetrics {
                discovery_ms: 4.0,
                allocated_pack_ms: 1.0,
                allocated_materialize_ms: 0.5,
                client_overhead_ms,
            },
            exact_ok: true,
        }
    }

    #[test]
    fn overlay_offline_benchmark_is_strong_for_target_wins_and_neutral_controls() {
        let summaries = vec![
            build_overlay_offline_bucket_summary(
                "jsonl.signals".to_string(),
                &[&fake_overlay_metrics(3.2, 0.07, 7.0, 2)],
            ),
            build_overlay_offline_bucket_summary(
                "logs.operational_mixed".to_string(),
                &[&fake_overlay_metrics(2.1, 0.08, 8.0, 2)],
            ),
            build_overlay_offline_bucket_summary(
                "docs.spec".to_string(),
                &[&fake_overlay_metrics(1.0, 0.02, 2.0, 0)],
            ),
            build_overlay_offline_bucket_summary(
                "external.prose.web".to_string(),
                &[&fake_overlay_metrics(1.0, 0.01, 2.0, 0)],
            ),
        ];

        let verdict = overlay_offline_benchmark_verdict(&summaries, 0, 1.43, 0.07);

        assert_eq!(verdict, OverlayOfflineBenchmarkVerdict::Strong);
    }

    #[test]
    fn overlay_offline_benchmark_fails_on_control_regression() {
        let summaries = vec![
            build_overlay_offline_bucket_summary(
                "jsonl.signals".to_string(),
                &[&fake_overlay_metrics(2.5, 0.08, 7.0, 2)],
            ),
            build_overlay_offline_bucket_summary(
                "logs.operational_mixed".to_string(),
                &[&fake_overlay_metrics(1.6, 0.09, 8.0, 2)],
            ),
            build_overlay_offline_bucket_summary(
                "external.prose.web".to_string(),
                &[&fake_overlay_metrics(0.90, 0.01, 2.0, 0)],
            ),
        ];

        let verdict = overlay_offline_benchmark_verdict(&summaries, 0, 1.38, 0.09);

        assert_eq!(verdict, OverlayOfflineBenchmarkVerdict::Fail);
    }

    #[test]
    fn verdict_is_red_on_hard_gate_failure() {
        let results = vec![fake_record(
            "logs.repetition_heavy",
            CorpusSplit::Evaluation,
            SourceFamily::LocalFawx,
            FakeRecordMetrics {
                ratio: 8.0,
                motif_reuse_count: 3,
                roundtrip_ok: false,
                chunk_utf8_ok: true,
                collation_ok: true,
                byte_fallback_tokens: 0,
                exact_hits: 0,
                prototype_hits: 0,
            },
        )];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Red);
        assert_eq!(verdict.roundtrip_failures, 1);
    }

    #[test]
    fn verdict_is_yellow_on_weak_logs_or_suspicious_nonlogs() {
        let weak_logs = vec![fake_record(
            "logs.repetition_heavy",
            CorpusSplit::Evaluation,
            SourceFamily::LocalFawx,
            FakeRecordMetrics {
                ratio: 3.0,
                motif_reuse_count: 1,
                roundtrip_ok: true,
                chunk_utf8_ok: true,
                collation_ok: true,
                byte_fallback_tokens: 0,
                exact_hits: 0,
                prototype_hits: 0,
            },
        )];
        let suspicious_nonlogs = vec![
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 8.0,
                    motif_reuse_count: 3,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "code.rust",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 6.0,
                    motif_reuse_count: 4,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
        ];

        assert_eq!(
            summarize_verdict(&weak_logs).verdict,
            BakeoffVerdict::Yellow
        );
        assert_eq!(
            summarize_verdict(&suspicious_nonlogs).verdict,
            BakeoffVerdict::Yellow
        );
    }

    #[test]
    fn verdict_is_yellow_on_extreme_nonlog_held_out_compression() {
        let results = vec![
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 8.0,
                    motif_reuse_count: 1,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "docs.spec",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: HELD_OUT_NONLOG_CAUTION_RATIO + 1.0,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
        ];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Yellow);
        assert_eq!(verdict.suspicious_nonlog_overcollapse_docs, 1);
    }

    #[test]
    fn verdict_is_green_for_healthy_selective_run() {
        let results = vec![
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 8.0,
                    motif_reuse_count: 3,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "jsonl.signals",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 2.0,
                    motif_reuse_count: 1,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "code.rust",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 1.8,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "docs.spec",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 1.6,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
        ];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Green);
        assert!(verdict
            .reasons
            .iter()
            .any(|reason| reason == "hard_gates_clear"));
    }

    #[test]
    fn verdict_is_red_without_evaluation_docs() {
        let results = vec![fake_record(
            "logs.repetition_heavy",
            CorpusSplit::Induction,
            SourceFamily::LocalFawx,
            FakeRecordMetrics {
                ratio: 8.0,
                motif_reuse_count: 1,
                roundtrip_ok: true,
                chunk_utf8_ok: true,
                collation_ok: true,
                byte_fallback_tokens: 0,
                exact_hits: 0,
                prototype_hits: 0,
            },
        )];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Red);
        assert!(verdict
            .reasons
            .iter()
            .any(|reason| reason == "evaluation_docs=0"));
    }

    #[test]
    fn verdict_ignores_induction_docs_when_evaluation_is_clean() {
        let results = vec![
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Induction,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 1.0,
                    motif_reuse_count: 6,
                    roundtrip_ok: false,
                    chunk_utf8_ok: false,
                    collation_ok: false,
                    byte_fallback_tokens: 4,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 8.0,
                    motif_reuse_count: 1,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
        ];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Green);
        assert_eq!(verdict.roundtrip_failures, 0);
        assert_eq!(verdict.chunk_utf8_failures, 0);
        assert_eq!(verdict.collation_failures, 0);
        assert_eq!(verdict.byte_fallback_docs, 0);
    }

    #[test]
    fn verdict_is_yellow_for_hybrid_without_external_structural_hits() {
        let results = vec![
            fake_record(
                "external.code.python",
                CorpusSplit::Evaluation,
                SourceFamily::ExternalHf,
                FakeRecordMetrics {
                    ratio: 0.9,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "external.code.js_ts",
                CorpusSplit::Evaluation,
                SourceFamily::ExternalHf,
                FakeRecordMetrics {
                    ratio: 0.95,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
            fake_record(
                "logs.repetition_heavy",
                CorpusSplit::Evaluation,
                SourceFamily::LocalFawx,
                FakeRecordMetrics {
                    ratio: 8.0,
                    motif_reuse_count: 0,
                    roundtrip_ok: true,
                    chunk_utf8_ok: true,
                    collation_ok: true,
                    byte_fallback_tokens: 0,
                    exact_hits: 0,
                    prototype_hits: 0,
                },
            ),
        ];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Yellow);
        assert_eq!(verdict.external_structural_hit_docs, 0);
        assert_eq!(verdict.external_code_buckets_below_parity, 2);
    }
}
