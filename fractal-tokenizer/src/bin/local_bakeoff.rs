use burn::backend::Candle;
use fractal_tokenizer::{
    EncodedDocument, FaceoffChunkLimits, FaceoffEmissionPolicy, FaceoffTokenizer, FaceoffVocab,
    HuggingFaceNativeTokenizer, ModelFacingBatch, ModelFacingDocument, NativeCollationSpec,
    NativeCompatibilityAdapter,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashSet},
    env,
    error::Error,
    fs,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

type Backend = Candle<f32, i64>;

const DEFAULT_OUTPUT_DIR: &str =
    "/Users/joseph/fractal-tokenizer-checkout/fractal-tokenizer/benchmarks/.local";
const DEFAULT_FAWX_ROOT: &str = "/Users/joseph/fawx";
const DEFAULT_HOME_STATE_ROOT: &str = "/Users/joseph/.fawx";
const DEFAULT_CORPUS_LIMIT: usize = 120;
const DEFAULT_LOG_WINDOW_LINES: usize = 120;
const DEFAULT_JSON_WINDOW_LINES: usize = 100;
const DEFAULT_PAD_MULTIPLE: usize = 8;
const DEFAULT_CHUNK_LIMIT_TOKENS: usize = 8;
const DEFAULT_CHUNK_LIMIT_BYTES: usize = 4096;

#[derive(Clone, Debug, Serialize)]
struct CorpusDocument {
    id: String,
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
    fallback_unknown_motifs: usize,
    fallback_recursed_to_children: usize,
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
    models: BTreeMap<String, ModelMetrics>,
}

#[derive(Clone, Debug)]
struct CorpusCandidate {
    corpus: CorpusDocument,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerdictSummary {
    verdict: BakeoffVerdict,
    roundtrip_failures: usize,
    chunk_utf8_failures: usize,
    collation_failures: usize,
    byte_fallback_docs: usize,
    suspicious_nonlog_overcollapse_docs: usize,
    weak_log_buckets: usize,
    reasons: Vec<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.output_dir)?;

    let corpus = build_corpus(&args)?;
    write_jsonl(
        args.output_dir.join("local_bakeoff_corpus.jsonl"),
        corpus.iter().map(|candidate| &candidate.corpus),
    )?;

    let (works, vocab) = build_fractal_documents(&corpus, &args)?;
    let model_sources = discover_model_tokenizers(&args)?;
    let model_results = run_model_bakeoff(&works, &model_sources)?;

    let results = merge_results(works, model_results);
    write_jsonl(
        args.output_dir.join("local_bakeoff_results.jsonl"),
        results.iter(),
    )?;

    print_summary(&results, &vocab, &args);
    print_review_list(&results, &args);

    Ok(())
}

struct Args {
    output_dir: PathBuf,
    corpus_limit: usize,
    fawx_root: PathBuf,
    home_state_root: PathBuf,
    max_review_count: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
        let mut corpus_limit = DEFAULT_CORPUS_LIMIT;
        let mut fawx_root = PathBuf::from(DEFAULT_FAWX_ROOT);
        let mut home_state_root = PathBuf::from(DEFAULT_HOME_STATE_ROOT);
        let mut max_review_count = 10usize;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--output-dir" => {
                    output_dir = PathBuf::from(args.next().ok_or("--output-dir requires a value")?);
                }
                "--corpus-limit" => {
                    corpus_limit = args
                        .next()
                        .ok_or("--corpus-limit requires a value")?
                        .parse()?;
                }
                "--fawx-root" => {
                    fawx_root = PathBuf::from(args.next().ok_or("--fawx-root requires a value")?);
                }
                "--home-state-root" => {
                    home_state_root =
                        PathBuf::from(args.next().ok_or("--home-state-root requires a value")?);
                }
                "--max-review-count" => {
                    max_review_count = args
                        .next()
                        .ok_or("--max-review-count requires a value")?
                        .parse()?;
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

        Ok(Self {
            output_dir,
            corpus_limit,
            fawx_root,
            home_state_root,
            max_review_count,
        })
    }
}

fn print_help() {
    eprintln!(
        "Usage: cargo run -p fractal-tokenizer --bin local_bakeoff -- [--output-dir DIR] [--corpus-limit N] [--fawx-root DIR] [--home-state-root DIR] [--max-review-count N]"
    );
}

fn build_corpus(args: &Args) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut seen = HashSet::<u64>::new();
    let buckets = vec![
        collect_log_candidates(&args.home_state_root)?,
        collect_jsonl_candidates(&args.home_state_root)?,
        collect_code_candidates(&args.fawx_root)?,
        collect_markdown_candidates(&args.fawx_root)?,
    ];

    let mut candidates = Vec::new();
    round_robin_extend(buckets, &mut candidates, args.corpus_limit * 2);

    let mut deduped = Vec::new();
    for candidate in candidates {
        let fingerprint = fnv1a64(&candidate.corpus.text);
        if seen.insert(fingerprint) {
            deduped.push(candidate);
        }
        if deduped.len() >= args.corpus_limit {
            break;
        }
    }

    if deduped.len() < args.corpus_limit {
        return Err(format!(
            "only built {} unique documents, wanted {}",
            deduped.len(),
            args.corpus_limit
        )
        .into());
    }

    Ok(deduped)
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
    round_robin_extend(buckets, &mut out, 36);
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
    round_robin_extend(buckets, &mut out, 24);
    Ok(out)
}

fn collect_code_candidates(fawx_root: &Path) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut buckets = Vec::new();

    let mut rust_candidates = Vec::new();
    for path in sorted_files_with_extension(&fawx_root.join("engine"), "rs")?
        .into_iter()
        .take(24)
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
        .take(12)
    {
        if let Ok(text) = fs::read_to_string(&path) {
            swift_candidates.push(prefix_file_document(&text, &path, "code.swift", 12_000));
        }
    }
    if !swift_candidates.is_empty() {
        buckets.push(swift_candidates);
    }

    let mut out = Vec::new();
    round_robin_extend(buckets, &mut out, 36);
    Ok(out)
}

fn collect_markdown_candidates(fawx_root: &Path) -> Result<Vec<CorpusCandidate>, Box<dyn Error>> {
    let mut candidates = Vec::new();

    for path in sorted_files_with_extension(&fawx_root.join("docs"), "md")?
        .into_iter()
        .take(24)
    {
        if let Ok(text) = fs::read_to_string(&path) {
            candidates.push(prefix_file_document(&text, &path, "docs.spec", 12_000));
        }
    }

    Ok(candidates)
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
                source_path.to_string_lossy().to_string(),
                start + 1,
                end,
                slice,
            ),
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
            source_path.to_string_lossy().to_string(),
            1,
            line_count(text),
            text.to_string(),
        ),
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
            source_path.to_string_lossy().to_string(),
            1,
            end_line,
            slice,
        ),
    }
}

fn build_corpus_document(
    id: String,
    bucket: String,
    source_path: String,
    start_line: usize,
    end_line: usize,
    text: String,
) -> CorpusDocument {
    let byte_len = text.len();
    let char_len = text.chars().count();
    CorpusDocument {
        id,
        bucket,
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

fn build_fractal_documents(
    corpus: &[CorpusCandidate],
    _args: &Args,
) -> Result<(Vec<DocumentWork>, FaceoffVocab), Box<dyn Error>> {
    let device = Default::default();
    let tokenizer = FaceoffTokenizer::new(Default::default());
    let texts = corpus
        .iter()
        .map(|candidate| candidate.corpus.text.as_str())
        .collect::<Vec<_>>();
    let vocab = tokenizer.induce_vocab_from_texts::<Backend>(&texts, &device)?;
    let limits = FaceoffChunkLimits::new(DEFAULT_CHUNK_LIMIT_TOKENS, DEFAULT_CHUNK_LIMIT_BYTES);
    let mut works = Vec::with_capacity(corpus.len());

    for candidate in corpus {
        let started = Instant::now();
        let encoded = tokenizer.encode_text_v2_with_policy::<Backend>(
            &candidate.corpus.text,
            &vocab,
            &device,
            FaceoffEmissionPolicy::NoveltyAware,
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
            fallback_unknown_motifs: encoded.fallback.unknown_motifs,
            fallback_recursed_to_children: encoded.fallback.recursed_to_children,
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

fn print_summary(results: &[BakeoffRecord], vocab: &FaceoffVocab, args: &Args) {
    let total_docs = results.len();
    let total_chars: usize = results.iter().map(|record| record.corpus.char_len).sum();
    let total_bytes: usize = results.iter().map(|record| record.corpus.byte_len).sum();
    let total_frontier: usize = results
        .iter()
        .map(|record| record.fractal.frontier_token_count)
        .sum();

    println!("BAKEOFF_DOCUMENTS={total_docs}");
    println!("BAKEOFF_CORPUS_BYTES={total_bytes}");
    println!("BAKEOFF_CORPUS_CHARS={total_chars}");
    println!("BAKEOFF_FRONTIER_TOKENS={total_frontier}");
    println!("BAKEOFF_VOCAB_MOTIFS={}", vocab.motif_count());
    println!("BAKEOFF_OUTPUT_DIR={}", args.output_dir.display());

    let verdict = summarize_verdict(results);
    println!(
        "BAKEOFF_HARD_GATES roundtrip_failures={} chunk_utf8_failures={} collation_failures={} byte_fallback_docs={}",
        verdict.roundtrip_failures,
        verdict.chunk_utf8_failures,
        verdict.collation_failures,
        verdict.byte_fallback_docs
    );
    println!(
        "BAKEOFF_HEURISTICS suspicious_nonlog_overcollapse_docs={} weak_log_buckets={}",
        verdict.suspicious_nonlog_overcollapse_docs,
        verdict.weak_log_buckets
    );

    for bucket in summarize_buckets(results) {
        println!(
            "BUCKET_SUMMARY bucket={} docs={} median_best_ratio={:.2} median_motif_reuse={:.2} byte_fallback_docs={}",
            bucket.bucket,
            bucket.doc_count,
            bucket.median_best_ratio,
            bucket.median_motif_reuse,
            bucket.byte_fallback_docs
        );
    }

    let mut model_totals: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for record in results {
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
            "MODEL_SUMMARY label={label} docs={count} native_tokens={native_tokens} avg_chars_per_native_token={:.2}",
            chars as f64 / native_tokens.max(1) as f64
        );
    }

    println!("BAKEOFF_VERDICT={}", verdict.verdict.as_str());
    for reason in verdict.reasons {
        println!("BAKEOFF_VERDICT_REASON={reason}");
    }
}

fn print_review_list(results: &[BakeoffRecord], args: &Args) {
    let mut review = results
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
        "BAKEOFF_REVIEW_SET={}",
        args.max_review_count.min(review.len())
    );
    for (ratio, id, bucket, motif_reuse, roundtrip_ok, chunk_utf8_ok, collation_ok) in
        review.into_iter().take(args.max_review_count)
    {
        println!(
            "review id={id} bucket={bucket} best_ratio={ratio:.2} motif_reuse={motif_reuse} roundtrip={roundtrip_ok} chunk_utf8={chunk_utf8_ok} collation={collation_ok}"
        );
    }
}

fn summarize_buckets(results: &[BakeoffRecord]) -> Vec<BucketSummary> {
    let mut per_bucket = BTreeMap::<String, Vec<&BakeoffRecord>>::new();
    for record in results {
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

fn summarize_verdict(results: &[BakeoffRecord]) -> VerdictSummary {
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
    let suspicious_nonlog_overcollapse_docs = results
        .iter()
        .filter(|record| {
            !record.corpus.bucket.starts_with("logs.")
                && record.fractal.motif_reuse_count > 2
                && best_ratio(record).map(|ratio| ratio > 5.0).unwrap_or(false)
        })
        .count();

    let bucket_summaries = summarize_buckets(results);
    let weak_log_buckets = bucket_summaries
        .iter()
        .filter(|bucket| {
            bucket.bucket == "logs.repetition_heavy" && bucket.median_best_ratio < 5.0
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
                "/tmp/source".to_string(),
                1,
                1,
                format!("{label}-{idx}"),
            ),
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

    fn fake_record(
        bucket: &str,
        ratio: f64,
        motif_reuse_count: usize,
        roundtrip_ok: bool,
        chunk_utf8_ok: bool,
        collation_ok: bool,
        byte_fallback_tokens: usize,
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
                compression_ratio_vs_native: ratio,
                native_chunk_count: 1,
                retokenize_ms: 1.0,
                collate_ms: 1.0,
                collation_ok,
            },
        );

        BakeoffRecord {
            corpus: CorpusDocument {
                id: format!("{bucket}-doc"),
                bucket: bucket.to_string(),
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
                motif_reuse_count,
                fallback_motif_hits: 0,
                fallback_unknown_motifs: 0,
                fallback_recursed_to_children: 0,
                fallback_byte_fallback_tokens: byte_fallback_tokens,
                roundtrip_ok,
                chunk_utf8_ok,
                collation_ok,
                wall_time_ms: 1.0,
            },
            models,
        }
    }

    #[test]
    fn verdict_is_red_on_hard_gate_failure() {
        let results = vec![fake_record(
            "logs.repetition_heavy",
            8.0,
            3,
            false,
            true,
            true,
            0,
        )];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Red);
        assert_eq!(verdict.roundtrip_failures, 1);
    }

    #[test]
    fn verdict_is_yellow_on_weak_logs_or_suspicious_nonlogs() {
        let weak_logs = vec![fake_record(
            "logs.repetition_heavy",
            3.0,
            1,
            true,
            true,
            true,
            0,
        )];
        let suspicious_nonlogs = vec![
            fake_record("logs.repetition_heavy", 8.0, 3, true, true, true, 0),
            fake_record("code.rust", 6.0, 4, true, true, true, 0),
        ];

        assert_eq!(summarize_verdict(&weak_logs).verdict, BakeoffVerdict::Yellow);
        assert_eq!(
            summarize_verdict(&suspicious_nonlogs).verdict,
            BakeoffVerdict::Yellow
        );
    }

    #[test]
    fn verdict_is_green_for_healthy_selective_run() {
        let results = vec![
            fake_record("logs.repetition_heavy", 8.0, 3, true, true, true, 0),
            fake_record("jsonl.signals", 2.0, 1, true, true, true, 0),
            fake_record("code.rust", 1.8, 0, true, true, true, 0),
            fake_record("docs.spec", 1.6, 0, true, true, true, 0),
        ];

        let verdict = summarize_verdict(&results);

        assert_eq!(verdict.verdict, BakeoffVerdict::Green);
        assert!(verdict.reasons.iter().any(|reason| reason == "hard_gates_clear"));
    }
}
