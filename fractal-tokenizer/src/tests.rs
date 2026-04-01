use burn::backend::Candle;
use fractal_core::{
    rule_trait::{ApplyContext, FractalRule},
    state::FractalState,
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tiktoken::cl100k_base;
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, PaddingDirection,
    PaddingParams, PaddingStrategy, Tokenizer,
};
use tokio::{process::Command as TokioCommand, runtime::Builder as TokioRuntimeBuilder};

use crate::{
    revived_primitive_factories, tokenizer::p1_dynamic_lever_factory,
    validate_tokenizer_primitive_name, B1FractalGated, B3FractalHierarchical, B4Universal,
    EncodedDocument, EncodedTokenKind, FaceoffChunkLimits, FaceoffEmissionPolicy,
    FaceoffLexemeKind, FaceoffTokenizer, FaceoffVocab, FaceoffVocabConfig,
    HuggingFaceNativeTokenizer, ModelFacingBatch, ModelFacingDocument, NativeCollationSpec,
    NativeCompatibilityAdapter, NativeTokenizer, P1FractalHybrid, P2Mandelbrot,
    PrimitiveRunSummary, RecursiveTokenizer, TokenRecord, TokenizerConfig,
    FACEOFF_VOCAB_FORMAT_VERSION,
};

type TestBackend = Candle<f32, i64>;

#[test]
fn revived_primitives_preserve_declared_layout_and_shape() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let primitives: Vec<Box<dyn FractalRule<TestBackend>>> = vec![
        Box::new(B1FractalGated::new(config.dim, &device)),
        Box::new(P1FractalHybrid::new(config.dim, &device)),
        Box::new(P2Mandelbrot::new(config.dim, &device)),
        Box::new(B3FractalHierarchical::new(
            config.dim,
            config.levels,
            &device,
        )),
        Box::new(B4Universal::new(config.dim, config.levels, &device)),
    ];

    for primitive in primitives {
        let state =
            FractalState::zeros(primitive.state_layout(), 1, primitive.hidden_dim(), &device)
                .unwrap();
        let tokenizer = RecursiveTokenizer::new(config);
        let tokens = tokenizer
            .tokenize(primitive.as_ref(), "tiny proof", &device)
            .unwrap();
        let next = primitive
            .apply(
                &state,
                &burn::tensor::Tensor::<TestBackend, 2>::zeros([1, config.dim], &device),
                ApplyContext {
                    depth: 0,
                    max_depth: config.max_depth,
                },
            )
            .unwrap();

        assert_eq!(primitive.state_layout(), next.layout());
        assert!(!tokens.is_empty());
    }
}

#[test]
fn proving_ground_runs_all_revived_primitives_with_fixed_seed() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let sentence = "The cat sat on the mat.";
    let first = collect_summaries(&tokenizer, sentence, &device);
    let second = collect_summaries(&tokenizer, sentence, &device);

    assert_eq!(
        digest_sequences(&first),
        digest_sequences(&second),
        "fixed seed should make proving-ground output reproducible"
    );

    for summary in &first {
        println!("{}", format_summary(summary));
        assert!(!summary.tokens.is_empty());
    }

    assert_eq!(first.len(), 5);
    assert!(first.iter().all(|summary| summary.produced >= 3));
}

#[test]
fn revived_tokenizer_factory_names_follow_convention() {
    for factory in revived_primitive_factories::<TestBackend>() {
        validate_tokenizer_primitive_name(factory.name).unwrap();
    }
}

#[test]
fn real_text_top_primitive_comparison() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let sentence =
        "The quick brown fox jumps over the lazy dog and the cat sat on the mat while watching the birds.";
    let summaries = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .filter(|factory| matches!(factory.name, "p1_fractal_hybrid_v1" | "b1_fractal_gated_v1"))
        .map(|factory| tokenizer.run_factory(sentence, &device, factory).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(summaries.len(), 2);

    for summary in &summaries {
        let unique_by_depth = unique_tokens_by_depth(summary);
        println!("{}", format_summary(summary));
        println!(
            "unique_by_depth={} total_by_depth={}",
            format_depth_counts(&unique_by_depth),
            format_depth_counts(&token_counts_by_depth(summary))
        );
        println!("pattern={}", describe_pattern(summary, &unique_by_depth));
        assert!(!summary.tokens.is_empty());
    }
}

#[test]
fn longer_real_text_p1_hybrid_follow_up() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let paragraph = "The quick brown fox jumps over the lazy dog and the cat sat on the mat while watching the birds fly high above the old oak tree in the quiet meadow on a sunny afternoon.";
    let factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let summary = tokenizer.run_factory(paragraph, &device, factory).unwrap();
    let unique_by_depth = unique_tokens_by_depth(&summary);

    println!("{}", format_summary(&summary));
    println!("unique_by_depth={}", format_depth_counts(&unique_by_depth));
    println!("pattern={}", describe_pattern(&summary, &unique_by_depth));
    println!(
        "balanced_note={}",
        balanced_pattern_note(&summary, &unique_by_depth)
    );

    assert!(!summary.tokens.is_empty());
    assert!(balanced_recursive_split_holds(&summary));
}

#[test]
fn motif_reuse_p1_hybrid_follow_up() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let sentence =
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat.";
    let factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let summary = tokenizer.run_factory(sentence, &device, factory).unwrap();
    let unique_by_depth = unique_tokens_by_depth(&summary);

    println!("{}", format_summary(&summary));
    println!("unique_by_depth={}", format_depth_counts(&unique_by_depth));
    println!("pattern={}", describe_pattern(&summary, &unique_by_depth));
    println!("motif_reuse={}", describe_motif_reuse(&summary));

    assert!(!summary.tokens.is_empty());
}

#[test]
fn motif_amplification_p1_hybrid_follow_up() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let amplified_sentence =
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat. The cat sat on the mat again.";
    let static_factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();
    let static_summary = tokenizer
        .run_factory(amplified_sentence, &device, static_factory)
        .unwrap();
    let dynamic_summary = tokenizer
        .run_factory(amplified_sentence, &device, dynamic_factory)
        .unwrap();
    let static_unique_by_depth = unique_tokens_by_depth(&static_summary);
    let dynamic_unique_by_depth = unique_tokens_by_depth(&dynamic_summary);

    println!("{}", format_summary(&static_summary));
    println!(
        "static_unique_tokens_by_depth={}",
        format_depth_counts(&static_unique_by_depth)
    );
    println!(
        "static_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&static_summary)
    );
    println!(
        "static_amplification_note={}",
        describe_pattern(&static_summary, &static_unique_by_depth)
    );
    println!("{}", format_summary(&dynamic_summary));
    println!("dynamic_lever_type=v2-self-regulating");
    println!(
        "dynamic_unique_tokens_by_depth={}",
        format_depth_counts(&dynamic_unique_by_depth)
    );
    println!(
        "dynamic_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&dynamic_summary)
    );
    println!(
        "amplification_note={}",
        dynamic_lever_note_v2(&static_summary, &dynamic_summary)
    );

    assert!(!static_summary.tokens.is_empty());
    assert!(!dynamic_summary.tokens.is_empty());
    assert!(balanced_recursive_split_holds(&static_summary));
    assert!(balanced_recursive_split_holds(&dynamic_summary));
    assert_eq!(
        dynamic_unique_by_depth,
        token_counts_by_depth(&dynamic_summary)
    );
}

#[test]
fn motif_amplification_p1_hybrid_v2_stress() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let repeated_paragraph =
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat.";
    let stress_input = std::iter::repeat_n(repeated_paragraph, 20)
        .collect::<Vec<_>>()
        .join(" ")
        + " The cat sat on the mat once more.";
    let static_factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();

    let static_summary = tokenizer
        .run_factory(&stress_input, &device, static_factory)
        .unwrap();
    let dynamic_summary = tokenizer
        .run_factory(&stress_input, &device, dynamic_factory)
        .unwrap();
    let static_unique_by_depth = unique_tokens_by_depth(&static_summary);
    let dynamic_unique_by_depth = unique_tokens_by_depth(&dynamic_summary);

    println!("{}", format_summary_preview(&static_summary, 20));
    println!(
        "static_unique_tokens_by_depth={}",
        format_depth_counts(&static_unique_by_depth)
    );
    println!(
        "static_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&static_summary)
    );
    println!(
        "static_hierarchy_note={}",
        hierarchy_balance_note(&static_summary, &static_unique_by_depth)
    );
    println!(
        "{}",
        format_final_token_spans_preview(&static_summary, &stress_input, 10)
    );
    println!("REUSED MOTIFS (cross-depth) [{}]", static_summary.primitive);
    println!(
        "{}",
        format_reused_motif_spans(&static_summary, &stress_input)
    );
    println!("{}", format_summary_preview(&dynamic_summary, 20));
    println!("dynamic_lever_type=v2-self-regulating");
    println!(
        "dynamic_unique_tokens_by_depth={}",
        format_depth_counts(&dynamic_unique_by_depth)
    );
    println!(
        "dynamic_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&dynamic_summary)
    );
    println!(
        "dynamic_hierarchy_note={}",
        hierarchy_balance_note(&dynamic_summary, &dynamic_unique_by_depth)
    );
    println!(
        "{}",
        format_final_token_spans_preview(&dynamic_summary, &stress_input, 10)
    );
    println!(
        "REUSED MOTIFS (cross-depth) [{}]",
        dynamic_summary.primitive
    );
    println!(
        "{}",
        format_reused_motif_spans(&dynamic_summary, &stress_input)
    );

    assert!(!static_summary.tokens.is_empty());
    assert!(!dynamic_summary.tokens.is_empty());
    assert!(balanced_recursive_split_holds(&static_summary));
    assert!(balanced_recursive_split_holds(&dynamic_summary));
}

#[test]
fn motif_amplification_p1_hybrid_v2_mixed_domain() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let news_paragraph = "City officials said Tuesday that transit service resumed across the river corridor after overnight storms flooded two low-lying stations, while crews continued inspecting power lines and drainage pumps before the evening commute.";
    let code_comment_paragraph = "This cache invalidation path keeps a rolling checksum for each segment so repeated blocks can be recognized without recomputing the full buffer; if a checksum disagrees, rebuild the branch and log the span that changed for debugging.";
    let literature_paragraph = "By the time the lamps were lit, the street had gone quiet enough for the distant train to sound like weather, and the old bookseller stood in his doorway listening as if the night itself were turning a page.";
    let mixed_input = [
        "=== NEWS ===",
        news_paragraph,
        "=== CODE COMMENT ===",
        code_comment_paragraph,
        "=== LITERATURE ===",
        literature_paragraph,
    ]
    .join("\n");
    let static_factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();

    let static_summary = tokenizer
        .run_factory(&mixed_input, &device, static_factory)
        .unwrap();
    let dynamic_summary = tokenizer
        .run_factory(&mixed_input, &device, dynamic_factory)
        .unwrap();
    let static_unique_by_depth = unique_tokens_by_depth(&static_summary);
    let dynamic_unique_by_depth = unique_tokens_by_depth(&dynamic_summary);

    println!("{}", format_summary_preview(&static_summary, 20));
    println!(
        "static_unique_tokens_by_depth={}",
        format_depth_counts(&static_unique_by_depth)
    );
    println!(
        "static_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&static_summary)
    );
    println!(
        "static_hierarchy_note={}",
        hierarchy_balance_note(&static_summary, &static_unique_by_depth)
    );
    println!("REUSED MOTIFS (cross-depth) [{}]", static_summary.primitive);
    println!(
        "{}",
        format_reused_motif_spans(&static_summary, &mixed_input)
    );
    println!("{}", format_summary_preview(&dynamic_summary, 20));
    println!("dynamic_lever_type=v2-self-regulating");
    println!(
        "dynamic_unique_tokens_by_depth={}",
        format_depth_counts(&dynamic_unique_by_depth)
    );
    println!(
        "dynamic_motif_reuse_count={}",
        cross_depth_motif_reuse_count(&dynamic_summary)
    );
    println!(
        "dynamic_hierarchy_note={}",
        hierarchy_balance_note(&dynamic_summary, &dynamic_unique_by_depth)
    );
    println!(
        "REUSED MOTIFS (cross-depth) [{}]",
        dynamic_summary.primitive
    );
    println!(
        "{}",
        format_reused_motif_spans(&dynamic_summary, &mixed_input)
    );

    assert!(!static_summary.tokens.is_empty());
    assert!(!dynamic_summary.tokens.is_empty());
    assert!(balanced_recursive_split_holds(&static_summary));
    assert!(balanced_recursive_split_holds(&dynamic_summary));
}

#[test]
fn tokenizer_sota_sanity_check() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let tiktoken = cl100k_base().unwrap();
    let stress_input = std::iter::repeat_n(
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat.",
        20,
    )
    .collect::<Vec<_>>()
    .join(" ")
        + " The cat sat on the mat once more.";
    let mixed_input = [
        "=== NEWS ===",
        "City officials said Tuesday that transit service resumed across the river corridor after overnight storms flooded two low-lying stations, while crews continued inspecting power lines and drainage pumps before the evening commute.",
        "=== CODE COMMENT ===",
        "This cache invalidation path keeps a rolling checksum for each segment so repeated blocks can be recognized without recomputing the full buffer; if a checksum disagrees, rebuild the branch and log the span that changed for debugging.",
        "=== LITERATURE ===",
        "By the time the lamps were lit, the street had gone quiet enough for the distant train to sound like weather, and the old bookseller stood in his doorway listening as if the night itself were turning a page.",
    ]
    .join("\n");
    let static_factory = revived_primitive_factories::<TestBackend>()
        .into_iter()
        .find(|factory| factory.name == "p1_fractal_hybrid_v1")
        .unwrap();
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();

    for (label, input) in [
        ("stress-20x-repetition", stress_input.as_str()),
        ("mixed-domain", mixed_input.as_str()),
    ] {
        let v1_summary = tokenizer
            .run_factory(input, &device, static_factory.clone())
            .unwrap();
        let v2_summary = tokenizer
            .run_factory(input, &device, dynamic_factory.clone())
            .unwrap();
        let tiktoken_count = tiktoken.encode_ordinary(input).len();

        println!("INPUT={label}");
        println!(
            "v2: token_count={} avg_chars_per_token={:.2} motif_reuse_count={}",
            v2_summary.tokens.len(),
            avg_chars_per_token(input, v2_summary.tokens.len()),
            cross_depth_motif_reuse_count(&v2_summary)
        );
        println!(
            "tiktoken_cl100k_base: token_count={} avg_chars_per_token={:.2}",
            tiktoken_count,
            avg_chars_per_token(input, tiktoken_count)
        );
        println!("REUSED MOTIFS (cross-depth) [{}]", v2_summary.primitive);
        println!("{}", format_reused_motif_spans(&v2_summary, input));
        assert_roundtrip("v1", &v1_summary, input);
        assert_roundtrip("v2", &v2_summary, input);

        assert!(!v1_summary.tokens.is_empty());
        assert!(!v2_summary.tokens.is_empty());
        assert!(balanced_recursive_split_holds(&v1_summary));
        assert!(balanced_recursive_split_holds(&v2_summary));
        assert!(tiktoken_count > 0);
    }
}

#[test]
fn tokenizer_integration_tests() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();
    let stress_input = std::iter::repeat_n(
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat.",
        20,
    )
    .collect::<Vec<_>>()
    .join(" ")
        + " The cat sat on the mat once more.";
    let mixed_input = [
        "=== NEWS ===",
        "City officials said Tuesday that transit service resumed across the river corridor after overnight storms flooded two low-lying stations, while crews continued inspecting power lines and drainage pumps before the evening commute.",
        "=== CODE COMMENT ===",
        "This cache invalidation path keeps a rolling checksum for each segment so repeated blocks can be recognized without recomputing the full buffer; if a checksum disagrees, rebuild the branch and log the span that changed for debugging.",
        "=== LITERATURE ===",
        "By the time the lamps were lit, the street had gone quiet enough for the distant train to sound like weather, and the old bookseller stood in his doorway listening as if the night itself were turning a page.",
    ]
    .join("\n");
    for (label, input, print_reused) in [
        ("stress-roundtrip", stress_input.as_str(), false),
        ("mixed-domain-roundtrip", mixed_input.as_str(), false),
    ] {
        let summary = tokenizer
            .run_factory(input, &device, dynamic_factory.clone())
            .unwrap();
        let unique_by_depth = unique_tokens_by_depth(&summary);
        let motif_reuse = cross_depth_motif_reuse_count(&summary);

        println!("SCENARIO={label}");
        println!("input_length={}", input.len());
        println!("final_token_count={}", summary.tokens.len());
        println!(
            "avg_chars_per_token={:.2}",
            avg_chars_per_token(input, summary.tokens.len())
        );
        println!("motif_reuse_count={motif_reuse}");
        println!(
            "hierarchy_note={}",
            hierarchy_balance_note(&summary, &unique_by_depth)
        );
        assert_roundtrip_integration(&summary, input);

        assert!(!summary.tokens.is_empty());
        assert!(balanced_recursive_split_holds(&summary));
        assert!(
            !print_reused,
            "streaming scenario should be handled separately"
        );
    }

    let stream_separator = "\n\n=== STREAM CHUNK ===\n\n";
    let streaming_chunks = [
        [stress_input.as_str(), stress_input.as_str()].join(stream_separator),
        [stress_input.as_str(), mixed_input.as_str()].join(stream_separator),
    ];
    let streaming_input = streaming_chunks.concat();
    let mut stream_offset = 0usize;
    let mut streaming_chunk_summaries = Vec::new();
    for chunk in &streaming_chunks {
        let summary = tokenizer
            .run_factory(chunk, &device, dynamic_factory.clone())
            .unwrap();
        streaming_chunk_summaries.push((stream_offset, summary));
        stream_offset += chunk.len();
    }
    let streaming_summary = merge_streaming_summaries(&streaming_chunk_summaries);
    let streaming_motif_reuse = cross_depth_motif_reuse_count(&streaming_summary);
    let streaming_balanced = streaming_chunk_summaries
        .iter()
        .all(|(_, summary)| balanced_recursive_split_holds(summary));

    println!("SCENARIO=streaming-corpus");
    println!("input_length={}", streaming_input.len());
    println!("final_token_count={}", streaming_summary.tokens.len());
    println!(
        "avg_chars_per_token={:.2}",
        avg_chars_per_token(&streaming_input, streaming_summary.tokens.len())
    );
    println!("motif_reuse_count={streaming_motif_reuse}");
    println!(
        "hierarchy_note={}",
        if streaming_balanced {
            "hierarchy remains perfectly balanced across streaming chunks"
        } else {
            "hierarchy drifted across streaming chunks"
        }
    );
    assert_roundtrip_integration(&streaming_summary, &streaming_input);
    println!(
        "REUSED MOTIFS (cross-depth) [{}]",
        streaming_summary.primitive
    );
    println!(
        "{}",
        format_reused_motif_spans(&streaming_summary, &streaming_input)
    );

    assert!(
        streaming_input.len() > 5_000,
        "streaming corpus should exceed 5k chars"
    );
    assert!(streaming_balanced);
    assert!(
        (10..=15).contains(&streaming_motif_reuse),
        "streaming motif_reuse_count should scale into the 10-15 target window, got {streaming_motif_reuse}"
    );
}

#[test]
fn faceoff_chunking_model_packaging() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let encoded = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();
        let packaged = encoded
            .package(crate::faceoff::FaceoffChunkLimits::new(8, 4096))
            .unwrap();
        let packaged_again = encoded
            .package(crate::faceoff::FaceoffChunkLimits::new(8, 4096))
            .unwrap();
        let reconstructed = packaged.reconstruct().unwrap();

        assert_eq!(faceoff.decode_document(&encoded).unwrap(), input);
        assert_eq!(
            packaged, packaged_again,
            "chunk packaging must be deterministic"
        );
        assert_eq!(
            reconstructed, input,
            "chunk payload concatenation must reconstruct input"
        );
        assert_eq!(encoded.fallback.unknown_motifs, 0);
        assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
        assert!(encoded.tokens.len() >= packaged.chunks.len());

        println!("FACEOFF_PACKAGING_INPUT={name}");
        println!("frontier_token_count={}", packaged.frontier_token_count);
        println!("packaged_chunk_count={}", packaged.chunks.len());
        println!(
            "chunk_shape_summary={}",
            packaged
                .chunks
                .iter()
                .map(|chunk| format!(
                    "c{}:{}tok/{}b",
                    chunk.index, chunk.token_count, chunk.byte_count
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "fallback_stats=motif_hits:{} unknown:{} recursed:{} byte:{}",
            encoded.fallback.motif_hits,
            encoded.fallback.unknown_motifs,
            encoded.fallback.recursed_to_children,
            encoded.fallback.byte_fallback_tokens
        );
        println!("roundtrip_status=OK");
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WhitespaceTokenizer;

impl NativeTokenizer for WhitespaceTokenizer {
    type Token = String;
    type Error = std::convert::Infallible;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
        Ok(text.split_whitespace().map(str::to_owned).collect())
    }
}

fn build_hf_native_tokenizer(inputs: &[&str]) -> HuggingFaceNativeTokenizer {
    let mut vocab = HashMap::new();
    vocab.insert("[PAD]".to_string(), 0);
    vocab.insert("[UNK]".to_string(), 1);

    let mut next_id = 2u32;
    let mut pieces = BTreeSet::<String>::new();
    for input in inputs {
        pieces.extend(input.split_whitespace().map(str::to_owned));
    }
    for piece in pieces {
        vocab.insert(piece, next_id);
        next_id += 1;
    }

    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));

    HuggingFaceNativeTokenizer::new(tokenizer)
}

fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "fractal-tokenizer-{prefix}-{}-{stamp}{suffix}",
        std::process::id()
    ))
}

fn save_hf_tokenizer_to_temp_file(tokenizer: &HuggingFaceNativeTokenizer, prefix: &str) -> PathBuf {
    let path = unique_temp_path(prefix, ".json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    tokenizer.tokenizer().save(&path, false).unwrap();
    assert!(
        path.exists(),
        "expected tokenizer.json to be written to disk"
    );
    path
}

#[derive(Clone, Copy, Debug)]
struct PretrainedTokenizerSource {
    label: &'static str,
    env_var: &'static str,
    owner_fragment: &'static str,
    repo_fragment: &'static str,
}

impl PretrainedTokenizerSource {
    fn env_path(self) -> Option<PathBuf> {
        let value = std::env::var(self.env_var).ok()?;
        let path = PathBuf::from(value);
        path.is_file().then_some(path)
    }

    fn cache_path(self, cache_root: &Path) -> Option<PathBuf> {
        find_tokenizer_json_under(cache_root, self.owner_fragment, self.repo_fragment)
    }
}

fn pretrained_hf_tokenizer_json_paths() -> Vec<(String, PathBuf)> {
    const SOURCES: &[PretrainedTokenizerSource] = &[
        PretrainedTokenizerSource {
            label: "llama31",
            env_var: "HF_LLAMA31_TOKENIZER_JSON",
            owner_fragment: "meta-llama",
            repo_fragment: "Llama-3.1-8B-Instruct",
        },
        PretrainedTokenizerSource {
            label: "mistral7",
            env_var: "HF_MISTRAL7_TOKENIZER_JSON",
            owner_fragment: "mistralai",
            repo_fragment: "Mistral-7B-Instruct-v0.3",
        },
        PretrainedTokenizerSource {
            label: "qwen25",
            env_var: "HF_QWEN25_TOKENIZER_JSON",
            owner_fragment: "Qwen",
            repo_fragment: "Qwen2.5-7B-Instruct",
        },
        PretrainedTokenizerSource {
            label: "phi3mini",
            env_var: "HF_PHI3MINI_TOKENIZER_JSON",
            owner_fragment: "microsoft",
            repo_fragment: "Phi-3-mini-4k-instruct",
        },
        PretrainedTokenizerSource {
            label: "mixtral8x7b",
            env_var: "HF_MIXTRAL8X7B_TOKENIZER_JSON",
            owner_fragment: "mistralai",
            repo_fragment: "Mixtral-8x7B-Instruct-v0.1",
        },
    ];

    let mut paths = Vec::new();

    for source in SOURCES {
        if let Some(path) = source.env_path() {
            paths.push((source.label.to_string(), path));
        }
    }

    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return dedupe_tokenizer_paths(paths);
    };
    let cache_root = home.join(".cache/huggingface/hub");
    if !cache_root.is_dir() {
        return dedupe_tokenizer_paths(paths);
    }

    for source in SOURCES {
        if let Some(path) = source.cache_path(&cache_root) {
            paths.push((source.label.to_string(), path));
        }
    }

    dedupe_tokenizer_paths(paths)
}

fn find_tokenizer_json_under(
    root: &Path,
    owner_fragment: &str,
    repo_fragment: &str,
) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().and_then(|name| name.to_str()) != Some("tokenizer.json") {
                continue;
            }
            let path_str = path.to_string_lossy();
            if path_str.contains(owner_fragment) && path_str.contains(repo_fragment) {
                return Some(path);
            }
        }
    }
    None
}

fn dedupe_tokenizer_paths(paths: Vec<(String, PathBuf)>) -> Vec<(String, PathBuf)> {
    let mut seen = BTreeSet::<PathBuf>::new();
    let mut deduped = Vec::new();
    for (label, path) in paths {
        if seen.insert(path.clone()) {
            deduped.push((label, path));
        }
    }
    deduped
}

fn write_json_value(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_string_pretty(value).unwrap()).unwrap();
}

#[test]
fn model_face_document_roundtrip_on_benchmark_inputs() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let encoded = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();
        let document = ModelFacingDocument::from_encoded(encoded, limits).unwrap();

        assert_eq!(document.decode().unwrap(), input);
        assert_eq!(document.reconstruct().unwrap(), input);
        assert!(document.validate().is_ok());
        assert_eq!(document.fallback().unknown_motifs, 0);
        assert_eq!(document.fallback().byte_fallback_tokens, 0);
        assert_eq!(
            document.frontier_token_count(),
            document.encoded().tokens.len()
        );
        assert_eq!(document.chunk_count(), document.chunked().chunks.len());

        println!("MODEL_FACE_INPUT={name}");
        println!("frontier_token_count={}", document.frontier_token_count());
        println!("packaged_chunk_count={}", document.chunk_count());
        println!(
            "fallback_stats=motif_hits:{} unknown:{} recursed:{} byte:{}",
            document.fallback().motif_hits,
            document.fallback().unknown_motifs,
            document.fallback().recursed_to_children,
            document.fallback().byte_fallback_tokens
        );
        println!("roundtrip_status=OK");
    }
}

#[test]
fn model_face_native_adapter_preserves_chunk_order_and_is_deterministic() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer = WhitespaceTokenizer;

    let documents = vec![
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &stress,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
    ]
    .into_iter()
    .map(|encoded| ModelFacingDocument::from_encoded(encoded, limits).unwrap())
    .collect::<Vec<_>>();
    let batch = ModelFacingBatch::from(documents);
    let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
    let native_again = adapter.retokenize_batch(&batch, &tokenizer).unwrap();

    assert_eq!(native, native_again);
    assert_eq!(native.len(), batch.len());

    for (index, (document, native_document)) in batch.iter().zip(native.iter()).enumerate() {
        assert_eq!(document.input_len(), native_document.input_len);
        assert_eq!(
            document.frontier_token_count(),
            native_document.frontier_token_count
        );
        assert_eq!(document.chunk_count(), native_document.chunk_count());
        assert_eq!(document.reconstruct().unwrap(), document.decode().unwrap());
        assert!(native_document
            .chunks
            .iter()
            .enumerate()
            .all(|(chunk_index, chunk)| chunk.source.index == chunk_index));
        assert!(native_document
            .chunks
            .windows(2)
            .all(|pair| pair[0].source.index < pair[1].source.index));
        assert_eq!(
            native_document.native_token_count(),
            native_document
                .chunks
                .iter()
                .map(|chunk| chunk.native_tokens.len())
                .sum::<usize>()
        );

        println!("MODEL_FACE_BATCH_DOCUMENT_INDEX={index}");
        println!("native_chunk_count={}", native_document.chunk_count());
        println!(
            "native_token_count={}",
            native_document.native_token_count()
        );
        println!("roundtrip_status=OK");
    }
}

#[test]
fn model_face_hf_native_adapter_preserves_chunk_order_and_is_deterministic() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer = build_hf_native_tokenizer(&corpus);

    let documents = vec![
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &stress,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
    ]
    .into_iter()
    .map(|encoded| ModelFacingDocument::from_encoded(encoded, limits).unwrap())
    .collect::<Vec<_>>();
    let batch = ModelFacingBatch::from(documents);

    for (name, expected, document) in [
        (
            "stress-20x-repetition",
            stress.as_str(),
            &batch.documents()[0],
        ),
        ("mixed-domain", mixed.as_str(), &batch.documents()[1]),
    ] {
        assert_eq!(document.decode().unwrap(), expected);
        assert_eq!(document.reconstruct().unwrap(), expected);
        assert!(document.validate().is_ok());
        println!("MODEL_FACE_HF_INPUT={name}");
        println!("frontier_token_count={}", document.frontier_token_count());
        println!("packaged_chunk_count={}", document.chunk_count());
        println!("roundtrip_status=OK");
    }

    let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
    let native_again = adapter.retokenize_batch(&batch, &tokenizer).unwrap();

    assert_eq!(native, native_again);
    assert_eq!(native.len(), batch.len());

    for (index, (document, native_document)) in batch.iter().zip(native.iter()).enumerate() {
        assert_eq!(document.input_len(), native_document.input_len);
        assert_eq!(
            document.frontier_token_count(),
            native_document.frontier_token_count
        );
        assert_eq!(document.chunk_count(), native_document.chunk_count());
        assert!(native_document
            .chunks
            .iter()
            .enumerate()
            .all(|(chunk_index, chunk)| chunk.source.index == chunk_index));
        assert!(native_document
            .chunks
            .windows(2)
            .all(|pair| pair[0].source.index < pair[1].source.index));
        assert_eq!(
            native_document.native_token_count(),
            native_document
                .chunks
                .iter()
                .map(|chunk| chunk.native_tokens.len())
                .sum::<usize>()
        );

        println!("MODEL_FACE_HF_BATCH_DOCUMENT_INDEX={index}");
        println!("native_chunk_count={}", native_document.chunk_count());
        println!(
            "native_token_count={}",
            native_document.native_token_count()
        );
        println!("roundtrip_status=OK");
    }
}

#[test]
fn model_face_native_batch_collation_preserves_order_padding_and_attention_mask() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer = build_hf_native_tokenizer(&corpus);
    let pad_id = tokenizer
        .tokenizer()
        .get_padding()
        .map(|padding| padding.pad_id)
        .unwrap_or(0);
    let spec = NativeCollationSpec::try_new(pad_id, Some(8)).unwrap();

    let documents = vec![
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &stress,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
    ]
    .into_iter()
    .map(|encoded| ModelFacingDocument::from_encoded(encoded, limits).unwrap())
    .collect::<Vec<_>>();
    let batch = ModelFacingBatch::from(documents);

    let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
    let native_again = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
    let collated = native.collate(&spec).unwrap();
    let collated_again = native_again.collate(&spec).unwrap();

    assert_eq!(native, native_again);
    assert_eq!(collated, collated_again);
    assert_eq!(collated.len(), batch.len());
    assert_eq!(collated.spec, spec);
    assert_eq!(
        collated.chunk_count(),
        native.iter().map(|doc| doc.chunk_count()).sum::<usize>()
    );
    assert_eq!(collated.sequence_len % 8, 0);

    for (document_index, (native_document, collated_document)) in
        native.iter().zip(collated.iter()).enumerate()
    {
        assert_eq!(collated_document.source_document_index, document_index);
        assert_eq!(collated_document.input_len, native_document.input_len);
        assert_eq!(
            collated_document.frontier_token_count,
            native_document.frontier_token_count
        );
        assert_eq!(
            collated_document.chunk_count(),
            native_document.chunk_count()
        );
        assert_eq!(
            collated_document.native_token_count(),
            native_document.native_token_count()
        );

        for (chunk_index, (native_chunk, collated_chunk)) in native_document
            .chunks
            .iter()
            .zip(collated_document.chunks.iter())
            .enumerate()
        {
            let valid_len = native_chunk.native_tokens.len();

            assert_eq!(collated_chunk.source_document_index, document_index);
            assert_eq!(collated_chunk.source_chunk_index, chunk_index);
            assert_eq!(collated_chunk.source, *native_chunk);
            assert_eq!(collated_chunk.valid_token_count(), valid_len);
            assert_eq!(collated_chunk.padded_token_count(), collated.sequence_len);
            assert_eq!(
                &collated_chunk.padded_tokens[..valid_len],
                native_chunk.native_tokens.as_slice()
            );
            assert!(collated_chunk.padded_tokens[valid_len..]
                .iter()
                .all(|token| *token == pad_id));
            assert!(collated_chunk.attention_mask[..valid_len]
                .iter()
                .all(|mask| *mask));
            assert!(collated_chunk.attention_mask[valid_len..]
                .iter()
                .all(|mask| !*mask));
        }
    }

    assert_eq!(
        collated.documents[0].chunk_count(),
        collated.documents[2].chunk_count()
    );
    for (left, right) in collated.documents[0]
        .chunks
        .iter()
        .zip(collated.documents[2].chunks.iter())
    {
        assert_eq!(left.padded_tokens, right.padded_tokens);
        assert_eq!(left.attention_mask, right.attention_mask);
    }

    println!("MODEL_FACE_COLLATION_ORDER=mixed,stress,mixed");
    println!("MODEL_FACE_COLLATION_PAD_ID={pad_id}");
    println!("MODEL_FACE_COLLATION_DOCUMENTS={}", collated.len());
    println!("MODEL_FACE_COLLATION_CHUNKS={}", collated.chunk_count());
    println!(
        "MODEL_FACE_COLLATION_SEQUENCE_LEN={}",
        collated.sequence_len
    );
}

#[test]
fn model_face_file_backed_hf_native_adapter_preserves_chunk_order_and_is_deterministic() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let in_memory_tokenizer = build_hf_native_tokenizer(&corpus);
    let tokenizer_path = save_hf_tokenizer_to_temp_file(&in_memory_tokenizer, "hf-file-backed");
    let file_backed_tokenizer = HuggingFaceNativeTokenizer::from_file(&tokenizer_path).unwrap();

    let documents = vec![
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &stress,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
    ]
    .into_iter()
    .map(|encoded| ModelFacingDocument::from_encoded(encoded, limits).unwrap())
    .collect::<Vec<_>>();
    let batch = ModelFacingBatch::from(documents);

    let file_backed_native = adapter
        .retokenize_batch(&batch, &file_backed_tokenizer)
        .unwrap();
    let file_backed_again = adapter
        .retokenize_batch(&batch, &file_backed_tokenizer)
        .unwrap();
    let in_memory_native = adapter
        .retokenize_batch(&batch, &in_memory_tokenizer)
        .unwrap();

    assert!(file_backed_tokenizer.tokenizer().get_padding().is_some());
    assert_eq!(file_backed_native, file_backed_again);
    assert_eq!(file_backed_native, in_memory_native);
    assert_eq!(file_backed_native.len(), batch.len());

    for (index, (document, native_document)) in
        batch.iter().zip(file_backed_native.iter()).enumerate()
    {
        assert_eq!(document.input_len(), native_document.input_len);
        assert_eq!(
            document.frontier_token_count(),
            native_document.frontier_token_count
        );
        assert_eq!(document.chunk_count(), native_document.chunk_count());
        assert_eq!(document.reconstruct().unwrap(), document.decode().unwrap());
        assert!(native_document
            .chunks
            .iter()
            .enumerate()
            .all(|(chunk_index, chunk)| chunk.source.index == chunk_index));
        assert!(native_document
            .chunks
            .windows(2)
            .all(|pair| pair[0].source.index < pair[1].source.index));

        println!("MODEL_FACE_HF_FILE_INPUT_INDEX={index}");
        println!("tokenizer_json_path={}", tokenizer_path.display());
        println!("native_chunk_count={}", native_document.chunk_count());
        println!(
            "native_token_count={}",
            native_document.native_token_count()
        );
        println!("roundtrip_status=OK");
    }
}

#[test]
fn model_face_pretrained_hf_native_adapter_preserves_chunk_order_and_is_deterministic() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer_paths = pretrained_hf_tokenizer_json_paths();

    if tokenizer_paths.is_empty() {
        println!(
            "MODEL_FACE_PRETRAINED_HF_SKIP=no local pretrained tokenizer.json configured/found"
        );
        return;
    }

    let documents = vec![
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &stress,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
        faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                &mixed,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap(),
    ]
    .into_iter()
    .map(|encoded| ModelFacingDocument::from_encoded(encoded, limits).unwrap())
    .collect::<Vec<_>>();
    let batch = ModelFacingBatch::from(documents);

    for (label, tokenizer_path) in tokenizer_paths {
        let tokenizer =
            HuggingFaceNativeTokenizer::from_file(&tokenizer_path).unwrap_or_else(|error| {
                panic!(
                    "failed to load pretrained tokenizer.json for {label} at {}: {error}",
                    tokenizer_path.display()
                )
            });
        let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
        let native_again = adapter.retokenize_batch(&batch, &tokenizer).unwrap();

        assert_eq!(native, native_again);
        assert_eq!(native.len(), batch.len());

        for (index, (document, native_document)) in batch.iter().zip(native.iter()).enumerate() {
            assert_eq!(document.input_len(), native_document.input_len);
            assert_eq!(
                document.frontier_token_count(),
                native_document.frontier_token_count
            );
            assert_eq!(document.chunk_count(), native_document.chunk_count());
            assert_eq!(document.reconstruct().unwrap(), document.decode().unwrap());
            assert!(native_document
                .chunks
                .iter()
                .enumerate()
                .all(|(chunk_index, chunk)| chunk.source.index == chunk_index));
            assert!(native_document
                .chunks
                .windows(2)
                .all(|pair| pair[0].source.index < pair[1].source.index));

            println!("MODEL_FACE_PRETRAINED_HF_LABEL={label}");
            println!("MODEL_FACE_PRETRAINED_HF_INPUT_INDEX={index}");
            println!("tokenizer_json_path={}", tokenizer_path.display());
            println!("native_chunk_count={}", native_document.chunk_count());
            println!(
                "native_token_count={}",
                native_document.native_token_count()
            );
            println!("roundtrip_status=OK");
        }
    }
}

#[test]
fn faceoff_vocab_induction_is_deterministic() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];

    let first = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let second = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    assert_eq!(first.entries(), second.entries());
    assert!(first.motif_count() > 0);
}

#[test]
fn faceoff_vocab_config_filters_one_off_large_motifs() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let shared = "error: retry later\n".repeat(16);
    let unique_large = format!("unique start {}\n{}", "A".repeat(2048), stress_input());
    let corpus = vec![shared.as_str(), shared.as_str(), unique_large.as_str()];

    let default_vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let recurring_only = faceoff
        .induce_vocab_from_texts_with_config::<TestBackend>(
            &corpus,
            &device,
            FaceoffVocabConfig {
                min_occurrence_count: 2,
                min_doc_count: 2,
                max_token_bytes: Some(512),
                ..FaceoffVocabConfig::default()
            },
        )
        .unwrap();

    assert!(recurring_only.motif_count() > 0);
    assert!(recurring_only.motif_count() < default_vocab.motif_count());
    assert!(recurring_only
        .entries()
        .iter()
        .all(|entry| entry.occurrence_count >= 2));
    assert!(recurring_only
        .entries()
        .iter()
        .all(|entry| entry.doc_count >= 2));
    assert!(recurring_only
        .entries()
        .iter()
        .all(|entry| entry.max_byte_len <= 512));
}

#[test]
fn faceoff_vocab_recover_descendant_cover_on_held_out_input() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let shared = "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat.";
    let induction_a = format!("ALPHA HEADER\n{shared}\nALPHA FOOTER");
    let induction_b = format!("BETA HEADER\n{shared}\nBETA FOOTER");
    let held_out = format!("GAMMA HEADER\n{shared}\nGAMMA FOOTER");
    let corpus = vec![induction_a.as_str(), induction_b.as_str()];

    let vocab = faceoff
        .induce_vocab_from_texts_with_config::<TestBackend>(
            &corpus,
            &device,
            FaceoffVocabConfig {
                min_occurrence_count: 2,
                min_doc_count: 2,
                max_token_bytes: Some(512),
                ..FaceoffVocabConfig::default()
            },
        )
        .unwrap();

    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&held_out, &vocab, &device)
        .unwrap();

    assert_eq!(faceoff.decode_document(&encoded).unwrap(), held_out);
    assert!(encoded.fallback.unknown_motifs > 0);
    assert!(encoded.fallback.recursed_to_children > 0);
    assert!(encoded.fallback.motif_hits > 0);
    assert!(
        encoded.fallback.byte_fallback_tokens < held_out.len(),
        "held-out recovery should not devolve to all-byte fallback"
    );
}

#[test]
fn faceoff_vocab_recovers_held_out_shape_aliases() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let induction_a = "fn render_home() {\n    let auth_provider = 2026;\n}\n";
    let induction_b = "fn render_settings() {\n    let oauth_flow = 2027;\n}\n";
    let held_out = "fn render_usage() {\n    let experiments = 2028;\n}\n";
    let corpus = vec![induction_a, induction_b];

    let vocab = faceoff
        .induce_vocab_from_texts_with_config::<TestBackend>(
            &corpus,
            &device,
            FaceoffVocabConfig {
                min_occurrence_count: 2,
                min_doc_count: 2,
                max_token_bytes: Some(128),
                ..FaceoffVocabConfig::default()
            },
        )
        .unwrap();

    assert!(!vocab.shape_entries().is_empty());

    let encoded = faceoff
        .encode_text_v2::<TestBackend>(held_out, &vocab, &device)
        .unwrap();

    assert_eq!(faceoff.decode_document(&encoded).unwrap(), held_out);
    assert!(
        encoded.fallback.shape_hits > 0,
        "held-out shape-equivalent text should recover at least one shape-based structural hit"
    );
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
}

#[test]
fn faceoff_vocab_persistence_round_trip_is_exact() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let vocab_path = unique_temp_path("faceoff-vocab", ".json");
    if let Some(parent) = vocab_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    vocab.save_to_file(&vocab_path).unwrap();
    let loaded = FaceoffVocab::load_from_file(&vocab_path).unwrap();

    assert_eq!(vocab, loaded);
    assert_eq!(vocab.entries(), loaded.entries());
    assert_eq!(vocab.motif_count(), loaded.motif_count());
    assert_eq!(vocab.byte_fallback_base(), loaded.byte_fallback_base());
    assert_eq!(
        vocab.decode_byte_id(vocab.byte_id(0)),
        loaded.decode_byte_id(loaded.byte_id(0))
    );
    assert_eq!(FaceoffVocab::FORMAT_VERSION, FACEOFF_VOCAB_FORMAT_VERSION);
}

#[test]
fn faceoff_vocab_persistence_rejects_invalid_version_and_duplicate_motifs() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let vocab_path = unique_temp_path("faceoff-vocab-invalid", ".json");
    if let Some(parent) = vocab_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }

    vocab.save_to_file(&vocab_path).unwrap();
    let persisted: Value = serde_json::from_str(&fs::read_to_string(&vocab_path).unwrap()).unwrap();

    let mut invalid_version = persisted.clone();
    invalid_version["version"] = Value::from(FaceoffVocab::FORMAT_VERSION + 1);
    let invalid_version_path = unique_temp_path("faceoff-vocab-invalid-version", ".json");
    write_json_value(&invalid_version_path, &invalid_version);
    let err = FaceoffVocab::load_from_file(&invalid_version_path).unwrap_err();
    assert!(
        format!("{err}").contains("unsupported faceoff vocab version"),
        "expected invalid version to be rejected"
    );

    let mut duplicate_motifs = persisted;
    let motifs = duplicate_motifs["motifs"].as_array_mut().unwrap();
    let first_motif = motifs[0].clone();
    motifs.push(first_motif);
    duplicate_motifs["byte_fallback_base"] = Value::from(motifs.len() as u64);
    let duplicate_path = unique_temp_path("faceoff-vocab-duplicate", ".json");
    write_json_value(&duplicate_path, &duplicate_motifs);
    let err = FaceoffVocab::load_from_file(&duplicate_path).unwrap_err();
    assert!(
        format!("{err}").contains("sorted and unique"),
        "expected duplicate motifs to be rejected"
    );
}

#[test]
fn faceoff_roundtrip_stress_input_is_exact() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&stress, &vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();

    assert_eq!(decoded, stress);
    assert!(
        encoded.tokens.len() > 1,
        "full-vocab faceoff encoding should emit a useful frontier, not a single root token"
    );
    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
}

#[test]
fn faceoff_roundtrip_mixed_domain_input_is_exact() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&mixed, &vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();

    assert_eq!(decoded, mixed);
    assert!(
        encoded.tokens.len() > 1,
        "full-vocab faceoff encoding should emit a useful frontier, not a single root token"
    );
    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
}

#[test]
fn faceoff_fallback_activates_with_partial_vocab() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let empty_records = Vec::<TokenRecord>::new();
    let partial_vocab = FaceoffVocab::from_token_records(empty_records.iter()).unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&stress, &partial_vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();

    assert_eq!(decoded, stress);
    assert!(encoded.fallback.unknown_motifs > 0);
    assert!(encoded.fallback.recursed_to_children > 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
    assert!(encoded
        .tokens
        .iter()
        .any(|token| matches!(token.kind, EncodedTokenKind::Lexical { .. })));
}

#[test]
fn faceoff_lexical_fallback_classifies_typed_atoms_deterministically() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let input = "AuthProvider 2026-03-31 ::git-push{ x }\n    next_line";
    let partial_vocab = FaceoffVocab::from_token_records([].iter()).unwrap();

    let first = faceoff
        .encode_text_v2::<TestBackend>(input, &partial_vocab, &device)
        .unwrap();
    let second = faceoff
        .encode_text_v2::<TestBackend>(input, &partial_vocab, &device)
        .unwrap();

    assert_eq!(faceoff.decode_document(&first).unwrap(), input);
    assert_eq!(first, second);
    assert!(first.tokens.iter().any(|token| matches!(
        token.kind,
        EncodedTokenKind::Lexical {
            kind: FaceoffLexemeKind::Identifier
        }
    )));
    assert!(first.tokens.iter().any(|token| matches!(
        token.kind,
        EncodedTokenKind::Lexical {
            kind: FaceoffLexemeKind::Number
        }
    )));
    assert!(first.tokens.iter().any(|token| matches!(
        token.kind,
        EncodedTokenKind::Lexical {
            kind: FaceoffLexemeKind::Whitespace
        }
    )));
    assert!(first.tokens.iter().any(|token| matches!(
        token.kind,
        EncodedTokenKind::Lexical {
            kind: FaceoffLexemeKind::NewlineIndent
        }
    )));
}

#[test]
fn faceoff_mixed_domain_false_positive_reuse_stays_near_zero() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&mixed, &vocab, &device)
        .unwrap();
    let reuse_count = encoded_cross_depth_motif_reuse_count(&encoded);

    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
    assert_eq!(
        reuse_count, 0,
        "expected near-zero cross-depth motif reuse on mixed-domain input"
    );
}

#[test]
fn faceoff_unicode_heavy_roundtrip_is_exact() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let unicode = unicode_heavy_input();
    let json_code_log = json_code_log_input();
    let near_repetition = near_repetition_input();
    let corpus = vec![
        unicode.as_str(),
        json_code_log.as_str(),
        near_repetition.as_str(),
    ];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&unicode, &vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();

    assert_eq!(decoded, unicode);
    assert!(encoded.tokens.len() > 1);
    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
}

#[test]
fn faceoff_json_code_log_blend_avoids_false_reuse() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let unicode = unicode_heavy_input();
    let json_code_log = json_code_log_input();
    let near_repetition = near_repetition_input();
    let corpus = vec![
        unicode.as_str(),
        json_code_log.as_str(),
        near_repetition.as_str(),
    ];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&json_code_log, &vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();
    let reuse_count = encoded_cross_depth_motif_reuse_count(&encoded);

    assert_eq!(decoded, json_code_log);
    assert!(encoded.tokens.len() > 1);
    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
    assert!(
        reuse_count <= 1,
        "json/code/log blend should not create excessive false-positive reuse"
    );
}

#[test]
fn faceoff_near_repetition_does_not_overcollapse() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let unicode = unicode_heavy_input();
    let json_code_log = json_code_log_input();
    let near_repetition = near_repetition_input();
    let corpus = vec![
        unicode.as_str(),
        json_code_log.as_str(),
        near_repetition.as_str(),
    ];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let encoded = faceoff
        .encode_text_v2::<TestBackend>(&near_repetition, &vocab, &device)
        .unwrap();
    let decoded = faceoff.decode_document(&encoded).unwrap();
    let reuse_count = encoded_cross_depth_motif_reuse_count(&encoded);

    assert_eq!(decoded, near_repetition);
    assert!(
        encoded.tokens.len() > 1,
        "near repetition should preserve multiple frontier tokens"
    );
    assert_eq!(encoded.fallback.unknown_motifs, 0);
    assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
    assert!(
        reuse_count <= 2,
        "near repetition should not overcollapse into excessive cross-depth reuse"
    );
}

#[test]
fn model_face_pretrained_adapter_handles_edge_case_documents_deterministically() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let unicode = unicode_heavy_input();
    let json_code_log = json_code_log_input();
    let near_repetition = near_repetition_input();
    let hard_edge_input = [
        "=== JSON CODE LOG MIX ===",
        json_code_log.as_str(),
        "=== NEAR REPETITION MIX ===",
        near_repetition.as_str(),
    ]
    .join("\n");
    let corpus = vec![
        unicode.as_str(),
        json_code_log.as_str(),
        near_repetition.as_str(),
    ];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(8, 4096);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer_paths = pretrained_hf_tokenizer_json_paths();

    if tokenizer_paths.is_empty() {
        println!("MODEL_FACE_EDGE_SKIP=no local pretrained tokenizer.json configured/found");
        return;
    }

    let document = faceoff
        .encode_text_v2_with_policy::<TestBackend>(
            &hard_edge_input,
            &vocab,
            &device,
            FaceoffEmissionPolicy::NoveltyAware,
        )
        .unwrap();
    let document = ModelFacingDocument::from_encoded(document, limits).unwrap();
    let batch = ModelFacingBatch::from(vec![document]);

    for (label, tokenizer_path) in tokenizer_paths {
        let tokenizer =
            HuggingFaceNativeTokenizer::from_file(&tokenizer_path).unwrap_or_else(|error| {
                panic!(
                    "failed to load pretrained tokenizer.json for {label} at {}: {error}",
                    tokenizer_path.display()
                )
            });
        let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();
        let native_again = adapter.retokenize_batch(&batch, &tokenizer).unwrap();

        assert_eq!(native, native_again);
        assert_eq!(native.len(), batch.len());

        for (index, (document, native_document)) in batch.iter().zip(native.iter()).enumerate() {
            assert_eq!(document.input_len(), native_document.input_len);
            assert_eq!(
                document.frontier_token_count(),
                native_document.frontier_token_count
            );
            assert_eq!(document.chunk_count(), native_document.chunk_count());
            assert_eq!(document.reconstruct().unwrap(), document.decode().unwrap());
            assert!(native_document
                .chunks
                .iter()
                .enumerate()
                .all(|(chunk_index, chunk)| chunk.source.index == chunk_index));
            assert!(native_document
                .chunks
                .windows(2)
                .all(|pair| pair[0].source.index < pair[1].source.index));

            println!("MODEL_FACE_EDGE_LABEL={label}");
            println!("MODEL_FACE_EDGE_INPUT_INDEX={index}");
            println!("tokenizer_json_path={}", tokenizer_path.display());
            println!("native_chunk_count={}", native_document.chunk_count());
            println!(
                "native_token_count={}",
                native_document.native_token_count()
            );
            println!("roundtrip_status=OK");
        }
    }
}

#[test]
fn model_face_pretrained_adapter_unicode_heavy_documents_are_utf8_safe() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let unicode = unicode_heavy_input();
    let json_code_log = json_code_log_input();
    let near_repetition = near_repetition_input();
    let corpus = vec![json_code_log.as_str(), near_repetition.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();
    let limits = FaceoffChunkLimits::new(1, 8);
    let adapter = NativeCompatibilityAdapter;
    let tokenizer_paths = pretrained_hf_tokenizer_json_paths();

    if tokenizer_paths.is_empty() {
        println!("MODEL_FACE_UNICODE_SKIP=no local pretrained tokenizer.json configured/found");
        return;
    }

    let document = faceoff
        .encode_text_v2_with_policy::<TestBackend>(
            &unicode,
            &vocab,
            &device,
            FaceoffEmissionPolicy::NoveltyAware,
        )
        .unwrap();
    let document = ModelFacingDocument::from_encoded(document, limits).unwrap();
    let batch = ModelFacingBatch::from(vec![document]);

    for (label, tokenizer_path) in tokenizer_paths {
        let tokenizer =
            HuggingFaceNativeTokenizer::from_file(&tokenizer_path).unwrap_or_else(|error| {
                panic!(
                    "failed to load pretrained tokenizer.json for {label} at {}: {error}",
                    tokenizer_path.display()
                )
            });
        let native = adapter.retokenize_batch(&batch, &tokenizer).unwrap();

        assert_eq!(native.len(), 1);
        let document = batch.iter().next().unwrap();
        let native_document = native.documents.first().unwrap();
        assert_eq!(native_document.input_len, unicode.len());
        assert_eq!(
            native_document.frontier_token_count,
            document.frontier_token_count()
        );
        assert!(native_document.native_token_count() > 0);
        assert!(native_document
            .chunks
            .iter()
            .all(|chunk| std::str::from_utf8(&chunk.source.payload).is_ok()));
    }
}

#[test]
fn faceoff_tokenizer_slice_report() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let encoded = faceoff
            .encode_text_v2::<TestBackend>(input, &vocab, &device)
            .unwrap();
        let decoded = faceoff.decode_document(&encoded).unwrap();
        assert_eq!(decoded, input);
        assert!(
            encoded.tokens.len() > 1,
            "full-vocab faceoff encoding should emit a useful frontier, not a single root token"
        );
        assert_eq!(encoded.fallback.unknown_motifs, 0);
        assert_eq!(encoded.fallback.byte_fallback_tokens, 0);
        println!("FACEOFF_INPUT={name}");
        println!("token_count={}", encoded.tokens.len());
        println!(
            "avg_chars_per_token={:.2}",
            avg_chars_per_token(input, encoded.tokens.len())
        );
        println!(
            "frontier_status={}",
            if encoded.tokens.len() > 1 {
                "useful-frontier"
            } else {
                "single-root-token"
            }
        );
        println!(
            "fallback_stats=motif_hits:{} unknown_motifs:{} recursed_to_children:{} byte_fallback_tokens:{}",
            encoded.fallback.motif_hits,
            encoded.fallback.unknown_motifs,
            encoded.fallback.recursed_to_children,
            encoded.fallback.byte_fallback_tokens
        );
        println!("ROUNDTRIP: OK");
    }
}

#[test]
fn faceoff_state_aware_vs_finest_known_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let finest = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::FinestKnown,
            )
            .unwrap();
        let state_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::StateAware,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&finest).unwrap(), input);
        assert_eq!(faceoff.decode_document(&state_aware).unwrap(), input);
        assert_eq!(finest.fallback.unknown_motifs, 0);
        assert_eq!(state_aware.fallback.unknown_motifs, 0);
        assert_eq!(finest.fallback.byte_fallback_tokens, 0);
        assert_eq!(state_aware.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "finest_known token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            finest.tokens.len(),
            avg_chars_per_token(input, finest.tokens.len()),
            finest.fallback.motif_hits,
            finest.fallback.unknown_motifs,
            finest.fallback.recursed_to_children,
            finest.fallback.byte_fallback_tokens
        );
        println!(
            "state_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            state_aware.tokens.len(),
            avg_chars_per_token(input, state_aware.tokens.len()),
            state_aware.fallback.motif_hits,
            state_aware.fallback.unknown_motifs,
            state_aware.fallback.recursed_to_children,
            state_aware.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            state_aware.tokens.len() as isize - finest.tokens.len() as isize
        );
    }
}

#[test]
fn faceoff_reuse_aware_vs_finest_known_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let finest = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::FinestKnown,
            )
            .unwrap();
        let reuse_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::ReuseAware,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&finest).unwrap(), input);
        assert_eq!(faceoff.decode_document(&reuse_aware).unwrap(), input);
        assert_eq!(finest.fallback.unknown_motifs, 0);
        assert_eq!(reuse_aware.fallback.unknown_motifs, 0);
        assert_eq!(finest.fallback.byte_fallback_tokens, 0);
        assert_eq!(reuse_aware.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "finest_known token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            finest.tokens.len(),
            avg_chars_per_token(input, finest.tokens.len()),
            finest.fallback.motif_hits,
            finest.fallback.unknown_motifs,
            finest.fallback.recursed_to_children,
            finest.fallback.byte_fallback_tokens
        );
        println!(
            "reuse_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            reuse_aware.tokens.len(),
            avg_chars_per_token(input, reuse_aware.tokens.len()),
            reuse_aware.fallback.motif_hits,
            reuse_aware.fallback.unknown_motifs,
            reuse_aware.fallback.recursed_to_children,
            reuse_aware.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            reuse_aware.tokens.len() as isize - finest.tokens.len() as isize
        );
    }
}

#[test]
fn faceoff_novelty_aware_vs_finest_known_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let finest = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::FinestKnown,
            )
            .unwrap();
        let novelty_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&finest).unwrap(), input);
        assert_eq!(faceoff.decode_document(&novelty_aware).unwrap(), input);
        assert_eq!(finest.fallback.unknown_motifs, 0);
        assert_eq!(novelty_aware.fallback.unknown_motifs, 0);
        assert_eq!(finest.fallback.byte_fallback_tokens, 0);
        assert_eq!(novelty_aware.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "finest_known token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            finest.tokens.len(),
            avg_chars_per_token(input, finest.tokens.len()),
            finest.fallback.motif_hits,
            finest.fallback.unknown_motifs,
            finest.fallback.recursed_to_children,
            finest.fallback.byte_fallback_tokens
        );
        println!(
            "novelty_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            novelty_aware.tokens.len(),
            avg_chars_per_token(input, novelty_aware.tokens.len()),
            novelty_aware.fallback.motif_hits,
            novelty_aware.fallback.unknown_motifs,
            novelty_aware.fallback.recursed_to_children,
            novelty_aware.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            novelty_aware.tokens.len() as isize - finest.tokens.len() as isize
        );
    }
}

#[test]
fn faceoff_span_length_aware_vs_novelty_aware_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let novelty_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();
        let span_length_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::SpanLengthAware,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&novelty_aware).unwrap(), input);
        assert_eq!(faceoff.decode_document(&span_length_aware).unwrap(), input);
        assert_eq!(novelty_aware.fallback.unknown_motifs, 0);
        assert_eq!(span_length_aware.fallback.unknown_motifs, 0);
        assert_eq!(novelty_aware.fallback.byte_fallback_tokens, 0);
        assert_eq!(span_length_aware.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "novelty_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            novelty_aware.tokens.len(),
            avg_chars_per_token(input, novelty_aware.tokens.len()),
            novelty_aware.fallback.motif_hits,
            novelty_aware.fallback.unknown_motifs,
            novelty_aware.fallback.recursed_to_children,
            novelty_aware.fallback.byte_fallback_tokens
        );
        println!(
            "span_length_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            span_length_aware.tokens.len(),
            avg_chars_per_token(input, span_length_aware.tokens.len()),
            span_length_aware.fallback.motif_hits,
            span_length_aware.fallback.unknown_motifs,
            span_length_aware.fallback.recursed_to_children,
            span_length_aware.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            span_length_aware.tokens.len() as isize - novelty_aware.tokens.len() as isize
        );
    }
}

#[test]
fn faceoff_budgeted_vs_novelty_aware_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let novelty_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();
        let budgeted = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::Budgeted,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&novelty_aware).unwrap(), input);
        assert_eq!(faceoff.decode_document(&budgeted).unwrap(), input);
        assert_eq!(novelty_aware.fallback.unknown_motifs, 0);
        assert_eq!(budgeted.fallback.unknown_motifs, 0);
        assert_eq!(novelty_aware.fallback.byte_fallback_tokens, 0);
        assert_eq!(budgeted.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "novelty_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            novelty_aware.tokens.len(),
            avg_chars_per_token(input, novelty_aware.tokens.len()),
            novelty_aware.fallback.motif_hits,
            novelty_aware.fallback.unknown_motifs,
            novelty_aware.fallback.recursed_to_children,
            novelty_aware.fallback.byte_fallback_tokens
        );
        println!(
            "budgeted token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            budgeted.tokens.len(),
            avg_chars_per_token(input, budgeted.tokens.len()),
            budgeted.fallback.motif_hits,
            budgeted.fallback.unknown_motifs,
            budgeted.fallback.recursed_to_children,
            budgeted.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            budgeted.tokens.len() as isize - novelty_aware.tokens.len() as isize
        );
    }
}

#[test]
fn faceoff_hybrid_structural_vs_novelty_aware_side_by_side() {
    let device = Default::default();
    let faceoff = FaceoffTokenizer::new(TokenizerConfig::default());
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let corpus = vec![stress.as_str(), mixed.as_str()];
    let vocab = faceoff
        .induce_vocab_from_texts::<TestBackend>(&corpus, &device)
        .unwrap();

    for (name, input) in [
        ("stress-20x-repetition", stress.as_str()),
        ("mixed-domain", mixed.as_str()),
    ] {
        let novelty_aware = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::NoveltyAware,
            )
            .unwrap();
        let hybrid_structural = faceoff
            .encode_text_v2_with_policy::<TestBackend>(
                input,
                &vocab,
                &device,
                FaceoffEmissionPolicy::HybridStructural,
            )
            .unwrap();

        assert_eq!(faceoff.decode_document(&novelty_aware).unwrap(), input);
        assert_eq!(faceoff.decode_document(&hybrid_structural).unwrap(), input);
        assert_eq!(novelty_aware.fallback.unknown_motifs, 0);
        assert_eq!(hybrid_structural.fallback.unknown_motifs, 0);
        assert_eq!(novelty_aware.fallback.byte_fallback_tokens, 0);
        assert_eq!(hybrid_structural.fallback.byte_fallback_tokens, 0);

        println!("FACEOFF_POLICY_COMPARISON_INPUT={name}");
        println!(
            "novelty_aware token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            novelty_aware.tokens.len(),
            avg_chars_per_token(input, novelty_aware.tokens.len()),
            novelty_aware.fallback.motif_hits,
            novelty_aware.fallback.unknown_motifs,
            novelty_aware.fallback.recursed_to_children,
            novelty_aware.fallback.byte_fallback_tokens
        );
        println!(
            "hybrid_structural token_count={} avg_chars_per_token={:.2} fallback=motif_hits:{} unknown:{} recursed:{} byte:{}",
            hybrid_structural.tokens.len(),
            avg_chars_per_token(input, hybrid_structural.tokens.len()),
            hybrid_structural.fallback.motif_hits,
            hybrid_structural.fallback.unknown_motifs,
            hybrid_structural.fallback.recursed_to_children,
            hybrid_structural.fallback.byte_fallback_tokens
        );
        println!(
            "roundtrip_status=OK delta_tokens={}",
            hybrid_structural.tokens.len() as isize - novelty_aware.tokens.len() as isize
        );
    }
}

#[test]
fn tokenizer_oss_benchmark() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let dynamic_factory = p1_dynamic_lever_factory::<TestBackend>();
    let corpus = oss_benchmark_corpus();

    assert!(
        corpus.len() >= 10_000,
        "OSS benchmark corpus should be at least 10k chars, got {}",
        corpus.len()
    );

    for (model_name, env_var) in [
        ("Llama 3.1 8B", "LLAMA31_8B_GGUF_PATH"),
        ("Mistral 7B", "MISTRAL_7B_GGUF_PATH"),
    ] {
        let v2_started = Instant::now();
        let v2_summary = tokenizer
            .run_factory(&corpus, &device, dynamic_factory.clone())
            .unwrap();
        let v2_elapsed_ms = v2_started.elapsed().as_millis();
        let v2_unique_by_depth = unique_tokens_by_depth(&v2_summary);

        println!("model_name={model_name}");
        println!("tokenizer_type=v2");
        println!("token_count={}", v2_summary.tokens.len());
        println!(
            "avg_chars_per_token={:.2}",
            avg_chars_per_token(&corpus, v2_summary.tokens.len())
        );
        println!(
            "motif_reuse_count={}",
            cross_depth_motif_reuse_count(&v2_summary)
        );
        println!("rough_perplexity=N/A");
        println!("wall_time_ms={v2_elapsed_ms}");
        println!(
            "hierarchy_note={}",
            hierarchy_balance_note(&v2_summary, &v2_unique_by_depth)
        );
        println!("REUSED MOTIFS (cross-depth) [{}]", v2_summary.primitive);
        println!("{}", format_reused_motif_spans(&v2_summary, &corpus));

        assert!(!v2_summary.tokens.is_empty());
        assert!(balanced_recursive_split_holds(&v2_summary));

        let native_started = Instant::now();
        match maybe_native_llama_token_count(env_var, &corpus) {
            Ok(native_count) => {
                let native_elapsed_ms = native_started.elapsed().as_millis();
                println!("model_name={model_name}");
                println!("tokenizer_type=native");
                println!("token_count={native_count}");
                println!(
                    "avg_chars_per_token={:.2}",
                    avg_chars_per_token(&corpus, native_count)
                );
                println!("motif_reuse_count=N/A");
                println!("rough_perplexity=N/A");
                println!("wall_time_ms={native_elapsed_ms}");
            }
            Err(status) => {
                println!("model_name={model_name}");
                println!("tokenizer_type=native");
                println!("token_count=N/A");
                println!("avg_chars_per_token=N/A");
                println!("motif_reuse_count=N/A");
                println!("rough_perplexity=N/A");
                println!("wall_time_ms=N/A");
                println!("status={status}");
            }
        }
    }
}

fn collect_summaries(
    tokenizer: &RecursiveTokenizer,
    sentence: &str,
    device: &<TestBackend as burn::tensor::backend::Backend>::Device,
) -> Vec<PrimitiveRunSummary> {
    revived_primitive_factories::<TestBackend>()
        .into_iter()
        .map(|factory| tokenizer.run_factory(sentence, device, factory).unwrap())
        .collect()
}

fn digest_sequences(summaries: &[PrimitiveRunSummary]) -> Vec<(&'static str, Vec<String>)> {
    summaries
        .iter()
        .map(|summary| {
            (
                summary.primitive,
                summary
                    .tokens
                    .iter()
                    .map(|token| token.token.clone())
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

fn format_summary(summary: &PrimitiveRunSummary) -> String {
    let digests = summary
        .tokens
        .iter()
        .map(|token| token.token.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{:<24} | {}", summary.primitive, digests)
}

fn format_summary_preview(summary: &PrimitiveRunSummary, limit: usize) -> String {
    let digests = summary
        .tokens
        .iter()
        .take(limit)
        .map(|token| token.token.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{:<24} | {}", summary.primitive, digests)
}

fn format_final_token_spans_preview(
    summary: &PrimitiveRunSummary,
    input: &str,
    limit: usize,
) -> String {
    let final_depth = summary
        .tokens
        .iter()
        .map(|token| token.depth)
        .max()
        .unwrap_or_default();
    let lines = summary
        .tokens
        .iter()
        .filter(|token| token.depth == final_depth)
        .take(limit)
        .map(|token| {
            format!(
                "{} {}..{} \"{}\"",
                token.token,
                token.start,
                token.end,
                truncated_token_span(input, token.start, token.end)
            )
        })
        .collect::<Vec<_>>();

    format!(
        "{}_final_d{final_depth}_spans={}",
        summary.primitive,
        lines.join(" | ")
    )
}

fn truncated_token_span(input: &str, start: usize, end: usize) -> String {
    let span = input.get(start..end).unwrap_or_default();
    if span.chars().count() <= 80 {
        span.to_string()
    } else {
        let truncated = span.chars().take(77).collect::<String>();
        format!("{truncated}...")
    }
}

fn format_reused_motif_spans(summary: &PrimitiveRunSummary, input: &str) -> String {
    let repeated = repeated_cross_depth_motifs(summary);
    if repeated.is_empty() {
        return "(none)".to_string();
    }

    repeated
        .into_iter()
        .map(|motif| {
            format!(
                "digest={} | total_reuse_count={} | ranges={} | text=\"{}\"",
                motif.digest,
                motif.total_reuse_count,
                motif
                    .ranges
                    .iter()
                    .map(|(start, end)| format!("{start}..{end}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                truncated_token_span(input, motif.example_start, motif.example_end)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unique_tokens_by_depth(summary: &PrimitiveRunSummary) -> BTreeMap<usize, usize> {
    let mut tokens_by_depth = BTreeMap::<usize, BTreeSet<&str>>::new();
    for token in &summary.tokens {
        tokens_by_depth
            .entry(token.depth)
            .or_default()
            .insert(token.token.as_str());
    }

    tokens_by_depth
        .into_iter()
        .map(|(depth, tokens)| (depth, tokens.len()))
        .collect()
}

fn token_counts_by_depth(summary: &PrimitiveRunSummary) -> BTreeMap<usize, usize> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for token in &summary.tokens {
        *counts.entry(token.depth).or_default() += 1;
    }
    counts
}

fn format_depth_counts(counts: &BTreeMap<usize, usize>) -> String {
    counts
        .iter()
        .map(|(depth, count)| format!("d{depth}:{count}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn avg_chars_per_token(text: &str, token_count: usize) -> f64 {
    text.chars().count() as f64 / token_count.max(1) as f64
}

fn assert_roundtrip(label: &str, summary: &PrimitiveRunSummary, input: &str) {
    let reconstructed = reconstruct_from_final_token_spans(summary, input);
    assert_eq!(
        reconstructed, input,
        "ROUNDTRIP CHECK [{label}] failed for {}",
        summary.primitive
    );
    println!("ROUNDTRIP CHECK [{label}]: OK");
}

fn reconstruct_from_final_token_spans(summary: &PrimitiveRunSummary, input: &str) -> String {
    let final_depth = summary
        .tokens
        .iter()
        .map(|token| token.depth)
        .max()
        .unwrap_or_default();
    let mut final_tokens = summary
        .tokens
        .iter()
        .filter(|token| token.depth == final_depth)
        .collect::<Vec<_>>();
    final_tokens.sort_by_key(|token| token.start);

    final_tokens
        .into_iter()
        .map(|token| {
            input
                .get(token.start..token.end)
                .unwrap_or_else(|| {
                    panic!(
                        "invalid final token span {}..{} for {}",
                        token.start, token.end, summary.primitive
                    )
                })
                .to_string()
        })
        .collect()
}

fn assert_roundtrip_integration(summary: &PrimitiveRunSummary, input: &str) {
    let reconstructed = reconstruct_from_final_token_spans(summary, input);
    assert_eq!(
        reconstructed, input,
        "ROUNDTRIP failed for {}",
        summary.primitive
    );
    println!("ROUNDTRIP: OK");
}

fn merge_streaming_summaries(
    streaming_chunk_summaries: &[(usize, PrimitiveRunSummary)],
) -> PrimitiveRunSummary {
    let primitive = streaming_chunk_summaries
        .first()
        .map(|(_, summary)| summary.primitive)
        .unwrap_or("p1_fractal_hybrid_dyn-state-norm_v2");
    let produced = streaming_chunk_summaries
        .iter()
        .map(|(_, summary)| summary.produced)
        .sum();
    let tokens = streaming_chunk_summaries
        .iter()
        .flat_map(|(offset, summary)| {
            summary.tokens.iter().map(move |token| TokenRecord {
                depth: token.depth,
                start: token.start + offset,
                end: token.end + offset,
                text: token.text.clone(),
                token: token.token.clone(),
            })
        })
        .collect();

    PrimitiveRunSummary {
        primitive,
        produced,
        tokens,
    }
}

fn stress_input() -> String {
    std::iter::repeat_n(
        "The cat sat on the mat. The dog sat on the mat. The bird sat on the mat. The fox sat on the mat.",
        20,
    )
    .collect::<Vec<_>>()
    .join(" ")
        + " The cat sat on the mat once more."
}

fn mixed_domain_input() -> String {
    [
        "=== NEWS ===",
        "City officials said Tuesday that transit service resumed across the river corridor after overnight storms flooded two low-lying stations, while crews continued inspecting power lines and drainage pumps before the evening commute.",
        "=== CODE COMMENT ===",
        "This cache invalidation path keeps a rolling checksum for each segment so repeated blocks can be recognized without recomputing the full buffer; if a checksum disagrees, rebuild the branch and log the span that changed for debugging.",
        "=== LITERATURE ===",
        "By the time the lamps were lit, the street had gone quiet enough for the distant train to sound like weather, and the old bookseller stood in his doorway listening as if the night itself were turning a page.",
    ]
    .join("\n")
}

fn unicode_heavy_input() -> String {
    [
        "日本語の段落です。🙂 “Quoted text” keeps its punctuation, and café remains café.",
        "العربية هنا أيضًا، مع أرقام ١٢٣ وبعض الرموز — plus English mix.",
        "Emoji burst: 🧪🚀✨ and family sequence 👨‍👩‍👧‍👦 with zero-width joiners intact.",
    ]
    .join("\n")
}

fn json_code_log_input() -> String {
    [
        r#"{"ts":"2026-03-31T12:00:00Z","level":"info","event":"cache_hit","request_id":"abc123"}"#,
        r#"fn refresh_cache(key: &str, value: &str) -> Result<(), CacheError> { if key.is_empty() { return Err(CacheError::EmptyKey); } Ok(()) }"#,
        "log=warn request_id=abc123 action=refresh_cache status=retrying checksum=1a2b3c4d",
        r#"{"ts":"2026-03-31T12:00:01Z","level":"info","event":"cache_hit","request_id":"abc123","count":3}"#,
    ]
    .join("\n")
}

fn near_repetition_input() -> String {
    [
        "record id=1001 user=alice status=ok region=us-west shard=alpha timestamp=2026-03-31T12:00:00Z",
        "record id=1002 user=alice status=ok region=us-west shard=alpha timestamp=2026-03-31T12:00:01Z",
        "record id=1003 user=bob status=ok region=us-west shard=alpha timestamp=2026-03-31T12:00:02Z",
        "record id=1004 user=bob status=ok region=us-west shard=beta timestamp=2026-03-31T12:00:03Z",
        "record id=1005 user=carol status=ok region=us-west shard=beta timestamp=2026-03-31T12:00:04Z",
    ]
    .join("\n")
}

fn oss_benchmark_corpus() -> String {
    let stress = stress_input();
    let mixed = mixed_domain_input();
    let code_block = r#"fn stitch_token_spans(tokens: &[TokenRecord], input: &str) -> String {
    tokens
        .iter()
        .filter(|token| token.depth == 5)
        .map(|token| &input[token.start..token.end])
        .collect::<String>()
}

fn grouped_reuse(digests: &[String]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for digest in digests {
        *counts.entry(digest.clone()).or_insert(0) += 1;
    }
    counts
}"#;

    [
        stress.as_str(),
        stress.as_str(),
        stress.as_str(),
        stress.as_str(),
        mixed.as_str(),
        code_block,
        mixed.as_str(),
        code_block,
        code_block,
    ]
    .join("\n\n=== OSS BENCHMARK CHUNK ===\n\n")
}

fn maybe_native_llama_token_count(env_var: &str, input: &str) -> Result<usize, String> {
    let Some(model_path) = native_model_path(env_var) else {
        return Err(format!("MODEL NOT FOUND — SKIPPED (set {env_var})"));
    };

    native_llama_token_count(&model_path, input)
        .map_err(|err| format!("MODEL NOT FOUND — SKIPPED ({err})"))
}

fn native_model_path(env_var: &str) -> Option<PathBuf> {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .filter(|path| path.is_file())
}

fn native_llama_token_count(model_path: &Path, input: &str) -> Result<usize, String> {
    let Some(tokenizer_bin) = native_tokenizer_bin() else {
        return Err(
            "native tokenizer binary not configured (set LLAMA_CPP_TOKENIZE_BIN or install llama-tokenize)"
                .to_string(),
        );
    };
    let temp_path = std::env::temp_dir().join(format!(
        "fractal-tokenizer-oss-benchmark-{}.txt",
        std::process::id()
    ));
    fs::write(&temp_path, input).map_err(|err| format!("failed to write temp corpus: {err}"))?;

    let runtime = TokioRuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| format!("failed to build tokio runtime: {err}"))?;
    let output = runtime
        .block_on(async {
            TokioCommand::new(&tokenizer_bin)
                .arg("-m")
                .arg(model_path)
                .arg("-f")
                .arg(&temp_path)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        })
        .map_err(|err| format!("failed to run {}: {err}", tokenizer_bin.display()));
    let _ = fs::remove_file(&temp_path);
    let output = output?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "{} exited with status {}: {}",
            tokenizer_bin.display(),
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "signal".to_string()),
            stderr.trim()
        ));
    }

    parse_native_token_count(
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    )
}

fn native_tokenizer_bin() -> Option<PathBuf> {
    std::env::var_os("LLAMA_CPP_TOKENIZE_BIN")
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| command_on_path("llama-tokenize"))
}

fn command_on_path(command: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|path| {
        std::env::split_paths(&path)
            .map(|entry| entry.join(command))
            .find(|candidate| candidate.is_file())
    })
}

fn parse_native_token_count(stdout: &str, stderr: &str) -> Result<usize, String> {
    let combined = format!("{stdout}\n{stderr}");
    for line in combined.lines() {
        if !line.to_ascii_lowercase().contains("token") {
            continue;
        }
        if let Some(number) = line
            .split(|ch: char| !ch.is_ascii_digit())
            .find(|part| !part.is_empty())
        {
            return number
                .parse::<usize>()
                .map_err(|err| format!("failed to parse token count from `{line}`: {err}"));
        }
    }

    Err("native tokenizer output did not include a parseable token count".to_string())
}

fn encoded_cross_depth_motif_reuse_count(document: &EncodedDocument) -> usize {
    let mut motif_hits = BTreeMap::<String, BTreeSet<usize>>::new();
    for token in &document.tokens {
        if let EncodedTokenKind::Motif { digest } = &token.kind {
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

fn describe_pattern(
    summary: &PrimitiveRunSummary,
    unique_by_depth: &BTreeMap<usize, usize>,
) -> String {
    let totals = token_counts_by_depth(summary);
    let repeated_depths = totals
        .iter()
        .filter_map(|(depth, total)| {
            let unique = unique_by_depth.get(depth).copied().unwrap_or_default();
            (unique < *total).then_some(format!("d{depth} ({unique}/{total} unique)"))
        })
        .collect::<Vec<_>>();
    let repetition_note = if repeated_depths.is_empty() {
        "no repeated digests by depth".to_string()
    } else {
        format!("digest reuse appears at {}", repeated_depths.join(", "))
    };
    let deepest = totals.keys().next_back().copied().unwrap_or_default();

    format!("balanced recursive split through depth {deepest}; {repetition_note}")
}

fn balanced_pattern_note(
    summary: &PrimitiveRunSummary,
    unique_by_depth: &BTreeMap<usize, usize>,
) -> &'static str {
    if balanced_recursive_split_holds(summary) && unique_by_depth == &token_counts_by_depth(summary)
    {
        "balanced recursive split pattern holds on longer text"
    } else {
        "balanced recursive split pattern drifted on longer text"
    }
}

fn hierarchy_balance_note(
    summary: &PrimitiveRunSummary,
    unique_by_depth: &BTreeMap<usize, usize>,
) -> &'static str {
    if balanced_recursive_split_holds(summary) && unique_by_depth == &token_counts_by_depth(summary)
    {
        "hierarchy remains perfectly balanced"
    } else if balanced_recursive_split_holds(summary) {
        "hierarchy remains balanced but shows within-depth reuse"
    } else {
        "hierarchy drifted from the balanced split"
    }
}

fn balanced_recursive_split_holds(summary: &PrimitiveRunSummary) -> bool {
    let totals = token_counts_by_depth(summary);
    let mut expected = 1;
    for total in totals.values() {
        if *total != expected {
            return false;
        }
        expected *= 2;
    }
    true
}

fn describe_motif_reuse(summary: &PrimitiveRunSummary) -> String {
    let mut motif_hits = BTreeMap::<String, BTreeSet<usize>>::new();

    for token in &summary.tokens {
        let motif = token_digest(token).to_string();
        motif_hits.entry(motif).or_default().insert(token.depth);
    }

    let repeated = motif_hits
        .into_iter()
        .filter(|(_, depths)| depths.len() > 1)
        .map(|(motif, depths)| {
            format!(
                "{} across {}",
                motif,
                depths
                    .into_iter()
                    .map(|depth| format!("d{depth}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>();

    if repeated.is_empty() {
        "no repeated motif digests across any depth".into()
    } else {
        format!("repeated motif digests: {}", repeated.join("; "))
    }
}

fn dynamic_lever_note_v2(
    static_summary: &PrimitiveRunSummary,
    dynamic_summary: &PrimitiveRunSummary,
) -> String {
    let static_hits = cross_depth_motif_reuse_count(static_summary);
    let dynamic_hits = cross_depth_motif_reuse_count(dynamic_summary);

    if (2..=4).contains(&dynamic_hits) {
        format!(
            "v2 self-regulating lever hit the target window ({dynamic_hits} repeated motifs, static={static_hits})"
        )
    } else if dynamic_hits > static_hits {
        format!(
            "v2 self-regulating lever increased motif reuse but landed outside the target window ({dynamic_hits} repeated motifs, static={static_hits})"
        )
    } else {
        format!(
            "v2 self-regulating lever matched the static motif reuse count ({dynamic_hits} repeated motifs)"
        )
    }
}

fn cross_depth_motif_reuse_count(summary: &PrimitiveRunSummary) -> usize {
    repeated_cross_depth_motifs(summary).len()
}

fn repeated_cross_depth_motifs(summary: &PrimitiveRunSummary) -> Vec<RepeatedMotif> {
    let mut motif_hits = BTreeMap::<String, RepeatedMotifAccumulator>::new();
    for token in &summary.tokens {
        let entry = motif_hits
            .entry(token_digest(token).to_string())
            .or_insert_with(|| RepeatedMotifAccumulator {
                depths: BTreeSet::new(),
                ranges: Vec::new(),
                example_start: token.start,
                example_end: token.end,
            });
        entry.depths.insert(token.depth);
        entry.ranges.push((token.start, token.end));
        if entry.ranges.len() == 1 {
            entry.example_start = token.start;
            entry.example_end = token.end;
        }
    }

    motif_hits
        .into_iter()
        .filter_map(|(digest, entry)| {
            (entry.depths.len() > 1 && entry.ranges.len() >= 2).then_some(RepeatedMotif {
                digest,
                total_reuse_count: entry.ranges.len(),
                ranges: entry.ranges,
                example_start: entry.example_start,
                example_end: entry.example_end,
            })
        })
        .collect()
}

fn token_digest(token: &TokenRecord) -> &str {
    token
        .token
        .rsplit_once('-')
        .map(|(_, digest)| digest)
        .unwrap_or(token.token.as_str())
}

struct RepeatedMotifAccumulator {
    depths: BTreeSet<usize>,
    ranges: Vec<(usize, usize)>,
    example_start: usize,
    example_end: usize,
}

struct RepeatedMotif {
    digest: String,
    total_reuse_count: usize,
    ranges: Vec<(usize, usize)>,
    example_start: usize,
    example_end: usize,
}
