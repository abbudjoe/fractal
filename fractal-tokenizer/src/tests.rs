use burn::backend::Candle;
use fractal_core::{rule_trait::FractalRule, state::FractalState};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    revived_primitive_factories, tokenizer::p1_dynamic_lever_factory, tokenizer_tracker_reminder,
    validate_tokenizer_primitive_name, B1FractalGated, B3FractalHierarchical, B4Universal,
    P1FractalHybrid, P2Mandelbrot, PrimitiveRunSummary, RecursiveTokenizer, TokenRecord,
    TokenizerConfig,
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
    println!("{}", tokenizer_tracker_reminder());

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
    println!("{}", tokenizer_tracker_reminder());
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
    println!("{}", tokenizer_tracker_reminder());

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
    println!("{}", tokenizer_tracker_reminder());

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
    println!("{}", tokenizer_tracker_reminder());

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
    println!("{}", tokenizer_tracker_reminder());

    assert!(!static_summary.tokens.is_empty());
    assert!(!dynamic_summary.tokens.is_empty());
    assert!(balanced_recursive_split_holds(&static_summary));
    assert!(balanced_recursive_split_holds(&dynamic_summary));
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
