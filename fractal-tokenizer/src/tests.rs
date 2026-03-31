use burn::backend::Candle;
use fractal_core::{rule_trait::FractalRule, state::FractalState};

use crate::{
    revived_primitive_factories, tokenizer_tracker_reminder, validate_tokenizer_primitive_name,
    B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot,
    PrimitiveRunSummary, RecursiveTokenizer, TokenizerConfig,
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
