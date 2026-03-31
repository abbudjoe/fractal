use burn::backend::Candle;
use fractal_core::{rule_trait::FractalRule, state::FractalState};

use crate::{
    revived_primitive_factories, B1FractalGated, B3FractalHierarchical, B4Universal,
    P1FractalHybrid, P2Mandelbrot, PrimitiveRunSummary, RecursiveTokenizer, TokenizerConfig,
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
fn proving_ground_runs_all_revived_primitives() {
    let device = Default::default();
    let config = TokenizerConfig::default();
    let tokenizer = RecursiveTokenizer::new(config);
    let sentence = "Fractal tokenizers sketch recursive chunks.";
    let mut summaries = Vec::new();

    for factory in revived_primitive_factories::<TestBackend>() {
        let summary = tokenizer.run_factory(sentence, &device, factory).unwrap();
        println!("{}", format_summary(&summary));
        assert!(!summary.tokens.is_empty());
        summaries.push(summary);
    }

    assert_eq!(summaries.len(), 5);
    assert!(summaries.iter().all(|summary| summary.produced >= 3));
}

fn format_summary(summary: &PrimitiveRunSummary) -> String {
    let preview = summary
        .tokens
        .iter()
        .take(4)
        .map(|token| {
            format!(
                "{}:{}-{}:{}",
                token.depth, token.start, token.end, token.token
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "primitive={} produced={} preview={}",
        summary.primitive, summary.produced, preview
    )
}
