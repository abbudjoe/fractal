use std::sync::atomic::{AtomicUsize, Ordering};

use burn::{
    backend::Candle,
    module::Module,
    tensor::{backend::Backend, Tensor, TensorData},
};

use crate::{
    data_generator::{GeneratorConfig, SimpleHierarchicalGenerator, MIN_VOCAB_SIZE, PAD_TOKEN},
    lifecycle::{Tournament, TournamentConfig},
    model::FractalModel,
    primitives::{
        b1_fractal_gated::B1FractalGated, b2_stable_hierarchical::B2StableHierarchical,
        b3_fractal_hierarchical::B3FractalHierarchical, b4_universal::B4Universal, complex_square,
        p1_contractive::P1Contractive, p2_mandelbrot::P2Mandelbrot,
        p3_hierarchical::P3Hierarchical,
    },
    router::EarlyExitRouter,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

type TestBackend = Candle<f32, i64>;
static APPLY_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
fn complex_square_matches_hand_computed_values() {
    let device = Default::default();
    let tensor = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(vec![2.0f32, 3.0, 4.0, 5.0], [1, 4]),
        &device,
    );
    let squared = complex_square(tensor).into_data().to_vec::<f32>().unwrap();

    assert_eq!(squared, vec![-12.0, -16.0, 16.0, 30.0]);
}

#[test]
fn primitives_preserve_declared_layout_and_shape() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::zeros([2, 8], &device);

    let primitives: Vec<Box<dyn FractalRule<TestBackend>>> = vec![
        Box::new(P1Contractive::new(8, &device)),
        Box::new(P2Mandelbrot::new(8, &device)),
        Box::new(P3Hierarchical::new(8, 3, &device)),
        Box::new(B1FractalGated::new(8, &device)),
        Box::new(B2StableHierarchical::new(8, 3, &device)),
        Box::new(B3FractalHierarchical::new(8, 3, &device)),
        Box::new(B4Universal::new(8, 3, &device)),
    ];

    for primitive in primitives {
        let state =
            FractalState::zeros(primitive.state_layout(), 2, primitive.hidden_dim(), &device)
                .unwrap();
        let next = primitive.apply(&state, &x).unwrap();
        assert_eq!(primitive.state_layout(), next.layout());
    }
}

#[test]
fn clone_box_preserves_metadata() {
    let device = Default::default();
    let primitive: Box<dyn FractalRule<TestBackend>> = Box::new(B4Universal::new(8, 3, &device));
    let clone = primitive.clone_box();

    assert_eq!(primitive.name(), clone.name());
    assert_eq!(primitive.hidden_dim(), clone.hidden_dim());
    assert_eq!(primitive.state_layout(), clone.state_layout());
}

#[test]
fn hierarchical_rules_update_all_levels_without_shape_collapse() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::zeros([2, 8], &device);
    let rule = B2StableHierarchical::new(8, 4, &device);
    let state =
        FractalState::zeros(StateLayout::Hierarchical { levels: 4 }, 2, 8, &device).unwrap();
    let next = rule.apply(&state, &x).unwrap();

    match next {
        FractalState::Hierarchical(tensor) => assert_eq!(tensor.dims(), [2, 4, 8]),
        _ => panic!("expected hierarchical state"),
    }
}

#[test]
fn router_never_mutates_readout_shape() {
    let device = Default::default();
    let router = EarlyExitRouter::<TestBackend>::new(8, 0.9, &device);
    let readout = Tensor::<TestBackend, 2>::zeros([3, 8], &device);
    let scores = router.scores(readout.clone());

    assert_eq!(readout.dims(), [3, 8]);
    assert_eq!(scores.dims(), [3]);
}

#[test]
fn tournament_returns_exactly_seven_results() {
    let tournament = Tournament::new(TournamentConfig::fast_test()).unwrap();
    let results = tournament.run_generation().unwrap();

    assert_eq!(results.len(), 7);
}

#[test]
fn tournament_rejects_invalid_workspace_config() {
    let error = Tournament::new(TournamentConfig {
        levels: 1,
        ..TournamentConfig::fast_test()
    })
    .unwrap_err();

    assert!(matches!(
        error,
        crate::error::FractalError::InvalidConfig(_)
    ));
}

#[test]
fn generator_rejects_vocab_that_cannot_encode_reserved_tokens() {
    let error = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: MIN_VOCAB_SIZE - 1,
        ..GeneratorConfig::default()
    })
    .unwrap_err();

    assert!(matches!(
        error,
        crate::error::FractalError::InvalidConfig(_)
    ));
}

#[derive(Module, Debug)]
struct CountingRule<B: Backend> {
    hidden_dim: usize,
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend> CountingRule<B> {
    fn new(hidden_dim: usize) -> Self {
        Self {
            hidden_dim,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> FractalRule<B> for CountingRule<B> {
    fn apply(
        &self,
        state: &FractalState<B>,
        _x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, crate::error::FractalError> {
        APPLY_COUNTER.fetch_add(1, Ordering::SeqCst);
        Ok(state.clone())
    }

    fn name(&self) -> &'static str {
        "counting_rule"
    }

    fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    fn state_layout(&self) -> StateLayout {
        StateLayout::Flat
    }

    fn clone_box(&self) -> Box<dyn FractalRule<B>> {
        Box::new(Self::new(self.hidden_dim))
    }
}

#[test]
fn recurrent_model_uses_only_rule_apply_for_state_transitions() {
    let device = Default::default();
    APPLY_COUNTER.store(0, Ordering::SeqCst);
    let rule = CountingRule::<TestBackend>::new(8);
    let model = FractalModel::new(64, 8, 3, 1.1, PAD_TOKEN, rule, &device);
    let input_ids = Tensor::<TestBackend, 2, burn::tensor::Int>::zeros([2, 4], &device);

    let _ = model.forward_tokens(input_ids).unwrap();

    assert_eq!(APPLY_COUNTER.load(Ordering::SeqCst), 12);
}
