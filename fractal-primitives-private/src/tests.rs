use burn::{backend::Candle, tensor::Tensor};
use fractal_core::{
    registry::SpeciesId,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

use crate::{
    species_registry, B1FractalGated, B2StableHierarchical, B3FractalHierarchical, B4Universal,
    P1Contractive, P2Mandelbrot, P3Hierarchical,
};

type TestBackend = Candle<f32, i64>;

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
fn species_registry_lists_every_species_once() {
    let ids = species_registry()
        .iter()
        .map(|species| species.id)
        .collect::<Vec<_>>();

    assert_eq!(ids, SpeciesId::ALL.to_vec());
}
