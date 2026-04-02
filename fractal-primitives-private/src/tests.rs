use burn::{
    backend::Candle,
    module::Param,
    tensor::{Tensor, TensorData},
};
use fractal_core::{
    diagnostics::{
        DiagnosticIdentity, DiagnosticProbeKind, DiagnosticProbeRequest, DiagnosticsPolicy,
        DiagnosticsRecorder, ProbeCadence, RuleProjectionDiagnosticContext, RuleProjectionKind,
        TensorLayoutTransform,
    },
    registry::SpeciesId,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
    StructuredDiagnosticsOutput, RunPhase,
};

use crate::{
    species_registry, B1FractalGated, B2StableHierarchical, B3FractalHierarchical, B4Universal,
    GeneralizedMobius, Ifs, JuliaRecursiveEscape, LogisticChaoticMap,
    MandelboxRecursiveDynEscapeRadius, P1Contractive, P1FractalHybrid, P1FractalHybridComposite,
    P1FractalHybridDynGate, P2Mandelbrot, P3Hierarchical,
};

type TestBackend = Candle<f32, i64>;

#[test]
fn primitives_preserve_declared_layout_and_shape() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::zeros([2, 8], &device);

    let primitives: Vec<Box<dyn FractalRule<TestBackend>>> = vec![
        Box::new(P1Contractive::new(8, &device)),
        Box::new(P3Hierarchical::new(8, 3, &device)),
        Box::new(B2StableHierarchical::new(8, 3, &device)),
        Box::new(B1FractalGated::new(8, &device)),
        Box::new(P1FractalHybrid::new(8, &device)),
        Box::new(P1FractalHybridComposite::new(8, &device)),
        Box::new(P1FractalHybridDynGate::new(8, &device)),
        Box::new(P2Mandelbrot::new(8, &device)),
        Box::new(B3FractalHierarchical::new(8, 3, &device)),
        Box::new(B4Universal::new(8, 3, &device)),
        Box::new(Ifs::new(8, &device)),
        Box::new(GeneralizedMobius::new(8, &device)),
        Box::new(LogisticChaoticMap::new(8, &device)),
        Box::new(JuliaRecursiveEscape::new(8, &device)),
        Box::new(MandelboxRecursiveDynEscapeRadius::new(8, &device)),
    ];

    for primitive in primitives {
        let state =
            FractalState::zeros(primitive.state_layout(), 2, primitive.hidden_dim(), &device)
                .unwrap();
        let next = primitive
            .apply(
                &state,
                &x,
                ApplyContext {
                    depth: 1,
                    max_depth: 4,
                },
            )
            .unwrap();
        assert_eq!(primitive.state_layout(), next.layout());
    }
}

#[test]
fn clone_box_preserves_metadata() {
    let device = Default::default();
    let primitive: Box<dyn FractalRule<TestBackend>> = Box::new(GeneralizedMobius::new(8, &device));
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
    let next = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 4,
            },
        )
        .unwrap();

    match next {
        FractalState::Hierarchical(tensor) => assert_eq!(tensor.dims(), [2, 4, 8]),
        _ => panic!("expected hierarchical state"),
    }
}

#[test]
fn p1_contractive_emits_typed_rule_projection_diagnostics() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 8], &device);
    let state = FractalState::zeros(StateLayout::Flat, 1, 8, &device).unwrap();
    let rule = P1Contractive::new(8, &device);
    let mut recorder = DiagnosticsRecorder::new(
        DiagnosticsPolicy {
            required: false,
            probes: vec![DiagnosticProbeRequest {
                kind: DiagnosticProbeKind::RuleProjection,
                cadence: ProbeCadence::EveryStep,
                position_interval: Some(1),
            }],
            structured_output: StructuredDiagnosticsOutput::Jsonl,
        },
        DiagnosticIdentity {
            experiment_run_id: "run-123".to_owned(),
            experiment_logical_name: Some("rule-projection-test".to_owned()),
            species: SpeciesId::P1Contractive.as_str().to_owned(),
            variant_name: "p1_contractive_v1".to_owned(),
        },
    );

    let next = rule
        .apply_with_diagnostics(
            &state,
            &x,
            ApplyContext {
                depth: 2,
                max_depth: 4,
            },
            Some(&mut recorder),
            Some(RuleProjectionDiagnosticContext {
                step: 0,
                tokens_seen: 0,
                position: 3,
                sequence_length: 4,
                recursion_depth: 2,
                max_recursion_depth: 4,
            }),
        )
        .unwrap();

    assert_eq!(next.layout(), StateLayout::Flat);
    let artifact = recorder.artifact();
    let events = artifact.events;
    assert_eq!(events.len(), 3);
    assert!(events.iter().all(|event| event.phase == RunPhase::Train));
    let projections = events
        .iter()
        .map(|event| match &event.event {
            fractal_core::DiagnosticEventKind::RuleProjection {
                position,
                sequence_length,
                recursion_depth,
                max_recursion_depth,
                projection,
                spec,
            } => {
                assert_eq!((*position, *sequence_length), (3, 4));
                assert_eq!((*recursion_depth, *max_recursion_depth), (2, 4));
                assert_eq!(spec.identity.rule_name, "p1_contractive");
                assert_eq!(spec.input_layout.shape, vec![1, 8]);
                assert_eq!(spec.output_layout.shape, vec![1, 8]);
                assert_eq!(
                    spec.linear_layout
                        .as_ref()
                        .expect("linear contract metadata should exist")
                        .backward_input_grad_rhs
                        .transform,
                    TensorLayoutTransform::Identity
                );
                assert_eq!(
                    spec.linear_layout
                        .as_ref()
                        .expect("linear contract metadata should exist")
                        .forward_rhs
                        .transform,
                    TensorLayoutTransform::TransposedUnsqueezedView
                );
                *projection
            }
            other => panic!("expected rule projection event, got {other:?}"),
        })
        .collect::<Vec<_>>();

    assert_eq!(
        projections,
        vec![
            RuleProjectionKind::Gate,
            RuleProjectionKind::StateMix,
            RuleProjectionKind::InputMix,
        ]
    );
}

#[test]
fn species_registry_lists_every_species_once() {
    let ids = species_registry()
        .iter()
        .map(|species| species.id)
        .collect::<Vec<_>>();

    assert_eq!(ids, SpeciesId::ALL.to_vec());
}

#[test]
fn ifs_dynamic_radius_shrinks_with_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state = FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device));
    let rule = Ifs::new(4, &device);

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let deep = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 10,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, deep);
}

#[test]
fn p2_dynamic_gate_clamp_changes_with_state_norm() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let low_state = FractalState::Complex(Tensor::<TestBackend, 2>::ones([1, 8], &device));
    let high_state =
        FractalState::Complex(Tensor::<TestBackend, 2>::ones([1, 8], &device).mul_scalar(8.0));
    let mut rule = P2Mandelbrot::new(4, &device);
    rule.g_proj.weight = Param::from_data(TensorData::new(vec![10.0f32; 32], [4, 8]), &device);
    rule.g_proj.bias = Some(Param::from_data(
        TensorData::new(vec![10.0f32; 8], [8]),
        &device,
    ));
    rule.c_proj.weight = Param::from_data(TensorData::new(vec![0.0f32; 32], [4, 8]), &device);
    rule.c_proj.bias = Some(Param::from_data(
        TensorData::new(vec![0.0f32; 8], [8]),
        &device,
    ));

    let low = rule
        .apply(
            &low_state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let high = rule
        .apply(
            &high_state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(low, high);
}

#[test]
fn p1_fractal_hybrid_dynamic_clamp_changes_with_state_norm() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let low_state = FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device));
    let high_state =
        FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device).mul_scalar(8.0));
    let rule = P1FractalHybrid::new(4, &device);

    let low = rule
        .apply(
            &low_state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 8,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let high = rule
        .apply(
            &high_state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 8,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(low, high);
}

#[test]
fn p1_fractal_hybrid_composite_only_injects_inner_rule_at_controlled_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state = FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device));
    let rule = P1FractalHybridComposite::new(4, &device);

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let injected = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 5,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, injected);
}

#[test]
fn p1_fractal_hybrid_dyn_gate_tightens_with_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state = FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device));
    let mut rule = P1FractalHybridDynGate::new(4, &device);
    rule.core.g_proj.weight = Param::from_data(TensorData::new(vec![10.0f32; 16], [4, 4]), &device);
    rule.core.g_proj.bias = Some(Param::from_data(
        TensorData::new(vec![10.0f32; 4], [4]),
        &device,
    ));

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let deep = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 10,
                max_depth: 10,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, deep);
}

#[test]
fn julia_recursive_escape_tightens_escape_radius_with_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state =
        FractalState::Complex(Tensor::<TestBackend, 2>::ones([1, 8], &device).mul_scalar(3.0));
    let rule = JuliaRecursiveEscape::new(4, &device);

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 8,
            },
        )
        .unwrap()
        .complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let deep = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 8,
                max_depth: 8,
            },
        )
        .unwrap()
        .complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, deep);
}

#[test]
fn mandelbox_recursive_escape_radius_changes_with_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state = FractalState::Flat(Tensor::<TestBackend, 2>::ones([1, 4], &device));
    let mut rule = MandelboxRecursiveDynEscapeRadius::new(4, &device);
    rule.drive_proj.weight = Param::from_data(TensorData::new(vec![0.0f32; 16], [4, 4]), &device);
    rule.drive_proj.bias = Some(Param::from_data(
        TensorData::new(vec![0.0f32; 4], [4]),
        &device,
    ));

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 6,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let deep = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 6,
                max_depth: 6,
            },
        )
        .unwrap()
        .flat()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, deep);
}

#[test]
fn b3_dynamic_radius_changes_hierarchical_complex_update_with_depth() {
    let device = Default::default();
    let x = Tensor::<TestBackend, 2>::ones([1, 4], &device);
    let state =
        FractalState::HierarchicalComplex(Tensor::<TestBackend, 3>::ones([1, 3, 8], &device));
    let rule = B3FractalHierarchical::new(4, 3, &device);

    let shallow = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 1,
                max_depth: 10,
            },
        )
        .unwrap()
        .hierarchical_complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let deep = rule
        .apply(
            &state,
            &x,
            ApplyContext {
                depth: 10,
                max_depth: 10,
            },
        )
        .unwrap()
        .hierarchical_complex()
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_ne!(shallow, deep);
}
