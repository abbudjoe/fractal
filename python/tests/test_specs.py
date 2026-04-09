from __future__ import annotations

import unittest

import torch

from python.runtime import resolve_torch_device
from python.specs.mini_moe import (
    LearnedGateTeacherKind,
    MiniMoeArchitectureSpec,
    MiniMoeBackboneSpec,
    MiniMoeDispatchExecutionStrategy,
    MiniMoeDispatchSpec,
    MiniMoeLayerSchedule,
    MiniMoeLayerScheduleKind,
    MiniMoeObservabilitySpec,
    MiniMoeRouterSpec,
    MiniMoeRuntimeSpec,
    MiniMoeStackSpec,
    MiniMoeSurfaceSpec,
    OneShotRouterSpec,
    RecurrentPreExpertRouterSpec,
    RecurrentRoundExecutionStrategy,
    MiniMoePreset,
    RecurrentRoundGateKind,
    RecurrentRoundGateSpec,
    contiguous_layer_bands,
    transfer_round2_layer_bands_by_anchor_fill,
    transfer_round2_layer_bands_by_scaled_span,
    transfer_round2_layer_indices_by_depth_fraction,
)
from python.specs.common import DeviceRuntimeSpec, ValidationError
from python.specs.path1 import (
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    ReferenceSsmProfile,
    phase1_attention_only_variant,
    phase1_baseline_matrix,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)
from python.specs.runtime import (
    PrimitiveStateTransformMode as SharedPrimitiveStateTransformMode,
    RuntimeOptimizationFamily,
)


class Path1SpecTests(unittest.TestCase):
    def test_runtime_spec_accepts_compile_modes(self) -> None:
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", compile_mode="reduce-overhead").validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(backend="cuda", dtype="bf16", compile_mode="made-up-mode").validate()

    def test_runtime_spec_accepts_env_kinds(self) -> None:
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", env_kind="compile-safe").validate()
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", env_kind="primitive-triton").validate()
        DeviceRuntimeSpec(backend="mps", dtype="fp32", env_kind="requirements-only").validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(backend="cuda", dtype="bf16", env_kind="mystery-env").validate()

    def test_runtime_spec_rejects_invalid_triton_backend_contract(self) -> None:
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(
                backend="cuda",
                dtype="bf16",
                env_kind="compile-safe",
                primitive_runtime_backend="triton",
            ).validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(
                backend="cuda",
                dtype="bf16",
                env_kind="primitive-triton",
                compile_mode="reduce-overhead",
            ).validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(
                backend="mps",
                dtype="fp32",
                env_kind="primitive-triton",
                primitive_runtime_backend="triton",
            ).validate()
        DeviceRuntimeSpec(
            backend="cuda",
            dtype="bf16",
            env_kind="primitive-triton",
            primitive_runtime_backend="triton",
        ).validate()

    def test_baseline_matrix_validates(self) -> None:
        matrix = phase1_baseline_matrix(
            reference_profile=ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
            primitive_profile=PrimitiveProfile.P23,
            residual_mode=PrimitiveResidualMode.GATED,
            readout_mode=PrimitiveReadoutMode.PROJECTED_NORM,
            norm_mode=PrimitiveNormMode.RESIDUAL_RENORM,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        matrix.validate()
        self.assertEqual(matrix.reference_ssm_hybrid.reference_ssm_profile, ReferenceSsmProfile.MAMBA3_SISO_RUNTIME)
        self.assertEqual(matrix.primitive_hybrid.primitive_profile, PrimitiveProfile.P23)

    def test_primitive_variant_requires_wrapper_modes(self) -> None:
        variant = phase1_primitive_variant()
        variant.validate()

    def test_reference_variant_tracks_profile(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.MAMBA3_MIMO_REFERENCE)
        variant.validate()
        self.assertTrue(variant.reference_ssm_profile.is_mimo)
        self.assertEqual(variant.label, "reference-ssm-hybrid-mamba3-mimo-reference")

    def test_primitive_variant_label_includes_wrapper_identity(self) -> None:
        variant_a = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P23,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.GATED,
            readout_mode=PrimitiveReadoutMode.PROJECTED_NORM,
            norm_mode=PrimitiveNormMode.RESIDUAL_RENORM,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        variant_b = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P23,
            execution_profile=PrimitiveExecutionProfile.REFERENCE,
            residual_mode=PrimitiveResidualMode.PLAIN,
            readout_mode=PrimitiveReadoutMode.PROJECTED,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.MAMBA_RMS,
            state_transform_mode=PrimitiveStateTransformMode.DENSE,
        )
        self.assertNotEqual(variant_a.label, variant_b.label)
        self.assertIn("runtime", variant_a.label)
        self.assertIn("gated", variant_a.label)
        self.assertIn("projected-norm", variant_a.label)
        self.assertIn("block-diagonal-4", variant_a.label)

    def test_primitive_variant_label_tracks_block_diagonal_2(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P20,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.SCALED,
            readout_mode=PrimitiveReadoutMode.PROJECTED,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )
        variant.validate()
        self.assertIn("block-diagonal-2", variant.label)

    def test_path1_reuses_shared_state_transform_mode_contract(self) -> None:
        self.assertIs(PrimitiveStateTransformMode.BLOCK_DIAGONAL_2, SharedPrimitiveStateTransformMode.BLOCK_DIAGONAL_2)

    def test_attention_only_variant_uses_pure_transformer_runtime_family(self) -> None:
        variant = phase1_attention_only_variant()
        self.assertEqual(
            variant.runtime_optimization_family,
            RuntimeOptimizationFamily.PURE_TRANSFORMER,
        )

    def test_hybrid_variants_use_recurrent_scan_runtime_family(self) -> None:
        reference_variant = phase1_reference_ssm_variant()
        primitive_variant = phase1_primitive_variant()
        self.assertEqual(
            reference_variant.runtime_optimization_family,
            RuntimeOptimizationFamily.RECURRENT_SCAN_HYBRID,
        )
        self.assertEqual(
            primitive_variant.runtime_optimization_family,
            RuntimeOptimizationFamily.RECURRENT_SCAN_HYBRID,
        )


class MiniMoeSpecTests(unittest.TestCase):
    def test_mini_moe_surface_resolves_all_layers(self) -> None:
        surface = MiniMoeSurfaceSpec(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=None,
                label="phase1-mini-moe-reference",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=8,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(kind=MiniMoeLayerScheduleKind.ALL_LAYERS),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(kind="one_shot", one_shot=OneShotRouterSpec()),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )
        surface.validate()
        self.assertEqual(surface.architecture.resolved_layout().moe_layers, tuple(range(8)))

    def test_phase1_defaults_validate(self) -> None:
        reference = MiniMoeSurfaceSpec.phase1_reference_default()
        recurrent = MiniMoeSurfaceSpec.phase1_recurrent_default()
        recurrent_gated = MiniMoeSurfaceSpec.phase1_recurrent_gated_default()
        recurrent_scaled_gated = MiniMoeSurfaceSpec.phase1_recurrent_scaled_margin_gated_default()
        recurrent_fraction_gated = MiniMoeSurfaceSpec.phase1_recurrent_fraction_gated_default()
        recurrent_entropy_gated = MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default()
        recurrent_learned_gated = MiniMoeSurfaceSpec.phase1_recurrent_learned_gated_default()
        recurrent_learned_score_gated = MiniMoeSurfaceSpec.phase1_recurrent_learned_score_gated_default()
        reference.validate()
        recurrent.validate()
        recurrent_gated.validate()
        recurrent_scaled_gated.validate()
        recurrent_fraction_gated.validate()
        recurrent_entropy_gated.validate()
        recurrent_learned_gated.validate()
        recurrent_learned_score_gated.validate()
        self.assertEqual(reference.architecture.preset, MiniMoePreset.PHASE1_REFERENCE)
        self.assertEqual(recurrent.architecture.preset, MiniMoePreset.PHASE1_RECURRENT)
        self.assertEqual(recurrent_gated.architecture.preset, MiniMoePreset.PHASE1_RECURRENT_GATED)
        self.assertEqual(
            recurrent_scaled_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW,
        )
        self.assertEqual(
            recurrent_fraction_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.TARGET_APPLIED_FRACTION,
        )
        self.assertEqual(
            recurrent_entropy_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
        )
        self.assertEqual(
            recurrent_learned_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
        )
        self.assertEqual(
            recurrent_learned_score_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
        )
        self.assertEqual(reference.architecture.resolved_layout().moe_layers, tuple(range(8)))
        self.assertEqual(recurrent.architecture.router.recurrent_pre_expert.round_count, 2)
        self.assertEqual(
            recurrent_gated.architecture.router.recurrent_pre_expert.gate.kind,
            RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
        )
        self.assertEqual(
            recurrent_gated.architecture.router.recurrent_pre_expert.execution_strategy,
            RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        self.assertEqual(
            reference.resolved_dispatch().execution_strategy,
            MiniMoeDispatchExecutionStrategy.DENSE_GATHER,
        )
        self.assertEqual(
            reference.runtime_optimization_family,
            RuntimeOptimizationFamily.TRANSFORMER_MOE_ROUTING,
        )

    def test_mini_moe_uses_transformer_moe_routing_runtime_family(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_default()
        self.assertEqual(
            surface.runtime_optimization_family,
            RuntimeOptimizationFamily.TRANSFORMER_MOE_ROUTING,
        )

    def test_observability_trace_budget_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "max_token_route_traces_per_layer must be non-negative",
        ):
            MiniMoeObservabilitySpec(max_token_route_traces_per_layer=-1).validate()

    def test_recurrent_gate_threshold_must_be_in_open_unit_interval(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "threshold must be in the open interval",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
                threshold=0.0,
            ).validate()

    def test_recurrent_scaled_margin_gate_requires_reference_expert_count(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "must set reference_experts_per_block",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW,
                threshold=0.02,
            ).validate()

    def test_recurrent_gate_target_applied_fraction_must_be_in_unit_interval(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "target_applied_fraction must be in the interval",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.TARGET_APPLIED_FRACTION,
                target_applied_fraction=0.0,
            ).validate()

    def test_recurrent_gate_normalized_entropy_threshold_must_be_in_unit_interval(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "normalized_entropy_threshold must be in the interval",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
                normalized_entropy_threshold=0.0,
            ).validate()

    def test_recurrent_gate_learned_prior_must_be_in_open_unit_interval(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "learned_prior_probability must be in the open interval",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                learned_hidden_dim=8,
                learned_prior_probability=1.0,
                teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                teacher_supervision_weight=1.0,
            ).validate()

    def test_recurrent_gate_learned_score_top_fraction_requires_fraction(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "must set target_applied_fraction",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                learned_hidden_dim=8,
                learned_prior_probability=0.2,
                teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                teacher_supervision_weight=1.0,
            ).validate()

    def test_recurrent_gate_learned_teacher_weight_must_be_positive(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "teacher_supervision_weight must be greater than zero",
        ):
            RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                learned_hidden_dim=8,
                learned_prior_probability=0.2,
                target_applied_fraction=0.2,
                teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                teacher_supervision_weight=0.0,
            ).validate()

    def test_recurrent_round2_layer_indices_resolve_to_all_moe_layers_by_default(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default()
        surface.validate()
        self.assertEqual(
            surface.architecture.resolved_round2_layer_indices(),
            tuple(range(surface.architecture.backbone.total_layers)),
        )

    def test_recurrent_round2_layer_indices_must_target_moe_layers(self) -> None:
        surface = MiniMoeSurfaceSpec(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=None,
                label="phase1-mini-moe-selective-round2-invalid",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=128,
                    head_count=4,
                    total_layers=4,
                    local_window=256,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.EXPLICIT,
                        explicit_layers=(0, 2),
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=64,
                        gate=RecurrentRoundGateSpec(
                            kind=RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
                            normalized_entropy_threshold=0.95,
                        ),
                        round2_layer_indices=(1,),
                    ),
                ),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )
        with self.assertRaisesRegex(
            ValidationError,
            "is not a routed MoE layer",
        ):
            surface.validate()

    def test_transfer_round2_layer_indices_by_depth_fraction_scales_mask(self) -> None:
        self.assertEqual(
            transfer_round2_layer_indices_by_depth_fraction(
                source_layer_indices=(2, 3, 4, 6, 7),
                source_total_layers=8,
                target_total_layers=16,
            ),
            (4, 6, 8, 12, 14),
        )

    def test_transfer_round2_layer_indices_by_depth_fraction_rejects_invalid_source(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "contains out-of-range layer 8",
        ):
            transfer_round2_layer_indices_by_depth_fraction(
                source_layer_indices=(2, 8),
                source_total_layers=8,
                target_total_layers=16,
            )

    def test_contiguous_layer_bands_groups_adjacent_layers(self) -> None:
        self.assertEqual(
            contiguous_layer_bands((2, 3, 4, 6, 7)),
            ((2, 3, 4), (6, 7)),
        )

    def test_transfer_round2_layer_bands_by_anchor_fill_preserves_band_topology(self) -> None:
        self.assertEqual(
            transfer_round2_layer_bands_by_anchor_fill(
                source_layer_indices=(2, 3, 4, 6, 7),
                source_total_layers=8,
                target_total_layers=16,
            ),
            (4, 5, 6, 7, 8, 12, 13, 14),
        )

    def test_transfer_round2_layer_bands_by_scaled_span_preserves_fractional_band_width(self) -> None:
        self.assertEqual(
            transfer_round2_layer_bands_by_scaled_span(
                source_layer_indices=(2, 3, 4, 6, 7),
                source_total_layers=8,
                target_total_layers=16,
            ),
            (4, 5, 6, 7, 8, 9, 12, 13, 14, 15),
        )


class RuntimeSpecTests(unittest.TestCase):
    def test_mps_runtime_spec_validates_with_fp32(self) -> None:
        spec = DeviceRuntimeSpec(backend="mps", dtype="fp32")
        spec.validate()

    def test_mps_runtime_spec_rejects_bf16(self) -> None:
        with self.assertRaisesRegex(ValidationError, "bf16 is only supported for backend=cuda"):
            DeviceRuntimeSpec(backend="mps", dtype="bf16").validate()

    def test_resolve_torch_device_supports_mps_when_available(self) -> None:
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available in this environment")
        device = resolve_torch_device(DeviceRuntimeSpec(backend="mps", dtype="fp32"))
        self.assertEqual(device.type, "mps")


if __name__ == "__main__":
    unittest.main()
