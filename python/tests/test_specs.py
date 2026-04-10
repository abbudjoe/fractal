from __future__ import annotations

import unittest

from python.specs.common import DeviceRuntimeSpec, ValidationError
from python.specs.mini_moe import (
    MiniMoeArchitectureSpec,
    MiniMoeBackboneSpec,
    MiniMoeDispatchSpec,
    MiniMoeLayerSchedule,
    MiniMoeLayerScheduleKind,
    MiniMoeObservabilitySpec,
    MiniMoeRouterSpec,
    MiniMoeRuntimeSpec,
    MiniMoeStackSpec,
    MiniMoeSurfaceSpec,
    OneShotRouterSpec,
)
from python.specs.path1 import (
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    ReferenceSsmProfile,
    phase1_baseline_matrix,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)
from python.specs.runtime import PrimitiveStateTransformMode as SharedPrimitiveStateTransformMode
from python.specs.runtime import RuntimeOptimizationTarget, runtime_optimization_profile


class Path1SpecTests(unittest.TestCase):
    def test_runtime_spec_accepts_compile_modes(self) -> None:
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", compile_mode="reduce-overhead").validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(backend="cuda", dtype="bf16", compile_mode="made-up-mode").validate()

    def test_runtime_spec_accepts_env_kinds(self) -> None:
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", env_kind="compile-safe").validate()
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", env_kind="primitive-triton").validate()
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

    def test_p1_fractal_hybrid_variant_is_typed_distinct_from_p1(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P1_FRACTAL_HYBRID,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
        )
        variant.validate()
        self.assertIn("p1-fractal-hybrid", variant.label)

    def test_path1_reuses_shared_state_transform_mode_contract(self) -> None:
        self.assertIs(PrimitiveStateTransformMode.BLOCK_DIAGONAL_2, SharedPrimitiveStateTransformMode.BLOCK_DIAGONAL_2)

    def test_recurrent_scan_runtime_optimization_profile_exposes_scan_targets(self) -> None:
        profile = runtime_optimization_profile("recurrent-scan-hybrid")

        self.assertIn(RuntimeOptimizationTarget.PACKED_PROJECTIONS, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.SEQUENCE_SCAN_KERNEL, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.STRUCTURED_STATE_TRANSFORM, profile.targets)
        self.assertNotIn(RuntimeOptimizationTarget.ATTENTION_KERNEL, profile.targets)

    def test_pure_transformer_runtime_optimization_profile_exposes_transformer_targets(self) -> None:
        profile = runtime_optimization_profile("pure-transformer")

        self.assertIn(RuntimeOptimizationTarget.ATTENTION_KERNEL, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.KV_CACHE_LAYOUT, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.MLP_FUSION, profile.targets)
        self.assertNotIn(RuntimeOptimizationTarget.SEQUENCE_SCAN_KERNEL, profile.targets)


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


if __name__ == "__main__":
    unittest.main()
