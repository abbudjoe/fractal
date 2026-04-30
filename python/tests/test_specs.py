from __future__ import annotations

import unittest

from python.specs.common import BenchmarkBudgetSpec, DeviceRuntimeSpec, ValidationError
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
    AttentionKernelProfile,
    FeedForwardProfile,
    HybridAttentionLayerRole,
    Path1ScaffoldProfile,
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    ReferenceSsmProfile,
    layer_schedule_signature,
    parse_layer_index_spec,
    parse_int_tuple_spec,
    parse_layer_schedule_spec,
    parse_reference_ssm_profile_schedule_spec,
    phase1_attention_only_variant,
    phase1_baseline_matrix,
    Path1ModelShape,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)
from python.specs.runtime import PrimitiveStateTransformMode as SharedPrimitiveStateTransformMode
from python.specs.runtime import RuntimeOptimizationTarget, runtime_optimization_profile


class Path1SpecTests(unittest.TestCase):
    def test_budget_spec_accepts_sparse_train_loss_recording(self) -> None:
        BenchmarkBudgetSpec(train_loss_record_interval=128).validate()
        with self.assertRaises(ValidationError):
            BenchmarkBudgetSpec(train_loss_record_interval=0).validate()

    def test_budget_spec_accepts_muon_reference_optimizer_profile(self) -> None:
        BenchmarkBudgetSpec(optimizer_profile="adam-fused").validate()
        BenchmarkBudgetSpec(optimizer_profile="adam-triton-2d").validate()
        BenchmarkBudgetSpec(
            optimizer_profile="muon-reference",
            muon_weight_decay=0.01,
            muon_momentum=0.9,
            muon_ns_steps=4,
            muon_adjust_lr_fn="match_rms_adamw",
        ).validate()
        with self.assertRaises(ValidationError):
            BenchmarkBudgetSpec(optimizer_profile="mystery").validate()
        with self.assertRaises(ValidationError):
            BenchmarkBudgetSpec(muon_ns_steps=0).validate()
        with self.assertRaises(ValidationError):
            BenchmarkBudgetSpec(muon_adjust_lr_fn="mystery").validate()

    def test_path1_shape_accepts_explicit_attention_kernel(self) -> None:
        shape = Path1ModelShape(attention_kernel=AttentionKernelProfile.FLEX_LOCAL)

        shape.validate()

        self.assertEqual(shape.attention_kernel, AttentionKernelProfile.FLEX_LOCAL)

    def test_path1_shape_rejects_flex_local_non_power_of_two_head_dim(self) -> None:
        shape = Path1ModelShape(
            d_model=480,
            head_count=10,
            attention_kernel=AttentionKernelProfile.FLEX_LOCAL,
        )

        with self.assertRaisesRegex(ValidationError, "power-of-two head_dim"):
            shape.validate()

    def test_path1_shape_allows_flash_local_non_power_of_two_head_dim(self) -> None:
        shape = Path1ModelShape(
            d_model=480,
            head_count=10,
            attention_kernel=AttentionKernelProfile.FLASH_LOCAL,
        )

        shape.validate()

        self.assertEqual(shape.attention_kernel, AttentionKernelProfile.FLASH_LOCAL)

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

    def test_runtime_spec_tracks_native_kernel_backend_contracts(self) -> None:
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", head_loss_backend="streaming-kernel").validate()
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", ffn_backend="manual-autograd").validate()
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", ffn_backend="triton-gelu").validate()
        DeviceRuntimeSpec(backend="cuda", dtype="bf16", ffn_backend="recompute").validate()
        with self.assertRaisesRegex(ValidationError, "streaming-kernel"):
            DeviceRuntimeSpec(backend="cpu", dtype="fp32", head_loss_backend="streaming-kernel").validate()
        with self.assertRaisesRegex(ValidationError, "ffn_backend=manual-autograd"):
            DeviceRuntimeSpec(backend="mps", dtype="fp32", ffn_backend="manual-autograd").validate()
        with self.assertRaisesRegex(ValidationError, "ffn_backend=triton-gelu"):
            DeviceRuntimeSpec(backend="cpu", dtype="fp32", ffn_backend="triton-gelu").validate()
        with self.assertRaisesRegex(ValidationError, "head_loss_backend"):
            DeviceRuntimeSpec(backend="cuda", dtype="bf16", head_loss_backend="mystery").validate()
        with self.assertRaisesRegex(ValidationError, "ffn_backend"):
            DeviceRuntimeSpec(backend="cuda", dtype="bf16", ffn_backend="mystery").validate()

    def test_runtime_spec_accepts_mps_fp32_only(self) -> None:
        DeviceRuntimeSpec(backend="mps", dtype="fp32").validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(backend="mps", dtype="bf16").validate()

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

    def test_attention_only_surgical_feed_forward_profile_tracks_layers(self) -> None:
        variant = phase1_attention_only_variant(
            feed_forward_profile=FeedForwardProfile.MLP_EML_GATED,
            feed_forward_layer_indices=(3, 4),
            eml_slot_count=4,
            eml_tree_depth=2,
        )

        variant.validate()

        self.assertEqual(variant.feed_forward_layer_indices, (3, 4))
        self.assertEqual(variant.label, "attention-only-mlp-eml-gated-slots4-depth2-layers3-4")

    def test_parse_layer_index_spec_rejects_duplicate_indices(self) -> None:
        self.assertEqual(parse_layer_index_spec("4, 3"), (3, 4))
        with self.assertRaises(ValidationError):
            parse_layer_index_spec("2 2")

    def test_attention_only_routed_feed_forward_profile_tracks_route_fraction(self) -> None:
        variant = phase1_attention_only_variant(
            feed_forward_profile=FeedForwardProfile.MLP_EML_ROUTED,
            feed_forward_layer_indices=(4,),
            eml_slot_count=4,
            eml_tree_depth=2,
            eml_route_fraction=0.25,
        )

        variant.validate()

        self.assertEqual(variant.feed_forward_layer_indices, (4,))
        self.assertEqual(variant.eml_route_fraction, 0.25)
        self.assertEqual(variant.label, "attention-only-mlp-eml-routed-slots4-depth2-route25pct-layers4")

    def test_attention_only_parcae_scaffold_tracks_loop_count(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=3,
        )

        variant.validate()

        self.assertEqual(variant.scaffold_profile, Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION)
        self.assertEqual(variant.parcae_loop_count, 3)
        self.assertEqual(variant.label, "attention-only-parcae-looped-attention-loops3")

    def test_attention_only_parcae_scaffold_tracks_cuda_parity_knobs(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_discretization="stable-exp",
            parcae_control_stride=4,
            parcae_recurrent_compile_mode="max-autotune",
            parcae_loop_update_backend="compiled",
        )

        variant.validate()

        self.assertEqual(variant.parcae_backward_steps, 1)
        self.assertEqual(variant.parcae_prelude_norm_kind, "rmsnorm")
        self.assertEqual(variant.parcae_discretization, "stable-exp")
        self.assertEqual(variant.parcae_control_stride, 4)
        self.assertEqual(variant.parcae_recurrent_compile_mode, "max-autotune")
        self.assertEqual(variant.parcae_loop_update_backend, "compiled")
        self.assertEqual(
            variant.label,
            "attention-only-parcae-p20-control-looped-attention-loops2-bwd1-prenorm-rmsnorm-ctrlstride4-rcompile-max-autotune-loopupdate-compiled",
        )

        manual_variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_update_backend="manual-autograd",
        )

        manual_variant.validate()
        self.assertEqual(manual_variant.parcae_loop_update_backend, "manual-autograd")
        self.assertIn("loopupdate-manual-autograd", manual_variant.label)

        lean_variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_update_backend="lean-eager",
        )

        lean_variant.validate()
        self.assertEqual(lean_variant.parcae_loop_update_backend, "lean-eager")
        self.assertIn("loopupdate-lean-eager", lean_variant.label)

        triton_variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_update_backend="triton-glue",
        )

        triton_variant.validate()
        self.assertEqual(triton_variant.parcae_loop_update_backend, "triton-glue")
        self.assertIn("loopupdate-triton-glue", triton_variant.label)

        triton_forward_variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_update_backend="triton-loop-forward",
        )

        triton_forward_variant.validate()
        self.assertEqual(triton_forward_variant.parcae_loop_update_backend, "triton-loop-forward")
        self.assertIn("loopupdate-triton-loop-forward", triton_forward_variant.label)

    def test_standard_attention_ignores_parcae_runtime_knobs(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.STANDARD,
            parcae_loop_count=7,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_discretization="zoh",
            parcae_loop_d_model=64,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            parcae_loop_layer_count=2,
            parcae_hourglass_band_schedule=(2, 1, 2),
            parcae_control_position_kind="learned",
            parcae_control_stride=4,
            parcae_control_state_transform="trainable-block-diagonal-8",
            parcae_recurrent_compile_mode="max-autotune",
            parcae_loop_update_backend="triton-loop-forward",
            parcae_scaffold_backend="compiled",
            parcae_band_block_contract="compiled-direct",
            parcae_band_prepare_backend="compiled",
            parcae_output_mix_backend="triton",
            parcae_fuse_first_state_mix=True,
            attention_position_contract="attention-only",
        )

        variant.validate()

        self.assertEqual(variant.label, "attention-only-attnpos-attention-only")
        self.assertEqual(variant.scaffold_profile, Path1ScaffoldProfile.STANDARD)
        self.assertEqual(variant.parcae_loop_update_backend, "eager")
        self.assertEqual(variant.parcae_band_prepare_backend, "standard")
        self.assertEqual(variant.parcae_loop_d_model, None)
        self.assertFalse(variant.parcae_fuse_first_state_mix)

    def test_attention_only_parcae_rejects_retired_tiny_native_backward_backends(self) -> None:
        for backend in (
            "triton-loop-forward-native-bwd",
            "triton-loop-forward-frozen-control-bwd",
        ):
            with self.subTest(backend=backend):
                variant = phase1_attention_only_variant(
                    scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
                    parcae_loop_update_backend=backend,
                )

                with self.assertRaisesRegex(ValidationError, "parcae_loop_update_backend"):
                    variant.validate()

    def test_attention_only_attention_position_contract_tracks_label(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            position_encoding_kind="learned",
            attention_position_contract="attention-only",
            parcae_control_position_kind="learned",
        )

        variant.validate()

        self.assertEqual(variant.attention_position_contract, "attention-only")
        self.assertIn("attnpos-attention-only", variant.label)

    def test_parcae_hourglass_pass_count_requires_hourglass_scaffold(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_hourglass_pass_count=2,
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
        )

        variant.validate()

        self.assertEqual(variant.parcae_hourglass_pass_count, 2)
        self.assertIn("passes2", variant.label)

        invalid = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_hourglass_pass_count=2,
        )
        with self.assertRaisesRegex(ValidationError, "hourglass pass count"):
            invalid.validate()

    def test_parcae_hourglass_band_schedule_requires_hourglass_scaffold(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=12),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_hourglass_band_schedule=(3, 2, 3, 2, 2),
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
        )

        variant.validate()

        self.assertEqual(variant.parcae_hourglass_band_schedule, (3, 2, 3, 2, 2))
        self.assertIn("bands3x2x3x2x2", variant.label)

        invalid_sum = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=12),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_hourglass_band_schedule=(3, 2, 3, 2, 1),
        )
        with self.assertRaisesRegex(ValidationError, "band_schedule must sum"):
            invalid_sum.validate()

        invalid_scaffold = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=12),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_hourglass_band_schedule=(3, 2, 3, 2, 2),
        )
        with self.assertRaisesRegex(ValidationError, "hourglass band schedule"):
            invalid_scaffold.validate()

    def test_parse_int_tuple_spec_preserves_order(self) -> None:
        self.assertEqual(
            parse_int_tuple_spec("3, 2 3,2,2", name="band schedule"),
            (3, 2, 3, 2, 2),
        )
        with self.assertRaisesRegex(ValidationError, "positive integers"):
            parse_int_tuple_spec("3,0,2", name="band schedule")

    def test_parcae_loop_layer_count_requires_hourglass_scaffold(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=8),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_d_model=64,
            parcae_loop_head_count=4,
            parcae_loop_layer_count=1,
        )

        variant.validate()

        self.assertEqual(variant.parcae_loop_layer_count, 1)
        self.assertIn("looplayers1", variant.label)

        non_hourglass = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_layer_count=1,
        )
        with self.assertRaises(ValidationError):
            non_hourglass.validate()

    def test_attention_only_parcae_bx_and_p20_control_scaffolds_validate(self) -> None:
        bx = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )

        bx.validate()
        p20.validate()

        self.assertEqual(bx.label, "attention-only-parcae-bx-looped-attention-loops2")
        self.assertEqual(p20.label, "attention-only-parcae-p20-control-looped-attention-loops2")

    def test_parcae_control_position_requires_p20_control_scaffold(self) -> None:
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_control_position_kind="learned",
        )
        p20.validate()

        self.assertEqual(p20.parcae_control_position_kind, "learned")
        self.assertIn("ctrlpos-learned", p20.label)

        bx = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            parcae_control_position_kind="learned",
        )
        with self.assertRaises(ValidationError):
            bx.validate()

    def test_parcae_control_stride_requires_p20_control_scaffold(self) -> None:
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_control_stride=4,
        )
        p20.validate()

        self.assertEqual(p20.parcae_control_stride, 4)
        self.assertIn("ctrlstride4", p20.label)

        bx = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            parcae_control_stride=4,
        )
        with self.assertRaises(ValidationError):
            bx.validate()

    def test_parcae_control_state_transform_requires_p20_control_scaffold(self) -> None:
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_control_state_transform="frozen-identity",
        )
        p20.validate()

        self.assertIn("ctrlx-frozen-identity", p20.label)

        bx = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            parcae_control_state_transform="frozen-identity",
        )
        with self.assertRaises(ValidationError):
            bx.validate()

    def test_parcae_control_state_transform_accepts_block_diagonal_8_profile(self) -> None:
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_control_state_transform="trainable-block-diagonal-8",
        )
        p20.validate()

        self.assertIn("ctrlx-trainable-block-diagonal-8", p20.label)

    def test_attention_position_contract_rejects_unknown_value(self) -> None:
        variant = phase1_attention_only_variant(attention_position_contract="mystery-contract")
        with self.assertRaises(ValidationError):
            variant.validate()

    def test_attention_variant_accepts_layernorm_final_norm(self) -> None:
        variant = phase1_attention_only_variant(final_norm_kind="layernorm")

        variant.validate()

        self.assertEqual(variant.final_norm_kind, "layernorm")

    def test_gated_deltanet_reference_variant_tracks_profile(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_TORCH)

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertEqual(variant.label, "reference-ssm-hybrid-gated-deltanet-torch")

    def test_fla_gated_deltanet_reference_variant_tracks_runtime_profile(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_FLA)

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertTrue(variant.reference_ssm_profile.is_fla_gated_deltanet)
        self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertEqual(variant.label, "reference-ssm-hybrid-gated-deltanet-fla")

    def test_fla_gdnp_compatible_reference_variant_tracks_runtime_profile(self) -> None:
        expected_laws = {
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT: "single-read",
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ: "multi-read",
        }

        for profile, law in expected_laws.items():
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)

                variant.validate()

                self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
                self.assertTrue(variant.reference_ssm_profile.is_fla_gdnp_compatible)
                self.assertEqual(variant.reference_ssm_profile.fla_gdnp_compatible_law, law)
                self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
                self.assertEqual(variant.final_norm_kind, "rmsnorm")
                self.assertIn(profile.value, variant.label)

    def test_fla_gdnp_control_conditioned_reference_variant_tracks_runtime_profile(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY,
        )

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertTrue(variant.reference_ssm_profile.is_fla_gdnp_control_conditioned)
        self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertIn("gated-deltanet-fla-p20-control-tiny", variant.label)

    def test_fla_gdn_control_shell_reference_variant_tracks_runtime_profile(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL,
        )

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertTrue(variant.reference_ssm_profile.is_fla_gdn_control_shell)
        self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertIn("gated-deltanet-fla-control-shell", variant.label)

    def test_reference_variant_accepts_sparse_reference_profile_schedule(self) -> None:
        layer_schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = parse_reference_ssm_profile_schedule_spec(
            "gated-deltanet-torch gated-deltanet-torch gated-deltanet-torch "
            "gated-deltanet-torch gated-deltanet-fla-control-shell "
            "gated-deltanet-torch gated-deltanet-torch gated-deltanet-torch "
            "gated-deltanet-torch gated-deltanet-torch"
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(total_layers=len(layer_schedule)),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=layer_schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )

        variant.validate()

        self.assertEqual(variant.scaffold_profile, Path1ScaffoldProfile.PR5_HYBRID_GDN)
        self.assertEqual(len(variant.reference_ssm_profile_schedule), 10)
        self.assertEqual(
            variant.reference_profile_for_ordinal(4),
            ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL,
        )
        self.assertEqual(variant.reference_p20_ramp_init, 0.01)
        self.assertIn("pr5-hybrid-gdn", variant.label)
        self.assertIn("profiles-gdnx4-gdn-fla-shell-gdnx5", variant.label)

    def test_reference_variant_rejects_invalid_p20_ramp_init(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT,
            reference_p20_ramp_init=1.0,
        )

        with self.assertRaisesRegex(ValidationError, "reference_p20_ramp_init"):
            variant.validate()

    def test_reference_variant_rejects_profile_schedule_length_mismatch(self) -> None:
        layer_schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(total_layers=len(layer_schedule)),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=layer_schedule,
            profile_schedule=(ReferenceSsmProfile.GATED_DELTANET_TORCH,),
        )

        with self.assertRaisesRegex(ValidationError, "reference_ssm_profile_schedule length"):
            variant.validate()

    def test_gdnp_fused_reference_variant_tracks_profile(self) -> None:
        expected_laws = {
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH: "value",
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH: "beta",
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH: "qkv",
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH: "residual-readout",
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH: "multi-read",
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH: "all",
        }

        for profile, law in expected_laws.items():
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)

                variant.validate()

                self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
                self.assertTrue(variant.reference_ssm_profile.is_gdnp_fused)
                self.assertFalse(variant.reference_ssm_profile.is_composite)
                self.assertEqual(variant.reference_ssm_profile.gdnp_fused_law, law)
                self.assertEqual(variant.final_norm_kind, "rmsnorm")
                self.assertIn(profile.value, variant.label)

    def test_composite_reference_variants_expose_branch_contracts(self) -> None:
        expected = {
            ReferenceSsmProfile.GATED_DELTANET_P20_TORCH: ("gdn", "p20"),
            ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH: ("gdn", "p20_thin"),
            ReferenceSsmProfile.P20_MAMBA3_TORCH: ("p20", "mamba3"),
            ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH: ("gdn", "mamba3"),
            ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH: ("gdn", "p20", "mamba3"),
        }

        for profile, branches in expected.items():
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)

                variant.validate()

                self.assertTrue(profile.is_composite)
                self.assertEqual(profile.composite_branches, branches)
                self.assertEqual(variant.final_norm_kind, "rmsnorm")
                self.assertIn(profile.value, variant.label)

    def test_p20_reference_scan_variants_expose_width_contracts(self) -> None:
        full = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.P20_TORCH)
        thin = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.P20_THIN_TORCH)

        full.validate()
        thin.validate()

        self.assertTrue(full.reference_ssm_profile.is_p20_scan)
        self.assertEqual(full.reference_ssm_profile.p20_branch_width_factor, 1.0)
        self.assertTrue(thin.reference_ssm_profile.is_p20_scan)
        self.assertEqual(thin.reference_ssm_profile.p20_branch_width_factor, 0.5)
        self.assertEqual(full.final_norm_kind, "rmsnorm")
        self.assertEqual(thin.final_norm_kind, "rmsnorm")

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

    def test_parse_layer_schedule_spec_accepts_compact_schedule_strings(self) -> None:
        schedule = parse_layer_schedule_spec("AAAAAPAAAAA")

        self.assertEqual(len(schedule), 11)
        self.assertEqual(schedule[5], HybridAttentionLayerRole.PRIMITIVE)
        self.assertEqual(layer_schedule_signature(schedule), "aaaaapaaaaa")

    def test_parse_layer_schedule_spec_accepts_shared_swa_token(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")

        self.assertEqual(len(schedule), 12)
        self.assertEqual(schedule[5], HybridAttentionLayerRole.EXACT_ATTENTION)
        self.assertEqual(schedule[11], HybridAttentionLayerRole.SHARED_EXACT_ATTENTION)
        self.assertEqual(layer_schedule_signature(schedule), "rrrrrarrrrrs")

    def test_parse_layer_schedule_spec_rejects_unknown_tokens(self) -> None:
        with self.assertRaises(ValidationError):
            parse_layer_schedule_spec("AAAXP")

    def test_primitive_variant_tracks_custom_schedule_in_label(self) -> None:
        schedule = parse_layer_schedule_spec("AAAAAPAAAAA")
        variant = phase1_primitive_variant(
            shape=Path1ModelShape(total_layers=len(schedule)),
            primitive_profile=PrimitiveProfile.P20,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.SCALED,
            readout_mode=PrimitiveReadoutMode.PROJECTED,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
            layer_schedule=schedule,
        )

        variant.validate()
        self.assertEqual(variant.shape.total_layers, 11)
        self.assertIn("schedule-aaaaapaaaaa", variant.label)

    def test_attention_variant_tracks_shared_swa_schedule_in_label(self) -> None:
        schedule = parse_layer_schedule_spec("AAAAAS")
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=len(schedule)),
            layer_schedule=schedule,
        )

        variant.validate()
        self.assertEqual(variant.label, "attention-only-schedule-aaaaas")

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
