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
    AttentionProfile,
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
    RecurrentHaltingProfile,
    RecurrentTokenRoutingProfile,
    ReferenceSsmProfile,
    TokenRoutingProfile,
    layer_schedule_signature,
    parse_layer_index_spec,
    parse_layer_schedule_spec,
    parse_reference_ssm_profile_schedule_spec,
    phase1_attention_only_variant,
    phase1_baseline_matrix,
    Path1ModelShape,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)
from python.specs.runtime import (
    PrimitiveStateTransformMode as SharedPrimitiveStateTransformMode,
)
from python.specs.runtime import RuntimeOptimizationTarget, runtime_optimization_profile


class Path1SpecTests(unittest.TestCase):
    def test_runtime_spec_accepts_compile_modes(self) -> None:
        DeviceRuntimeSpec(
            backend="cuda", dtype="bf16", compile_mode="reduce-overhead"
        ).validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(
                backend="cuda", dtype="bf16", compile_mode="made-up-mode"
            ).validate()

    def test_runtime_spec_accepts_env_kinds(self) -> None:
        DeviceRuntimeSpec(
            backend="cuda", dtype="bf16", env_kind="compile-safe"
        ).validate()
        DeviceRuntimeSpec(
            backend="cuda", dtype="bf16", env_kind="primitive-triton"
        ).validate()
        with self.assertRaises(ValidationError):
            DeviceRuntimeSpec(
                backend="cuda", dtype="bf16", env_kind="mystery-env"
            ).validate()

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
        self.assertEqual(
            matrix.reference_ssm_hybrid.reference_ssm_profile,
            ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
        )
        self.assertEqual(
            matrix.primitive_hybrid.primitive_profile, PrimitiveProfile.P23
        )

    def test_primitive_variant_requires_wrapper_modes(self) -> None:
        variant = phase1_primitive_variant()
        variant.validate()

    def test_reference_variant_tracks_profile(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.MAMBA3_MIMO_REFERENCE
        )
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
        self.assertEqual(
            variant.label, "attention-only-mlp-eml-gated-slots4-depth2-layers3-4"
        )

    def test_parse_layer_index_spec_rejects_duplicate_indices(self) -> None:
        self.assertEqual(parse_layer_index_spec("4, 3"), (3, 4))
        with self.assertRaises(ValidationError):
            parse_layer_index_spec("2 2")

    def test_attention_only_routed_feed_forward_profile_tracks_route_fraction(
        self,
    ) -> None:
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
        self.assertEqual(
            variant.label,
            "attention-only-mlp-eml-routed-slots4-depth2-route25pct-layers4",
        )

    def test_attention_only_parcae_scaffold_tracks_loop_count(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=3,
        )

        variant.validate()

        self.assertEqual(
            variant.scaffold_profile, Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION
        )
        self.assertEqual(variant.parcae_loop_count, 3)
        self.assertEqual(variant.label, "attention-only-parcae-looped-attention-loops3")

    def test_attention_only_looped_transformer_scaffolds_track_loop_count(
        self,
    ) -> None:
        expected_labels = {
            Path1ScaffoldProfile.FIXED_LOOPED_LM: (
                "attention-only-fixed-looped-lm-loops3"
            ),
            Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT: (
                "attention-only-looped-additive-input-loops3"
            ),
            Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE: (
                "attention-only-huginn-adapter-recurrence-loops3"
            ),
        }

        for scaffold_profile, expected_label in expected_labels.items():
            with self.subTest(scaffold_profile=scaffold_profile.value):
                variant = phase1_attention_only_variant(
                    scaffold_profile=scaffold_profile,
                    parcae_loop_count=3,
                )

                variant.validate()

                self.assertEqual(variant.scaffold_profile, scaffold_profile)
                self.assertEqual(variant.parcae_loop_count, 3)
                self.assertEqual(variant.label, expected_label)

    def test_looped_transformer_scaffold_rejects_hybrid_controls(self) -> None:
        cases = (
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.FIXED_LOOPED_LM,
                attention_profile=AttentionProfile.MODA_DEPTH_KV,
            ),
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
            ),
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
                feed_forward_profile=FeedForwardProfile.TINY_GLU_GATED,
            ),
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.FIXED_LOOPED_LM,
                layer_schedule=(
                    HybridAttentionLayerRole.EXACT_ATTENTION,
                    HybridAttentionLayerRole.SHARED_EXACT_ATTENTION,
                ),
                shape=Path1ModelShape(total_layers=2),
            ),
        )

        for variant in cases:
            with self.subTest(label=variant.label):
                with self.assertRaises(ValidationError):
                    variant.validate()

    def test_attention_only_universal_transformer_scaffolds_track_contract(
        self,
    ) -> None:
        fixed = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
            parcae_loop_count=3,
        )
        act = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
            parcae_loop_count=4,
            act_halting_threshold=0.95,
            act_ponder_loss_weight=0.02,
        )

        fixed.validate()
        act.validate()

        self.assertEqual(fixed.label, "attention-only-universal-transformer-loops3")
        self.assertEqual(act.label, "attention-only-universal-transformer-act-loops4")
        self.assertEqual(act.act_halting_threshold, 0.95)
        self.assertEqual(act.act_ponder_loss_weight, 0.02)

    def test_attention_only_ouro_learned_exit_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
            parcae_loop_count=3,
            ouro_entropy_weight=0.07,
            ouro_q_exit_threshold=0.75,
        )

        variant.validate()

        self.assertEqual(variant.label, "attention-only-ouro-learned-exit-loops3")
        self.assertEqual(variant.ouro_entropy_weight, 0.07)
        self.assertEqual(variant.ouro_q_exit_threshold, 0.75)

    def test_ouro_contract_rejects_invalid_controls(self) -> None:
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
                ouro_entropy_weight=-0.1,
            ).validate()
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
                ouro_q_exit_threshold=0.0,
            ).validate()

    def test_attention_only_rrt_cycle_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=4),
            scaffold_profile=Path1ScaffoldProfile.RRT_CYCLE,
            parcae_loop_count=2,
        )

        variant.validate()

        self.assertEqual(variant.label, "attention-only-rrt-cycle-loops2")
        self.assertEqual(variant.parcae_loop_count, 2)

    def test_rrt_cycle_requires_even_cycle_groups(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=3),
            scaffold_profile=Path1ScaffoldProfile.RRT_CYCLE,
            parcae_loop_count=2,
        )

        with self.assertRaises(ValidationError):
            variant.validate()

    def test_attention_only_mor_expert_choice_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(total_layers=3),
            scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
            parcae_loop_count=3,
            recurrent_token_route_fraction=0.5,
            mor_router_aux_loss_weight=0.02,
            mor_update_scale=0.2,
        )

        variant.validate()

        self.assertEqual(
            variant.label, "attention-only-mor-expert-choice-loops3-route50pct"
        )
        self.assertEqual(variant.mor_router_aux_loss_weight, 0.02)
        self.assertEqual(variant.mor_update_scale, 0.2)

    def test_mor_expert_choice_rejects_invalid_contracts(self) -> None:
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                shape=Path1ModelShape(total_layers=2),
                scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
            ).validate()
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                shape=Path1ModelShape(total_layers=3),
                scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
                mor_router_aux_loss_weight=-0.1,
            ).validate()
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                shape=Path1ModelShape(total_layers=3),
                scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
                mor_update_scale=0.0,
            ).validate()

    def test_act_contract_rejects_invalid_thresholds(self) -> None:
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
                act_halting_threshold=1.0,
            ).validate()
        with self.assertRaises(ValidationError):
            phase1_attention_only_variant(
                scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
                act_ponder_loss_weight=-0.1,
            ).validate()

    def test_attention_only_depth_attention_tracks_memory_contract(self) -> None:
        variant = phase1_attention_only_variant(
            attention_profile=AttentionProfile.MODA_DEPTH_KV,
            depth_memory_layers=3,
        )

        variant.validate()

        self.assertEqual(variant.attention_profile, AttentionProfile.MODA_DEPTH_KV)
        self.assertEqual(variant.depth_memory_layers, 3)
        self.assertEqual(variant.label, "attention-only-moda-depth-kv-depthmem3")

    def test_attention_only_paper_moda_tracks_memory_contract(self) -> None:
        variant = phase1_attention_only_variant(
            attention_profile=AttentionProfile.PAPER_MODA_DEPTH_KV,
            depth_memory_layers=3,
        )

        variant.validate()

        self.assertEqual(
            variant.attention_profile, AttentionProfile.PAPER_MODA_DEPTH_KV
        )
        self.assertEqual(variant.depth_memory_layers, 3)
        self.assertEqual(variant.label, "attention-only-paper-moda-depth-kv-depthmem3")

    def test_paper_moda_rejects_looped_scaffold_until_contract_exists(self) -> None:
        variant = phase1_attention_only_variant(
            attention_profile=AttentionProfile.PAPER_MODA_DEPTH_KV,
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
        )

        with self.assertRaises(ValidationError):
            variant.validate()

    def test_attention_only_recurrent_halting_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=4,
            recurrent_halting_profile=RecurrentHaltingProfile.ACCELERATION,
            recurrent_min_steps=2,
            recurrent_halting_threshold=0.02,
        )

        variant.validate()

        self.assertEqual(
            variant.recurrent_halting_profile, RecurrentHaltingProfile.ACCELERATION
        )
        self.assertEqual(variant.recurrent_min_steps, 2)
        self.assertIn("halt-acceleration-min2-t0p02", variant.label)

    def test_attention_only_vector_acceleration_halting_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=5,
            recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
            recurrent_min_steps=2,
            recurrent_halting_threshold=0.5,
        )

        variant.validate()

        self.assertEqual(
            variant.recurrent_halting_profile,
            RecurrentHaltingProfile.VECTOR_ACCELERATION,
        )
        self.assertIn("halt-vector-acceleration-min2-t0p5", variant.label)

    def test_attention_only_parcae_p20_value_scale_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=5,
            parcae_p20_value_scale=0.5,
        )

        variant.validate()

        self.assertEqual(variant.parcae_p20_value_scale, 0.5)
        self.assertIn("vscale0p5", variant.label)

    def test_adaptive_halting_requires_looped_scaffold(self) -> None:
        variant = phase1_attention_only_variant(
            recurrent_halting_profile=RecurrentHaltingProfile.NORMALIZED_STEP_NORM,
        )

        with self.assertRaises(ValidationError):
            variant.validate()

    def test_attention_only_token_routing_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
            token_route_fraction=0.25,
            token_routing_layer_indices=(2, 4),
        )

        variant.validate()

        self.assertEqual(
            variant.token_routing_profile, TokenRoutingProfile.CAUSAL_TOPK_BLOCK
        )
        self.assertEqual(variant.token_route_fraction, 0.25)
        self.assertEqual(variant.token_routing_layer_indices, (2, 4))
        self.assertIn("causal-topk-block-route25pct-layers2-4", variant.label)

    def test_attention_only_mod_train_topc_routing_tracks_contract(self) -> None:
        variant = phase1_attention_only_variant(
            token_routing_profile=TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK,
            token_route_fraction=0.5,
            token_routing_layer_indices=(2,),
        )

        variant.validate()

        self.assertEqual(
            variant.token_routing_profile, TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK
        )
        self.assertEqual(variant.token_route_fraction, 0.5)
        self.assertEqual(variant.token_routing_layer_indices, (2,))
        self.assertIn("mod-train-topc-block-route50pct-layers2", variant.label)

    def test_recurrent_token_routing_requires_looped_scaffold(self) -> None:
        variant = phase1_attention_only_variant(
            recurrent_token_routing_profile=(
                RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE
            ),
            recurrent_token_route_fraction=0.5,
        )

        with self.assertRaises(ValidationError):
            variant.validate()

    def test_recurrent_token_routing_tracks_loop_contract(self) -> None:
        variant = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=3,
            recurrent_token_routing_profile=(
                RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE
            ),
            recurrent_token_route_fraction=0.5,
        )

        variant.validate()

        self.assertEqual(
            variant.recurrent_token_routing_profile,
            RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE,
        )
        self.assertIn("causal-topk-state-route50pct", variant.label)

    def test_attention_only_parcae_bx_and_p20_control_scaffolds_validate(self) -> None:
        bx = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        p20 = phase1_attention_only_variant(
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        thin_p20 = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        thin_p20_gate = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        quarter_p20 = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        thin_p20_value = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        thin_p20_gate_baseblend = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        thin_p20_baseblend = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
        )
        mod_gate_bias = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
            token_route_fraction=0.5,
        )
        mod_value_scale = phase1_attention_only_variant(
            scaffold_profile=(
                Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION
            ),
            parcae_loop_count=2,
            token_route_fraction=0.5,
        )

        bx.validate()
        p20.validate()
        thin_p20.validate()
        thin_p20_gate.validate()
        quarter_p20.validate()
        thin_p20_value.validate()
        thin_p20_gate_baseblend.validate()
        thin_p20_baseblend.validate()
        mod_gate_bias.validate()
        mod_value_scale.validate()
        self.assertIn("mod-gate-bias", mod_gate_bias.label)
        self.assertIn("route50pct", mod_gate_bias.label)
        self.assertIn("mod-value-scale", mod_value_scale.label)
        self.assertIn("route50pct", mod_value_scale.label)

        self.assertEqual(bx.label, "attention-only-parcae-bx-looped-attention-loops2")
        self.assertEqual(
            p20.label, "attention-only-parcae-p20-control-looped-attention-loops2"
        )
        self.assertEqual(
            thin_p20.label,
            "attention-only-parcae-p20-thin-control-looped-attention-loops2",
        )
        self.assertEqual(
            thin_p20_gate.label,
            "attention-only-parcae-p20-thin-gate-control-looped-attention-loops2",
        )
        self.assertEqual(
            quarter_p20.label,
            "attention-only-parcae-p20-quarter-control-looped-attention-loops2",
        )
        self.assertEqual(
            thin_p20_value.label,
            "attention-only-parcae-p20-thin-value-control-looped-attention-loops2",
        )
        self.assertEqual(
            thin_p20_gate_baseblend.label,
            "attention-only-parcae-p20-thin-gate-baseblend-looped-attention-loops2",
        )
        self.assertEqual(
            thin_p20_baseblend.label,
            "attention-only-parcae-p20-thin-baseblend-control-looped-attention-loops2",
        )

    def test_gated_deltanet_reference_variant_tracks_profile(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH
        )

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertEqual(variant.label, "reference-ssm-hybrid-gated-deltanet-torch")

    def test_fla_gated_deltanet_reference_variant_tracks_runtime_profile(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA
        )

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
                self.assertEqual(
                    variant.reference_ssm_profile.fla_gdnp_compatible_law, law
                )
                self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
                self.assertEqual(variant.final_norm_kind, "rmsnorm")
                self.assertIn(profile.value, variant.label)

    def test_fla_gdnp_control_conditioned_reference_variant_tracks_runtime_profile(
        self,
    ) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY,
        )

        variant.validate()

        self.assertTrue(variant.reference_ssm_profile.is_gated_deltanet)
        self.assertTrue(variant.reference_ssm_profile.is_fla_gdnp_control_conditioned)
        self.assertTrue(variant.reference_ssm_profile.runtime_oriented)
        self.assertEqual(variant.final_norm_kind, "rmsnorm")
        self.assertIn("gated-deltanet-fla-p20-control-tiny", variant.label)

    def test_fla_gdn_control_shell_reference_variant_tracks_runtime_profile(
        self,
    ) -> None:
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

        with self.assertRaisesRegex(
            ValidationError, "reference_ssm_profile_schedule length"
        ):
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
            ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH: (
                "gdn",
                "p20",
                "mamba3",
            ),
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
        self.assertIs(
            PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
            SharedPrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )

    def test_recurrent_scan_runtime_optimization_profile_exposes_scan_targets(
        self,
    ) -> None:
        profile = runtime_optimization_profile("recurrent-scan-hybrid")

        self.assertIn(RuntimeOptimizationTarget.PACKED_PROJECTIONS, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.SEQUENCE_SCAN_KERNEL, profile.targets)
        self.assertIn(
            RuntimeOptimizationTarget.STRUCTURED_STATE_TRANSFORM, profile.targets
        )
        self.assertNotIn(RuntimeOptimizationTarget.ATTENTION_KERNEL, profile.targets)

    def test_pure_transformer_runtime_optimization_profile_exposes_transformer_targets(
        self,
    ) -> None:
        profile = runtime_optimization_profile("pure-transformer")

        self.assertIn(RuntimeOptimizationTarget.ATTENTION_KERNEL, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.KV_CACHE_LAYOUT, profile.targets)
        self.assertIn(RuntimeOptimizationTarget.MLP_FUSION, profile.targets)
        self.assertNotIn(
            RuntimeOptimizationTarget.SEQUENCE_SCAN_KERNEL, profile.targets
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
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(kind="one_shot", one_shot=OneShotRouterSpec()),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )
        surface.validate()
        self.assertEqual(
            surface.architecture.resolved_layout().moe_layers, tuple(range(8))
        )


if __name__ == "__main__":
    unittest.main()
