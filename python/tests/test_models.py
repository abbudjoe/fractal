from __future__ import annotations

import importlib.util
import unittest

import torch

from python.models.mini_moe import (
    OneShotTopKRouter,
    RoutePlan,
    SparseTopKDispatcher,
)
from python.models.mini_moe import MiniMoeBackboneModel
from python.models.common import (
    GatedEmlFeedForward,
    GenericTreeExpertFeedForward,
    PositionWiseFeedForward,
    RoutedEmlFeedForward,
    TinyGluExpertFeedForward,
    TinyMlpExpertFeedForward,
    leading_state_slice,
)
from python.models.primitives import build_sequence_primitive
from python.models.path1 import build_path1_model
from python.models.reference_ssm import GdnpFusedSequenceMixer, resolve_reference_ssm_config
from python.models.transformer import LocalCausalSelfAttention, local_causal_attention_bias
from python.runtime.recurrent import PackedLinearProjection
from python.specs.mini_moe import (
    MiniMoeArchitectureSpec,
    MiniMoeBackboneSpec,
    MiniMoeDispatchSpec,
    MiniMoeDispatchMode,
    MiniMoeLayerSchedule,
    MiniMoeLayerScheduleKind,
    MiniMoeObservabilitySpec,
    ResolvedDispatchContract,
    MiniMoeRouterSpec,
    MiniMoeRuntimeSpec,
    MiniMoeStackSpec,
    MiniMoeSurfaceSpec,
    OneShotRouterSpec,
)
from python.specs.path1 import (
    FeedForwardProfile,
    Path1ScaffoldProfile,
    PrimitiveExecutionProfile,
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveStateTransformMode,
    PrimitiveWrapperMode,
    Path1ModelShape,
    ReferenceSsmProfile,
    parse_layer_schedule_spec,
    phase1_attention_only_variant,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)


class Path1ModelTests(unittest.TestCase):
    def test_attention_only_forward_cpu(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))

    def test_attention_only_parcae_looped_scaffold_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=3,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("parcae_looped_attention", diagnostics)
        self.assertEqual(diagnostics["parcae_looped_attention"]["loop_count"], 3)
        self.assertEqual(len(diagnostics["parcae_looped_attention"]["last_recurrent_state_norms"]), 3)

    def test_attention_only_parcae_bx_and_p20_control_scaffolds_forward_cpu(self) -> None:
        for scaffold_profile in (
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
        ):
            with self.subTest(scaffold_profile=scaffold_profile.value):
                variant = phase1_attention_only_variant(
                    shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
                    scaffold_profile=scaffold_profile,
                    parcae_loop_count=2,
                )
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

                logits = model.forward_logits(input_ids)
                diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(diagnostics["profile"], scaffold_profile.value)
                self.assertIsNotNone(diagnostics["last_injection_gate_mean"])
                self.assertIsNotNone(diagnostics["last_injection_norm"])

    def test_attention_only_parcae_p20_control_receives_runtime_policy(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        self.assertIsNotNone(model.parcae_p20_controller)
        self.assertIsNone(model.parcae_p20_controller._compiled_scan_impl)

        model.configure_runtime_policy(
            compile_mode="reduce-overhead",
            primitive_runtime_backend="torch",
        )

        self.assertIsNotNone(model.parcae_p20_controller._compiled_scan_impl)

    def test_attention_only_eml_tree_feed_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=2, ffn_multiplier=2),
            feed_forward_profile=FeedForwardProfile.EML_TREE,
            eml_slot_count=6,
            eml_tree_depth=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("eml_tree", model.model_label)
        self.assertEqual(diagnostics["feed_forward_profile"], "eml-tree")
        self.assertEqual(diagnostics["eml_inspired_feed_forward"]["leaf_count"], 4)

    def test_attention_only_gated_eml_feed_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=2, ffn_multiplier=2),
            feed_forward_profile=FeedForwardProfile.MLP_EML_GATED,
            eml_slot_count=6,
            eml_tree_depth=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIn("mlp_eml_gated", model.model_label)

    def test_attention_only_surgical_gated_eml_targets_selected_layer_only(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=4, ffn_multiplier=2),
            feed_forward_profile=FeedForwardProfile.MLP_EML_GATED,
            feed_forward_layer_indices=(2,),
            eml_slot_count=4,
            eml_tree_depth=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.blocks[0].ffn, PositionWiseFeedForward)
        self.assertIsInstance(model.blocks[1].ffn, PositionWiseFeedForward)
        self.assertIsInstance(model.blocks[2].ffn, GatedEmlFeedForward)
        self.assertIsInstance(model.blocks[3].ffn, PositionWiseFeedForward)
        self.assertEqual(diagnostics["eml_inspired_feed_forward"]["layer_indices"], (2,))

    def test_attention_only_routed_eml_targets_selected_layer_only(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=4, ffn_multiplier=2),
            feed_forward_profile=FeedForwardProfile.MLP_EML_ROUTED,
            feed_forward_layer_indices=(2,),
            eml_slot_count=4,
            eml_tree_depth=2,
            eml_route_fraction=0.25,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.blocks[0].ffn, PositionWiseFeedForward)
        self.assertIsInstance(model.blocks[1].ffn, PositionWiseFeedForward)
        self.assertIsInstance(model.blocks[2].ffn, RoutedEmlFeedForward)
        self.assertIsInstance(model.blocks[3].ffn, PositionWiseFeedForward)
        self.assertEqual(diagnostics["eml_inspired_feed_forward"]["layer_indices"], (2,))
        self.assertEqual(diagnostics["eml_inspired_feed_forward"]["route_fraction"], 0.25)

    def test_attention_only_expert_controls_target_selected_layer_only(self) -> None:
        cases = (
            (FeedForwardProfile.TINY_MLP_GATED, TinyMlpExpertFeedForward),
            (FeedForwardProfile.TINY_GLU_GATED, TinyGluExpertFeedForward),
            (FeedForwardProfile.GENERIC_TREE_GATED, GenericTreeExpertFeedForward),
        )
        for profile, expected_type in cases:
            with self.subTest(profile=profile.value):
                variant = phase1_attention_only_variant(
                    shape=Path1ModelShape(d_model=32, head_count=4, total_layers=4, ffn_multiplier=2),
                    feed_forward_profile=profile,
                    feed_forward_layer_indices=(2,),
                    eml_slot_count=4,
                    eml_tree_depth=2,
                )
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

                logits = model.forward_logits(input_ids)
                diagnostics = model.diagnostic_payload()

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsInstance(model.blocks[0].ffn, PositionWiseFeedForward)
                self.assertIsInstance(model.blocks[2].ffn, expected_type)
                self.assertEqual(diagnostics["feed_forward_experts"][0]["layer_index"], 2)

    def test_primitive_forward_cpu(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P23,
            execution_profile=PrimitiveExecutionProfile.REFERENCE,
            residual_mode=PrimitiveResidualMode.GATED,
            readout_mode=PrimitiveReadoutMode.PROJECTED_NORM,
            norm_mode=PrimitiveNormMode.RESIDUAL_RENORM,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("gated", model.model_label)
        self.assertIn("projected_norm", model.model_label)

    def test_p1_fractal_hybrid_forward_cpu(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P1_FRACTAL_HYBRID,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.PLAIN,
            readout_mode=PrimitiveReadoutMode.DIRECT,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.STANDARD,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("p1_fractal_hybrid", model.model_label)

    def test_p20_gdn_role_forward_cpu(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P20_GDN_ROLE,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.SCALED,
            readout_mode=PrimitiveReadoutMode.DIRECT,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.MAMBA_RMS,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("p2_0_gdn_role", model.model_label)

    def test_gated_deltanet_reference_forward_cpu(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("gated_deltanet_torch", model.model_label)

    def test_gated_deltanet_reference_is_causal(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 6), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(torch.allclose(logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]))

    def test_gated_deltanet_pr_topology_reuses_shared_swa_attention(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
        )

        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIs(model.blocks[5].attention, model.blocks[11].attention)
        self.assertIsNot(model.blocks[5], model.blocks[11])
        self.assertIn("schedule_rrrrrarrrrrs", model.model_label)

    def test_pr5_scaffold_gdn_topology_forward_cpu(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 6), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 6, 257))
        self.assertIs(model.blocks[5].attention, model.blocks[11].attention)
        self.assertEqual(diagnostics["scaffold"]["profile"], "pr5-hybrid-gdn")
        self.assertTrue(diagnostics["scaffold"]["hash_context_embedding"])
        self.assertTrue(diagnostics["scaffold"]["smear_gate"])
        self.assertTrue(diagnostics["reference_ssm_blocks"][0]["pr5_scaffold"])

    def test_pr5_scaffold_sparse_p20_insertion_uses_only_selected_recurrent_slot(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
            if ordinal == 4
            else ReferenceSsmProfile.GATED_DELTANET_TORCH
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (1, 5, 257))
        mixer_kinds = [block["mixer"]["kind"] for block in diagnostics["reference_ssm_blocks"]]
        self.assertEqual(mixer_kinds.count("gdnp-fused"), 1)
        self.assertEqual(mixer_kinds.count("parallel-composite"), 0)
        self.assertEqual(diagnostics["reference_ssm_blocks"][4]["profile"], "gated-deltanet-p20-fused-multi-read-torch")

    def test_pr5_scaffold_tiny_p20_conditioner_keeps_gdn_readout_owner(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY
            if ordinal == 4
            else ReferenceSsmProfile.GATED_DELTANET_TORCH
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (1, 5, 257))
        mixer = diagnostics["reference_ssm_blocks"][4]["mixer"]
        self.assertEqual(mixer["kind"], "fla-gdnp-control-conditioned")
        self.assertEqual(mixer["readout"], "gdn-only")
        self.assertEqual(mixer["conditioned_controls"], ("q", "k", "v", "beta"))
        self.assertEqual(mixer["conditioner"]["kind"], "p20-tiny-control-conditioner")
        self.assertLess(mixer["conditioner"]["bottleneck_width"], 32)

    def test_pr5_scaffold_control_shell_has_no_p20_conditioner(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL
            if ordinal == 4
            else ReferenceSsmProfile.GATED_DELTANET_TORCH
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (1, 5, 257))
        mixer = diagnostics["reference_ssm_blocks"][4]["mixer"]
        self.assertEqual(mixer["kind"], "fla-gdn-control-shell")
        self.assertEqual(mixer["readout"], "gdn-only")
        self.assertEqual(mixer["conditioned_controls"], ())
        self.assertNotIn("conditioner", mixer)

    def test_pr5_scaffold_sparse_p20_insertion_is_causal(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
            if ordinal == 4
            else ReferenceSsmProfile.GATED_DELTANET_TORCH
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(torch.allclose(logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]))

    def test_pr5_scaffold_optimizer_groups_are_disjoint(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
            if ordinal == 4
            else ReferenceSsmProfile.GATED_DELTANET_TORCH
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=len(schedule), local_window=4),
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            layer_schedule=schedule,
            profile_schedule=profile_schedule,
            scaffold_profile=Path1ScaffoldProfile.PR5_HYBRID_GDN,
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        groups = model.optimizer_parameter_groups(1.0e-3)
        group_names = {group["name"] for group in groups}
        self.assertIn("pr5_context", group_names)
        self.assertIn("pr5_recurrent", group_names)
        self.assertIn("pr5_gates_controls", group_names)
        self.assertIn("pr5_readout", group_names)
        self.assertIn("pr5_scalars", group_names)
        grouped_params = [param for group in groups for param in group["params"]]
        self.assertEqual(len({id(param) for param in grouped_params}), len(grouped_params))
        self.assertEqual(
            {id(param) for param in model.parameters() if param.requires_grad},
            {id(param) for param in grouped_params},
        )
        lr_by_name = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(lr_by_name["pr5_recurrent"], 5.0e-4)
        self.assertEqual(lr_by_name["pr5_gates_controls"], 5.0e-4)

    def test_gated_deltanet_p20_composite_forward_cpu(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_P20_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        first_composite = model.blocks[1].mixer
        self.assertEqual(first_composite.branch_names, ("gdn", "p20"))
        self.assertIn("gated_deltanet_p20_torch", model.model_label)
        diagnostics = model.diagnostic_payload()
        summary = diagnostics["composite_branch_weight_summary"]
        self.assertEqual(summary["gdn"]["layer_count"], 4)
        self.assertAlmostEqual(summary["gdn"]["mean_weight_across_layers"], 0.5)
        self.assertAlmostEqual(summary["p20"]["mean_weight_across_layers"], 0.5)

    def test_gated_deltanet_thin_p20_composite_forward_cpu(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        first_composite = model.blocks[1].mixer
        self.assertEqual(first_composite.branch_names, ("gdn", "p20_thin"))
        p20_branch = first_composite.branches["p20_thin"]
        self.assertEqual(p20_branch.bottleneck_width, 64)
        diagnostics = model.diagnostic_payload()
        first_block = diagnostics["reference_ssm_blocks"][0]
        thin_branch = first_block["mixer"]["branches"][1]
        self.assertEqual(thin_branch["module"]["bottleneck_width"], 64)

    def test_p20_reference_ssm_scan_forward_cpu(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.P20_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("p20_torch", model.model_label)

    def test_gdnp_fused_reference_forward_cpu(self) -> None:
        profiles = (
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_BETA_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_QKV_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_RESIDUAL_READOUT_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_ALL_TORCH,
        )
        for profile in profiles:
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

                logits = model.forward_logits(input_ids)

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertIn(profile.value.replace("-", "_"), model.model_label)
                diagnostics = model.diagnostic_payload()
                first_block = diagnostics["reference_ssm_blocks"][0]
                self.assertEqual(first_block["mixer"]["kind"], "gdnp-fused")
                self.assertEqual(first_block["mixer"]["law"], profile.gdnp_fused_law)
                self.assertEqual(first_block["mixer"]["vector_state_width"], 128)

    def test_gdnp_fused_reference_is_causal(self) -> None:
        variant = phase1_reference_ssm_variant(profile=ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH)
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 6), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(torch.allclose(logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]))

    def test_gdnp_fused_triton_policy_routes_vector_scan_to_sequence_kernel(self) -> None:
        config = resolve_reference_ssm_config(
            d_model=32,
            head_count=4,
            profile=ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH,
            dtype_mode="fp32",
        )
        mixer = GdnpFusedSequenceMixer(config)

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0
                self.matrix_calls = 0

            def scan_rotary_state_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                return (
                    torch.full_like(update_gate, 0.25),
                    torch.full_like(initial_state, 0.5),
                )

            def scan_gdnp_matrix_multi_read(
                self,
                *,
                queries: torch.Tensor,
                keys: torch.Tensor,
                value_bases: torch.Tensor,
                vector_states: torch.Tensor,
                alpha_gates: torch.Tensor,
                beta_gates: torch.Tensor,
                aux_query_state_scale: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.matrix_calls += 1
                self.last_matrix_shapes = (
                    tuple(queries.shape),
                    tuple(keys.shape),
                    tuple(value_bases.shape),
                    tuple(vector_states.shape),
                    tuple(alpha_gates.shape),
                    tuple(beta_gates.shape),
                    tuple(aux_query_state_scale.shape),
                )
                return (
                    torch.full_like(queries, 0.75),
                    torch.full_like(queries, 0.125),
                )

        fake_backend = FakeBackend()
        mixer._primitive_runtime_backend = "triton"
        mixer._triton_backend = fake_backend

        outputs = mixer(torch.randn(2, 5, 32))

        self.assertEqual(tuple(outputs.shape), (2, 5, 32))
        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(fake_backend.matrix_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 32),
                (2, 5, 32),
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 32),
                (2, 32),
                (2, 16, 16),
                (32,),
            ),
        )
        self.assertEqual(
            fake_backend.last_matrix_shapes,
            (
                (2, 5, 4, 8),
                (2, 5, 4, 8),
                (2, 5, 4, 8),
                (2, 5, 4, 8),
                (2, 5, 4),
                (2, 5, 4),
                (4, 8),
            ),
        )
        self.assertEqual(mixer.diagnostic_payload()["primitive_runtime_backend"], "triton")
        self.assertTrue(mixer.diagnostic_payload()["triton_matrix_scan"])

    def test_fla_gdnp_compatible_profile_exposes_dependency_boundary(self) -> None:
        has_fla = importlib.util.find_spec("fla") is not None
        profiles = (
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_COMPAT,
            ReferenceSsmProfile.GATED_DELTANET_FLA_P20_MULTI_READ,
        )

        for profile in profiles:
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)
                if has_fla:
                    model = build_path1_model(variant, dtype_mode="fp32")
                    input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
                    logits = model.forward_logits(input_ids)
                    self.assertEqual(tuple(logits.shape), (2, 8, 257))
                    diagnostics = model.diagnostic_payload()
                    first_block = diagnostics["reference_ssm_blocks"][0]
                    self.assertEqual(first_block["mixer"]["kind"], "fla-gdnp-compatible")
                    self.assertEqual(first_block["mixer"]["law"], profile.fla_gdnp_compatible_law)
                else:
                    with self.assertRaisesRegex(RuntimeError, "FLA gated-delta-rule import failed"):
                        build_path1_model(variant, dtype_mode="fp32")

    def test_full_window_attention_matches_explicit_causal_mask(self) -> None:
        attention = LocalCausalSelfAttention(d_model=16, head_count=4)
        hidden = torch.randn(2, 6, 16)
        explicit_mask = local_causal_attention_bias(
            seq_len=hidden.shape[1],
            local_window=hidden.shape[1],
            device=hidden.device,
            dtype=hidden.dtype,
        )

        implicit = attention(hidden, None)
        explicit = attention(hidden, explicit_mask)

        self.assertTrue(torch.allclose(implicit, explicit, atol=1.0e-5, rtol=1.0e-5))

    def test_mamba_composite_profiles_keep_explicit_dependency_boundary(self) -> None:
        has_official_mamba = importlib.util.find_spec("mamba_ssm") is not None
        mamba_profiles = (
            ReferenceSsmProfile.GATED_DELTANET_MAMBA3_TORCH,
            ReferenceSsmProfile.GATED_DELTANET_P20_MAMBA3_TORCH,
            ReferenceSsmProfile.P20_MAMBA3_TORCH,
        )
        for profile in mamba_profiles:
            with self.subTest(profile=profile.value):
                variant = phase1_reference_ssm_variant(profile=profile)
                if has_official_mamba:
                    model = build_path1_model(variant, dtype_mode="fp32")
                    input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
                    logits = model.forward_logits(input_ids)
                    self.assertEqual(tuple(logits.shape), (2, 8, 257))
                else:
                    with self.assertRaisesRegex(RuntimeError, "official PyTorch Mamba3 import failed"):
                        build_path1_model(variant, dtype_mode="fp32")

    def test_legacy_primitive_ports_scan_cpu(self) -> None:
        inputs = torch.randn(2, 5, 16)
        legacy_profiles = (
            PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE,
            PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE,
            PrimitiveProfile.P2_MANDELBROT,
            PrimitiveProfile.P3_HIERARCHICAL,
            PrimitiveProfile.B1_FRACTAL_GATED,
            PrimitiveProfile.B2_STABLE_HIERARCHICAL,
            PrimitiveProfile.B3_FRACTAL_HIERARCHICAL,
            PrimitiveProfile.B4_UNIVERSAL,
            PrimitiveProfile.IFS,
            PrimitiveProfile.GENERALIZED_MOBIUS,
            PrimitiveProfile.LOGISTIC_CHAOTIC_MAP,
            PrimitiveProfile.JULIA_RECURSIVE_ESCAPE,
            PrimitiveProfile.MANDELBOX_RECURSIVE,
        )
        for primitive_profile in legacy_profiles:
            with self.subTest(primitive_profile=primitive_profile.value):
                primitive = build_sequence_primitive(
                    primitive_profile,
                    16,
                    PrimitiveExecutionProfile.RUNTIME,
                )
                result = primitive.scan(inputs)
                self.assertEqual(tuple(result.emitted_outputs.shape), (2, 5, 16))
                self.assertEqual(result.final_state.shape[0], 2)

    def test_runtime_primitive_matches_reference_math_for_p20(self) -> None:
        inputs = torch.randn(2, 5, 16)
        for primitive_profile in PrimitiveProfile:
            with self.subTest(primitive_profile=primitive_profile.value):
                reference = build_sequence_primitive(
                    primitive_profile,
                    16,
                    PrimitiveExecutionProfile.REFERENCE,
                )
                runtime = build_sequence_primitive(
                    primitive_profile,
                    16,
                    PrimitiveExecutionProfile.RUNTIME,
                )
                runtime.load_state_dict(reference.state_dict())
                reference_result = reference.scan(inputs)
                runtime_plan = runtime.prepare_runtime_plan(inputs)
                runtime_result = runtime.scan_with_runtime_plan(
                    runtime_plan,
                    batch_size=inputs.shape[0],
                    device=inputs.device,
                    dtype=inputs.dtype,
                    seq_len=inputs.shape[1],
                )
                runtime_scan_result = runtime.scan(inputs)
                self.assertTrue(
                    torch.allclose(
                        reference_result.emitted_outputs,
                        runtime_result.emitted_outputs,
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        reference_result.final_state,
                        runtime_result.final_state,
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        runtime_result.emitted_outputs,
                        runtime_scan_result.emitted_outputs,
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        runtime_result.final_state,
                        runtime_scan_result.final_state,
                        atol=1.0e-5,
                        rtol=1.0e-5,
                    )
                )

    def test_compiled_runtime_p20_matches_uncompiled_runtime(self) -> None:
        inputs = torch.randn(2, 5, 16)
        baseline = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
        )
        compiled = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
        )
        compiled.load_state_dict(baseline.state_dict())
        compiled.configure_runtime_policy(compile_mode="reduce-overhead")

        baseline_result = baseline.scan(inputs)
        compiled_result = compiled.scan(inputs)

        self.assertTrue(
            torch.allclose(
                baseline_result.emitted_outputs,
                compiled_result.emitted_outputs,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                baseline_result.final_state,
                compiled_result.final_state,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )

    def test_runtime_p20_block_diagonal_matches_reference(self) -> None:
        inputs = torch.randn(2, 5, 16)
        reference = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.REFERENCE,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        runtime.load_state_dict(reference.state_dict())

        reference_result = reference.scan(inputs)
        runtime_result = runtime.scan(inputs)

        self.assertTrue(
            torch.allclose(
                reference_result.emitted_outputs,
                runtime_result.emitted_outputs,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                reference_result.final_state,
                runtime_result.final_state,
                atol=1.0e-5,
                rtol=1.0e-5,
            )
        )

    def test_runtime_p20_triton_backend_boundary_is_explicit(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
        )
        has_triton = importlib.util.find_spec("triton") is not None
        if has_triton:
            runtime.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="triton",
            )
            return

        with self.assertRaisesRegex(
            RuntimeError,
            "primitive_runtime_backend=triton requires the primitive-triton CUDA env",
        ):
            runtime.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="triton",
            )

    def test_p1_fractal_hybrid_rejects_non_dense_state_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support non-dense state transforms"):
            build_sequence_primitive(
                PrimitiveProfile.P1_FRACTAL_HYBRID,
                16,
                PrimitiveExecutionProfile.RUNTIME,
                state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
            )

    def test_legacy_ports_reject_non_dense_state_transform(self) -> None:
        legacy_profiles = (
            PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE,
            PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE,
            PrimitiveProfile.P2_MANDELBROT,
            PrimitiveProfile.P3_HIERARCHICAL,
            PrimitiveProfile.B1_FRACTAL_GATED,
            PrimitiveProfile.B2_STABLE_HIERARCHICAL,
            PrimitiveProfile.B3_FRACTAL_HIERARCHICAL,
            PrimitiveProfile.B4_UNIVERSAL,
            PrimitiveProfile.IFS,
            PrimitiveProfile.GENERALIZED_MOBIUS,
            PrimitiveProfile.LOGISTIC_CHAOTIC_MAP,
            PrimitiveProfile.JULIA_RECURSIVE_ESCAPE,
            PrimitiveProfile.MANDELBOX_RECURSIVE,
        )
        for primitive_profile in legacy_profiles:
            with self.subTest(primitive_profile=primitive_profile.value):
                with self.assertRaisesRegex(ValueError, "does not support non-dense state transforms"):
                    build_sequence_primitive(
                        primitive_profile,
                        16,
                        PrimitiveExecutionProfile.RUNTIME,
                        state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                    )

    def test_p20_gdn_role_rejects_non_dense_state_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support non-dense state transforms"):
            build_sequence_primitive(
                PrimitiveProfile.P20_GDN_ROLE,
                16,
                PrimitiveExecutionProfile.RUNTIME,
                state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
            )

    def test_primitives_use_shared_packed_input_projection_surface(self) -> None:
        expected_split_sizes = {
            PrimitiveProfile.P1: (16, 16),
            PrimitiveProfile.P1_FRACTAL_HYBRID: (16, 16),
            PrimitiveProfile.P1_FRACTAL_HYBRID_COMPOSITE: (16, 16),
            PrimitiveProfile.P1_FRACTAL_HYBRID_DYN_GATE: (16, 16),
            PrimitiveProfile.P20: (16, 8, 16, 16),
            PrimitiveProfile.P20_GDN_ROLE: (16, 16, 16),
            PrimitiveProfile.P2: (16, 8, 16, 16),
            PrimitiveProfile.P23: (16, 16, 8, 16, 16),
            PrimitiveProfile.P21: (32, 16, 32, 16),
            PrimitiveProfile.P22: (32, 16, 32, 16),
            PrimitiveProfile.P2_MANDELBROT: (32, 32),
            PrimitiveProfile.P3_HIERARCHICAL: (16, 16, 16, 16),
            PrimitiveProfile.B1_FRACTAL_GATED: (32, 32),
            PrimitiveProfile.B2_STABLE_HIERARCHICAL: (16, 16, 16),
            PrimitiveProfile.B3_FRACTAL_HIERARCHICAL: (32, 32),
            PrimitiveProfile.B4_UNIVERSAL: (32, 32, 32),
            PrimitiveProfile.IFS: (4,),
            PrimitiveProfile.GENERALIZED_MOBIUS: (16, 16, 16, 16),
            PrimitiveProfile.LOGISTIC_CHAOTIC_MAP: (16, 16),
            PrimitiveProfile.JULIA_RECURSIVE_ESCAPE: (32,),
            PrimitiveProfile.MANDELBOX_RECURSIVE: (16,),
        }

        for primitive_profile, split_sizes in expected_split_sizes.items():
            with self.subTest(primitive_profile=primitive_profile.value):
                primitive = build_sequence_primitive(
                    primitive_profile,
                    16,
                    PrimitiveExecutionProfile.RUNTIME,
                )

                self.assertIsInstance(primitive.in_projection, PackedLinearProjection)
                self.assertEqual(primitive.in_projection.split_sizes, split_sizes)

    def test_p20_gdn_role_causal_prefix_invariance(self) -> None:
        primitive = build_sequence_primitive(
            PrimitiveProfile.P20_GDN_ROLE,
            16,
            PrimitiveExecutionProfile.RUNTIME,
        )
        prefix = torch.randn(2, 6, 16)
        suffix = torch.randn(2, 3, 16)
        prefix_outputs = primitive.scan(prefix).emitted_outputs
        extended_outputs = primitive.scan(torch.cat([prefix, suffix], dim=1)).emitted_outputs[:, :6, :]
        self.assertTrue(torch.allclose(prefix_outputs, extended_outputs, atol=1.0e-5, rtol=1.0e-5))

    def test_p20_gdn_role_optimizer_groups_are_disjoint(self) -> None:
        variant = phase1_primitive_variant(
            primitive_profile=PrimitiveProfile.P20_GDN_ROLE,
            execution_profile=PrimitiveExecutionProfile.RUNTIME,
            residual_mode=PrimitiveResidualMode.SCALED,
            readout_mode=PrimitiveReadoutMode.DIRECT,
            norm_mode=PrimitiveNormMode.PRE_NORM_ONLY,
            wrapper_mode=PrimitiveWrapperMode.MAMBA_RMS,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        groups = model.optimizer_parameter_groups(1.0e-3)
        group_names = {group["name"] for group in groups}
        self.assertIn("p20_gdn_recurrent", group_names)
        self.assertIn("p20_gdn_gates", group_names)
        self.assertIn("p20_gdn_readout", group_names)
        self.assertIn("p20_gdn_scalars", group_names)
        grouped_params = [param for group in groups for param in group["params"]]
        self.assertEqual(len({id(param) for param in grouped_params}), len(grouped_params))
        self.assertEqual(
            {id(param) for param in model.parameters() if param.requires_grad},
            {id(param) for param in grouped_params},
        )

    def test_runtime_p20_block_diagonal_triton_routes_to_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_p20_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                output_gate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(output_gate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                return (
                    torch.full_like(update_gate, 3.0),
                    torch.full_like(initial_state, 4.0),
                )

            def fused_p20_update_readout(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError("block-diagonal Triton runtime should not fall back to the step kernel")

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 8),
                (2, 5, 8),
                (2, 5, 16),
                (2, 5, 16),
                (2, 16),
                (4, 4, 4),
                (16,),
            ),
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, torch.full_like(result.emitted_outputs, 3.0)))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0)))

    def test_runtime_p20_block_diagonal_2_triton_routes_to_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_p20_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                output_gate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(output_gate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                return (
                    torch.full_like(update_gate, 7.0),
                    torch.full_like(initial_state, 8.0),
                )

            def scan_p20_dense_sequence(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError("block-diagonal Triton runtime should not route to the dense sequence kernel")

            def fused_p20_update_readout(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError("block-diagonal Triton runtime should not fall back to the step kernel")

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 8),
                (2, 5, 8),
                (2, 5, 16),
                (2, 5, 16),
                (2, 16),
                (2, 8, 8),
                (16,),
            ),
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, torch.full_like(result.emitted_outputs, 7.0)))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 8.0)))

    def test_runtime_p20_dense_triton_routes_to_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.DENSE,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_p20_dense_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                output_gate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(output_gate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                return (
                    torch.full_like(update_gate, 5.0),
                    torch.full_like(initial_state, 6.0),
                )

            def scan_p20_block_diagonal_sequence(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError("dense Triton runtime should not route to the block-diagonal sequence kernel")

            def fused_p20_update_readout(self, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError("dense Triton runtime should not fall back to the step kernel")

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 8),
                (2, 5, 8),
                (2, 5, 16),
                (2, 5, 16),
                (2, 16),
                (16, 16),
                (16,),
            ),
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, torch.full_like(result.emitted_outputs, 5.0)))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 6.0)))

    def test_runtime_p2_block_diagonal_2_triton_routes_to_state_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P2,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_rotary_state_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                state_outputs = torch.full_like(update_gate, 2.0)
                final_state = torch.full_like(initial_state, 4.0)
                return state_outputs, final_state

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 8),
                (2, 5, 8),
                (2, 5, 16),
                (2, 16),
                (2, 8, 8),
                (16,),
            ),
        )
        expected_outputs = runtime_plan.output_gates * runtime.output_projection(torch.full_like(runtime_plan.update_gates, 2.0))
        self.assertTrue(torch.allclose(result.emitted_outputs, expected_outputs))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0)))

    def test_runtime_p21_block_diagonal_2_triton_routes_to_state_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P21,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_rotary_state_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                state_outputs = torch.full_like(update_gate, 2.0)
                final_state = torch.full_like(initial_state, 4.0)
                return state_outputs, final_state

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 32),
                (2, 5, 32),
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 32),
                (2, 32),
                (2, 16, 16),
                (32,),
            ),
        )
        expected_outputs = runtime_plan.output_gates * leading_state_slice(
            torch.full_like(runtime_plan.update_gates, 2.0),
            runtime.d_model,
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, expected_outputs))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0)))

    def test_runtime_p22_block_diagonal_2_triton_routes_to_state_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P22,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
        )

        class FakeBackend:
            def __init__(self) -> None:
                self.sequence_calls = 0

            def scan_rotary_state_block_diagonal_sequence(
                self,
                *,
                update_gate: torch.Tensor,
                retain_gate: torch.Tensor,
                angle_cos: torch.Tensor,
                angle_sin: torch.Tensor,
                candidate: torch.Tensor,
                initial_state: torch.Tensor,
                transform_weight: torch.Tensor,
                transform_bias: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.last_shapes = (
                    tuple(update_gate.shape),
                    tuple(retain_gate.shape),
                    tuple(angle_cos.shape),
                    tuple(angle_sin.shape),
                    tuple(candidate.shape),
                    tuple(initial_state.shape),
                    tuple(transform_weight.shape),
                    tuple(transform_bias.shape),
                )
                state_outputs = torch.full_like(update_gate, 3.0)
                final_state = torch.full_like(initial_state, 5.0)
                return state_outputs, final_state

        fake_backend = FakeBackend()
        runtime._primitive_runtime_backend = "triton"
        runtime._triton_backend = fake_backend

        inputs = torch.randn(2, 5, 16)
        runtime_plan = runtime.prepare_runtime_plan(inputs)
        result = runtime.scan_with_runtime_plan(
            runtime_plan,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            seq_len=inputs.shape[1],
        )

        self.assertEqual(fake_backend.sequence_calls, 1)
        self.assertEqual(
            fake_backend.last_shapes,
            (
                (2, 5, 32),
                (2, 5, 32),
                (2, 5, 16),
                (2, 5, 16),
                (2, 5, 32),
                (2, 32),
                (2, 16, 16),
                (32,),
            ),
        )
        expected_outputs = runtime_plan.output_gates * runtime.output_projection(
            torch.full_like(runtime_plan.update_gates, 3.0)
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, expected_outputs))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 5.0)))

    def test_reference_ssm_boundary_is_explicit(self) -> None:
        has_official_mamba = importlib.util.find_spec("mamba_ssm") is not None
        if has_official_mamba:
            model = build_path1_model(phase1_reference_ssm_variant(), dtype_mode="fp32")
            input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
            logits = model.forward_logits(input_ids)
            self.assertEqual(tuple(logits.shape), (2, 8, 257))
            return

        with self.assertRaisesRegex(
            RuntimeError,
            "official PyTorch Mamba3 import failed",
        ):
            build_path1_model(phase1_reference_ssm_variant(), dtype_mode="fp32")

    def test_reference_ssm_profiles_resolve_distinct_execution_configs(self) -> None:
        reference = resolve_reference_ssm_config(
            d_model=128,
            head_count=4,
            profile=ReferenceSsmProfile.MAMBA3_SISO_REFERENCE,
            dtype_mode="fp32",
        )
        runtime = resolve_reference_ssm_config(
            d_model=128,
            head_count=4,
            profile=ReferenceSsmProfile.MAMBA3_SISO_RUNTIME,
            dtype_mode="fp32",
        )
        self.assertEqual(reference.chunk_size, 1)
        self.assertEqual(runtime.chunk_size, 8)
        self.assertFalse(reference.runtime_oriented)
        self.assertTrue(runtime.runtime_oriented)

    def test_gated_deltanet_reference_profile_is_internal_torch_baseline(self) -> None:
        config = resolve_reference_ssm_config(
            d_model=128,
            head_count=4,
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH,
            dtype_mode="fp32",
        )

        self.assertFalse(config.runtime_oriented)
        self.assertTrue(config.profile.is_gated_deltanet)


class MiniMoeModelTests(unittest.TestCase):
    def test_router_emits_dense_routing_intent(self) -> None:
        router = OneShotTopKRouter(hidden_dim=16, experts_per_block=4)
        hidden = torch.randn(2, 3, 16)
        plan = router.plan(hidden)
        self.assertIsInstance(plan, RoutePlan)
        self.assertEqual(tuple(plan.expert_logits.shape), (2, 3, 4))

    def test_dispatcher_owns_topk_selection(self) -> None:
        dispatcher = SparseTopKDispatcher(
            ResolvedDispatchContract(
                mode=MiniMoeDispatchMode.SPARSE_TOP_K,
                active_experts_per_token=2,
            )
        )
        experts = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
        hidden = torch.randn(2, 3, 8)
        plan = RoutePlan(
            expert_logits=torch.tensor(
                [
                    [[4.0, 1.0, 0.5, -1.0], [0.0, 2.0, 1.0, -2.0], [3.0, 2.0, 1.0, 0.0]],
                    [[1.0, 0.5, 4.0, 3.5], [2.0, 1.0, 0.0, -1.0], [0.2, 0.1, 0.0, -0.1]],
                ]
            )
        )
        mixed, observation = dispatcher.dispatch(hidden, plan, experts)
        self.assertEqual(tuple(mixed.shape), tuple(hidden.shape))
        self.assertEqual(observation.active_expert_count, 2)

    def test_mini_moe_backbone_dense_debug_forward_cpu(self) -> None:
        surface = MiniMoeSurfaceSpec(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=None,
                label="phase1-mini-moe-reference",
                backbone=MiniMoeBackboneSpec(
                    vocab_size=257,
                    hidden_dim=64,
                    head_count=4,
                    total_layers=4,
                    local_window=32,
                    ffn_multiplier=4,
                ),
                moe=MiniMoeStackSpec(
                    experts_per_block=4,
                    active_experts_per_token=1,
                    moe_layer_schedule=MiniMoeLayerSchedule(kind=MiniMoeLayerScheduleKind.EXPLICIT, explicit_layers=(0, 2)),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(kind="one_shot", one_shot=OneShotRouterSpec()),
            ),
            runtime=MiniMoeRuntimeSpec(dispatch=MiniMoeDispatchSpec()),
            observability=MiniMoeObservabilitySpec(),
        )
        model = MiniMoeBackboneModel(surface)
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))


if __name__ == "__main__":
    unittest.main()
