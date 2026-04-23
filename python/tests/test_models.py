from __future__ import annotations

import importlib.util
import unittest

import torch
import torch.nn.functional as F

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
from python.models.reference_ssm import (
    GdnpFusedSequenceMixer,
    resolve_reference_ssm_config,
)
from python.models.transformer import (
    DepthAugmentedCausalSelfAttention,
    LocalCausalSelfAttention,
    PaperMoDACausalSelfAttention,
    PaperMoDTrainTopCTransformerBlock,
    RotarySoftGatedLocalCausalTransformerBlock,
    SoftGatedLocalCausalTransformerBlock,
    TokenRoutedLocalCausalTransformerBlock,
    local_causal_attention_bias,
)
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
    AttentionProfile,
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
    RecurrentHaltingProfile,
    RecurrentTokenRoutingProfile,
    ReferenceSsmProfile,
    TokenRoutingProfile,
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
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
            ),
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
        self.assertEqual(
            len(diagnostics["parcae_looped_attention"]["last_recurrent_state_norms"]), 3
        )
        self.assertEqual(
            len(diagnostics["parcae_looped_attention"]["last_step_cosines"]), 3
        )
        self.assertEqual(
            len(diagnostics["parcae_looped_attention"]["last_step_accelerations"]), 3
        )
        self.assertEqual(
            len(diagnostics["parcae_looped_attention"]["last_drift_norms"]), 3
        )
        self.assertEqual(diagnostics["parcae_looped_attention"]["last_steps_used"], 3)
        self.assertAlmostEqual(
            diagnostics["parcae_looped_attention"]["average_steps_used"], 3.0
        )

    def test_attention_only_fixed_looped_lm_reuses_block_group(self) -> None:
        shape = Path1ModelShape(
            d_model=32, head_count=4, total_layers=2, ffn_multiplier=2
        )
        variant = phase1_attention_only_variant(
            shape=shape,
            scaffold_profile=Path1ScaffoldProfile.FIXED_LOOPED_LM,
            parcae_loop_count=3,
        )
        deeper_variant = phase1_attention_only_variant(
            shape=shape,
            scaffold_profile=Path1ScaffoldProfile.FIXED_LOOPED_LM,
            parcae_loop_count=5,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        deeper_model = build_path1_model(deeper_variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["looped_transformer"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(
            model.diagnostic_payload()["parameter_count"],
            deeper_model.diagnostic_payload()["parameter_count"],
        )
        self.assertEqual(diagnostics["profile"], "fixed-looped-lm")
        self.assertEqual(diagnostics["loop_count"], 3)
        self.assertEqual(diagnostics["stored_block_count"], 2)
        self.assertEqual(diagnostics["effective_layer_count"], 6)
        self.assertTrue(diagnostics["parameters_shared_across_loops"])
        self.assertTrue(diagnostics["embedding_applied_once"])
        self.assertTrue(diagnostics["output_head_applied_once"])
        self.assertEqual(diagnostics["input_injection_mode"], "none")
        self.assertEqual(diagnostics["initial_state"], "input-embedding")
        self.assertEqual(len(diagnostics["last_hidden_norms"]), 3)

    def test_attention_only_looped_additive_input_reinjects_prompt(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=2, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
            parcae_loop_count=3,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["looped_transformer"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["profile"], "looped-additive-input")
        self.assertEqual(diagnostics["initial_state"], "zero")
        self.assertEqual(diagnostics["input_injection_mode"], "additive")
        self.assertTrue(diagnostics["prompt_injected_each_loop"])
        self.assertEqual(diagnostics["last_initial_state_norm"], 0.0)
        self.assertEqual(len(diagnostics["last_input_injection_norms"]), 3)
        self.assertGreater(diagnostics["last_input_injection_norms"][0], 0.0)

    def test_attention_only_huginn_adapter_recurrence_depends_on_prompt(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=8, head_count=2, total_layers=2, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.huginn_adapter)
        model.huginn_adapter_norm = torch.nn.Identity()
        with torch.no_grad():
            model.huginn_adapter.weight.zero_()
            model.huginn_adapter.bias.zero_()
            for width_index in range(variant.shape.d_model):
                model.huginn_adapter.weight[
                    width_index, variant.shape.d_model + width_index
                ] = 1.0

        state = torch.zeros(1, 3, variant.shape.d_model)
        prompt_a = torch.ones_like(state)
        prompt_b = prompt_a * 2.0
        adapted_a = model._looped_recurrence_input(state, prompt_a)
        adapted_b = model._looped_recurrence_input(state, prompt_b)

        self.assertTrue(torch.allclose(adapted_a, prompt_a))
        self.assertTrue(torch.allclose(adapted_b, prompt_b))
        self.assertFalse(torch.allclose(adapted_a, adapted_b))

        logits = model.forward_logits(
            torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        )
        diagnostics = model.diagnostic_payload()["looped_transformer"]
        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual(diagnostics["profile"], "huginn-adapter-recurrence")
        self.assertEqual(diagnostics["input_injection_mode"], "concat-adapter")
        self.assertEqual(diagnostics["adapter"]["kind"], "concat-linear")
        self.assertEqual(len(diagnostics["last_adapter_input_norms"]), 2)
        self.assertEqual(len(diagnostics["last_adapter_output_norms"]), 2)

    def test_attention_only_universal_transformer_applies_coordinates_each_step(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=1, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
            parcae_loop_count=3,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        state = torch.zeros(1, 4, 32)
        step0 = model._universal_transformer_recurrence_input(state, step_index=0)
        model._last_ut_step_coordinate_norms = []
        model._last_ut_position_coordinate_norm = None
        step1 = model._universal_transformer_recurrence_input(state, step_index=1)
        self.assertFalse(torch.allclose(step0, step1))

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["universal_transformer"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["profile"], "universal-transformer")
        self.assertTrue(diagnostics["parameters_shared_across_steps"])
        self.assertTrue(diagnostics["coordinate_embeddings_each_step"])
        self.assertFalse(diagnostics["act_enabled"])
        self.assertEqual(diagnostics["stored_block_count"], 1)
        self.assertEqual(diagnostics["effective_layer_count"], 3)
        self.assertEqual(len(diagnostics["last_step_coordinate_norms"]), 3)
        self.assertEqual(len(diagnostics["last_hidden_norms"]), 3)

    def test_attention_only_universal_transformer_act_weighted_interpolation(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=4, head_count=1, total_layers=1, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
            parcae_loop_count=3,
            act_halting_threshold=0.99,
            act_ponder_loss_weight=0.01,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.act_halt_projection)
        with torch.no_grad():
            model.act_halt_projection.weight.zero_()
            model.act_halt_projection.bias.zero_()

        def fake_group(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            del mask
            return hidden.new_ones(hidden.shape) * (
                len(model._last_ut_hidden_norms) + 1
            )

        model._forward_looped_block_group = fake_group
        state = model._forward_universal_transformer(
            torch.zeros(1, 2, variant.shape.d_model),
            None,
        )
        diagnostics = model.diagnostic_payload()["universal_transformer"]
        act = diagnostics["act"]

        self.assertTrue(torch.allclose(state, torch.full_like(state, 1.5)))
        self.assertTrue(diagnostics["act_enabled"])
        self.assertAlmostEqual(act["update_weight_sum_min"], 1.0)
        self.assertAlmostEqual(act["update_weight_sum_max"], 1.0)
        self.assertAlmostEqual(act["remainder_mean"], 0.5)
        self.assertAlmostEqual(act["update_count_mean"], 2.0)
        self.assertGreater(act["ponder_loss"], 0.0)

    def test_attention_only_universal_transformer_act_forces_final_halt(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=8, head_count=2, total_layers=1, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
            parcae_loop_count=2,
            act_halting_threshold=0.99,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.act_halt_projection)
        with torch.no_grad():
            model.act_halt_projection.weight.zero_()
            model.act_halt_projection.bias.fill_(-20.0)

        logits = model.forward_logits(
            torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        )
        diagnostics = model.diagnostic_payload()["universal_transformer"]
        act = diagnostics["act"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertAlmostEqual(act["forced_final_halt_fraction"], 1.0)
        self.assertAlmostEqual(act["update_weight_sum_min"], 1.0)
        self.assertAlmostEqual(act["update_weight_sum_max"], 1.0)

    def test_attention_only_ouro_exit_pdf_assigns_final_survival(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=8, head_count=2, total_layers=1, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
            parcae_loop_count=3,
            ouro_q_exit_threshold=0.5,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        zeros = torch.zeros(1, 2)

        exit_pdf = model._ouro_exit_pdf((zeros, zeros, zeros))
        q_exit_steps = model._ouro_q_exit_steps(exit_pdf)

        self.assertEqual(tuple(exit_pdf.shape), (1, 2, 3))
        self.assertTrue(torch.allclose(exit_pdf.sum(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.allclose(exit_pdf[0, 0], torch.tensor([0.5, 0.25, 0.25])))
        self.assertTrue(torch.equal(q_exit_steps, torch.ones(1, 2, dtype=torch.long)))

    def test_attention_only_ouro_learned_exit_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                vocab_size=17,
                d_model=8,
                head_count=2,
                total_layers=1,
                ffn_multiplier=2,
            ),
            scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
            parcae_loop_count=3,
            ouro_entropy_weight=0.05,
            ouro_q_exit_threshold=0.5,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.ouro_exit_gate)
        with torch.no_grad():
            model.ouro_exit_gate.weight.zero_()
            model.ouro_exit_gate.bias.zero_()

        input_ids = torch.randint(low=0, high=17, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        loss = model.supervised_loss(
            logits,
            input_ids,
            pad_token=999,
            training=True,
        )
        diagnostics = model.diagnostic_payload()["ouro_learned_exit"]

        self.assertEqual(tuple(logits.shape), (2, 8, 17))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(diagnostics["profile"], "ouro-learned-exit")
        self.assertTrue(diagnostics["parameters_shared_across_steps"])
        self.assertTrue(diagnostics["output_head_applied_each_step"])
        self.assertEqual(diagnostics["stored_block_count"], 1)
        self.assertEqual(diagnostics["effective_layer_count"], 3)
        self.assertEqual(len(diagnostics["exit_pdf_mean_by_step"]), 3)
        self.assertAlmostEqual(diagnostics["exit_pdf_sum_min"], 1.0)
        self.assertAlmostEqual(diagnostics["exit_pdf_sum_max"], 1.0)
        self.assertAlmostEqual(diagnostics["q_exit_step_mean"], 1.0)
        self.assertLess(diagnostics["last_entropy_regularization"], 0.0)
        self.assertEqual(len(diagnostics["last_per_step_ce_mean"]), 3)

    def test_attention_only_ouro_supervised_loss_uses_expected_ce(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                vocab_size=3,
                d_model=4,
                head_count=1,
                total_layers=1,
                ffn_multiplier=2,
            ),
            scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
            parcae_loop_count=2,
            ouro_entropy_weight=0.1,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        step0 = torch.tensor([[[2.0, 0.0, 0.0]]])
        step1 = torch.tensor([[[0.0, 3.0, 0.0]]])
        exit_pdf = torch.tensor([[[0.25, 0.75]]])
        target_ids = torch.tensor([[1]])
        model._last_ouro_step_logits = (step0, step1)
        model._last_ouro_exit_pdf = exit_pdf

        loss = model.supervised_loss(
            torch.zeros(1, 1, 3),
            target_ids,
            pad_token=999,
            training=True,
        )
        expected = 0.25 * F.cross_entropy(step0.view(1, 3), target_ids.view(-1))
        expected = expected + 0.75 * F.cross_entropy(
            step1.view(1, 3), target_ids.view(-1)
        )
        entropy = -(exit_pdf * exit_pdf.log()).sum()

        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(model.auxiliary_loss(), -0.1 * entropy))

    def test_attention_only_rrt_cycle_reuses_shared_blocks(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.RRT_CYCLE,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        self.assertEqual(len(model.blocks), 2)
        self.assertIs(
            model._rrt_block_for_absolute_depth(0),
            model._rrt_block_for_absolute_depth(2),
        )
        self.assertIs(
            model._rrt_block_for_absolute_depth(1),
            model._rrt_block_for_absolute_depth(3),
        )
        self.assertIs(
            model.blocks[0].input_norm.weight,
            model._rrt_block_for_absolute_depth(2).input_norm.weight,
        )

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["rrt_cycle"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["profile"], "rrt-cycle")
        self.assertEqual(diagnostics["stored_layer_count"], 2)
        self.assertEqual(diagnostics["effective_layer_count"], 4)
        self.assertTrue(diagnostics["parameters_shared_by_cycle"])
        self.assertTrue(diagnostics["strict_recursion_shares_norms"])
        self.assertEqual(diagnostics["last_shared_layer_indices"], [0, 1, 0, 1])
        self.assertEqual(diagnostics["absolute_depth_cache_keys"], [0, 1, 2, 3])
        self.assertEqual(diagnostics["cache_key_mode"], "absolute_depth")
        self.assertFalse(diagnostics["relaxed_lora_enabled"])
        self.assertEqual(diagnostics["lora_rank"], 0)
        self.assertEqual(len(diagnostics["last_hidden_norms"]), 4)

    def test_attention_only_mor_expert_choice_shrinks_active_tokens(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=16, head_count=4, total_layers=3, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
            parcae_loop_count=2,
            recurrent_token_route_fraction=0.5,
            mor_router_aux_loss_weight=0.01,
            mor_update_scale=0.1,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.mor_routers)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["mor_expert_choice"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["profile"], "mor-expert-choice")
        self.assertEqual(diagnostics["stored_layer_count"], 3)
        self.assertEqual(diagnostics["unique_prelude_layer_count"], 1)
        self.assertEqual(diagnostics["shared_middle_layer_count"], 1)
        self.assertEqual(diagnostics["unique_coda_layer_count"], 1)
        self.assertEqual(diagnostics["last_active_token_counts"], [8.0, 4.0])
        self.assertEqual(diagnostics["last_selected_token_counts"], [4.0, 2.0])
        self.assertTrue(diagnostics["selected_indices_sorted"])
        self.assertTrue(diagnostics["unselected_tokens_stop_recursing"])
        self.assertEqual(
            diagnostics["kv_policy"], "recursion-wise-selected-subsequence-reference"
        )
        self.assertFalse(diagnostics["decode_safe"])
        self.assertGreater(diagnostics["router_aux_loss"], 0.0)
        self.assertGreater(model.auxiliary_loss().detach().item(), 0.0)
        self.assertEqual(len(diagnostics["last_selected_positions"]), 2)
        self.assertEqual(len(diagnostics["last_hidden_norms"]), 2)

    def test_attention_only_depth_augmented_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            attention_profile=AttentionProfile.MODA_DEPTH_KV,
            depth_memory_layers=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(
            model.blocks[0].attention, DepthAugmentedCausalSelfAttention
        )
        self.assertEqual(diagnostics["attention_profile"], "moda-depth-kv")
        self.assertEqual(
            diagnostics["depth_augmented_attention"]["depth_memory_layers"], 2
        )
        self.assertEqual(
            len(diagnostics["depth_augmented_attention"]["attention_blocks"]), 4
        )
        first_payload = diagnostics["depth_augmented_attention"]["attention_blocks"][1][
            "payload"
        ]
        self.assertTrue(first_payload["joint_softmax"])
        self.assertFalse(first_payload["same_token_depth_memory"])
        self.assertIn("last_depth_attention_mass", first_payload)

    def test_attention_only_paper_moda_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            attention_profile=AttentionProfile.PAPER_MODA_DEPTH_KV,
            depth_memory_layers=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.blocks[0].attention, PaperMoDACausalSelfAttention)
        self.assertEqual(diagnostics["attention_profile"], "paper-moda-depth-kv")
        paper_payload = diagnostics["depth_augmented_attention"]["attention_blocks"][1][
            "payload"
        ]
        self.assertEqual(paper_payload["kind"], "paper-moda-depth-kv-reference")
        self.assertTrue(paper_payload["joint_softmax"])
        self.assertTrue(paper_payload["same_token_depth_memory"])
        self.assertGreater(paper_payload["last_depth_attention_mass"], 0.0)

    def test_paper_moda_depth_memory_is_same_token_only(self) -> None:
        attention = PaperMoDACausalSelfAttention(d_model=2, head_count=1)
        with torch.no_grad():
            attention.qkv_projection.weight.zero_()
            attention.qkv_projection.bias.zero_()
            attention.output_projection.weight.copy_(torch.eye(2))
            attention.output_projection.bias.zero_()

        hidden = torch.zeros(1, 2, 2)
        depth_k = torch.zeros(1, 1, 2, 2)
        depth_v = torch.tensor([[[[9.0, 0.0], [0.0, 12.0]]]])

        output = attention(hidden, depth_memory=[(depth_k, depth_v)])
        payload = attention.diagnostic_payload()

        # Query 0 has one visible sequence key plus its own depth slot. It must
        # not see token 1's depth value.
        self.assertTrue(torch.allclose(output[0, 0], torch.tensor([4.5, 0.0])))
        # Query 1 has two visible sequence keys plus only token 1's depth slot.
        self.assertTrue(torch.allclose(output[0, 1], torch.tensor([0.0, 4.0])))
        self.assertAlmostEqual(payload["last_depth_attention_mass"], 5.0 / 12.0)

    def test_attention_only_depth_augmented_is_causal(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            attention_profile=AttentionProfile.MODA_DEPTH_KV,
            depth_memory_layers=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(
            torch.allclose(
                logits_a[:, : prefix.shape[1], :],
                logits_b[:, : prefix.shape[1], :],
                atol=1.0e-6,
            )
        )

    def test_attention_only_token_routed_block_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
            token_route_fraction=0.5,
            token_routing_layer_indices=(2,),
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.blocks[2], TokenRoutedLocalCausalTransformerBlock)
        self.assertEqual(diagnostics["token_routing_profile"], "causal-topk-block")
        self.assertEqual(diagnostics["token_routing"]["layer_indices"], (2,))
        self.assertEqual(len(diagnostics["token_routing"]["blocks"]), 1)
        payload = diagnostics["token_routing"]["blocks"][0]["payload"]
        self.assertGreater(payload["last_selected_count"], 0)
        self.assertLessEqual(
            payload["last_selected_count"], payload["last_token_count"]
        )
        self.assertEqual(
            payload["last_skipped_count"],
            payload["last_token_count"] - payload["last_selected_count"],
        )
        self.assertEqual(
            payload["selected_attention_scope"],
            "selected-query/full-causal-key-value",
        )
        self.assertTrue(payload["causal_decode_safe"])

    def test_attention_only_mod_train_topc_block_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            token_routing_profile=TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK,
            token_route_fraction=0.5,
            token_routing_layer_indices=(2,),
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.blocks[2], PaperMoDTrainTopCTransformerBlock)
        payload = diagnostics["token_routing"]["blocks"][0]["payload"]
        self.assertEqual(payload["kind"], "mod-train-topc-block")
        self.assertEqual(payload["capacity"], 4)
        self.assertEqual(payload["last_skipped_count"], 8)
        self.assertEqual(payload["selected_attention_scope"], "selected-only")
        self.assertFalse(payload["causal_decode_safe"])
        self.assertEqual(
            diagnostics["token_routing"]["label"],
            "Paper MoD training-time full-sequence top-C block routing",
        )
        routing_tensors = model.blocks[2].last_routing_tensors()
        self.assertEqual(tuple(routing_tensors["router_scores"].shape), (2, 8))
        self.assertEqual(tuple(routing_tensors["selected_mask"].shape), (2, 8))
        self.assertEqual(int(routing_tensors["selected_mask"].sum().item()), 8)

    def test_mod_train_topc_skipped_tokens_are_identity(self) -> None:
        block = PaperMoDTrainTopCTransformerBlock(
            d_model=16,
            head_count=4,
            d_ff=32,
            route_fraction=0.5,
        )
        hidden = torch.randn(2, 8, 16)
        selected_positions, _ = block._route(hidden)

        output = block(hidden)
        payload = block.token_routing_payload()

        self.assertEqual(payload["capacity"], 4)
        self.assertEqual(payload["last_skipped_count"], 8)
        for batch_index, positions in enumerate(selected_positions):
            selected = set(int(position) for position in positions.tolist())
            skipped = [
                position
                for position in range(hidden.shape[1])
                if position not in selected
            ]
            if skipped:
                skipped_tensor = torch.tensor(skipped, dtype=torch.long)
                self.assertTrue(
                    torch.allclose(
                        output[batch_index].index_select(0, skipped_tensor),
                        hidden[batch_index].index_select(0, skipped_tensor),
                    )
                )

    def test_attention_only_token_routing_is_causal(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
            token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
            token_route_fraction=0.5,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 5), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 3), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(
            torch.allclose(
                logits_a[:, : prefix.shape[1], :],
                logits_b[:, : prefix.shape[1], :],
                atol=1.0e-6,
            )
        )

    def test_attention_only_soft_gated_blocks_forward_cpu(self) -> None:
        cases = (
            (TokenRoutingProfile.SOFT_GATE_BLOCK, SoftGatedLocalCausalTransformerBlock),
            (
                TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK,
                RotarySoftGatedLocalCausalTransformerBlock,
            ),
        )
        for profile, expected_type in cases:
            with self.subTest(profile=profile.value):
                variant = phase1_attention_only_variant(
                    shape=Path1ModelShape(
                        d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
                    ),
                    token_routing_profile=profile,
                    token_route_fraction=0.25,
                    token_routing_layer_indices=(2,),
                )
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(
                    low=0, high=257, size=(2, 8), dtype=torch.long
                )

                logits = model.forward_logits(input_ids)
                diagnostics = model.diagnostic_payload()
                payload = diagnostics["token_routing"]["blocks"][0]["payload"]

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsInstance(model.blocks[2], expected_type)
                self.assertEqual(payload["kind"], profile.value)
                self.assertEqual(payload["gate_floor"], 0.25)
                self.assertGreaterEqual(payload["last_min_gate"], 0.25)
                self.assertLessEqual(payload["last_max_gate"], 1.0)
                self.assertGreater(payload["last_accepted_delta_norm"], 0.0)
                self.assertTrue(payload["causal_decode_safe"])
                if profile is TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK:
                    self.assertGreater(payload["last_controller_norm"], 0.0)

    def test_attention_only_parcae_acceleration_halting_reports_steps(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=4,
            recurrent_halting_profile=RecurrentHaltingProfile.ACCELERATION,
            recurrent_min_steps=2,
            recurrent_halting_threshold=1.0e9,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["halting_profile"], "acceleration")
        self.assertEqual(diagnostics["last_steps_used"], 2)
        self.assertTrue(diagnostics["last_halted_early"])
        self.assertEqual(diagnostics["early_exit_count"], 1)

    def test_attention_only_parcae_vector_acceleration_halting_reports_steps(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=5,
            recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
            recurrent_min_steps=2,
            recurrent_halting_threshold=1.0e9,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["halting_profile"], "vector-acceleration")
        self.assertEqual(diagnostics["last_steps_used"], 2)
        self.assertTrue(diagnostics["last_halted_early"])
        self.assertEqual(diagnostics["early_exit_count"], 1)
        self.assertGreater(diagnostics["last_halting_metric"], 0.0)

    def test_attention_only_parcae_token_selective_recurrence_reports_fraction(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
            ),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=3,
            recurrent_token_routing_profile=(
                RecurrentTokenRoutingProfile.CAUSAL_TOPK_STATE
            ),
            recurrent_token_route_fraction=0.5,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["token_routing_profile"], "causal-topk-state")
        self.assertEqual(len(diagnostics["last_selected_token_fractions"]), 3)
        self.assertGreater(diagnostics["average_selected_token_fraction"], 0.0)
        self.assertLessEqual(diagnostics["average_selected_token_fraction"], 1.0)

    def test_attention_only_parcae_bx_and_p20_control_scaffolds_forward_cpu(
        self,
    ) -> None:
        for scaffold_profile in (
            Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
            Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
        ):
            with self.subTest(scaffold_profile=scaffold_profile.value):
                variant = phase1_attention_only_variant(
                    shape=Path1ModelShape(
                        d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
                    ),
                    scaffold_profile=scaffold_profile,
                    parcae_loop_count=2,
                    token_route_fraction=0.5,
                )
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(
                    low=0, high=257, size=(2, 8), dtype=torch.long
                )

                logits = model.forward_logits(input_ids)
                diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(diagnostics["profile"], scaffold_profile.value)
                self.assertIsNotNone(diagnostics["last_injection_gate_mean"])
                self.assertIsNotNone(diagnostics["last_injection_norm"])
                self.assertEqual(diagnostics["p20_value_scale"], 1.0)
                if scaffold_profile in {
                    Path1ScaffoldProfile.PARCAE_P20_THIN_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
                }:
                    self.assertEqual(diagnostics["p20_control_width"], 16)
                if (
                    scaffold_profile
                    is Path1ScaffoldProfile.PARCAE_P20_QUARTER_CONTROL_LOOPED_ATTENTION
                ):
                    self.assertEqual(diagnostics["p20_control_width"], 8)
                if (
                    scaffold_profile
                    is Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_CONTROL_LOOPED_ATTENTION
                    or scaffold_profile
                    is Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION
                ):
                    self.assertTrue(diagnostics["p20_gate_only_control"])
                if (
                    scaffold_profile
                    is Path1ScaffoldProfile.PARCAE_P20_THIN_VALUE_CONTROL_LOOPED_ATTENTION
                ):
                    self.assertTrue(diagnostics["p20_value_only_control"])
                if scaffold_profile in {
                    Path1ScaffoldProfile.PARCAE_P20_THIN_GATE_BASEBLEND_LOOPED_ATTENTION,
                    Path1ScaffoldProfile.PARCAE_P20_THIN_BASEBLEND_CONTROL_LOOPED_ATTENTION,
                }:
                    self.assertTrue(diagnostics["p20_base_gate_blend"])
                if scaffold_profile is not Path1ScaffoldProfile.PARCAE_BX_LOOPED_ATTENTION:
                    control_diagnostics = diagnostics["control_diagnostics"]
                    self.assertIn("controller/gate_mean", control_diagnostics)
                    self.assertIn(
                        "controller/injection_delta_to_loop_input_ratio",
                        control_diagnostics,
                    )
                    self.assertIn("loop/state_norm_step_0", control_diagnostics)
                    self.assertIn(
                        "loop/step_delta_norm_step_0", control_diagnostics
                    )
                    self.assertIn(
                        "loop/acceleration_norm_step_0", control_diagnostics
                    )
                    self.assertIn("stability/nan_or_inf_seen", control_diagnostics)
                    if scaffold_profile in {
                        Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION,
                        Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION,
                    }:
                        self.assertIn(
                            "mod_router/selected_fraction", control_diagnostics
                        )
                        self.assertGreater(
                            control_diagnostics["mod_router/selected_fraction"], 0.0
                        )
                    if (
                        scaffold_profile
                        is Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION
                    ):
                        self.assertIn(
                            "mod_router/gate_bias_scale", control_diagnostics
                        )
                    if (
                        scaffold_profile
                        is Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION
                    ):
                        self.assertIn(
                            "mod_router/value_multiplier_mean", control_diagnostics
                        )

    def test_attention_only_parcae_p20_value_scale_reduces_value_diagnostics(
        self,
    ) -> None:
        shape = Path1ModelShape(
            d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
        )
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        ratios: list[float] = []
        for value_scale in (1.0, 0.5):
            torch.manual_seed(19)
            variant = phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
                parcae_loop_count=2,
                parcae_p20_value_scale=value_scale,
            )
            model = build_path1_model(variant, dtype_mode="fp32")
            logits = model.forward_logits(input_ids)
            self.assertTrue(torch.isfinite(logits).all())
            diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
            self.assertEqual(diagnostics["p20_value_scale"], value_scale)
            ratios.append(
                diagnostics["control_diagnostics"][
                    "controller/value_to_loop_input_norm_ratio"
                ]
            )

        self.assertAlmostEqual(ratios[1], 0.5 * ratios[0], delta=1.0e-5)

    def test_attention_only_parcae_p20_control_receives_runtime_policy(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=6, ffn_multiplier=2
            ),
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
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=2, ffn_multiplier=2
            ),
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
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=2, ffn_multiplier=2
            ),
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

    def test_attention_only_surgical_gated_eml_targets_selected_layer_only(
        self,
    ) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
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
        self.assertEqual(
            diagnostics["eml_inspired_feed_forward"]["layer_indices"], (2,)
        )

    def test_attention_only_routed_eml_targets_selected_layer_only(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
            ),
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
        self.assertEqual(
            diagnostics["eml_inspired_feed_forward"]["layer_indices"], (2,)
        )
        self.assertEqual(
            diagnostics["eml_inspired_feed_forward"]["route_fraction"], 0.25
        )

    def test_attention_only_expert_controls_target_selected_layer_only(self) -> None:
        cases = (
            (FeedForwardProfile.TINY_MLP_GATED, TinyMlpExpertFeedForward),
            (FeedForwardProfile.TINY_GLU_GATED, TinyGluExpertFeedForward),
            (FeedForwardProfile.GENERIC_TREE_GATED, GenericTreeExpertFeedForward),
        )
        for profile, expected_type in cases:
            with self.subTest(profile=profile.value):
                variant = phase1_attention_only_variant(
                    shape=Path1ModelShape(
                        d_model=32, head_count=4, total_layers=4, ffn_multiplier=2
                    ),
                    feed_forward_profile=profile,
                    feed_forward_layer_indices=(2,),
                    eml_slot_count=4,
                    eml_tree_depth=2,
                )
                model = build_path1_model(variant, dtype_mode="fp32")
                input_ids = torch.randint(
                    low=0, high=257, size=(2, 8), dtype=torch.long
                )

                logits = model.forward_logits(input_ids)
                diagnostics = model.diagnostic_payload()

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsInstance(model.blocks[0].ffn, PositionWiseFeedForward)
                self.assertIsInstance(model.blocks[2].ffn, expected_type)
                self.assertEqual(
                    diagnostics["feed_forward_experts"][0]["layer_index"], 2
                )

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
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("gated_deltanet_torch", model.model_label)

    def test_gated_deltanet_reference_is_causal(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_TORCH
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 6), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(
            torch.allclose(
                logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]
            )
        )

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
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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

    def test_pr5_scaffold_sparse_p20_insertion_uses_only_selected_recurrent_slot(
        self,
    ) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            (
                ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
                if ordinal == 4
                else ReferenceSsmProfile.GATED_DELTANET_TORCH
            )
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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
        mixer_kinds = [
            block["mixer"]["kind"] for block in diagnostics["reference_ssm_blocks"]
        ]
        self.assertEqual(mixer_kinds.count("gdnp-fused"), 1)
        self.assertEqual(mixer_kinds.count("parallel-composite"), 0)
        self.assertEqual(
            diagnostics["reference_ssm_blocks"][4]["profile"],
            "gated-deltanet-p20-fused-multi-read-torch",
        )

    def test_pr5_scaffold_tiny_p20_conditioner_keeps_gdn_readout_owner(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            (
                ReferenceSsmProfile.GATED_DELTANET_FLA_P20_CONTROL_TINY
                if ordinal == 4
                else ReferenceSsmProfile.GATED_DELTANET_TORCH
            )
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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
            (
                ReferenceSsmProfile.GATED_DELTANET_FLA_CONTROL_SHELL
                if ordinal == 4
                else ReferenceSsmProfile.GATED_DELTANET_TORCH
            )
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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
            (
                ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
                if ordinal == 4
                else ReferenceSsmProfile.GATED_DELTANET_TORCH
            )
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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

        self.assertTrue(
            torch.allclose(
                logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]
            )
        )

    def test_pr5_scaffold_optimizer_groups_are_disjoint(self) -> None:
        schedule = parse_layer_schedule_spec("RRRRRARRRRRS")
        profile_schedule = tuple(
            (
                ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_MULTI_READ_TORCH
                if ordinal == 4
                else ReferenceSsmProfile.GATED_DELTANET_TORCH
            )
            for ordinal in range(10)
        )
        variant = phase1_reference_ssm_variant(
            shape=Path1ModelShape(
                d_model=32, head_count=4, total_layers=len(schedule), local_window=4
            ),
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
        self.assertEqual(
            len({id(param) for param in grouped_params}), len(grouped_params)
        )
        self.assertEqual(
            {id(param) for param in model.parameters() if param.requires_grad},
            {id(param) for param in grouped_params},
        )
        lr_by_name = {group["name"]: group["lr"] for group in groups}
        self.assertEqual(lr_by_name["pr5_recurrent"], 5.0e-4)
        self.assertEqual(lr_by_name["pr5_gates_controls"], 5.0e-4)

    def test_gated_deltanet_p20_composite_forward_cpu(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_P20_TORCH
        )
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
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_P20_THIN_TORCH
        )
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
                input_ids = torch.randint(
                    low=0, high=257, size=(2, 8), dtype=torch.long
                )

                logits = model.forward_logits(input_ids)

                self.assertEqual(tuple(logits.shape), (2, 8, 257))
                self.assertIn(profile.value.replace("-", "_"), model.model_label)
                diagnostics = model.diagnostic_payload()
                first_block = diagnostics["reference_ssm_blocks"][0]
                self.assertEqual(first_block["mixer"]["kind"], "gdnp-fused")
                self.assertEqual(first_block["mixer"]["law"], profile.gdnp_fused_law)
                self.assertEqual(first_block["mixer"]["vector_state_width"], 128)

    def test_gdnp_fused_reference_is_causal(self) -> None:
        variant = phase1_reference_ssm_variant(
            profile=ReferenceSsmProfile.GATED_DELTANET_P20_FUSED_TORCH
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        prefix = torch.randint(low=0, high=257, size=(1, 6), dtype=torch.long)
        suffix_a = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)
        suffix_b = torch.randint(low=0, high=257, size=(1, 4), dtype=torch.long)

        logits_a = model.forward_logits(torch.cat((prefix, suffix_a), dim=1))
        logits_b = model.forward_logits(torch.cat((prefix, suffix_b), dim=1))

        self.assertTrue(
            torch.allclose(
                logits_a[:, : prefix.shape[1], :], logits_b[:, : prefix.shape[1], :]
            )
        )

    def test_gdnp_fused_triton_policy_routes_vector_scan_to_sequence_kernel(
        self,
    ) -> None:
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
        self.assertEqual(
            mixer.diagnostic_payload()["primitive_runtime_backend"], "triton"
        )
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
                    input_ids = torch.randint(
                        low=0, high=257, size=(2, 8), dtype=torch.long
                    )
                    logits = model.forward_logits(input_ids)
                    self.assertEqual(tuple(logits.shape), (2, 8, 257))
                    diagnostics = model.diagnostic_payload()
                    first_block = diagnostics["reference_ssm_blocks"][0]
                    self.assertEqual(
                        first_block["mixer"]["kind"], "fla-gdnp-compatible"
                    )
                    self.assertEqual(
                        first_block["mixer"]["law"], profile.fla_gdnp_compatible_law
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "FLA gated-delta-rule import failed"
                    ):
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
                    input_ids = torch.randint(
                        low=0, high=257, size=(2, 8), dtype=torch.long
                    )
                    logits = model.forward_logits(input_ids)
                    self.assertEqual(tuple(logits.shape), (2, 8, 257))
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "official PyTorch Mamba3 import failed"
                    ):
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
        with self.assertRaisesRegex(
            ValueError, "does not support non-dense state transforms"
        ):
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
                with self.assertRaisesRegex(
                    ValueError, "does not support non-dense state transforms"
                ):
                    build_sequence_primitive(
                        primitive_profile,
                        16,
                        PrimitiveExecutionProfile.RUNTIME,
                        state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_2,
                    )

    def test_p20_gdn_role_rejects_non_dense_state_transform(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "does not support non-dense state transforms"
        ):
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
        extended_outputs = primitive.scan(
            torch.cat([prefix, suffix], dim=1)
        ).emitted_outputs[:, :6, :]
        self.assertTrue(
            torch.allclose(prefix_outputs, extended_outputs, atol=1.0e-5, rtol=1.0e-5)
        )

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
        self.assertEqual(
            len({id(param) for param in grouped_params}), len(grouped_params)
        )
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

            def fused_p20_update_readout(
                self, **_: object
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError(
                    "block-diagonal Triton runtime should not fall back to the step kernel"
                )

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
        self.assertTrue(
            torch.allclose(
                result.emitted_outputs, torch.full_like(result.emitted_outputs, 3.0)
            )
        )
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0))
        )

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

            def scan_p20_dense_sequence(
                self, **_: object
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError(
                    "block-diagonal Triton runtime should not route to the dense sequence kernel"
                )

            def fused_p20_update_readout(
                self, **_: object
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError(
                    "block-diagonal Triton runtime should not fall back to the step kernel"
                )

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
        self.assertTrue(
            torch.allclose(
                result.emitted_outputs, torch.full_like(result.emitted_outputs, 7.0)
            )
        )
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 8.0))
        )

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

            def scan_p20_block_diagonal_sequence(
                self, **_: object
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError(
                    "dense Triton runtime should not route to the block-diagonal sequence kernel"
                )

            def fused_p20_update_readout(
                self, **_: object
            ) -> tuple[torch.Tensor, torch.Tensor]:
                raise AssertionError(
                    "dense Triton runtime should not fall back to the step kernel"
                )

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
        self.assertTrue(
            torch.allclose(
                result.emitted_outputs, torch.full_like(result.emitted_outputs, 5.0)
            )
        )
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 6.0))
        )

    def test_runtime_p2_block_diagonal_2_triton_routes_to_state_sequence_scan(
        self,
    ) -> None:
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
        expected_outputs = runtime_plan.output_gates * runtime.output_projection(
            torch.full_like(runtime_plan.update_gates, 2.0)
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, expected_outputs))
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0))
        )

    def test_runtime_p21_block_diagonal_2_triton_routes_to_state_sequence_scan(
        self,
    ) -> None:
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
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 4.0))
        )

    def test_runtime_p22_block_diagonal_2_triton_routes_to_state_sequence_scan(
        self,
    ) -> None:
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
        self.assertTrue(
            torch.allclose(result.final_state, torch.full_like(result.final_state, 5.0))
        )

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
                    [
                        [4.0, 1.0, 0.5, -1.0],
                        [0.0, 2.0, 1.0, -2.0],
                        [3.0, 2.0, 1.0, 0.0],
                    ],
                    [
                        [1.0, 0.5, 4.0, 3.5],
                        [2.0, 1.0, 0.0, -1.0],
                        [0.2, 0.1, 0.0, -0.1],
                    ],
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
                    moe_layer_schedule=MiniMoeLayerSchedule(
                        kind=MiniMoeLayerScheduleKind.EXPLICIT, explicit_layers=(0, 2)
                    ),
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
