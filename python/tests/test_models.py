from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from python.models.mini_moe import (
    CollectingMiniMoeObservabilitySink,
    OneShotRouter,
    RecurrentPreExpertRouter,
    RoutePlan,
    RouteRoundSummary,
    SparseTopKDispatcher,
)
from python.models.mini_moe import MiniMoeBackboneModel
from python.models.primitives import build_sequence_primitive
from python.models.path1 import build_path1_model
from python.models.reference_ssm import resolve_reference_ssm_config
from python.reporting.render import (
    render_mini_moe_round_transition_table,
    render_mini_moe_token_trace_table,
)
from python.reporting.schema import (
    BenchmarkReport,
    EvalSummary,
    RuntimeSummary,
    TrainStepRecord,
    read_report,
    write_report,
)
from python.specs.mini_moe import (
    LearnedGateTeacherKind,
    MiniMoeArchitectureSpec,
    MiniMoeBackboneSpec,
    MiniMoeDispatchExecutionStrategy,
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
    RecurrentRoundExecutionStrategy,
    RecurrentRoundGateKind,
    RecurrentRoundGateSpec,
    RecurrentPreExpertRouterSpec,
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
    phase1_attention_only_variant,
    phase1_primitive_variant,
    phase1_reference_ssm_variant,
)


class RecordingScaleExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.last_token_count = 0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.last_token_count = int(hidden.shape[0])
        return hidden * self.scale


class Path1ModelTests(unittest.TestCase):
    def test_attention_only_forward_cpu(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 257))

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


class MiniMoeModelTests(unittest.TestCase):
    def test_router_emits_dense_routing_intent(self) -> None:
        router = OneShotRouter(hidden_dim=16, experts_per_block=4)
        hidden = torch.randn(2, 3, 16)
        plan = router.plan(hidden)
        self.assertIsInstance(plan, RoutePlan)
        self.assertEqual(tuple(plan.expert_logits.shape), (2, 3, 4))
        self.assertEqual(tuple(plan.expert_weights.shape), (2, 3, 4))
        self.assertEqual(len(plan.round_summaries), 1)

    def test_recurrent_router_emits_multiple_round_summaries(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        hidden = torch.randn(2, 3, 16)
        plan = router.plan(hidden)
        self.assertEqual(tuple(plan.expert_logits.shape), (2, 3, 4))
        self.assertEqual(tuple(plan.expert_weights.shape), (2, 3, 4))
        self.assertEqual(len(plan.round_summaries), 2)
        self.assertEqual(plan.round_summaries[1].applied_token_fraction, 1.0)

    def test_recurrent_router_skips_round2_when_disabled_for_layer(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
                normalized_entropy_threshold=0.95,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
            round2_enabled=False,
        )
        hidden = torch.randn(2, 3, 16)
        plan = router.plan(hidden)
        self.assertEqual(len(plan.round_summaries), 1)
        self.assertEqual(plan.round_summaries[0].applied_token_fraction, 1.0)

    def test_recurrent_router_hoisted_logits_match_expanded_state_formulation(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        hidden = torch.randn(2, 3, 16)
        plan = router.plan(hidden)

        batch_size, seq_len, _ = hidden.shape
        token_state = torch.tanh(router.token_state_projection(hidden))
        pooled_token_state = token_state.mean(dim=1, keepdim=True)
        controller_state = torch.zeros(
            batch_size,
            1,
            pooled_token_state.shape[-1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        manual_rounds: list[RouteRoundSummary] = []
        for _ in range(router.round_count):
            expanded_state = controller_state.expand(-1, seq_len, -1)
            expert_logits = router.token_route_projection(hidden) + router.state_route_projection(expanded_state)
            expert_weights = torch.softmax(expert_logits, dim=-1)
            manual_rounds.append(
                RouteRoundSummary(
                    expert_logits=expert_logits,
                    expert_weights=expert_weights,
                )
            )
            pooled_feedback = router.route_feedback_projection(expert_weights.mean(dim=1, keepdim=True))
            state_input = pooled_token_state + pooled_feedback
            reset_gate = torch.sigmoid(router.reset_gate_projection(state_input))
            update_gate = torch.sigmoid(router.update_gate_projection(state_input))
            candidate_state = torch.tanh(
                router.candidate_input_projection(state_input)
                + router.candidate_state_projection(reset_gate * controller_state)
            )
            controller_state = (1.0 - update_gate) * controller_state + update_gate * candidate_state

        self.assertTrue(torch.allclose(plan.expert_logits, manual_rounds[-1].expert_logits))
        self.assertTrue(torch.allclose(plan.expert_weights, manual_rounds[-1].expert_weights))
        self.assertEqual(len(plan.round_summaries), len(manual_rounds))
        for actual, expected in zip(plan.round_summaries, manual_rounds):
            self.assertTrue(torch.allclose(actual.expert_logits, expected.expert_logits))
            self.assertTrue(torch.allclose(actual.expert_weights, expected.expert_weights))

    def test_recurrent_router_winner_margin_gate_masks_confident_tokens(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
                threshold=0.05,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        previous_weights = torch.tensor(
            [
                [[0.60, 0.20, 0.10, 0.10], [0.27, 0.26, 0.24, 0.23]],
                [[0.40, 0.39, 0.11, 0.10], [0.80, 0.10, 0.05, 0.05]],
            ]
        )
        active_mask, _, _, _ = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=torch.zeros(2, 2, 8),
        )
        self.assertEqual(active_mask.tolist(), [[False, True], [True, False]])

    def test_recurrent_router_target_applied_fraction_gate_selects_lowest_margins(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.TARGET_APPLIED_FRACTION,
                target_applied_fraction=0.5,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        previous_weights = torch.tensor(
            [
                [[0.60, 0.20, 0.10, 0.10], [0.27, 0.26, 0.24, 0.23]],
                [[0.40, 0.39, 0.11, 0.10], [0.80, 0.10, 0.05, 0.05]],
            ]
        )
        active_mask, _, _, _ = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=torch.zeros(2, 2, 8),
        )
        self.assertEqual(active_mask.tolist(), [[False, True], [True, False]])

    def test_recurrent_router_scaled_margin_gate_adjusts_for_expert_count(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=8,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.SCALED_WINNER_MARGIN_BELOW,
                threshold=0.02,
                reference_experts_per_block=4,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        previous_weights = torch.tensor(
            [
                [
                    [0.30, 0.286, 0.10, 0.09, 0.08, 0.06, 0.05, 0.034],
                    [0.20, 0.18, 0.14, 0.12, 0.11, 0.10, 0.08, 0.07],
                ]
            ]
        )
        active_mask, _, _, _ = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=torch.zeros(1, 2, 8),
        )
        self.assertEqual(active_mask.tolist(), [[False, False]])

    def test_recurrent_router_normalized_entropy_gate_selects_high_entropy_tokens(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.NORMALIZED_ENTROPY_ABOVE,
                normalized_entropy_threshold=0.8,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        previous_weights = torch.tensor(
            [
                [[0.60, 0.20, 0.10, 0.10], [0.27, 0.26, 0.24, 0.23]],
                [[0.40, 0.39, 0.11, 0.10], [0.80, 0.10, 0.05, 0.05]],
            ]
        )
        active_mask, _, _, _ = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=torch.zeros(2, 2, 8),
        )
        self.assertEqual(active_mask.tolist(), [[False, True], [True, False]])

    def test_recurrent_router_learned_gate_produces_hard_mask_and_soft_probability(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.LEARNED_SCORE_ABOVE,
                learned_hidden_dim=8,
                learned_prior_probability=0.2,
                teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                teacher_supervision_weight=1.0,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        assert router.learned_gate_network is not None
        final_linear = router.learned_gate_network[-1]
        self.assertIsInstance(final_linear, torch.nn.Linear)
        with torch.no_grad():
            final_linear.bias.fill_(10.0)
        previous_weights = torch.tensor(
            [
                [[0.60, 0.20, 0.10, 0.10], [0.27, 0.26, 0.24, 0.23]],
            ]
        )
        active_mask, gate_values, mean_gate_probability, auxiliary_loss = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=torch.zeros(1, 2, 8),
        )
        self.assertEqual(active_mask.tolist(), [[True, True]])
        self.assertTrue(torch.allclose(gate_values, torch.ones_like(gate_values)))
        self.assertIsNotNone(mean_gate_probability)
        assert mean_gate_probability is not None
        self.assertGreater(mean_gate_probability, 0.99)
        self.assertIsNotNone(auxiliary_loss)

    def test_recurrent_router_learned_score_top_fraction_selects_highest_scores(self) -> None:
        router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.LEARNED_SCORE_TOP_FRACTION,
                learned_hidden_dim=1,
                learned_prior_probability=0.2,
                target_applied_fraction=0.5,
                teacher_kind=LearnedGateTeacherKind.BLENDED_UNCERTAINTY,
                teacher_supervision_weight=1.0,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        assert router.learned_gate_network is not None
        first_linear = router.learned_gate_network[0]
        final_linear = router.learned_gate_network[-1]
        self.assertIsInstance(first_linear, torch.nn.Linear)
        self.assertIsInstance(final_linear, torch.nn.Linear)
        with torch.no_grad():
            first_linear.weight.zero_()
            first_linear.bias.zero_()
            first_linear.weight[0, 0] = 1.0
            final_linear.weight.fill_(5.0)
            final_linear.bias.zero_()
        previous_weights = torch.tensor(
            [[[0.60, 0.20, 0.10, 0.10], [0.27, 0.26, 0.24, 0.23]]]
        )
        token_state = torch.zeros(1, 2, 8)
        token_state[0, 0, 0] = 0.1
        token_state[0, 1, 0] = 2.0
        active_mask, gate_values, mean_gate_probability, auxiliary_loss = router._resolve_gate(
            previous_weights=previous_weights,
            token_state=token_state,
        )
        self.assertEqual(active_mask.tolist(), [[False, True]])
        self.assertLess(float(gate_values[0, 0].item()), float(gate_values[0, 1].item()))
        self.assertIsNotNone(mean_gate_probability)
        self.assertIsNotNone(auxiliary_loss)

    def test_mini_moe_model_surfaces_auxiliary_loss_from_learned_gate(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_learned_score_gated_default(
            learned_hidden_dim=8,
            learned_prior_probability=0.2,
            target_applied_fraction=0.2,
            teacher_supervision_weight=1.0,
        )
        model = MiniMoeBackboneModel(surface)
        input_ids = torch.randint(0, surface.architecture.backbone.vocab_size, (1, 4))
        logits = model.forward_logits(input_ids)
        self.assertEqual(logits.shape, (1, 4, surface.architecture.backbone.vocab_size))
        auxiliary_loss = model.pop_auxiliary_loss()
        self.assertIsNotNone(auxiliary_loss)
        assert auxiliary_loss is not None
        self.assertTrue(torch.isfinite(auxiliary_loss))

    def test_mini_moe_model_only_enables_round2_for_selected_layers(self) -> None:
        selected_layers = (3, 4, 6, 7)
        surface = MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default(
            normalized_entropy_threshold=0.95,
            round2_layer_indices=selected_layers,
        )
        model = MiniMoeBackboneModel(surface)
        for layer_index, block in enumerate(model.blocks):
            router = getattr(getattr(block, "ffn", None), "router", None)
            self.assertIsInstance(router, RecurrentPreExpertRouter)
            self.assertEqual(router.round2_enabled, layer_index in selected_layers)

    def test_recurrent_round_execution_strategies_match(self) -> None:
        dense_router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
                threshold=0.05,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
        )
        masked_router = RecurrentPreExpertRouter(
            hidden_dim=16,
            experts_per_block=4,
            state_dim=8,
            round_count=2,
            gate=RecurrentRoundGateSpec(
                kind=RecurrentRoundGateKind.WINNER_MARGIN_BELOW,
                threshold=0.05,
            ),
            execution_strategy=RecurrentRoundExecutionStrategy.MASKED_TOKEN_UPDATE,
        )
        masked_router.load_state_dict(dense_router.state_dict())
        hidden = torch.randn(2, 3, 16)
        dense_plan = dense_router.plan(hidden)
        masked_plan = masked_router.plan(hidden)
        self.assertTrue(torch.allclose(dense_plan.expert_logits, masked_plan.expert_logits))
        self.assertTrue(torch.allclose(dense_plan.expert_weights, masked_plan.expert_weights))
        self.assertEqual(len(dense_plan.round_summaries), len(masked_plan.round_summaries))
        for dense_round, masked_round in zip(dense_plan.round_summaries, masked_plan.round_summaries):
            self.assertTrue(torch.allclose(dense_round.expert_logits, masked_round.expert_logits))
            self.assertTrue(torch.allclose(dense_round.expert_weights, masked_round.expert_weights))
            self.assertEqual(dense_round.applied_token_fraction, masked_round.applied_token_fraction)

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
            ),
            expert_weights=torch.tensor(
                [
                    [[0.87, 0.08, 0.04, 0.01], [0.09, 0.66, 0.24, 0.01], [0.64, 0.24, 0.09, 0.03]],
                    [[0.04, 0.03, 0.55, 0.38], [0.64, 0.24, 0.09, 0.03], [0.29, 0.26, 0.23, 0.22]],
                ]
            ),
            round_summaries=(
                RouteRoundSummary(
                    expert_logits=torch.tensor(
                        [
                            [[4.0, 1.0, 0.5, -1.0], [0.0, 2.0, 1.0, -2.0], [3.0, 2.0, 1.0, 0.0]],
                            [[1.0, 0.5, 4.0, 3.5], [2.0, 1.0, 0.0, -1.0], [0.2, 0.1, 0.0, -0.1]],
                        ]
                    ),
                    expert_weights=torch.tensor(
                        [
                            [[0.87, 0.08, 0.04, 0.01], [0.09, 0.66, 0.24, 0.01], [0.64, 0.24, 0.09, 0.03]],
                            [[0.04, 0.03, 0.55, 0.38], [0.64, 0.24, 0.09, 0.03], [0.29, 0.26, 0.23, 0.22]],
                        ]
                    ),
                ),
            ),
        )
        dispatch_plan = dispatcher.compile(layer_index=0, route_plan=plan)
        self.assertEqual(tuple(dispatch_plan.selected_expert_indices.shape), (2, 3, 2))
        mixed = dispatcher.dispatch(hidden, experts, dispatch_plan)
        self.assertEqual(tuple(mixed.shape), tuple(hidden.shape))

    def test_sparse_top1_dispatch_executes_only_selected_tokens(self) -> None:
        dispatcher = SparseTopKDispatcher(
            ResolvedDispatchContract(
                mode=MiniMoeDispatchMode.SPARSE_TOP_K,
                active_experts_per_token=1,
                execution_strategy=MiniMoeDispatchExecutionStrategy.TOKEN_PACKED_SPARSE,
            )
        )
        experts = torch.nn.ModuleList(
            [RecordingScaleExpert(scale=float(index + 1)) for index in range(4)]
        )
        hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
        route_plan = RoutePlan(
            expert_logits=torch.zeros(2, 3, 4),
            expert_weights=torch.tensor(
                [
                    [[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.1, 0.1, 0.7, 0.1]],
                    [[0.0, 0.2, 0.1, 0.7], [0.8, 0.1, 0.1, 0.0], [0.1, 0.6, 0.1, 0.2]],
                ]
            ),
            round_summaries=(),
        )
        dispatch_plan = dispatcher.compile(layer_index=0, route_plan=route_plan)
        mixed = dispatcher.dispatch(hidden, experts, dispatch_plan)
        expected_scales = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [1.0], [2.0]],
            ]
        )
        self.assertTrue(torch.allclose(mixed, hidden * expected_scales))
        self.assertEqual([expert.last_token_count for expert in experts], [2, 2, 1, 1])

    def test_sparse_topk_dispatch_matches_manual_weighted_mix(self) -> None:
        dispatcher = SparseTopKDispatcher(
            ResolvedDispatchContract(
                mode=MiniMoeDispatchMode.SPARSE_TOP_K,
                active_experts_per_token=2,
                execution_strategy=MiniMoeDispatchExecutionStrategy.TOKEN_PACKED_SPARSE,
            )
        )
        experts = torch.nn.ModuleList(
            [RecordingScaleExpert(scale=float(index + 1)) for index in range(4)]
        )
        hidden = torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2)
        route_plan = RoutePlan(
            expert_logits=torch.zeros(2, 2, 4),
            expert_weights=torch.tensor(
                [
                    [[0.60, 0.30, 0.10, 0.00], [0.10, 0.20, 0.65, 0.05]],
                    [[0.05, 0.55, 0.10, 0.30], [0.25, 0.15, 0.20, 0.40]],
                ]
            ),
            round_summaries=(),
        )
        dispatch_plan = dispatcher.compile(layer_index=0, route_plan=route_plan)
        mixed = dispatcher.dispatch(hidden, experts, dispatch_plan)
        expected = torch.zeros_like(hidden)
        scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for batch_index in range(hidden.shape[0]):
            for token_index in range(hidden.shape[1]):
                token_hidden = hidden[batch_index, token_index]
                token_expected = torch.zeros_like(token_hidden)
                for slot_index in range(dispatch_plan.selected_expert_indices.shape[-1]):
                    expert_index = int(dispatch_plan.selected_expert_indices[batch_index, token_index, slot_index])
                    expert_weight = dispatch_plan.selected_expert_weights[batch_index, token_index, slot_index]
                    token_expected = token_expected + token_hidden * scales[expert_index] * expert_weight
                expected[batch_index, token_index] = token_expected
        self.assertTrue(torch.allclose(mixed, expected))

    def test_observability_sink_finalizes_summary(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_reference_default()
        sink = CollectingMiniMoeObservabilitySink(surface)
        dispatcher = SparseTopKDispatcher(
            ResolvedDispatchContract(
                mode=MiniMoeDispatchMode.SPARSE_TOP_K,
                active_experts_per_token=2,
            )
        )
        route_plan = RoutePlan(
            expert_logits=torch.tensor(
                [
                    [[4.0, 1.0, 0.5, -1.0], [0.0, 2.0, 1.0, -2.0], [3.0, 2.0, 1.0, 0.0]],
                    [[1.0, 0.5, 4.0, 3.5], [2.0, 1.0, 0.0, -1.0], [0.2, 0.1, 0.0, -0.1]],
                ]
            ),
            expert_weights=torch.tensor(
                [
                    [[0.87, 0.08, 0.04, 0.01], [0.09, 0.66, 0.24, 0.01], [0.64, 0.24, 0.09, 0.03]],
                    [[0.04, 0.03, 0.55, 0.38], [0.64, 0.24, 0.09, 0.03], [0.29, 0.26, 0.23, 0.22]],
                ]
            ),
            round_summaries=(
                RouteRoundSummary(
                    expert_logits=torch.tensor(
                        [
                            [[4.0, 1.0, 0.5, -1.0], [0.0, 2.0, 1.0, -2.0], [3.0, 2.0, 1.0, 0.0]],
                            [[1.0, 0.5, 4.0, 3.5], [2.0, 1.0, 0.0, -1.0], [0.2, 0.1, 0.0, -0.1]],
                        ]
                    ),
                    expert_weights=torch.tensor(
                        [
                            [[0.87, 0.08, 0.04, 0.01], [0.09, 0.66, 0.24, 0.01], [0.64, 0.24, 0.09, 0.03]],
                            [[0.04, 0.03, 0.55, 0.38], [0.64, 0.24, 0.09, 0.03], [0.29, 0.26, 0.23, 0.22]],
                        ]
                    ),
                ),
            ),
        )
        sink.record_input_batch(
            torch.tensor(
                [
                    [65, 66, 67],
                    [68, 69, 70],
                ],
                dtype=torch.long,
            )
        )
        dispatch_plan = dispatcher.compile(layer_index=0, route_plan=route_plan)
        sink.record_route_plan(0, route_plan)
        sink.record_dispatch_plan(dispatch_plan)
        summary = sink.finalize()
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.routing.layer_count, 1)
        self.assertEqual(summary.routing.active_expert_count, 2)
        self.assertEqual(len(summary.layers), 1)
        self.assertGreater(len(summary.token_traces), 0)
        trace = summary.token_traces[0]
        self.assertEqual(trace.layer_index, 0)
        self.assertEqual(trace.forward_pass_index, 1)
        self.assertEqual(len(trace.rounds), 1)
        self.assertIn(trace.token_label, {"A", "B", "C", "D", "E", "F"})

    def test_observability_sink_captures_rerouted_token_trace(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_default()
        sink = CollectingMiniMoeObservabilitySink(surface)
        sink.record_input_batch(torch.tensor([[65, 10]], dtype=torch.long))
        route_plan = RoutePlan(
            expert_logits=torch.tensor([[[3.0, 1.0, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
            expert_weights=torch.tensor([[[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
            round_summaries=(
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[3.0, 1.0, 0.0, -1.0], [1.6, 0.9, 0.2, -0.1]]]),
                    expert_weights=torch.tensor([[[0.75, 0.15, 0.05, 0.05], [0.55, 0.25, 0.1, 0.1]]]),
                    applied_token_fraction=1.0,
                ),
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[2.0, 2.8, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
                    expert_weights=torch.tensor([[[0.2, 0.6, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
                    applied_token_fraction=0.5,
                ),
            ),
        )
        sink.record_route_plan(0, route_plan)
        summary = sink.finalize()
        assert summary is not None
        rerouted = [trace for trace in summary.token_traces if trace.rerouted]
        self.assertEqual(len(rerouted), 1)
        trace = rerouted[0]
        self.assertEqual(trace.token_label, "A")
        self.assertEqual(trace.forward_pass_index, 1)
        self.assertEqual(trace.first_winner_expert_id, 0)
        self.assertEqual(trace.final_winner_expert_id, 1)
        self.assertEqual(len(trace.rounds), 2)
        self.assertGreater(trace.total_adjustment_l1, 0.0)
        round_two = [row for row in summary.controller_rounds if row.round_index == 2][0]
        self.assertEqual(round_two.applied_token_fraction, 0.5)

    def test_observability_sink_reduced_mode_matches_summary_metrics(self) -> None:
        traced_surface = replace(
            MiniMoeSurfaceSpec.phase1_recurrent_default(),
            observability=MiniMoeObservabilitySpec(max_token_route_traces_per_layer=4),
        )
        reduced_surface = replace(
            MiniMoeSurfaceSpec.phase1_recurrent_default(),
            observability=MiniMoeObservabilitySpec(max_token_route_traces_per_layer=0),
        )
        traced_sink = CollectingMiniMoeObservabilitySink(traced_surface)
        reduced_sink = CollectingMiniMoeObservabilitySink(reduced_surface)
        dispatcher = SparseTopKDispatcher(
            ResolvedDispatchContract(
                mode=MiniMoeDispatchMode.SPARSE_TOP_K,
                active_experts_per_token=2,
            )
        )
        input_ids = torch.tensor([[65, 10]], dtype=torch.long)
        route_plan = RoutePlan(
            expert_logits=torch.tensor([[[2.0, 2.8, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
            expert_weights=torch.tensor([[[0.2, 0.6, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
            round_summaries=(
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[3.0, 1.0, 0.0, -1.0], [1.6, 0.9, 0.2, -0.1]]]),
                    expert_weights=torch.tensor([[[0.75, 0.15, 0.05, 0.05], [0.55, 0.25, 0.1, 0.1]]]),
                    applied_token_fraction=1.0,
                ),
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[2.0, 2.8, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
                    expert_weights=torch.tensor([[[0.2, 0.6, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
                    applied_token_fraction=0.5,
                ),
            ),
        )
        dispatch_plan = dispatcher.compile(layer_index=0, route_plan=route_plan)
        for sink in (traced_sink, reduced_sink):
            sink.record_input_batch(input_ids)
            sink.record_route_plan(0, route_plan)
            sink.record_dispatch_plan(dispatch_plan)

        traced_summary = traced_sink.finalize()
        reduced_summary = reduced_sink.finalize()
        assert traced_summary is not None
        assert reduced_summary is not None
        self.assertGreater(len(traced_summary.token_traces), 0)
        self.assertEqual(reduced_summary.token_traces, [])
        self.assertEqual(traced_summary.routing.sampled_tokens, reduced_summary.routing.sampled_tokens)
        self.assertEqual(traced_summary.routing.winner_counts, reduced_summary.routing.winner_counts)
        self.assertAlmostEqual(
            traced_summary.routing.mean_route_entropy_bits,
            reduced_summary.routing.mean_route_entropy_bits,
        )
        self.assertAlmostEqual(
            traced_summary.routing.mean_winner_margin,
            reduced_summary.routing.mean_winner_margin,
        )
        self.assertEqual(
            traced_summary.dispatch[0].selected_expert_counts,
            reduced_summary.dispatch[0].selected_expert_counts,
        )
        self.assertAlmostEqual(
            traced_summary.controller_rounds[1].applied_token_fraction,
            reduced_summary.controller_rounds[1].applied_token_fraction,
        )

    def test_render_helpers_include_round_and_token_trace_information(self) -> None:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_default()
        sink = CollectingMiniMoeObservabilitySink(surface)
        sink.record_input_batch(torch.tensor([[65, 10]], dtype=torch.long))
        route_plan = RoutePlan(
            expert_logits=torch.tensor([[[3.0, 1.0, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
            expert_weights=torch.tensor([[[0.1, 0.7, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
            round_summaries=(
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[3.0, 1.0, 0.0, -1.0], [1.6, 0.9, 0.2, -0.1]]]),
                    expert_weights=torch.tensor([[[0.75, 0.15, 0.05, 0.05], [0.55, 0.25, 0.1, 0.1]]]),
                    applied_token_fraction=1.0,
                ),
                RouteRoundSummary(
                    expert_logits=torch.tensor([[[2.0, 2.8, 0.0, -1.0], [1.4, 0.8, 0.0, -0.2]]]),
                    expert_weights=torch.tensor([[[0.2, 0.6, 0.1, 0.1], [0.6, 0.2, 0.1, 0.1]]]),
                    applied_token_fraction=0.5,
                ),
            ),
        )
        sink.record_route_plan(0, route_plan)
        summary = sink.finalize()
        assert summary is not None
        report = BenchmarkReport(
            model_label="mini_moe_test",
            implementation_kind="python_native",
            note="",
            config={},
            corpus={},
            initial_eval=EvalSummary(batch_count=1, mean_loss=1.0, perplexity=2.0),
            final_eval=EvalSummary(batch_count=1, mean_loss=0.5, perplexity=1.5),
            runtime=RuntimeSummary(
                total_wall_time_ms=1.0,
                initial_eval_wall_time_ms=0.1,
                train_wall_time_ms=0.7,
                final_eval_wall_time_ms=0.2,
                train_tokens_seen=64,
                eval_tokens_per_pass=32,
                train_tokens_per_second=100.0,
                overall_tokens_per_second=90.0,
                process_memory_metric="rss",
                peak_process_memory_bytes=1024,
                peak_process_memory_delta_bytes=512,
                cuda_device_memory=None,
                memory_note="",
            ),
            train_steps=[TrainStepRecord(step=1, learning_rate=1.0e-3, train_loss=1.0, train_perplexity=2.0, seen_tokens=32)],
            mini_moe_summary=summary,
        )
        round_table = render_mini_moe_round_transition_table(report, round_index=2)
        token_table = render_mini_moe_token_trace_table(report, limit_per_layer=2)
        self.assertIn("layer=0", round_table)
        self.assertIn("rerouted=", round_table)
        self.assertIn("applied=0.5000", round_table)
        self.assertIn("token=A", token_table)
        self.assertIn("pass=1", token_table)
        self.assertIn("r2:e1", token_table)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            write_report(report, path)
            loaded = read_report(path)
        self.assertIsNotNone(loaded.mini_moe_summary)
        assert loaded.mini_moe_summary is not None
        self.assertEqual(len(loaded.mini_moe_summary.token_traces), len(summary.token_traces))

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

    def test_mini_moe_backbone_recurrent_forward_cpu(self) -> None:
        surface = MiniMoeSurfaceSpec(
            architecture=MiniMoeArchitectureSpec(
                schema_version=1,
                preset=None,
                label="phase1-mini-moe-recurrent-r2-s16",
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
                        kind=MiniMoeLayerScheduleKind.ALL_LAYERS
                    ),
                    expert_ffn_multiplier=4,
                    load_balance_loss_weight=0.01,
                ),
                router=MiniMoeRouterSpec(
                    kind="recurrent_pre_expert",
                    recurrent_pre_expert=RecurrentPreExpertRouterSpec(
                        round_count=2,
                        state_dim=16,
                        gate=RecurrentRoundGateSpec(),
                        execution_strategy=RecurrentRoundExecutionStrategy.DENSE_BLEND,
                    ),
                ),
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
