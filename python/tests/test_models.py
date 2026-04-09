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
from python.models.primitives import build_sequence_primitive
from python.models.path1 import build_path1_model
from python.models.reference_ssm import resolve_reference_ssm_config
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
