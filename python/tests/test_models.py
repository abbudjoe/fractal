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
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
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
