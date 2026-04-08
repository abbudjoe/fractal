from __future__ import annotations

import importlib.util
import unittest

import torch

from python.models.mini_moe import MiniMoeBackboneModel
from python.models.path1 import build_path1_model
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
    PrimitiveNormMode,
    PrimitiveProfile,
    PrimitiveReadoutMode,
    PrimitiveResidualMode,
    PrimitiveWrapperMode,
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


class MiniMoeModelTests(unittest.TestCase):
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
