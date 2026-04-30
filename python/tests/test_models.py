from __future__ import annotations

import importlib.util
from contextlib import contextmanager
import unittest
from unittest import mock

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
from python.models.reference_ssm import GdnpFusedSequenceMixer, resolve_reference_ssm_config
from python.models.transformer import LocalCausalSelfAttention, local_causal_attention_bias
from python.models.transformer import LocalCausalTransformerBlock, Pr5LocalCausalTransformerBlock
from python.runtime import apply_runtime_policy
from python.runtime.recurrent import PackedLinearProjection
from python.specs.common import DeviceRuntimeSpec
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
    AttentionKernelProfile,
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

    def test_path1_forward_exposes_lm_head_timing_regions(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        region_names: list[str] = []

        @contextmanager
        def record_region(name: str):
            region_names.append(name)
            yield

        with mock.patch("python.models.path1.timed_region", record_region):
            logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIn("path1.lm_head.total", region_names)
        self.assertIn("path1.lm_head.final_norm", region_names)
        self.assertIn("path1.lm_head.output_projection", region_names)

    def test_path1_forward_loss_matches_forward_logits_cross_entropy(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=2, ffn_multiplier=2)
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        target_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        target_ids[0, 0] = 0

        logits = model.forward_logits(input_ids)
        expected = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=0,
        )
        actual = model.forward_loss(input_ids, target_ids, pad_token=0)

        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6))

    def test_path1_runtime_policy_configures_head_loss_backend_without_model_compile(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")

        apply_runtime_policy(
            model,
            DeviceRuntimeSpec(
                backend="cpu",
                dtype="fp32",
                primitive_runtime_backend="torch",
                head_loss_backend="compiled",
            ),
        )

        self.assertEqual(model.diagnostic_payload()["head_loss_backend"], "compiled")

    def test_path1_streaming_head_loss_backend_fails_without_registered_kernel(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")
        model.configure_runtime_policy(
            compile_mode=None,
            primitive_runtime_backend="torch",
            head_loss_backend="streaming-kernel",
        )
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        target_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        self.assertEqual(model.diagnostic_payload()["head_loss_backend"], "streaming-kernel")
        with self.assertRaisesRegex(RuntimeError, "no streaming LM-head cross-entropy kernel"):
            model.forward_loss(input_ids, target_ids, pad_token=0)

    def test_path1_runtime_policy_configures_transformer_ffn_backend(self) -> None:
        model = build_path1_model(phase1_attention_only_variant(), dtype_mode="fp32")

        with mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn):
            apply_runtime_policy(
                model,
                DeviceRuntimeSpec(
                    backend="cpu",
                    dtype="fp32",
                    primitive_runtime_backend="torch",
                    ffn_backend="compiled",
                ),
            )

        self.assertEqual(model.diagnostic_payload()["ffn_backend"], "compiled")
        self.assertTrue(
            all(
                getattr(block, "_ffn_backend", None) == "compiled"
                for block in model.blocks
                if isinstance(block, LocalCausalTransformerBlock)
            )
        )

    def test_manual_autograd_transformer_ffn_backend_matches_dense_forward_and_backward(self) -> None:
        dense = LocalCausalTransformerBlock(16, 4, 32)
        manual = LocalCausalTransformerBlock(16, 4, 32)
        manual.load_state_dict(dense.state_dict())
        hidden_dense = torch.randn(2, 5, 16, requires_grad=True)
        hidden_manual = hidden_dense.detach().clone().requires_grad_(True)
        manual.configure_runtime_policy(compile_mode=None, ffn_backend="manual-autograd")

        dense_out = dense(hidden_dense)
        manual_out = manual(hidden_manual)
        self.assertTrue(torch.allclose(manual_out, dense_out, atol=1.0e-5, rtol=1.0e-5))

        dense_loss = dense_out.square().mean()
        manual_loss = manual_out.square().mean()
        dense_loss.backward()
        manual_loss.backward()

        self.assertTrue(torch.allclose(hidden_manual.grad, hidden_dense.grad, atol=1.0e-5, rtol=1.0e-5))
        for (dense_name, dense_parameter), (manual_name, manual_parameter) in zip(
            dense.named_parameters(),
            manual.named_parameters(),
            strict=True,
        ):
            self.assertEqual(manual_name, dense_name)
            self.assertIsNotNone(dense_parameter.grad, dense_name)
            self.assertIsNotNone(manual_parameter.grad, manual_name)
            self.assertTrue(
                torch.allclose(manual_parameter.grad, dense_parameter.grad, atol=1.0e-5, rtol=1.0e-5),
                f"gradient mismatch for {manual_name}",
            )

    def test_recompute_transformer_ffn_backend_matches_dense_forward_and_backward(self) -> None:
        dense = LocalCausalTransformerBlock(16, 4, 32)
        recompute = LocalCausalTransformerBlock(16, 4, 32)
        recompute.load_state_dict(dense.state_dict())
        hidden_dense = torch.randn(2, 5, 16, requires_grad=True)
        hidden_recompute = hidden_dense.detach().clone().requires_grad_(True)
        recompute.configure_runtime_policy(compile_mode=None, ffn_backend="recompute")

        dense_out = dense(hidden_dense)
        recompute_out = recompute(hidden_recompute)
        self.assertTrue(torch.allclose(recompute_out, dense_out, atol=1.0e-5, rtol=1.0e-5))

        dense_loss = dense_out.square().mean()
        recompute_loss = recompute_out.square().mean()
        dense_loss.backward()
        recompute_loss.backward()

        self.assertTrue(torch.allclose(hidden_recompute.grad, hidden_dense.grad, atol=1.0e-5, rtol=1.0e-5))
        for (dense_name, dense_parameter), (recompute_name, recompute_parameter) in zip(
            dense.named_parameters(),
            recompute.named_parameters(),
            strict=True,
        ):
            self.assertEqual(recompute_name, dense_name)
            self.assertIsNotNone(dense_parameter.grad, dense_name)
            self.assertIsNotNone(recompute_parameter.grad, recompute_name)
            self.assertTrue(
                torch.allclose(recompute_parameter.grad, dense_parameter.grad, atol=1.0e-5, rtol=1.0e-5),
                f"gradient mismatch for {recompute_name}",
            )

    def test_recompute_transformer_ffn_backend_supports_bf16_autocast_backward(self) -> None:
        block = LocalCausalTransformerBlock(16, 4, 32)
        block.configure_runtime_policy(compile_mode=None, ffn_backend="recompute")
        hidden = torch.randn(2, 5, 16, requires_grad=True)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = block(hidden)
            loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_pr5_transformer_manual_autograd_ffn_backend_fails_without_registered_path(self) -> None:
        block = Pr5LocalCausalTransformerBlock(d_model=32, head_count=4, d_ff=64)
        block.configure_runtime_policy(compile_mode=None, ffn_backend="manual-autograd")
        hidden = torch.randn(2, 8, 32)

        with self.assertRaisesRegex(RuntimeError, "no manual-autograd FFN path"):
            block(hidden)

    def test_compiled_transformer_ffn_backend_matches_dense_path(self) -> None:
        dense = LocalCausalTransformerBlock(16, 4, 32)
        compiled = LocalCausalTransformerBlock(16, 4, 32)
        compiled.load_state_dict(dense.state_dict())
        hidden = torch.randn(2, 5, 16)

        with mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn):
            compiled.configure_runtime_policy(compile_mode=None, ffn_backend="compiled")

        self.assertTrue(torch.allclose(compiled(hidden), dense(hidden), atol=1.0e-6, rtol=1.0e-6))

    def test_triton_gelu_transformer_ffn_backend_routes_activation(self) -> None:
        block = LocalCausalTransformerBlock(16, 4, 32)

        class FakeTritonBackend:
            def __init__(self) -> None:
                self.gelu_calls = 0

            def gelu(self, inputs: torch.Tensor) -> torch.Tensor:
                self.gelu_calls += 1
                return F.gelu(inputs)

        fake_backend = FakeTritonBackend()
        with (
            mock.patch("python.models.transformer.ensure_triton_runtime_available"),
            mock.patch("python.models.transformer.build_triton_primitive_backend", return_value=fake_backend),
        ):
            block.configure_runtime_policy(compile_mode=None, ffn_backend="triton-gelu")

        hidden = torch.randn(2, 5, 16)
        output = block(hidden)

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertEqual(fake_backend.gelu_calls, 1)

    def test_compiled_transformer_full_block_backend_matches_dense_path(self) -> None:
        dense = LocalCausalTransformerBlock(16, 4, 32)
        compiled = LocalCausalTransformerBlock(16, 4, 32)
        compiled.load_state_dict(dense.state_dict())
        hidden = torch.randn(2, 5, 16)

        with mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn):
            compiled.configure_full_block_compile(compile_mode="reduce-overhead")

        self.assertIsNotNone(compiled._compiled_full_block_impl)
        self.assertTrue(torch.allclose(compiled(hidden), dense(hidden), atol=1.0e-6, rtol=1.0e-6))

        compiled.configure_full_block_compile(enabled=False)
        self.assertIsNone(compiled._compiled_full_block_impl)

    def test_attention_only_learned_position_embeddings_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=2, ffn_multiplier=2),
            position_encoding_kind="learned",
            max_position_embeddings=16,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["position_encoding_kind"], "learned")
        self.assertEqual(diagnostics["attention_position_contract"], "shared-input")
        self.assertEqual(diagnostics["attention_position_embedding_widths"], [])
        self.assertEqual(diagnostics["max_position_embeddings"], 16)

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
                if scaffold_profile is Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION:
                    self.assertIsNotNone(model.parcae_p20_control_projection)
                    self.assertEqual(diagnostics["p20_control_projection"], "packed-value-gate")

    def test_attention_only_parcae_cuda_parity_knobs_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_discretization="stable-exp",
            parcae_control_stride=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["backward_steps"], 1)
        self.assertEqual(diagnostics["prelude_norm_kind"], "rmsnorm")
        self.assertEqual(diagnostics["discretization"], "stable-exp")
        self.assertEqual(diagnostics["control_stride"], 2)
        self.assertEqual(diagnostics["last_p20_control_steps"], 4)

    def test_attention_only_parcae_truncated_loop_steps_run_without_autograd(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        recurrent_block = model.blocks[model.parcae_loop_start]
        original_forward = recurrent_block.forward
        grad_modes: list[bool] = []

        def wrapped_forward(*args, **kwargs):
            grad_modes.append(torch.is_grad_enabled())
            return original_forward(*args, **kwargs)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        with mock.patch.object(recurrent_block, "forward", side_effect=wrapped_forward):
            logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual(grad_modes, [False, True])

    def test_attention_only_parcae_p20_control_receives_runtime_policy(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        self.assertIsNotNone(model.parcae_p20_controller)
        self.assertIsNone(model.parcae_p20_controller._compiled_scan_impl)
        self.assertIsNone(model._compiled_parcae_p20_post_scan_injection_impl)

        model.configure_runtime_policy(
            compile_mode="reduce-overhead",
            primitive_runtime_backend="torch",
        )

        self.assertIsNotNone(model.parcae_p20_controller._compiled_scan_impl)
        self.assertIsNotNone(model._compiled_parcae_p20_post_scan_injection_impl)

    def test_attention_only_parcae_p20_control_keeps_state_and_residual_mix_eager_with_compiled_ffn_backend(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        with (
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        self.assertIsNotNone(model._compiled_parcae_p20_post_scan_injection_impl)
        self.assertIsNotNone(model._compiled_parcae_loop_input_projection_impl)
        self.assertIsNotNone(model._compiled_parcae_loop_output_projection_impl)
        self.assertIsNone(model._compiled_parcae_state_mix_impl)
        self.assertIsNone(model._compiled_parcae_residual_mix_impl)
        self.assertIsNone(model._compiled_parcae_loop_iteration_impl)
        self.assertFalse(model.parcae_runtime_diagnostics)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(diagnostics["recurrent_block_backend"], "compiled-full-block")
        self.assertEqual(diagnostics["loop_projection_backend"], "compiled")
        self.assertEqual(diagnostics["recurrent_compile_mode"], "reduce-overhead")
        self.assertEqual(diagnostics["loop_update_backend"], "eager")
        self.assertFalse(diagnostics["runtime_diagnostics"])
        for layer_index, block in enumerate(model.blocks):
            if model.parcae_loop_start <= layer_index < model.parcae_loop_end:
                self.assertIsNotNone(block._compiled_full_block_impl)
            else:
                self.assertIsNone(block._compiled_full_block_impl)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertIsNone(diagnostics["last_injection_gate_mean"])
        self.assertIsNone(diagnostics["last_injection_norm"])
        self.assertIsNone(diagnostics["last_p20_control_norm"])
        self.assertEqual(diagnostics["last_recurrent_state_norms"], [])

    def test_attention_only_parcae_p20_control_can_compile_loop_iteration_as_opt_in(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32,
                head_count=4,
                total_layers=6,
                ffn_multiplier=2,
                local_window=4,
                attention_kernel=AttentionKernelProfile.FLEX_LOCAL,
            ),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_update_backend="compiled",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        with (
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertIsNotNone(model._compiled_parcae_loop_iteration_impl)
        self.assertEqual(diagnostics["loop_update_backend"], "compiled")
        self.assertEqual(diagnostics["recurrent_block_backend"], "standard")
        for block in model.blocks[model.parcae_loop_start : model.parcae_loop_end]:
            self.assertIsNone(block._compiled_full_block_impl)

        recurrent_blocks = model.blocks[model.parcae_loop_start : model.parcae_loop_end]
        flex_mask_mocks = []
        full_block_mocks = []
        for block in model.blocks:
            if block not in recurrent_blocks:
                block.forward = mock.Mock(side_effect=lambda hidden, *_args, **_kwargs: hidden)  # type: ignore[method-assign]
                continue
            flex_mask_mock = mock.Mock(return_value=object())
            full_block_mock = mock.Mock(side_effect=lambda hidden, *_args: hidden + 0.01)
            block._full_block_flex_block_mask = flex_mask_mock  # type: ignore[method-assign]
            block._full_block_impl_no_timing = full_block_mock  # type: ignore[method-assign]
            flex_mask_mocks.append(flex_mask_mock)
            full_block_mocks.append(full_block_mock)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual([mask_mock.call_count for mask_mock in flex_mask_mocks], [1, 1])
        self.assertEqual([block_mock.call_count for block_mock in full_block_mocks], [2, 2])

    def test_attention_only_parcae_p20_control_manual_autograd_loop_update_preserves_full_block_compile(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_update_backend="manual-autograd",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        with (
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(diagnostics["loop_update_backend"], "manual-autograd")
        self.assertEqual(diagnostics["recurrent_block_backend"], "compiled-full-block")
        self.assertIsNone(model._compiled_parcae_loop_iteration_impl)
        for block in model.blocks[model.parcae_loop_start : model.parcae_loop_end]:
            self.assertIsNotNone(block._compiled_full_block_impl)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        loss = model.forward_loss(input_ids, input_ids, pad_token=-100)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    def test_attention_only_parcae_p20_control_lean_eager_precomputes_loop_context(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(
                d_model=32,
                head_count=4,
                total_layers=6,
                ffn_multiplier=2,
                local_window=4,
                attention_kernel=AttentionKernelProfile.FLEX_LOCAL,
            ),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_update_backend="lean-eager",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        with (
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(diagnostics["loop_update_backend"], "lean-eager")
        self.assertEqual(diagnostics["recurrent_block_backend"], "compiled-full-block")

        recurrent_blocks = model.blocks[model.parcae_loop_start : model.parcae_loop_end]
        flex_mask_mocks = []
        compiled_block_mocks = []
        for block in model.blocks:
            if block not in recurrent_blocks:
                block.forward = mock.Mock(side_effect=lambda hidden, *_args, **_kwargs: hidden)  # type: ignore[method-assign]
                continue
            flex_mask_mock = mock.Mock(return_value=object())
            compiled_block_mock = mock.Mock(side_effect=lambda hidden, *_args: hidden + 0.01)
            block._full_block_flex_block_mask = flex_mask_mock  # type: ignore[method-assign]
            block._compiled_full_block_impl = compiled_block_mock
            flex_mask_mocks.append(flex_mask_mock)
            compiled_block_mocks.append(compiled_block_mock)

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual([mask_mock.call_count for mask_mock in flex_mask_mocks], [1, 1])
        self.assertEqual([block_mock.call_count for block_mock in compiled_block_mocks], [2, 2])

    def test_attention_only_parcae_p20_control_triton_glue_routes_loop_update_backend(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_update_backend="triton-glue",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        class FakeTritonBackend:
            def __init__(self) -> None:
                self.state_mix_calls = 0
                self.residual_mix_calls = 0
                self.output_mix_calls = 0

            def parcae_state_mix(self, state, decay, injection):
                self.state_mix_calls += 1
                return decay * state + injection

            def parcae_residual_mix(self, mixed, block_out, nonlinear):
                self.residual_mix_calls += 1
                return mixed + nonlinear * (block_out - mixed)

            def parcae_output_mix(self, anchor, delta, gate):
                self.output_mix_calls += 1
                return anchor + gate * delta

        fake_backend = FakeTritonBackend()

        with (
            mock.patch("python.models.path1.ensure_triton_runtime_available"),
            mock.patch("python.models.path1.build_triton_primitive_backend", return_value=fake_backend),
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(diagnostics["loop_update_backend"], "triton-glue")
        self.assertTrue(diagnostics["loop_update_triton"])
        self.assertEqual(diagnostics["recurrent_block_backend"], "compiled-full-block")

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual(fake_backend.state_mix_calls, 2)
        self.assertEqual(fake_backend.residual_mix_calls, 4)
        self.assertEqual(fake_backend.output_mix_calls, 0)

    def test_attention_only_parcae_p20_control_triton_loop_forward_routes_first_block_update(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_update_backend="triton-loop-forward",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        class FakeTritonBackend:
            def __init__(self) -> None:
                self.state_mix_calls = 0
                self.residual_mix_calls = 0
                self.loop_update_calls = 0
                self.output_mix_calls = 0

            def parcae_state_mix(self, state, decay, injection):
                self.state_mix_calls += 1
                return decay * state + injection

            def parcae_residual_mix(self, mixed, block_out, nonlinear):
                self.residual_mix_calls += 1
                return mixed + nonlinear * (block_out - mixed)

            def parcae_loop_update(self, state, decay, injection, block_out, nonlinear):
                self.loop_update_calls += 1
                mixed = decay * state + injection
                return mixed + nonlinear * (block_out - mixed)

            def parcae_output_mix(self, anchor, delta, gate):
                self.output_mix_calls += 1
                return anchor + gate * delta

        fake_backend = FakeTritonBackend()

        with (
            mock.patch("python.models.path1.ensure_triton_runtime_available"),
            mock.patch("python.models.path1.build_triton_primitive_backend", return_value=fake_backend),
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(diagnostics["loop_update_backend"], "triton-loop-forward")
        self.assertTrue(diagnostics["loop_update_triton"])
        self.assertEqual(diagnostics["recurrent_block_backend"], "compiled-full-block")

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual(fake_backend.state_mix_calls, 2)
        self.assertEqual(fake_backend.loop_update_calls, 2)
        self.assertEqual(fake_backend.residual_mix_calls, 2)
        self.assertEqual(fake_backend.output_mix_calls, 0)

    def test_parcae_loop_controls_normalize_to_loop_state_dtype(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_d_model=16,
            parcae_loop_head_count=4,
            parcae_loop_update_backend="triton-loop-forward",
            parcae_band_prepare_backend="compiled",
        )
        model = build_path1_model(variant, dtype_mode="bf16")
        loop_input = torch.randn(2, 8, 16, dtype=torch.bfloat16)
        decay = torch.ones(1, 1, 16, dtype=torch.float32)
        injection = torch.randn(2, 8, 16, dtype=torch.float32)
        nonlinear = torch.zeros(1, 1, 16, dtype=torch.float32)

        normalized = model._normalize_parcae_loop_controls(
            loop_input,
            decay,
            injection,
            nonlinear,
        )

        for tensor in normalized:
            self.assertEqual(tensor.dtype, loop_input.dtype)
            self.assertEqual(tensor.device, loop_input.device)
            self.assertTrue(tensor.is_contiguous())

    def test_attention_only_parcae_hourglass_triton_loop_forward_routes_output_mix(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=48, head_count=6, total_layers=3, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=1,
            parcae_hourglass_band_schedule=(1, 1, 1),
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            parcae_control_position_kind="learned",
            parcae_loop_update_backend="triton-loop-forward",
            parcae_band_block_contract="compiled-direct",
            parcae_output_mix_backend="triton",
        )
        model = build_path1_model(variant, dtype_mode="fp32")

        class FakeTritonBackend:
            def __init__(self) -> None:
                self.state_mix_calls = 0
                self.residual_mix_calls = 0
                self.loop_update_calls = 0
                self.output_mix_calls = 0

            def parcae_state_mix(self, state, decay, injection):
                self.state_mix_calls += 1
                return decay * state + injection

            def parcae_residual_mix(self, mixed, block_out, nonlinear):
                self.residual_mix_calls += 1
                return mixed + nonlinear * (block_out - mixed)

            def parcae_loop_update(self, state, decay, injection, block_out, nonlinear):
                self.loop_update_calls += 1
                mixed = decay * state + injection
                return mixed + nonlinear * (block_out - mixed)

            def parcae_output_mix(self, anchor, delta, gate):
                self.output_mix_calls += 1
                return anchor + gate * delta

        fake_backend = FakeTritonBackend()

        with (
            mock.patch("python.models.path1.ensure_triton_runtime_available"),
            mock.patch("python.models.path1.build_triton_primitive_backend", return_value=fake_backend),
            mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn),
            mock.patch("python.models.transformer.torch.compile", side_effect=lambda fn, mode=None: fn),
        ):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertEqual(fake_backend.output_mix_calls, 1)

    def test_attention_only_parcae_recurrent_compile_mode_reaches_full_block_compile(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_recurrent_compile_mode="max-autotune",
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        compile_modes: list[str] = []

        def fake_configure(*, enabled=True, compile_mode="reduce-overhead"):
            self.assertTrue(enabled)
            compile_modes.append(compile_mode)

        for block in model.blocks[model.parcae_loop_start : model.parcae_loop_end]:
            block.configure_full_block_compile = fake_configure  # type: ignore[method-assign]

        with mock.patch("python.models.path1.torch.compile", side_effect=lambda fn, mode=None: fn):
            model.configure_runtime_policy(
                compile_mode=None,
                primitive_runtime_backend="torch",
                ffn_backend="compiled",
            )

        self.assertEqual(compile_modes, ["max-autotune", "max-autotune"])
        self.assertEqual(
            model.diagnostic_payload()["parcae_looped_attention"]["recurrent_compile_mode"],
            "max-autotune",
        )

    def test_attention_only_parcae_p20_control_can_freeze_identity_state_transform(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_control_state_transform="frozen-identity",
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.parcae_p20_controller)
        self.assertTrue(model.parcae_p20_controller._triton_identity_state_transform)
        projection = model.parcae_p20_controller.state_transform_projection

        for parameter in projection.parameters():
            self.assertFalse(parameter.requires_grad)
        if projection.weight.ndim == 3:
            identity = torch.eye(projection.weight.shape[-1]).view(1, projection.weight.shape[-1], projection.weight.shape[-1])
            self.assertTrue(torch.allclose(projection.weight, identity.expand_as(projection.weight)))
        else:
            self.assertTrue(torch.allclose(projection.weight, torch.eye(projection.weight.shape[0])))
        self.assertTrue(torch.allclose(projection.bias, torch.zeros_like(projection.bias)))

        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["control_state_transform"], "frozen-identity")

    def test_attention_only_parcae_p20_control_can_use_block_diagonal_8_state_transform(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_control_state_transform="trainable-block-diagonal-8",
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        self.assertIsNotNone(model.parcae_p20_controller)
        projection = model.parcae_p20_controller.state_transform_projection

        self.assertEqual(projection.blocks, 8)
        self.assertEqual(projection.block_width, 4)
        self.assertFalse(model.parcae_p20_controller._triton_identity_state_transform)
        self.assertEqual(
            model.diagnostic_payload()["parcae_looped_attention"]["control_state_transform"],
            "trainable-block-diagonal-8",
        )

    def test_attention_only_parcae_hourglass_p20_control_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_hourglass_pass_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(diagnostics["hourglass"])
        self.assertEqual(diagnostics["hourglass_pass_count"], 2)
        self.assertEqual(diagnostics["wide_d_model"], 64)
        self.assertEqual(diagnostics["loop_d_model"], 32)
        self.assertEqual(diagnostics["prelude_layers"], 2)
        self.assertEqual(diagnostics["recurrent_layers"], 2)
        self.assertEqual(diagnostics["coda_layers"], 2)
        self.assertIsNotNone(diagnostics["last_p20_control_norm"])

    def test_attention_only_parcae_hourglass_band_schedule_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=12, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=1,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            parcae_hourglass_band_schedule=(3, 2, 3, 2, 2),
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["hourglass_band_schedule"], (3, 2, 3, 2, 2))
        self.assertEqual(diagnostics["loop_ranges"], ((3, 5), (8, 10)))
        self.assertEqual(diagnostics["recurrent_layers_total"], 4)
        self.assertEqual(diagnostics["last_p20_control_steps"], 8)
        self.assertEqual(model.blocks[3].input_norm.normalized_shape, (32,))
        self.assertEqual(model.blocks[8].input_norm.normalized_shape, (32,))
        self.assertEqual(model.blocks[5].input_norm.normalized_shape, (64,))

    def test_attention_only_parcae_hourglass_loop_layer_count_forward_cpu(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            parcae_loop_layer_count=1,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["configured_loop_layer_count"], 1)
        self.assertEqual(diagnostics["prelude_layers"], 2)
        self.assertEqual(diagnostics["recurrent_layers"], 1)
        self.assertEqual(diagnostics["coda_layers"], 3)

    def test_attention_only_parcae_p20_control_can_use_position_features(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            position_encoding_kind="learned",
            max_position_embeddings=16,
            parcae_control_position_kind="learned",
            parcae_control_position_scale_init=0.02,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(model.parcae_p20_position_embedding)
        self.assertEqual(diagnostics["parcae_control_position_kind"], "learned")
        self.assertAlmostEqual(diagnostics["parcae_control_position_scale"], 0.02, places=6)

    def test_attention_only_parcae_p20_control_stride_uses_causal_left_anchors(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            max_position_embeddings=16,
            parcae_control_position_kind="learned",
            parcae_control_position_scale_init=0.02,
            parcae_control_stride=4,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 10), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()
        parcae_diagnostics = diagnostics["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 10, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(diagnostics["parcae_control_stride"], 4)
        self.assertEqual(parcae_diagnostics["control_stride"], 4)
        self.assertEqual(parcae_diagnostics["last_p20_control_steps"], 3)

    def test_parcae_forward_defers_diagnostic_item_syncs(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=32, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)
        original_item = torch.Tensor.item

        def forbid_item(tensor):
            del tensor
            raise AssertionError("Tensor.item() should not run inside Parcae forward")

        with unittest.mock.patch.object(torch.Tensor, "item", forbid_item):
            logits = model.forward_logits(input_ids)

        diagnostics = model.diagnostic_payload()["parcae_looped_attention"]
        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIs(torch.Tensor.item, original_item)
        self.assertIsNotNone(diagnostics["last_injection_gate_mean"])
        self.assertEqual(len(diagnostics["last_recurrent_state_norms"]), 2)

    def test_attention_only_contract_uses_attention_local_position_tables(self) -> None:
        variant = phase1_attention_only_variant(
            shape=Path1ModelShape(d_model=64, head_count=4, total_layers=6, ffn_multiplier=2),
            scaffold_profile=Path1ScaffoldProfile.PARCAE_HOURGLASS_P20_CONTROL_LOOPED_ATTENTION,
            parcae_loop_count=2,
            parcae_backward_steps=1,
            parcae_prelude_norm_kind="rmsnorm",
            parcae_loop_d_model=32,
            parcae_loop_head_count=4,
            parcae_loop_ffn_multiplier=2,
            position_encoding_kind="learned",
            attention_position_contract="attention-only",
            max_position_embeddings=16,
            parcae_control_position_kind="learned",
            parcae_control_position_scale_init=0.02,
        )
        model = build_path1_model(variant, dtype_mode="fp32")
        input_ids = torch.randint(low=0, high=257, size=(2, 8), dtype=torch.long)

        logits = model.forward_logits(input_ids)
        diagnostics = model.diagnostic_payload()
        parcae_diagnostics = diagnostics["parcae_looped_attention"]

        self.assertEqual(tuple(logits.shape), (2, 8, 257))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNone(model.position_embedding)
        self.assertEqual(sorted(int(key) for key in model.attention_position_embeddings.keys()), [32, 64])
        self.assertEqual(diagnostics["attention_position_contract"], "attention-only")
        self.assertEqual(diagnostics["attention_position_embedding_widths"], [32, 64])
        self.assertEqual(parcae_diagnostics["attention_position_contract"], "attention-only")

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
                identity_transform: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.identity_transform = identity_transform
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

    def test_flex_local_attention_contract_is_explicit_cuda_local_kernel(self) -> None:
        attention = LocalCausalSelfAttention(
            d_model=16,
            head_count=4,
            local_window=4,
            attention_kernel=AttentionKernelProfile.FLEX_LOCAL,
        )
        hidden = torch.randn(2, 6, 16)
        mask = local_causal_attention_bias(
            seq_len=hidden.shape[1],
            local_window=4,
            device=hidden.device,
            dtype=hidden.dtype,
        )

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            attention(hidden, mask)

    def test_flex_local_attention_rejects_non_power_of_two_head_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "power-of-two head_dim"):
            LocalCausalSelfAttention(
                d_model=480,
                head_count=10,
                local_window=128,
                attention_kernel=AttentionKernelProfile.FLEX_LOCAL,
            )

    def test_flash_local_attention_allows_head_dim_48_contract(self) -> None:
        attention = LocalCausalSelfAttention(
            d_model=480,
            head_count=10,
            local_window=128,
            attention_kernel=AttentionKernelProfile.FLASH_LOCAL,
        )

        self.assertEqual(attention.head_dim, 48)
        self.assertEqual(attention.attention_kernel, AttentionKernelProfile.FLASH_LOCAL)

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
                identity_transform: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.identity_transform = identity_transform
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
        self.assertFalse(fake_backend.identity_transform)
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
                identity_transform: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.identity_transform = identity_transform
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
        self.assertFalse(fake_backend.identity_transform)
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

    def test_runtime_p20_block_diagonal_8_triton_routes_to_sequence_scan(self) -> None:
        runtime = build_sequence_primitive(
            PrimitiveProfile.P20,
            16,
            PrimitiveExecutionProfile.RUNTIME,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_8,
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
                identity_transform: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.sequence_calls += 1
                self.identity_transform = identity_transform
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
                    torch.full_like(update_gate, 9.0),
                    torch.full_like(initial_state, 10.0),
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
        self.assertFalse(fake_backend.identity_transform)
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
                (8, 2, 2),
                (16,),
            ),
        )
        self.assertTrue(torch.allclose(result.emitted_outputs, torch.full_like(result.emitted_outputs, 9.0)))
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(result.final_state, 10.0)))

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
