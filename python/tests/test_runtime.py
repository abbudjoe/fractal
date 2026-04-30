from __future__ import annotations

import inspect
import os
import unittest
from contextlib import contextmanager
import threading

import torch
import torch.nn as nn

from python.runtime.compilation import apply_runtime_policy
from python.runtime.cuda_timing import timed_region, use_cuda_timing, use_named_timing_regions
from python.runtime.cuda_setup_patch import patch_cuda_setup_text
from python.runtime.parcae_loop_region import (
    PARCAE_LOOP_REGION_TIMING_NAMES,
    ParcaeLoopRegionConfig,
    ParcaeLoopRegionControls,
    ParcaeLoopRegionKernels,
    ParcaeLoopRegionTensorLayout,
    run_parcae_loop_region,
)
from python.runtime.triton_primitives import (
    _P20BlockDiagonalSequenceScan,
    _p20_atomic_transform_grad_enabled,
    _sum_to_broadcast_owner,
    _triton_width_owner_reduce_supported,
)
from python.runtime.recurrent import (
    BlockDiagonalLinear,
    eggroll_update_matrix,
    make_eggroll_factors,
    materialized_eggroll_linear,
    PackedLinearProjection,
    build_state_transform_projection,
    packed_linear_chunks,
    virtual_eggroll_linear,
)
from python.runtime.train_eval import perplexity_from_loss
from python.specs.common import DeviceRuntimeSpec, ValidationError
from python.specs.runtime import PrimitiveStateTransformMode


class CudaSetupPatchTests(unittest.TestCase):
    def test_patches_causal_conv1d_arch_block(self) -> None:
        original = """
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
""".lstrip(
            "\n"
        )
        patched = patch_cuda_setup_text(original, "8.9")
        self.assertIn('cc_flag.append("arch=compute_89,code=sm_89")', patched)
        self.assertNotIn("arch=compute_75,code=sm_75", patched)
        self.assertIn("# HACK:", patched)

    def test_patches_causal_conv1d_arch_block_with_ptx_fallback(self) -> None:
        original = """
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
""".lstrip(
            "\n"
        )
        patched = patch_cuda_setup_text(original, "9.0+PTX")
        self.assertIn('cc_flag.append("arch=compute_90,code=sm_90")', patched)
        self.assertIn('cc_flag.append("arch=compute_90,code=compute_90")', patched)
        self.assertNotIn("arch=compute_80,code=sm_80", patched)
        self.assertIn("# HACK:", patched)

    def test_raises_when_arch_block_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed to locate CUDA arch block"):
            patch_cuda_setup_text("print('no cuda block here')\n", "8.9")


class ParcaeLoopRegionRuntimeTests(unittest.TestCase):
    def test_loop_region_exposes_stable_timing_contract_names(self) -> None:
        timing_names = PARCAE_LOOP_REGION_TIMING_NAMES

        self.assertEqual(timing_names.scoped("path1.parcae.band0", timing_names.loop_step), "path1.parcae.band0.loop_step")
        self.assertEqual(
            timing_names.scoped("path1.parcae.band0", timing_names.recurrent_residual_mix),
            "path1.parcae.band0.recurrent_residual_mix",
        )
        with self.assertRaisesRegex(ValueError, "unknown Parcae loop timing name"):
            timing_names.scoped("path1.parcae.band0", "surprise_region")

    def test_loop_region_tensor_layout_requires_rank3_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank-3"):
            ParcaeLoopRegionTensorLayout.from_state(torch.zeros(2, 3))

    def test_loop_region_runs_typed_state_residual_contract(self) -> None:
        initial_state = torch.zeros(1, 2, 3)
        controls = ParcaeLoopRegionControls(
            decay=torch.full((1, 1, 3), 0.5),
            injection=torch.ones(1, 2, 3),
            nonlinear=torch.full((1, 1, 3), 0.25),
        )

        def state_mix(state: torch.Tensor, decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
            return decay * state + injection

        def forward_recurrent_block(block_index: int, mixed: torch.Tensor) -> torch.Tensor:
            return mixed + float(block_index + 1)

        def apply_recurrent_residual(
            block_index: int,
            current_state: torch.Tensor,
            mixed: torch.Tensor,
            block_out: torch.Tensor,
            loop_controls: ParcaeLoopRegionControls,
        ) -> torch.Tensor:
            del block_index, current_state
            return mixed + loop_controls.nonlinear * (block_out - mixed)

        result = run_parcae_loop_region(
            initial_state=initial_state,
            controls=controls,
            config=ParcaeLoopRegionConfig(
                loop_count=2,
                gradient_start_step=1,
                recurrent_block_count=2,
                timing_prefix="test.parcae",
                diagnostics_enabled=True,
            ),
            kernels=ParcaeLoopRegionKernels(
                state_mix=state_mix,
                forward_recurrent_block=forward_recurrent_block,
                apply_recurrent_residual=apply_recurrent_residual,
            ),
        )

        self.assertTrue(torch.allclose(result.final_state, torch.full_like(initial_state, 2.625)))
        self.assertEqual(len(result.norm_history), 2)

    def test_loop_region_keeps_first_block_update_inside_boundary(self) -> None:
        initial_state = torch.zeros(1, 1, 2)
        controls = ParcaeLoopRegionControls(
            decay=torch.ones(1, 1, 2),
            injection=torch.ones(1, 1, 2),
            nonlinear=torch.full((1, 1, 2), 0.5),
        )
        update_calls: list[int] = []

        def state_mix(state: torch.Tensor, decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
            return decay * state + injection

        def forward_recurrent_block(block_index: int, mixed: torch.Tensor) -> torch.Tensor:
            return mixed + float(block_index + 1)

        def apply_recurrent_residual(
            block_index: int,
            current_state: torch.Tensor,
            mixed: torch.Tensor,
            block_out: torch.Tensor,
            loop_controls: ParcaeLoopRegionControls,
        ) -> torch.Tensor:
            if block_index == 0:
                update_calls.append(block_index)
                return current_state + loop_controls.injection + block_out
            return mixed + loop_controls.nonlinear * (block_out - mixed)

        result = run_parcae_loop_region(
            initial_state=initial_state,
            controls=controls,
            config=ParcaeLoopRegionConfig(
                loop_count=1,
                gradient_start_step=0,
                recurrent_block_count=2,
                timing_prefix="test.parcae",
            ),
            kernels=ParcaeLoopRegionKernels(
                state_mix=state_mix,
                forward_recurrent_block=forward_recurrent_block,
                apply_recurrent_residual=apply_recurrent_residual,
            ),
        )

        self.assertEqual(update_calls, [0])
        self.assertTrue(torch.allclose(result.final_state, torch.full_like(initial_state, 4.0)))

    def test_loop_region_validates_control_layout_before_running_kernels(self) -> None:
        initial_state = torch.zeros(2, 3, 4)
        controls = ParcaeLoopRegionControls(
            decay=torch.ones(1, 1, 5),
            injection=torch.ones(2, 3, 4),
            nonlinear=torch.ones(1, 1, 4),
        )

        def should_not_run(*args, **kwargs):
            raise AssertionError("kernel should not run when controls violate layout")

        with self.assertRaisesRegex(ValueError, "decay width"):
            run_parcae_loop_region(
                initial_state=initial_state,
                controls=controls,
                config=ParcaeLoopRegionConfig(
                    loop_count=1,
                    gradient_start_step=0,
                    recurrent_block_count=1,
                    timing_prefix="test.parcae",
                ),
                kernels=ParcaeLoopRegionKernels(
                    state_mix=should_not_run,
                    forward_recurrent_block=should_not_run,
                    apply_recurrent_residual=should_not_run,
                ),
            )

    def test_loop_region_expands_broadcast_injection_when_first_state_mix_is_fused(self) -> None:
        initial_state = torch.zeros(2, 3, 4)
        controls = ParcaeLoopRegionControls(
            decay=torch.ones(1, 1, 4),
            injection=torch.ones(1, 1, 4),
            nonlinear=torch.zeros(1, 1, 4),
        )

        seen_shapes: list[torch.Size] = []

        def state_mix(state: torch.Tensor, decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
            return decay * state + injection

        def forward_recurrent_block(block_index: int, mixed: torch.Tensor) -> torch.Tensor:
            del block_index
            seen_shapes.append(mixed.shape)
            return mixed

        def apply_recurrent_residual(
            block_index: int,
            current_state: torch.Tensor,
            mixed: torch.Tensor,
            block_out: torch.Tensor,
            loop_controls: ParcaeLoopRegionControls,
        ) -> torch.Tensor:
            del block_index, current_state, block_out, loop_controls
            return mixed

        result = run_parcae_loop_region(
            initial_state=initial_state,
            controls=controls,
            config=ParcaeLoopRegionConfig(
                loop_count=1,
                gradient_start_step=0,
                recurrent_block_count=1,
                timing_prefix="test.parcae",
                fuse_first_state_mix=True,
            ),
            kernels=ParcaeLoopRegionKernels(
                state_mix=state_mix,
                forward_recurrent_block=forward_recurrent_block,
                apply_recurrent_residual=apply_recurrent_residual,
            ),
        )

        self.assertEqual(seen_shapes, [initial_state.shape])
        self.assertTrue(torch.allclose(result.final_state, torch.ones_like(initial_state)))

    def test_loop_region_validates_runtime_contract_shape_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "gradient_start_step"):
            ParcaeLoopRegionConfig(
                loop_count=2,
                gradient_start_step=3,
                recurrent_block_count=1,
                timing_prefix="test.parcae",
            ).validate()

        with self.assertRaisesRegex(ValueError, "recurrent_block_count"):
            ParcaeLoopRegionConfig(
                loop_count=2,
                gradient_start_step=0,
                recurrent_block_count=0,
                timing_prefix="test.parcae",
            ).validate()


class RecurrentRuntimeTests(unittest.TestCase):
    @contextmanager
    def _temporary_env(self, name: str, value: str | None):
        previous = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_perplexity_from_loss_handles_large_finite_loss(self) -> None:
        self.assertEqual(perplexity_from_loss(1.0e6), float("inf"))

    def test_perplexity_from_loss_preserves_special_values(self) -> None:
        self.assertTrue(torch.isnan(torch.tensor(perplexity_from_loss(float("nan")))))
        self.assertEqual(perplexity_from_loss(float("inf")), float("inf"))
        self.assertEqual(perplexity_from_loss(float("-inf")), 0.0)

    def test_build_state_transform_projection_uses_shared_runtime_modes(self) -> None:
        dense = build_state_transform_projection(8, PrimitiveStateTransformMode.DENSE)
        block = build_state_transform_projection(8, PrimitiveStateTransformMode.BLOCK_DIAGONAL_2)
        block8 = build_state_transform_projection(16, PrimitiveStateTransformMode.BLOCK_DIAGONAL_8)

        self.assertIsInstance(dense, nn.Linear)
        self.assertIsInstance(block, BlockDiagonalLinear)
        self.assertEqual(block.blocks, 2)
        self.assertIsInstance(block8, BlockDiagonalLinear)
        self.assertEqual(block8.blocks, 8)
        self.assertEqual(block8.block_width, 2)

    def test_p20_atomic_transform_grad_policy_defaults_on_for_wide_tiles(self) -> None:
        with self._temporary_env("FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD", None):
            self.assertFalse(_p20_atomic_transform_grad_enabled(32))
            self.assertTrue(_p20_atomic_transform_grad_enabled(64))

    def test_p20_atomic_transform_grad_policy_respects_explicit_env(self) -> None:
        with self._temporary_env("FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD", "false"):
            self.assertFalse(_p20_atomic_transform_grad_enabled(64))
        with self._temporary_env("FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD", "true"):
            self.assertTrue(_p20_atomic_transform_grad_enabled(32))

    def test_triton_broadcast_owner_reduction_matches_sum_to_size(self) -> None:
        gradient = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

        for target_shape in (torch.Size([1, 1, 4]), torch.Size([2, 1, 4]), torch.Size([4])):
            actual = _sum_to_broadcast_owner(gradient, target_shape)
            expected = gradient.sum_to_size(target_shape)
            self.assertTrue(torch.equal(actual, expected))

    def test_triton_broadcast_owner_reduction_rejects_invalid_shape(self) -> None:
        gradient = torch.zeros(2, 3, 4)

        with self.assertRaisesRegex(RuntimeError, "not broadcast-compatible"):
            _sum_to_broadcast_owner(gradient, torch.Size([2, 2, 4]))

    def test_triton_width_owner_reduction_supports_only_cuda_width_owner_shapes(self) -> None:
        gradient = torch.zeros(2, 3, 4)

        self.assertFalse(_triton_width_owner_reduce_supported(gradient, torch.Size([1, 1, 4])))
        self.assertFalse(_triton_width_owner_reduce_supported(gradient, torch.Size([2, 1, 4])))

    def test_packed_linear_chunks_matches_individual_layers(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 4)
        layer_a = nn.Linear(4, 5)
        layer_b = nn.Linear(4, 6)

        actual_a, actual_b = packed_linear_chunks(inputs, layer_a, layer_b)

        self.assertTrue(torch.allclose(actual_a, layer_a(inputs)))
        self.assertTrue(torch.allclose(actual_b, layer_b(inputs)))

    def test_virtual_eggroll_linear_matches_materialized_reference(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(3, 2, 5, 4)
        weight = torch.randn(6, 4)
        bias = torch.randn(6)
        a, b = make_eggroll_factors(
            population_size=3,
            d_out=6,
            d_in=4,
            rank=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=123,
        )

        virtual = virtual_eggroll_linear(inputs, weight, perturbation_a=a, perturbation_b=b, sigma=0.01, bias=bias)
        materialized = materialized_eggroll_linear(
            inputs,
            weight,
            perturbation_a=a,
            perturbation_b=b,
            sigma=0.01,
            bias=bias,
        )

        self.assertTrue(torch.allclose(virtual, materialized, atol=1.0e-6, rtol=1.0e-6))

    def test_eggroll_update_matrix_matches_manual_score_weighted_sum(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(4, 3, 2)
        b = torch.randn(4, 5, 2)
        fitness = torch.tensor([1.0, -0.5, 0.25, 2.0])

        actual = eggroll_update_matrix(a, b, fitness, learning_rate=0.1)
        expected = 0.1 * torch.einsum("por,pir->oi", a * fitness.view(-1, 1, 1), b) / (4 * 2)

        self.assertTrue(torch.allclose(actual, expected))

    def test_block_diagonal_triton_scan_accepts_identity_transform_contract(self) -> None:
        signature = inspect.signature(_P20BlockDiagonalSequenceScan.forward)

        self.assertIn("identity_transform", signature.parameters)
        self.assertEqual(signature.parameters["identity_transform"].default, False)


class RuntimePolicyTests(unittest.TestCase):
    def test_apply_runtime_policy_rejects_non_default_backends_without_configure_hook(self) -> None:
        model = nn.Linear(4, 4)

        with self.assertRaisesRegex(ValidationError, "configure_runtime_policy"):
            apply_runtime_policy(
                model,
                DeviceRuntimeSpec(backend="cuda", dtype="bf16", head_loss_backend="compiled"),
            )

    def test_apply_runtime_policy_rejects_missing_backend_parameter(self) -> None:
        class CompileOnlyModel(nn.Module):
            def configure_runtime_policy(self, *, compile_mode, primitive_runtime_backend):
                del compile_mode, primitive_runtime_backend

        with self.assertRaisesRegex(ValidationError, "head_loss_backend"):
            apply_runtime_policy(
                CompileOnlyModel(),
                DeviceRuntimeSpec(backend="cuda", dtype="bf16", head_loss_backend="compiled"),
            )


class CudaTimingControlPlaneTests(unittest.TestCase):
    def test_timed_region_uses_active_collector_across_worker_thread(self) -> None:
        class FakeCollector:
            def __init__(self) -> None:
                self.names: list[str] = []

            @contextmanager
            def record(self, name: str):
                self.names.append(name)
                yield

        collector = FakeCollector()

        def worker() -> None:
            with timed_region("unit.worker_region"):
                pass

        with use_cuda_timing(collector):  # type: ignore[arg-type]
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

        self.assertEqual(collector.names, ["unit.worker_region"])

    def test_timed_region_skips_record_function_when_named_regions_are_disabled(self) -> None:
        calls: list[str] = []

        @contextmanager
        def fake_record_function(name: str):
            calls.append(name)
            yield

        with unittest.mock.patch("python.runtime.cuda_timing.record_function", fake_record_function):
            with timed_region("unit.disabled_named_region"):
                pass

        self.assertEqual(calls, [])

    def test_timed_region_emits_record_function_when_named_regions_are_enabled(self) -> None:
        calls: list[str] = []

        @contextmanager
        def fake_record_function(name: str):
            calls.append(name)
            yield

        with unittest.mock.patch("python.runtime.cuda_timing.record_function", fake_record_function):
            with use_named_timing_regions(True):
                with timed_region("unit.enabled_named_region"):
                    pass

        self.assertEqual(calls, ["unit.enabled_named_region"])

    def test_timed_region_emits_nvtx_only_when_requested(self) -> None:
        pushes: list[str] = []
        pops: list[str] = []

        class FakeNvtx:
            @staticmethod
            def range_push(name: str) -> None:
                pushes.append(name)

            @staticmethod
            def range_pop() -> None:
                pops.append("pop")

        with (
            unittest.mock.patch.dict("os.environ", {"FRACTAL_ENABLE_NVTX_RANGES": "true"}),
            unittest.mock.patch("torch.cuda.is_available", return_value=True),
            unittest.mock.patch("torch.cuda.nvtx", FakeNvtx, create=True),
        ):
            with timed_region("unit.nvtx_region"):
                pass

        self.assertEqual(pushes, ["unit.nvtx_region"])
        self.assertEqual(pops, ["pop"])

    def test_packed_linear_projection_matches_manual_linear_split(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 4)
        projection = PackedLinearProjection(4, (5, 6))

        expected = projection.projection(inputs).split((5, 6), dim=-1)
        actual = projection(inputs)

        self.assertEqual(len(actual), 2)
        self.assertTrue(torch.allclose(actual[0], expected[0]))
        self.assertTrue(torch.allclose(actual[1], expected[1]))
