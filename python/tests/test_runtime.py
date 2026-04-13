from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from python.runtime.cuda_setup_patch import patch_cuda_setup_text
from python.runtime.recurrent import (
    BlockDiagonalLinear,
    PackedLinearProjection,
    build_state_transform_projection,
    packed_linear_chunks,
)
from python.runtime.train_eval import perplexity_from_loss
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


class RecurrentRuntimeTests(unittest.TestCase):
    def test_perplexity_from_loss_handles_large_finite_loss(self) -> None:
        self.assertEqual(perplexity_from_loss(1.0e6), float("inf"))

    def test_perplexity_from_loss_preserves_special_values(self) -> None:
        self.assertTrue(torch.isnan(torch.tensor(perplexity_from_loss(float("nan")))))
        self.assertEqual(perplexity_from_loss(float("inf")), float("inf"))
        self.assertEqual(perplexity_from_loss(float("-inf")), 0.0)

    def test_build_state_transform_projection_uses_shared_runtime_modes(self) -> None:
        dense = build_state_transform_projection(8, PrimitiveStateTransformMode.DENSE)
        block = build_state_transform_projection(8, PrimitiveStateTransformMode.BLOCK_DIAGONAL_2)

        self.assertIsInstance(dense, nn.Linear)
        self.assertIsInstance(block, BlockDiagonalLinear)
        self.assertEqual(block.blocks, 2)

    def test_packed_linear_chunks_matches_individual_layers(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 4)
        layer_a = nn.Linear(4, 5)
        layer_b = nn.Linear(4, 6)

        actual_a, actual_b = packed_linear_chunks(inputs, layer_a, layer_b)

        self.assertTrue(torch.allclose(actual_a, layer_a(inputs)))
        self.assertTrue(torch.allclose(actual_b, layer_b(inputs)))

    def test_packed_linear_projection_matches_manual_linear_split(self) -> None:
        torch.manual_seed(0)
        inputs = torch.randn(2, 3, 4)
        projection = PackedLinearProjection(4, (5, 6))

        expected = projection.projection(inputs).split((5, 6), dim=-1)
        actual = projection(inputs)

        self.assertEqual(len(actual), 2)
        self.assertTrue(torch.allclose(actual[0], expected[0]))
        self.assertTrue(torch.allclose(actual[1], expected[1]))
