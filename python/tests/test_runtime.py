from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from python.runtime.cuda_setup_patch import patch_cuda_setup_text
from python.runtime.recurrent import (
    BlockDiagonalLinear,
    build_state_transform_projection,
    packed_linear_chunks,
)
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

    def test_raises_when_arch_block_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed to locate CUDA arch block"):
            patch_cuda_setup_text("print('no cuda block here')\n", "8.9")


class RecurrentRuntimeTests(unittest.TestCase):
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
