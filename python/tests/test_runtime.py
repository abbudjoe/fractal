from __future__ import annotations

import unittest

from python.runtime.cuda_setup_patch import patch_cuda_setup_text


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
