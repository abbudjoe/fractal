from __future__ import annotations

import importlib.util
import unittest

import torch

from python.models.reference_ssm import (
    FlaGatedDeltaNetControlShellSequenceMixer,
    FlaGatedDeltaNetSequenceMixer,
    FlaGdnpControlConditionedSequenceMixer,
    resolve_reference_ssm_config,
)
from python.specs.path1 import ReferenceSsmProfile


def _has_fla_runtime() -> bool:
    return importlib.util.find_spec("fla") is not None and torch.cuda.is_available()


@unittest.skipUnless(_has_fla_runtime(), "FLA parity tests require CUDA and flash-linear-attention")
class FlaGatedDeltaNetParityTests(unittest.TestCase):
    def _config(self):
        return resolve_reference_ssm_config(
            d_model=64,
            head_count=4,
            profile=ReferenceSsmProfile.GATED_DELTANET_FLA,
            dtype_mode="bf16",
            layer_index=0,
        )

    def test_native_control_shell_matches_native_fla_gdn_at_initialization(self) -> None:
        device = torch.device("cuda")
        dtype = torch.bfloat16
        config = self._config()
        torch.manual_seed(1234)
        native = FlaGatedDeltaNetSequenceMixer(config).to(device=device, dtype=dtype)
        torch.manual_seed(1234)
        shell = FlaGatedDeltaNetControlShellSequenceMixer(config).to(device=device, dtype=dtype)
        hidden = torch.randn(2, 16, config.d_model, device=device, dtype=dtype)

        native_out = native(hidden)
        shell_out = shell(hidden)

        max_error = (native_out - shell_out).abs().max().item()
        mean_error = (native_out - shell_out).abs().mean().item()
        self.assertLess(max_error, 1.0e-2, f"max_error={max_error}, mean_error={mean_error}")
        self.assertLess(mean_error, 1.0e-3, f"max_error={max_error}, mean_error={mean_error}")

    def test_zero_delta_p20_conditioner_matches_control_shell(self) -> None:
        device = torch.device("cuda")
        dtype = torch.bfloat16
        config = self._config()
        torch.manual_seed(5678)
        shell = FlaGatedDeltaNetControlShellSequenceMixer(config).to(device=device, dtype=dtype)
        torch.manual_seed(5678)
        conditioned = FlaGdnpControlConditionedSequenceMixer(config).to(device=device, dtype=dtype)
        hidden = torch.randn(2, 16, config.d_model, device=device, dtype=dtype)

        shell_out = shell(hidden)
        conditioned_out = conditioned(hidden)

        max_error = (shell_out - conditioned_out).abs().max().item()
        mean_error = (shell_out - conditioned_out).abs().mean().item()
        self.assertLess(max_error, 1.0e-2, f"max_error={max_error}, mean_error={mean_error}")
        self.assertLess(mean_error, 1.0e-3, f"max_error={max_error}, mean_error={mean_error}")


if __name__ == "__main__":
    unittest.main()
