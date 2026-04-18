from __future__ import annotations

import unittest

from python.jax_tpu import (
    JaxTpuBenchmarkSpec,
    JaxTpuDatasetSpec,
    JaxTpuDatasetType,
    JaxTpuModelShape,
    JaxTpuParallelismSpec,
    JaxTpuRunBudget,
    build_maxtext_command,
    get_candidate,
    render_shell_command,
)
from python.specs.common import ValidationError


class JaxTpuContractTests(unittest.TestCase):
    def test_attention_baseline_emits_maxtext_pretrain_command(self) -> None:
        spec = JaxTpuBenchmarkSpec(
            run_name="fractal-scout",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("attention-baseline"),
            shape=JaxTpuModelShape(sequence_length=512, d_model=256, num_layers=4, num_heads=4),
            budget=JaxTpuRunBudget(steps=5, per_device_batch_size=2),
        )

        command = build_maxtext_command(spec)
        rendered = render_shell_command(command)

        self.assertEqual(command[:3], ["python3", "-m", "maxtext.trainers.pre_train.train"])
        self.assertIn("run_name=fractal-scout", rendered)
        self.assertIn("dataset_type=synthetic", rendered)
        self.assertIn("max_target_length=512", rendered)
        self.assertIn("base_emb_dim=256", rendered)
        self.assertIn("base_num_decoder_layers=4", rendered)
        self.assertIn("base_num_query_heads=4", rendered)
        self.assertIn("base_num_kv_heads=4", rendered)
        self.assertNotIn("base_num_heads=", rendered)
        self.assertNotIn("enable_profiler=", rendered)

    def test_p20_candidate_requires_patched_maxtext(self) -> None:
        spec = JaxTpuBenchmarkSpec(
            run_name="p20-scout",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("rotary-gated-recurrent-state-update"),
        )

        with self.assertRaises(ValidationError):
            build_maxtext_command(spec)

        command = build_maxtext_command(spec, allow_patched_maxtext=True)

        self.assertIn(
            "fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update",
            command,
        )

    def test_p20_candidate_contract_tracks_state_carry(self) -> None:
        candidate = get_candidate("rotary-gated-recurrent-state-update")

        self.assertTrue(candidate.requires_patched_maxtext)
        self.assertTrue(candidate.kernel_contract.carries_state_across_tokens)
        self.assertEqual(candidate.kernel_contract.recurrence_axis, "tokens")
        self.assertEqual(candidate.kernel_contract.fusion_boundary, "ffn-side-recurrent-state-update")

    def test_tfds_dataset_requires_path_and_name(self) -> None:
        with self.assertRaises(ValidationError):
            JaxTpuDatasetSpec(dataset_type=JaxTpuDatasetType.TFDS).validate()

        dataset = JaxTpuDatasetSpec(
            dataset_type=JaxTpuDatasetType.TFDS,
            dataset_path="gs://fractal-data",
            dataset_name="c4/en:3.0.1",
        )

        dataset.validate()
        self.assertEqual(dataset.to_overrides()["dataset_type"], "tfds")

    def test_shape_and_parallelism_validate_explicit_contracts(self) -> None:
        JaxTpuModelShape(d_model=512, num_heads=8, moe_experts=16, moe_top_k=2).validate()
        with self.assertRaises(ValidationError):
            JaxTpuModelShape(d_model=510, num_heads=8).validate()
        with self.assertRaises(ValidationError):
            JaxTpuParallelismSpec(ici_fsdp_parallelism=0).validate()


if __name__ == "__main__":
    unittest.main()
