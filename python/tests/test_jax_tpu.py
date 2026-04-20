from __future__ import annotations

import unittest

from python.jax_tpu import (
    JaxTpuBenchmarkSpec,
    JaxTpuDatasetSpec,
    JaxTpuDatasetType,
    JaxTpuModelShape,
    JaxTpuParallelismSpec,
    JaxTpuRunBudget,
    JaxTpuTokenizerSpec,
    build_maxtext_command,
    get_candidate,
    render_shell_command,
)
from python.jax_tpu import lm_smoke
from python.jax_tpu.adapters import rotary_gated_recurrent_state_update as rgrp_adapter
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
        self.assertIn("head_dim=64", rendered)
        self.assertIn("vocab_size=32000", rendered)
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
        self.assertIn("decoder_block=default", command)
        self.assertIn("fractal_rgrp_state_transform=block-diagonal-4-masked-dense", command)
        self.assertIn("fractal_rgrp_scan_unroll=3", command)

    def test_p20_candidate_contract_tracks_state_carry(self) -> None:
        candidate = get_candidate("rotary-gated-recurrent-state-update")

        self.assertTrue(candidate.requires_patched_maxtext)
        self.assertTrue(candidate.kernel_contract.carries_state_across_tokens)
        self.assertEqual(candidate.kernel_contract.recurrence_axis, "tokens")
        self.assertEqual(candidate.kernel_contract.fusion_boundary, "ffn-side-recurrent-state-update")

    def test_rgrp_mlp_sidecar_candidate_keeps_transformer_backbone(self) -> None:
        spec = JaxTpuBenchmarkSpec(
            run_name="rgrp-sidecar-scout",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("rotary-gated-recurrent-state-update-mlp-sidecar"),
        )

        command = build_maxtext_command(spec, allow_patched_maxtext=True)
        rendered = render_shell_command(command)

        self.assertIn("decoder_block=default", rendered)
        self.assertIn("scan_layers=false", rendered)
        self.assertIn("fractal_candidate=rotary-gated-recurrent-state-update", rendered)
        self.assertIn("fractal_rgrp_integration_mode=mlp-sidecar", rendered)
        self.assertIn("fractal_rgrp_layers=1", rendered)
        self.assertIn("fractal_rgrp_bottleneck_dim=64", rendered)
        self.assertIn("fractal_rgrp_sidecar_type=rgrp", rendered)
        self.assertIn("fractal_rgrp_side_scale=0.1", rendered)
        self.assertIn("fractal_rgrp_output_init=xavier", rendered)
        self.assertEqual(
            spec.candidate.kernel_contract.fusion_boundary,
            "selected-mlp-sidecar-recurrent-state-update",
        )

    def test_matched_sidecar_controls_swap_only_sidecar_operator(self) -> None:
        for slug, sidecar_type in (
            ("tiny-mlp-mlp-sidecar-control", "tiny-mlp"),
            ("tiny-glu-mlp-sidecar-control", "tiny-glu"),
            ("binary-tree-mlp-sidecar-control", "binary-tree"),
        ):
            with self.subTest(slug=slug):
                spec = JaxTpuBenchmarkSpec(
                    run_name=f"{sidecar_type}-control",
                    base_output_directory="gs://fractal-maxtext-runs",
                    candidate=get_candidate(slug),
                )

                command = build_maxtext_command(spec, allow_patched_maxtext=True)
                rendered = render_shell_command(command)

                self.assertIn("decoder_block=default", rendered)
                self.assertIn("scan_layers=false", rendered)
                self.assertIn("fractal_candidate=rotary-gated-recurrent-state-update", rendered)
                self.assertIn("fractal_rgrp_integration_mode=mlp-sidecar", rendered)
                self.assertIn("fractal_rgrp_layers=1", rendered)
                self.assertIn("fractal_rgrp_bottleneck_dim=64", rendered)
                self.assertIn(f"fractal_rgrp_sidecar_type={sidecar_type}", rendered)
                self.assertIn("fractal_rgrp_output_init=zero", rendered)
                self.assertEqual(
                    spec.candidate.kernel_contract.fusion_boundary,
                    "selected-mlp-sidecar-control",
                )

    def test_parcae_rgrp_control_candidate_matches_looped_scaffold_contract(self) -> None:
        spec = JaxTpuBenchmarkSpec(
            run_name="parcae-rgrp-scout",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("parcae-rgrp-control-looped-attention"),
            shape=JaxTpuModelShape(vocab_size=32_000, sequence_length=256, d_model=128, num_layers=8, num_heads=4),
        )

        command = build_maxtext_command(spec, allow_patched_maxtext=True)
        rendered = render_shell_command(command)

        self.assertIn("base_num_decoder_layers=8", rendered)
        self.assertIn("base_emb_dim=128", rendered)
        self.assertIn("base_mlp_dim=512", rendered)
        self.assertIn("scan_layers=false", rendered)
        self.assertIn("fractal_candidate=parcae-rgrp-control-looped-attention", rendered)
        self.assertIn("fractal_parcae_loop_count=2", rendered)
        self.assertIn("fractal_parcae_loop_policy=fixed", rendered)
        self.assertIn("fractal_parcae_mu_rec=2.0", rendered)
        self.assertIn("fractal_parcae_mu_bwd=2", rendered)
        self.assertIn("fractal_parcae_discretization=stable-exp", rendered)
        self.assertIn("fractal_rgrp_state_transform=block-diagonal-4-masked-dense", rendered)
        self.assertIn("fractal_rgrp_scan_unroll=3", rendered)
        self.assertTrue(spec.candidate.kernel_contract.carries_state_across_tokens)
        self.assertEqual(
            spec.candidate.kernel_contract.fusion_boundary,
            "decoder-stack-loop-injection-rgrp-controller",
        )

    def test_parcae_controls_keep_same_looped_scaffold_without_rgrp_state(self) -> None:
        for slug in ("parcae-looped-attention", "parcae-bx-looped-attention"):
            with self.subTest(slug=slug):
                spec = JaxTpuBenchmarkSpec(
                    run_name=slug,
                    base_output_directory="gs://fractal-maxtext-runs",
                    candidate=get_candidate(slug),
                )

                command = build_maxtext_command(spec, allow_patched_maxtext=True)
                rendered = render_shell_command(command)

                self.assertIn(f"fractal_candidate={slug}", rendered)
                self.assertIn("fractal_parcae_loop_count=2", rendered)
                self.assertIn("fractal_parcae_loop_policy=fixed", rendered)
                self.assertIn("fractal_parcae_depth_distribution=poisson", rendered)
                self.assertIn("scan_layers=false", rendered)
                self.assertFalse(spec.candidate.kernel_contract.carries_state_across_tokens)
                self.assertIn("looped-middle", spec.candidate.kernel_contract.lowering)

    def test_parcae_per_sequence_loop_contract_is_explicit_override(self) -> None:
        spec = JaxTpuBenchmarkSpec(
            run_name="parcae-bx-perseq",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("parcae-bx-looped-attention"),
            extra_overrides={
                "fractal_parcae_loop_policy": "per-sequence",
                "fractal_parcae_mu_rec": 2.0,
                "fractal_parcae_mu_bwd": 1,
                "fractal_parcae_max_loop_count": 4,
                "fractal_parcae_discretization": "zoh",
            },
        )

        rendered = render_shell_command(build_maxtext_command(spec, allow_patched_maxtext=True))

        self.assertIn("fractal_candidate=parcae-bx-looped-attention", rendered)
        self.assertIn("fractal_parcae_loop_policy=per-sequence", rendered)
        self.assertIn("fractal_parcae_mu_rec=2.0", rendered)
        self.assertIn("fractal_parcae_mu_bwd=1", rendered)
        self.assertIn("fractal_parcae_max_loop_count=4", rendered)
        self.assertIn("fractal_parcae_discretization=zoh", rendered)

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

    def test_hf_dataset_requires_path_and_tokenizer(self) -> None:
        with self.assertRaises(ValidationError):
            JaxTpuDatasetSpec(dataset_type=JaxTpuDatasetType.HF).validate()

        spec = JaxTpuBenchmarkSpec(
            run_name="hf-ingress",
            base_output_directory="gs://fractal-maxtext-runs",
            candidate=get_candidate("attention-baseline"),
            shape=JaxTpuModelShape(vocab_size=50_257, sequence_length=256, d_model=256, num_layers=4, num_heads=4),
            dataset=JaxTpuDatasetSpec(dataset_type=JaxTpuDatasetType.HF, hf_path="roneneldan/TinyStories"),
        )
        with self.assertRaises(ValidationError):
            build_maxtext_command(spec)

        command = build_maxtext_command(
            JaxTpuBenchmarkSpec(
                run_name="hf-ingress",
                base_output_directory="gs://fractal-maxtext-runs",
                candidate=get_candidate("attention-baseline"),
                shape=JaxTpuModelShape(
                    vocab_size=50_257,
                    sequence_length=256,
                    d_model=256,
                    num_layers=4,
                    num_heads=4,
                ),
                dataset=JaxTpuDatasetSpec(dataset_type=JaxTpuDatasetType.HF, hf_path="roneneldan/TinyStories"),
                tokenizer=JaxTpuTokenizerSpec(tokenizer_type="huggingface", tokenizer_path="gpt2"),
            )
        )
        rendered = render_shell_command(command)
        self.assertIn("dataset_type=hf", rendered)
        self.assertIn("hf_path=roneneldan/TinyStories", rendered)
        self.assertIn("tokenizer_type=huggingface", rendered)
        self.assertIn("tokenizer_path=gpt2", rendered)
        self.assertIn("vocab_size=50257", rendered)

    def test_shape_and_parallelism_validate_explicit_contracts(self) -> None:
        JaxTpuModelShape(d_model=512, num_heads=8, moe_experts=16, moe_top_k=2).validate()
        with self.assertRaises(ValidationError):
            JaxTpuModelShape(d_model=510, num_heads=8).validate()
        with self.assertRaises(ValidationError):
            JaxTpuParallelismSpec(ici_fsdp_parallelism=0).validate()

    def test_rgrp_adapter_imports_without_jax(self) -> None:
        self.assertEqual(rgrp_adapter.ADAPTER_NAME, "rotary-gated-recurrent-state-update")
        config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4",
            dtype="float32",
        )
        config.validate()
        self.assertEqual(config.packed_width, 56)
        if not rgrp_adapter.jax_available():
            with self.assertRaisesRegex(RuntimeError, "JAX is required"):
                rgrp_adapter.require_jax()

    def test_rgrp_masked_dense_transform_contract_validates(self) -> None:
        config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="bfloat16",
        )
        config.validate()
        self.assertEqual(config.block_count, 4)
        self.assertTrue(config.stores_block_transform_as_dense)

    def test_rgrp_lowering_knobs_validate(self) -> None:
        config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="bfloat16",
            scan_unroll=4,
            projection_mode="sequence",
            trig_mode="scan",
        )
        config.validate()

        with self.assertRaisesRegex(ValueError, "scan_unroll"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(d_model=16, scan_unroll=0).validate()
        with self.assertRaisesRegex(ValueError, "projection_mode"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(d_model=16, projection_mode="packed").validate()
        with self.assertRaisesRegex(ValueError, "trig_mode"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(d_model=16, trig_mode="lookup").validate()
        with self.assertRaisesRegex(ValueError, "requires trig_mode='scan'"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
                d_model=16,
                projection_mode="scan",
                trig_mode="precompute",
            ).validate()
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(d_model=16, execution_mode="warp-drive").validate()
        with self.assertRaisesRegex(ValueError, "pallas_chunk_size"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(d_model=16, pallas_chunk_size=0).validate()
        with self.assertRaisesRegex(ValueError, "requires projection_mode='sequence'"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
                d_model=16,
                projection_mode="scan",
                trig_mode="scan",
                execution_mode="pallas-forward",
            ).validate()
        with self.assertRaisesRegex(ValueError, "requires trig_mode='precompute'"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
                d_model=16,
                trig_mode="scan",
                execution_mode="pallas-forward",
            ).validate()
        with self.assertRaisesRegex(ValueError, "dense-shaped"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
                d_model=16,
                state_transform="block-diagonal-4",
                execution_mode="pallas-forward",
            ).validate()
        with self.assertRaisesRegex(ValueError, "masked block-diagonal"):
            rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
                d_model=16,
                state_transform="dense",
                execution_mode="pallas-block-tiled-forward",
            ).validate()

    def test_jax_lm_smoke_config_validates_without_jax(self) -> None:
        baseline = lm_smoke.JaxLmSmokeConfig(
            variant="mlp",
            vocab_size=128,
            seq_len=16,
            batch_size=2,
            d_model=32,
            layers=1,
            heads=4,
        )
        baseline.validate()

        recurrent = lm_smoke.JaxLmSmokeConfig(
            variant="rgrp",
            vocab_size=128,
            seq_len=16,
            batch_size=2,
            d_model=32,
            layers=1,
            heads=4,
            rgrp_state_transform="block-diagonal-4-masked-dense",
            rgrp_scan_unroll=2,
            rgrp_projection_mode="sequence",
            rgrp_trig_mode="scan",
            rgrp_execution_mode="scan",
            rgrp_pallas_chunk_size=128,
        )
        recurrent.validate()
        self.assertEqual(recurrent.d_ff, 128)
        self.assertEqual(recurrent.head_dim, 8)
        if not lm_smoke.jax_available():
            with self.assertRaisesRegex(RuntimeError, "JAX is required"):
                lm_smoke.require_jax()

    @unittest.skipUnless(rgrp_adapter.jax_available(), "JAX is not installed in this environment")
    def test_rgrp_jax_adapter_matches_torch_p20_block_diagonal_scan(self) -> None:
        import numpy as np
        import torch

        from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer
        from python.specs.runtime import PrimitiveStateTransformMode

        torch.manual_seed(7)
        inputs = torch.randn(2, 5, 16)
        torch_primitive = P20RotaryStateOutputRuntimeSequenceMixer(
            16,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        torch_result = torch_primitive.scan(inputs)

        config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4",
            dtype="float32",
        )
        params = rgrp_adapter.params_from_torch_state_dict(torch_primitive.state_dict(), config)
        jax_inputs = rgrp_adapter.jnp.asarray(inputs.detach().numpy(), dtype=rgrp_adapter.jnp.float32)
        jax_outputs, jax_final_state = rgrp_adapter.scan(params, jax_inputs, config)

        np.testing.assert_allclose(
            np.asarray(jax_outputs),
            torch_result.emitted_outputs.detach().numpy(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        np.testing.assert_allclose(
            np.asarray(jax_final_state),
            torch_result.final_state.detach().numpy(),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    @unittest.skipUnless(rgrp_adapter.jax_available(), "JAX is not installed in this environment")
    def test_rgrp_masked_dense_matches_grouped_block_diagonal(self) -> None:
        import numpy as np
        import torch

        from python.models.primitives import P20RotaryStateOutputRuntimeSequenceMixer
        from python.specs.runtime import PrimitiveStateTransformMode

        torch.manual_seed(11)
        inputs = torch.randn(2, 5, 16)
        torch_primitive = P20RotaryStateOutputRuntimeSequenceMixer(
            16,
            state_transform_mode=PrimitiveStateTransformMode.BLOCK_DIAGONAL_4,
        )
        grouped_config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4",
            dtype="float32",
        )
        masked_config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="float32",
        )
        grouped_params = rgrp_adapter.params_from_torch_state_dict(torch_primitive.state_dict(), grouped_config)
        masked_params = rgrp_adapter.params_from_torch_state_dict(torch_primitive.state_dict(), masked_config)
        jax_inputs = rgrp_adapter.jnp.asarray(inputs.detach().numpy(), dtype=rgrp_adapter.jnp.float32)

        grouped_outputs, grouped_state = rgrp_adapter.scan(grouped_params, jax_inputs, grouped_config)
        masked_outputs, masked_state = rgrp_adapter.scan(masked_params, jax_inputs, masked_config)

        np.testing.assert_allclose(
            np.asarray(masked_outputs),
            np.asarray(grouped_outputs),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        np.testing.assert_allclose(
            np.asarray(masked_state),
            np.asarray(grouped_state),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    @unittest.skipUnless(rgrp_adapter.jax_available(), "JAX is not installed in this environment")
    def test_rgrp_lowering_modes_match_reference(self) -> None:
        import numpy as np

        key = rgrp_adapter.jax.random.PRNGKey(19)
        param_key, input_key = rgrp_adapter.jax.random.split(key)
        reference_config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="float32",
            projection_mode="sequence",
            trig_mode="precompute",
        )
        trig_scan_config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="float32",
            projection_mode="sequence",
            trig_mode="scan",
        )
        projection_scan_config = rgrp_adapter.RotaryGatedRecurrentStateUpdateConfig(
            d_model=16,
            state_transform="block-diagonal-4-masked-dense",
            dtype="float32",
            projection_mode="scan",
            trig_mode="scan",
        )
        params = rgrp_adapter.init_params(param_key, reference_config)
        inputs = rgrp_adapter.jax.random.normal(input_key, (2, 5, 16), dtype=rgrp_adapter.jnp.float32)

        reference_outputs, reference_state = rgrp_adapter.scan(params, inputs, reference_config)
        trig_outputs, trig_state = rgrp_adapter.scan(params, inputs, trig_scan_config)
        projection_outputs, projection_state = rgrp_adapter.scan(params, inputs, projection_scan_config)

        np.testing.assert_allclose(np.asarray(trig_outputs), np.asarray(reference_outputs), atol=1.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(np.asarray(trig_state), np.asarray(reference_state), atol=1.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(
            np.asarray(projection_outputs),
            np.asarray(reference_outputs),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        np.testing.assert_allclose(np.asarray(projection_state), np.asarray(reference_state), atol=1.0e-5, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
