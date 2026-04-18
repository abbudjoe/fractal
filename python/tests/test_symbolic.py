from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from python.specs.common import ValidationError
from python.specs.symbolic import (
    SymbolicBenchmarkManifest,
    SymbolicDatasetSpec,
    SymbolicModelFamily,
    SymbolicPreset,
    SymbolicTrainSpec,
    SymbolicTreeOptimizer,
)
from python.symbolic.autodiff import autodiff_loss_and_gradient
from python.symbolic.bridge import TokenQuantizer, run_symbolic_bridge, select_safety_calibration_rows
from python.symbolic.bridge_canary import resolve_device, run_bridge_canary
from python.symbolic.bridge_corpus import run_bridge_corpus
from python.symbolic.bridge_lm import run_symbolic_bridge_lm
from python.symbolic.bridge_sequence import run_sequence_bridge
from python.symbolic.formulas import default_symbolic_tasks, sample_symbolic_dataset, tier0_exact_recovery_tasks
from python.symbolic.models import build_symbolic_model
from python.symbolic.runner import (
    enumerate_paper_root_candidates,
    evaluate_predictions,
    refit_paper_readout_after_hardening,
    run_symbolic_benchmark,
    run_single_symbolic_case,
)
from python.symbolic.torch_backend import resolve_torch_device


class _FakeAvailability:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _FakeAvailability(cuda_available)
        self.backends = type("Backends", (), {"mps": _FakeAvailability(mps_available)})()

    def device(self, name: str) -> str:
        return name


class SymbolicDatasetTests(unittest.TestCase):
    def test_default_tasks_cover_depth_bins_with_finite_splits(self) -> None:
        spec = SymbolicDatasetSpec(train_samples=8, validation_samples=10, extrapolation_samples=10)
        tasks = default_symbolic_tasks(tasks_per_depth=1)

        self.assertEqual([task.difficulty_depth for task in tasks], [1, 2, 3, 4])
        for task in tasks:
            dataset = sample_symbolic_dataset(task, spec, seed=123)
            self.assertTrue(dataset.train.finite())
            self.assertTrue(dataset.validation.finite())
            self.assertTrue(dataset.extrapolation.finite())
            self.assertIn("source_formula", dataset.metadata()["task"])

    def test_default_tasks_support_larger_depth_bins(self) -> None:
        spec = SymbolicDatasetSpec(train_samples=8, validation_samples=10, extrapolation_samples=10, tasks_per_depth=4)
        spec.validate()
        tasks = default_symbolic_tasks(tasks_per_depth=4)

        self.assertEqual(len(tasks), 16)
        self.assertEqual({depth: sum(task.difficulty_depth == depth for task in tasks) for depth in (1, 2, 3, 4)}, {1: 4, 2: 4, 3: 4, 4: 4})
        for task in tasks:
            dataset = sample_symbolic_dataset(task, spec, seed=456)
            self.assertTrue(dataset.train.finite())
            self.assertTrue(dataset.validation.finite())
            self.assertTrue(dataset.extrapolation.finite())

    def test_tier0_tasks_are_shallow_and_finite(self) -> None:
        spec = SymbolicDatasetSpec(train_samples=8, validation_samples=10, extrapolation_samples=10)
        tasks = tier0_exact_recovery_tasks()

        self.assertGreaterEqual(len(tasks), 4)
        self.assertTrue(all(task.difficulty_depth <= 2 for task in tasks))
        self.assertIn("t0_exp_x", {task.task_id for task in tasks})
        self.assertIn("t0_nested_exp", {task.task_id for task in tasks})
        for task in tasks:
            dataset = sample_symbolic_dataset(task, spec, seed=321)
            self.assertTrue(dataset.train.finite())
            self.assertTrue(dataset.validation.finite())
            self.assertTrue(dataset.extrapolation.finite())


class SymbolicModelTests(unittest.TestCase):
    def test_torch_backend_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(resolve_torch_device(_FakeTorch(cuda_available=True, mps_available=True), "auto"), "cuda")
        self.assertEqual(resolve_torch_device(_FakeTorch(cuda_available=False, mps_available=True), "auto"), "mps")
        self.assertEqual(resolve_torch_device(_FakeTorch(cuda_available=False, mps_available=False), "auto"), "cpu")

    def test_torch_backend_cuda_request_requires_cuda(self) -> None:
        self.assertEqual(resolve_torch_device(_FakeTorch(cuda_available=True, mps_available=False), "cuda"), "cuda")
        with self.assertRaisesRegex(RuntimeError, "backend=cuda requested"):
            resolve_torch_device(_FakeTorch(cuda_available=False, mps_available=False), "cuda")

    def test_bridge_device_auto_prefers_cuda_then_mps_then_cpu(self) -> None:
        self.assertEqual(resolve_device(_FakeTorch(cuda_available=True, mps_available=True), "auto"), "cuda")
        self.assertEqual(resolve_device(_FakeTorch(cuda_available=False, mps_available=True), "auto"), "mps")
        self.assertEqual(resolve_device(_FakeTorch(cuda_available=False, mps_available=False), "auto"), "cpu")

    def test_bridge_device_cuda_request_requires_cuda(self) -> None:
        self.assertEqual(resolve_device(_FakeTorch(cuda_available=True, mps_available=False), "cuda"), "cuda")
        with self.assertRaisesRegex(RuntimeError, "requested device=cuda"):
            resolve_device(_FakeTorch(cuda_available=False, mps_available=False), "cuda")

    def test_symbolic_manifest_accepts_cuda_for_torch_autodiff(self) -> None:
        manifest = SymbolicBenchmarkManifest(
            run_label="unit-cuda",
            preset=SymbolicPreset.SMOKE,
            model_families=(SymbolicModelFamily.PAPER_COMPLEX_EML,),
            seeds=(1,),
            dataset=SymbolicDatasetSpec(tasks_per_depth=1),
            train=SymbolicTrainSpec(tree_optimizer=SymbolicTreeOptimizer.TORCH_AUTODIFF),
            backend="cuda",
        )

        manifest.validate()

    def test_symbolic_manifest_requires_torch_autodiff_for_cuda(self) -> None:
        manifest = SymbolicBenchmarkManifest(
            run_label="unit-cuda",
            preset=SymbolicPreset.SMOKE,
            model_families=(SymbolicModelFamily.PAPER_COMPLEX_EML,),
            seeds=(1,),
            dataset=SymbolicDatasetSpec(tasks_per_depth=1),
            train=SymbolicTrainSpec(tree_optimizer=SymbolicTreeOptimizer.AUTODIFF),
            backend="cuda",
        )

        with self.assertRaisesRegex(ValidationError, "backend=cuda"):
            manifest.validate()

    def test_all_model_families_forward_and_harden(self) -> None:
        xs = (-0.5, 0.0, 0.5)
        for family in SymbolicModelFamily:
            with self.subTest(family=family.value):
                model = build_symbolic_model(family, depth=2, seed=7, hidden_units=4)
                predictions = model.predict(xs)
                self.assertEqual(len(predictions), len(xs))
                self.assertTrue(all(math.isfinite(value) for value in predictions))
                hardened = model.harden()
                compiled = hardened.compile()
                compiled_values = [compiled(value) for value in xs]
                self.assertTrue(all(math.isfinite(value) for value in compiled_values))

    def test_paper_complex_export_names_eml_and_records_surrogate_note(self) -> None:
        model = build_symbolic_model(SymbolicModelFamily.PAPER_COMPLEX_EML, depth=1, seed=3, hidden_units=4)
        hardened = model.harden()

        self.assertIn("eml", hardened.expression)
        self.assertTrue(any("complex" in note for note in hardened.notes))

    def test_autodiff_gradient_is_finite_for_tree_families(self) -> None:
        task = tier0_exact_recovery_tasks()[0]
        dataset = sample_symbolic_dataset(
            task,
            SymbolicDatasetSpec(train_samples=5, validation_samples=5, extrapolation_samples=5),
            seed=17,
        )
        for family in (
            SymbolicModelFamily.PAPER_COMPLEX_EML,
            SymbolicModelFamily.STABLE_REAL_EML,
            SymbolicModelFamily.GENERIC_TREE,
        ):
            with self.subTest(family=family.value):
                model = build_symbolic_model(family, depth=task.difficulty_depth, seed=19, hidden_units=4)
                loss, gradient = autodiff_loss_and_gradient(
                    model,
                    dataset.train,
                    y_scale=1.0,
                    temperature=0.7,
                    entropy_weight=0.01,
                )
                self.assertTrue(math.isfinite(loss))
                self.assertEqual(len(gradient), len(model.parameters()))
                self.assertTrue(any(abs(value) > 1.0e-12 for value in gradient))

    def test_torch_autodiff_path_trains_when_torch_is_available(self) -> None:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed in the active interpreter")
        task = tier0_exact_recovery_tasks()[0]
        dataset = sample_symbolic_dataset(
            task,
            SymbolicDatasetSpec(train_samples=8, validation_samples=8, extrapolation_samples=8),
            seed=23,
        )

        result = run_single_symbolic_case(
            dataset=dataset,
            model_family=SymbolicModelFamily.PAPER_COMPLEX_EML,
            seed=29,
            train_spec=SymbolicTrainSpec(
                steps=2,
                hidden_units=4,
                tree_optimizer=SymbolicTreeOptimizer.TORCH_AUTODIFF,
            ),
            backend="cpu",
        )

        self.assertTrue(result.compile_success)
        self.assertTrue(math.isfinite(result.train_loss_final))

    def test_paper_readout_refit_recovers_exact_eml_native_scale(self) -> None:
        task = tier0_exact_recovery_tasks()[1]
        dataset = sample_symbolic_dataset(
            task,
            SymbolicDatasetSpec(train_samples=12, validation_samples=12, extrapolation_samples=12),
            seed=31,
        )
        model = build_symbolic_model(SymbolicModelFamily.PAPER_COMPLEX_EML, depth=1, seed=37, hidden_units=4)
        params = model.parameters()
        root = model.root
        assert root.left_selector is not None and root.right_selector is not None
        for index in range(4):
            params[root.left_selector + index] = -4.0
            params[root.right_selector + index] = -4.0
        params[root.left_selector] = 4.0
        params[root.right_selector + 1] = 4.0
        params[model.readout_offset] = 0.5
        params[model.readout_offset + 1] = 0.25
        model.set_parameters(params)

        refit = refit_paper_readout_after_hardening(model, dataset.train, model.harden())
        compiled = refit.compile()
        metrics = evaluate_predictions(
            tuple(compiled(value) for value in dataset.validation.xs),
            dataset.validation.ys,
        )

        self.assertLess(metrics.rmse, 1.0e-6)

    def test_depth2_paper_hardening_search_can_recover_nested_eml_target(self) -> None:
        tasks = {task.task_id: task for task in tier0_exact_recovery_tasks()}
        task = tasks["t0_nested_exp"]
        dataset = sample_symbolic_dataset(
            task,
            SymbolicDatasetSpec(train_samples=12, validation_samples=12, extrapolation_samples=12),
            seed=41,
        )

        result = run_single_symbolic_case(
            dataset=dataset,
            model_family=SymbolicModelFamily.PAPER_COMPLEX_EML,
            seed=43,
            train_spec=SymbolicTrainSpec(
                steps=1,
                hidden_units=4,
                tree_optimizer=SymbolicTreeOptimizer.AUTODIFF,
            ),
        )

        self.assertTrue(result.exact_recovery)
        self.assertIn("eml(eml(x, 1), 1)", result.expression)

    def test_depth2_paper_search_space_contains_nested_candidates(self) -> None:
        candidates = enumerate_paper_root_candidates(2)
        expressions = {candidate.expression for candidate in candidates}

        self.assertIn("eml(eml(x, 1), 1)", expressions)
        self.assertIn("eml(1, eml(1, x))", expressions)
        self.assertIn("eml(((2 * ((2.7182818 - eml(1, x))))), 1)", expressions)
        self.assertIn(
            "eml((((2.7182818 - eml(1, x))) + (-((2.7182818 - eml(1, (x + 2)))))), 1)",
            expressions,
        )

    def test_log_lift_search_can_recover_square_and_safe_ratio_controls(self) -> None:
        tasks = {task.task_id: task for task in tier0_exact_recovery_tasks()}
        for task_id in ("t0_square", "t0_safe_ratio"):
            with self.subTest(task_id=task_id):
                dataset = sample_symbolic_dataset(
                    tasks[task_id],
                    SymbolicDatasetSpec(train_samples=12, validation_samples=16, extrapolation_samples=16),
                    seed=47,
                )

                result = run_single_symbolic_case(
                    dataset=dataset,
                    model_family=SymbolicModelFamily.PAPER_COMPLEX_EML,
                    seed=53,
                    train_spec=SymbolicTrainSpec(
                        steps=1,
                        hidden_units=4,
                        tree_optimizer=SymbolicTreeOptimizer.AUTODIFF,
                    ),
                )

                self.assertTrue(result.exact_recovery)

    def test_sparse_readout_recovers_compact_product_and_shifted_ratio(self) -> None:
        tasks = {task.task_id: task for task in default_symbolic_tasks(tasks_per_depth=2)}
        for task_id in ("d2_quadratic_mix", "d2_reciprocal_shift"):
            with self.subTest(task_id=task_id):
                dataset = sample_symbolic_dataset(
                    tasks[task_id],
                    SymbolicDatasetSpec(train_samples=12, validation_samples=16, extrapolation_samples=16),
                    seed=59,
                )

                result = run_single_symbolic_case(
                    dataset=dataset,
                    model_family=SymbolicModelFamily.PAPER_COMPLEX_EML,
                    seed=61,
                    train_spec=SymbolicTrainSpec(
                        steps=1,
                        hidden_units=4,
                        tree_optimizer=SymbolicTreeOptimizer.AUTODIFF,
                    ),
                )

                self.assertTrue(result.exact_recovery)


class SymbolicEvaluationTests(unittest.TestCase):
    def test_error_metrics_track_finite_fraction(self) -> None:
        metrics = evaluate_predictions((1.0, math.nan, 3.0), (1.5, 2.0, 2.0))

        self.assertEqual(metrics.finite_fraction, 2 / 3)
        self.assertGreater(metrics.rmse, 0.0)

    def test_single_case_trains_and_exports(self) -> None:
        task = default_symbolic_tasks(tasks_per_depth=1)[0]
        dataset = sample_symbolic_dataset(
            task,
            SymbolicDatasetSpec(train_samples=8, validation_samples=8, extrapolation_samples=8),
            seed=5,
        )

        result = run_single_symbolic_case(
            dataset=dataset,
            model_family=SymbolicModelFamily.SMALL_MLP,
            seed=11,
            train_spec=SymbolicTrainSpec(steps=4, hidden_units=4),
        )

        self.assertEqual(result.task_id, task.task_id)
        self.assertTrue(result.compile_success)
        self.assertGreaterEqual(result.parameter_count, 1)

    def test_benchmark_writes_artifact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "symbolic"
            manifest = SymbolicBenchmarkManifest(
                run_label="unit-symbolic",
                preset=SymbolicPreset.SMOKE,
                model_families=(SymbolicModelFamily.GENERIC_TREE,),
                seeds=(1,),
                dataset=SymbolicDatasetSpec(
                    train_samples=6,
                    validation_samples=6,
                    extrapolation_samples=6,
                    tasks_per_depth=1,
                ),
                train=SymbolicTrainSpec(steps=2, hidden_units=4),
            )

            report = run_symbolic_benchmark(manifest, output_dir)

            self.assertEqual(report.summary["total_runs"], 4)
            self.assertIn("per_depth", report.summary)
            self.assertIn("symbolic_recovery_leader", report.summary)
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())

    def test_tier0_benchmark_uses_exact_recovery_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "tier0"
            manifest = SymbolicBenchmarkManifest(
                run_label="unit-tier0",
                preset=SymbolicPreset.TIER0_EXACT,
                model_families=(SymbolicModelFamily.PAPER_COMPLEX_EML,),
                seeds=(1,),
                dataset=SymbolicDatasetSpec(
                    train_samples=5,
                    validation_samples=5,
                    extrapolation_samples=5,
                    tasks_per_depth=1,
                ),
                train=SymbolicTrainSpec(
                    steps=1,
                    hidden_units=4,
                    tree_optimizer=SymbolicTreeOptimizer.AUTODIFF,
                ),
            )

            report = run_symbolic_benchmark(manifest, output_dir)

            self.assertEqual(report.summary["total_runs"], len(tier0_exact_recovery_tasks()))
            task_ids = {result.task_id for result in report.results}
            self.assertIn("t0_exp_x", task_ids)

    def test_symbolic_bridge_scores_compiled_summary_as_token_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "summary.json"
            output_dir = root / "bridge"
            source_path.write_text(
                json.dumps(
                    {
                        "manifest": {
                            "preset": "tier0-exact",
                            "seeds": [42],
                            "dataset": {
                                "train_samples": 6,
                                "validation_samples": 6,
                                "extrapolation_samples": 6,
                                "tasks_per_depth": 1,
                            },
                        },
                        "results": [
                            {
                                "task_id": "t0_exp_x",
                                "model_family": "paper-complex-eml",
                                "seed": 42,
                                "expression": "exp(x)",
                                "expression_source": "lambda x: float(math.exp(x))",
                                "complexity": 2,
                                "active_ops": ["exp"],
                                "export_success": True,
                                "exact_recovery": True,
                            }
                        ],
                    }
                )
            )

            report = run_symbolic_bridge(source_path, output_dir, run_label="unit-bridge", token_bins=8)

            self.assertEqual(report.summary["best_extrapolation_token_accuracy"], "paper-complex-eml")
            self.assertIn("frozen_side_channel_probe", report.summary)
            self.assertIn("router_target_probe", report.summary)
            self.assertIn("feature_table", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "feature_table.jsonl").exists())
            feature_rows = [
                json.loads(line)
                for line in (output_dir / "feature_table.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertIn("safety_calibration", {row["split"] for row in feature_rows})
            self.assertIn("split_safe_expert_coverage", report.summary["feature_table"])

    def test_safety_calibration_prefers_out_of_range_safe_rows_near_abstain(self) -> None:
        rows = [
            {"index": 0, "x": -5.0, "in_training_range": False, "oracle_has_safe_expert": True},
            {"index": 1, "x": 1.9, "in_training_range": False, "oracle_has_safe_expert": True},
            {"index": 2, "x": 2.0, "in_training_range": False, "oracle_has_safe_expert": False},
            {"index": 3, "x": 2.05, "in_training_range": True, "oracle_has_safe_expert": True},
            {"index": 4, "x": 2.2, "in_training_range": False, "oracle_has_safe_expert": False},
            {"index": 5, "x": 2.3, "in_training_range": False, "oracle_has_safe_expert": True},
        ]

        selected = select_safety_calibration_rows(rows, target_count=4)

        selected_xs = {round(float(row["x"]), 2) for row in selected}
        self.assertEqual(selected_xs, {1.9, 2.0, 2.2, 2.3})

    def test_token_quantizer_clamps_to_vocab_range(self) -> None:
        quantizer = TokenQuantizer.from_values((0.0, 1.0), bins=4)

        self.assertEqual(quantizer.encode(-10.0), 0)
        self.assertEqual(quantizer.encode(10.0), 3)
        self.assertEqual(quantizer.encode(math.nan), -1)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for default unit tests")
    def test_symbolic_bridge_canary_trains_from_feature_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_table = root / "feature_table.jsonl"
            bridge_summary = root / "summary.json"
            output_dir = root / "canary"
            rows = []
            for split, values in {
                "train": (-1.0, -0.5, 0.5, 1.0),
                "validation": (-0.75, -0.25, 0.25, 0.75),
                "extrapolation": (-1.5, -1.25, 1.25, 1.5),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split != "extrapolation",
                            "best_expert_id": "paper-complex-eml",
                            "oracle_has_safe_expert": True,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": True,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": True,
                                },
                            },
                        }
                    )
            with feature_table.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(feature_table),
                                "row_count": len(rows),
                                "safe_expert_coverage": 1.0,
                            }
                        },
                    }
                )
            )

            report = run_bridge_canary(
                bridge_summary,
                output_dir,
                run_label="unit-canary",
                epochs=8,
                learning_rate=0.03,
                device="cpu",
            )

            self.assertIn("best_trained_extrapolation_token_accuracy", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for default unit tests")
    def test_symbolic_sequence_bridge_trains_router_call_from_feature_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_table = root / "feature_table.jsonl"
            bridge_summary = root / "summary.json"
            output_dir = root / "sequence"
            rows = []
            for split, values in {
                "train": (-1.0, -0.5, 0.5, 1.0),
                "validation": (-0.75, -0.25, 0.25, 0.75),
                "extrapolation": (-1.5, -1.25, 1.25, 1.5),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split != "extrapolation",
                            "best_expert_id": "paper-complex-eml",
                            "oracle_has_safe_expert": True,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": True,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": True,
                                },
                            },
                        }
                    )
            with feature_table.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(feature_table),
                                "row_count": len(rows),
                                "safe_expert_coverage": 1.0,
                            }
                        },
                    }
                )
            )

            report = run_sequence_bridge(
                bridge_summary,
                output_dir,
                run_label="unit-sequence",
                epochs=4,
                learning_rate=0.02,
                hidden_units=4,
                device="cpu",
            )

            self.assertIn("best_trained_router_extrapolation_accuracy", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for default unit tests")
    def test_symbolic_bridge_lm_contract_records_abstain_and_unsafe_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_table = root / "feature_table.jsonl"
            bridge_summary = root / "summary.json"
            extra_feature_table = root / "extra_feature_table.jsonl"
            extra_bridge_summary = root / "extra_summary.json"
            output_dir = root / "bridge-lm"
            rows = []
            for split, values in {
                "train": (-1.0, -0.5, 0.5, 1.0),
                "validation": (-0.75, -0.25, 0.25, 0.75),
                "extrapolation": (-1.5, -1.25, 1.25, 1.5),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    has_safe_expert = x_value < 1.0
                    rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split != "extrapolation",
                            "best_expert_id": "paper-complex-eml" if has_safe_expert else "generic-tree",
                            "oracle_has_safe_expert": has_safe_expert,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": has_safe_expert,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": has_safe_expert,
                                },
                            },
                        }
                    )
            with feature_table.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(feature_table),
                                "row_count": len(rows),
                                "safe_expert_coverage": 0.75,
                            }
                        },
                    }
                )
            )
            extra_rows = []
            for split, values in {
                "train": (-0.9, 0.9),
                "safety_calibration": (-1.1, 1.1),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    has_safe_expert = abs(x_value) <= 1.0
                    extra_rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 2,
                            "split": split,
                            "sequence_id": f"extra-{split}",
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split == "train",
                            "best_expert_id": "paper-complex-eml" if has_safe_expert else "generic-tree",
                            "oracle_has_safe_expert": has_safe_expert,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": has_safe_expert,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": has_safe_expert,
                                },
                            },
                        }
                    )
            with extra_feature_table.open("w") as handle:
                for row in extra_rows:
                    handle.write(json.dumps(row) + "\n")
            extra_bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(extra_feature_table),
                                "row_count": len(extra_rows),
                                "safe_expert_coverage": 0.5,
                            }
                        },
                    }
                )
            )

            report = run_symbolic_bridge_lm(
                bridge_summary,
                output_dir,
                run_label="unit-bridge-lm",
                epochs=4,
                learning_rate=0.003,
                hidden_units=4,
                abstain_class_weight=2.0,
                unsafe_call_loss_weight=0.5,
                call_abstain_loss_weight=0.75,
                answer_call_abstain_loss_weight=0.8,
                answer_unsafe_loss_weight=0.6,
                non_answer_abstain_loss_weight=0.7,
                unsafe_margin_loss_weight=1.25,
                unsafe_margin=0.4,
                router_call_threshold=0.25,
                expert_logit_scale=3.0,
                device="cpu",
                extra_fit_bridge_summary_paths=(extra_bridge_summary,),
            )

            self.assertEqual(report.backbone, "gru")
            self.assertEqual(report.backbone_config["type"], "gru")
            self.assertEqual(report.abstain_index, 2)
            self.assertEqual(report.abstain_class_weight, 2.0)
            self.assertEqual(report.unsafe_call_loss_weight, 0.5)
            self.assertEqual(report.call_abstain_loss_weight, 0.75)
            self.assertEqual(report.answer_call_abstain_loss_weight, 0.8)
            self.assertEqual(report.answer_unsafe_loss_weight, 0.6)
            self.assertEqual(report.non_answer_abstain_loss_weight, 0.7)
            self.assertEqual(report.unsafe_margin_loss_weight, 1.25)
            self.assertEqual(report.unsafe_margin, 0.4)
            self.assertEqual(report.router_call_threshold, 0.25)
            self.assertEqual(report.expert_logit_scale, 3.0)
            self.assertEqual(report.extra_fit_bridge_summary_paths, (str(extra_bridge_summary.resolve()),))
            self.assertEqual(report.extra_fit_feature_table_paths, (str(extra_feature_table.resolve()),))
            self.assertEqual(report.summary["extra_fit_row_count"], len(extra_rows))
            self.assertEqual(report.summary["fit_row_count"], 4 + len(extra_rows))
            run_names = {run.name for run in report.runs}
            self.assertIn("lm-router-logit-fusion", run_names)
            self.assertIn("lm-router-prob-mixture", run_names)
            prob_mixture = next(run for run in report.runs if run.name == "lm-router-prob-mixture")
            self.assertIsNotNone(prob_mixture.role_metrics)
            assert prob_mixture.role_metrics is not None
            self.assertIn("symbolic", prob_mixture.role_metrics["extrapolation"])
            self.assertIn("router_contract_unsafe_call_rate", report.summary)
            self.assertIn("router_contract_abstain_recall", report.summary)
            self.assertIn("best_extrapolation_final_nll", report.summary)
            self.assertIn("logit_fusion_extrapolation_nll_delta_vs_side_channel", report.summary)
            self.assertIn("prob_mixture_extrapolation_nll_delta_vs_side_channel", report.summary)
            self.assertIn("prob_mixture_extrapolation_nll_delta_vs_token_only", report.summary)
            self.assertIn("math_answer_extrapolation_metrics", report.summary)
            self.assertIn("prob_mixture_math_answer_contract_confirmed", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())

    def test_bridge_corpus_v1_generates_language_math_and_math_only_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_feature_table = root / "source_feature_table.jsonl"
            source_summary = root / "source_summary.json"
            source_rows = []
            for split in ("train", "safety_calibration", "validation", "extrapolation"):
                for index in range(2):
                    target_token = 2 + index if split != "extrapolation" else 3 - index
                    source_rows.append(
                        {
                            "task_id": "toy_formula",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": 0.5 + index,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 4,
                            "quantizer": {"minimum": 0.0, "maximum": 4.0},
                            "in_training_range": split in {"train", "validation"},
                            "best_expert_id": "paper-complex-eml",
                            "oracle_has_safe_expert": True,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": True,
                                "small-mlp": False,
                                "stable-real-eml": False,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 0.0,
                                    "token": 0,
                                    "residual": 1.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": True,
                                },
                                "small-mlp": {
                                    "prediction": 1.0,
                                    "token": 1,
                                    "residual": 1.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "stable-real-eml": {
                                    "prediction": 1.0,
                                    "token": 1,
                                    "residual": 1.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                            },
                        }
                    )
            with source_feature_table.open("w") as handle:
                for row in source_rows:
                    handle.write(json.dumps(row) + "\n")
            source_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 4,
                        "summary": {
                            "feature_table": {
                                "path": str(source_feature_table),
                                "row_count": len(source_rows),
                            }
                        },
                    }
                )
            )

            language_report = run_bridge_corpus(
                root / "language-math",
                run_label="unit-language-math",
                corpus_kind="language-math",
                source_bridge_summary_path=source_summary,
                language_train_per_group=2,
                language_safety_per_group=2,
                language_eval_per_group=2,
            )
            math_report = run_bridge_corpus(
                root / "math-only",
                run_label="unit-math-only",
                corpus_kind="math-only",
                source_bridge_summary_path=source_summary,
            )
            pure_report = run_bridge_corpus(
                root / "pure-language",
                run_label="unit-pure-language",
                corpus_kind="pure-language",
                pure_sequences_per_split=2,
            )
            ablation_report = run_bridge_corpus(
                root / "paper-only",
                run_label="unit-paper-only",
                corpus_kind="expert-ablation",
                source_corpus_summary_path=root / "language-math" / "summary.json",
                expert_subset=("paper-complex-eml",),
            )
            shuffle_report = run_bridge_corpus(
                root / "paper-shuffled",
                run_label="unit-paper-shuffled",
                corpus_kind="expert-shuffle",
                source_corpus_summary_path=root / "language-math" / "summary.json",
                expert_subset=("paper-complex-eml",),
                shuffle_seed=9,
            )
            target_randomized_report = run_bridge_corpus(
                root / "target-randomized",
                run_label="unit-target-randomized",
                corpus_kind="target-randomized",
                source_corpus_summary_path=root / "language-math" / "summary.json",
                shuffle_seed=11,
            )
            wrong_expert_report = run_bridge_corpus(
                root / "wrong-expert",
                run_label="unit-wrong-expert",
                corpus_kind="wrong-expert",
                source_corpus_summary_path=root / "language-math" / "summary.json",
                shuffle_seed=13,
            )

            self.assertIn("math_answer", language_report.summary["feature_table"]["role_counts"])
            self.assertGreater(language_report.summary["feature_table"]["split_safe_expert_coverage"]["train"], 0.0)
            self.assertEqual(math_report.summary["feature_table"]["role_counts"], {"math_only": 8})
            self.assertEqual(pure_report.summary["feature_table"]["role_counts"]["prose"], 72)
            self.assertEqual(pure_report.summary["feature_table"]["safe_expert_coverage"], 0.0)
            self.assertEqual(ablation_report.summary["expert_transform"]["selected_experts"], ["paper-complex-eml"])
            self.assertEqual(shuffle_report.summary["expert_transform"]["shuffle_seed"], 9)
            with Path(ablation_report.feature_table_path).open() as handle:
                ablation_rows = [json.loads(line) for line in handle if line.strip()]
            with Path(shuffle_report.feature_table_path).open() as handle:
                shuffled_rows = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(all(tuple(row["experts"]) == ("paper-complex-eml",) for row in ablation_rows))
            self.assertTrue(all(tuple(row["safe_expert_mask"]) == ("paper-complex-eml",) for row in shuffled_rows))
            shuffled_answers = [row for row in shuffled_rows if row["eval_role"] == "math_answer"]
            self.assertTrue(any(not row["oracle_has_safe_expert"] for row in shuffled_answers))
            self.assertEqual(target_randomized_report.summary["expert_transform"]["control_kind"], "target-randomized")
            self.assertGreater(target_randomized_report.summary["expert_transform"]["target_changed_rate"], 0.0)
            self.assertEqual(wrong_expert_report.summary["expert_transform"]["control_kind"], "wrong-expert")
            with Path(wrong_expert_report.feature_table_path).open() as handle:
                wrong_expert_rows = [json.loads(line) for line in handle if line.strip()]
            wrong_expert_answers = [row for row in wrong_expert_rows if row["eval_role"] == "math_answer"]
            self.assertTrue(any(not row["oracle_has_safe_expert"] for row in wrong_expert_answers))
            self.assertTrue((root / "language-math" / "feature_table.jsonl").exists())
            self.assertTrue((root / "language-math" / "summary.json").exists())

    def test_bridge_corpus_heldout_templates_separate_formula_and_language_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_feature_table = root / "source_feature_table.jsonl"
            source_summary = root / "source_summary.json"
            source_rows = []
            task_ids = ("d1_alpha", "d1_beta", "d2_alpha", "d2_beta")
            for task_offset, task_id in enumerate(task_ids):
                for split in ("train", "safety_calibration", "validation", "extrapolation"):
                    for index in range(3):
                        target_token = (task_offset + index) % 4
                        source_rows.append(
                            {
                                "task_id": task_id,
                                "seed": 1,
                                "split": split,
                                "index": index,
                                "x": 0.25 + index,
                                "target_y": float(target_token),
                                "target_token": target_token,
                                "token_bins": 4,
                                "quantizer": {"minimum": 0.0, "maximum": 4.0},
                                "in_training_range": split in {"train", "validation"},
                                "best_expert_id": "paper-complex-eml",
                                "oracle_has_safe_expert": True,
                                "safe_expert_mask": {
                                    "generic-tree": False,
                                    "paper-complex-eml": True,
                                    "small-mlp": False,
                                    "stable-real-eml": False,
                                },
                                "experts": {
                                    "generic-tree": {
                                        "prediction": 0.0,
                                        "token": 0,
                                        "residual": 1.0,
                                        "abs_residual": 1.0,
                                        "token_match": False,
                                    },
                                    "paper-complex-eml": {
                                        "prediction": float(target_token),
                                        "token": target_token,
                                        "residual": 0.0,
                                        "abs_residual": 0.0,
                                        "token_match": True,
                                    },
                                    "small-mlp": {
                                        "prediction": 1.0,
                                        "token": 1,
                                        "residual": 1.0,
                                        "abs_residual": 1.0,
                                        "token_match": False,
                                    },
                                    "stable-real-eml": {
                                        "prediction": 1.0,
                                        "token": 1,
                                        "residual": 1.0,
                                        "abs_residual": 1.0,
                                        "token_match": False,
                                    },
                                },
                            }
                        )
            with source_feature_table.open("w") as handle:
                for row in source_rows:
                    handle.write(json.dumps(row) + "\n")
            source_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 4,
                        "summary": {
                            "feature_table": {
                                "path": str(source_feature_table),
                                "row_count": len(source_rows),
                            }
                        },
                    }
                )
            )

            report = run_bridge_corpus(
                root / "heldout",
                run_label="unit-heldout",
                corpus_kind="language-math-heldout-templates",
                source_bridge_summary_path=source_summary,
                seed=5,
                language_train_per_group=2,
                language_safety_per_group=2,
                language_eval_per_group=2,
            )
            with Path(report.feature_table_path).open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            train_tasks = {row["task_id"] for row in rows if row["split"] in {"train", "safety_calibration"}}
            eval_tasks = {row["task_id"] for row in rows if row["split"] in {"validation", "extrapolation"}}
            self.assertTrue(train_tasks)
            self.assertTrue(eval_tasks)
            self.assertTrue(train_tasks.isdisjoint(eval_tasks))
            self.assertEqual(
                {row["formula_template_split"] for row in rows if row["split"] == "train"},
                {"seen_formula"},
            )
            self.assertEqual(
                {row["language_template_split"] for row in rows if row["split"] == "extrapolation"},
                {"heldout_language_template"},
            )
            self.assertIn("heldout_template", report.summary)
            self.assertIn("math_answer_index_counts", report.summary["feature_table"])
            self.assertGreater(len(report.summary["feature_table"]["math_answer_index_counts"]), 1)
            variance_a = run_bridge_corpus(
                root / "heldout-variance-a",
                run_label="unit-heldout-variance-a",
                corpus_kind="language-math-heldout-variance",
                source_bridge_summary_path=source_summary,
                seed=1,
                language_train_per_group=1,
                language_safety_per_group=1,
                language_eval_per_group=1,
            )
            variance_b = run_bridge_corpus(
                root / "heldout-variance-b",
                run_label="unit-heldout-variance-b",
                corpus_kind="language-math-heldout-variance",
                source_bridge_summary_path=source_summary,
                seed=2,
                language_train_per_group=1,
                language_safety_per_group=1,
                language_eval_per_group=1,
            )
            self.assertNotEqual(
                variance_a.summary["heldout_template"]["seen_formula_tasks"],
                variance_b.summary["heldout_template"]["seen_formula_tasks"],
            )
            self.assertEqual(variance_a.summary["heldout_template"]["formula_split_rotation_seed"], 1)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for default unit tests")
    def test_symbolic_bridge_lm_transformer_backbone_runs_true_token_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_table = root / "feature_table.jsonl"
            bridge_summary = root / "summary.json"
            output_dir = root / "bridge-lm-transformer"
            rows = []
            for split, values in {
                "train": (-1.0, -0.25, 0.25, 1.0),
                "safety_calibration": (-1.2, 1.2),
                "validation": (-0.75, 0.75),
                "extrapolation": (-1.5, 1.5),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    has_safe_expert = abs(x_value) <= 1.2
                    rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split in {"train", "validation"},
                            "best_expert_id": "paper-complex-eml" if has_safe_expert else "generic-tree",
                            "oracle_has_safe_expert": has_safe_expert,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": has_safe_expert,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": has_safe_expert,
                                },
                            },
                        }
                    )
            with feature_table.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(feature_table),
                                "row_count": len(rows),
                                "safe_expert_coverage": 0.8,
                            }
                        },
                    }
                )
            )

            report = run_symbolic_bridge_lm(
                bridge_summary,
                output_dir,
                run_label="unit-bridge-lm-transformer",
                epochs=2,
                learning_rate=0.003,
                hidden_units=8,
                backbone="transformer",
                transformer_layers=1,
                transformer_heads=2,
                transformer_ffn_multiplier=2,
                device="cpu",
            )

            self.assertEqual(report.backbone, "transformer")
            self.assertEqual(report.backbone_config["type"], "decoder-only-causal-transformer")
            self.assertFalse(report.backbone_config["feature_projection_bias"])
            run_names = {run.name for run in report.runs}
            self.assertIn("lm-token-only", run_names)
            self.assertIn("lm-router-prob-mixture", run_names)
            self.assertIn("best_trained_extrapolation_final_nll", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is optional for default unit tests")
    def test_symbolic_bridge_path1_uses_path1_backbone_contract(self) -> None:
        from python.symbolic.bridge_path1 import run_symbolic_bridge_path1

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            feature_table = root / "feature_table.jsonl"
            bridge_summary = root / "summary.json"
            output_dir = root / "bridge-path1"
            rows = []
            for split, values in {
                "train": (-1.0, -0.5, 0.5, 1.0),
                "safety_calibration": (-1.2, 1.2, 1.4, 1.6),
                "validation": (-0.75, -0.25, 0.25, 0.75),
                "extrapolation": (-1.5, -1.25, 1.25, 1.5),
            }.items():
                for index, x_value in enumerate(values):
                    target_token = 0 if x_value < 0.0 else 1
                    has_safe_expert = x_value < 1.4
                    rows.append(
                        {
                            "task_id": "toy_sign",
                            "seed": 1,
                            "split": split,
                            "index": index,
                            "x": x_value,
                            "target_y": float(target_token),
                            "target_token": target_token,
                            "token_bins": 2,
                            "quantizer": {"minimum": 0.0, "maximum": 1.0},
                            "in_training_range": split in {"train", "validation"},
                            "best_expert_id": "paper-complex-eml" if has_safe_expert else "generic-tree",
                            "oracle_has_safe_expert": has_safe_expert,
                            "safe_expert_mask": {
                                "generic-tree": False,
                                "paper-complex-eml": has_safe_expert,
                            },
                            "experts": {
                                "generic-tree": {
                                    "prediction": 1.0 - float(target_token),
                                    "token": 1 - target_token,
                                    "residual": 0.0,
                                    "abs_residual": 1.0,
                                    "token_match": False,
                                },
                                "paper-complex-eml": {
                                    "prediction": float(target_token),
                                    "token": target_token,
                                    "residual": 0.0,
                                    "abs_residual": 0.0,
                                    "token_match": has_safe_expert,
                                },
                            },
                        }
                    )
            with feature_table.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            bridge_summary.write_text(
                json.dumps(
                    {
                        "token_bins": 2,
                        "summary": {
                            "feature_table": {
                                "path": str(feature_table),
                                "row_count": len(rows),
                                "safe_expert_coverage": 0.75,
                            }
                        },
                    }
                )
            )

            report = run_symbolic_bridge_path1(
                bridge_summary,
                output_dir,
                run_label="unit-bridge-path1",
                epochs=2,
                learning_rate=0.004,
                d_model=16,
                total_layers=1,
                head_count=4,
                ffn_multiplier=2,
                device="cpu",
            )

            self.assertEqual(report.backbone, "path1")
            self.assertIn("path1_variant", report.summary)
            self.assertEqual(report.summary["path1_variant"]["d_model"], 16)
            self.assertIn("router_contract_unsafe_call_rate", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
