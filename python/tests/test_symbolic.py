from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from python.specs.symbolic import (
    SymbolicBenchmarkManifest,
    SymbolicDatasetSpec,
    SymbolicModelFamily,
    SymbolicPreset,
    SymbolicTrainSpec,
    SymbolicTreeOptimizer,
)
from python.symbolic.autodiff import autodiff_loss_and_gradient
from python.symbolic.bridge import TokenQuantizer, run_symbolic_bridge
from python.symbolic.bridge_canary import run_bridge_canary
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

            report = run_symbolic_bridge_lm(
                bridge_summary,
                output_dir,
                run_label="unit-bridge-lm",
                epochs=4,
                learning_rate=0.003,
                hidden_units=4,
                abstain_class_weight=2.0,
                unsafe_call_loss_weight=0.5,
                router_call_threshold=0.25,
                device="cpu",
            )

            self.assertEqual(report.abstain_index, 2)
            self.assertEqual(report.abstain_class_weight, 2.0)
            self.assertEqual(report.unsafe_call_loss_weight, 0.5)
            self.assertEqual(report.router_call_threshold, 0.25)
            self.assertIn("router_contract_unsafe_call_rate", report.summary)
            self.assertIn("router_contract_abstain_recall", report.summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
