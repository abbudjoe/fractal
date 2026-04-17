from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative
from python.specs.symbolic import SymbolicDatasetSpec, SymbolicModelFamily
from python.symbolic.formulas import (
    FormulaDataset,
    FormulaSplit,
    default_symbolic_tasks,
    sample_symbolic_dataset,
    tier0_exact_recovery_tasks,
)
from python.symbolic.models import HardenedExpression
from python.symbolic.runner import compile_expression


@dataclass(frozen=True)
class TokenQuantizer:
    minimum: float
    maximum: float
    bins: int

    @classmethod
    def from_values(cls, values: tuple[float, ...], bins: int) -> "TokenQuantizer":
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return cls(0.0, 1.0, bins)
        minimum = min(finite_values)
        maximum = max(finite_values)
        if maximum - minimum <= 1.0e-12:
            maximum = minimum + 1.0
        return cls(minimum, maximum, bins)

    def encode(self, value: float) -> int:
        if not math.isfinite(value):
            return -1
        fraction = (value - self.minimum) / (self.maximum - self.minimum)
        index = int(math.floor(fraction * self.bins))
        return max(0, min(self.bins - 1, index))


@dataclass(frozen=True)
class BridgeRunResult:
    task_id: str
    model_family: str
    seed: int
    expression: str
    validation_token_accuracy: float
    extrapolation_token_accuracy: float
    validation_token_nll: float
    extrapolation_token_nll: float
    validation_continuous_rmse: float
    extrapolation_continuous_rmse: float
    compile_success: bool
    source_exact_recovery: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_family": self.model_family,
            "seed": self.seed,
            "expression": self.expression,
            "validation_token_accuracy": self.validation_token_accuracy,
            "extrapolation_token_accuracy": self.extrapolation_token_accuracy,
            "validation_token_nll": self.validation_token_nll,
            "extrapolation_token_nll": self.extrapolation_token_nll,
            "validation_continuous_rmse": self.validation_continuous_rmse,
            "extrapolation_continuous_rmse": self.extrapolation_continuous_rmse,
            "compile_success": self.compile_success,
            "source_exact_recovery": self.source_exact_recovery,
        }


@dataclass(frozen=True)
class BridgeReport:
    source_summary_path: str
    run_label: str
    token_bins: int
    results: tuple[BridgeRunResult, ...]
    summary: dict[str, Any]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_summary_path": self.source_summary_path,
            "run_label": self.run_label,
            "token_bins": self.token_bins,
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
            "output_dir": self.output_dir,
        }


def run_symbolic_bridge(
    source_summary_path: Path,
    output_dir: Path,
    *,
    run_label: str,
    token_bins: int = 32,
) -> BridgeReport:
    source = json.loads(source_summary_path.read_text())
    datasets = datasets_from_symbolic_summary(source)
    results: list[BridgeRunResult] = []
    for row in source.get("results", []):
        task_id = row["task_id"]
        dataset = datasets.get(task_id)
        if dataset is None:
            continue
        results.append(evaluate_symbolic_result_on_tokens(row, dataset, token_bins))
    add_majority_baselines(results, datasets, source, token_bins)
    summary = summarize_bridge_results(results)
    source_rows = tuple(row for row in source.get("results", []) if row.get("model_family") != "token-majority")
    summary["frozen_side_channel_probe"] = summarize_side_channel_probe(source_rows, datasets, token_bins)
    summary["router_target_probe"] = summarize_router_target_probe(source_rows, datasets, token_bins)
    feature_rows = build_bridge_feature_table(source_rows, datasets, token_bins)
    summary["feature_table"] = {
        "path": repo_relative(output_dir / "feature_table.jsonl"),
        "row_count": len(feature_rows),
        "split_counts": split_counts(feature_rows),
        "split_safe_expert_coverage": split_safe_expert_coverage(feature_rows),
        "safe_expert_coverage": mean(1.0 if row["oracle_has_safe_expert"] else 0.0 for row in feature_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report = BridgeReport(
        source_summary_path=repo_relative(source_summary_path),
        run_label=run_label,
        token_bins=token_bins,
        results=tuple(results),
        summary=summary,
        output_dir=repo_relative(output_dir),
    )
    write_bridge_report(report, output_dir)
    write_feature_table(feature_rows, output_dir / "feature_table.jsonl")
    return report


def datasets_from_symbolic_summary(source: dict[str, Any]) -> dict[str, FormulaDataset]:
    manifest = source["manifest"]
    dataset_spec = SymbolicDatasetSpec(**manifest["dataset"])
    first_seed = int(manifest["seeds"][0])
    preset = manifest["preset"]
    if preset == "tier0-exact":
        tasks = tier0_exact_recovery_tasks()
    else:
        tasks = default_symbolic_tasks(dataset_spec.tasks_per_depth)
    return {
        task.task_id: sample_symbolic_dataset(
            task,
            dataset_spec,
            seed=first_seed + task.difficulty_depth * 100,
        )
        for task in tasks
    }


def evaluate_symbolic_result_on_tokens(
    row: dict[str, Any],
    dataset: FormulaDataset,
    token_bins: int,
) -> BridgeRunResult:
    model_family = SymbolicModelFamily(row["model_family"])
    expression = HardenedExpression(
        model_family=model_family,
        expression=row["expression"],
        python_source=row["expression_source"],
        complexity=int(row.get("complexity", 1)),
        active_ops=tuple(row.get("active_ops", ())),
        symbolic_export=bool(row.get("export_success", False)),
    )
    func = compile_expression(expression)
    if func is None:
        return BridgeRunResult(
            task_id=row["task_id"],
            model_family=row["model_family"],
            seed=int(row["seed"]),
            expression=row["expression"],
            validation_token_accuracy=0.0,
            extrapolation_token_accuracy=0.0,
            validation_token_nll=math.inf,
            extrapolation_token_nll=math.inf,
            validation_continuous_rmse=math.inf,
            extrapolation_continuous_rmse=math.inf,
            compile_success=False,
            source_exact_recovery=bool(row.get("exact_recovery", False)),
        )
    quantizer = quantizer_for_dataset(dataset, token_bins)
    validation_predictions = tuple(float(func(value)) for value in dataset.validation.xs)
    extrapolation_predictions = tuple(float(func(value)) for value in dataset.extrapolation.xs)
    validation_accuracy = token_accuracy(validation_predictions, dataset.validation.ys, quantizer)
    extrapolation_accuracy = token_accuracy(extrapolation_predictions, dataset.extrapolation.ys, quantizer)
    return BridgeRunResult(
        task_id=row["task_id"],
        model_family=row["model_family"],
        seed=int(row["seed"]),
        expression=row["expression"],
        validation_token_accuracy=validation_accuracy,
        extrapolation_token_accuracy=extrapolation_accuracy,
        validation_token_nll=deterministic_token_nll(validation_accuracy, token_bins),
        extrapolation_token_nll=deterministic_token_nll(extrapolation_accuracy, token_bins),
        validation_continuous_rmse=continuous_rmse(validation_predictions, dataset.validation.ys),
        extrapolation_continuous_rmse=continuous_rmse(extrapolation_predictions, dataset.extrapolation.ys),
        compile_success=True,
        source_exact_recovery=bool(row.get("exact_recovery", False)),
    )


def add_majority_baselines(
    results: list[BridgeRunResult],
    datasets: dict[str, FormulaDataset],
    source: dict[str, Any],
    token_bins: int,
) -> None:
    seeds = tuple(int(seed) for seed in source["manifest"]["seeds"])
    for dataset in datasets.values():
        quantizer = quantizer_for_dataset(dataset, token_bins)
        train_tokens = [quantizer.encode(value) for value in dataset.train.ys]
        majority = max(set(train_tokens), key=train_tokens.count)
        for seed in seeds:
            validation_predictions = token_constant_values(majority, dataset.validation.ys, quantizer)
            extrapolation_predictions = token_constant_values(majority, dataset.extrapolation.ys, quantizer)
            validation_accuracy = token_match_accuracy(majority, dataset.validation.ys, quantizer)
            extrapolation_accuracy = token_match_accuracy(majority, dataset.extrapolation.ys, quantizer)
            results.append(
                BridgeRunResult(
                    task_id=dataset.task.task_id,
                    model_family="token-majority",
                    seed=seed,
                    expression=f"majority_token({majority})",
                    validation_token_accuracy=validation_accuracy,
                    extrapolation_token_accuracy=extrapolation_accuracy,
                    validation_token_nll=deterministic_token_nll(validation_accuracy, token_bins),
                    extrapolation_token_nll=deterministic_token_nll(extrapolation_accuracy, token_bins),
                    validation_continuous_rmse=continuous_rmse(validation_predictions, dataset.validation.ys),
                    extrapolation_continuous_rmse=continuous_rmse(extrapolation_predictions, dataset.extrapolation.ys),
                    compile_success=True,
                    source_exact_recovery=False,
                )
            )


def summarize_side_channel_probe(
    rows: tuple[dict[str, Any], ...],
    datasets: dict[str, FormulaDataset],
    token_bins: int,
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        dataset = datasets.get(row["task_id"])
        if dataset is None:
            continue
        compiled = compiled_row_function(row)
        if compiled is None:
            continue
        quantizer = quantizer_for_dataset(dataset, token_bins)
        train_predictions = tuple(float(compiled(value)) for value in dataset.train.xs)
        validation_predictions = tuple(float(compiled(value)) for value in dataset.validation.xs)
        extrapolation_predictions = tuple(float(compiled(value)) for value in dataset.extrapolation.xs)
        baseline_fit = fit_linear_readout((dataset.train.xs,), dataset.train.ys)
        side_fit = fit_linear_readout((dataset.train.xs, train_predictions), dataset.train.ys)
        if baseline_fit is None or side_fit is None:
            continue
        baseline_coefficients, baseline_bias = baseline_fit
        side_coefficients, side_bias = side_fit
        baseline_validation = apply_linear_readout((dataset.validation.xs,), baseline_coefficients, baseline_bias)
        baseline_extrapolation = apply_linear_readout((dataset.extrapolation.xs,), baseline_coefficients, baseline_bias)
        side_validation = apply_linear_readout((dataset.validation.xs, validation_predictions), side_coefficients, side_bias)
        side_extrapolation = apply_linear_readout((dataset.extrapolation.xs, extrapolation_predictions), side_coefficients, side_bias)
        by_model.setdefault(row["model_family"], []).append(
            {
                "baseline_validation_token_accuracy": token_accuracy(baseline_validation, dataset.validation.ys, quantizer),
                "baseline_extrapolation_token_accuracy": token_accuracy(baseline_extrapolation, dataset.extrapolation.ys, quantizer),
                "side_validation_token_accuracy": token_accuracy(side_validation, dataset.validation.ys, quantizer),
                "side_extrapolation_token_accuracy": token_accuracy(side_extrapolation, dataset.extrapolation.ys, quantizer),
                "side_validation_rmse": continuous_rmse(side_validation, dataset.validation.ys),
                "side_extrapolation_rmse": continuous_rmse(side_extrapolation, dataset.extrapolation.ys),
            }
        )
    summaries = {
        model: {
            "run_count": len(model_rows),
            "median_baseline_validation_token_accuracy": median(row["baseline_validation_token_accuracy"] for row in model_rows),
            "median_baseline_extrapolation_token_accuracy": median(row["baseline_extrapolation_token_accuracy"] for row in model_rows),
            "median_side_validation_token_accuracy": median(row["side_validation_token_accuracy"] for row in model_rows),
            "median_side_extrapolation_token_accuracy": median(row["side_extrapolation_token_accuracy"] for row in model_rows),
            "median_side_validation_rmse": median(row["side_validation_rmse"] for row in model_rows),
            "median_side_extrapolation_rmse": median(row["side_extrapolation_rmse"] for row in model_rows),
        }
        for model, model_rows in sorted(by_model.items())
        if model_rows
    }
    leader = max(
        summaries,
        key=lambda model: summaries[model]["median_side_extrapolation_token_accuracy"],
    ) if summaries else ""
    return {"model_families": summaries, "best_side_channel_extrapolation": leader}


def summarize_router_target_probe(
    rows: tuple[dict[str, Any], ...],
    datasets: dict[str, FormulaDataset],
    token_bins: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task_id"], int(row["seed"])), []).append(row)
    routed: list[dict[str, Any]] = []
    oracle: list[dict[str, float]] = []
    selected_counts: dict[str, int] = {}
    for (task_id, seed), group in sorted(grouped.items()):
        dataset = datasets.get(task_id)
        if dataset is None:
            continue
        candidates = compile_router_candidates(group, dataset, token_bins)
        if not candidates:
            continue
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["train_token_accuracy"],
                -candidate["train_rmse"],
            ),
        )
        selected_counts[selected["model_family"]] = selected_counts.get(selected["model_family"], 0) + 1
        routed.append(
            {
                "task_id": task_id,
                "seed": seed,
                "selected_model_family": selected["model_family"],
                "train_token_accuracy": selected["train_token_accuracy"],
                "validation_token_accuracy": selected["validation_token_accuracy"],
                "extrapolation_token_accuracy": selected["extrapolation_token_accuracy"],
                "validation_rmse": selected["validation_rmse"],
                "extrapolation_rmse": selected["extrapolation_rmse"],
            }
        )
        oracle.append(oracle_router_scores(candidates, dataset, token_bins))
    return {
        "run_count": len(routed),
        "selected_model_counts": selected_counts,
        "median_train_token_accuracy": median(row["train_token_accuracy"] for row in routed) if routed else 0.0,
        "median_validation_token_accuracy": median(row["validation_token_accuracy"] for row in routed) if routed else 0.0,
        "median_extrapolation_token_accuracy": median(row["extrapolation_token_accuracy"] for row in routed) if routed else 0.0,
        "mean_validation_token_accuracy": mean(row["validation_token_accuracy"] for row in routed) if routed else 0.0,
        "mean_extrapolation_token_accuracy": mean(row["extrapolation_token_accuracy"] for row in routed) if routed else 0.0,
        "median_validation_rmse": median(row["validation_rmse"] for row in routed) if routed else math.inf,
        "median_extrapolation_rmse": median(row["extrapolation_rmse"] for row in routed) if routed else math.inf,
        "oracle_validation_token_accuracy": median(row["validation_token_accuracy"] for row in oracle) if oracle else 0.0,
        "oracle_extrapolation_token_accuracy": median(row["extrapolation_token_accuracy"] for row in oracle) if oracle else 0.0,
        "oracle_mean_validation_token_accuracy": mean(row["validation_token_accuracy"] for row in oracle) if oracle else 0.0,
        "oracle_mean_extrapolation_token_accuracy": mean(row["extrapolation_token_accuracy"] for row in oracle) if oracle else 0.0,
        "routes": routed,
    }


def build_bridge_feature_table(
    rows: tuple[dict[str, Any], ...],
    datasets: dict[str, FormulaDataset],
    token_bins: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task_id"], int(row["seed"])), []).append(row)
    feature_rows: list[dict[str, Any]] = []
    for (task_id, seed), group in sorted(grouped.items()):
        dataset = datasets.get(task_id)
        if dataset is None:
            continue
        compiled = tuple(
            (row["model_family"], compiled_row_function(row))
            for row in group
        )
        compiled = tuple((model, func) for model, func in compiled if func is not None)
        if not compiled:
            continue
        quantizer = quantizer_for_dataset(dataset, token_bins)
        safety_split = sample_safety_calibration_split(dataset, target_count=len(dataset.train.xs))
        for split_name, xs, ys in (("train", dataset.train.xs, dataset.train.ys),):
            feature_rows.extend(
                feature_rows_for_split(
                    task_id=task_id,
                    seed=seed,
                    split_name=split_name,
                    xs=xs,
                    ys=ys,
                    dataset=dataset,
                    compiled=compiled,
                    quantizer=quantizer,
                    token_bins=token_bins,
                )
            )
        safety_candidate_rows = feature_rows_for_split(
            task_id=task_id,
            seed=seed,
            split_name="safety_calibration",
            xs=safety_split.xs,
            ys=safety_split.ys,
            dataset=dataset,
            compiled=compiled,
            quantizer=quantizer,
            token_bins=token_bins,
        )
        feature_rows.extend(select_safety_calibration_rows(safety_candidate_rows, target_count=len(dataset.train.xs)))
        for split_name, xs, ys in (
            ("validation", dataset.validation.xs, dataset.validation.ys),
            ("extrapolation", dataset.extrapolation.xs, dataset.extrapolation.ys),
        ):
            feature_rows.extend(
                feature_rows_for_split(
                    task_id=task_id,
                    seed=seed,
                    split_name=split_name,
                    xs=xs,
                    ys=ys,
                    dataset=dataset,
                    compiled=compiled,
                    quantizer=quantizer,
                    token_bins=token_bins,
                )
            )
    return feature_rows


def feature_rows_for_split(
    *,
    task_id: str,
    seed: int,
    split_name: str,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    dataset: FormulaDataset,
    compiled: tuple[tuple[str, Any], ...],
    quantizer: TokenQuantizer,
    token_bins: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (x_value, target) in enumerate(zip(xs, ys)):
        target_token = quantizer.encode(target)
        experts = {
            model: expert_feature_payload(func, x_value, target, target_token, quantizer)
            for model, func in compiled
        }
        best_expert_id = best_expert_for_payload(experts)
        safe_mask = {
            model: payload["token"] == target_token
            for model, payload in experts.items()
        }
        rows.append(
            {
                "task_id": task_id,
                "seed": seed,
                "split": split_name,
                "index": index,
                "x": x_value,
                "target_y": target,
                "target_token": target_token,
                "token_bins": token_bins,
                "quantizer": {
                    "minimum": quantizer.minimum,
                    "maximum": quantizer.maximum,
                },
                "in_training_range": dataset.task.train_range[0] <= x_value <= dataset.task.train_range[1],
                "best_expert_id": best_expert_id,
                "oracle_has_safe_expert": any(safe_mask.values()),
                "safe_expert_mask": safe_mask,
                "experts": experts,
            }
        )
    return rows


def sample_safety_calibration_split(dataset: FormulaDataset, *, target_count: int) -> FormulaSplit:
    task = dataset.task
    train_start, train_end = task.train_range
    extrap_start, extrap_end = task.extrapolation_range
    train_span = max(abs(train_end - train_start), 1.0e-6)
    extrap_span = max(abs(extrap_end - extrap_start), 1.0e-6)
    right_extrapolation = ((extrap_start + extrap_end) * 0.5) >= ((train_start + train_end) * 0.5)
    if right_extrapolation:
        candidate_ranges = (
            (train_start, train_end),
            (train_end + 0.08 * train_span, extrap_start - 0.08 * extrap_span),
            (extrap_start, extrap_end),
            (extrap_end + 0.08 * extrap_span, extrap_end + 0.95 * max(train_span, extrap_span)),
        )
    else:
        candidate_ranges = (
            (train_start, train_end),
            (extrap_end + 0.08 * extrap_span, train_start - 0.08 * train_span),
            (extrap_start, extrap_end),
            (extrap_start - 0.95 * max(train_span, extrap_span), extrap_start - 0.08 * extrap_span),
        )
    existing = {
        round(value, 12)
        for value in dataset.train.xs + dataset.validation.xs + dataset.extrapolation.xs
    }
    candidates: list[tuple[float, float]] = []
    per_range_count = max(target_count * 2, 24)
    for value_range in candidate_ranges:
        start, end = value_range
        if end <= start:
            continue
        for index in range(per_range_count):
            fraction = (index + 0.37) / per_range_count
            x_value = start + fraction * (end - start)
            if round(x_value, 12) in existing:
                continue
            target = task.expression.eval(x_value)
            if math.isfinite(x_value) and math.isfinite(target):
                candidates.append((x_value, target))
    if len(candidates) < target_count:
        raise ValueError(f"could not generate safety calibration split for task {task.task_id}")
    return FormulaSplit(
        xs=tuple(x_value for x_value, _target in candidates),
        ys=tuple(target for _x_value, target in candidates),
    )


def select_safety_calibration_rows(rows: list[dict[str, Any]], *, target_count: int) -> list[dict[str, Any]]:
    safe_rows = sorted(
        (row for row in rows if row["oracle_has_safe_expert"]),
        key=lambda row: (bool(row["in_training_range"]), float(row["x"])),
    )
    abstain_rows = sorted(
        (row for row in rows if not row["oracle_has_safe_expert"]),
        key=lambda row: (bool(row["in_training_range"]), float(row["x"])),
    )
    desired_abstain = 0
    if abstain_rows:
        desired_abstain = min(len(abstain_rows), max(1, int(round(target_count * 0.85))))
    selected = abstain_rows[:desired_abstain]
    selected.extend(safe_rows[: max(0, target_count - len(selected))])
    if len(selected) < target_count:
        selected_ids = {id(row) for row in selected}
        selected.extend(row for row in rows if id(row) not in selected_ids)
    selected = sorted(selected[:target_count], key=lambda row: float(row["x"]))
    reindexed = []
    for index, row in enumerate(selected):
        row_copy = dict(row)
        row_copy["index"] = index
        reindexed.append(row_copy)
    return reindexed


def expert_feature_payload(
    func: Any,
    x_value: float,
    target: float,
    target_token: int,
    quantizer: TokenQuantizer,
) -> dict[str, Any]:
    try:
        prediction = float(func(x_value))
    except (ArithmeticError, ValueError, OverflowError):
        prediction = math.nan
    finite_prediction = math.isfinite(prediction)
    token = quantizer.encode(prediction)
    residual = prediction - target if finite_prediction else math.inf
    return {
        "prediction": prediction if finite_prediction else None,
        "token": token,
        "residual": residual if math.isfinite(residual) else None,
        "abs_residual": abs(residual) if math.isfinite(residual) else 1.0e300,
        "token_match": token == target_token,
    }


def best_expert_for_payload(experts: dict[str, dict[str, Any]]) -> str:
    if not experts:
        return ""
    return min(
        experts,
        key=lambda model: (
            not bool(experts[model]["token_match"]),
            float(experts[model]["abs_residual"]),
            model,
        ),
    )


def compiled_row_function(row: dict[str, Any]) -> Any:
    try:
        model_family = SymbolicModelFamily(row["model_family"])
    except ValueError:
        return None
    expression = HardenedExpression(
        model_family=model_family,
        expression=row["expression"],
        python_source=row["expression_source"],
        complexity=int(row.get("complexity", 1)),
        active_ops=tuple(row.get("active_ops", ())),
        symbolic_export=bool(row.get("export_success", False)),
    )
    return compile_expression(expression)


def compile_router_candidates(
    group: list[dict[str, Any]],
    dataset: FormulaDataset,
    token_bins: int,
) -> list[dict[str, Any]]:
    quantizer = quantizer_for_dataset(dataset, token_bins)
    candidates: list[dict[str, Any]] = []
    for row in group:
        func = compiled_row_function(row)
        if func is None:
            continue
        train_predictions = tuple(float(func(value)) for value in dataset.train.xs)
        validation_predictions = tuple(float(func(value)) for value in dataset.validation.xs)
        extrapolation_predictions = tuple(float(func(value)) for value in dataset.extrapolation.xs)
        candidates.append(
            {
                "model_family": row["model_family"],
                "train_tokens": tuple(quantizer.encode(value) for value in train_predictions),
                "validation_tokens": tuple(quantizer.encode(value) for value in validation_predictions),
                "extrapolation_tokens": tuple(quantizer.encode(value) for value in extrapolation_predictions),
                "train_token_accuracy": token_accuracy(train_predictions, dataset.train.ys, quantizer),
                "validation_token_accuracy": token_accuracy(validation_predictions, dataset.validation.ys, quantizer),
                "extrapolation_token_accuracy": token_accuracy(extrapolation_predictions, dataset.extrapolation.ys, quantizer),
                "train_rmse": continuous_rmse(train_predictions, dataset.train.ys),
                "validation_rmse": continuous_rmse(validation_predictions, dataset.validation.ys),
                "extrapolation_rmse": continuous_rmse(extrapolation_predictions, dataset.extrapolation.ys),
            }
        )
    return candidates


def oracle_router_scores(
    candidates: list[dict[str, Any]],
    dataset: FormulaDataset,
    token_bins: int,
) -> dict[str, float]:
    quantizer = quantizer_for_dataset(dataset, token_bins)
    validation_targets = tuple(quantizer.encode(value) for value in dataset.validation.ys)
    extrapolation_targets = tuple(quantizer.encode(value) for value in dataset.extrapolation.ys)
    return {
        "validation_token_accuracy": oracle_token_accuracy(
            [candidate["validation_tokens"] for candidate in candidates],
            validation_targets,
        ),
        "extrapolation_token_accuracy": oracle_token_accuracy(
            [candidate["extrapolation_tokens"] for candidate in candidates],
            extrapolation_targets,
        ),
    }


def oracle_token_accuracy(candidate_tokens: list[tuple[int, ...]], targets: tuple[int, ...]) -> float:
    if not targets:
        return 0.0
    correct = 0
    for index, target in enumerate(targets):
        if any(tokens[index] == target for tokens in candidate_tokens):
            correct += 1
    return correct / len(targets)


def fit_linear_readout(
    columns: tuple[tuple[float, ...], ...],
    ys: tuple[float, ...],
) -> tuple[tuple[float, ...], float] | None:
    fit = solve_least_squares(columns, ys)
    if fit is None:
        return None
    coefficients, bias, _rmse = fit
    return coefficients, bias


def solve_least_squares(
    columns: tuple[tuple[float, ...], ...],
    ys: tuple[float, ...],
) -> tuple[tuple[float, ...], float, float] | None:
    dimension = len(columns) + 1
    matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]
    rhs = [0.0 for _ in range(dimension)]
    for row_index, target in enumerate(ys):
        row = [column[row_index] for column in columns] + [1.0]
        for i, left in enumerate(row):
            rhs[i] += left * target
            for j, right in enumerate(row):
                matrix[i][j] += left * right
    solution = solve_linear_system(matrix, rhs)
    if solution is None:
        return None
    coefficients = tuple(solution[:-1])
    bias = solution[-1]
    predictions = apply_linear_readout(columns, coefficients, bias)
    return coefficients, bias, continuous_rmse(predictions, ys)


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs)]
    for diagonal in range(size):
        pivot = max(range(diagonal, size), key=lambda row: abs(augmented[row][diagonal]))
        if abs(augmented[pivot][diagonal]) <= 1.0e-12:
            augmented[diagonal][diagonal] += 1.0e-10
            pivot = diagonal
        if abs(augmented[pivot][diagonal]) <= 1.0e-12:
            return None
        if pivot != diagonal:
            augmented[diagonal], augmented[pivot] = augmented[pivot], augmented[diagonal]
        pivot_value = augmented[diagonal][diagonal]
        for column in range(diagonal, size + 1):
            augmented[diagonal][column] /= pivot_value
        for row in range(size):
            if row == diagonal:
                continue
            factor = augmented[row][diagonal]
            for column in range(diagonal, size + 1):
                augmented[row][column] -= factor * augmented[diagonal][column]
    return [augmented[row][size] for row in range(size)]


def apply_linear_readout(
    columns: tuple[tuple[float, ...], ...],
    coefficients: tuple[float, ...],
    bias: float,
) -> tuple[float, ...]:
    if not columns:
        return ()
    return tuple(
        bias + sum(coefficient * column[index] for coefficient, column in zip(coefficients, columns))
        for index in range(len(columns[0]))
    )


def quantizer_for_dataset(dataset: FormulaDataset, token_bins: int) -> TokenQuantizer:
    values = dataset.train.ys + dataset.validation.ys + dataset.extrapolation.ys
    return TokenQuantizer.from_values(values, token_bins)


def token_accuracy(predictions: tuple[float, ...], targets: tuple[float, ...], quantizer: TokenQuantizer) -> float:
    if not targets:
        return 0.0
    correct = sum(1 for prediction, target in zip(predictions, targets) if quantizer.encode(prediction) == quantizer.encode(target))
    return correct / len(targets)


def token_match_accuracy(token: int, targets: tuple[float, ...], quantizer: TokenQuantizer) -> float:
    if not targets:
        return 0.0
    return sum(1 for target in targets if token == quantizer.encode(target)) / len(targets)


def deterministic_token_nll(accuracy: float, token_bins: int) -> float:
    epsilon = 0.01
    correct_prob = 1.0 - epsilon
    wrong_prob = epsilon / max(1, token_bins - 1)
    return -(accuracy * math.log(correct_prob) + (1.0 - accuracy) * math.log(wrong_prob))


def continuous_rmse(predictions: tuple[float, ...], targets: tuple[float, ...]) -> float:
    if not targets:
        return math.inf
    errors = [
        (prediction - target) ** 2
        for prediction, target in zip(predictions, targets)
        if math.isfinite(prediction) and math.isfinite(target)
    ]
    if len(errors) != len(targets):
        return math.inf
    return math.sqrt(sum(errors) / len(errors))


def token_constant_values(token: int, targets: tuple[float, ...], quantizer: TokenQuantizer) -> tuple[float, ...]:
    width = (quantizer.maximum - quantizer.minimum) / quantizer.bins
    center = quantizer.minimum + (token + 0.5) * width
    return tuple(center for _ in targets)


def summarize_bridge_results(results: list[BridgeRunResult]) -> dict[str, Any]:
    by_model: dict[str, list[BridgeRunResult]] = {}
    for result in results:
        by_model.setdefault(result.model_family, []).append(result)
    model_summaries = {
        model: {
            "run_count": len(rows),
            "median_validation_token_accuracy": median(row.validation_token_accuracy for row in rows),
            "median_extrapolation_token_accuracy": median(row.extrapolation_token_accuracy for row in rows),
            "median_validation_token_nll": median(row.validation_token_nll for row in rows),
            "median_extrapolation_token_nll": median(row.extrapolation_token_nll for row in rows),
            "median_validation_continuous_rmse": median(row.validation_continuous_rmse for row in rows),
            "median_extrapolation_continuous_rmse": median(row.extrapolation_continuous_rmse for row in rows),
            "compile_success_rate": mean(1.0 if row.compile_success else 0.0 for row in rows),
            "source_exact_recovery_rate": mean(1.0 if row.source_exact_recovery else 0.0 for row in rows),
        }
        for model, rows in sorted(by_model.items())
    }
    best_extrap = max(
        model_summaries,
        key=lambda model: model_summaries[model]["median_extrapolation_token_accuracy"],
    )
    best_validation = max(
        model_summaries,
        key=lambda model: model_summaries[model]["median_validation_token_accuracy"],
    )
    return {
        "total_runs": len(results),
        "model_families": model_summaries,
        "best_validation_token_accuracy": best_validation,
        "best_extrapolation_token_accuracy": best_extrap,
    }


def median(values: Any) -> float:
    return float(statistics.median(tuple(values)))


def mean(values: Any) -> float:
    values_tuple = tuple(values)
    if not values_tuple:
        return 0.0
    return sum(values_tuple) / len(values_tuple)


def write_bridge_report(report: BridgeReport, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    with (output_dir / "runs.jsonl").open("w") as handle:
        for result in report.results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_bridge_markdown(report))


def write_feature_table(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row["split"])
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def split_safe_expert_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    coverage: dict[str, list[float]] = {}
    for row in rows:
        coverage.setdefault(str(row["split"]), []).append(1.0 if row["oracle_has_safe_expert"] else 0.0)
    return {split: mean(values) for split, values in sorted(coverage.items())}


def render_bridge_markdown(report: BridgeReport) -> str:
    lines = [
        f"# Symbolic Bridge: {report.run_label}",
        "",
        f"Source summary: `{report.source_summary_path}`",
        f"Token bins: `{report.token_bins}`",
        "",
        "| model | val token acc | extrap token acc | val RMSE | extrap RMSE | source exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, stats in report.summary["model_families"].items():
        lines.append(
            "| "
            f"`{model}` | "
            f"{stats['median_validation_token_accuracy']:.3f} | "
            f"{stats['median_extrapolation_token_accuracy']:.3f} | "
            f"{stats['median_validation_continuous_rmse']:.4g} | "
            f"{stats['median_extrapolation_continuous_rmse']:.4g} | "
            f"{stats['source_exact_recovery_rate']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Best validation token accuracy: `{report.summary['best_validation_token_accuracy']}`",
            f"Best extrapolation token accuracy: `{report.summary['best_extrapolation_token_accuracy']}`",
            f"Feature table: `{report.summary['feature_table']['path']}`",
            f"Feature rows: `{report.summary['feature_table']['row_count']}`",
            f"Feature split counts: `{report.summary['feature_table']['split_counts']}`",
            f"Feature split safe-expert coverage: `{report.summary['feature_table']['split_safe_expert_coverage']}`",
            f"Safe-expert coverage: `{report.summary['feature_table']['safe_expert_coverage']:.3f}`",
            "",
            "## Frozen Side Channel",
            "",
            "| model | baseline extrap acc | side extrap acc | side extrap RMSE |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for model, stats in report.summary["frozen_side_channel_probe"]["model_families"].items():
        lines.append(
            "| "
            f"`{model}` | "
            f"{stats['median_baseline_extrapolation_token_accuracy']:.3f} | "
            f"{stats['median_side_extrapolation_token_accuracy']:.3f} | "
            f"{stats['median_side_extrapolation_rmse']:.4g} |"
        )
    router = report.summary["router_target_probe"]
    lines.extend(
        [
            "",
            "## Router Target",
            "",
            f"Task-router validation token accuracy: `{router['median_validation_token_accuracy']:.3f}`",
            f"Task-router extrapolation token accuracy: `{router['median_extrapolation_token_accuracy']:.3f}`",
            f"Task-router mean extrapolation token accuracy: `{router['mean_extrapolation_token_accuracy']:.3f}`",
            f"Oracle-router validation token accuracy: `{router['oracle_validation_token_accuracy']:.3f}`",
            f"Oracle-router extrapolation token accuracy: `{router['oracle_extrapolation_token_accuracy']:.3f}`",
            f"Oracle-router mean extrapolation token accuracy: `{router['oracle_mean_extrapolation_token_accuracy']:.3f}`",
            f"Selected model counts: `{router['selected_model_counts']}`",
            "",
        ]
    )
    return "\n".join(lines)
