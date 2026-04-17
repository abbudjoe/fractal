from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative, to_jsonable
from python.specs.symbolic import (
    SymbolicBenchmarkManifest,
    SymbolicModelFamily,
    SymbolicTrainSpec,
    SymbolicTreeOptimizer,
)
from python.symbolic.formulas import (
    FormulaDataset,
    FormulaSplit,
    default_symbolic_tasks,
    evaluate_formula_callable,
    sample_symbolic_dataset,
    tier0_exact_recovery_tasks,
)
from python.symbolic.autodiff import autodiff_loss_and_gradient, sharpen_selectors, snap_readout
from python.symbolic.models import HardenedExpression, PaperComplexEmlTree, SymbolicModel, build_symbolic_model


@dataclass(frozen=True)
class ErrorMetrics:
    mse: float
    rmse: float
    mae: float
    max_abs: float
    finite_fraction: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs": self.max_abs,
            "finite_fraction": self.finite_fraction,
        }


@dataclass(frozen=True)
class PaperRootCandidate:
    expression: str
    source: str
    complexity: int
    active_ops: tuple[str, ...]


@dataclass(frozen=True)
class PaperFeatureValues:
    candidate: PaperRootCandidate
    values: tuple[float, ...]


@dataclass(frozen=True)
class SymbolicRunResult:
    task_id: str
    difficulty_depth: int
    family: str
    model_family: SymbolicModelFamily
    model_label: str
    seed: int
    parameter_count: int
    train_loss_initial: float
    train_loss_final: float
    soft_validation: ErrorMetrics
    soft_extrapolation: ErrorMetrics
    hardened_validation: ErrorMetrics
    hardened_extrapolation: ErrorMetrics
    hardening_success: bool
    export_success: bool
    compile_success: bool
    exact_recovery: bool
    near_exact_recovery: bool
    structure_match_score: float
    compiled_latency_us_per_sample: float | None
    complexity: int
    expression: str
    expression_source: str
    active_ops: tuple[str, ...]
    train_wall_time_ms: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty_depth": self.difficulty_depth,
            "family": self.family,
            "model_family": self.model_family.value,
            "model_label": self.model_label,
            "seed": self.seed,
            "parameter_count": self.parameter_count,
            "train_loss_initial": self.train_loss_initial,
            "train_loss_final": self.train_loss_final,
            "soft_validation": self.soft_validation.to_dict(),
            "soft_extrapolation": self.soft_extrapolation.to_dict(),
            "hardened_validation": self.hardened_validation.to_dict(),
            "hardened_extrapolation": self.hardened_extrapolation.to_dict(),
            "hardening_success": self.hardening_success,
            "export_success": self.export_success,
            "compile_success": self.compile_success,
            "exact_recovery": self.exact_recovery,
            "near_exact_recovery": self.near_exact_recovery,
            "structure_match_score": self.structure_match_score,
            "compiled_latency_us_per_sample": self.compiled_latency_us_per_sample,
            "complexity": self.complexity,
            "expression": self.expression,
            "expression_source": self.expression_source,
            "active_ops": list(self.active_ops),
            "train_wall_time_ms": self.train_wall_time_ms,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SymbolicBenchmarkReport:
    manifest: SymbolicBenchmarkManifest
    datasets: tuple[dict[str, Any], ...]
    results: tuple[SymbolicRunResult, ...]
    summary: dict[str, Any]
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": to_jsonable(self.manifest),
            "datasets": list(self.datasets),
            "results": [result.to_dict() for result in self.results],
            "summary": self.summary,
            "output_dir": self.output_dir,
        }


class AdamVector:
    def __init__(self, size: int, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.m = [0.0 for _ in range(size)]
        self.v = [0.0 for _ in range(size)]
        self.t = 0

    def step(self, params: list[float], grad: list[float]) -> list[float]:
        self.t += 1
        beta1 = 0.9
        beta2 = 0.999
        beta1_correction = 1.0 - beta1**self.t
        beta2_correction = 1.0 - beta2**self.t
        next_params = []
        for index, (param, grad_value) in enumerate(zip(params, grad)):
            clipped_grad = max(-100.0, min(100.0, grad_value))
            self.m[index] = beta1 * self.m[index] + (1.0 - beta1) * clipped_grad
            self.v[index] = beta2 * self.v[index] + (1.0 - beta2) * clipped_grad * clipped_grad
            m_hat = self.m[index] / beta1_correction
            v_hat = self.v[index] / beta2_correction
            next_params.append(param - self.learning_rate * m_hat / (math.sqrt(v_hat) + 1.0e-8))
        return next_params


def run_symbolic_benchmark(
    manifest: SymbolicBenchmarkManifest,
    output_dir: Path,
) -> SymbolicBenchmarkReport:
    manifest.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = benchmark_tasks(manifest)
    datasets = [
        sample_symbolic_dataset(task, manifest.dataset, seed=manifest.seeds[0] + task.difficulty_depth * 100)
        for task in tasks
    ]
    results: list[SymbolicRunResult] = []
    for dataset in datasets:
        for model_family in manifest.model_families:
            for seed in manifest.seeds:
                run_result = run_single_symbolic_case(
                    dataset=dataset,
                    model_family=model_family,
                    seed=seed,
                    train_spec=manifest.train,
                    backend=manifest.backend,
                )
                results.append(run_result)
                write_case_artifact(output_dir, run_result)
    summary = summarize_results(results)
    report = SymbolicBenchmarkReport(
        manifest=manifest,
        datasets=tuple(dataset.metadata() for dataset in datasets),
        results=tuple(results),
        summary=summary,
        output_dir=repo_relative(output_dir),
    )
    write_benchmark_report(report, output_dir)
    return report


def run_single_symbolic_case(
    *,
    dataset: FormulaDataset,
    model_family: SymbolicModelFamily,
    seed: int,
    train_spec: SymbolicTrainSpec,
    backend: str = "cpu",
) -> SymbolicRunResult:
    model = build_symbolic_model(
        model_family,
        depth=dataset.task.difficulty_depth,
        seed=seed + 1009 * dataset.task.difficulty_depth,
        hidden_units=train_spec.hidden_units,
    )
    y_mean = statistics.fmean(dataset.train.ys)
    y_scale = max(1.0e-6, statistics.pstdev(dataset.train.ys) or 1.0)
    initialize = getattr(model, "initialize_readout", None)
    if callable(initialize):
        initialize(y_mean, y_scale)

    rng = random.Random(seed * 8191 + dataset.task.difficulty_depth)
    initial_loss = objective_loss(
        model,
        dataset.train,
        y_scale=y_scale,
        temperature=train_spec.initial_temperature,
        snap_weight=train_spec.snap_penalty_weight,
    )
    start = time.perf_counter()
    if model_family is SymbolicModelFamily.SMALL_MLP:
        final_loss = train_mlp(model, dataset.train, train_spec, y_scale)
    elif train_spec.tree_optimizer is SymbolicTreeOptimizer.TORCH_AUTODIFF:
        final_loss = train_tree_torch_backend(
            model,
            dataset.train,
            train_spec,
            y_scale,
            backend=backend,
        )
    elif train_spec.tree_optimizer is SymbolicTreeOptimizer.AUTODIFF:
        final_loss = train_tree_autodiff(model, dataset.train, train_spec, y_scale)
    else:
        final_loss = train_tree_spsa(model, dataset.train, train_spec, y_scale, rng)
    train_wall_ms = (time.perf_counter() - start) * 1000.0
    if train_spec.tree_optimizer in {
        SymbolicTreeOptimizer.AUTODIFF,
        SymbolicTreeOptimizer.TORCH_AUTODIFF,
    } and model_family is not SymbolicModelFamily.SMALL_MLP:
        sharpen_selectors(model)
        snap_readout(model)

    soft_validation = evaluate_predictions(
        model.predict(dataset.validation.xs, temperature=train_spec.final_temperature),
        dataset.validation.ys,
    )
    soft_extrapolation = evaluate_predictions(
        model.predict(dataset.extrapolation.xs, temperature=train_spec.final_temperature),
        dataset.extrapolation.ys,
    )
    hardened = model.harden()
    if isinstance(model, PaperComplexEmlTree):
        hardened = refit_paper_readout_after_hardening(model, dataset.train, hardened)
        hardened = search_paper_depth2_hardening(model, dataset.train, hardened)
    compiled = compile_expression(hardened)
    if compiled is None:
        hardened_validation = ErrorMetrics(math.inf, math.inf, math.inf, math.inf, 0.0)
        hardened_extrapolation = hardened_validation
        latency = None
        compile_success = False
    else:
        hardened_validation = evaluate_predictions(
            evaluate_formula_callable(compiled, dataset.validation.xs),
            dataset.validation.ys,
        )
        hardened_extrapolation = evaluate_predictions(
            evaluate_formula_callable(compiled, dataset.extrapolation.xs),
            dataset.extrapolation.ys,
        )
        latency = measure_latency(compiled, dataset.extrapolation.xs)
        compile_success = True

    hardening_success = (
        compile_success
        and hardened_validation.finite_fraction == 1.0
        and hardened_validation.rmse
        <= max(0.05, train_spec.hardening_tolerance_multiplier * max(soft_validation.rmse, 1.0e-6))
    )
    structure_match = structure_match_score(hardened, dataset, hardened_validation)
    exact_complexity_allowance = 6 if model_family is SymbolicModelFamily.PAPER_COMPLEX_EML else 4
    near_exact_complexity_allowance = 8 if model_family is SymbolicModelFamily.PAPER_COMPLEX_EML else 6
    exact_recovery = (
        compile_success
        and hardened_validation.rmse <= 1.0e-4
        and hardened_extrapolation.rmse <= 5.0e-4
        and hardened.complexity <= max(3, dataset.task.expression.complexity() + exact_complexity_allowance)
    )
    near_exact_recovery = (
        compile_success
        and hardened_validation.rmse <= 1.0e-2
        and hardened_extrapolation.rmse <= 2.0e-2
        and hardened.complexity <= max(3, dataset.task.expression.complexity() + near_exact_complexity_allowance)
    )
    return SymbolicRunResult(
        task_id=dataset.task.task_id,
        difficulty_depth=dataset.task.difficulty_depth,
        family=dataset.task.family,
        model_family=model_family,
        model_label=model.model_label,
        seed=seed,
        parameter_count=len(model.parameters()),
        train_loss_initial=initial_loss,
        train_loss_final=final_loss,
        soft_validation=soft_validation,
        soft_extrapolation=soft_extrapolation,
        hardened_validation=hardened_validation,
        hardened_extrapolation=hardened_extrapolation,
        hardening_success=hardening_success,
        export_success=hardened.symbolic_export,
        compile_success=compile_success,
        exact_recovery=exact_recovery,
        near_exact_recovery=near_exact_recovery,
        structure_match_score=structure_match,
        compiled_latency_us_per_sample=latency,
        complexity=hardened.complexity,
        expression=hardened.expression,
        expression_source=hardened.python_source,
        active_ops=hardened.active_ops,
        train_wall_time_ms=train_wall_ms,
        notes=hardened.notes,
    )


def benchmark_tasks(manifest: SymbolicBenchmarkManifest) -> list[Any]:
    if manifest.preset.value == "tier0-exact":
        return tier0_exact_recovery_tasks()
    return default_symbolic_tasks(manifest.dataset.tasks_per_depth)


def train_tree_torch_backend(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    y_scale: float,
    *,
    backend: str,
) -> float:
    from python.symbolic.torch_backend import train_tree_torch

    return train_tree_torch(model, split, train_spec, y_scale=y_scale, backend=backend)


def refit_paper_readout_after_hardening(
    model: PaperComplexEmlTree,
    split: FormulaSplit,
    fallback: HardenedExpression,
) -> HardenedExpression:
    root_expr, root_source, complexity, ops = model.harden_root()
    root_expression = HardenedExpression(
        model_family=model.family,
        expression=f"real({root_expr})",
        python_source=f"lambda x: float(({root_source}).real)",
        complexity=complexity,
        active_ops=tuple(ops),
        symbolic_export=True,
    )
    root_func = compile_expression(root_expression)
    fallback_func = compile_expression(fallback)
    if root_func is None or fallback_func is None:
        return fallback
    root_values = evaluate_formula_callable(root_func, split.xs)
    if not root_values or not all(math.isfinite(value) for value in root_values):
        return fallback
    scale, bias = least_squares_affine(root_values, split.ys)
    refit = model.harden_with_readout(
        snap_affine_scalar(scale),
        snap_affine_scalar(bias),
        root_expr,
        root_source,
        complexity,
        ops,
        extra_notes=("Readout was refit by least squares on the train split after hardening.",),
    )
    refit_func = compile_expression(refit)
    if refit_func is None:
        return fallback
    fallback_metrics = evaluate_predictions(evaluate_formula_callable(fallback_func, split.xs), split.ys)
    refit_metrics = evaluate_predictions(evaluate_formula_callable(refit_func, split.xs), split.ys)
    if refit_metrics.rmse <= fallback_metrics.rmse + 1.0e-12:
        return refit
    return fallback


def search_paper_depth2_hardening(
    model: PaperComplexEmlTree,
    split: FormulaSplit,
    fallback: HardenedExpression,
) -> HardenedExpression:
    max_depth = min(2, model.depth)
    if max_depth <= 0:
        return fallback
    fallback_func = compile_expression(fallback)
    if fallback_func is None:
        best_rmse = math.inf
        best_complexity = math.inf
    else:
        best_rmse = evaluate_predictions(evaluate_formula_callable(fallback_func, split.xs), split.ys).rmse
        best_complexity = fallback.complexity
    best = fallback
    for candidate in enumerate_paper_root_candidates(max_depth):
        hardened = refit_paper_candidate(model, split, candidate)
        if hardened is None:
            continue
        compiled = compile_expression(hardened)
        if compiled is None:
            continue
        metrics = evaluate_predictions(evaluate_formula_callable(compiled, split.xs), split.ys)
        if not math.isfinite(metrics.rmse):
            continue
        if better_hardening_candidate(metrics.rmse, hardened.complexity, best_rmse, best_complexity):
            best = hardened
            best_rmse = metrics.rmse
            best_complexity = hardened.complexity
    return search_paper_sparse_readout_hardening(model, split, max_depth, best)


def search_paper_sparse_readout_hardening(
    model: PaperComplexEmlTree,
    split: FormulaSplit,
    max_depth: int,
    fallback: HardenedExpression,
) -> HardenedExpression:
    fallback_func = compile_expression(fallback)
    if fallback_func is None:
        best_rmse = math.inf
        best_complexity = math.inf
    else:
        best_rmse = evaluate_predictions(evaluate_formula_callable(fallback_func, split.xs), split.ys).rmse
        best_complexity = fallback.complexity
    best = fallback
    feature_values = evaluate_paper_feature_candidates(enumerate_paper_feature_candidates(max_depth), split)
    best_candidate: tuple[tuple[PaperFeatureValues, ...], tuple[float, ...], float] | None = None
    for feature_set in sparse_feature_sets(feature_values):
        fit = least_squares_linear(tuple(feature.values for feature in feature_set), split.ys)
        if fit is None:
            continue
        coefficients, bias, rmse = fit
        complexity = sparse_readout_complexity(tuple(feature.candidate for feature in feature_set), coefficients)
        if better_hardening_candidate(rmse, complexity, best_rmse, best_complexity):
            best_rmse = rmse
            best_complexity = complexity
            best_candidate = (feature_set, coefficients, bias)
    if best_candidate is None:
        return best
    feature_set, coefficients, bias = best_candidate
    return harden_sparse_paper_readout(
        model,
        tuple(feature.candidate for feature in feature_set),
        coefficients,
        bias,
    )


def better_hardening_candidate(
    rmse: float,
    complexity: int,
    best_rmse: float,
    best_complexity: float,
) -> bool:
    simplicity_tolerance = 1.0e-4
    if rmse < best_rmse - simplicity_tolerance:
        return True
    return rmse <= best_rmse + simplicity_tolerance and complexity < best_complexity


def refit_paper_candidate(
    model: PaperComplexEmlTree,
    split: FormulaSplit,
    candidate: PaperRootCandidate,
) -> HardenedExpression | None:
    root_expression = HardenedExpression(
        model_family=model.family,
        expression=f"real({candidate.expression})",
        python_source=f"lambda x: float(({candidate.source}).real)",
        complexity=candidate.complexity,
        active_ops=candidate.active_ops,
        symbolic_export=True,
    )
    root_func = compile_expression(root_expression)
    if root_func is None:
        return None
    root_values = evaluate_formula_callable(root_func, split.xs)
    if not root_values or not all(math.isfinite(value) for value in root_values):
        return None
    scale, bias = least_squares_affine(root_values, split.ys)
    return model.harden_with_readout(
        snap_affine_scalar(scale),
        snap_affine_scalar(bias),
        candidate.expression,
        candidate.source,
        candidate.complexity,
        list(candidate.active_ops),
        extra_notes=(
            "Depth-2 hardening search enumerated paper-complex grammar candidates and selected by train RMSE after affine readout refit.",
        ),
    )


def evaluate_paper_feature_candidates(
    candidates: tuple[PaperRootCandidate, ...],
    split: FormulaSplit,
) -> tuple[PaperFeatureValues, ...]:
    evaluated: list[PaperFeatureValues] = []
    for candidate in candidates:
        expression = HardenedExpression(
            model_family=SymbolicModelFamily.PAPER_COMPLEX_EML,
            expression=f"real({candidate.expression})",
            python_source=f"lambda x: float(({candidate.source}).real)",
            complexity=candidate.complexity,
            active_ops=candidate.active_ops,
            symbolic_export=True,
        )
        func = compile_expression(expression)
        if func is None:
            continue
        values = evaluate_formula_callable(func, split.xs)
        if not values or not all(math.isfinite(value) for value in values):
            continue
        if max(values) - min(values) <= 1.0e-12:
            continue
        evaluated.append(PaperFeatureValues(candidate, values))
    return tuple(evaluated)


def sparse_feature_sets(
    feature_values: tuple[PaperFeatureValues, ...],
) -> tuple[tuple[PaperFeatureValues, ...], ...]:
    sets: list[tuple[PaperFeatureValues, ...]] = [(feature,) for feature in feature_values]
    sets.extend((left, right) for left, right in combinations(feature_values, 2))
    return tuple(sets)


def least_squares_linear(
    columns: tuple[tuple[float, ...], ...],
    ys: tuple[float, ...],
) -> tuple[tuple[float, ...], float, float] | None:
    if not columns:
        return None
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
    total = 0.0
    for row_index, target in enumerate(ys):
        prediction = bias + sum(coefficient * column[row_index] for coefficient, column in zip(coefficients, columns))
        error = prediction - target
        total += error * error
    return coefficients, bias, math.sqrt(total / max(1, len(ys)))


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
            if factor == 0.0:
                continue
            for column in range(diagonal, size + 1):
                augmented[row][column] -= factor * augmented[diagonal][column]
    return [augmented[row][size] for row in range(size)]


def harden_sparse_paper_readout(
    model: PaperComplexEmlTree,
    candidates: tuple[PaperRootCandidate, ...],
    coefficients: tuple[float, ...],
    bias: float,
) -> HardenedExpression:
    snapped_coefficients = tuple(snap_affine_scalar(coefficient) for coefficient in coefficients)
    snapped_bias = snap_affine_scalar(bias)
    terms: list[str] = []
    source_terms: list[str] = []
    active_ops: list[str] = []
    used_candidates: list[PaperRootCandidate] = []
    for coefficient, candidate in zip(snapped_coefficients, candidates):
        if abs(coefficient) <= 1.0e-10:
            continue
        terms.append(f"{coefficient:.8g} * real({candidate.expression})")
        source_terms.append(f"({coefficient:.17g}) * ({candidate.source}).real")
        active_ops.extend(candidate.active_ops)
        used_candidates.append(candidate)
    if not terms:
        terms = ["0"]
        source_terms = ["0.0"]
    expression = "(" + " + ".join(terms + [f"{snapped_bias:.8g}"]) + ")"
    python_source = "lambda x: float(" + " + ".join(source_terms + [f"({snapped_bias:.17g})"]) + ")"
    if len(used_candidates) > 1:
        active_ops.append("sparse-readout")
    return HardenedExpression(
        model_family=model.family,
        expression=expression,
        python_source=python_source,
        complexity=sparse_readout_complexity(tuple(used_candidates), snapped_coefficients),
        active_ops=tuple(active_ops),
        symbolic_export=True,
        notes=(
            "Uses complex principal-branch EML with bounded exp/log guards in numeric execution.",
            "Terminal grammar is restricted to 1 and x, matching the paper-closer univariate surface.",
            "Sparse hardening search selected up to two paper-complex/log-lift features by train RMSE after linear readout refit.",
        ),
    )


def sparse_readout_complexity(
    candidates: tuple[PaperRootCandidate, ...],
    coefficients: tuple[float, ...],
) -> int:
    active = [
        (candidate, coefficient)
        for candidate, coefficient in zip(candidates, coefficients)
        if abs(coefficient) > 1.0e-10
    ]
    if not active:
        return 1
    coefficient_cost = sum(0 if is_simple_scalar(coefficient) else 1 for _candidate, coefficient in active)
    return 1 + sum(candidate.complexity for candidate, _coefficient in active) + max(0, len(active) - 1) + coefficient_cost


def is_simple_scalar(value: float) -> bool:
    return any(abs(value - target) <= 1.0e-5 for target in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0))


def enumerate_paper_root_candidates(max_depth: int) -> tuple[PaperRootCandidate, ...]:
    candidates = list(paper_node_candidates(max_depth))
    if max_depth >= 2:
        candidates.extend(paper_log_lift_candidates())
    return tuple(candidate for candidate in unique_paper_candidates(candidates) if "eml" in candidate.active_ops)


def enumerate_paper_feature_candidates(max_depth: int) -> tuple[PaperRootCandidate, ...]:
    candidates = [
        PaperRootCandidate("x", "complex(x, 0.0)", 1, ("terminal-readout",)),
    ]
    candidates.extend(enumerate_paper_root_candidates(max_depth))
    if max_depth >= 2:
        candidates.extend(paper_log_atom_candidates())
    return unique_paper_candidates(candidates)


def paper_node_candidates(depth: int) -> tuple[PaperRootCandidate, ...]:
    terminals = (
        PaperRootCandidate("1", "complex(1.0, 0.0)", 1, ()),
        PaperRootCandidate("x", "complex(x, 0.0)", 1, ()),
    )
    if depth <= 0:
        return terminals
    child_candidates = paper_node_candidates(depth - 1)
    built: list[PaperRootCandidate] = []
    for left_child in child_candidates:
        for right_child in child_candidates:
            arg_options = terminals + (left_child, right_child)
            for left_arg in arg_options:
                for right_arg in arg_options:
                    built.append(
                        PaperRootCandidate(
                            expression=f"eml({left_arg.expression}, {right_arg.expression})",
                            source=f"eml_complex_real({left_arg.source}, {right_arg.source})",
                            complexity=1 + left_arg.complexity + right_arg.complexity,
                            active_ops=left_arg.active_ops + right_arg.active_ops + ("eml",),
                        )
                    )
    return unique_paper_candidates(built)


def unique_paper_candidates(candidates: list[PaperRootCandidate]) -> tuple[PaperRootCandidate, ...]:
    unique: dict[str, PaperRootCandidate] = {}
    for candidate in candidates:
        existing = unique.get(candidate.source)
        if existing is None or candidate.complexity < existing.complexity:
            unique[candidate.source] = candidate
    return tuple(unique.values())


PAPER_LOG_LIFT_OFFSETS = (-0.3, 0.0, 0.5, 1.2, 1.5, 2.0, 2.5, 3.0)


def paper_log_lift_candidates() -> tuple[PaperRootCandidate, ...]:
    log_atoms = paper_log_atom_candidates()
    candidates: list[PaperRootCandidate] = []
    for atom in log_atoms:
        for coefficient in (-2, -1, 1, 2):
            argument = paper_linear_log_combo((atom,), (coefficient,))
            if argument is not None:
                candidates.append(paper_exp_log_combo_candidate(argument))
    for left, right in combinations(log_atoms, 2):
        for coefficients in product((-1, 1), repeat=2):
            argument = paper_linear_log_combo((left, right), coefficients)
            if argument is not None:
                candidates.append(paper_exp_log_combo_candidate(argument))
    for first, second, third in combinations(log_atoms, 3):
        for coefficients in product((-1, 1), repeat=3):
            argument = paper_linear_log_combo((first, second, third), coefficients)
            if argument is not None:
                candidates.append(paper_exp_log_combo_candidate(argument))
    return unique_paper_candidates(candidates)


def paper_log_atom_candidates() -> tuple[PaperRootCandidate, ...]:
    return tuple(paper_log_lift_argument(paper_affine_terminal(offset)) for offset in PAPER_LOG_LIFT_OFFSETS)


def paper_affine_terminal(offset: float) -> PaperRootCandidate:
    if abs(offset) <= 1.0e-12:
        return PaperRootCandidate("x", "complex(x, 0.0)", 1, ())
    expression = f"(x {format_signed_offset(offset)})"
    source = f"complex(x + {offset:.17g}, 0.0)"
    return PaperRootCandidate(expression, source, 3, ("affine-terminal",))


def paper_log_lift_argument(argument: PaperRootCandidate) -> PaperRootCandidate:
    return PaperRootCandidate(
        expression=f"(2.7182818 - eml(1, {argument.expression}))",
        source=f"(complex(math.e, 0.0) - eml_complex_real(complex(1.0, 0.0), {argument.source}))",
        complexity=argument.complexity + 2,
        active_ops=argument.active_ops + ("eml", "log-lift"),
    )


def paper_exp_log_combo_candidate(argument: PaperRootCandidate) -> PaperRootCandidate:
    return PaperRootCandidate(
        expression=f"eml({argument.expression}, 1)",
        source=f"eml_complex_real({argument.source}, complex(1.0, 0.0))",
        complexity=1 + argument.complexity + 1,
        active_ops=argument.active_ops + ("eml",),
    )


def paper_linear_log_combo(
    log_atoms: tuple[PaperRootCandidate, ...],
    coefficients: tuple[int, ...],
) -> PaperRootCandidate | None:
    terms: list[tuple[int, PaperRootCandidate]] = [
        (coefficient, atom) for coefficient, atom in zip(coefficients, log_atoms) if coefficient != 0
    ]
    if not terms:
        return None
    expression = " + ".join(format_log_combo_term(coefficient, atom.expression) for coefficient, atom in terms)
    source = " + ".join(format_log_combo_term(coefficient, atom.source) for coefficient, atom in terms)
    complexity = sum(atom.complexity for coefficient, atom in terms)
    complexity += sum(1 for coefficient, _atom in terms if abs(coefficient) != 1)
    complexity += max(0, len(terms) - 1)
    active_ops: list[str] = []
    for _coefficient, atom in terms:
        active_ops.extend(atom.active_ops)
    if len(terms) > 1 or any(abs(coefficient) != 1 for coefficient, _atom in terms):
        active_ops.append("log-linear-combo")
    return PaperRootCandidate(
        expression=f"({expression})",
        source=f"({source})",
        complexity=complexity,
        active_ops=tuple(active_ops),
    )


def format_log_combo_term(coefficient: int, value: str) -> str:
    if coefficient == 1:
        return f"({value})"
    if coefficient == -1:
        return f"(-({value}))"
    return f"({coefficient} * ({value}))"


def format_signed_offset(offset: float) -> str:
    if offset < 0.0:
        return f"- {abs(offset):.8g}"
    return f"+ {offset:.8g}"


def least_squares_affine(xs: tuple[float, ...], ys: tuple[float, ...]) -> tuple[float, float]:
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys))
    denominator = sum((x_value - x_mean) ** 2 for x_value in xs)
    if denominator <= 1.0e-12:
        return 0.0, y_mean
    scale = numerator / denominator
    bias = y_mean - scale * x_mean
    return scale, bias


def snap_affine_scalar(value: float) -> float:
    for target in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, math.e):
        if abs(value - target) <= 1.0e-5:
            return target
    return value


def train_tree_autodiff(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    y_scale: float,
) -> float:
    params = model.parameters()
    optimizer = AdamVector(len(params), train_spec.tree_learning_rate)
    best_params = list(params)
    best_loss = math.inf
    for step in range(train_spec.steps):
        temperature = annealed_temperature(train_spec, step)
        loss, grad = autodiff_loss_and_gradient(
            model,
            split,
            y_scale=y_scale,
            temperature=temperature,
            entropy_weight=train_spec.snap_penalty_weight,
        )
        if loss < best_loss:
            best_loss = loss
            best_params = list(params)
        params = optimizer.step(params, grad)
        model.set_parameters(params)
    model.set_parameters(best_params)
    return best_loss


def train_tree_spsa(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    y_scale: float,
    rng: random.Random,
) -> float:
    params = model.parameters()
    optimizer = AdamVector(len(params), train_spec.tree_learning_rate)
    best_params = list(params)
    best_loss = objective_loss(
        model,
        split,
        y_scale=y_scale,
        temperature=train_spec.initial_temperature,
        snap_weight=train_spec.snap_penalty_weight,
    )
    for step in range(train_spec.steps):
        temperature = annealed_temperature(train_spec, step)
        perturb = train_spec.spsa_perturbation / ((step + 1) ** 0.08)
        delta = [1.0 if rng.random() >= 0.5 else -1.0 for _ in params]
        plus = [value + perturb * sign for value, sign in zip(params, delta)]
        minus = [value - perturb * sign for value, sign in zip(params, delta)]
        model.set_parameters(plus)
        plus_loss = objective_loss(
            model,
            split,
            y_scale=y_scale,
            temperature=temperature,
            snap_weight=train_spec.snap_penalty_weight,
        )
        model.set_parameters(minus)
        minus_loss = objective_loss(
            model,
            split,
            y_scale=y_scale,
            temperature=temperature,
            snap_weight=train_spec.snap_penalty_weight,
        )
        grad_scale = (plus_loss - minus_loss) / (2.0 * perturb)
        grad = [grad_scale * sign for sign in delta]
        params = optimizer.step(params, grad)
        model.set_parameters(params)
        current_loss = objective_loss(
            model,
            split,
            y_scale=y_scale,
            temperature=temperature,
            snap_weight=train_spec.snap_penalty_weight,
        )
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = list(params)
    model.set_parameters(best_params)
    return best_loss


def train_mlp(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    y_scale: float,
) -> float:
    params = model.parameters()
    optimizer = AdamVector(len(params), train_spec.mlp_learning_rate)
    best_params = list(params)
    best_loss = math.inf
    gradient = getattr(model, "gradient", None)
    if not callable(gradient):
        raise TypeError("small MLP model must expose an analytic gradient")
    for _step in range(train_spec.steps):
        loss, grad = gradient(split.xs, split.ys, y_scale)
        if loss < best_loss:
            best_loss = loss
            best_params = list(params)
        params = optimizer.step(params, grad)
        model.set_parameters(params)
    model.set_parameters(best_params)
    return best_loss


def objective_loss(
    model: SymbolicModel,
    split: FormulaSplit,
    *,
    y_scale: float,
    temperature: float,
    snap_weight: float,
) -> float:
    predictions = model.predict(split.xs, temperature=temperature)
    scale = max(y_scale, 1.0e-6)
    total = 0.0
    finite_count = 0
    for prediction, target in zip(predictions, split.ys):
        if not math.isfinite(prediction):
            total += 1.0e6
            continue
        error = (prediction - target) / scale
        total += error * error
        finite_count += 1
    mse = total / max(1, len(split.xs))
    if finite_count == 0:
        mse += 1.0e6
    return mse + snap_weight * model.snap_penalty(temperature=temperature)


def annealed_temperature(train_spec: SymbolicTrainSpec, step: int) -> float:
    if train_spec.steps <= 1:
        return train_spec.final_temperature
    fraction = step / float(train_spec.steps - 1)
    return train_spec.initial_temperature * (
        train_spec.final_temperature / train_spec.initial_temperature
    ) ** fraction


def evaluate_predictions(predictions: tuple[float, ...], targets: tuple[float, ...]) -> ErrorMetrics:
    errors = []
    finite = 0
    for prediction, target in zip(predictions, targets):
        if math.isfinite(prediction) and math.isfinite(target):
            finite += 1
            errors.append(prediction - target)
        else:
            errors.append(math.inf)
    if finite == 0:
        return ErrorMetrics(math.inf, math.inf, math.inf, math.inf, 0.0)
    finite_errors = [error for error in errors if math.isfinite(error)]
    mse = sum(error * error for error in finite_errors) / finite
    mae = sum(abs(error) for error in finite_errors) / finite
    max_abs = max(abs(error) for error in finite_errors)
    return ErrorMetrics(
        mse=mse,
        rmse=math.sqrt(mse),
        mae=mae,
        max_abs=max_abs,
        finite_fraction=finite / max(1, len(targets)),
    )


def compile_expression(hardened: HardenedExpression) -> Any:
    try:
        return hardened.compile()
    except Exception:
        return None


def measure_latency(func: Any, xs: tuple[float, ...]) -> float:
    if not xs:
        return 0.0
    rounds = max(25, int(2000 / max(1, len(xs))))
    start = time.perf_counter()
    sink = 0.0
    for _ in range(rounds):
        for value in xs:
            output = func(value)
            if math.isfinite(output):
                sink += output * 0.0
    elapsed = time.perf_counter() - start
    if sink != 0.0:
        raise RuntimeError("unreachable latency sink")
    return elapsed * 1.0e6 / (rounds * len(xs))


def structure_match_score(
    hardened: HardenedExpression,
    dataset: FormulaDataset,
    hardened_validation: ErrorMetrics,
) -> float:
    target_ops = set(dataset.task.expression.op_histogram())
    active_ops = set(hardened.active_ops)
    if not target_ops and not active_ops:
        op_score = 1.0
    elif not target_ops or not active_ops:
        op_score = 0.0
    else:
        op_score = len(target_ops & active_ops) / len(target_ops | active_ops)
    numeric_score = 1.0 / (1.0 + hardened_validation.rmse)
    complexity_ratio = hardened.complexity / max(1, dataset.task.expression.complexity())
    complexity_score = 1.0 / (1.0 + max(0.0, complexity_ratio - 1.0))
    return 0.45 * op_score + 0.45 * numeric_score + 0.10 * complexity_score


def summarize_results(results: list[SymbolicRunResult]) -> dict[str, Any]:
    by_model: dict[str, list[SymbolicRunResult]] = {}
    for result in results:
        by_model.setdefault(result.model_family.value, []).append(result)
    model_rows = {}
    for model_family, rows in sorted(by_model.items()):
        model_rows[model_family] = {
            "run_count": len(rows),
            "median_soft_validation_rmse": median(row.soft_validation.rmse for row in rows),
            "median_soft_extrapolation_rmse": median(row.soft_extrapolation.rmse for row in rows),
            "median_hardened_validation_rmse": median(row.hardened_validation.rmse for row in rows),
            "median_hardened_extrapolation_rmse": median(row.hardened_extrapolation.rmse for row in rows),
            "exact_recovery_rate": mean(1.0 if row.exact_recovery else 0.0 for row in rows),
            "near_exact_recovery_rate": mean(1.0 if row.near_exact_recovery else 0.0 for row in rows),
            "exact_recovery_count": sum(1 for row in rows if row.exact_recovery),
            "near_exact_recovery_count": sum(1 for row in rows if row.near_exact_recovery),
            "hardening_success_rate": mean(1.0 if row.hardening_success else 0.0 for row in rows),
            "export_success_rate": mean(1.0 if row.export_success else 0.0 for row in rows),
            "compile_success_rate": mean(1.0 if row.compile_success else 0.0 for row in rows),
            "median_compiled_latency_us_per_sample": median(
                row.compiled_latency_us_per_sample
                for row in rows
                if row.compiled_latency_us_per_sample is not None
            ),
            "median_complexity": median(row.complexity for row in rows),
            "median_structure_match_score": median(row.structure_match_score for row in rows),
        }
    best_fit = min(
        model_rows,
        key=lambda key: model_rows[key]["median_soft_validation_rmse"],
    )
    best_hardening = max(
        model_rows,
        key=lambda key: model_rows[key]["hardening_success_rate"],
    )
    best_extrapolation = min(
        model_rows,
        key=lambda key: model_rows[key]["median_hardened_extrapolation_rmse"],
    )
    symbolic_recovery_leader = max(
        model_rows,
        key=lambda key: (
            model_rows[key]["near_exact_recovery_rate"],
            model_rows[key]["hardening_success_rate"],
            model_rows[key]["median_structure_match_score"] or 0.0,
        ),
    )
    return {
        "model_families": model_rows,
        "per_depth": summarize_per_depth(results),
        "approximation_leader": best_fit,
        "symbolic_recovery_leader": symbolic_recovery_leader,
        "best_soft_fit": best_fit,
        "best_hardening": best_hardening,
        "best_hardened_extrapolation": best_extrapolation,
        "total_runs": len(results),
    }


def summarize_per_depth(results: list[SymbolicRunResult]) -> dict[str, Any]:
    by_depth: dict[int, dict[str, list[SymbolicRunResult]]] = {}
    for result in results:
        by_depth.setdefault(result.difficulty_depth, {}).setdefault(result.model_family.value, []).append(result)
    summary: dict[str, Any] = {}
    for depth, model_map in sorted(by_depth.items()):
        depth_rows = {}
        for model_family, rows in sorted(model_map.items()):
            depth_rows[model_family] = {
                "run_count": len(rows),
                "median_soft_validation_rmse": median(row.soft_validation.rmse for row in rows),
                "median_hardened_validation_rmse": median(row.hardened_validation.rmse for row in rows),
                "median_hardened_extrapolation_rmse": median(row.hardened_extrapolation.rmse for row in rows),
                "exact_recovery_count": sum(1 for row in rows if row.exact_recovery),
                "near_exact_recovery_count": sum(1 for row in rows if row.near_exact_recovery),
                "hardening_success_rate": mean(1.0 if row.hardening_success else 0.0 for row in rows),
            }
        summary[str(depth)] = depth_rows
    return summary


def median(values: Any) -> float | None:
    collected = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not collected:
        return None
    return statistics.median(collected)


def mean(values: Any) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return sum(collected) / len(collected)


def write_case_artifact(output_dir: Path, result: SymbolicRunResult) -> None:
    path = (
        output_dir
        / "cases"
        / result.task_id
        / result.model_family.value
        / f"seed{result.seed}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_benchmark_report(report: SymbolicBenchmarkReport, output_dir: Path) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(to_jsonable(report.manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for result in report.results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_markdown_summary(report), encoding="utf-8")


def render_markdown_summary(report: SymbolicBenchmarkReport) -> str:
    lines = [
        "# Symbolic EML Benchmark Summary",
        "",
        f"Run label: `{report.manifest.run_label}`",
        f"Preset: `{report.manifest.preset.value}`",
        f"Total runs: {report.summary['total_runs']}",
        "",
        "## Model Summary",
        "",
        "| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile | latency us/sample |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_family, row in report.summary["model_families"].items():
        lines.append(
            "| "
            f"{model_family} | "
            f"{fmt(row['median_soft_validation_rmse'])} | "
            f"{fmt(row['median_hardened_validation_rmse'])} | "
            f"{fmt(row['median_hardened_extrapolation_rmse'])} | "
            f"{row['exact_recovery_rate']:.2f} | "
            f"{row['near_exact_recovery_rate']:.2f} | "
            f"{row['hardening_success_rate']:.2f} | "
            f"{row['export_success_rate']:.2f} | "
            f"{row['compile_success_rate']:.2f} | "
            f"{fmt(row['median_compiled_latency_us_per_sample'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `paper-complex-eml` uses `eml(x, y) = exp(x) - log(y)` with complex intermediates and bounded numeric guards.",
            "- `stable-real-eml` is a real-valued gated square surrogate aligned with the repo's practical P1-style primitive, not a faithful paper implementation.",
            "- Exact recovery is numeric and structural: hardened validation RMSE <= 1e-4, extrapolation RMSE <= 5e-4, and compact complexity.",
            "- Near-exact recovery relaxes those thresholds to 1e-2 in-range and 2e-2 extrapolation.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf"
    if abs(value) >= 1000.0 or abs(value) < 0.001 and value != 0.0:
        return f"{value:.3e}"
    return f"{value:.4f}"
