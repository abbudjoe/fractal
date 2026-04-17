from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from python.specs.symbolic import SymbolicDatasetSpec


Number = float


@dataclass(frozen=True)
class FormulaExpr:
    kind: str
    args: tuple["FormulaExpr", ...] = ()
    value: float | None = None

    def eval(self, x: float) -> float:
        if self.kind == "x":
            return x
        if self.kind == "const":
            return float(self.value if self.value is not None else 0.0)
        if self.kind == "add":
            return self.args[0].eval(x) + self.args[1].eval(x)
        if self.kind == "sub":
            return self.args[0].eval(x) - self.args[1].eval(x)
        if self.kind == "mul":
            return self.args[0].eval(x) * self.args[1].eval(x)
        if self.kind == "div":
            denominator = self.args[1].eval(x)
            if abs(denominator) < 1.0e-8:
                return math.nan
            return self.args[0].eval(x) / denominator
        if self.kind == "exp":
            return math.exp(self.args[0].eval(x))
        if self.kind == "log":
            value = self.args[0].eval(x)
            if value <= 0.0:
                return math.nan
            return math.log(value)
        raise ValueError(f"unsupported formula kind: {self.kind}")

    def depth(self) -> int:
        if not self.args:
            return 0
        return 1 + max(arg.depth() for arg in self.args)

    def complexity(self) -> int:
        return 1 + sum(arg.complexity() for arg in self.args)

    def op_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.kind not in {"x", "const"}:
            counts[self.kind] = 1
        for arg in self.args:
            for key, value in arg.op_histogram().items():
                counts[key] = counts.get(key, 0) + value
        return counts

    def source(self) -> str:
        if self.kind == "x":
            return "x"
        if self.kind == "const":
            value = float(self.value if self.value is not None else 0.0)
            return f"{value:.8g}"
        if self.kind == "add":
            return f"({self.args[0].source()} + {self.args[1].source()})"
        if self.kind == "sub":
            return f"({self.args[0].source()} - {self.args[1].source()})"
        if self.kind == "mul":
            return f"({self.args[0].source()} * {self.args[1].source()})"
        if self.kind == "div":
            return f"({self.args[0].source()} / {self.args[1].source()})"
        if self.kind == "exp":
            return f"exp({self.args[0].source()})"
        if self.kind == "log":
            return f"log({self.args[0].source()})"
        raise ValueError(f"unsupported formula kind: {self.kind}")


def x() -> FormulaExpr:
    return FormulaExpr("x")


def c(value: float) -> FormulaExpr:
    return FormulaExpr("const", value=value)


def add(left: FormulaExpr, right: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("add", (left, right))


def sub(left: FormulaExpr, right: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("sub", (left, right))


def mul(left: FormulaExpr, right: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("mul", (left, right))


def div(left: FormulaExpr, right: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("div", (left, right))


def exp(arg: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("exp", (arg,))


def log(arg: FormulaExpr) -> FormulaExpr:
    return FormulaExpr("log", (arg,))


@dataclass(frozen=True)
class SymbolicTask:
    task_id: str
    difficulty_depth: int
    family: str
    expression: FormulaExpr
    train_range: tuple[float, float]
    validation_range: tuple[float, float]
    extrapolation_range: tuple[float, float]

    def metadata(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "difficulty_depth": self.difficulty_depth,
            "family": self.family,
            "source_formula": self.expression.source(),
            "ast_depth": self.expression.depth(),
            "ast_complexity": self.expression.complexity(),
            "operator_histogram": self.expression.op_histogram(),
            "train_range": list(self.train_range),
            "validation_range": list(self.validation_range),
            "extrapolation_range": list(self.extrapolation_range),
        }


@dataclass(frozen=True)
class FormulaSplit:
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    def finite(self) -> bool:
        return all(math.isfinite(x) and math.isfinite(y) for x, y in zip(self.xs, self.ys))


@dataclass(frozen=True)
class FormulaDataset:
    task: SymbolicTask
    train: FormulaSplit
    validation: FormulaSplit
    extrapolation: FormulaSplit

    def metadata(self) -> dict[str, object]:
        return {
            "task": self.task.metadata(),
            "split_sizes": {
                "train": len(self.train.xs),
                "validation": len(self.validation.xs),
                "extrapolation": len(self.extrapolation.xs),
            },
        }


def default_symbolic_tasks(tasks_per_depth: int = 1) -> list[SymbolicTask]:
    candidates = [
        SymbolicTask(
            task_id="d1_affine",
            difficulty_depth=1,
            family="linear",
            expression=sub(mul(c(1.25), x()), c(0.5)),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d1_exp_soft",
            difficulty_depth=1,
            family="exponential",
            expression=exp(mul(c(0.35), x())),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d2_quadratic_mix",
            difficulty_depth=2,
            family="polynomial",
            expression=add(mul(x(), x()), mul(c(0.5), x())),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d2_reciprocal_shift",
            difficulty_depth=2,
            family="safe-ratio",
            expression=add(div(c(1.0), add(x(), c(2.5))), mul(c(0.2), x())),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d3_log_quadratic",
            difficulty_depth=3,
            family="log-polynomial",
            expression=add(log(add(x(), c(2.0))), mul(c(0.3), mul(x(), x()))),
            train_range=(-0.8, 1.0),
            validation_range=(-0.8, 1.0),
            extrapolation_range=(1.1, 2.0),
        ),
        SymbolicTask(
            task_id="d3_ratio_product",
            difficulty_depth=3,
            family="safe-ratio-product",
            expression=div(mul(add(x(), c(1.2)), sub(x(), c(0.3))), add(x(), c(2.5))),
            train_range=(-0.9, 1.0),
            validation_range=(-0.9, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d4_exp_log_product",
            difficulty_depth=4,
            family="composition",
            expression=mul(log(add(exp(mul(c(0.35), x())), c(1.5))), add(x(), c(0.5))),
            train_range=(-0.9, 1.0),
            validation_range=(-0.9, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
        SymbolicTask(
            task_id="d4_nested_ratio_exp",
            difficulty_depth=4,
            family="composition-ratio",
            expression=add(
                div(exp(mul(c(0.2), x())), add(x(), c(3.0))),
                log(add(mul(x(), x()), c(1.5))),
            ),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.15, 2.05),
        ),
    ]
    tasks: list[SymbolicTask] = []
    for depth in (1, 2, 3, 4):
        depth_tasks = [task for task in candidates if task.difficulty_depth == depth]
        tasks.extend(depth_tasks[:tasks_per_depth])
    return tasks


def tier0_exact_recovery_tasks() -> list[SymbolicTask]:
    return [
        SymbolicTask(
            task_id="t0_exp_x",
            difficulty_depth=1,
            family="eml-native-exp",
            expression=exp(x()),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.1, 1.8),
        ),
        SymbolicTask(
            task_id="t0_e_minus_log_x",
            difficulty_depth=1,
            family="eml-native-log",
            expression=sub(c(math.e), log(x())),
            train_range=(0.25, 1.25),
            validation_range=(0.25, 1.25),
            extrapolation_range=(1.35, 2.0),
        ),
        SymbolicTask(
            task_id="t0_eml_x_x",
            difficulty_depth=1,
            family="eml-native-exp-log",
            expression=sub(exp(x()), log(x())),
            train_range=(0.25, 1.25),
            validation_range=(0.25, 1.25),
            extrapolation_range=(1.35, 2.0),
        ),
        SymbolicTask(
            task_id="t0_nested_exp",
            difficulty_depth=2,
            family="eml-native-depth2-exp",
            expression=exp(exp(x())),
            train_range=(-0.75, 0.25),
            validation_range=(-0.75, 0.25),
            extrapolation_range=(0.3, 0.7),
        ),
        SymbolicTask(
            task_id="t0_nested_log",
            difficulty_depth=2,
            family="eml-native-depth2-log",
            expression=sub(c(math.e), log(sub(c(math.e), log(x())))),
            train_range=(0.5, 1.5),
            validation_range=(0.5, 1.5),
            extrapolation_range=(1.6, 2.2),
        ),
        SymbolicTask(
            task_id="t0_eml_log_then_exp",
            difficulty_depth=2,
            family="eml-native-depth2-exp-log",
            expression=exp(sub(c(math.e), log(x()))),
            train_range=(0.8, 1.6),
            validation_range=(0.8, 1.6),
            extrapolation_range=(1.7, 2.5),
        ),
        SymbolicTask(
            task_id="t0_square",
            difficulty_depth=2,
            family="polynomial",
            expression=mul(x(), x()),
            train_range=(-1.0, 1.0),
            validation_range=(-1.0, 1.0),
            extrapolation_range=(1.1, 1.8),
        ),
        SymbolicTask(
            task_id="t0_safe_ratio",
            difficulty_depth=2,
            family="safe-ratio",
            expression=div(x(), add(x(), c(2.0))),
            train_range=(-0.8, 1.0),
            validation_range=(-0.8, 1.0),
            extrapolation_range=(1.1, 1.8),
        ),
    ]


def sample_symbolic_dataset(
    task: SymbolicTask,
    spec: SymbolicDatasetSpec,
    *,
    seed: int,
) -> FormulaDataset:
    rng = random.Random(seed)
    train_xs = _random_samples(task.train_range, spec.train_samples, rng)
    validation_xs = _linspace(task.validation_range, spec.validation_samples)
    extrapolation_xs = _linspace(task.extrapolation_range, spec.extrapolation_samples)
    dataset = FormulaDataset(
        task=task,
        train=_evaluate_split(task.expression, train_xs),
        validation=_evaluate_split(task.expression, validation_xs),
        extrapolation=_evaluate_split(task.expression, extrapolation_xs),
    )
    if not dataset.train.finite() or not dataset.validation.finite() or not dataset.extrapolation.finite():
        raise ValueError(f"generated non-finite dataset for task {task.task_id}")
    return dataset


def _linspace(value_range: tuple[float, float], count: int) -> tuple[float, ...]:
    if count == 1:
        return ((value_range[0] + value_range[1]) * 0.5,)
    start, end = value_range
    step = (end - start) / float(count - 1)
    return tuple(start + index * step for index in range(count))


def _random_samples(value_range: tuple[float, float], count: int, rng: random.Random) -> tuple[float, ...]:
    start, end = value_range
    return tuple(rng.uniform(start, end) for _ in range(count))


def _evaluate_split(expression: FormulaExpr, xs: tuple[float, ...]) -> FormulaSplit:
    return FormulaSplit(xs=xs, ys=tuple(expression.eval(value) for value in xs))


def evaluate_formula_callable(func: Callable[[float], float], xs: tuple[float, ...]) -> tuple[float, ...]:
    values = []
    for value in xs:
        try:
            output = float(func(value))
        except (ArithmeticError, ValueError, OverflowError):
            output = math.nan
        values.append(output)
    return tuple(values)
