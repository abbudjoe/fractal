from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass
from typing import Callable, Protocol

from python.specs.symbolic import SymbolicModelFamily


MAX_REAL_MAGNITUDE = 1.0e4
EXP_REAL_CLAMP = 8.0
LOG_EPSILON = 1.0e-8


class SymbolicModel(Protocol):
    family: SymbolicModelFamily
    model_label: str

    def parameters(self) -> list[float]:
        ...

    def set_parameters(self, values: list[float]) -> None:
        ...

    def predict(self, xs: tuple[float, ...], *, temperature: float = 1.0) -> tuple[float, ...]:
        ...

    def snap_penalty(self, *, temperature: float = 1.0) -> float:
        ...

    def harden(self) -> "HardenedExpression":
        ...


@dataclass(frozen=True)
class HardenedExpression:
    model_family: SymbolicModelFamily
    expression: str
    python_source: str
    complexity: int
    active_ops: tuple[str, ...]
    symbolic_export: bool
    notes: tuple[str, ...] = ()

    def compile(self) -> Callable[[float], float]:
        namespace = {
            "float": float,
            "complex": complex,
            "math": math,
            "cmath": cmath,
            "safe_div": safe_div,
            "safe_exp": safe_exp,
            "safe_log_abs": safe_log_abs,
            "stable_node": stable_node,
            "eml_complex_real": eml_complex_real,
        }
        code = compile(self.python_source, f"<{self.model_family.value}-expression>", "eval")
        return eval(code, {"__builtins__": {}, **namespace}, {})


@dataclass
class TreeNode:
    depth: int
    terminal_selector: int | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None
    left_selector: int | None = None
    right_selector: int | None = None
    op_selector: int | None = None
    op_param_offset: int | None = None


class ParameterAllocator:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.values: list[float] = []

    def add_logits(self, count: int, *, scale: float = 0.05) -> int:
        offset = len(self.values)
        self.values.extend(self.rng.uniform(-scale, scale) for _ in range(count))
        return offset

    def add_values(self, values: tuple[float, ...]) -> int:
        offset = len(self.values)
        self.values.extend(values)
        return offset


def build_tree(
    depth: int,
    allocator: ParameterAllocator,
    *,
    terminal_count: int,
    internal_option_count: int,
    op_count: int | None = None,
    op_param_count: int = 0,
) -> TreeNode:
    if depth <= 0:
        return TreeNode(
            depth=0,
            terminal_selector=allocator.add_logits(terminal_count),
        )
    left = build_tree(
        depth - 1,
        allocator,
        terminal_count=terminal_count,
        internal_option_count=internal_option_count,
        op_count=op_count,
        op_param_count=op_param_count,
    )
    right = build_tree(
        depth - 1,
        allocator,
        terminal_count=terminal_count,
        internal_option_count=internal_option_count,
        op_count=op_count,
        op_param_count=op_param_count,
    )
    return TreeNode(
        depth=depth,
        left=left,
        right=right,
        left_selector=allocator.add_logits(internal_option_count),
        right_selector=allocator.add_logits(internal_option_count),
        op_selector=None if op_count is None else allocator.add_logits(op_count),
        op_param_offset=None if op_param_count == 0 else allocator.add_values((0.0,) * op_param_count),
    )


class BaseParameterModel:
    family: SymbolicModelFamily
    model_label: str

    def __init__(self, values: list[float]) -> None:
        self._parameters = values

    def parameters(self) -> list[float]:
        return list(self._parameters)

    def set_parameters(self, values: list[float]) -> None:
        if len(values) != len(self._parameters):
            raise ValueError(
                f"parameter length mismatch for {self.model_label}: got {len(values)}, expected {len(self._parameters)}"
            )
        self._parameters = list(values)

    @property
    def parameter_count(self) -> int:
        return len(self._parameters)

    def initialize_readout(self, mean: float, scale: float) -> None:
        del mean, scale

    def snap_penalty(self, *, temperature: float = 1.0) -> float:
        del temperature
        return 0.0


class PaperComplexEmlTree(BaseParameterModel):
    family = SymbolicModelFamily.PAPER_COMPLEX_EML

    def __init__(self, depth: int, seed: int) -> None:
        allocator = ParameterAllocator(random.Random(seed))
        self.depth = depth
        self.terminal_values = ("1", "x")
        self.root = build_tree(
            depth,
            allocator,
            terminal_count=2,
            internal_option_count=4,
        )
        self.readout_offset = allocator.add_values((1.0, 0.0))
        super().__init__(allocator.values)
        self.model_label = f"paper-complex-eml-depth{depth}"

    def initialize_readout(self, mean: float, scale: float) -> None:
        params = self.parameters()
        params[self.readout_offset] = max(0.1, scale)
        params[self.readout_offset + 1] = mean
        self.set_parameters(params)

    def predict(self, xs: tuple[float, ...], *, temperature: float = 1.0) -> tuple[float, ...]:
        scale = self._parameters[self.readout_offset]
        bias = self._parameters[self.readout_offset + 1]
        values = []
        for value in xs:
            raw = self._eval_node(self.root, value, hard=False, temperature=temperature)
            values.append(_finite_real(scale * raw.real + bias))
        return tuple(values)

    def snap_penalty(self, *, temperature: float = 1.0) -> float:
        probs = self._selector_max_probs(self.root, temperature)
        if not probs:
            return 0.0
        return sum(1.0 - prob for prob in probs) / len(probs)

    def harden(self) -> HardenedExpression:
        root_expr, root_source, complexity, ops = self._harden_node(self.root)
        scale = snap_scalar(self._parameters[self.readout_offset])
        bias = snap_scalar(self._parameters[self.readout_offset + 1])
        return self.harden_with_readout(scale, bias, root_expr, root_source, complexity, ops)

    def harden_root(self) -> tuple[str, str, int, list[str]]:
        return self._harden_node(self.root)

    def harden_with_readout(
        self,
        scale: float,
        bias: float,
        root_expr: str,
        root_source: str,
        complexity: int,
        ops: list[str],
        *,
        extra_notes: tuple[str, ...] = (),
    ) -> HardenedExpression:
        expression = f"({scale:.8g} * real({root_expr}) + {bias:.8g})"
        python_source = (
            "lambda x: "
            f"float(({scale:.17g}) * ({root_source}).real + ({bias:.17g}))"
        )
        return HardenedExpression(
            model_family=self.family,
            expression=expression,
            python_source=python_source,
            complexity=complexity + 1,
            active_ops=tuple(ops),
            symbolic_export=True,
            notes=(
                "Uses complex principal-branch EML with bounded exp/log guards in numeric execution.",
                "Terminal grammar is restricted to 1 and x, matching the paper-closer univariate surface.",
            )
            + extra_notes,
        )

    def _eval_node(self, node: TreeNode, x_value: float, *, hard: bool, temperature: float) -> complex:
        if node.depth == 0:
            options = (1.0 + 0.0j, complex(x_value, 0.0))
            return select_value(self._parameters, node.terminal_selector, options, hard=hard, temperature=temperature)
        assert node.left is not None and node.right is not None
        left = self._eval_node(node.left, x_value, hard=hard, temperature=temperature)
        right = self._eval_node(node.right, x_value, hard=hard, temperature=temperature)
        options = (1.0 + 0.0j, complex(x_value, 0.0), left, right)
        left_arg = select_value(self._parameters, node.left_selector, options, hard=hard, temperature=temperature)
        right_arg = select_value(self._parameters, node.right_selector, options, hard=hard, temperature=temperature)
        return eml_complex(left_arg, right_arg)

    def _selector_max_probs(self, node: TreeNode, temperature: float) -> list[float]:
        if node.depth == 0:
            assert node.terminal_selector is not None
            return [max(softmax(self._parameters[node.terminal_selector : node.terminal_selector + 2], temperature))]
        assert node.left is not None and node.right is not None
        probs = self._selector_max_probs(node.left, temperature)
        probs.extend(self._selector_max_probs(node.right, temperature))
        assert node.left_selector is not None and node.right_selector is not None
        probs.append(max(softmax(self._parameters[node.left_selector : node.left_selector + 4], temperature)))
        probs.append(max(softmax(self._parameters[node.right_selector : node.right_selector + 4], temperature)))
        return probs

    def _harden_node(self, node: TreeNode) -> tuple[str, str, int, list[str]]:
        if node.depth == 0:
            assert node.terminal_selector is not None
            index = argmax(self._parameters[node.terminal_selector : node.terminal_selector + 2])
            expr = "1" if index == 0 else "x"
            source = "complex(1.0, 0.0)" if index == 0 else "complex(x, 0.0)"
            return expr, source, 1, []
        assert node.left is not None and node.right is not None
        left_expr, left_source, left_complexity, left_ops = self._harden_node(node.left)
        right_expr, right_source, right_complexity, right_ops = self._harden_node(node.right)
        options = [
            ("1", "complex(1.0, 0.0)", 1, []),
            ("x", "complex(x, 0.0)", 1, []),
            (left_expr, left_source, left_complexity, left_ops),
            (right_expr, right_source, right_complexity, right_ops),
        ]
        assert node.left_selector is not None and node.right_selector is not None
        left_index = argmax(self._parameters[node.left_selector : node.left_selector + 4])
        right_index = argmax(self._parameters[node.right_selector : node.right_selector + 4])
        left_choice = options[left_index]
        right_choice = options[right_index]
        expr = f"eml({left_choice[0]}, {right_choice[0]})"
        source = f"eml_complex_real({left_choice[1]}, {right_choice[1]})"
        ops = list(left_choice[3]) + list(right_choice[3]) + ["eml"]
        return expr, source, 1 + left_choice[2] + right_choice[2], ops


class StableRealEmlTree(BaseParameterModel):
    family = SymbolicModelFamily.STABLE_REAL_EML

    def __init__(self, depth: int, seed: int) -> None:
        allocator = ParameterAllocator(random.Random(seed))
        self.depth = depth
        self.terminal_count = 4
        self.internal_option_count = 6
        self.root = build_tree(
            depth,
            allocator,
            terminal_count=self.terminal_count,
            internal_option_count=self.internal_option_count,
            op_param_count=5,
        )
        _initialize_stable_params(self.root, allocator.values)
        self.readout_offset = allocator.add_values((1.0, 0.0))
        super().__init__(allocator.values)
        self.model_label = f"stable-real-eml-depth{depth}"

    def initialize_readout(self, mean: float, scale: float) -> None:
        params = self.parameters()
        params[self.readout_offset] = max(0.1, scale)
        params[self.readout_offset + 1] = mean
        self.set_parameters(params)

    def predict(self, xs: tuple[float, ...], *, temperature: float = 1.0) -> tuple[float, ...]:
        scale = self._parameters[self.readout_offset]
        bias = self._parameters[self.readout_offset + 1]
        return tuple(
            _finite_real(scale * self._eval_node(self.root, value, hard=False, temperature=temperature) + bias)
            for value in xs
        )

    def snap_penalty(self, *, temperature: float = 1.0) -> float:
        probs = stable_selector_max_probs(self.root, self._parameters, temperature)
        if not probs:
            return 0.0
        return sum(1.0 - prob for prob in probs) / len(probs)

    def harden(self) -> HardenedExpression:
        root_expr, root_source, complexity, ops = self._harden_node(self.root)
        scale = snap_scalar(self._parameters[self.readout_offset])
        bias = snap_scalar(self._parameters[self.readout_offset + 1])
        expression = f"({scale:.8g} * {root_expr} + {bias:.8g})"
        python_source = "lambda x: float(" f"({scale:.17g}) * ({root_source}) + ({bias:.17g}))"
        return HardenedExpression(
            model_family=self.family,
            expression=expression,
            python_source=python_source,
            complexity=complexity + 1,
            active_ops=tuple(ops),
            symbolic_export=True,
            notes=(
                "Real-valued gated square surrogate patterned after the repo P1 stable primitive.",
                "This is not the paper's complex EML operator.",
            ),
        )

    def _eval_node(self, node: TreeNode, x_value: float, *, hard: bool, temperature: float) -> float:
        if node.depth == 0:
            options = (1.0, x_value, -1.0, 0.0)
            return float(select_value(self._parameters, node.terminal_selector, options, hard=hard, temperature=temperature))
        assert node.left is not None and node.right is not None
        left = self._eval_node(node.left, x_value, hard=hard, temperature=temperature)
        right = self._eval_node(node.right, x_value, hard=hard, temperature=temperature)
        options = (1.0, x_value, -1.0, 0.0, left, right)
        left_arg = float(select_value(self._parameters, node.left_selector, options, hard=hard, temperature=temperature))
        right_arg = float(select_value(self._parameters, node.right_selector, options, hard=hard, temperature=temperature))
        assert node.op_param_offset is not None
        gate_logit, state_weight, input_weight, square_weight, bias = self._parameters[
            node.op_param_offset : node.op_param_offset + 5
        ]
        return stable_node(left_arg, right_arg, gate_logit, state_weight, input_weight, square_weight, bias)

    def _harden_node(self, node: TreeNode) -> tuple[str, str, int, list[str]]:
        if node.depth == 0:
            assert node.terminal_selector is not None
            index = argmax(self._parameters[node.terminal_selector : node.terminal_selector + self.terminal_count])
            options = [
                ("1", "1.0"),
                ("x", "x"),
                ("-1", "-1.0"),
                ("0", "0.0"),
            ]
            expr, source = options[index]
            return expr, source, 1, []
        assert node.left is not None and node.right is not None
        left_expr, left_source, left_complexity, left_ops = self._harden_node(node.left)
        right_expr, right_source, right_complexity, right_ops = self._harden_node(node.right)
        options = [
            ("1", "1.0", 1, []),
            ("x", "x", 1, []),
            ("-1", "-1.0", 1, []),
            ("0", "0.0", 1, []),
            (left_expr, left_source, left_complexity, left_ops),
            (right_expr, right_source, right_complexity, right_ops),
        ]
        assert node.left_selector is not None and node.right_selector is not None and node.op_param_offset is not None
        left_index = argmax(self._parameters[node.left_selector : node.left_selector + self.internal_option_count])
        right_index = argmax(self._parameters[node.right_selector : node.right_selector + self.internal_option_count])
        left_choice = options[left_index]
        right_choice = options[right_index]
        gate_logit, state_weight, input_weight, square_weight, bias = (
            snap_scalar(value) for value in self._parameters[node.op_param_offset : node.op_param_offset + 5]
        )
        expr = (
            "stable("
            f"{left_choice[0]}, {right_choice[0]}; "
            f"g={gate_logit:.4g}, sw={state_weight:.4g}, iw={input_weight:.4g}, "
            f"qw={square_weight:.4g}, b={bias:.4g})"
        )
        source = (
            "stable_node("
            f"{left_choice[1]}, {right_choice[1]}, {gate_logit:.17g}, {state_weight:.17g}, "
            f"{input_weight:.17g}, {square_weight:.17g}, {bias:.17g})"
        )
        ops = list(left_choice[3]) + list(right_choice[3]) + ["stable_eml_surrogate"]
        return expr, source, 1 + left_choice[2] + right_choice[2], ops


class GenericTreeControl(BaseParameterModel):
    family = SymbolicModelFamily.GENERIC_TREE
    ops = ("add", "sub", "mul", "div", "exp_left", "log_abs_left", "id_left", "id_right")

    def __init__(self, depth: int, seed: int) -> None:
        allocator = ParameterAllocator(random.Random(seed))
        self.depth = depth
        self.terminal_count = 6
        self.internal_option_count = 8
        self.root = build_tree(
            depth,
            allocator,
            terminal_count=self.terminal_count,
            internal_option_count=self.internal_option_count,
            op_count=len(self.ops),
        )
        self.readout_offset = allocator.add_values((1.0, 0.0))
        super().__init__(allocator.values)
        self.model_label = f"generic-tree-depth{depth}"

    def initialize_readout(self, mean: float, scale: float) -> None:
        params = self.parameters()
        params[self.readout_offset] = max(0.1, scale)
        params[self.readout_offset + 1] = mean
        self.set_parameters(params)

    def predict(self, xs: tuple[float, ...], *, temperature: float = 1.0) -> tuple[float, ...]:
        scale = self._parameters[self.readout_offset]
        bias = self._parameters[self.readout_offset + 1]
        return tuple(
            _finite_real(scale * self._eval_node(self.root, value, hard=False, temperature=temperature) + bias)
            for value in xs
        )

    def snap_penalty(self, *, temperature: float = 1.0) -> float:
        probs = generic_selector_max_probs(self.root, self._parameters, temperature, len(self.ops))
        if not probs:
            return 0.0
        return sum(1.0 - prob for prob in probs) / len(probs)

    def harden(self) -> HardenedExpression:
        root_expr, root_source, complexity, ops = self._harden_node(self.root)
        scale = snap_scalar(self._parameters[self.readout_offset])
        bias = snap_scalar(self._parameters[self.readout_offset + 1])
        expression = f"({scale:.8g} * {root_expr} + {bias:.8g})"
        python_source = "lambda x: float(" f"({scale:.17g}) * ({root_source}) + ({bias:.17g}))"
        return HardenedExpression(
            model_family=self.family,
            expression=expression,
            python_source=python_source,
            complexity=complexity + 1,
            active_ops=tuple(ops),
            symbolic_export=True,
            notes=("Generic differentiable binary tree with primitive choices; no EML-specific claim.",),
        )

    def _eval_node(self, node: TreeNode, x_value: float, *, hard: bool, temperature: float) -> float:
        if node.depth == 0:
            options = (1.0, x_value, -1.0, 0.0, 0.5, 2.0)
            return float(select_value(self._parameters, node.terminal_selector, options, hard=hard, temperature=temperature))
        assert node.left is not None and node.right is not None
        left = self._eval_node(node.left, x_value, hard=hard, temperature=temperature)
        right = self._eval_node(node.right, x_value, hard=hard, temperature=temperature)
        options = (1.0, x_value, -1.0, 0.0, 0.5, 2.0, left, right)
        left_arg = float(select_value(self._parameters, node.left_selector, options, hard=hard, temperature=temperature))
        right_arg = float(select_value(self._parameters, node.right_selector, options, hard=hard, temperature=temperature))
        op_values = (
            safe_add(left_arg, right_arg),
            safe_sub(left_arg, right_arg),
            safe_mul(left_arg, right_arg),
            safe_div(left_arg, right_arg),
            safe_exp(left_arg),
            safe_log_abs(left_arg),
            left_arg,
            right_arg,
        )
        assert node.op_selector is not None
        return float(select_value(self._parameters, node.op_selector, op_values, hard=hard, temperature=temperature))

    def _harden_node(self, node: TreeNode) -> tuple[str, str, int, list[str]]:
        if node.depth == 0:
            assert node.terminal_selector is not None
            index = argmax(self._parameters[node.terminal_selector : node.terminal_selector + self.terminal_count])
            options = [
                ("1", "1.0"),
                ("x", "x"),
                ("-1", "-1.0"),
                ("0", "0.0"),
                ("0.5", "0.5"),
                ("2", "2.0"),
            ]
            expr, source = options[index]
            return expr, source, 1, []
        assert node.left is not None and node.right is not None
        left_expr, left_source, left_complexity, left_ops = self._harden_node(node.left)
        right_expr, right_source, right_complexity, right_ops = self._harden_node(node.right)
        options = [
            ("1", "1.0", 1, []),
            ("x", "x", 1, []),
            ("-1", "-1.0", 1, []),
            ("0", "0.0", 1, []),
            ("0.5", "0.5", 1, []),
            ("2", "2.0", 1, []),
            (left_expr, left_source, left_complexity, left_ops),
            (right_expr, right_source, right_complexity, right_ops),
        ]
        assert node.left_selector is not None and node.right_selector is not None and node.op_selector is not None
        left_index = argmax(self._parameters[node.left_selector : node.left_selector + self.internal_option_count])
        right_index = argmax(self._parameters[node.right_selector : node.right_selector + self.internal_option_count])
        op_index = argmax(self._parameters[node.op_selector : node.op_selector + len(self.ops)])
        left_choice = options[left_index]
        right_choice = options[right_index]
        op = self.ops[op_index]
        expr, source = render_generic_op(op, left_choice[0], right_choice[0], left_choice[1], right_choice[1])
        ops = list(left_choice[3]) + list(right_choice[3]) + [op]
        return expr, source, 1 + left_choice[2] + right_choice[2], ops


class SmallMlpBaseline(BaseParameterModel):
    family = SymbolicModelFamily.SMALL_MLP

    def __init__(self, hidden_units: int, seed: int) -> None:
        rng = random.Random(seed)
        self.hidden_units = hidden_units
        values: list[float] = []
        values.extend(rng.uniform(-0.6, 0.6) for _ in range(hidden_units))
        values.extend(rng.uniform(-0.1, 0.1) for _ in range(hidden_units))
        values.extend(rng.uniform(-0.6, 0.6) for _ in range(hidden_units))
        values.append(0.0)
        super().__init__(values)
        self.model_label = f"small-mlp-h{hidden_units}"

    def initialize_readout(self, mean: float, scale: float) -> None:
        del scale
        params = self.parameters()
        params[-1] = mean
        self.set_parameters(params)

    def predict(self, xs: tuple[float, ...], *, temperature: float = 1.0) -> tuple[float, ...]:
        del temperature
        return tuple(_finite_real(self._forward(value)) for value in xs)

    def gradient(self, xs: tuple[float, ...], ys: tuple[float, ...], y_scale: float) -> tuple[float, list[float]]:
        count = max(1, len(xs))
        params = self._parameters
        width = self.hidden_units
        w1 = params[:width]
        b1 = params[width : 2 * width]
        w2 = params[2 * width : 3 * width]
        grad = [0.0 for _ in params]
        loss = 0.0
        scale = max(y_scale, 1.0e-6)
        for x_value, target in zip(xs, ys):
            hidden = [math.tanh(w1[index] * x_value + b1[index]) for index in range(width)]
            pred = sum(w2[index] * hidden[index] for index in range(width)) + params[-1]
            err = (pred - target) / scale
            loss += err * err
            d_pred = 2.0 * err / (scale * count)
            for index, activation in enumerate(hidden):
                grad[2 * width + index] += d_pred * activation
                common = d_pred * w2[index] * (1.0 - activation * activation)
                grad[index] += common * x_value
                grad[width + index] += common
            grad[-1] += d_pred
        return loss / count, grad

    def harden(self) -> HardenedExpression:
        params = [snap_scalar(value) for value in self._parameters]
        width = self.hidden_units
        terms = []
        source_terms = []
        for index in range(width):
            w = params[index]
            b = params[width + index]
            v = params[2 * width + index]
            if abs(v) < 1.0e-8:
                continue
            terms.append(f"{v:.6g}*tanh({w:.6g}*x + {b:.6g})")
            source_terms.append(f"({v:.17g})*math.tanh(({w:.17g})*x + ({b:.17g}))")
        bias = params[-1]
        expression = " + ".join(terms + [f"{bias:.6g}"]) if terms else f"{bias:.6g}"
        python_source = "lambda x: float(" + " + ".join(source_terms + [f"({bias:.17g})"]) + ")"
        return HardenedExpression(
            model_family=self.family,
            expression=expression,
            python_source=python_source,
            complexity=len(terms) + 1,
            active_ops=tuple("tanh" for _ in terms),
            symbolic_export=False,
            notes=("Dense MLP baseline exports executable weights, not a recovered symbolic formula.",),
        )

    def _forward(self, x_value: float) -> float:
        width = self.hidden_units
        params = self._parameters
        total = params[-1]
        for index in range(width):
            total += params[2 * width + index] * math.tanh(params[index] * x_value + params[width + index])
        return total


def build_symbolic_model(
    family: SymbolicModelFamily,
    *,
    depth: int,
    seed: int,
    hidden_units: int,
) -> SymbolicModel:
    if family is SymbolicModelFamily.PAPER_COMPLEX_EML:
        return PaperComplexEmlTree(depth=depth, seed=seed)
    if family is SymbolicModelFamily.STABLE_REAL_EML:
        return StableRealEmlTree(depth=depth, seed=seed)
    if family is SymbolicModelFamily.GENERIC_TREE:
        return GenericTreeControl(depth=depth, seed=seed)
    if family is SymbolicModelFamily.SMALL_MLP:
        return SmallMlpBaseline(hidden_units=hidden_units, seed=seed)
    raise ValueError(f"unsupported symbolic model family: {family}")


def softmax(values: list[float], temperature: float) -> list[float]:
    inv_temp = 1.0 / max(temperature, 1.0e-6)
    scaled = [value * inv_temp for value in values]
    offset = max(scaled)
    exp_values = [math.exp(value - offset) for value in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def select_value(
    params: list[float],
    offset: int | None,
    options: tuple[complex, ...] | tuple[float, ...],
    *,
    hard: bool,
    temperature: float,
) -> complex | float:
    if offset is None:
        raise ValueError("selector offset is missing")
    logits = params[offset : offset + len(options)]
    if hard:
        return options[argmax(logits)]
    weights = softmax(logits, temperature)
    total = options[0] * 0.0
    for weight, option in zip(weights, options):
        total += option * weight
    return total


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def safe_add(left: float, right: float) -> float:
    return _finite_real(left + right)


def safe_sub(left: float, right: float) -> float:
    return _finite_real(left - right)


def safe_mul(left: float, right: float) -> float:
    return _finite_real(left * right)


def safe_div(left: float, right: float) -> float:
    denominator = right if abs(right) >= LOG_EPSILON else math.copysign(LOG_EPSILON, right if right != 0.0 else 1.0)
    return _finite_real(left / denominator)


def safe_exp(value: float) -> float:
    return _finite_real(math.exp(max(-EXP_REAL_CLAMP, min(EXP_REAL_CLAMP, value))))


def safe_log_abs(value: float) -> float:
    return _finite_real(math.log(abs(value) + LOG_EPSILON))


def stable_node(
    state: float,
    inputs: float,
    gate_logit: float,
    state_weight: float,
    input_weight: float,
    square_weight: float,
    bias: float,
) -> float:
    gate = sigmoid(gate_logit)
    squared = max(-64.0, min(64.0, state * state))
    update = state_weight * state + input_weight * inputs + square_weight * squared + bias
    return _finite_real(gate * update + (1.0 - gate) * state)


def eml_complex(left: complex, right: complex) -> complex:
    exp_arg = complex(max(-EXP_REAL_CLAMP, min(EXP_REAL_CLAMP, left.real)), left.imag)
    safe_right = right
    if abs(safe_right) < LOG_EPSILON:
        safe_right += complex(LOG_EPSILON, 0.0)
    try:
        value = cmath.exp(exp_arg) - cmath.log(safe_right)
    except (OverflowError, ValueError):
        return complex(MAX_REAL_MAGNITUDE, 0.0)
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        return complex(MAX_REAL_MAGNITUDE, 0.0)
    if magnitude > MAX_REAL_MAGNITUDE:
        return value / magnitude * MAX_REAL_MAGNITUDE
    return value


def eml_complex_real(left: complex, right: complex) -> complex:
    return eml_complex(left, right)


def _finite_real(value: float) -> float:
    if not math.isfinite(value):
        return math.copysign(MAX_REAL_MAGNITUDE, value if value != 0.0 else 1.0)
    return max(-MAX_REAL_MAGNITUDE, min(MAX_REAL_MAGNITUDE, value))


def snap_scalar(value: float) -> float:
    targets = (-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    nearest = min(targets, key=lambda target: abs(target - value))
    if abs(value - nearest) <= 0.08:
        return nearest
    return value


def _initialize_stable_params(node: TreeNode, params: list[float]) -> None:
    if node.depth == 0:
        return
    assert node.left is not None and node.right is not None
    _initialize_stable_params(node.left, params)
    _initialize_stable_params(node.right, params)
    if node.op_param_offset is None:
        return
    defaults = (0.0, 0.55, 0.45, 0.08, 0.0)
    for index, value in enumerate(defaults):
        params[node.op_param_offset + index] = value


def stable_selector_max_probs(node: TreeNode, params: list[float], temperature: float) -> list[float]:
    if node.depth == 0:
        assert node.terminal_selector is not None
        return [max(softmax(params[node.terminal_selector : node.terminal_selector + 4], temperature))]
    assert node.left is not None and node.right is not None
    values = stable_selector_max_probs(node.left, params, temperature)
    values.extend(stable_selector_max_probs(node.right, params, temperature))
    assert node.left_selector is not None and node.right_selector is not None
    values.append(max(softmax(params[node.left_selector : node.left_selector + 6], temperature)))
    values.append(max(softmax(params[node.right_selector : node.right_selector + 6], temperature)))
    return values


def generic_selector_max_probs(
    node: TreeNode,
    params: list[float],
    temperature: float,
    op_count: int,
) -> list[float]:
    if node.depth == 0:
        assert node.terminal_selector is not None
        return [max(softmax(params[node.terminal_selector : node.terminal_selector + 6], temperature))]
    assert node.left is not None and node.right is not None
    values = generic_selector_max_probs(node.left, params, temperature, op_count)
    values.extend(generic_selector_max_probs(node.right, params, temperature, op_count))
    assert node.left_selector is not None and node.right_selector is not None and node.op_selector is not None
    values.append(max(softmax(params[node.left_selector : node.left_selector + 8], temperature)))
    values.append(max(softmax(params[node.right_selector : node.right_selector + 8], temperature)))
    values.append(max(softmax(params[node.op_selector : node.op_selector + op_count], temperature)))
    return values


def render_generic_op(
    op: str,
    left_expr: str,
    right_expr: str,
    left_source: str,
    right_source: str,
) -> tuple[str, str]:
    if op == "add":
        return f"({left_expr} + {right_expr})", f"(({left_source}) + ({right_source}))"
    if op == "sub":
        return f"({left_expr} - {right_expr})", f"(({left_source}) - ({right_source}))"
    if op == "mul":
        return f"({left_expr} * {right_expr})", f"(({left_source}) * ({right_source}))"
    if op == "div":
        return f"({left_expr} / {right_expr})", f"safe_div(({left_source}), ({right_source}))"
    if op == "exp_left":
        return f"exp({left_expr})", f"safe_exp(({left_source}))"
    if op == "log_abs_left":
        return f"log_abs({left_expr})", f"safe_log_abs(({left_source}))"
    if op == "id_left":
        return left_expr, left_source
    if op == "id_right":
        return right_expr, right_source
    raise ValueError(f"unsupported generic op: {op}")
