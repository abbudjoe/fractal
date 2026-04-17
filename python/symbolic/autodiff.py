from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from python.symbolic.formulas import FormulaSplit
from python.symbolic.models import (
    EXP_REAL_CLAMP,
    LOG_EPSILON,
    MAX_REAL_MAGNITUDE,
    GenericTreeControl,
    PaperComplexEmlTree,
    StableRealEmlTree,
    TreeNode,
    argmax,
    safe_exp,
    sigmoid,
    snap_scalar,
)


@dataclass(frozen=True)
class Dual:
    value: complex
    grad: tuple[complex, ...]

    def real(self) -> "Dual":
        return Dual(
            complex(self.value.real, 0.0),
            tuple(complex(entry.real, 0.0) for entry in self.grad),
        )


def autodiff_loss_and_gradient(
    model: object,
    split: FormulaSplit,
    *,
    y_scale: float,
    temperature: float,
    entropy_weight: float,
) -> tuple[float, list[float]]:
    params = model.parameters()
    dual_params = dual_parameters(params)
    total_loss = 0.0
    total_grad = [0.0 for _ in params]
    scale = max(y_scale, 1.0e-6)
    for x_value, target in zip(split.xs, split.ys):
        prediction = predict_dual(model, dual_params, x_value, temperature)
        pred_value = prediction.value.real
        if not math.isfinite(pred_value):
            total_loss += 1.0e6
            continue
        error = (pred_value - target) / scale
        total_loss += error * error
        coeff = 2.0 * error / scale
        for index, grad_value in enumerate(prediction.grad):
            total_grad[index] += coeff * grad_value.real
    selector_penalty = entropy_penalty(model, dual_params, temperature)
    total_loss += entropy_weight * selector_penalty.value.real * max(1, len(split.xs))
    for index, grad_value in enumerate(selector_penalty.grad):
        total_grad[index] += entropy_weight * grad_value.real * max(1, len(split.xs))
    normalizer = max(1, len(split.xs))
    return total_loss / normalizer, [value / normalizer for value in total_grad]


def predict_dual(model: object, dual_params: list[Dual], x_value: float, temperature: float) -> Dual:
    if isinstance(model, PaperComplexEmlTree):
        root = paper_node_dual(model.root, dual_params, x_value, temperature)
        scale = dual_params[model.readout_offset]
        bias = dual_params[model.readout_offset + 1]
        return dual_add(dual_mul(scale, root.real()), bias)
    if isinstance(model, StableRealEmlTree):
        root = stable_node_dual(model.root, dual_params, x_value, temperature)
        scale = dual_params[model.readout_offset]
        bias = dual_params[model.readout_offset + 1]
        return dual_add(dual_mul(scale, root), bias)
    if isinstance(model, GenericTreeControl):
        root = generic_node_dual(model.root, dual_params, x_value, temperature)
        scale = dual_params[model.readout_offset]
        bias = dual_params[model.readout_offset + 1]
        return dual_add(dual_mul(scale, root), bias)
    raise TypeError(f"unsupported autodiff model: {type(model).__name__}")


def dual_parameters(params: list[float]) -> list[Dual]:
    size = len(params)
    duals = []
    for index, value in enumerate(params):
        grad = [0j for _ in range(size)]
        grad[index] = 1.0 + 0j
        duals.append(Dual(complex(value, 0.0), tuple(grad)))
    return duals


def dual_const(value: complex | float, size: int) -> Dual:
    return Dual(complex(value), tuple(0j for _ in range(size)))


def dual_add(left: Dual, right: Dual) -> Dual:
    return Dual(left.value + right.value, tuple(a + b for a, b in zip(left.grad, right.grad)))


def dual_sub(left: Dual, right: Dual) -> Dual:
    return Dual(left.value - right.value, tuple(a - b for a, b in zip(left.grad, right.grad)))


def dual_mul(left: Dual, right: Dual) -> Dual:
    return Dual(
        left.value * right.value,
        tuple(left.value * b + right.value * a for a, b in zip(left.grad, right.grad)),
    )


def dual_div(left: Dual, right: Dual) -> Dual:
    denominator = right.value
    if abs(denominator) < LOG_EPSILON:
        denominator += complex(LOG_EPSILON, 0.0)
    return Dual(
        left.value / denominator,
        tuple((a * denominator - left.value * b) / (denominator * denominator) for a, b in zip(left.grad, right.grad)),
    )


def dual_exp_complex(arg: Dual) -> Dual:
    raw_real = arg.value.real
    clamped_real = max(-EXP_REAL_CLAMP, min(EXP_REAL_CLAMP, raw_real))
    value = cmath.exp(complex(clamped_real, arg.value.imag))
    real_active = -EXP_REAL_CLAMP < raw_real < EXP_REAL_CLAMP
    grad = []
    for entry in arg.grad:
        dz = complex(entry.real if real_active else 0.0, entry.imag)
        grad.append(value * dz)
    return Dual(value, tuple(grad))


def dual_exp_real(arg: Dual) -> Dual:
    raw = arg.value.real
    clamped = max(-EXP_REAL_CLAMP, min(EXP_REAL_CLAMP, raw))
    value = math.exp(clamped)
    active = -EXP_REAL_CLAMP < raw < EXP_REAL_CLAMP
    return Dual(
        complex(value, 0.0),
        tuple(complex(value * entry.real if active else 0.0, 0.0) for entry in arg.grad),
    )


def dual_log_complex(arg: Dual) -> Dual:
    value = arg.value
    if abs(value) < LOG_EPSILON:
        value += complex(LOG_EPSILON, 0.0)
    return Dual(cmath.log(value), tuple(entry / value for entry in arg.grad))


def dual_log_abs(arg: Dual) -> Dual:
    raw = arg.value.real
    sign = 1.0 if raw >= 0.0 else -1.0
    denom = abs(raw) + LOG_EPSILON
    return Dual(
        complex(math.log(denom), 0.0),
        tuple(complex(sign * entry.real / denom, 0.0) for entry in arg.grad),
    )


def dual_sigmoid(arg: Dual) -> Dual:
    value = sigmoid(arg.value.real)
    factor = value * (1.0 - value)
    return Dual(complex(value, 0.0), tuple(complex(factor * entry.real, 0.0) for entry in arg.grad))


def dual_square_clamped(arg: Dual) -> Dual:
    raw = arg.value.real * arg.value.real
    if raw > 64.0:
        return dual_const(64.0, len(arg.grad))
    return Dual(
        complex(raw, 0.0),
        tuple(complex(2.0 * arg.value.real * entry.real, 0.0) for entry in arg.grad),
    )


def softmax_dual(logits: list[Dual], temperature: float) -> list[Dual]:
    offset = max(logit.value.real for logit in logits)
    scaled = [
        Dual(
            complex((logit.value.real - offset) / max(temperature, 1.0e-6), 0.0),
            tuple(entry / max(temperature, 1.0e-6) for entry in logit.grad),
        )
        for logit in logits
    ]
    exp_values = [dual_exp_real(logit) for logit in scaled]
    total = exp_values[0]
    for value in exp_values[1:]:
        total = dual_add(total, value)
    return [dual_div(value, total) for value in exp_values]


def select_dual(
    dual_params: list[Dual],
    offset: int | None,
    options: tuple[Dual, ...],
    temperature: float,
) -> Dual:
    if offset is None:
        raise ValueError("selector offset is missing")
    weights = softmax_dual(dual_params[offset : offset + len(options)], temperature)
    total = dual_const(0.0, len(dual_params))
    for weight, option in zip(weights, options):
        total = dual_add(total, dual_mul(weight, option))
    return total


def paper_node_dual(node: TreeNode, dual_params: list[Dual], x_value: float, temperature: float) -> Dual:
    size = len(dual_params)
    if node.depth == 0:
        return select_dual(
            dual_params,
            node.terminal_selector,
            (dual_const(1.0, size), dual_const(x_value, size)),
            temperature,
        )
    assert node.left is not None and node.right is not None
    left = paper_node_dual(node.left, dual_params, x_value, temperature)
    right = paper_node_dual(node.right, dual_params, x_value, temperature)
    options = (dual_const(1.0, size), dual_const(x_value, size), left, right)
    left_arg = select_dual(dual_params, node.left_selector, options, temperature)
    right_arg = select_dual(dual_params, node.right_selector, options, temperature)
    return dual_sub(dual_exp_complex(left_arg), dual_log_complex(right_arg))


def stable_node_dual(node: TreeNode, dual_params: list[Dual], x_value: float, temperature: float) -> Dual:
    size = len(dual_params)
    if node.depth == 0:
        return select_dual(
            dual_params,
            node.terminal_selector,
            (
                dual_const(1.0, size),
                dual_const(x_value, size),
                dual_const(-1.0, size),
                dual_const(0.0, size),
            ),
            temperature,
        )
    assert node.left is not None and node.right is not None and node.op_param_offset is not None
    left = stable_node_dual(node.left, dual_params, x_value, temperature)
    right = stable_node_dual(node.right, dual_params, x_value, temperature)
    options = (
        dual_const(1.0, size),
        dual_const(x_value, size),
        dual_const(-1.0, size),
        dual_const(0.0, size),
        left,
        right,
    )
    state = select_dual(dual_params, node.left_selector, options, temperature)
    inputs = select_dual(dual_params, node.right_selector, options, temperature)
    gate = dual_sigmoid(dual_params[node.op_param_offset])
    state_weight = dual_params[node.op_param_offset + 1]
    input_weight = dual_params[node.op_param_offset + 2]
    square_weight = dual_params[node.op_param_offset + 3]
    bias = dual_params[node.op_param_offset + 4]
    update = dual_add(
        dual_add(dual_mul(state_weight, state), dual_mul(input_weight, inputs)),
        dual_add(dual_mul(square_weight, dual_square_clamped(state)), bias),
    )
    return dual_add(dual_mul(gate, update), dual_mul(dual_sub(dual_const(1.0, size), gate), state))


def generic_node_dual(node: TreeNode, dual_params: list[Dual], x_value: float, temperature: float) -> Dual:
    size = len(dual_params)
    if node.depth == 0:
        return select_dual(
            dual_params,
            node.terminal_selector,
            (
                dual_const(1.0, size),
                dual_const(x_value, size),
                dual_const(-1.0, size),
                dual_const(0.0, size),
                dual_const(0.5, size),
                dual_const(2.0, size),
            ),
            temperature,
        )
    assert node.left is not None and node.right is not None and node.op_selector is not None
    left = generic_node_dual(node.left, dual_params, x_value, temperature)
    right = generic_node_dual(node.right, dual_params, x_value, temperature)
    options = (
        dual_const(1.0, size),
        dual_const(x_value, size),
        dual_const(-1.0, size),
        dual_const(0.0, size),
        dual_const(0.5, size),
        dual_const(2.0, size),
        left,
        right,
    )
    left_arg = select_dual(dual_params, node.left_selector, options, temperature)
    right_arg = select_dual(dual_params, node.right_selector, options, temperature)
    op_values = (
        dual_add(left_arg, right_arg),
        dual_sub(left_arg, right_arg),
        dual_mul(left_arg, right_arg),
        dual_div(left_arg, right_arg),
        dual_exp_real(left_arg),
        dual_log_abs(left_arg),
        left_arg,
        right_arg,
    )
    return select_dual(dual_params, node.op_selector, op_values, temperature)


def entropy_penalty(model: object, dual_params: list[Dual], temperature: float) -> Dual:
    total = dual_const(0.0, len(dual_params))
    for offset, count in selector_specs(model):
        weights = softmax_dual(dual_params[offset : offset + count], temperature)
        for weight in weights:
            value = max(weight.value.real, 1.0e-12)
            log_weight = math.log(value)
            grad = tuple(entry * (log_weight + 1.0) for entry in weight.grad)
            total = dual_sub(total, Dual(complex(value * log_weight, 0.0), grad))
    return total


def selector_specs(model: object) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    if isinstance(model, PaperComplexEmlTree):
        collect_specs(model.root, specs, leaf_count=2, internal_count=4, op_count=0)
    elif isinstance(model, StableRealEmlTree):
        collect_specs(model.root, specs, leaf_count=4, internal_count=6, op_count=0)
    elif isinstance(model, GenericTreeControl):
        collect_specs(model.root, specs, leaf_count=6, internal_count=8, op_count=len(model.ops))
    else:
        raise TypeError(f"unsupported selector spec model: {type(model).__name__}")
    return specs


def collect_specs(
    node: TreeNode,
    specs: list[tuple[int, int]],
    *,
    leaf_count: int,
    internal_count: int,
    op_count: int,
) -> None:
    if node.depth == 0:
        assert node.terminal_selector is not None
        specs.append((node.terminal_selector, leaf_count))
        return
    assert node.left is not None and node.right is not None
    collect_specs(node.left, specs, leaf_count=leaf_count, internal_count=internal_count, op_count=op_count)
    collect_specs(node.right, specs, leaf_count=leaf_count, internal_count=internal_count, op_count=op_count)
    assert node.left_selector is not None and node.right_selector is not None
    specs.append((node.left_selector, internal_count))
    specs.append((node.right_selector, internal_count))
    if op_count > 0:
        assert node.op_selector is not None
        specs.append((node.op_selector, op_count))


def sharpen_selectors(model: object, strength: float = 1.5) -> None:
    params = model.parameters()
    for offset, count in selector_specs(model):
        values = params[offset : offset + count]
        winner = argmax(values)
        for index in range(count):
            params[offset + index] = values[index] - strength
        params[offset + winner] = values[winner] + strength
    model.set_parameters(params)


def snap_readout(model: object) -> None:
    if not hasattr(model, "readout_offset"):
        return
    params = model.parameters()
    offset = model.readout_offset
    params[offset] = snap_scalar(params[offset])
    params[offset + 1] = snap_scalar(params[offset + 1])
    model.set_parameters(params)
