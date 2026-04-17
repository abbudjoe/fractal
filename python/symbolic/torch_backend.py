from __future__ import annotations

import math
from typing import Any

from python.specs.symbolic import SymbolicTrainSpec
from python.symbolic.autodiff import selector_specs
from python.symbolic.formulas import FormulaSplit
from python.symbolic.models import (
    EXP_REAL_CLAMP,
    HardenedExpression,
    LOG_EPSILON,
    MAX_REAL_MAGNITUDE,
    GenericTreeControl,
    PaperComplexEmlTree,
    StableRealEmlTree,
    SymbolicModel,
    TreeNode,
)


def train_tree_torch(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    *,
    y_scale: float,
    backend: str,
) -> float:
    torch = import_torch()
    device = resolve_torch_device(torch, backend)
    starts = paper_restart_parameters(model, train_spec.paper_restarts) if isinstance(model, PaperComplexEmlTree) else [model.parameters()]
    best_loss = math.inf
    best_score = math.inf
    best_params = list(model.parameters())
    for start_params in starts:
        loss, trained = train_tree_torch_from_parameters(
            model,
            split,
            train_spec,
            y_scale=y_scale,
            device=device,
            torch=torch,
            start_params=start_params,
        )
        model.set_parameters(trained)
        score = paper_hardened_refit_rmse(model, split) if isinstance(model, PaperComplexEmlTree) else loss
        if score < best_score or (abs(score - best_score) <= 1.0e-12 and loss < best_loss):
            best_score = score
            best_loss = loss
            best_params = trained
    model.set_parameters(best_params)
    return best_loss


def train_tree_torch_from_parameters(
    model: SymbolicModel,
    split: FormulaSplit,
    train_spec: SymbolicTrainSpec,
    *,
    y_scale: float,
    device: Any,
    torch: Any,
    start_params: list[float],
) -> tuple[float, list[float]]:
    dtype = torch.float32
    params = torch.nn.Parameter(torch.tensor(start_params, dtype=dtype, device=device))
    xs = torch.tensor(split.xs, dtype=dtype, device=device)
    ys = torch.tensor(split.ys, dtype=dtype, device=device)
    optimizer = torch.optim.Adam([params], lr=train_spec.tree_learning_rate)
    scale = max(y_scale, 1.0e-6)
    best_loss = math.inf
    best_params = list(start_params)
    for step in range(train_spec.steps):
        temperature = annealed_temperature(train_spec, step)
        optimizer.zero_grad(set_to_none=True)
        predictions = predict_torch(model, params, xs, temperature, torch)
        predictions = torch.nan_to_num(
            predictions,
            nan=MAX_REAL_MAGNITUDE,
            posinf=MAX_REAL_MAGNITUDE,
            neginf=-MAX_REAL_MAGNITUDE,
        )
        mse = torch.mean(((predictions - ys) / scale) ** 2)
        loss = mse + train_spec.snap_penalty_weight * entropy_penalty_torch(model, params, temperature, torch)
        if not bool(torch.isfinite(loss).detach().cpu().item()):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([params], max_norm=25.0)
        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = tensor_to_list(params.detach(), torch)
        optimizer.step()
    return best_loss, best_params


def paper_restart_parameters(model: PaperComplexEmlTree, restart_count: int) -> list[list[float]]:
    starts = [model.parameters()]
    if model.root.depth == 0:
        return starts
    templates = [
        (1, 0),  # eml(x, 1) = exp(x)
        (0, 1),  # eml(1, x) = e - log(x)
        (1, 1),  # eml(x, x)
        (0, 0),  # constant eml(1, 1)
        (2, 3),  # composed left/right subtrees
        (3, 2),
    ]
    for index, (left_choice, right_choice) in enumerate(templates):
        if len(starts) >= restart_count:
            break
        params = model.parameters()
        bias_selector(params, model.root.left_selector, 4, left_choice, margin=4.0 + 0.25 * index)
        bias_selector(params, model.root.right_selector, 4, right_choice, margin=4.0 + 0.25 * index)
        bias_leaf_subtree(params, model.root.left, prefer_x=left_choice == 1)
        bias_leaf_subtree(params, model.root.right, prefer_x=right_choice == 1)
        starts.append(params)
    return starts[:restart_count]


def paper_hardened_refit_rmse(model: PaperComplexEmlTree, split: FormulaSplit) -> float:
    root_expr, root_source, _complexity, ops = model.harden_root()
    root_expression = HardenedExpression(
        model_family=model.family,
        expression=f"real({root_expr})",
        python_source=f"lambda x: float(({root_source}).real)",
        complexity=1,
        active_ops=tuple(ops),
        symbolic_export=True,
    )
    try:
        root_func = root_expression.compile()
        values = tuple(float(root_func(value)) for value in split.xs)
    except Exception:
        return math.inf
    if not values or not all(math.isfinite(value) for value in values):
        return math.inf
    scale, bias = least_squares_affine(values, split.ys)
    total = 0.0
    for value, target in zip(values, split.ys):
        error = scale * value + bias - target
        total += error * error
    return math.sqrt(total / max(1, len(values)))


def least_squares_affine(xs: tuple[float, ...], ys: tuple[float, ...]) -> tuple[float, float]:
    x_mean = sum(xs) / max(1, len(xs))
    y_mean = sum(ys) / max(1, len(ys))
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(xs, ys))
    denominator = sum((x_value - x_mean) ** 2 for x_value in xs)
    if denominator <= 1.0e-12:
        return 0.0, y_mean
    scale = numerator / denominator
    bias = y_mean - scale * x_mean
    return scale, bias


def bias_selector(
    params: list[float],
    offset: int | None,
    count: int,
    choice: int,
    *,
    margin: float,
) -> None:
    if offset is None:
        return
    for index in range(count):
        params[offset + index] = -margin
    params[offset + choice] = margin


def bias_leaf_subtree(params: list[float], node: TreeNode | None, *, prefer_x: bool) -> None:
    if node is None:
        return
    if node.depth == 0:
        bias_selector(params, node.terminal_selector, 2, 1 if prefer_x else 0, margin=2.0)
        return
    bias_leaf_subtree(params, node.left, prefer_x=prefer_x)
    bias_leaf_subtree(params, node.right, prefer_x=prefer_x)


def annealed_temperature(train_spec: SymbolicTrainSpec, step: int) -> float:
    if train_spec.steps <= 1:
        return train_spec.final_temperature
    fraction = step / float(train_spec.steps - 1)
    return train_spec.initial_temperature * (
        train_spec.final_temperature / train_spec.initial_temperature
    ) ** fraction


def import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "tree_optimizer=torch-autodiff requires PyTorch. Use "
            "`uv run --python 3.12 --with torch python scripts/symbolic_benchmark.py ...`."
        ) from error
    return torch


def resolve_torch_device(torch: Any, backend: str) -> Any:
    if backend == "cpu":
        return torch.device("cpu")
    if backend in {"mps", "auto"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if backend == "mps":
        raise RuntimeError("backend=mps requested but torch.backends.mps is not available")
    return torch.device("cpu")


def tensor_to_list(tensor: Any, torch: Any) -> list[float]:
    return [float(value) for value in tensor.detach().to("cpu").tolist()]


def predict_torch(model: SymbolicModel, params: Any, xs: Any, temperature: float, torch: Any) -> Any:
    if isinstance(model, PaperComplexEmlTree):
        real, _imag = paper_node_torch(model.root, params, xs, temperature, torch)
        return params[model.readout_offset] * real + params[model.readout_offset + 1]
    if isinstance(model, StableRealEmlTree):
        root = stable_node_torch(model.root, params, xs, temperature, torch)
        return params[model.readout_offset] * root + params[model.readout_offset + 1]
    if isinstance(model, GenericTreeControl):
        root = generic_node_torch(model.root, params, xs, temperature, torch)
        return params[model.readout_offset] * root + params[model.readout_offset + 1]
    raise TypeError(f"unsupported torch symbolic model: {type(model).__name__}")


def select_torch(params: Any, offset: int | None, options: tuple[Any, ...], temperature: float, torch: Any) -> Any:
    if offset is None:
        raise ValueError("selector offset is missing")
    weights = torch.softmax(params[offset : offset + len(options)] / max(temperature, 1.0e-6), dim=0)
    total = options[0] * weights[0]
    for index in range(1, len(options)):
        total = total + options[index] * weights[index]
    return total


def select_complex_torch(
    params: Any,
    offset: int | None,
    options: tuple[tuple[Any, Any], ...],
    temperature: float,
    torch: Any,
) -> tuple[Any, Any]:
    if offset is None:
        raise ValueError("selector offset is missing")
    weights = torch.softmax(params[offset : offset + len(options)] / max(temperature, 1.0e-6), dim=0)
    real = options[0][0] * weights[0]
    imag = options[0][1] * weights[0]
    for index in range(1, len(options)):
        real = real + options[index][0] * weights[index]
        imag = imag + options[index][1] * weights[index]
    return real, imag


def paper_node_torch(node: TreeNode, params: Any, xs: Any, temperature: float, torch: Any) -> tuple[Any, Any]:
    zeros = torch.zeros_like(xs)
    ones = torch.ones_like(xs)
    if node.depth == 0:
        return select_complex_torch(
            params,
            node.terminal_selector,
            ((ones, zeros), (xs, zeros)),
            temperature,
            torch,
        )
    assert node.left is not None and node.right is not None
    left = paper_node_torch(node.left, params, xs, temperature, torch)
    right = paper_node_torch(node.right, params, xs, temperature, torch)
    options = ((ones, zeros), (xs, zeros), left, right)
    left_arg = select_complex_torch(params, node.left_selector, options, temperature, torch)
    right_arg = select_complex_torch(params, node.right_selector, options, temperature, torch)
    return eml_complex_torch(left_arg, right_arg, torch)


def eml_complex_torch(left: tuple[Any, Any], right: tuple[Any, Any], torch: Any) -> tuple[Any, Any]:
    left_real, left_imag = left
    right_real, right_imag = right
    exp_real = torch.exp(torch.clamp(left_real, min=-EXP_REAL_CLAMP, max=EXP_REAL_CLAMP))
    exp_out_real = exp_real * torch.cos(left_imag)
    exp_out_imag = exp_real * torch.sin(left_imag)
    radius = torch.sqrt(right_real * right_real + right_imag * right_imag + LOG_EPSILON)
    log_real = torch.log(torch.clamp(radius, min=LOG_EPSILON))
    log_imag = torch.atan2(right_imag, right_real + LOG_EPSILON)
    return (
        torch.clamp(exp_out_real - log_real, min=-MAX_REAL_MAGNITUDE, max=MAX_REAL_MAGNITUDE),
        torch.clamp(exp_out_imag - log_imag, min=-MAX_REAL_MAGNITUDE, max=MAX_REAL_MAGNITUDE),
    )


def stable_node_torch(node: TreeNode, params: Any, xs: Any, temperature: float, torch: Any) -> Any:
    zeros = torch.zeros_like(xs)
    ones = torch.ones_like(xs)
    if node.depth == 0:
        return select_torch(params, node.terminal_selector, (ones, xs, -ones, zeros), temperature, torch)
    assert node.left is not None and node.right is not None and node.op_param_offset is not None
    left = stable_node_torch(node.left, params, xs, temperature, torch)
    right = stable_node_torch(node.right, params, xs, temperature, torch)
    options = (ones, xs, -ones, zeros, left, right)
    state = select_torch(params, node.left_selector, options, temperature, torch)
    inputs = select_torch(params, node.right_selector, options, temperature, torch)
    gate = torch.sigmoid(params[node.op_param_offset])
    state_weight = params[node.op_param_offset + 1]
    input_weight = params[node.op_param_offset + 2]
    square_weight = params[node.op_param_offset + 3]
    bias = params[node.op_param_offset + 4]
    squared = torch.clamp(state * state, min=-64.0, max=64.0)
    update = state_weight * state + input_weight * inputs + square_weight * squared + bias
    return torch.clamp(gate * update + (1.0 - gate) * state, min=-MAX_REAL_MAGNITUDE, max=MAX_REAL_MAGNITUDE)


def generic_node_torch(node: TreeNode, params: Any, xs: Any, temperature: float, torch: Any) -> Any:
    zeros = torch.zeros_like(xs)
    ones = torch.ones_like(xs)
    if node.depth == 0:
        return select_torch(
            params,
            node.terminal_selector,
            (ones, xs, -ones, zeros, ones * 0.5, ones * 2.0),
            temperature,
            torch,
        )
    assert node.left is not None and node.right is not None and node.op_selector is not None
    left = generic_node_torch(node.left, params, xs, temperature, torch)
    right = generic_node_torch(node.right, params, xs, temperature, torch)
    options = (ones, xs, -ones, zeros, ones * 0.5, ones * 2.0, left, right)
    left_arg = select_torch(params, node.left_selector, options, temperature, torch)
    right_arg = select_torch(params, node.right_selector, options, temperature, torch)
    denominator = torch.where(
        torch.abs(right_arg) >= LOG_EPSILON,
        right_arg,
        torch.where(right_arg >= 0.0, ones * LOG_EPSILON, -ones * LOG_EPSILON),
    )
    op_values = (
        left_arg + right_arg,
        left_arg - right_arg,
        left_arg * right_arg,
        left_arg / denominator,
        torch.exp(torch.clamp(left_arg, min=-EXP_REAL_CLAMP, max=EXP_REAL_CLAMP)),
        torch.log(torch.abs(left_arg) + LOG_EPSILON),
        left_arg,
        right_arg,
    )
    return torch.clamp(select_torch(params, node.op_selector, op_values, temperature, torch), min=-MAX_REAL_MAGNITUDE, max=MAX_REAL_MAGNITUDE)


def entropy_penalty_torch(model: SymbolicModel, params: Any, temperature: float, torch: Any) -> Any:
    penalty = params[0] * 0.0
    for offset, count in selector_specs(model):
        weights = torch.softmax(params[offset : offset + count] / max(temperature, 1.0e-6), dim=0)
        penalty = penalty - torch.sum(weights * torch.log(weights + 1.0e-12))
    return penalty
