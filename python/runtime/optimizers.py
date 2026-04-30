from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from python.specs.common import BenchmarkBudgetSpec, ValidationError

try:  # pragma: no cover - availability depends on CUDA runtime image.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - local CPU/MPS envs do not need Triton.
    triton = None
    tl = None


_EMBEDDING_OR_HEAD_MARKERS = (
    "embedding.",
    "position_embedding.",
    "attention_position_embeddings.",
    "context_embedding.embedding",
    "output.weight",
)

_MUON_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_DEFAULT_EPS = 1.0e-7
_TRITON_ADAM_MIN_ELEMENTS = 4096


if triton is not None and tl is not None:  # pragma: no branch

    @triton.jit
    def _adam_2d_update_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2_sqrt,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        grad += weight_decay * param
        exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
        exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
        step_size = lr * bias_correction2_sqrt / bias_correction1
        update = exp_avg / (tl.sqrt(exp_avg_sq) + eps)
        param = param - step_size * update

        tl.store(param_ptr + offsets, param, mask=mask)
        tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class CompositeOptimizer:
    """Small optimizer facade for heterogeneous parameter update laws."""

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]) -> None:
        self.optimizers = [optimizer for optimizer in optimizers]
        if not self.optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer")

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            result = optimizer.step(closure=closure)
            if result is not None:
                loss = result
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(self.optimizers):
            raise ValueError("CompositeOptimizer state_dict has incompatible optimizer state count")
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)


class ReferenceMuon(torch.optim.Optimizer):
    """Small local Muon reference for PyTorch builds that do not ship torch.optim.Muon."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _MUON_DEFAULT_NS_COEFFICIENTS,
        eps: float = _MUON_DEFAULT_EPS,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if momentum < 0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if ns_steps <= 0 or ns_steps >= 100:
            raise ValueError(f"ns_steps must be in [1, 99], got {ns_steps}")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must contain exactly three values")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError(f"unsupported adjust_lr_fn: {adjust_lr_fn}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError(
                        "ReferenceMuon only supports 2D parameters; non-2D tensors "
                        "must stay in the AdamW fallback optimizer"
                    )

    @staticmethod
    def _adjust_lr(lr: float, adjust_lr_fn: str | None, parameter: torch.Tensor) -> float:
        rows, cols = parameter.shape[:2]
        if adjust_lr_fn is None or adjust_lr_fn == "original":
            return lr * math.sqrt(max(1.0, rows / cols))
        if adjust_lr_fn == "match_rms_adamw":
            return lr * 0.2 * math.sqrt(max(rows, cols))
        return lr

    @staticmethod
    def _zeropower_via_newton_schulz(
        update: torch.Tensor,
        *,
        ns_coefficients: tuple[float, float, float],
        ns_steps: int,
        eps: float,
    ) -> torch.Tensor:
        if update.ndim != 2:
            raise ValueError("ReferenceMuon update must be a 2D matrix")
        a, b, c = ns_coefficients
        work = update.bfloat16() if update.device.type == "cuda" else update.float()
        transposed = work.shape[0] > work.shape[1]
        if transposed:
            work = work.T
        work = work / work.norm().clamp(min=eps)
        for _ in range(ns_steps):
            gram = work @ work.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            work = torch.addmm(work, gram_update, work, beta=a)
        if transposed:
            work = work.T
        return work.to(dtype=update.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_coefficients = group["ns_coefficients"]
            ns_steps = int(group["ns_steps"])
            eps = float(group["eps"])
            adjust_lr_fn = group["adjust_lr_fn"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if torch.is_complex(parameter):
                    raise RuntimeError("ReferenceMuon does not support complex parameters")
                if parameter.grad.is_sparse:
                    raise RuntimeError("ReferenceMuon does not support sparse gradients")
                grad = parameter.grad
                if grad.ndim != 2:
                    raise ValueError("ReferenceMuon gradients must be 2D matrices")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad,
                        memory_format=torch.preserve_format,
                    )
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buffer, momentum) if nesterov else buffer
                update = self._zeropower_via_newton_schulz(
                    update,
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                )
                adjusted_lr = self._adjust_lr(lr, adjust_lr_fn, parameter)
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


class TritonAdam2D(torch.optim.Optimizer):
    """Prototype Triton Adam for large contiguous 2D hidden matrices."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        block_size: int = 256,
    ) -> None:
        if triton is None or tl is None:
            raise RuntimeError("TritonAdam2D requires a runtime with triton installed")
        if lr < 0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if eps < 0:
            raise ValueError(f"eps should be >= 0 but is: {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "block_size": block_size,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("TritonAdam2D only supports 2D parameters")
                if parameter.device.type != "cuda":
                    raise ValueError("TritonAdam2D only supports CUDA parameters")
                if not parameter.is_contiguous():
                    raise ValueError("TritonAdam2D only supports contiguous parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            beta1 = float(beta1)
            beta2 = float(beta2)
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            block_size = int(group["block_size"])
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("TritonAdam2D does not support sparse gradients")
                if grad.ndim != 2 or not grad.is_contiguous():
                    raise RuntimeError("TritonAdam2D requires contiguous 2D gradients")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
                state["step"] += 1
                step = int(state["step"])
                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                n_elements = parameter.numel()
                grid = (triton.cdiv(n_elements, block_size),)
                _adam_2d_update_kernel[grid](
                    parameter,
                    grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    n_elements,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    bias_correction1=bias_correction1,
                    bias_correction2_sqrt=bias_correction2_sqrt,
                    BLOCK_SIZE=block_size,
                )
        return loss


@dataclass(frozen=True)
class MuonParameterSplit:
    muon_parameter_count: int
    fallback_parameter_count: int
    muon_tensor_count: int
    fallback_tensor_count: int


@dataclass(frozen=True)
class NativeAdamParameterSplit:
    native_parameter_count: int
    fallback_parameter_count: int
    native_tensor_count: int
    fallback_tensor_count: int


def _eligible_for_muon(name: str, parameter: torch.nn.Parameter) -> bool:
    if parameter.ndim != 2:
        return False
    return not any(marker in name for marker in _EMBEDDING_OR_HEAD_MARKERS)


def _eligible_for_triton_adam_2d(name: str, parameter: torch.nn.Parameter) -> bool:
    if not _eligible_for_muon(name, parameter):
        return False
    return (
        parameter.device.type == "cuda"
        and parameter.is_contiguous()
        and parameter.numel() >= _TRITON_ADAM_MIN_ELEMENTS
    )


def _group_lrs_by_parameter(model: torch.nn.Module, base_lr: float) -> dict[int, tuple[float, str]]:
    parameter_groups = getattr(model, "optimizer_parameter_groups", None)
    groups = (
        parameter_groups(base_lr)
        if callable(parameter_groups)
        else [{"name": "default", "params": list(model.parameters()), "lr": base_lr}]
    )
    by_parameter: dict[int, tuple[float, str]] = {}
    for group_index, group in enumerate(groups):
        group_name = str(group.get("name", f"group_{group_index}"))
        lr = float(group.get("lr", base_lr))
        for parameter in group.get("params", []):
            by_parameter[id(parameter)] = (lr, group_name)
    return by_parameter


def _append_group(
    groups_by_key: dict[tuple[float, str], dict[str, Any]],
    *,
    lr: float,
    group_name: str,
    parameter: torch.nn.Parameter,
) -> None:
    key = (lr, group_name)
    group = groups_by_key.setdefault(
        key,
        {"name": group_name, "lr": lr, "params": []},
    )
    group["params"].append(parameter)


def split_muon_parameters(model: torch.nn.Module, base_lr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], MuonParameterSplit]:
    group_lrs = _group_lrs_by_parameter(model, base_lr)
    muon_groups_by_key: dict[tuple[float, str], dict[str, Any]] = {}
    fallback_groups_by_key: dict[tuple[float, str], dict[str, Any]] = {}
    split = {
        "muon_parameter_count": 0,
        "fallback_parameter_count": 0,
        "muon_tensor_count": 0,
        "fallback_tensor_count": 0,
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr, source_group = group_lrs.get(id(parameter), (base_lr, "default"))
        if _eligible_for_muon(name, parameter):
            _append_group(
                muon_groups_by_key,
                lr=lr,
                group_name=f"muon:{source_group}",
                parameter=parameter,
            )
            split["muon_parameter_count"] += parameter.numel()
            split["muon_tensor_count"] += 1
        else:
            _append_group(
                fallback_groups_by_key,
                lr=lr,
                group_name=f"adamw_fallback:{source_group}",
                parameter=parameter,
            )
            split["fallback_parameter_count"] += parameter.numel()
            split["fallback_tensor_count"] += 1

    return (
        list(muon_groups_by_key.values()),
        list(fallback_groups_by_key.values()),
        MuonParameterSplit(**split),
    )


def split_triton_adam_2d_parameters(
    model: torch.nn.Module,
    base_lr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], NativeAdamParameterSplit]:
    group_lrs = _group_lrs_by_parameter(model, base_lr)
    native_groups_by_key: dict[tuple[float, str], dict[str, Any]] = {}
    fallback_groups_by_key: dict[tuple[float, str], dict[str, Any]] = {}
    split = {
        "native_parameter_count": 0,
        "fallback_parameter_count": 0,
        "native_tensor_count": 0,
        "fallback_tensor_count": 0,
    }

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr, source_group = group_lrs.get(id(parameter), (base_lr, "default"))
        if _eligible_for_triton_adam_2d(name, parameter):
            _append_group(
                native_groups_by_key,
                lr=lr,
                group_name=f"triton_adam_2d:{source_group}",
                parameter=parameter,
            )
            split["native_parameter_count"] += parameter.numel()
            split["native_tensor_count"] += 1
        else:
            _append_group(
                fallback_groups_by_key,
                lr=lr,
                group_name=f"adam_fused_fallback:{source_group}",
                parameter=parameter,
            )
            split["fallback_parameter_count"] += parameter.numel()
            split["fallback_tensor_count"] += 1

    return (
        list(native_groups_by_key.values()),
        list(fallback_groups_by_key.values()),
        NativeAdamParameterSplit(**split),
    )


def _first_trainable_parameter_device(model: torch.nn.Module) -> torch.device | None:
    for parameter in model.parameters():
        if parameter.requires_grad:
            return parameter.device
    return None


def build_optimizer(
    model: torch.nn.Module,
    budget: BenchmarkBudgetSpec,
    *,
    capturable: bool = False,
) -> torch.optim.Optimizer | CompositeOptimizer:
    budget.validate()
    base_lr = budget.learning_rate
    parameter_groups = getattr(model, "optimizer_parameter_groups", None)
    if budget.optimizer_profile == "adam":
        params = (
            parameter_groups(base_lr)
            if callable(parameter_groups)
            else model.parameters()
        )
        kwargs: dict[str, Any] = {}
        if capturable:
            kwargs["capturable"] = True
        return torch.optim.Adam(params, lr=base_lr, **kwargs)

    if budget.optimizer_profile == "adam-fused":
        device = _first_trainable_parameter_device(model)
        if device is None:
            raise ValidationError("optimizer_profile=adam-fused found no trainable parameters")
        if device.type != "cuda":
            raise ValidationError("optimizer_profile=adam-fused requires CUDA parameters")
        params = (
            parameter_groups(base_lr)
            if callable(parameter_groups)
            else model.parameters()
        )
        kwargs = {"fused": True}
        if capturable:
            kwargs["capturable"] = True
        return torch.optim.Adam(params, lr=base_lr, **kwargs)

    if budget.optimizer_profile == "adam-triton-2d":
        device = _first_trainable_parameter_device(model)
        if device is None:
            raise ValidationError("optimizer_profile=adam-triton-2d found no trainable parameters")
        if device.type != "cuda":
            raise ValidationError("optimizer_profile=adam-triton-2d requires CUDA parameters")
        if triton is None or tl is None:
            raise ValidationError("optimizer_profile=adam-triton-2d requires triton")
        native_groups, fallback_groups, split = split_triton_adam_2d_parameters(model, base_lr)
        if not native_groups:
            raise ValidationError("optimizer_profile=adam-triton-2d found no eligible large 2D hidden parameters")
        optimizers: list[torch.optim.Optimizer] = [
            TritonAdam2D(native_groups, lr=base_lr, weight_decay=0.0),
        ]
        if fallback_groups:
            fallback_kwargs: dict[str, Any] = {"fused": True}
            if capturable:
                fallback_kwargs["capturable"] = True
            optimizers.append(torch.optim.Adam(fallback_groups, lr=base_lr, **fallback_kwargs))
        optimizer = CompositeOptimizer(optimizers)
        setattr(optimizer, "native_adam_parameter_split", split)
        setattr(optimizer, "native_optimizer_kind", "TritonAdam2D")
        return optimizer

    if budget.optimizer_profile != "muon-reference":
        raise ValidationError(f"unsupported optimizer_profile: {budget.optimizer_profile}")
    if capturable:
        raise ValidationError("optimizer_profile=muon-reference is not supported with cuda_graph_step")

    muon_groups, fallback_groups, split = split_muon_parameters(model, base_lr)
    if not muon_groups:
        raise ValidationError("optimizer_profile=muon-reference found no eligible 2D hidden parameters")
    muon_cls = getattr(torch.optim, "Muon", ReferenceMuon)

    optimizers: list[torch.optim.Optimizer] = [
        muon_cls(
            muon_groups,
            lr=base_lr,
            weight_decay=budget.muon_weight_decay,
            momentum=budget.muon_momentum,
            ns_steps=budget.muon_ns_steps,
            adjust_lr_fn=budget.muon_adjust_lr_fn,
        )
    ]
    if fallback_groups:
        optimizers.append(torch.optim.AdamW(fallback_groups, lr=base_lr, weight_decay=0.0))
    optimizer = CompositeOptimizer(optimizers)
    setattr(optimizer, "muon_parameter_split", split)
    setattr(optimizer, "muon_optimizer_kind", "torch.optim.Muon" if hasattr(torch.optim, "Muon") else "ReferenceMuon")
    return optimizer
