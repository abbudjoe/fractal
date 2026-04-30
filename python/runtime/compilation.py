from __future__ import annotations

import inspect

import torch.nn as nn

from python.specs.common import DeviceRuntimeSpec, ValidationError


def apply_runtime_policy(model: nn.Module, runtime_spec: DeviceRuntimeSpec) -> nn.Module:
    runtime_spec.validate()
    compile_mode = runtime_spec.compile_mode
    primitive_runtime_backend = runtime_spec.primitive_runtime_backend
    if (
        compile_mode is None
        and primitive_runtime_backend in {None, "torch"}
        and runtime_spec.head_loss_backend == "dense"
        and runtime_spec.ffn_backend == "dense"
    ):
        return model
    if compile_mode is None:
        compile_mode = None
    configure = getattr(model, "configure_runtime_policy", None)
    if not callable(configure):
        if runtime_spec.head_loss_backend != "dense" or runtime_spec.ffn_backend != "dense":
            raise ValidationError(
                "non-default head_loss_backend or ffn_backend requires model.configure_runtime_policy"
            )
        return model
    available = inspect.signature(configure).parameters
    if runtime_spec.head_loss_backend != "dense" and "head_loss_backend" not in available:
        raise ValidationError("requested head_loss_backend requires configure_runtime_policy(head_loss_backend=...)")
    if runtime_spec.ffn_backend != "dense" and "ffn_backend" not in available:
        raise ValidationError("requested ffn_backend requires configure_runtime_policy(ffn_backend=...)")
    kwargs: dict[str, object] = {
        "compile_mode": compile_mode,
        "primitive_runtime_backend": primitive_runtime_backend,
    }
    if "head_loss_backend" in available:
        kwargs["head_loss_backend"] = runtime_spec.head_loss_backend
    if "ffn_backend" in available:
        kwargs["ffn_backend"] = runtime_spec.ffn_backend
    configure(**kwargs)
    return model
