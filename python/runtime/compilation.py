from __future__ import annotations

import torch.nn as nn

from python.specs.common import DeviceRuntimeSpec


def apply_runtime_policy(model: nn.Module, runtime_spec: DeviceRuntimeSpec) -> nn.Module:
    runtime_spec.validate()
    compile_mode = runtime_spec.compile_mode
    primitive_runtime_backend = runtime_spec.primitive_runtime_backend
    if compile_mode is None and primitive_runtime_backend in {None, "torch"}:
        return model
    if compile_mode is None:
        compile_mode = None
    configure = getattr(model, "configure_runtime_policy", None)
    if callable(configure):
        configure(
            compile_mode=compile_mode,
            primitive_runtime_backend=primitive_runtime_backend,
        )
    return model
