from __future__ import annotations

import torch.nn as nn

from python.specs.common import DeviceRuntimeSpec


def apply_runtime_policy(model: nn.Module, runtime_spec: DeviceRuntimeSpec) -> nn.Module:
    runtime_spec.validate()
    compile_mode = runtime_spec.compile_mode
    if compile_mode is None:
        return model
    configure = getattr(model, "configure_runtime_policy", None)
    if callable(configure):
        configure(compile_mode=compile_mode)
    return model
