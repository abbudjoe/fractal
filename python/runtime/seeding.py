from __future__ import annotations

import random

import torch

from python.specs.common import DeviceRuntimeSpec, SeedSpec


def configure_reproducibility(seed_spec: SeedSpec, runtime_spec: DeviceRuntimeSpec) -> None:
    seed_spec.validate()
    runtime_spec.validate()
    random.seed(seed_spec.model_seed)
    torch.manual_seed(seed_spec.model_seed)
    if runtime_spec.backend == "cuda":
        torch.cuda.manual_seed_all(seed_spec.model_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_torch_device(runtime_spec: DeviceRuntimeSpec) -> torch.device:
    runtime_spec.validate()
    if runtime_spec.backend == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{runtime_spec.cuda_device}")


def resolve_autocast_dtype(runtime_spec: DeviceRuntimeSpec) -> torch.dtype | None:
    runtime_spec.validate()
    if runtime_spec.backend != "cuda":
        return None
    return torch.bfloat16 if runtime_spec.dtype == "bf16" else None

