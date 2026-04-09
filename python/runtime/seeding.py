from __future__ import annotations

import random

import torch

from python.specs.common import DeviceRuntimeSpec, SeedSpec, ValidationError


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
    if runtime_spec.backend == "mps":
        if not torch.backends.mps.is_built():
            raise ValidationError("runtime.backend=mps requires a PyTorch build with MPS support")
        if not torch.backends.mps.is_available():
            raise ValidationError("runtime.backend=mps requested, but MPS is not available on this machine")
        return torch.device("mps")
    if not torch.cuda.is_available():
        raise ValidationError("runtime.backend=cuda requested, but CUDA is not available")
    if runtime_spec.cuda_device >= torch.cuda.device_count():
        raise ValidationError(
            f"runtime.cuda_device {runtime_spec.cuda_device} is out of range for {torch.cuda.device_count()} visible CUDA devices"
        )
    return torch.device(f"cuda:{runtime_spec.cuda_device}")


def resolve_autocast_dtype(runtime_spec: DeviceRuntimeSpec) -> torch.dtype | None:
    runtime_spec.validate()
    if runtime_spec.backend != "cuda":
        return None
    return torch.bfloat16 if runtime_spec.dtype == "bf16" else None
