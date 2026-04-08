from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a typed spec violates its explicit contract."""


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


def ensure_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValidationError(f"{name} must be greater than zero, got {value}")


def ensure_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class SeedSpec:
    model_seed: int
    data_seed: int | None = None

    def validate(self) -> None:
        ensure_non_negative(self.model_seed, "seed_spec.model_seed")
        if self.data_seed is not None:
            ensure_non_negative(self.data_seed, "seed_spec.data_seed")


@dataclass(frozen=True)
class JsonlCorpusSpec:
    train_path: Path
    eval_path: Path
    corpus_name: str
    text_field: str = "text"

    def validate(self) -> None:
        if not self.corpus_name.strip():
            raise ValidationError("corpus.corpus_name must not be empty")
        if not self.text_field.strip():
            raise ValidationError("corpus.text_field must not be empty")
        if not self.train_path.exists():
            raise ValidationError(f"corpus.train_path does not exist: {self.train_path}")
        if not self.eval_path.exists():
            raise ValidationError(f"corpus.eval_path does not exist: {self.eval_path}")


@dataclass(frozen=True)
class BenchmarkBudgetSpec:
    seq_len: int = 32
    window_stride: int = 32
    batch_size: int = 1
    train_steps: int = 8
    eval_batches: int = 2
    full_train_pass: bool = False
    full_eval_pass: bool = False
    learning_rate: float = 1.0e-3
    warmup_eval_batches: int = 1
    warmup_train_steps: int = 1

    def validate(self) -> None:
        ensure_positive(self.seq_len, "budget.seq_len")
        ensure_positive(self.window_stride, "budget.window_stride")
        ensure_positive(self.batch_size, "budget.batch_size")
        ensure_positive(self.train_steps, "budget.train_steps")
        ensure_positive(self.eval_batches, "budget.eval_batches")
        ensure_non_negative(self.warmup_eval_batches, "budget.warmup_eval_batches")
        ensure_non_negative(self.warmup_train_steps, "budget.warmup_train_steps")
        if self.learning_rate <= 0:
            raise ValidationError(
                f"budget.learning_rate must be greater than zero, got {self.learning_rate}"
            )


@dataclass(frozen=True)
class DeviceRuntimeSpec:
    backend: str = "cuda"
    cuda_device: int = 0
    dtype: str = "bf16"
    env_kind: str | None = None
    compile_mode: str | None = None

    def validate(self) -> None:
        if self.backend not in {"cpu", "cuda"}:
            raise ValidationError(
                f"runtime.backend must be one of cpu|cuda, got {self.backend}"
            )
        ensure_non_negative(self.cuda_device, "runtime.cuda_device")
        if self.dtype not in {"fp32", "bf16"}:
            raise ValidationError(
                f"runtime.dtype must be one of fp32|bf16, got {self.dtype}"
            )
        if self.backend == "cpu" and self.dtype == "bf16":
            raise ValidationError("runtime.dtype=bf16 is only supported for backend=cuda")
        if self.env_kind not in {None, "requirements-only", "official-mamba3", "compile-safe"}:
            raise ValidationError(
                "runtime.env_kind must be one of requirements-only|official-mamba3|compile-safe or omitted"
            )
        if self.compile_mode not in {None, "default", "reduce-overhead", "max-autotune"}:
            raise ValidationError(
                "runtime.compile_mode must be one of default|reduce-overhead|max-autotune or omitted"
            )


@dataclass(frozen=True)
class BenchmarkRunManifest:
    run_label: str
    implementation_kind: str
    seed_spec: SeedSpec
    corpus: JsonlCorpusSpec
    budget: BenchmarkBudgetSpec
    runtime: DeviceRuntimeSpec
    benchmark_name: str | None = None
    note: str = ""

    def validate(self) -> None:
        if not self.run_label.strip():
            raise ValidationError("manifest.run_label must not be empty")
        if not self.implementation_kind.strip():
            raise ValidationError("manifest.implementation_kind must not be empty")
        self.seed_spec.validate()
        self.corpus.validate()
        self.budget.validate()
        self.runtime.validate()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {field_name: to_jsonable(getattr(value, field_name)) for field_name in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
