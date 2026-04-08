from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python.specs.common import repo_relative, to_jsonable


@dataclass(frozen=True)
class EvalSummary:
    batch_count: int
    mean_loss: float
    perplexity: float


@dataclass(frozen=True)
class TrainStepRecord:
    step: int
    learning_rate: float
    train_loss: float
    train_perplexity: float
    seen_tokens: int


@dataclass(frozen=True)
class RuntimeSummary:
    total_wall_time_ms: float
    initial_eval_wall_time_ms: float
    train_wall_time_ms: float
    final_eval_wall_time_ms: float
    train_tokens_seen: int
    eval_tokens_per_pass: int
    train_tokens_per_second: float
    overall_tokens_per_second: float
    process_memory_metric: str
    peak_process_memory_bytes: int
    peak_process_memory_delta_bytes: int
    cuda_device_memory: dict[str, Any] | None
    memory_note: str


@dataclass
class BenchmarkReport:
    model_label: str
    implementation_kind: str
    note: str
    config: dict[str, Any]
    corpus: dict[str, Any]
    initial_eval: EvalSummary
    final_eval: EvalSummary
    runtime: RuntimeSummary
    train_steps: list[TrainStepRecord]
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        return payload


def write_report(report: BenchmarkReport, report_path: Path) -> None:
    report.report_path = repo_relative(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_ledger_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(entry), sort_keys=True) + "\n")

