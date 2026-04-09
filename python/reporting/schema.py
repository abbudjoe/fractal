from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python.specs.mini_moe import MiniMoeDispatchMode
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


@dataclass(frozen=True)
class MiniMoeRoutingSummary:
    sampled_tokens: int
    layer_count: int
    round_count: int
    mean_route_entropy_bits: float
    mean_winner_margin: float
    mean_expert_weights: list[float]
    winner_counts: list[int]
    active_expert_count: int
    mean_round_adjustment_l1: list[float]


@dataclass(frozen=True)
class ExpertUsageSummary:
    expert_id: int
    selection_count: int
    mean_weight: float


@dataclass(frozen=True)
class MiniMoeLayerSummary:
    layer_index: int
    sampled_tokens: int
    route_entropy_bits: float
    reroute_fraction: float
    expert_usage: list[ExpertUsageSummary]


@dataclass(frozen=True)
class MiniMoeDispatchSummary:
    layer_index: int
    mode: MiniMoeDispatchMode
    selected_expert_counts: list[int]
    dropped_token_fraction: float | None = None


@dataclass(frozen=True)
class MiniMoeControllerRoundSummary:
    layer_index: int
    round_index: int
    mean_route_entropy_bits: float
    mean_winner_margin: float
    mean_route_adjustment_l1: float | None
    rerouted_token_fraction: float
    applied_token_fraction: float
    mean_gate_probability: float | None = None


@dataclass(frozen=True)
class MiniMoeTokenRoundTrace:
    round_index: int
    winner_expert_id: int
    winner_weight: float
    route_entropy_bits: float
    winner_margin: float
    route_adjustment_l1: float | None


@dataclass(frozen=True)
class MiniMoeTokenRouteTrace:
    layer_index: int
    forward_pass_index: int
    batch_index: int
    position_index: int
    token_id: int
    token_label: str
    rerouted: bool
    first_winner_expert_id: int
    final_winner_expert_id: int
    total_adjustment_l1: float
    rounds: list[MiniMoeTokenRoundTrace]


@dataclass(frozen=True)
class MiniMoeReportSummary:
    routing: MiniMoeRoutingSummary
    layers: list[MiniMoeLayerSummary]
    dispatch: list[MiniMoeDispatchSummary]
    controller_rounds: list[MiniMoeControllerRoundSummary]
    token_traces: list[MiniMoeTokenRouteTrace]


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
    mini_moe_summary: MiniMoeReportSummary | None = None
    report_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        return payload


def write_report(report: BenchmarkReport, report_path: Path) -> None:
    report.report_path = repo_relative(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_report(report_path: Path) -> BenchmarkReport:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    mini_moe_summary_payload = payload.get("mini_moe_summary")
    mini_moe_summary = None
    if mini_moe_summary_payload is not None:
        routing_payload = mini_moe_summary_payload["routing"]
        layers_payload = mini_moe_summary_payload["layers"]
        dispatch_payload = mini_moe_summary_payload["dispatch"]
        controller_rounds_payload = mini_moe_summary_payload["controller_rounds"]
        token_traces_payload = mini_moe_summary_payload.get("token_traces", [])
        mini_moe_summary = MiniMoeReportSummary(
            routing=MiniMoeRoutingSummary(**routing_payload),
            layers=[
                MiniMoeLayerSummary(
                    layer_index=layer["layer_index"],
                    sampled_tokens=layer["sampled_tokens"],
                    route_entropy_bits=layer["route_entropy_bits"],
                    reroute_fraction=layer["reroute_fraction"],
                    expert_usage=[
                        ExpertUsageSummary(**expert_usage)
                        for expert_usage in layer["expert_usage"]
                    ],
                )
                for layer in layers_payload
            ],
            dispatch=[
                MiniMoeDispatchSummary(
                    layer_index=dispatch["layer_index"],
                    mode=MiniMoeDispatchMode(dispatch["mode"]),
                    selected_expert_counts=dispatch["selected_expert_counts"],
                    dropped_token_fraction=dispatch.get("dropped_token_fraction"),
                )
                for dispatch in dispatch_payload
            ],
            controller_rounds=[
                MiniMoeControllerRoundSummary(**round_summary)
                for round_summary in controller_rounds_payload
            ],
            token_traces=[
                MiniMoeTokenRouteTrace(
                    layer_index=trace["layer_index"],
                    forward_pass_index=trace["forward_pass_index"],
                    batch_index=trace["batch_index"],
                    position_index=trace["position_index"],
                    token_id=trace["token_id"],
                    token_label=trace["token_label"],
                    rerouted=trace["rerouted"],
                    first_winner_expert_id=trace["first_winner_expert_id"],
                    final_winner_expert_id=trace["final_winner_expert_id"],
                    total_adjustment_l1=trace["total_adjustment_l1"],
                    rounds=[
                        MiniMoeTokenRoundTrace(**round_trace)
                        for round_trace in trace["rounds"]
                    ],
                )
                for trace in token_traces_payload
            ],
        )
    return BenchmarkReport(
        model_label=payload["model_label"],
        implementation_kind=payload["implementation_kind"],
        note=payload["note"],
        config=payload["config"],
        corpus=payload["corpus"],
        initial_eval=EvalSummary(**payload["initial_eval"]),
        final_eval=EvalSummary(**payload["final_eval"]),
        runtime=RuntimeSummary(**payload["runtime"]),
        train_steps=[TrainStepRecord(**step) for step in payload["train_steps"]],
        mini_moe_summary=mini_moe_summary,
        report_path=payload.get("report_path"),
    )


def append_ledger_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(entry), sort_keys=True) + "\n")
