from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from python.runners.mini_moe import MiniMoeRunnerRequest, run_mini_moe_variant
from python.specs.common import BenchmarkRunManifest
from python.specs.mini_moe import MiniMoeSurfaceSpec


@dataclass(frozen=True)
class MiniMoePolicySearchCandidate:
    name: str
    surface: MiniMoeSurfaceSpec
    note: str = ""


@dataclass(frozen=True)
class MiniMoePolicySearchRequest:
    benchmark_name: str
    manifest_template: BenchmarkRunManifest
    candidates: tuple[MiniMoePolicySearchCandidate, ...]
    seeds: tuple[int, ...]
    output_dir: Path
    output_format: str = "json"
    ledger_path: Path | None = None


@dataclass(frozen=True)
class MiniMoePolicySearchCandidateResult:
    candidate_name: str
    report_paths: tuple[str, ...]
    avg_final_loss: float
    avg_train_toks_per_s: float
    avg_overall_toks_per_s: float
    avg_peak_process_memory_delta_mb: float
    avg_overall_round2_fraction: float
    avg_mean_active_round2_fraction: float


@dataclass(frozen=True)
class MiniMoePolicySearchSummary:
    benchmark_name: str
    seeds: tuple[int, ...]
    candidate_results: tuple[MiniMoePolicySearchCandidateResult, ...]
    summary_path: str


def _candidate_payload(report: Any, surface: MiniMoeSurfaceSpec) -> tuple[float, float]:
    controller_rounds = report.mini_moe_summary.controller_rounds if report.mini_moe_summary else []
    round2_rows = [row for row in controller_rounds if row.round_index == 2]
    total_layers = surface.architecture.backbone.total_layers
    overall_round2_fraction = (
        sum(row.applied_token_fraction for row in round2_rows) / total_layers
        if total_layers > 0
        else 0.0
    )
    mean_active_round2_fraction = (
        sum(row.applied_token_fraction for row in round2_rows) / len(round2_rows)
        if round2_rows
        else 0.0
    )
    return overall_round2_fraction, mean_active_round2_fraction


def run_mini_moe_policy_search(
    request: MiniMoePolicySearchRequest,
) -> MiniMoePolicySearchSummary:
    request.manifest_template.validate()
    if not request.candidates:
        raise ValueError("mini_moe_policy_search requires at least one candidate")
    if not request.seeds:
        raise ValueError("mini_moe_policy_search requires at least one seed")

    request.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results: list[MiniMoePolicySearchCandidateResult] = []
    for candidate in request.candidates:
        candidate.surface.validate()
        report_paths: list[str] = []
        final_losses: list[float] = []
        train_toks_per_s: list[float] = []
        overall_toks_per_s: list[float] = []
        peak_process_memory_delta_mb: list[float] = []
        overall_round2_fraction: list[float] = []
        mean_active_round2_fraction: list[float] = []
        for seed in request.seeds:
            manifest = BenchmarkRunManifest(
                run_label=f"{request.manifest_template.run_label}-{candidate.name}-seed{seed}",
                implementation_kind=request.manifest_template.implementation_kind,
                seed_spec=type(request.manifest_template.seed_spec)(
                    model_seed=seed,
                    data_seed=request.manifest_template.seed_spec.data_seed,
                ),
                corpus=request.manifest_template.corpus,
                budget=request.manifest_template.budget,
                runtime=request.manifest_template.runtime,
                benchmark_name=request.benchmark_name,
                note=candidate.note or request.manifest_template.note,
            )
            runner_request = MiniMoeRunnerRequest(
                manifest=manifest,
                surface=candidate.surface,
                output_dir=request.output_dir,
                output_format=request.output_format,
                ledger_path=request.ledger_path,
                variant_output_name=f"{candidate.name}-seed{seed}",
                model_note=candidate.note or request.manifest_template.note,
            )
            report = run_mini_moe_variant(runner_request)
            report_paths.append(report.report_path or "")
            final_losses.append(report.final_eval.mean_loss)
            train_toks_per_s.append(report.runtime.train_tokens_per_second)
            overall_toks_per_s.append(report.runtime.overall_tokens_per_second)
            peak_process_memory_delta_mb.append(
                report.runtime.peak_process_memory_delta_bytes / (1024 * 1024)
            )
            overall_fraction, active_fraction = _candidate_payload(report, candidate.surface)
            overall_round2_fraction.append(overall_fraction)
            mean_active_round2_fraction.append(active_fraction)

        candidate_results.append(
            MiniMoePolicySearchCandidateResult(
                candidate_name=candidate.name,
                report_paths=tuple(report_paths),
                avg_final_loss=sum(final_losses) / len(final_losses),
                avg_train_toks_per_s=sum(train_toks_per_s) / len(train_toks_per_s),
                avg_overall_toks_per_s=sum(overall_toks_per_s) / len(overall_toks_per_s),
                avg_peak_process_memory_delta_mb=(
                    sum(peak_process_memory_delta_mb) / len(peak_process_memory_delta_mb)
                ),
                avg_overall_round2_fraction=(
                    sum(overall_round2_fraction) / len(overall_round2_fraction)
                ),
                avg_mean_active_round2_fraction=(
                    sum(mean_active_round2_fraction) / len(mean_active_round2_fraction)
                ),
            )
        )

    candidate_results.sort(
        key=lambda result: (result.avg_final_loss, -result.avg_train_toks_per_s)
    )
    summary_path = request.output_dir / "search_summary.json"
    payload = {
        "benchmark_name": request.benchmark_name,
        "seeds": list(request.seeds),
        "candidate_results": [
            {
                "candidate_name": result.candidate_name,
                "report_paths": list(result.report_paths),
                "avg_final_loss": result.avg_final_loss,
                "avg_train_toks_per_s": result.avg_train_toks_per_s,
                "avg_overall_toks_per_s": result.avg_overall_toks_per_s,
                "avg_peak_process_memory_delta_mb": result.avg_peak_process_memory_delta_mb,
                "avg_overall_round2_fraction": result.avg_overall_round2_fraction,
                "avg_mean_active_round2_fraction": result.avg_mean_active_round2_fraction,
            }
            for result in candidate_results
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MiniMoePolicySearchSummary(
        benchmark_name=request.benchmark_name,
        seeds=request.seeds,
        candidate_results=tuple(candidate_results),
        summary_path=str(summary_path),
    )
