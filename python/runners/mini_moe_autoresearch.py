from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import heapq
import json
import multiprocessing
from pathlib import Path
from typing import Any

from python.runners.mini_moe import MiniMoeRunnerRequest, run_mini_moe_variant
from python.specs.common import BenchmarkRunManifest
from python.specs.mini_moe import (
    MiniMoeSurfaceSpec,
    contiguous_layer_bands,
    transfer_round2_layer_bands_by_anchor_fill,
    transfer_round2_layer_bands_by_scaled_span,
    transfer_round2_layer_indices_by_depth_fraction,
)

STATE_SCHEMA_VERSION = 2


def mask_to_key(mask: tuple[int, ...]) -> str:
    if not mask:
        return "none"
    return "l" + "_".join(str(layer_index) for layer_index in mask)


def key_to_mask(mask_key: str) -> tuple[int, ...]:
    if mask_key == "none":
        return ()
    if not mask_key.startswith("l"):
        raise ValueError(f"unsupported mask key: {mask_key}")
    payload = mask_key[1:]
    if not payload:
        return ()
    return tuple(int(token) for token in payload.split("_"))


def neighbor_masks(mask: tuple[int, ...], total_layers: int) -> tuple[tuple[int, ...], ...]:
    active = set(mask)
    neighbors: set[tuple[int, ...]] = set()
    for layer_index in range(total_layers):
        if layer_index in active:
            next_mask = tuple(sorted(active - {layer_index}))
            if next_mask:
                neighbors.add(next_mask)
        else:
            next_mask = tuple(sorted(active | {layer_index}))
            if len(next_mask) < total_layers:
                neighbors.add(next_mask)
    return tuple(sorted(neighbors))


def mask_to_bitmask(mask: tuple[int, ...]) -> int:
    bitmask = 0
    for layer_index in mask:
        if layer_index < 0:
            raise ValueError(f"mask contains negative layer index: {layer_index}")
        bitmask |= 1 << layer_index
    return bitmask


def bitmask_to_mask(mask_id: int, total_layers: int) -> tuple[int, ...]:
    if mask_id < 0:
        raise ValueError(f"mask_id must be non-negative, got {mask_id}")
    return tuple(
        layer_index for layer_index in range(total_layers) if mask_id & (1 << layer_index)
    )


def bitmask_to_key(mask_id: int, total_layers: int) -> str:
    return mask_to_key(bitmask_to_mask(mask_id, total_layers))


def neighbor_mask_ids(mask_id: int, total_layers: int) -> tuple[int, ...]:
    return tuple(
        mask_to_bitmask(mask)
        for mask in neighbor_masks(bitmask_to_mask(mask_id, total_layers), total_layers)
    )


def iter_bitset_members(bitset: int) -> tuple[int, ...]:
    members: list[int] = []
    remaining = bitset
    while remaining:
        least_significant = remaining & -remaining
        members.append(least_significant.bit_length() - 1)
        remaining ^= least_significant
    return tuple(members)


@dataclass(frozen=True)
class MiniMoeAutoresearchBitmaskTable:
    total_layers: int
    evaluated_bitset: int = 0
    pending_bitset: int = 0

    def has_evaluated(self, mask_id: int) -> bool:
        return bool(self.evaluated_bitset & (1 << mask_id))

    def has_pending(self, mask_id: int) -> bool:
        return bool(self.pending_bitset & (1 << mask_id))

    def with_pending(self, mask_id: int) -> "MiniMoeAutoresearchBitmaskTable":
        return MiniMoeAutoresearchBitmaskTable(
            total_layers=self.total_layers,
            evaluated_bitset=self.evaluated_bitset,
            pending_bitset=self.pending_bitset | (1 << mask_id),
        )

    def without_pending(self, mask_id: int) -> "MiniMoeAutoresearchBitmaskTable":
        return MiniMoeAutoresearchBitmaskTable(
            total_layers=self.total_layers,
            evaluated_bitset=self.evaluated_bitset,
            pending_bitset=self.pending_bitset & ~(1 << mask_id),
        )

    def with_evaluated(self, mask_id: int) -> "MiniMoeAutoresearchBitmaskTable":
        return MiniMoeAutoresearchBitmaskTable(
            total_layers=self.total_layers,
            evaluated_bitset=self.evaluated_bitset | (1 << mask_id),
            pending_bitset=self.pending_bitset & ~(1 << mask_id),
        )

    def pending_count(self) -> int:
        return self.pending_bitset.bit_count()

    def to_payload(self) -> dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "evaluated_bitset_hex": format(self.evaluated_bitset, "x"),
            "pending_bitset_hex": format(self.pending_bitset, "x"),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MiniMoeAutoresearchBitmaskTable":
        return cls(
            total_layers=int(payload["total_layers"]),
            evaluated_bitset=int(payload.get("evaluated_bitset_hex", "0"), 16),
            pending_bitset=int(payload.get("pending_bitset_hex", "0"), 16),
        )

    @classmethod
    def from_records(
        cls,
        *,
        total_layers: int,
        evaluated_mask_ids: tuple[int, ...],
        pending_mask_ids: tuple[int, ...],
    ) -> "MiniMoeAutoresearchBitmaskTable":
        table = cls(total_layers=total_layers)
        for mask_id in evaluated_mask_ids:
            table = table.with_evaluated(mask_id)
        for mask_id in pending_mask_ids:
            if table.has_evaluated(mask_id):
                continue
            table = table.with_pending(mask_id)
        return table


@dataclass(frozen=True)
class MiniMoeAutoresearchCandidateResult:
    candidate_name: str
    mask: tuple[int, ...]
    report_paths: tuple[str, ...]
    avg_final_loss: float
    avg_train_toks_per_s: float
    avg_overall_toks_per_s: float
    avg_peak_process_memory_delta_mb: float
    avg_overall_round2_fraction: float
    avg_mean_active_round2_fraction: float


@dataclass(frozen=True)
class MiniMoeAutoresearchRequest:
    benchmark_name: str
    manifest_template: BenchmarkRunManifest
    output_dir: Path
    seeds: tuple[int, ...]
    total_layers: int
    experts_per_block: int
    normalized_entropy_threshold: float
    source_total_layers: int
    source_round2_layers: tuple[int, ...]
    ledger_path: Path | None = None
    max_new_candidates: int | None = None
    resume: bool = True
    stop_on_first_success: bool = True
    parallel_candidates: int = 1


@dataclass(frozen=True)
class MiniMoeAutoresearchSummary:
    benchmark_name: str
    status: str
    state_path: str
    summary_path: str
    baseline_reference: MiniMoeAutoresearchCandidateResult
    baseline_all_layers: MiniMoeAutoresearchCandidateResult
    best_selective: MiniMoeAutoresearchCandidateResult | None
    success_candidate: MiniMoeAutoresearchCandidateResult | None
    evaluated_selective_count: int
    pending_candidate_count: int
    total_selective_search_space: int


def _round2_fractions(report: Any, surface: MiniMoeSurfaceSpec) -> tuple[float, float]:
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


def _build_surface(
    request: MiniMoeAutoresearchRequest,
    *,
    kind: str,
    round2_layer_indices: tuple[int, ...] | None = None,
) -> MiniMoeSurfaceSpec:
    if kind == "reference":
        surface = MiniMoeSurfaceSpec.phase1_reference_default()
    else:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default(
            normalized_entropy_threshold=request.normalized_entropy_threshold,
            round2_layer_indices=round2_layer_indices,
        )
    from dataclasses import replace

    architecture = replace(
        surface.architecture,
        label=f"{surface.architecture.label}-e{request.experts_per_block}-d{request.total_layers}",
        backbone=replace(surface.architecture.backbone, total_layers=request.total_layers),
        moe=replace(surface.architecture.moe, experts_per_block=request.experts_per_block),
    )
    observability = replace(surface.observability, max_token_route_traces_per_layer=0)
    return replace(surface, architecture=architecture, observability=observability)


def _evaluate_candidate(
    request: MiniMoeAutoresearchRequest,
    *,
    candidate_name: str,
    surface: MiniMoeSurfaceSpec,
    note: str,
) -> MiniMoeAutoresearchCandidateResult:
    report_paths: list[str] = []
    final_losses: list[float] = []
    train_toks_per_s: list[float] = []
    overall_toks_per_s: list[float] = []
    peak_process_memory_delta_mb: list[float] = []
    overall_round2_fraction: list[float] = []
    mean_active_round2_fraction: list[float] = []
    for seed in request.seeds:
        manifest = BenchmarkRunManifest(
            run_label=f"{request.manifest_template.run_label}-{candidate_name}-seed{seed}",
            implementation_kind=request.manifest_template.implementation_kind,
            seed_spec=type(request.manifest_template.seed_spec)(
                model_seed=seed,
                data_seed=request.manifest_template.seed_spec.data_seed,
            ),
            corpus=request.manifest_template.corpus,
            budget=request.manifest_template.budget,
            runtime=request.manifest_template.runtime,
            benchmark_name=request.benchmark_name,
            note=note or request.manifest_template.note,
        )
        report = run_mini_moe_variant(
            MiniMoeRunnerRequest(
                manifest=manifest,
                surface=surface,
                output_dir=request.output_dir,
                output_format="json",
                ledger_path=request.ledger_path,
                variant_output_name=f"{candidate_name}-seed{seed}",
                model_note=note or request.manifest_template.note,
            )
        )
        report_paths.append(report.report_path or "")
        final_losses.append(report.final_eval.mean_loss)
        train_toks_per_s.append(report.runtime.train_tokens_per_second)
        overall_toks_per_s.append(report.runtime.overall_tokens_per_second)
        peak_process_memory_delta_mb.append(
            report.runtime.peak_process_memory_delta_bytes / (1024 * 1024)
        )
        overall_fraction, active_fraction = _round2_fractions(report, surface)
        overall_round2_fraction.append(overall_fraction)
        mean_active_round2_fraction.append(active_fraction)
    return MiniMoeAutoresearchCandidateResult(
        candidate_name=candidate_name,
        mask=surface.architecture.resolved_round2_layer_indices(),
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


def _beats_all_layer(
    candidate: MiniMoeAutoresearchCandidateResult,
    all_layer: MiniMoeAutoresearchCandidateResult,
) -> bool:
    return (
        candidate.avg_final_loss < all_layer.avg_final_loss
        and candidate.avg_train_toks_per_s > all_layer.avg_train_toks_per_s
    )


def _seed_masks(request: MiniMoeAutoresearchRequest) -> tuple[tuple[int, ...], ...]:
    return tuple(
        dict.fromkeys(
            [
                transfer_round2_layer_indices_by_depth_fraction(
                    source_layer_indices=request.source_round2_layers,
                    source_total_layers=request.source_total_layers,
                    target_total_layers=request.total_layers,
                ),
                transfer_round2_layer_bands_by_anchor_fill(
                    source_layer_indices=request.source_round2_layers,
                    source_total_layers=request.source_total_layers,
                    target_total_layers=request.total_layers,
                ),
                transfer_round2_layer_bands_by_scaled_span(
                    source_layer_indices=request.source_round2_layers,
                    source_total_layers=request.source_total_layers,
                    target_total_layers=request.total_layers,
                ),
            ]
        )
    )


def _candidate_priority(
    parent_result: MiniMoeAutoresearchCandidateResult,
    mask_id: int,
    total_layers: int,
) -> tuple[float, float, int, int, int]:
    mask = bitmask_to_mask(mask_id, total_layers)
    return (
        parent_result.avg_final_loss,
        -parent_result.avg_train_toks_per_s,
        len(mask),
        len(contiguous_layer_bands(mask)),
        mask_id,
    )


def _total_selective_search_space(total_layers: int) -> int:
    if total_layers < 2:
        return 0
    return (2**total_layers) - 2


def _result_sort_key(
    result: MiniMoeAutoresearchCandidateResult,
) -> tuple[float, float]:
    return (result.avg_final_loss, -result.avg_train_toks_per_s)


def _best_result(
    left: MiniMoeAutoresearchCandidateResult | None,
    right: MiniMoeAutoresearchCandidateResult | None,
) -> MiniMoeAutoresearchCandidateResult | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if _result_sort_key(left) <= _result_sort_key(right) else right


def _resume_is_terminal(status: str, *, stop_on_first_success: bool) -> bool:
    if status == "exhausted":
        return True
    if status == "success" and stop_on_first_success:
        return True
    return False


def _pop_live_pending_mask(
    pending_heap: list[tuple[float, float, int, int, int]],
    search_table: MiniMoeAutoresearchBitmaskTable,
) -> tuple[int | None, MiniMoeAutoresearchBitmaskTable]:
    table = search_table
    while pending_heap:
        _, _, _, _, mask_id = heapq.heappop(pending_heap)
        if table.has_pending(mask_id):
            return mask_id, table.without_pending(mask_id)
    return None, table


def _pop_live_pending_masks(
    pending_heap: list[tuple[float, float, int, int, int]],
    search_table: MiniMoeAutoresearchBitmaskTable,
    limit: int,
) -> tuple[list[int], MiniMoeAutoresearchBitmaskTable]:
    batch: list[int] = []
    table = search_table
    while len(batch) < limit:
        mask_id, table = _pop_live_pending_mask(pending_heap, table)
        if mask_id is None:
            break
        batch.append(mask_id)
    return batch, table


def _state_path(output_dir: Path) -> Path:
    return output_dir / "autoresearch_state.json"


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "autoresearch_summary.json"


def _save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _emit_progress(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True), flush=True)


def _evaluate_candidate_entry(
    request: MiniMoeAutoresearchRequest,
    candidate_name: str,
    surface: MiniMoeSurfaceSpec,
    note: str,
) -> MiniMoeAutoresearchCandidateResult:
    return _evaluate_candidate(
        request,
        candidate_name=candidate_name,
        surface=surface,
        note=note,
    )


def _result_from_payload(payload: dict[str, Any]) -> MiniMoeAutoresearchCandidateResult:
    return MiniMoeAutoresearchCandidateResult(
        candidate_name=payload["candidate_name"],
        mask=tuple(payload["mask"]),
        report_paths=tuple(payload["report_paths"]),
        avg_final_loss=payload["avg_final_loss"],
        avg_train_toks_per_s=payload["avg_train_toks_per_s"],
        avg_overall_toks_per_s=payload["avg_overall_toks_per_s"],
        avg_peak_process_memory_delta_mb=payload["avg_peak_process_memory_delta_mb"],
        avg_overall_round2_fraction=payload["avg_overall_round2_fraction"],
        avg_mean_active_round2_fraction=payload["avg_mean_active_round2_fraction"],
    )


def load_mini_moe_autoresearch_state(
    path: Path,
    *,
    total_layers: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selective_records, search_table = _records_and_table_from_state(
        payload,
        total_layers=total_layers,
    )
    baseline_reference = _result_from_payload(payload["baseline_reference"])
    baseline_all_layers = _result_from_payload(payload["baseline_all_layers"])
    success_payload = payload.get("success_candidate")
    return {
        "payload": payload,
        "baseline_reference": baseline_reference,
        "baseline_all_layers": baseline_all_layers,
        "selective_records": selective_records,
        "search_table": search_table,
        "success_candidate": (
            None if success_payload is None else _result_from_payload(success_payload)
        ),
    }


def top_selective_mask_ids_from_state(
    path: Path,
    *,
    total_layers: int,
    limit: int,
) -> tuple[int, ...]:
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    state = load_mini_moe_autoresearch_state(path, total_layers=total_layers)
    ranked = sorted(
        state["selective_records"].items(),
        key=lambda item: _result_sort_key(item[1]),
    )
    return tuple(mask_id for mask_id, _ in ranked[:limit])


def _result_payload(
    result: MiniMoeAutoresearchCandidateResult,
    *,
    mask_id: int | None = None,
    baseline_all_layers: MiniMoeAutoresearchCandidateResult | None = None,
) -> dict[str, Any]:
    payload = asdict(result)
    if mask_id is not None:
        payload["mask_id"] = mask_id
    if baseline_all_layers is not None:
        payload["loss_gap_vs_all_layers"] = (
            result.avg_final_loss - baseline_all_layers.avg_final_loss
        )
        payload["train_gap_vs_all_layers"] = (
            result.avg_train_toks_per_s - baseline_all_layers.avg_train_toks_per_s
        )
    return payload


def _leaderboard_payload(
    selective_records: dict[int, MiniMoeAutoresearchCandidateResult],
    *,
    baseline_all_layers: MiniMoeAutoresearchCandidateResult,
    limit: int = 10,
) -> list[dict[str, Any]]:
    top = sorted(selective_records.items(), key=lambda item: _result_sort_key(item[1]))[:limit]
    return [
        _result_payload(result, mask_id=mask_id, baseline_all_layers=baseline_all_layers)
        for mask_id, result in top
    ]


def _best_selective(
    selective_records: dict[int, MiniMoeAutoresearchCandidateResult],
) -> tuple[int, MiniMoeAutoresearchCandidateResult] | None:
    if not selective_records:
        return None
    return min(selective_records.items(), key=lambda item: _result_sort_key(item[1]))


def _build_runtime_state_payload(
    *,
    request: MiniMoeAutoresearchRequest,
    status: str,
    baseline_reference: MiniMoeAutoresearchCandidateResult,
    baseline_all_layers: MiniMoeAutoresearchCandidateResult,
    selective_records: dict[int, MiniMoeAutoresearchCandidateResult],
    search_table: MiniMoeAutoresearchBitmaskTable,
    success_candidate: MiniMoeAutoresearchCandidateResult | None,
    total_selective_search_space: int,
) -> dict[str, Any]:
    best_entry = _best_selective(selective_records)
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "benchmark_name": request.benchmark_name,
        "status": status,
        "baseline_reference": _result_payload(baseline_reference),
        "baseline_all_layers": _result_payload(baseline_all_layers),
        "best_selective": (
            None
            if best_entry is None
            else _result_payload(
                best_entry[1],
                mask_id=best_entry[0],
                baseline_all_layers=baseline_all_layers,
            )
        ),
        "success_candidate": (
            None
            if success_candidate is None
            else _result_payload(success_candidate, baseline_all_layers=baseline_all_layers)
        ),
        "evaluated_selective_count": len(selective_records),
        "pending_candidate_count": search_table.pending_count(),
        "total_selective_search_space": total_selective_search_space,
        "search_table": search_table.to_payload(),
        "evaluated_selective_records_by_id": {
            str(mask_id): _result_payload(result, mask_id=mask_id)
            for mask_id, result in sorted(selective_records.items())
        },
        "leaderboard": _leaderboard_payload(
            selective_records,
            baseline_all_layers=baseline_all_layers,
        ),
    }


def _build_summary_payload(
    *,
    request: MiniMoeAutoresearchRequest,
    status: str,
    baseline_reference: MiniMoeAutoresearchCandidateResult,
    baseline_all_layers: MiniMoeAutoresearchCandidateResult,
    selective_records: dict[int, MiniMoeAutoresearchCandidateResult],
    search_table: MiniMoeAutoresearchBitmaskTable,
    success_candidate: MiniMoeAutoresearchCandidateResult | None,
    total_selective_search_space: int,
) -> dict[str, Any]:
    best_entry = _best_selective(selective_records)
    best_selective_loss_gap_vs_all_layers = None
    best_selective_train_gap_vs_all_layers = None
    if best_entry is not None:
        best_selective_loss_gap_vs_all_layers = (
            best_entry[1].avg_final_loss - baseline_all_layers.avg_final_loss
        )
        best_selective_train_gap_vs_all_layers = (
            best_entry[1].avg_train_toks_per_s - baseline_all_layers.avg_train_toks_per_s
        )
    success_loss_gap_vs_all_layers = None
    success_train_gap_vs_all_layers = None
    if success_candidate is not None:
        success_loss_gap_vs_all_layers = (
            success_candidate.avg_final_loss - baseline_all_layers.avg_final_loss
        )
        success_train_gap_vs_all_layers = (
            success_candidate.avg_train_toks_per_s - baseline_all_layers.avg_train_toks_per_s
        )
    return {
        "state_schema_version": STATE_SCHEMA_VERSION,
        "benchmark_name": request.benchmark_name,
        "status": status,
        "success_found": 1 if success_candidate is not None else 0,
        "baseline_reference": _result_payload(baseline_reference),
        "baseline_all_layers": _result_payload(baseline_all_layers),
        "best_selective": (
            None
            if best_entry is None
            else _result_payload(
                best_entry[1],
                mask_id=best_entry[0],
                baseline_all_layers=baseline_all_layers,
            )
        ),
        "success_candidate": (
            None
            if success_candidate is None
            else _result_payload(success_candidate, baseline_all_layers=baseline_all_layers)
        ),
        "best_selective_loss_gap_vs_all_layers": best_selective_loss_gap_vs_all_layers,
        "best_selective_train_gap_vs_all_layers": best_selective_train_gap_vs_all_layers,
        "success_loss_gap_vs_all_layers": success_loss_gap_vs_all_layers,
        "success_train_gap_vs_all_layers": success_train_gap_vs_all_layers,
        "evaluated_selective_count": len(selective_records),
        "pending_candidate_count": search_table.pending_count(),
        "total_selective_search_space": total_selective_search_space,
        "leaderboard": _leaderboard_payload(
            selective_records,
            baseline_all_layers=baseline_all_layers,
        ),
    }


def _records_and_table_from_state(
    state_payload: dict[str, Any],
    *,
    total_layers: int,
) -> tuple[dict[int, MiniMoeAutoresearchCandidateResult], MiniMoeAutoresearchBitmaskTable]:
    if "evaluated_selective_records_by_id" in state_payload or "search_table" in state_payload:
        selective_records = {
            int(mask_id): _result_from_payload(payload)
            for mask_id, payload in state_payload.get("evaluated_selective_records_by_id", {}).items()
        }
        if "search_table" in state_payload:
            search_table = MiniMoeAutoresearchBitmaskTable.from_payload(
                state_payload["search_table"]
            )
        else:
            pending_mask_ids = tuple(int(mask_id) for mask_id in state_payload.get("pending_mask_ids", []))
            search_table = MiniMoeAutoresearchBitmaskTable.from_records(
                total_layers=total_layers,
                evaluated_mask_ids=tuple(selective_records.keys()),
                pending_mask_ids=pending_mask_ids,
            )
        return selective_records, search_table

    selective_records = {
        mask_to_bitmask(key_to_mask(mask_key)): _result_from_payload(payload)
        for mask_key, payload in state_payload.get("evaluated_selective", {}).items()
    }
    pending_mask_ids = tuple(
        mask_to_bitmask(key_to_mask(mask_key)) for mask_key in state_payload.get("pending_masks", [])
    )
    search_table = MiniMoeAutoresearchBitmaskTable.from_records(
        total_layers=total_layers,
        evaluated_mask_ids=tuple(selective_records.keys()),
        pending_mask_ids=pending_mask_ids,
    )
    return selective_records, search_table


def _summary_from_state_payload(
    request: MiniMoeAutoresearchRequest,
    state_payload: dict[str, Any],
    *,
    total_selective_search_space: int,
) -> MiniMoeAutoresearchSummary:
    baseline_reference = _result_from_payload(state_payload["baseline_reference"])
    baseline_all_layers = _result_from_payload(state_payload["baseline_all_layers"])
    best_payload = state_payload.get("best_selective")
    success_payload = state_payload.get("success_candidate")
    return MiniMoeAutoresearchSummary(
        benchmark_name=request.benchmark_name,
        status=state_payload["status"],
        state_path=str(_state_path(request.output_dir)),
        summary_path=str(_summary_path(request.output_dir)),
        baseline_reference=baseline_reference,
        baseline_all_layers=baseline_all_layers,
        best_selective=None if best_payload is None else _result_from_payload(best_payload),
        success_candidate=None if success_payload is None else _result_from_payload(success_payload),
        evaluated_selective_count=state_payload.get("evaluated_selective_count", 0),
        pending_candidate_count=state_payload.get("pending_candidate_count", 0),
        total_selective_search_space=state_payload.get(
            "total_selective_search_space",
            total_selective_search_space,
        ),
    )


def run_mini_moe_autoresearch(
    request: MiniMoeAutoresearchRequest,
) -> MiniMoeAutoresearchSummary:
    request.manifest_template.validate()
    if not request.seeds:
        raise ValueError("mini_moe_autoresearch requires at least one seed")
    if request.parallel_candidates < 1:
        raise ValueError("mini_moe_autoresearch parallel_candidates must be at least 1")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    total_selective_search_space = _total_selective_search_space(request.total_layers)

    state_path = _state_path(request.output_dir)
    if request.resume and state_path.exists():
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
        if _resume_is_terminal(
            state_payload.get("status", ""),
            stop_on_first_success=request.stop_on_first_success,
        ):
            return _summary_from_state_payload(
                request,
                state_payload,
                total_selective_search_space=total_selective_search_space,
            )

        baseline_reference = _result_from_payload(state_payload["baseline_reference"])
        baseline_all_layers = _result_from_payload(state_payload["baseline_all_layers"])
        selective_records, search_table = _records_and_table_from_state(
            state_payload,
            total_layers=request.total_layers,
        )
        success_payload = state_payload.get("success_candidate")
        success_candidate = (
            None if success_payload is None else _result_from_payload(success_payload)
        )
        pending_heap = [
            _candidate_priority(
                baseline_all_layers,
                mask_id,
                request.total_layers,
            )
            for mask_id in iter_bitset_members(search_table.pending_bitset)
        ]
        heapq.heapify(pending_heap)
        _emit_progress(
            "resume_loaded",
            benchmark_name=request.benchmark_name,
            evaluated_selective_count=len(selective_records),
            pending_candidate_count=search_table.pending_count(),
            total_selective_search_space=total_selective_search_space,
            success_found=1 if success_candidate is not None else 0,
        )
    else:
        baseline_reference = _evaluate_candidate(
            request,
            candidate_name="reference",
            surface=_build_surface(request, kind="reference"),
            note="One-shot standard MoE baseline",
        )
        baseline_all_layers = _evaluate_candidate(
            request,
            candidate_name="entropy_all_layers",
            surface=_build_surface(request, kind="recurrent"),
            note="All-layer entropy-gated recurrent baseline",
        )
        selective_records: dict[int, MiniMoeAutoresearchCandidateResult] = {}
        seed_mask_ids = tuple(mask_to_bitmask(mask) for mask in _seed_masks(request))
        search_table = MiniMoeAutoresearchBitmaskTable.from_records(
            total_layers=request.total_layers,
            evaluated_mask_ids=(),
            pending_mask_ids=seed_mask_ids,
        )
        pending_heap = [
            _candidate_priority(baseline_all_layers, mask_id, request.total_layers)
            for mask_id in seed_mask_ids
        ]
        heapq.heapify(pending_heap)
        success_candidate = None

    _save_state(
        state_path,
        _build_runtime_state_payload(
            request=request,
            status="running",
            baseline_reference=baseline_reference,
            baseline_all_layers=baseline_all_layers,
            selective_records=selective_records,
            search_table=search_table,
            success_candidate=success_candidate,
            total_selective_search_space=total_selective_search_space,
        ),
    )
    _emit_progress(
        "baselines_ready",
        benchmark_name=request.benchmark_name,
        baseline_reference_loss=baseline_reference.avg_final_loss,
        baseline_reference_train_toks_per_s=baseline_reference.avg_train_toks_per_s,
        baseline_all_layers_loss=baseline_all_layers.avg_final_loss,
        baseline_all_layers_train_toks_per_s=baseline_all_layers.avg_train_toks_per_s,
        pending_candidate_count=search_table.pending_count(),
        total_selective_search_space=total_selective_search_space,
    )

    evaluated_count = 0
    while search_table.pending_count() > 0:
        if request.max_new_candidates is not None and evaluated_count >= request.max_new_candidates:
            break
        remaining_budget = (
            None
            if request.max_new_candidates is None
            else max(0, request.max_new_candidates - evaluated_count)
        )
        batch_limit = request.parallel_candidates if remaining_budget is None else min(
            request.parallel_candidates,
            remaining_budget,
        )
        if batch_limit <= 0:
            break
        mask_ids, search_table = _pop_live_pending_masks(
            pending_heap,
            search_table,
            batch_limit,
        )
        if not mask_ids:
            break

        batch_specs: list[tuple[int, str, MiniMoeSurfaceSpec, tuple[int, ...]]] = []
        for mask_id in mask_ids:
            if mask_id in selective_records:
                continue
            mask = bitmask_to_mask(mask_id, request.total_layers)
            if not mask or len(mask) >= request.total_layers:
                continue
            candidate_key = bitmask_to_key(mask_id, request.total_layers)
            batch_specs.append(
                (
                    mask_id,
                    f"entropy_{candidate_key}",
                    _build_surface(request, kind="recurrent", round2_layer_indices=mask),
                    mask,
                )
            )
        if not batch_specs:
            continue

        _emit_progress(
            "batch_launched",
            batch_size=len(batch_specs),
            pending_candidate_count=search_table.pending_count(),
            evaluated_selective_count=len(selective_records),
            total_selective_search_space=total_selective_search_space,
            parallel_candidates=request.parallel_candidates,
        )

        batch_results: dict[int, MiniMoeAutoresearchCandidateResult] = {}
        if len(batch_specs) == 1:
            mask_id, candidate_name, surface, mask = batch_specs[0]
            batch_results[mask_id] = _evaluate_candidate(
                request,
                candidate_name=candidate_name,
                surface=surface,
                note=f"Autoresearch selective entropy-gated recurrent mask {mask}",
            )
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(batch_specs), mp_context=ctx) as executor:
                futures = {
                    mask_id: executor.submit(
                        _evaluate_candidate_entry,
                        request,
                        candidate_name,
                        surface,
                        f"Autoresearch selective entropy-gated recurrent mask {mask}",
                    )
                    for mask_id, candidate_name, surface, mask in batch_specs
                }
                for mask_id, future in futures.items():
                    batch_results[mask_id] = future.result()

        should_stop = False
        for mask_id, _, _, mask in batch_specs:
            candidate_result = batch_results[mask_id]
            search_table = search_table.with_evaluated(mask_id)
            selective_records[mask_id] = candidate_result
            evaluated_count += 1

            prior_success_candidate = success_candidate
            if _beats_all_layer(candidate_result, baseline_all_layers):
                success_candidate = _best_result(success_candidate, candidate_result)

            _emit_progress(
                "candidate_evaluated",
                candidate_name=candidate_result.candidate_name,
                mask=list(candidate_result.mask),
                avg_final_loss=candidate_result.avg_final_loss,
                avg_train_toks_per_s=candidate_result.avg_train_toks_per_s,
                avg_overall_round2_fraction=candidate_result.avg_overall_round2_fraction,
                avg_mean_active_round2_fraction=candidate_result.avg_mean_active_round2_fraction,
                evaluated_selective_count=len(selective_records),
                pending_candidate_count=search_table.pending_count(),
                total_selective_search_space=total_selective_search_space,
                success_found=1 if success_candidate is not None else 0,
            )
            if success_candidate is not None and success_candidate is not prior_success_candidate:
                _emit_progress(
                    "success_candidate_improved",
                    candidate_name=success_candidate.candidate_name,
                    mask=list(success_candidate.mask),
                    avg_final_loss=success_candidate.avg_final_loss,
                    avg_train_toks_per_s=success_candidate.avg_train_toks_per_s,
                )
            if _beats_all_layer(candidate_result, baseline_all_layers) and request.stop_on_first_success:
                _emit_progress(
                    "success_candidate_found",
                    candidate_name=success_candidate.candidate_name,
                    mask=list(success_candidate.mask),
                    avg_final_loss=success_candidate.avg_final_loss,
                    avg_train_toks_per_s=success_candidate.avg_train_toks_per_s,
                )
                should_stop = True

            for neighbor_id in neighbor_mask_ids(mask_id, request.total_layers):
                if search_table.has_evaluated(neighbor_id) or search_table.has_pending(neighbor_id):
                    continue
                heapq.heappush(
                    pending_heap,
                    _candidate_priority(candidate_result, neighbor_id, request.total_layers),
                )
                search_table = search_table.with_pending(neighbor_id)

        if should_stop:
            break

        _save_state(
            state_path,
            _build_runtime_state_payload(
                request=request,
                status=(
                    "success"
                    if success_candidate is not None and request.stop_on_first_success
                    else "running"
                ),
                baseline_reference=baseline_reference,
                baseline_all_layers=baseline_all_layers,
                selective_records=selective_records,
                search_table=search_table,
                success_candidate=success_candidate,
                total_selective_search_space=total_selective_search_space,
            ),
        )

    if search_table.pending_count() > 0:
        status = "running"
    elif success_candidate is not None:
        status = "success"
    else:
        status = "exhausted"

    summary_payload = _build_summary_payload(
        request=request,
        status=status,
        baseline_reference=baseline_reference,
        baseline_all_layers=baseline_all_layers,
        selective_records=selective_records,
        search_table=search_table,
        success_candidate=success_candidate,
        total_selective_search_space=total_selective_search_space,
    )
    _save_state(_summary_path(request.output_dir), summary_payload)
    _save_state(
        state_path,
        _build_runtime_state_payload(
            request=request,
            status=status,
            baseline_reference=baseline_reference,
            baseline_all_layers=baseline_all_layers,
            selective_records=selective_records,
            search_table=search_table,
            success_candidate=success_candidate,
            total_selective_search_space=total_selective_search_space,
        ),
    )
    _emit_progress(
        "search_finished",
        benchmark_name=request.benchmark_name,
        status=status,
        evaluated_selective_count=len(selective_records),
        pending_candidate_count=search_table.pending_count(),
        total_selective_search_space=total_selective_search_space,
    )

    best_entry = _best_selective(selective_records)
    return MiniMoeAutoresearchSummary(
        benchmark_name=request.benchmark_name,
        status=status,
        state_path=str(state_path),
        summary_path=str(_summary_path(request.output_dir)),
        baseline_reference=baseline_reference,
        baseline_all_layers=baseline_all_layers,
        best_selective=None if best_entry is None else best_entry[1],
        success_candidate=success_candidate,
        evaluated_selective_count=len(selective_records),
        pending_candidate_count=search_table.pending_count(),
        total_selective_search_space=total_selective_search_space,
    )
