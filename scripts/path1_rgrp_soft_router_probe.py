#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.reporting.render import render_path1_table
from python.runners.path1 import Path1RunnerRequest, run_path1_variant
from python.runners.path1_cli import _implementation_kind_for_variant
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
    to_jsonable,
)
from python.specs.path1 import (
    Path1ModelShape,
    TokenRoutingProfile,
    phase1_attention_only_variant,
)

SUITE_NAME = "path1-rgrp-soft-router-probe-v2"
DEFAULT_CORPUS = Path("experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    family: str
    note: str
    build_variant: Callable[[Path1ModelShape], object]


def _lanes() -> tuple[LaneSpec, ...]:
    return (
        LaneSpec(
            name="attention-control",
            family="control",
            note="Dense attention control.",
            build_variant=lambda shape: phase1_attention_only_variant(shape=shape),
        ),
        LaneSpec(
            name="causal-topk-route50-layer3",
            family="hard-router-control",
            note="Middle-band hard causal prefix-top-k router from the prior ladder.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(3,),
            ),
        ),
        LaneSpec(
            name="soft-gate-floor25-layer3",
            family="soft-router-control",
            note="Dense reference partial-update block with an MLP gate floor of 0.25.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.SOFT_GATE_BLOCK,
                token_route_fraction=0.25,
                token_routing_layer_indices=(3,),
            ),
        ),
        LaneSpec(
            name="soft-gate-floor50-layer3",
            family="soft-router-control",
            note="Dense reference partial-update block with an MLP gate floor of 0.50.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.SOFT_GATE_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(3,),
            ),
        ),
        LaneSpec(
            name="rotary-soft-gate-floor25-layer3",
            family="rotary-soft-router",
            note="Dense reference partial-update block controlled by a P20/RGRP-style rotary scan with gate floor 0.25.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK,
                token_route_fraction=0.25,
                token_routing_layer_indices=(3,),
            ),
        ),
        LaneSpec(
            name="rotary-soft-gate-floor50-layer3",
            family="rotary-soft-router",
            note="Dense reference partial-update block controlled by a P20/RGRP-style rotary scan with gate floor 0.50.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.ROTARY_SOFT_GATE_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(3,),
            ),
        ),
    )


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk) for chunk in value.replace(",", " ").split())
    if not seeds:
        raise argparse.ArgumentTypeError("must provide at least one seed")
    return seeds


def _selected_lanes(names: set[str] | None) -> tuple[LaneSpec, ...]:
    lanes = _lanes()
    if names is None:
        return lanes
    known = {lane.name for lane in lanes}
    unknown = sorted(names - known)
    if unknown:
        raise SystemExit(f"unknown lane(s): {', '.join(unknown)}")
    return tuple(lane for lane in lanes if lane.name in names)


def _diagnostics(report: dict[str, object]) -> dict[str, object]:
    diagnostics = report.get("diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def _routing_payload(diagnostics: dict[str, object]) -> dict[str, object] | None:
    token_routing = diagnostics.get("token_routing")
    if not isinstance(token_routing, dict):
        return None
    blocks = token_routing.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None
    block = blocks[0]
    if not isinstance(block, dict):
        return None
    payload = block.get("payload")
    return payload if isinstance(payload, dict) else None


def _float_payload(payload: dict[str, object] | None, key: str) -> float | None:
    if payload is None:
        return None
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _row(
    *, lane: LaneSpec, seed: int, report_path: Path, report: dict[str, object]
) -> dict[str, object]:
    diagnostics = _diagnostics(report)
    runtime = report["runtime"]
    final_eval = report["final_eval"]
    if not isinstance(runtime, dict) or not isinstance(final_eval, dict):
        raise TypeError("report runtime/final_eval payloads must be dicts")
    routing = _routing_payload(diagnostics)
    return {
        "lane": lane.name,
        "family": lane.family,
        "seed": seed,
        "final_loss": final_eval["mean_loss"],
        "train_tokens_per_second": runtime["train_tokens_per_second"],
        "parameter_count": diagnostics.get("parameter_count"),
        "selected_token_fraction": _float_payload(
            routing, "last_selected_fraction"
        ),
        "mean_gate": _float_payload(routing, "last_mean_gate"),
        "gate_floor": _float_payload(routing, "gate_floor"),
        "accepted_delta_ratio": _float_payload(
            routing, "last_accepted_delta_ratio"
        ),
        "controller_norm": _float_payload(routing, "last_controller_norm"),
        "raw_controller_norm": _float_payload(routing, "last_raw_controller_norm"),
        "gate_first_half_mean": _float_payload(
            routing, "last_gate_first_half_mean"
        ),
        "gate_second_half_mean": _float_payload(
            routing, "last_gate_second_half_mean"
        ),
        "report_path": str(report_path.resolve()),
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _std(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    mean = _mean(materialized)
    return math.sqrt(sum((value - mean) ** 2 for value in materialized) / len(materialized))


def _maybe_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row[key], (int, float))]
    return _mean(values) if values else None


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate: list[dict[str, object]] = []
    for lane in sorted({str(row["lane"]) for row in rows}):
        group = [row for row in rows if row["lane"] == lane]
        losses = [float(row["final_loss"]) for row in group]
        speeds = [float(row["train_tokens_per_second"]) for row in group]
        aggregate.append(
            {
                "lane": lane,
                "family": group[0]["family"],
                "mean_final_loss": _mean(losses),
                "std_final_loss": _std(losses),
                "mean_train_tokens_per_second": _mean(speeds),
                "parameter_count": group[0]["parameter_count"],
                "mean_selected_token_fraction": _maybe_mean(
                    group, "selected_token_fraction"
                ),
                "mean_gate": _maybe_mean(group, "mean_gate"),
                "mean_accepted_delta_ratio": _maybe_mean(
                    group, "accepted_delta_ratio"
                ),
                "mean_controller_norm": _maybe_mean(group, "controller_norm"),
                "mean_raw_controller_norm": _maybe_mean(
                    group, "raw_controller_norm"
                ),
                "mean_gate_first_half": _maybe_mean(group, "gate_first_half_mean"),
                "mean_gate_second_half": _maybe_mean(group, "gate_second_half_mean"),
                "seeds": [row["seed"] for row in group],
            }
        )
    aggregate.sort(key=lambda row: float(row["mean_final_loss"]))
    return aggregate


def _format_optional(value: object, digits: int = 3) -> str:
    if not isinstance(value, float):
        return ""
    if value != 0.0 and (abs(value) >= 1.0e5 or abs(value) < 1.0e-3):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _write_summary(
    output_dir: Path, rows: list[dict[str, object]], conditions: dict[str, object]
) -> None:
    aggregate = _aggregate(rows)
    payload = {
        "suite_name": SUITE_NAME,
        "conditions": conditions,
        "aggregate": aggregate,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {SUITE_NAME}",
        "",
        "Probe for replacing hard token skipping with soft partial-update routing, including a P20/RGRP-style rotary recurrent gate controller.",
        "",
        "## Conditions",
        "",
    ]
    for key, value in conditions.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "| Lane | Family | Loss | Loss Std | Tok/s | Params | Selected | Gate | Accepted Delta | Controller Norm | Raw Controller Norm | Gate H1 | Gate H2 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate:
        lines.append(
            "| {lane} | {family} | {loss:.4f} | {std:.4f} | {speed:.1f} | {params} | {selected} | {gate} | {accepted} | {controller} | {raw_controller} | {gate_h1} | {gate_h2} |".format(
                lane=row["lane"],
                family=row["family"],
                loss=row["mean_final_loss"],
                std=row["std_final_loss"],
                speed=row["mean_train_tokens_per_second"],
                params=row["parameter_count"],
                selected=_format_optional(row["mean_selected_token_fraction"]),
                gate=_format_optional(row["mean_gate"]),
                accepted=_format_optional(row["mean_accepted_delta_ratio"]),
                controller=_format_optional(row["mean_controller_norm"]),
                raw_controller=_format_optional(row["mean_raw_controller_norm"]),
                gate_h1=_format_optional(row["mean_gate_first_half"]),
                gate_h2=_format_optional(row["mean_gate_second_half"]),
            )
        )
    seed_values = sorted({int(row["seed"]) for row in rows})
    lines.extend(["", "## Per-Seed Final Loss", ""])
    lines.append("| Lane | " + " | ".join(f"Seed {seed}" for seed in seed_values) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in seed_values) + " |")
    for lane in sorted({str(row["lane"]) for row in rows}):
        by_seed = {
            int(row["seed"]): float(row["final_loss"])
            for row in rows
            if row["lane"] == lane
        }
        lines.append(
            "| "
            + lane
            + " | "
            + " | ".join(f"{by_seed[seed]:.4f}" for seed in seed_values)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a soft-router/RGRP-gate Path1 probe."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / SUITE_NAME)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(42, 43, 44))
    parser.add_argument("--lanes", help="Comma or whitespace separated lane names.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lane_names = set(args.lanes.replace(",", " ").split()) if args.lanes else None
    lanes = _selected_lanes(lane_names)
    shape = Path1ModelShape(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        head_count=args.heads,
        total_layers=args.layers,
        ffn_multiplier=args.ffn_multiplier,
    )
    shape.validate()
    budget = BenchmarkBudgetSpec(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        warmup_eval_batches=1,
        warmup_train_steps=1,
    )
    runtime = DeviceRuntimeSpec(
        backend=args.backend,
        cuda_device=args.cuda_device,
        dtype=args.dtype,
        primitive_runtime_backend="torch",
    )
    corpus = JsonlCorpusSpec(
        train_path=args.corpus_dir / "train.jsonl",
        eval_path=args.corpus_dir / "eval.jsonl",
        corpus_name="fineweb-stage0-local-bench-9row-v1",
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.ledger_path or output_dir / "ledger.jsonl"
    conditions = {
        "backend": args.backend,
        "dtype": args.dtype,
        "shape": f"d_model={args.d_model}, heads={args.heads}, layers={args.layers}, ffn_multiplier={args.ffn_multiplier}, vocab={args.vocab_size}",
        "train_steps": args.train_steps,
        "eval_batches": args.eval_batches,
        "seq_len": args.seq_len,
        "window_stride": args.window_stride,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_eval_batches": 1,
        "warmup_train_steps": 1,
        "corpus": corpus.corpus_name,
    }
    (output_dir / "suite_manifest.json").write_text(
        json.dumps(
            {
                "suite_name": SUITE_NAME,
                "conditions": conditions,
                "seeds": list(args.seeds),
                "lanes": [
                    {
                        "name": lane.name,
                        "family": lane.family,
                        "note": lane.note,
                        "variant": to_jsonable(lane.build_variant(shape)),
                    }
                    for lane in lanes
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for lane in lanes:
            variant = lane.build_variant(shape)
            run_label = f"{SUITE_NAME}-{lane.name}-s{seed}"
            report_path = output_dir / run_label / "report.json"
            if args.resume and report_path.exists():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                rows.append(
                    _row(
                        lane=lane, seed=seed, report_path=report_path, report=payload
                    )
                )
                continue
            manifest = BenchmarkRunManifest(
                run_label=run_label,
                implementation_kind=_implementation_kind_for_variant(
                    variant, primitive_runtime_backend="torch"
                ),
                benchmark_name=f"{SUITE_NAME}-{lane.name}",
                seed_spec=SeedSpec(model_seed=seed),
                corpus=corpus,
                budget=budget,
                runtime=runtime,
                note=lane.note,
            )
            print(f"RUN {lane.name} seed {seed}", flush=True)
            report = run_path1_variant(
                Path1RunnerRequest(
                    manifest=manifest,
                    variant=variant,
                    output_dir=output_dir,
                    output_format="table",
                    ledger_path=ledger_path,
                    variant_output_name=run_label,
                )
            )
            print(render_path1_table(report, variant.label), flush=True)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(
                _row(lane=lane, seed=seed, report_path=report_path, report=payload)
            )
    _write_summary(output_dir, rows, conditions)
    print((output_dir / "summary.md").resolve(), flush=True)


if __name__ == "__main__":
    main()
