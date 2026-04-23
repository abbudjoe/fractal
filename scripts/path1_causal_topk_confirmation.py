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
    Path1ScaffoldProfile,
    RecurrentHaltingProfile,
    TokenRoutingProfile,
    phase1_attention_only_variant,
)

SUITE_NAME = "path1-causal-topk-confirmation-v1"
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
            note="Standard local causal attention baseline.",
            build_variant=lambda shape: phase1_attention_only_variant(shape=shape),
        ),
        LaneSpec(
            name="plain-parcae-fixed3",
            family="control",
            note="Plain Parcae fixed recurrent-depth control.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="mod-train-topc",
            family="paper MoD train-time reference",
            note="Full-sequence top-C MoD training primitive; intentionally non-causal.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
            ),
        ),
        LaneSpec(
            name="p20-fixed5-proxy",
            family="Parcae-P20 proxy",
            note="Local Parcae P20-control fixed-depth proxy for the promoted RGRP lane.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
            ),
        ),
        LaneSpec(
            name="p20-smart5-proxy",
            family="Parcae-P20 proxy",
            note="Local Parcae P20-control proxy with vector-acceleration smart halting.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.55,
            ),
        ),
        LaneSpec(
            name="causal-topk-route50-layer1",
            family="decode-safe MoD",
            note="Decode-safe causal prefix-top-k routed block, route 50%, layer 1.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
            ),
        ),
        LaneSpec(
            name="causal-topk-route50-layer1-parcae-fixed3",
            family="decode-safe MoD + Parcae",
            note="Causal top-k routed middle block inside plain fixed Parcae recurrence.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
                scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="causal-topk-route50-layer1-p20-fixed5",
            family="decode-safe MoD + Parcae-P20",
            note="Causal top-k routed middle block inside fixed Parcae P20-control recurrence.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
            ),
        ),
        LaneSpec(
            name="causal-topk-route50-layer1-p20-smart5",
            family="decode-safe MoD + Parcae-P20",
            note="Causal top-k routed middle block inside smart-halting Parcae P20-control recurrence.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.55,
            ),
        ),
        LaneSpec(
            name="p20-mod-gate-bias-fixed5",
            family="MoD-conditioned Parcae-P20",
            note="Causal router logits bias the Parcae P20-control injection gate.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                token_route_fraction=0.5,
            ),
        ),
        LaneSpec(
            name="p20-mod-value-scale-fixed5",
            family="MoD-conditioned Parcae-P20",
            note="Causal router salience scales the Parcae P20-control value injection.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                token_route_fraction=0.5,
            ),
        ),
        LaneSpec(
            name="p20-mod-gate-bias-smart5",
            family="MoD-conditioned Parcae-P20",
            note="Causal router gate-bias control plus vector-acceleration smart halting.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_MOD_GATE_BIAS_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                token_route_fraction=0.5,
                recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.55,
            ),
        ),
        LaneSpec(
            name="p20-mod-value-scale-smart5",
            family="MoD-conditioned Parcae-P20",
            note="Causal router value-scale control plus vector-acceleration smart halting.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_MOD_VALUE_SCALE_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                token_route_fraction=0.5,
                recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.55,
            ),
        ),
        LaneSpec(
            name="p20-coda-topk-fixed5",
            family="MoD-coda Parcae-P20",
            note="Dense Parcae P20-control loop followed by a causal top-k coda block.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(2,),
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
            ),
        ),
        LaneSpec(
            name="p20-coda-topk-smart5",
            family="MoD-coda Parcae-P20",
            note="Dense smart-halting Parcae P20-control loop followed by a causal top-k coda block.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(2,),
                scaffold_profile=(
                    Path1ScaffoldProfile.PARCAE_P20_CONTROL_LOOPED_ATTENTION
                ),
                parcae_loop_count=5,
                recurrent_halting_profile=RecurrentHaltingProfile.VECTOR_ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.55,
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


def _token_selected_fraction(diagnostics: dict[str, object]) -> float | None:
    token_routing = diagnostics.get("token_routing")
    if not isinstance(token_routing, dict):
        parcae = diagnostics.get("parcae_looped_attention")
        if not isinstance(parcae, dict):
            return None
        control = parcae.get("control_diagnostics")
        if not isinstance(control, dict):
            return None
        value = control.get("mod_router/selected_fraction")
        return float(value) if isinstance(value, (int, float)) else None
    blocks = token_routing.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return None
    fractions: list[float] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        payload = block.get("payload")
        if not isinstance(payload, dict):
            continue
        value = payload.get("last_selected_fraction")
        if isinstance(value, (int, float)):
            fractions.append(float(value))
    return sum(fractions) / len(fractions) if fractions else None


def _parcae_steps(diagnostics: dict[str, object]) -> float | None:
    payload = diagnostics.get("parcae_looped_attention")
    if not isinstance(payload, dict):
        return None
    value = payload.get("average_steps_used")
    return float(value) if isinstance(value, (int, float)) else None


def _row(
    *, lane: LaneSpec, seed: int, report_path: Path, report: dict[str, object]
) -> dict[str, object]:
    diagnostics = _diagnostics(report)
    runtime = report["runtime"]
    initial_eval = report["initial_eval"]
    final_eval = report["final_eval"]
    if not isinstance(runtime, dict):
        raise TypeError("report runtime must be a dict")
    if not isinstance(initial_eval, dict) or not isinstance(final_eval, dict):
        raise TypeError("report eval payloads must be dicts")
    return {
        "lane": lane.name,
        "family": lane.family,
        "seed": seed,
        "initial_loss": initial_eval["mean_loss"],
        "final_loss": final_eval["mean_loss"],
        "train_tokens_per_second": runtime["train_tokens_per_second"],
        "overall_tokens_per_second": runtime["overall_tokens_per_second"],
        "parameter_count": diagnostics.get("parameter_count"),
        "average_steps_used": _parcae_steps(diagnostics),
        "selected_token_fraction": _token_selected_fraction(diagnostics),
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


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lanes = sorted({str(row["lane"]) for row in rows})
    aggregate: list[dict[str, object]] = []
    for lane in lanes:
        group = [row for row in rows if row["lane"] == lane]
        losses = [float(row["final_loss"]) for row in group]
        speeds = [float(row["train_tokens_per_second"]) for row in group]
        steps = [
            float(row["average_steps_used"])
            for row in group
            if isinstance(row["average_steps_used"], (int, float))
        ]
        selected = [
            float(row["selected_token_fraction"])
            for row in group
            if isinstance(row["selected_token_fraction"], (int, float))
        ]
        aggregate.append(
            {
                "lane": lane,
                "family": group[0]["family"],
                "mean_final_loss": _mean(losses),
                "std_final_loss": _std(losses),
                "mean_train_tokens_per_second": _mean(speeds),
                "parameter_count": group[0]["parameter_count"],
                "mean_average_steps_used": _mean(steps) if steps else None,
                "mean_selected_token_fraction": _mean(selected) if selected else None,
                "seeds": [row["seed"] for row in group],
            }
        )
    aggregate.sort(key=lambda row: float(row["mean_final_loss"]))
    return aggregate


def _write_summary(output_dir: Path, rows: list[dict[str, object]], conditions: dict[str, object]) -> None:
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
        "Formal local confirmation screen for the decode-safe causal prefix-top-k MoD lane and first Parcae hybrids.",
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
            "| Lane | Family | Loss | Loss Std | Tok/s | Params | Steps | Selected |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate:
        steps = row["mean_average_steps_used"]
        selected = row["mean_selected_token_fraction"]
        lines.append(
            "| {lane} | {family} | {loss:.4f} | {std:.4f} | {speed:.1f} | {params} | {steps} | {selected} |".format(
                lane=row["lane"],
                family=row["family"],
                loss=row["mean_final_loss"],
                std=row["std_final_loss"],
                speed=row["mean_train_tokens_per_second"],
                params=row["parameter_count"],
                steps=f"{steps:.2f}" if isinstance(steps, float) else "",
                selected=f"{selected:.3f}" if isinstance(selected, float) else "",
            )
        )
    seed_values = sorted({int(row["seed"]) for row in rows})
    lines.extend(["", "## Per-Seed Final Loss", ""])
    lines.append("| Lane | " + " | ".join(f"Seed {seed}" for seed in seed_values) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in seed_values) + " |")
    for lane in sorted({str(row["lane"]) for row in rows}):
        by_seed = {int(row["seed"]): float(row["final_loss"]) for row in rows if row["lane"] == lane}
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
        description="Run a matched Path1 confirmation screen for causal top-k MoD routing."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts") / SUITE_NAME)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(42, 43, 44, 45, 46))
    parser.add_argument("--lanes", help="Comma or whitespace separated lane names.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--train-steps", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    lane_names = set(args.lanes.replace(",", " ").split()) if args.lanes else None
    lanes = _selected_lanes(lane_names)
    shape = Path1ModelShape(
        d_model=32,
        head_count=4,
        total_layers=4,
        ffn_multiplier=2,
    )
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
    rows: list[dict[str, object]] = []
    conditions = {
        "backend": args.backend,
        "dtype": args.dtype,
        "shape": "d_model=32, heads=4, layers=4, ffn_multiplier=2",
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

    for seed in args.seeds:
        for lane in lanes:
            variant = lane.build_variant(shape)
            run_label = f"{SUITE_NAME}-{lane.name}-s{seed}"
            report_path = output_dir / run_label / "report.json"
            if args.resume and report_path.exists():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                rows.append(_row(lane=lane, seed=seed, report_path=report_path, report=payload))
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
            rows.append(_row(lane=lane, seed=seed, report_path=report_path, report=payload))
    _write_summary(output_dir, rows, conditions)
    print((output_dir / "summary.md").resolve(), flush=True)


if __name__ == "__main__":
    main()
