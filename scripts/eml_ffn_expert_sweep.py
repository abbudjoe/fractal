#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.runners.path1 import Path1RunnerRequest, run_path1_variant
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
    TokenIdCorpusSpec,
    to_jsonable,
)
from python.specs.path1 import FeedForwardProfile, Path1ModelShape, Path1ScaffoldProfile, phase1_attention_only_variant


@dataclass(frozen=True)
class ExpertLane:
    slug: str
    profile: FeedForwardProfile
    slot_count: int | None = None
    tree_depth: int | None = None
    route_fraction: float = 0.25


PHASE1_EXPERTS: tuple[ExpertLane, ...] = (
    ExpertLane("dense-eml", FeedForwardProfile.MLP_EML_GATED),
    ExpertLane("routed-eml-r25", FeedForwardProfile.MLP_EML_ROUTED, route_fraction=0.25),
    ExpertLane("tiny-mlp", FeedForwardProfile.TINY_MLP_GATED),
    ExpertLane("tiny-glu", FeedForwardProfile.TINY_GLU_GATED),
    ExpertLane("generic-tree", FeedForwardProfile.GENERIC_TREE_GATED),
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(token.strip()) for token in value.replace(",", " ").split() if token.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in value.replace(",", " ").split() if token.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded EML FFN-side expert sweep through Path 1.")
    parser.add_argument("--phase", default="phase1", choices=["phase1", "phase2"])
    parser.add_argument("--backend", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--slot-count", type=int, default=4)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--slot-counts", default="4,8")
    parser.add_argument("--tree-depths", default="2,3")
    parser.add_argument("--route-fractions", default="0.10,0.25,0.50")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--total-layers", type=int, default=8)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument(
        "--scaffold-profile",
        default=Path1ScaffoldProfile.STANDARD.value,
        choices=[profile.value for profile in Path1ScaffoldProfile],
    )
    parser.add_argument("--parcae-loop-count", type=int, default=2)
    parser.add_argument(
        "--jsonl-train-path",
        type=Path,
        default=REPO_ROOT / "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl",
    )
    parser.add_argument(
        "--jsonl-eval-path",
        type=Path,
        default=REPO_ROOT / "experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl",
    )
    parser.add_argument("--corpus-name", default="fineweb-stage0-local-bench-9row-v1")
    parser.add_argument("--tokenized-manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "eml-ffn-expert-sweep")
    parser.add_argument("--run-id")
    return parser


def _expert_lanes(args: argparse.Namespace) -> tuple[ExpertLane, ...]:
    if args.phase == "phase1":
        return tuple(
            ExpertLane(
                lane.slug,
                lane.profile,
                slot_count=args.slot_count,
                tree_depth=args.tree_depth,
                route_fraction=lane.route_fraction,
            )
            for lane in PHASE1_EXPERTS
        )

    lanes: list[ExpertLane] = [
        ExpertLane("tiny-mlp", FeedForwardProfile.TINY_MLP_GATED, slot_count=args.slot_count, tree_depth=args.tree_depth),
        ExpertLane("tiny-glu", FeedForwardProfile.TINY_GLU_GATED, slot_count=args.slot_count, tree_depth=args.tree_depth),
    ]
    for slot_count in _parse_csv_ints(args.slot_counts):
        for tree_depth in _parse_csv_ints(args.tree_depths):
            lanes.append(
                ExpertLane(
                    f"dense-eml-s{slot_count}-d{tree_depth}",
                    FeedForwardProfile.MLP_EML_GATED,
                    slot_count=slot_count,
                    tree_depth=tree_depth,
                )
            )
            lanes.append(
                ExpertLane(
                    f"generic-tree-s{slot_count}-d{tree_depth}",
                    FeedForwardProfile.GENERIC_TREE_GATED,
                    slot_count=slot_count,
                    tree_depth=tree_depth,
                )
            )
            for route_fraction in _parse_csv_floats(args.route_fractions):
                lanes.append(
                    ExpertLane(
                        f"routed-eml-s{slot_count}-d{tree_depth}-r{int(round(route_fraction * 100))}",
                        FeedForwardProfile.MLP_EML_ROUTED,
                        slot_count=slot_count,
                        tree_depth=tree_depth,
                        route_fraction=route_fraction,
                    )
                )
    return tuple(lanes)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stdev(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _report_row(
    *,
    run_label: str,
    seed: int,
    expert: str,
    layer: int | None,
    report: Any,
) -> dict[str, Any]:
    diagnostics = report.diagnostics or {}
    runtime = report.runtime
    memory_mb = runtime.peak_process_memory_bytes / (1024 * 1024)
    cuda_memory = runtime.cuda_device_memory
    if cuda_memory is not None:
        memory_mb = cuda_memory["peak_used_bytes"] / (1024 * 1024)
    return {
        "run_label": run_label,
        "seed": seed,
        "expert": expert,
        "layer": layer,
        "slot_count": (report.diagnostics or {}).get("eml_inspired_feed_forward", {}).get("slot_count"),
        "tree_depth": (report.diagnostics or {}).get("eml_inspired_feed_forward", {}).get("tree_depth"),
        "route_fraction": (report.diagnostics or {}).get("eml_inspired_feed_forward", {}).get("route_fraction"),
        "model_label": report.model_label,
        "parameter_count": diagnostics.get("parameter_count"),
        "initial_loss": report.initial_eval.mean_loss,
        "final_loss": report.final_eval.mean_loss,
        "train_tokens_per_second": runtime.train_tokens_per_second,
        "peak_memory_mb": memory_mb,
        "report_path": report.report_path,
        "diagnostics": diagnostics,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None, int | None, int | None, float | None], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                row["expert"],
                row["layer"],
                row.get("slot_count"),
                row.get("tree_depth"),
                row.get("route_fraction"),
            ),
            [],
        ).append(row)
    aggregated: list[dict[str, Any]] = []
    for (expert, layer, slot_count, tree_depth, route_fraction), group_rows in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
            -1 if item[0][2] is None else item[0][2],
            -1 if item[0][3] is None else item[0][3],
            -1.0 if item[0][4] is None else item[0][4],
        ),
    ):
        losses = [float(row["final_loss"]) for row in group_rows]
        toks = [float(row["train_tokens_per_second"]) for row in group_rows]
        mem = [float(row["peak_memory_mb"]) for row in group_rows]
        aggregated.append(
            {
                "expert": expert,
                "layer": layer,
                "slot_count": slot_count,
                "tree_depth": tree_depth,
                "route_fraction": route_fraction,
                "runs": len(group_rows),
                "mean_final_loss": _mean(losses),
                "stdev_final_loss": _stdev(losses),
                "mean_train_tokens_per_second": _mean(toks),
                "mean_peak_memory_mb": _mean(mem),
                "parameter_count": group_rows[0].get("parameter_count"),
            }
        )
    return aggregated


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for row in rows:
        loss = row["mean_final_loss"]
        speed = row["mean_train_tokens_per_second"]
        dominated = any(
            other["mean_final_loss"] <= loss
            and other["mean_train_tokens_per_second"] >= speed
            and (
                other["mean_final_loss"] < loss
                or other["mean_train_tokens_per_second"] > speed
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: item["mean_final_loss"])


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    aggregated = _aggregate(rows)
    frontier = _pareto(aggregated)
    payload = {
        "args": to_jsonable(vars(args)),
        "rows": rows,
        "aggregated": aggregated,
        "pareto_frontier": frontier,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    baseline = next((row for row in aggregated if row["expert"] == "baseline"), None)
    baseline_loss = baseline["mean_final_loss"] if baseline else None
    lines = [
        "# EML FFN Expert Sweep Summary",
        "",
        f"- backend: `{args.backend}`",
        f"- seeds: `{args.seeds}`",
        f"- steps: `{args.steps}`",
        f"- seq_len: `{args.seq_len}`",
        f"- batch_size: `{args.batch_size}`",
        f"- slot_count: `{args.slot_count}`",
        f"- tree_depth: `{args.tree_depth}`",
        "",
        "## Aggregated Results",
        "",
        "| expert | layer | runs | mean loss | delta vs baseline | tok/s | peak MB | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregated, key=lambda item: item["mean_final_loss"]):
        delta = ""
        if baseline_loss is not None:
            delta = f"{row['mean_final_loss'] - baseline_loss:+.4f}"
        layer = "all" if row["layer"] is None else str(row["layer"])
        expert_label = row["expert"]
        if row.get("slot_count") is not None and row.get("tree_depth") is not None:
            expert_label = f"{expert_label}/s{row['slot_count']}/d{row['tree_depth']}"
        if row.get("route_fraction") is not None and row["expert"].startswith("routed"):
            expert_label = f"{expert_label}/r{row['route_fraction']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    expert_label,
                    layer,
                    str(row["runs"]),
                    f"{row['mean_final_loss']:.4f}",
                    delta,
                    f"{row['mean_train_tokens_per_second']:.2f}",
                    f"{row['mean_peak_memory_mb']:.2f}",
                    str(row["parameter_count"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Pareto Frontier", ""])
    for row in frontier:
        layer = "all" if row["layer"] is None else str(row["layer"])
        expert_label = row["expert"]
        if row.get("slot_count") is not None and row.get("tree_depth") is not None:
            expert_label = f"{expert_label}/s{row['slot_count']}/d{row['tree_depth']}"
        if row.get("route_fraction") is not None and row["expert"].startswith("routed"):
            expert_label = f"{expert_label}/r{row['route_fraction']}"
        lines.append(
            f"- `{expert_label}` layer `{layer}`: loss `{row['mean_final_loss']:.4f}`, "
            f"tok/s `{row['mean_train_tokens_per_second']:.2f}`"
        )
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    inferred_vocab_size = args.vocab_size
    if args.tokenized_manifest_path is not None:
        manifest_payload = json.loads(args.tokenized_manifest_path.read_text(encoding="utf-8"))
        tokenizer_payload = manifest_payload.get("tokenizer", {})
        manifest_vocab_size = tokenizer_payload.get("vocab_size")
        if not isinstance(manifest_vocab_size, int) or manifest_vocab_size <= 0:
            raise SystemExit("tokenized manifest tokenizer.vocab_size must be a positive integer")
        if inferred_vocab_size is not None and inferred_vocab_size != manifest_vocab_size:
            raise SystemExit(
                f"--vocab-size {inferred_vocab_size} does not match tokenized manifest vocab_size {manifest_vocab_size}"
            )
        inferred_vocab_size = manifest_vocab_size
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "ledger.jsonl"

    seeds = _parse_csv_ints(args.seeds)
    layers = tuple(range(args.total_layers)) if args.layers == "all" else _parse_csv_ints(args.layers)
    shape = Path1ModelShape(
        vocab_size=inferred_vocab_size or 257,
        d_model=args.d_model,
        head_count=args.head_count,
        total_layers=args.total_layers,
        local_window=args.local_window,
        ffn_multiplier=args.ffn_multiplier,
    )
    if args.tokenized_manifest_path is not None:
        corpus = TokenIdCorpusSpec(manifest_path=args.tokenized_manifest_path)
    else:
        corpus = JsonlCorpusSpec(
            train_path=args.jsonl_train_path,
            eval_path=args.jsonl_eval_path,
            corpus_name=args.corpus_name,
        )
    budget = BenchmarkBudgetSpec(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_steps=args.steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        warmup_train_steps=args.warmup_train_steps,
        warmup_eval_batches=args.warmup_eval_batches,
    )
    runtime = DeviceRuntimeSpec(
        backend=args.backend,
        cuda_device=args.cuda_device,
        dtype=args.dtype,
        env_kind="requirements-only",
        primitive_runtime_backend="torch",
    )

    rows: list[dict[str, Any]] = []
    expert_lanes = _expert_lanes(args)
    scaffold_profile = Path1ScaffoldProfile(args.scaffold_profile)
    for seed in seeds:
        baseline = phase1_attention_only_variant(
            shape=shape,
            scaffold_profile=scaffold_profile,
            parcae_loop_count=args.parcae_loop_count,
        )
        run_label = f"{args.phase}-baseline-s{seed}"
        report = run_path1_variant(
            Path1RunnerRequest(
                manifest=BenchmarkRunManifest(
                    run_label=run_label,
                    implementation_kind=f"python_attention_{args.backend}",
                    benchmark_name="eml-ffn-expert-sweep",
                    seed_spec=SeedSpec(model_seed=seed, data_seed=None),
                    corpus=corpus,
                    budget=budget,
                    runtime=runtime,
                ),
                variant=baseline,
                output_dir=output_dir / "reports",
                output_format="table",
                ledger_path=ledger_path,
                variant_output_name=run_label,
            )
        )
        rows.append(_report_row(run_label=run_label, seed=seed, expert="baseline", layer=None, report=report))

        for layer in layers:
            for expert in expert_lanes:
                variant = phase1_attention_only_variant(
                    shape=shape,
                    feed_forward_profile=expert.profile,
                    feed_forward_layer_indices=(layer,),
                    eml_slot_count=expert.slot_count or args.slot_count,
                    eml_tree_depth=expert.tree_depth or args.tree_depth,
                    eml_route_fraction=expert.route_fraction,
                    scaffold_profile=scaffold_profile,
                    parcae_loop_count=args.parcae_loop_count,
                )
                run_label = f"{args.phase}-{expert.slug}-L{layer}-s{seed}"
                report = run_path1_variant(
                    Path1RunnerRequest(
                        manifest=BenchmarkRunManifest(
                            run_label=run_label,
                            implementation_kind=f"python_attention_{expert.slug}_{args.backend}",
                            benchmark_name="eml-ffn-expert-sweep",
                            seed_spec=SeedSpec(model_seed=seed, data_seed=None),
                            corpus=corpus,
                            budget=budget,
                            runtime=runtime,
                        ),
                        variant=variant,
                        output_dir=output_dir / "reports",
                        output_format="table",
                        ledger_path=ledger_path,
                        variant_output_name=run_label,
                    )
                )
                rows.append(_report_row(run_label=run_label, seed=seed, expert=expert.slug, layer=layer, report=report))
                _write_summary(output_dir, rows, args)

    _write_summary(output_dir, rows, args)
    print(f"summary={output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
