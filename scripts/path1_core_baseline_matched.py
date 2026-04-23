#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    AttentionProfile,
    Path1ModelShape,
    Path1ScaffoldProfile,
    RecurrentHaltingProfile,
    TokenRoutingProfile,
    phase1_attention_only_variant,
)

SUITE_NAME = "path1-core-baseline-matched-v1"
DEFAULT_CORPUS = Path("experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1")


@dataclass(frozen=True)
class LaneSpec:
    name: str
    benchmark_name: str
    fidelity: str
    concepts: tuple[str, ...]
    note: str
    build_variant: Callable[[Path1ModelShape], object]


def _base_lanes() -> tuple[LaneSpec, ...]:
    return (
        LaneSpec(
            name="attention-control",
            benchmark_name="path1-core-baseline-attention-control",
            fidelity="existing repo baseline",
            concepts=("pure attention",),
            note="Decoder-only local causal attention control.",
            build_variant=lambda shape: phase1_attention_only_variant(shape=shape),
        ),
        LaneSpec(
            name="moda-paper-depth-kv",
            benchmark_name="path1-core-baseline-moda-paper-ref",
            fidelity="paper-faithful slow attention-side reference",
            concepts=("MoDA", "depth KV"),
            note=(
                "Same-token prior-depth sequence KV reference; no Flash MoDA kernel "
                "or FFN-side lane."
            ),
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                attention_profile=AttentionProfile.PAPER_MODA_DEPTH_KV,
                depth_memory_layers=2,
            ),
        ),
        LaneSpec(
            name="mod-train-topc",
            benchmark_name="path1-core-baseline-mod-paper-train",
            fidelity="paper-faithful training-time top-C reference",
            concepts=("MoD", "token routing"),
            note="Full-sequence top-C MoD training primitive; intentionally non-causal.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
            ),
        ),
        LaneSpec(
            name="mod-causal-decode-control",
            benchmark_name="path1-core-baseline-mod-causal-decode",
            fidelity="causal practical approximation",
            concepts=("MoD", "causal token routing"),
            note="Causal prefix-top-k routing control for decode-safe comparison.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                token_routing_profile=TokenRoutingProfile.CAUSAL_TOPK_BLOCK,
                token_route_fraction=0.5,
                token_routing_layer_indices=(1,),
            ),
        ),
        LaneSpec(
            name="parcae-fixed-recurrent-depth",
            benchmark_name="path1-core-baseline-recurrent-depth-fixed",
            fidelity="practical recurrent-depth approximation",
            concepts=("recurrent depth", "latent refinement"),
            note="Middle block-group recurrence with a fixed loop count.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="parcae-acceleration-exit",
            benchmark_name="path1-core-baseline-recurrent-depth-accel-exit",
            fidelity="practical hidden-dynamics halting approximation",
            concepts=("recurrent depth", "acceleration halting"),
            note="Acceleration-style hidden-state early exit over the Parcae loop.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION,
                parcae_loop_count=3,
                recurrent_halting_profile=RecurrentHaltingProfile.ACCELERATION,
                recurrent_min_steps=2,
                recurrent_halting_threshold=0.6,
            ),
        ),
        LaneSpec(
            name="fixed-looped-lm",
            benchmark_name="path1-core-baseline-looped-lm",
            fidelity="paper-faithful core loop scaffold",
            concepts=("looped transformer", "fixed recurrence"),
            note="Shared k-layer decoder block group repeated for fixed latent compute.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.FIXED_LOOPED_LM,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="looped-additive-input",
            benchmark_name="path1-core-baseline-input-injected-loop",
            fidelity="paper-faithful recurrence equation, LM-adapted",
            concepts=("looped transformer", "input injection"),
            note="Input-injected loop profile implementing Y <- M(Y + P).",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="huginn-adapter-recurrence",
            benchmark_name="path1-core-baseline-huginn-adapter",
            fidelity="practical adapter-surface reference",
            concepts=("recurrent depth", "input-injected adapter"),
            note=(
                "Deterministic zero-state concat adapter R(e, s); sampled-depth "
                "Huginn recipe deferred."
            ),
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="universal-transformer",
            benchmark_name="path1-core-baseline-universal-transformer",
            fidelity="paper-faithful core recurrence approximation",
            concepts=("Universal Transformer", "coordinate recurrence"),
            note="Tied transition with per-step sinusoidal position/time coordinates.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER,
                parcae_loop_count=3,
            ),
        ),
        LaneSpec(
            name="universal-transformer-act",
            benchmark_name="path1-core-baseline-ut-act",
            fidelity="paper-faithful ACT reference surface",
            concepts=("Universal Transformer", "ACT"),
            note="Weighted per-token ACT halting with ponder auxiliary loss.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT,
                parcae_loop_count=3,
                act_halting_threshold=0.99,
                act_ponder_loss_weight=0.01,
            ),
        ),
        LaneSpec(
            name="ouro-stage1-learned-exit",
            benchmark_name="path1-core-baseline-ouro-exit",
            fidelity="paper-faithful Stage 1 approximation",
            concepts=("Ouro", "learned exit distribution"),
            note=(
                "Expected CE over per-step logits plus entropy auxiliary; Stage 2 "
                "gate training deferred."
            ),
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.OURO_LEARNED_EXIT,
                parcae_loop_count=3,
                ouro_entropy_weight=0.05,
                ouro_q_exit_threshold=0.5,
            ),
        ),
        LaneSpec(
            name="rrt-cycle",
            benchmark_name="path1-core-baseline-rrt-cycle",
            fidelity="paper-faithful strict recursion baseline",
            concepts=("Relaxed Recursive Transformer", "CYCLE sharing"),
            note="Strict CYCLE sharing with LoRA/SVD relaxation deferred.",
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.RRT_CYCLE,
                parcae_loop_count=2,
            ),
        ),
        LaneSpec(
            name="mor-expert-choice",
            benchmark_name="path1-core-baseline-mor-expert-choice",
            fidelity="paper-faithful train/eval reference surface",
            concepts=("Mixture of Recursions", "expert-choice routing"),
            note=(
                "Full-sequence expert-choice active-token shrinkage; not a "
                "decode-safe path."
            ),
            build_variant=lambda shape: phase1_attention_only_variant(
                shape=shape,
                scaffold_profile=Path1ScaffoldProfile.MOR_EXPERT_CHOICE,
                parcae_loop_count=3,
                recurrent_token_route_fraction=0.5,
                mor_router_aux_loss_weight=0.01,
                mor_update_scale=0.1,
            ),
        ),
    )


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for chunk in value.replace(",", " ").split():
        seeds.append(int(chunk))
    if not seeds:
        raise argparse.ArgumentTypeError("must provide at least one seed")
    return tuple(seeds)


def _selected_lanes(lane_names: set[str] | None) -> tuple[LaneSpec, ...]:
    lanes = _base_lanes()
    if lane_names is None:
        return lanes
    unknown = sorted(lane_names - {lane.name for lane in lanes})
    if unknown:
        raise SystemExit(f"unknown lane(s): {', '.join(unknown)}")
    return tuple(lane for lane in lanes if lane.name in lane_names)


def _diagnostic_brief(report: dict[str, object]) -> str:
    diagnostics = report.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return ""
    if "parcae_looped_attention" in diagnostics:
        payload = diagnostics["parcae_looped_attention"]
        if isinstance(payload, dict):
            return f"avg steps {payload.get('average_steps_used')}"
    if "looped_transformer" in diagnostics:
        payload = diagnostics["looped_transformer"]
        if isinstance(payload, dict):
            return f"effective layers {payload.get('effective_layer_count')}"
    if "universal_transformer" in diagnostics:
        payload = diagnostics["universal_transformer"]
        if isinstance(payload, dict):
            if payload.get("act_enabled"):
                act = payload.get("act", {})
                if isinstance(act, dict):
                    return f"ACT mean updates {act.get('update_count_mean')}"
            return f"effective layers {payload.get('effective_layer_count')}"
    if "ouro_learned_exit" in diagnostics:
        payload = diagnostics["ouro_learned_exit"]
        if isinstance(payload, dict):
            return (
                f"q-exit {payload.get('q_exit_step_mean')}, "
                f"expected exit {payload.get('expected_exit_step_mean')}"
            )
    if "rrt_cycle" in diagnostics:
        payload = diagnostics["rrt_cycle"]
        if isinstance(payload, dict):
            return (
                f"stored/effective {payload.get('stored_layer_count')}/"
                f"{payload.get('effective_layer_count')}"
            )
    if "mor_expert_choice" in diagnostics:
        payload = diagnostics["mor_expert_choice"]
        if isinstance(payload, dict):
            return (
                f"active {payload.get('last_active_token_counts')}, "
                f"selected {payload.get('last_selected_token_counts')}"
            )
    if "depth_augmented_attention" in diagnostics:
        return "depth attention"
    if "token_routing" in diagnostics:
        return "token routing"
    return "control"


def _report_row(
    *,
    lane: LaneSpec,
    seed: int,
    report_path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    runtime = payload["runtime"]
    initial_eval = payload["initial_eval"]
    final_eval = payload["final_eval"]
    diagnostics = payload.get("diagnostics", {})
    if not isinstance(runtime, dict) or not isinstance(initial_eval, dict):
        raise TypeError("report payload is missing runtime/eval dictionaries")
    if not isinstance(final_eval, dict) or not isinstance(diagnostics, dict):
        raise TypeError("report payload is missing final_eval/diagnostics")
    return {
        "lane": lane.name,
        "seed": seed,
        "variant": payload["config"]["variant"]["label"],
        "fidelity": lane.fidelity,
        "initial_loss": initial_eval["mean_loss"],
        "final_loss": final_eval["mean_loss"],
        "train_tokens_per_second": runtime["train_tokens_per_second"],
        "overall_tokens_per_second": runtime["overall_tokens_per_second"],
        "peak_process_memory_bytes": runtime["peak_process_memory_bytes"],
        "parameter_count": diagnostics.get("parameter_count"),
        "diagnostic_brief": _diagnostic_brief(payload),
        "report_path": str(report_path.resolve()),
    }


def _write_suite_manifest(
    *,
    path: Path,
    lanes: Iterable[LaneSpec],
    seeds: tuple[int, ...],
    shape: Path1ModelShape,
    budget: BenchmarkBudgetSpec,
    runtime: DeviceRuntimeSpec,
    corpus: JsonlCorpusSpec,
    run_prefix: str,
) -> None:
    lane_payloads = []
    for lane in lanes:
        variant = lane.build_variant(shape)
        lane_payloads.append(
            {
                "name": lane.name,
                "benchmark_name": lane.benchmark_name,
                "fidelity": lane.fidelity,
                "concepts": list(lane.concepts),
                "note": lane.note,
                "implementation_kind": _implementation_kind_for_variant(
                    variant,
                    primitive_runtime_backend=runtime.primitive_runtime_backend
                    or "torch",
                ),
                "variant": to_jsonable(variant),
            }
        )
    payload = {
        "schema_version": 1,
        "suite_name": SUITE_NAME,
        "run_prefix": run_prefix,
        "comparison_scope": (
            "Local compact matched Path 1 core-baseline run. All lanes share "
            "corpus, shape, optimizer budget, runtime, and seed list unless a "
            "paper primitive requires a profile-specific loop/routing control."
        ),
        "seeds": list(seeds),
        "shape": to_jsonable(shape),
        "budget": to_jsonable(budget),
        "runtime": to_jsonable(runtime),
        "corpus": to_jsonable(corpus),
        "lanes": lane_payloads,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    summary_json = {
        "suite_name": SUITE_NAME,
        "row_count": len(rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {SUITE_NAME}",
        "",
        "| Lane | Seed | Params | Initial Loss | Final Loss | Train Tok/s | Overall Tok/s | Peak RSS | Diagnostic |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {lane} | {seed} | {parameter_count} | {initial_loss:.4f} | "
            "{final_loss:.4f} | {train_tokens_per_second:.2f} | "
            "{overall_tokens_per_second:.2f} | {peak_process_memory_bytes} | "
            "{diagnostic_brief} |".format(**row)
        )
    lines.extend(
        [
            "",
            "These are matched local proving-ground runs, not promotion-grade results.",
            "Use the per-lane report paths in `summary.json` for full diagnostics.",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched Path 1 core-baseline manifests through the existing harness."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/core-baseline-matched-v1")
    )
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-prefix", default=SUITE_NAME)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=(42,))
    parser.add_argument(
        "--lanes",
        help="Comma or whitespace separated lane names. Defaults to all lanes.",
    )
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--total-layers", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=2)
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--corpus-name", default="fineweb-stage0-local-bench-9row-v1")
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip lanes whose report.json already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lane_names = None
    if args.lanes:
        lane_names = set(args.lanes.replace(",", " ").split())
    lanes = _selected_lanes(lane_names)
    shape = Path1ModelShape(
        d_model=args.d_model,
        head_count=args.head_count,
        total_layers=args.total_layers,
        local_window=args.local_window,
        ffn_multiplier=args.ffn_multiplier,
    )
    budget = BenchmarkBudgetSpec(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_steps=args.steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        warmup_eval_batches=args.warmup_eval_batches,
        warmup_train_steps=args.warmup_train_steps,
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
        corpus_name=args.corpus_name,
    )
    output_dir = args.output_dir
    ledger_path = args.ledger_path or output_dir / "ledger.jsonl"
    suite_manifest_path = output_dir / "suite_manifest.json"
    _write_suite_manifest(
        path=suite_manifest_path,
        lanes=lanes,
        seeds=args.seeds,
        shape=shape,
        budget=budget,
        runtime=runtime,
        corpus=corpus,
        run_prefix=args.run_prefix,
    )
    if args.dry_run:
        print(suite_manifest_path)
        return

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for lane in lanes:
            variant = lane.build_variant(shape)
            run_label = f"{args.run_prefix}-{lane.name}-s{seed}"
            report_path = output_dir / run_label / "report.json"
            if args.resume and report_path.exists():
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                rows.append(
                    _report_row(
                        lane=lane, seed=seed, report_path=report_path, payload=payload
                    )
                )
                continue
            manifest = BenchmarkRunManifest(
                run_label=run_label,
                implementation_kind=_implementation_kind_for_variant(
                    variant,
                    primitive_runtime_backend=runtime.primitive_runtime_backend
                    or "torch",
                ),
                benchmark_name=lane.benchmark_name,
                seed_spec=SeedSpec(model_seed=seed),
                corpus=corpus,
                budget=budget,
                runtime=runtime,
                note=f"{lane.fidelity}. {lane.note}",
            )
            report = run_path1_variant(
                Path1RunnerRequest(
                    manifest=manifest,
                    variant=variant,
                    output_dir=output_dir,
                    output_format=args.output,
                    ledger_path=ledger_path,
                    variant_output_name=run_label,
                )
            )
            print(render_path1_table(report, variant.label), flush=True)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append(
                _report_row(
                    lane=lane, seed=seed, report_path=report_path, payload=payload
                )
            )
    _write_summary(output_dir, rows)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
