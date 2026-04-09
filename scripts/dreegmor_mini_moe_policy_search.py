#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.runners.mini_moe_policy_search import (  # noqa: E402
    MiniMoePolicySearchCandidate,
    MiniMoePolicySearchRequest,
    run_mini_moe_policy_search,
)
from python.specs.common import (  # noqa: E402
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
)
from python.specs.mini_moe import (  # noqa: E402
    MiniMoeSurfaceSpec,
    contiguous_layer_bands,
    transfer_round2_layer_bands_by_anchor_fill,
    transfer_round2_layer_bands_by_scaled_span,
    transfer_round2_layer_indices_by_depth_fraction,
)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a structured mini-MoE policy search.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--jsonl-train-path", type=Path, required=True)
    parser.add_argument("--jsonl-eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--benchmark-name", default="dreegmor-mini-moe-policy-search")
    parser.add_argument("--corpus-name", default="fineweb-stage0-canary")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--experts-per-block", type=int, default=8)
    parser.add_argument("--total-layers", type=int, default=16)
    parser.add_argument("--source-total-layers", type=int, default=8)
    parser.add_argument("--source-round2-layers", default="2,3,4,6,7")
    parser.add_argument("--entropy-threshold", type=float, default=0.95)
    return parser.parse_args()


def _manifest(args: argparse.Namespace) -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        run_label=args.run_label,
        implementation_kind="python_native",
        benchmark_name=args.benchmark_name,
        note="Structured mini-MoE policy search over typed recurrent-routing candidates",
        seed_spec=SeedSpec(model_seed=0),
        corpus=JsonlCorpusSpec(
            train_path=args.jsonl_train_path,
            eval_path=args.jsonl_eval_path,
            corpus_name=args.corpus_name,
            text_field=args.corpus_text_field,
        ),
        budget=BenchmarkBudgetSpec(
            seq_len=args.seq_len,
            window_stride=args.window_stride,
            batch_size=args.batch_size,
            train_steps=args.steps,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            warmup_eval_batches=args.warmup_eval_batches,
            warmup_train_steps=args.warmup_train_steps,
        ),
        runtime=DeviceRuntimeSpec(backend=args.backend, dtype=args.dtype),
    )


def _build_surface(
    *,
    kind: str,
    total_layers: int,
    experts_per_block: int,
    normalized_entropy_threshold: float,
    round2_layer_indices: tuple[int, ...] | None = None,
) -> MiniMoeSurfaceSpec:
    if kind == "reference":
        surface = MiniMoeSurfaceSpec.phase1_reference_default()
    else:
        surface = MiniMoeSurfaceSpec.phase1_recurrent_entropy_gated_default(
            normalized_entropy_threshold=normalized_entropy_threshold,
            round2_layer_indices=round2_layer_indices,
        )
    architecture = replace(
        surface.architecture,
        label=f"{surface.architecture.label}-e{experts_per_block}-d{total_layers}",
        backbone=replace(surface.architecture.backbone, total_layers=total_layers),
        moe=replace(surface.architecture.moe, experts_per_block=experts_per_block),
    )
    observability = replace(surface.observability, max_token_route_traces_per_layer=0)
    return replace(surface, architecture=architecture, observability=observability)


def _candidates(args: argparse.Namespace) -> tuple[MiniMoePolicySearchCandidate, ...]:
    source_mask = _parse_int_tuple(args.source_round2_layers)
    point_transfer = transfer_round2_layer_indices_by_depth_fraction(
        source_layer_indices=source_mask,
        source_total_layers=args.source_total_layers,
        target_total_layers=args.total_layers,
    )
    filled_transfer = transfer_round2_layer_bands_by_anchor_fill(
        source_layer_indices=source_mask,
        source_total_layers=args.source_total_layers,
        target_total_layers=args.total_layers,
    )
    span_transfer = transfer_round2_layer_bands_by_scaled_span(
        source_layer_indices=source_mask,
        source_total_layers=args.source_total_layers,
        target_total_layers=args.total_layers,
    )
    source_bands = contiguous_layer_bands(source_mask)
    return (
        MiniMoePolicySearchCandidate(
            name="reference",
            surface=_build_surface(
                kind="reference",
                total_layers=args.total_layers,
                experts_per_block=args.experts_per_block,
                normalized_entropy_threshold=args.entropy_threshold,
            ),
            note="One-shot standard MoE baseline",
        ),
        MiniMoePolicySearchCandidate(
            name="entropy_all_layers",
            surface=_build_surface(
                kind="entropy_recurrent",
                total_layers=args.total_layers,
                experts_per_block=args.experts_per_block,
                normalized_entropy_threshold=args.entropy_threshold,
            ),
            note="All-layer entropy-gated recurrent routing",
        ),
        MiniMoePolicySearchCandidate(
            name="entropy_transfer_points",
            surface=_build_surface(
                kind="entropy_recurrent",
                total_layers=args.total_layers,
                experts_per_block=args.experts_per_block,
                normalized_entropy_threshold=args.entropy_threshold,
                round2_layer_indices=point_transfer,
            ),
            note=f"Depth-fraction point transfer from source mask {source_mask}",
        ),
        MiniMoePolicySearchCandidate(
            name="entropy_transfer_band_fill",
            surface=_build_surface(
                kind="entropy_recurrent",
                total_layers=args.total_layers,
                experts_per_block=args.experts_per_block,
                normalized_entropy_threshold=args.entropy_threshold,
                round2_layer_indices=filled_transfer,
            ),
            note=f"Contiguous band fill transfer from source bands {source_bands}",
        ),
        MiniMoePolicySearchCandidate(
            name="entropy_transfer_scaled_span",
            surface=_build_surface(
                kind="entropy_recurrent",
                total_layers=args.total_layers,
                experts_per_block=args.experts_per_block,
                normalized_entropy_threshold=args.entropy_threshold,
                round2_layer_indices=span_transfer,
            ),
            note=f"Scaled-span transfer from source bands {source_bands}",
        ),
    )


def main() -> int:
    args = parse_args()
    request = MiniMoePolicySearchRequest(
        benchmark_name=args.benchmark_name,
        manifest_template=_manifest(args),
        candidates=_candidates(args),
        seeds=_parse_int_tuple(args.seeds),
        output_dir=args.output_dir,
        ledger_path=args.ledger_path,
    )
    summary = run_mini_moe_policy_search(request)
    payload = {
        "benchmark_name": summary.benchmark_name,
        "seeds": list(summary.seeds),
        "summary_path": summary.summary_path,
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
            for result in summary.candidate_results
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
