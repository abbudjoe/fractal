#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.runners.mini_moe_autoresearch import (  # noqa: E402
    MiniMoeAutoresearchRequest,
    run_mini_moe_autoresearch,
)
from python.specs.common import (  # noqa: E402
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mini-MoE autoresearch hill-climber.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--jsonl-train-path", type=Path, required=True)
    parser.add_argument("--jsonl-eval-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--benchmark-name", default="dreegmor-mini-moe-autoresearch")
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
    parser.add_argument("--max-new-candidates", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-after-success", action="store_true")
    parser.add_argument("--parallel-candidates", type=int, default=1)
    return parser.parse_args()


def _manifest(args: argparse.Namespace) -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        run_label=args.run_label,
        implementation_kind="python_native",
        benchmark_name=args.benchmark_name,
        note="Autoresearch hill-climb over typed selective recurrent-routing masks",
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


def main() -> int:
    args = parse_args()
    summary = run_mini_moe_autoresearch(
        MiniMoeAutoresearchRequest(
            benchmark_name=args.benchmark_name,
            manifest_template=_manifest(args),
            output_dir=args.output_dir,
            seeds=_parse_int_tuple(args.seeds),
            total_layers=args.total_layers,
            experts_per_block=args.experts_per_block,
            normalized_entropy_threshold=args.entropy_threshold,
            source_total_layers=args.source_total_layers,
            source_round2_layers=_parse_int_tuple(args.source_round2_layers),
            ledger_path=args.ledger_path,
            max_new_candidates=args.max_new_candidates,
            resume=not args.no_resume,
            stop_on_first_success=not args.continue_after_success,
            parallel_candidates=args.parallel_candidates,
        )
    )
    best_selective_loss_gap_vs_all_layers = None
    best_selective_train_gap_vs_all_layers = None
    if summary.best_selective is not None:
        best_selective_loss_gap_vs_all_layers = (
            summary.best_selective.avg_final_loss - summary.baseline_all_layers.avg_final_loss
        )
        best_selective_train_gap_vs_all_layers = (
            summary.best_selective.avg_train_toks_per_s - summary.baseline_all_layers.avg_train_toks_per_s
        )

    success_loss_gap_vs_all_layers = None
    success_train_gap_vs_all_layers = None
    if summary.success_candidate is not None:
        success_loss_gap_vs_all_layers = (
            summary.success_candidate.avg_final_loss - summary.baseline_all_layers.avg_final_loss
        )
        success_train_gap_vs_all_layers = (
            summary.success_candidate.avg_train_toks_per_s - summary.baseline_all_layers.avg_train_toks_per_s
        )
    payload = {
        "benchmark_name": summary.benchmark_name,
        "status": summary.status,
        "success_found": 1 if summary.success_candidate is not None else 0,
        "state_path": summary.state_path,
        "summary_path": summary.summary_path,
        "total_selective_search_space": summary.total_selective_search_space,
        "parallel_candidates": args.parallel_candidates,
        "baseline_reference": {
            "avg_final_loss": summary.baseline_reference.avg_final_loss,
            "avg_train_toks_per_s": summary.baseline_reference.avg_train_toks_per_s,
        },
        "baseline_all_layers": {
            "avg_final_loss": summary.baseline_all_layers.avg_final_loss,
            "avg_train_toks_per_s": summary.baseline_all_layers.avg_train_toks_per_s,
        },
        "best_selective": (
            None
            if summary.best_selective is None
            else {
                "candidate_name": summary.best_selective.candidate_name,
                "mask": list(summary.best_selective.mask),
                "avg_final_loss": summary.best_selective.avg_final_loss,
                "avg_train_toks_per_s": summary.best_selective.avg_train_toks_per_s,
                "avg_overall_round2_fraction": summary.best_selective.avg_overall_round2_fraction,
            }
        ),
        "success_candidate": (
            None
            if summary.success_candidate is None
            else {
                "candidate_name": summary.success_candidate.candidate_name,
                "mask": list(summary.success_candidate.mask),
                "avg_final_loss": summary.success_candidate.avg_final_loss,
                "avg_train_toks_per_s": summary.success_candidate.avg_train_toks_per_s,
            }
        ),
        "evaluated_selective_count": summary.evaluated_selective_count,
        "pending_candidate_count": summary.pending_candidate_count,
        "best_selective_loss_gap_vs_all_layers": best_selective_loss_gap_vs_all_layers,
        "best_selective_train_gap_vs_all_layers": best_selective_train_gap_vs_all_layers,
        "success_loss_gap_vs_all_layers": success_loss_gap_vs_all_layers,
        "success_train_gap_vs_all_layers": success_train_gap_vs_all_layers,
        "explored_fraction": (
            0.0
            if summary.total_selective_search_space <= 0
            else summary.evaluated_selective_count / summary.total_selective_search_space
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
