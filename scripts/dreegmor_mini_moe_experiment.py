#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.runners.mini_moe import MiniMoeRunnerRequest, render_report, run_mini_moe_variant
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
)
from python.specs.mini_moe import MiniMoeSurfaceSpec


DEFAULT_NOTE = (
    "Phase 1 mini-MoE benchmark on the shared Python research substrate with all-blocks "
    "MoE FFN placement and package-native routing/dispatch reporting"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the phase 1 mini-MoE benchmark.")
    parser.add_argument("--variant", choices=["all", "reference", "recurrent"], default="all")
    parser.add_argument("--backend", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--env-kind",
        choices=["requirements-only", "official-mamba3", "compile-safe", "primitive-triton"],
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--primitive-runtime-backend",
        default="torch",
        choices=["torch", "triton"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--window-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--full-train-pass", action="store_true")
    parser.add_argument("--full-eval-pass", action="store_true")
    parser.add_argument("--warmup-eval-batches", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--jsonl-train-path", type=Path, required=True)
    parser.add_argument("--jsonl-eval-path", type=Path, required=True)
    parser.add_argument("--benchmark-name", default="dreegmor-mini-moe")
    parser.add_argument("--corpus-name", default="fineweb-stage0-canary")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32")
    return parser.parse_args()


def _manifest(args: argparse.Namespace) -> BenchmarkRunManifest:
    return BenchmarkRunManifest(
        run_label=args.run_label,
        implementation_kind="python_native",
        benchmark_name=args.benchmark_name,
        note=DEFAULT_NOTE,
        seed_spec=SeedSpec(model_seed=args.seed, data_seed=args.data_seed),
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
            full_train_pass=args.full_train_pass,
            full_eval_pass=args.full_eval_pass,
            learning_rate=args.learning_rate,
            warmup_eval_batches=args.warmup_eval_batches,
            warmup_train_steps=args.warmup_train_steps,
        ),
        runtime=DeviceRuntimeSpec(
            backend=args.backend,
            cuda_device=args.cuda_device,
            dtype=args.dtype,
            env_kind=args.env_kind,
            compile_mode=args.compile_mode,
            primitive_runtime_backend=args.primitive_runtime_backend,
        ),
    )


def _requests(args: argparse.Namespace) -> list[MiniMoeRunnerRequest]:
    manifest = _manifest(args)
    candidates: list[tuple[str, MiniMoeSurfaceSpec]] = []
    if args.variant in {"all", "reference"}:
        candidates.append(("reference", MiniMoeSurfaceSpec.phase1_reference_default()))
    if args.variant in {"all", "recurrent"}:
        candidates.append(("recurrent", MiniMoeSurfaceSpec.phase1_recurrent_default()))
    return [
        MiniMoeRunnerRequest(
            manifest=manifest,
            surface=surface,
            output_dir=args.output_dir,
            output_format=args.output,
            ledger_path=args.ledger_path,
            variant_output_name=name,
            model_note=DEFAULT_NOTE,
        )
        for name, surface in candidates
    ]


def main() -> int:
    args = parse_args()
    requests = _requests(args)
    reports = [run_mini_moe_variant(request) for request in requests]
    if args.output == "json":
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
        return 0

    for report, request in zip(reports, requests):
        print(render_report(report, request))
        print(f"report_path={report.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
