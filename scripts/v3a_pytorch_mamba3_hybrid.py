#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.reporting.render import render_path1_table
from python.runners.path1 import Path1RunnerRequest, run_path1_variant
from python.specs.common import (
    BenchmarkBudgetSpec,
    BenchmarkRunManifest,
    DeviceRuntimeSpec,
    JsonlCorpusSpec,
    SeedSpec,
)
from python.specs.path1 import ReferenceSsmProfile, phase1_reference_ssm_variant


DEFAULT_NOTE = (
    "Path 1 reference SSM hybrid baseline using official PyTorch Mamba3 blocks on the shared "
    "byte-level benchmark substrate through the upstream SISO runtime-oriented lane"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official PyTorch Mamba3 v3a hybrid benchmark.")
    parser.add_argument("--backend", default="cuda", choices=["cuda"])
    parser.add_argument("--cuda-device", type=int, default=0)
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
    parser.add_argument("--benchmark-name")
    parser.add_argument("--corpus-name", default="fineweb-stage0-canary")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = BenchmarkRunManifest(
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
        ),
    )
    report = run_path1_variant(
        Path1RunnerRequest(
            manifest=manifest,
            variant=phase1_reference_ssm_variant(
                profile=ReferenceSsmProfile.MAMBA3_SISO_RUNTIME
            ),
            output_dir=args.output_dir,
            output_format=args.output,
            ledger_path=args.ledger_path,
            variant_output_name="python-reference-ssm-hybrid-siso",
            model_note=DEFAULT_NOTE,
        )
    )
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_path1_table(report, "python-reference-ssm-hybrid-siso"))
        print(f"report_path={report.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
