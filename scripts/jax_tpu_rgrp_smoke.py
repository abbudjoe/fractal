#!/usr/bin/env python3
"""Smoke benchmark for the JAX rotary gated recurrent state update adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.jax_tpu.adapters.rotary_gated_recurrent_state_update import (
    SUPPORTED_STATE_TRANSFORMS,
    benchmark_scan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a JAX/JIT smoke benchmark for the rotary gated recurrent state update primitive."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--state-transform", choices=SUPPORTED_STATE_TRANSFORMS, default="block-diagonal-4")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--forward-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = benchmark_scan(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            state_transform=args.state_transform,
            dtype=args.dtype,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            include_grad=not args.forward_only,
        )
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
