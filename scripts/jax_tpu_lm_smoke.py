#!/usr/bin/env python3
"""Run a tiny JAX LM integration smoke for transformer-vs-RGRP TPU gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.jax_tpu.adapters.rotary_gated_recurrent_state_update import (
    SUPPORTED_EXECUTION_MODES,
    SUPPORTED_PROJECTION_MODES,
    SUPPORTED_STATE_TRANSFORMS,
    SUPPORTED_TRIG_MODES,
)
from python.jax_tpu.lm_smoke import SUPPORTED_VARIANTS, JaxLmSmokeConfig, benchmark_lm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny JAX LM integration smoke with either an MLP FFN or RGRP FFN seam."
    )
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, default="mlp")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--rgrp-state-transform", choices=SUPPORTED_STATE_TRANSFORMS, default="block-diagonal-4-masked-dense")
    parser.add_argument("--rgrp-scan-unroll", type=int, default=1)
    parser.add_argument("--rgrp-projection-mode", choices=SUPPORTED_PROJECTION_MODES, default="sequence")
    parser.add_argument("--rgrp-trig-mode", choices=SUPPORTED_TRIG_MODES, default="precompute")
    parser.add_argument("--rgrp-execution-mode", choices=SUPPORTED_EXECUTION_MODES, default="scan")
    parser.add_argument("--rgrp-pallas-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--forward-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = JaxLmSmokeConfig(
        variant=args.variant,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ffn_multiplier=args.ffn_multiplier,
        rgrp_state_transform=args.rgrp_state_transform,
        rgrp_scan_unroll=args.rgrp_scan_unroll,
        rgrp_projection_mode=args.rgrp_projection_mode,
        rgrp_trig_mode=args.rgrp_trig_mode,
        rgrp_execution_mode=args.rgrp_execution_mode,
        rgrp_pallas_chunk_size=args.rgrp_pallas_chunk_size,
        dtype=args.dtype,
    )
    try:
        result = benchmark_lm(
            config=config,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            forward_only=args.forward_only,
        )
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
