#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.runtime.recurrent import make_eggroll_factors, materialized_eggroll_linear, virtual_eggroll_linear


@dataclass(frozen=True)
class EggrollLinearCase:
    mode: str
    population_size: int
    batch_size: int
    seq_len: int
    d_in: int
    d_out: int
    rank: int
    dtype: str
    device: str
    success: bool
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    peak_cuda_memory_mb: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "population_size": self.population_size,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "rank": self.rank,
            "dtype": self.dtype,
            "device": self.device,
            "success": self.success,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "peak_cuda_memory_mb": self.peak_cuda_memory_mb,
            "error": self.error,
        }


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"all values must be positive, got {raw!r}")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def run_case(
    *,
    mode: str,
    population_size: int,
    batch_size: int,
    seq_len: int,
    d_in: int,
    d_out: int,
    rank: int,
    sigma: float,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
) -> EggrollLinearCase:
    dtype = dtype_from_name(dtype_name)
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        if mode == "base":
            inputs = torch.randn(batch_size * population_size, seq_len, d_in, device=device, dtype=dtype, generator=generator)
        else:
            inputs = torch.randn(population_size, batch_size, seq_len, d_in, device=device, dtype=dtype, generator=generator)
        weight = torch.randn(d_out, d_in, device=device, dtype=dtype, generator=generator)
        bias = torch.randn(d_out, device=device, dtype=dtype, generator=generator)
        a, b = make_eggroll_factors(
            population_size=population_size,
            d_out=d_out,
            d_in=d_in,
            rank=rank,
            device=device,
            dtype=dtype,
            seed=seed + 1,
        )

        def step() -> torch.Tensor:
            if mode == "base":
                return F.linear(inputs, weight, bias)
            if mode == "virtual":
                return virtual_eggroll_linear(inputs, weight, perturbation_a=a, perturbation_b=b, sigma=sigma, bias=bias)
            if mode == "materialized":
                return materialized_eggroll_linear(inputs, weight, perturbation_a=a, perturbation_b=b, sigma=sigma, bias=bias)
            raise ValueError(f"unsupported mode: {mode}")

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            step()
        synchronize(device)

        durations: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            output = step()
            synchronize(device)
            durations.append((time.perf_counter() - start) * 1000.0)
            if not torch.isfinite(output).all():
                raise RuntimeError("non-finite output")

        return EggrollLinearCase(
            mode=mode,
            population_size=population_size,
            batch_size=batch_size,
            seq_len=seq_len,
            d_in=d_in,
            d_out=d_out,
            rank=rank,
            dtype=dtype_name,
            device=str(device),
            success=True,
            mean_ms=sum(durations) / len(durations),
            min_ms=min(durations),
            max_ms=max(durations),
            peak_cuda_memory_mb=peak_memory_mb(device),
            error=None,
        )
    except Exception as exc:
        return EggrollLinearCase(
            mode=mode,
            population_size=population_size,
            batch_size=batch_size,
            seq_len=seq_len,
            d_in=d_in,
            d_out=d_out,
            rank=rank,
            dtype=dtype_name,
            device=str(device),
            success=False,
            mean_ms=None,
            min_ms=None,
            max_ms=None,
            peak_cuda_memory_mb=peak_memory_mb(device),
            error=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark EGGROLL-style virtual low-rank linear perturbations against materialized references."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--population-sizes", type=parse_csv_ints, default=parse_csv_ints("1,8,32,128"))
    parser.add_argument("--ranks", type=parse_csv_ints, default=parse_csv_ints("1"))
    parser.add_argument("--widths", type=parse_csv_ints, default=parse_csv_ints("256,320,384,448,512"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--modes", default="base,virtual,materialized")
    parser.add_argument("--sigma", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    device = resolve_device(args.device)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    rows: list[EggrollLinearCase] = []
    for width in args.widths:
        for rank in args.ranks:
            for population_size in args.population_sizes:
                for mode in modes:
                    rows.append(
                        run_case(
                            mode=mode,
                            population_size=population_size,
                            batch_size=args.batch_size,
                            seq_len=args.seq_len,
                            d_in=width,
                            d_out=width,
                            rank=rank,
                            sigma=args.sigma,
                            dtype_name=args.dtype,
                            device=device,
                            warmup=args.warmup,
                            iters=args.iters,
                            seed=args.seed,
                        )
                    )

    payload = {"rows": [row.to_dict() for row in rows]}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "eggroll_linear_microbench.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
