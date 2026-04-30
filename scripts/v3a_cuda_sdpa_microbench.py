#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

try:  # pragma: no cover - depends on PyTorch version.
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - reported as unsupported benchmark cases.
    create_block_mask = None
    flex_attention = None

try:  # pragma: no cover - optional CUDA package.
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover - reported as unsupported benchmark cases.
    flash_attn_func = None


DEFAULT_SHAPES = (
    "attn_d448_h7:448:7",
    "rgrp_d448_h8:448:8",
    "attn_d456_h19:456:19",
    "rgrp_d480_h10:480:10",
    "rgrp_d512_h8:512:8",
    "slow_d472_h8:472:8",
)

_COMPILED_FLEX_ATTENTION = None

KERNEL_KEY_SUBSTRINGS = (
    "scaled_dot_product",
    "flash_attention",
    "efficient_attention",
    "attention",
    "bmm",
    "matmul",
)


@dataclass(frozen=True)
class SdpaShapeSpec:
    name: str
    d_model: int
    head_count: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.head_count

    @property
    def cuda_friendly_head_dim(self) -> bool:
        return self.head_dim % 8 == 0


@dataclass(frozen=True)
class SdpaCaseResult:
    shape_name: str
    d_model: int
    head_count: int
    head_dim: int
    cuda_friendly_head_dim: bool
    batch_size: int
    seq_len: int
    local_window: int
    dtype: str
    device: str
    mask_mode: str
    backend: str
    mode: str
    success: bool
    mean_ms: float | None
    min_ms: float | None
    max_ms: float | None
    peak_cuda_memory_mb: float | None
    detected_profiler_keys: list[str]
    profile_table: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_name": self.shape_name,
            "d_model": self.d_model,
            "head_count": self.head_count,
            "head_dim": self.head_dim,
            "cuda_friendly_head_dim": self.cuda_friendly_head_dim,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "local_window": self.local_window,
            "dtype": self.dtype,
            "device": self.device,
            "mask_mode": self.mask_mode,
            "backend": self.backend,
            "mode": self.mode,
            "success": self.success,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "peak_cuda_memory_mb": self.peak_cuda_memory_mb,
            "detected_profiler_keys": self.detected_profiler_keys,
            "profile_table": self.profile_table,
            "error": self.error,
        }


def parse_shape_spec(raw: str) -> SdpaShapeSpec:
    parts = raw.split(":")
    if len(parts) == 2:
        d_model = int(parts[0])
        head_count = int(parts[1])
        name = f"d{d_model}_h{head_count}"
    elif len(parts) == 3:
        name = parts[0]
        d_model = int(parts[1])
        head_count = int(parts[2])
    else:
        raise ValueError(f"shape spec must be name:d_model:heads or d_model:heads, got {raw!r}")
    if d_model <= 0 or head_count <= 0:
        raise ValueError(f"d_model and head_count must be positive, got {raw!r}")
    if d_model % head_count != 0:
        raise ValueError(f"d_model must be divisible by head_count, got {raw!r}")
    return SdpaShapeSpec(name=name, d_model=d_model, head_count=head_count)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def local_keep_mask(seq_len: int, local_window: int, *, device: torch.device) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if local_window <= 0:
        raise ValueError("local_window must be positive")
    queries = torch.arange(seq_len, device=device).view(seq_len, 1)
    keys = torch.arange(seq_len, device=device).view(1, seq_len)
    return (keys <= queries) & (keys >= queries - (local_window - 1))


def additive_local_bias(
    seq_len: int,
    local_window: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    keep = local_keep_mask(seq_len, local_window, device=device)
    bias = torch.zeros((1, 1, seq_len, seq_len), dtype=dtype, device=device)
    return bias.masked_fill(~keep.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)


def bool_local_mask(seq_len: int, local_window: int, *, device: torch.device) -> torch.Tensor:
    return local_keep_mask(seq_len, local_window, device=device).view(1, 1, seq_len, seq_len)


def attention_mask(
    mask_mode: str,
    *,
    seq_len: int,
    local_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, bool]:
    if mask_mode == "causal":
        return None, True
    if mask_mode == "local-additive":
        return additive_local_bias(seq_len, local_window, device=device, dtype=dtype), False
    if mask_mode == "local-bool":
        return bool_local_mask(seq_len, local_window, device=device), False
    raise ValueError(f"unsupported mask mode: {mask_mode}")


@contextmanager
def forced_sdpa_backend(backend: str, *, device: torch.device) -> Iterator[None]:
    if backend == "auto":
        with nullcontext():
            yield
        return
    if backend in {"flex-local", "flash-local"}:
        with nullcontext():
            yield
        return
    if device.type != "cuda":
        raise RuntimeError(f"forced SDPA backend {backend!r} requires CUDA")

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend_map = {
            "flash": "FLASH_ATTENTION",
            "efficient": "EFFICIENT_ATTENTION",
            "math": "MATH",
        }
        enum_name = backend_map[backend]
        with sdpa_kernel([getattr(SDPBackend, enum_name)]):
            yield
        return
    except (ImportError, AttributeError, TypeError):
        pass

    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        raise RuntimeError("this PyTorch build does not expose an SDPA backend selector")

    enable_flash = backend == "flash"
    enable_efficient = backend == "efficient"
    enable_math = backend == "math"
    with torch.backends.cuda.sdp_kernel(
        enable_flash=enable_flash,
        enable_mem_efficient=enable_efficient,
        enable_math=enable_math,
    ):
        yield


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        raise RuntimeError("PyTorch FlexAttention is not available")
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


def flex_local_block_mask(
    *,
    seq_len: int,
    local_window: int,
    device: torch.device,
) -> object:
    if create_block_mask is None:
        raise RuntimeError("PyTorch FlexAttention block masks are not available")

    def local_causal_mask(batch, head, query_index, key_index):
        del batch, head
        return (key_index <= query_index) & (key_index >= query_index - (local_window - 1))

    return create_block_mask(
        local_causal_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=128,
    )


def _make_qkv(
    shape: SdpaShapeSpec,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor_shape = (batch_size, shape.head_count, seq_len, shape.head_dim)
    requires_grad = mode == "forward-backward"
    q = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v


def _one_iteration(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    mode: str,
    block_mask: object | None = None,
    flash_window: int | None = None,
) -> torch.Tensor:
    if q.grad is not None:
        q.grad = None
    if k.grad is not None:
        k.grad = None
    if v.grad is not None:
        v.grad = None
    if flash_window is not None:
        if flash_attn_func is None:
            raise RuntimeError("flash-attn is not available")
        out = flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            causal=True,
            window_size=(flash_window - 1, 0),
        ).transpose(1, 2).contiguous()
    elif block_mask is None:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    else:
        out = _compiled_flex_attention()(q, k, v, block_mask=block_mask)
    if mode == "forward-backward":
        loss = out.float().square().mean()
        loss.backward()
        return loss.detach()
    if mode == "forward":
        return out
    raise ValueError(f"unsupported mode: {mode}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    block_mask: object | None,
    flash_window: int | None,
    mode: str,
    device: torch.device,
    row_limit: int,
) -> tuple[list[str], str]:
    activities = [ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"
    with profile(activities=activities, record_shapes=True, profile_memory=device.type == "cuda") as prof:
        _one_iteration(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            block_mask=block_mask,
            flash_window=flash_window,
            mode=mode,
        )
    _synchronize(device)
    keys: list[str] = []
    for event in prof.key_averages():
        key = str(event.key)
        lower = key.lower()
        if any(fragment in lower for fragment in KERNEL_KEY_SUBSTRINGS):
            keys.append(key)
    table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    return sorted(set(keys)), table


def _timed_iterations(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    block_mask: object | None,
    flash_window: int | None,
    mode: str,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup):
        _one_iteration(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            block_mask=block_mask,
            flash_window=flash_window,
            mode=mode,
        )
    _synchronize(device)

    times_ms: list[float] = []
    if device.type == "cuda":
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _one_iteration(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                block_mask=block_mask,
                flash_window=flash_window,
                mode=mode,
            )
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
        return times_ms

    for _ in range(iters):
        start = time.perf_counter()
        _one_iteration(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            block_mask=block_mask,
            flash_window=flash_window,
            mode=mode,
        )
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def run_case(
    shape: SdpaShapeSpec,
    *,
    batch_size: int,
    seq_len: int,
    local_window: int,
    dtype_name: str,
    device: torch.device,
    mask_mode: str,
    backend: str,
    mode: str,
    warmup: int,
    iters: int,
    profile_row_limit: int,
) -> SdpaCaseResult:
    dtype = dtype_from_name(dtype_name)
    peak_memory: float | None = None
    try:
        q, k, v = _make_qkv(
            shape,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=dtype,
            device=device,
            mode=mode,
        )
        mask, is_causal = attention_mask(
            mask_mode,
            seq_len=seq_len,
            local_window=local_window,
            device=device,
            dtype=dtype,
        )
        block_mask = None
        flash_window = None
        if backend == "flex-local":
            if device.type != "cuda":
                raise RuntimeError("backend=flex-local requires CUDA")
            block_mask = flex_local_block_mask(
                seq_len=seq_len,
                local_window=seq_len if mask_mode == "causal" else local_window,
                device=device,
            )
            mask = None
            is_causal = False
        if backend == "flash-local":
            if device.type != "cuda":
                raise RuntimeError("backend=flash-local requires CUDA")
            if flash_attn_func is None:
                raise RuntimeError("flash-attn is not available")
            flash_window = seq_len if mask_mode == "causal" else local_window
            mask = None
            is_causal = False
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with forced_sdpa_backend(backend, device=device):
            times = _timed_iterations(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=is_causal,
                block_mask=block_mask,
                flash_window=flash_window,
                mode=mode,
                warmup=warmup,
                iters=iters,
                device=device,
            )
            detected_keys, profile_table = _profile_once(
                q,
                k,
                v,
                attn_mask=mask,
                is_causal=is_causal,
                block_mask=block_mask,
                flash_window=flash_window,
                mode=mode,
                device=device,
                row_limit=profile_row_limit,
            )
        if device.type == "cuda":
            peak_memory = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        return SdpaCaseResult(
            shape_name=shape.name,
            d_model=shape.d_model,
            head_count=shape.head_count,
            head_dim=shape.head_dim,
            cuda_friendly_head_dim=shape.cuda_friendly_head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            local_window=local_window,
            dtype=dtype_name,
            device=str(device),
            mask_mode=mask_mode,
            backend=backend,
            mode=mode,
            success=True,
            mean_ms=float(sum(times) / len(times)) if times else math.nan,
            min_ms=float(min(times)) if times else math.nan,
            max_ms=float(max(times)) if times else math.nan,
            peak_cuda_memory_mb=peak_memory,
            detected_profiler_keys=detected_keys,
            profile_table=profile_table,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going across unsupported backend cases.
        return SdpaCaseResult(
            shape_name=shape.name,
            d_model=shape.d_model,
            head_count=shape.head_count,
            head_dim=shape.head_dim,
            cuda_friendly_head_dim=shape.cuda_friendly_head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
            local_window=local_window,
            dtype=dtype_name,
            device=str(device),
            mask_mode=mask_mode,
            backend=backend,
            mode=mode,
            success=False,
            mean_ms=None,
            min_ms=None,
            max_ms=None,
            peak_cuda_memory_mb=peak_memory,
            detected_profiler_keys=[],
            profile_table=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _device_from_args(raw: str) -> torch.device:
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise SystemExit(f"{raw} requested, but torch.cuda.is_available() is false")
        return torch.device(raw)
    return torch.device(raw)


def _device_probe(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
        )
    return payload


def _summary_markdown(results: Sequence[SdpaCaseResult], *, device_probe: dict[str, Any]) -> str:
    lines = [
        "# CUDA SDPA Microbench",
        "",
        "This isolates `torch.nn.functional.scaled_dot_product_attention` for the Path 1 long-context shapes.",
        "",
        "```json",
        json.dumps(device_probe, indent=2, sort_keys=True),
        "```",
        "",
        "| Shape | d | h | head_dim | friendly | mask | backend | mode | success | mean ms | peak MB | detected kernels | error |",
        "|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for result in results:
        kernels = ", ".join(result.detected_profiler_keys[:4])
        if len(result.detected_profiler_keys) > 4:
            kernels += ", ..."
        mean_ms = "" if result.mean_ms is None else f"{result.mean_ms:.3f}"
        peak_mb = "" if result.peak_cuda_memory_mb is None else f"{result.peak_cuda_memory_mb:.2f}"
        error = (result.error or "").replace("|", "\\|")
        lines.append(
            f"| {result.shape_name} | {result.d_model} | {result.head_count} | {result.head_dim} | "
            f"{'yes' if result.cuda_friendly_head_dim else 'no'} | {result.mask_mode} | {result.backend} | "
            f"{result.mode} | {1 if result.success else 0} | {mean_ms} | {peak_mb} | {kernels} | {error} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Microbenchmark CUDA SDPA backend behavior for Path 1 local-attention shapes."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["forward", "forward-backward"], default="forward-backward")
    parser.add_argument(
        "--shape-specs",
        default=",".join(DEFAULT_SHAPES),
        help="Comma-separated shape specs: name:d_model:head_count or d_model:head_count.",
    )
    parser.add_argument(
        "--mask-modes",
        default="causal,local-additive,local-bool",
        help="Comma-separated mask modes: causal, local-additive, local-bool.",
    )
    parser.add_argument(
        "--backends",
        default="auto,flash,efficient,math,flex-local,flash-local",
        help="Comma-separated backend selectors: auto, flash, efficient, math, flex-local, flash-local.",
    )
    parser.add_argument("--profile-row-limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/cuda_sdpa_microbench/latest"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    device = _device_from_args(args.device)
    shape_specs = [parse_shape_spec(raw) for raw in parse_csv(args.shape_specs)]
    mask_modes = parse_csv(args.mask_modes)
    backends = parse_csv(args.backends)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_probe = _device_probe(device)
    print("device_probe=" + json.dumps(device_probe, sort_keys=True), flush=True)

    results: list[SdpaCaseResult] = []
    for shape in shape_specs:
        for mask_mode in mask_modes:
            for backend in backends:
                print(
                    "running "
                    f"shape={shape.name} d={shape.d_model} h={shape.head_count} hd={shape.head_dim} "
                    f"mask={mask_mode} backend={backend}",
                    flush=True,
                )
                result = run_case(
                    shape,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    local_window=args.local_window,
                    dtype_name=args.dtype,
                    device=device,
                    mask_mode=mask_mode,
                    backend=backend,
                    mode=args.mode,
                    warmup=args.warmup,
                    iters=args.iters,
                    profile_row_limit=args.profile_row_limit,
                )
                results.append(result)
                status = "ok" if result.success else f"failed: {result.error}"
                mean = "" if result.mean_ms is None else f" mean_ms={result.mean_ms:.3f}"
                print(f"result {status}{mean}", flush=True)

    payload = {
        "device_probe": device_probe,
        "results": [result.to_dict() for result in results],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _summary_markdown(results, device_probe=device_probe)
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
