from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
import time
from typing import Any

import torch
from torch.profiler import record_function


@dataclass(frozen=True)
class CudaTimingSummary:
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


class CudaEventTimingCollector:
    """Low-overhead CUDA-event collector for steady-state timing regions."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._wall_times_by_name: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._wall_times_by_name[name].append((time.perf_counter() - wall_start) * 1000.0)
            self._events.append((name, start, end))

    def summarize(self) -> dict[str, CudaTimingSummary]:
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        values_by_name: dict[str, list[float]] = defaultdict(list)
        for name, start, end in self._events:
            values_by_name[name].append(float(start.elapsed_time(end)))
        return {
            name: CudaTimingSummary(
                count=len(values),
                total_ms=sum(values),
                mean_ms=sum(values) / len(values),
                min_ms=min(values),
                max_ms=max(values),
            )
            for name, values in sorted(values_by_name.items())
            if values
        }

    def summarize_wall(self) -> dict[str, CudaTimingSummary]:
        if not self.enabled:
            return {}
        return {
            name: CudaTimingSummary(
                count=len(values),
                total_ms=sum(values),
                mean_ms=sum(values) / len(values),
                min_ms=min(values),
                max_ms=max(values),
            )
            for name, values in sorted(self._wall_times_by_name.items())
            if values
        }


_CURRENT_COLLECTOR: ContextVar[CudaEventTimingCollector | None] = ContextVar(
    "fractal_cuda_timing_collector",
    default=None,
)
_ACTIVE_COLLECTOR: CudaEventTimingCollector | None = None
_NAMED_REGIONS_ENABLED: ContextVar[bool] = ContextVar(
    "fractal_named_timing_regions_enabled",
    default=False,
)
_ACTIVE_NAMED_REGIONS_ENABLED = False


@contextmanager
def use_cuda_timing(collector: CudaEventTimingCollector | None) -> Iterator[None]:
    global _ACTIVE_COLLECTOR
    previous_collector = _ACTIVE_COLLECTOR
    _ACTIVE_COLLECTOR = collector
    token = _CURRENT_COLLECTOR.set(collector)
    try:
        yield
    finally:
        _CURRENT_COLLECTOR.reset(token)
        _ACTIVE_COLLECTOR = previous_collector


@contextmanager
def use_named_timing_regions(enabled: bool = True) -> Iterator[None]:
    """Enable torch-profiler record_function labels for explicit profiling runs.

    CUDA-event timing does not need profiler labels, and normal training should
    not pay profiler annotation overhead just because the code is instrumented.
    """

    global _ACTIVE_NAMED_REGIONS_ENABLED
    previous = _ACTIVE_NAMED_REGIONS_ENABLED
    _ACTIVE_NAMED_REGIONS_ENABLED = enabled
    token = _NAMED_REGIONS_ENABLED.set(enabled)
    try:
        yield
    finally:
        _NAMED_REGIONS_ENABLED.reset(token)
        _ACTIVE_NAMED_REGIONS_ENABLED = previous


@contextmanager
def timed_region(name: str) -> Iterator[None]:
    """Emit optional torch-profiler regions and optional CUDA-event timings."""

    collector = _CURRENT_COLLECTOR.get() or _ACTIVE_COLLECTOR
    named_regions_enabled = _NAMED_REGIONS_ENABLED.get() or _ACTIVE_NAMED_REGIONS_ENABLED
    nvtx_enabled = os.environ.get("FRACTAL_ENABLE_NVTX_RANGES", "").lower() in {"1", "true", "yes"}

    @contextmanager
    def maybe_record_function() -> Iterator[None]:
        if named_regions_enabled:
            with record_function(name):
                yield
        else:
            yield

    @contextmanager
    def maybe_nvtx_range() -> Iterator[None]:
        if nvtx_enabled and torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield

    with maybe_record_function(), maybe_nvtx_range():
        if collector is None:
            yield
        else:
            with collector.record(name):
                yield


def cuda_timing_summary_to_dict(summary: dict[str, CudaTimingSummary]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "count": value.count,
            "total_ms": value.total_ms,
            "mean_ms": value.mean_ms,
            "min_ms": value.min_ms,
            "max_ms": value.max_ms,
        }
        for name, value in summary.items()
    }
