from __future__ import annotations

import math
import resource
import sys
import time
from typing import Protocol

import torch
import torch.nn.functional as F

from python.data.byte_corpus import TokenBatch
from python.reporting.schema import BenchmarkReport, EvalSummary, RuntimeSummary, TrainStepRecord


class LanguageModelProtocol(Protocol):
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...


def perplexity_from_loss(loss: float) -> float:
    if math.isnan(loss):
        return math.nan
    if loss == math.inf:
        return math.inf
    if loss == -math.inf:
        return 0.0
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


def materialize_batch(batch: TokenBatch, device: torch.device) -> TokenBatch:
    return batch.to_device(device)


def process_peak_rss_bytes() -> int:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak_rss
    return peak_rss * 1024


def cuda_memory_stats(device: torch.device) -> tuple[int, int]:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)


def evaluate_model(
    model: LanguageModelProtocol,
    batches: list[TokenBatch],
    eval_batches: int,
    autocast_dtype: torch.dtype | None,
    *,
    pad_token: int,
    device: torch.device,
    device_type: str,
) -> EvalSummary:
    selected = batches[: min(eval_batches, len(batches))]
    total_loss = 0.0
    total_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in selected:
            batch_on_device = materialize_batch(batch, device)
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model.forward_logits(batch_on_device.input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch_on_device.target_ids.reshape(-1),
                    ignore_index=pad_token,
                )
            total_loss += float(loss.detach().float().item())
            total_batches += 1
    mean_loss = total_loss / max(total_batches, 1)
    return EvalSummary(
        batch_count=total_batches,
        mean_loss=mean_loss,
        perplexity=perplexity_from_loss(mean_loss),
    )


def warmup_model(
    model: LanguageModelProtocol,
    optimizer: torch.optim.Optimizer,
    train_batches: list[TokenBatch],
    eval_batches: list[TokenBatch],
    warmup_eval_batches: int,
    warmup_train_steps: int,
    autocast_dtype: torch.dtype | None,
    *,
    pad_token: int,
    device: torch.device,
    device_type: str,
) -> None:
    if warmup_eval_batches > 0:
        evaluate_model(
            model,
            eval_batches,
            warmup_eval_batches,
            autocast_dtype,
            pad_token=pad_token,
            device=device,
            device_type=device_type,
        )
    if warmup_train_steps > 0:
        model.train()
        for step in range(warmup_train_steps):
            batch = materialize_batch(train_batches[step % len(train_batches)], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model.forward_logits(batch.input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.target_ids.reshape(-1),
                    ignore_index=pad_token,
                )
            loss.backward()
        optimizer.zero_grad(set_to_none=True)


def run_training_benchmark(
    *,
    model: LanguageModelProtocol,
    optimizer: torch.optim.Optimizer,
    train_batches: list[TokenBatch],
    eval_batches: list[TokenBatch],
    train_steps: int,
    eval_batch_count: int,
    autocast_dtype: torch.dtype | None,
    pad_token: int,
    device: torch.device,
    report_model_label: str,
    implementation_kind: str,
    note: str,
    config_payload: dict[str, object],
    corpus_payload: dict[str, object],
) -> BenchmarkReport:
    device_type = device.type
    baseline_process_memory = process_peak_rss_bytes()
    baseline_cuda_used = 0
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_cuda_used, _ = cuda_memory_stats(device)
    peak_process_memory = baseline_process_memory

    total_start = time.perf_counter()
    initial_eval_start = time.perf_counter()
    initial_eval = evaluate_model(
        model,
        eval_batches,
        eval_batch_count,
        autocast_dtype,
        pad_token=pad_token,
        device=device,
        device_type=device_type,
    )
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    initial_eval_wall_time_ms = (time.perf_counter() - initial_eval_start) * 1000.0
    peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())

    seen_tokens = 0
    train_step_reports: list[TrainStepRecord] = []
    model.train()
    train_start = time.perf_counter()
    for step in range(train_steps):
        batch = materialize_batch(train_batches[step % len(train_batches)], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model.forward_logits(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_ids.reshape(-1),
                ignore_index=pad_token,
            )
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().float().item())
        seen_tokens += batch.token_count
        train_step_reports.append(
            TrainStepRecord(
                step=step + 1,
                learning_rate=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                train_perplexity=perplexity_from_loss(train_loss),
                seen_tokens=seen_tokens,
            )
        )
        peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    train_wall_time_ms = (time.perf_counter() - train_start) * 1000.0

    final_eval_start = time.perf_counter()
    final_eval = evaluate_model(
        model,
        eval_batches,
        eval_batch_count,
        autocast_dtype,
        pad_token=pad_token,
        device=device,
        device_type=device_type,
    )
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    final_eval_wall_time_ms = (time.perf_counter() - final_eval_start) * 1000.0
    total_wall_time_ms = (time.perf_counter() - total_start) * 1000.0
    peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())

    selected_eval_tokens = sum(batch.token_count for batch in eval_batches[:eval_batch_count])
    cuda_memory_payload = None
    if device_type == "cuda":
        _, peak_cuda_used = cuda_memory_stats(device)
        device_props = torch.cuda.get_device_properties(device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        cuda_memory_payload = {
            "device_index": int(device_index),
            "device_name": str(device_props.name),
            "compute_capability": f"{int(device_props.major)}.{int(device_props.minor)}",
            "multiprocessor_count": int(device_props.multi_processor_count),
            "memory_metric": "torch_max_memory_allocated",
            "peak_used_bytes": int(peak_cuda_used),
            "peak_used_delta_bytes": int(max(0, peak_cuda_used - baseline_cuda_used)),
            "total_bytes": int(device_props.total_memory),
            "note": "CUDA device memory metrics are sampled from torch.cuda.max_memory_allocated for the active device",
        }

    runtime = RuntimeSummary(
        total_wall_time_ms=total_wall_time_ms,
        initial_eval_wall_time_ms=initial_eval_wall_time_ms,
        train_wall_time_ms=train_wall_time_ms,
        final_eval_wall_time_ms=final_eval_wall_time_ms,
        train_tokens_seen=seen_tokens,
        eval_tokens_per_pass=selected_eval_tokens,
        train_tokens_per_second=seen_tokens / (train_wall_time_ms / 1000.0),
        overall_tokens_per_second=(seen_tokens + 2 * selected_eval_tokens)
        / (total_wall_time_ms / 1000.0),
        process_memory_metric="peak_rss",
        peak_process_memory_bytes=peak_process_memory,
        peak_process_memory_delta_bytes=max(0, peak_process_memory - baseline_process_memory),
        cuda_device_memory=cuda_memory_payload,
        memory_note="process memory metrics are sampled via getrusage(RUSAGE_SELF).ru_maxrss; CUDA memory uses torch.cuda.max_memory_allocated",
    )
    return BenchmarkReport(
        model_label=report_model_label,
        implementation_kind=implementation_kind,
        note=note,
        config=config_payload,
        corpus=corpus_payload,
        initial_eval=initial_eval,
        final_eval=final_eval,
        runtime=runtime,
        train_steps=train_step_reports,
    )
