#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None

    class _DummyModule:
        pass

    class _DummyNN:
        Module = _DummyModule

    nn = _DummyNN()
    F = None


BYTE_LEVEL_PAD_TOKEN = 0
BYTE_LEVEL_VOCAB_SIZE = 257
DEFAULT_SEQ_LEN = 32
DEFAULT_WINDOW_STRIDE = DEFAULT_SEQ_LEN
DEFAULT_BATCH_SIZE = 1
DEFAULT_STEPS = 8
DEFAULT_EVAL_BATCHES = 2
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_SEED = 42
DEFAULT_D_MODEL = 128
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 8
DEFAULT_LOCAL_WINDOW = 256
DEFAULT_D_STATE = 128
DEFAULT_EXPAND = 2
DEFAULT_IS_MIMO = False
DEFAULT_MIMO_RANK = 1
DEFAULT_DTYPE = "bf16"
DEFAULT_WARMUP_EVAL_BATCHES = 1
DEFAULT_WARMUP_TRAIN_STEPS = 1
DEFAULT_NOTE = (
    "Path 1 reference SSM hybrid baseline using official PyTorch Mamba3 blocks on the shared "
    "byte-level smoke lane through the upstream SISO runtime path"
)


@dataclass
class TokenBatch:
    input_ids: "torch.Tensor"
    target_ids: "torch.Tensor"
    token_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official PyTorch Mamba3 v3a hybrid benchmark.")
    parser.add_argument("--backend", default="cuda", choices=["cuda"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--window-stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--full-train-pass", action="store_true")
    parser.add_argument("--full-eval-pass", action="store_true")
    parser.add_argument("--warmup-eval-batches", type=int, default=DEFAULT_WARMUP_EVAL_BATCHES)
    parser.add_argument("--warmup-train-steps", type=int, default=DEFAULT_WARMUP_TRAIN_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--jsonl-train-path", type=Path, required=True)
    parser.add_argument("--jsonl-eval-path", type=Path, required=True)
    parser.add_argument("--benchmark-name")
    parser.add_argument("--corpus-name", default="fineweb-stage0-canary")
    parser.add_argument("--corpus-text-field", default="text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default=DEFAULT_DTYPE)
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def load_jsonl_text_documents(path: Path, text_field: str) -> list[bytes]:
    documents: list[bytes] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"failed to parse jsonl text corpus {path} line {line_index}: {exc}") from exc
            text = value.get(text_field)
            if not isinstance(text, str):
                raise SystemExit(
                    f"jsonl text corpus {path} line {line_index} is missing string field {text_field!r}"
                )
            if text:
                documents.append(text.encode("utf-8"))
    if not documents:
        raise SystemExit(f"jsonl text corpus {path} did not yield any non-empty documents")
    return documents


def byte_sequences_from_documents(documents: list[bytes], seq_len: int, window_stride: int) -> list[tuple[list[int], list[int]]]:
    required_len = seq_len + 1
    sequences: list[tuple[list[int], list[int]]] = []
    for raw in documents:
        if len(raw) < required_len:
            continue
        for start in range(0, len(raw) - required_len + 1, window_stride):
            window = raw[start : start + required_len]
            inputs = [int(window[index]) + 1 for index in range(seq_len)]
            targets = [int(window[index + 1]) + 1 for index in range(seq_len)]
            sequences.append((inputs, targets))
    if len(sequences) < 2:
        raise SystemExit(
            f"byte-level corpus must yield at least 2 sequences of length {seq_len}, got {len(sequences)}"
        )
    return sequences


def sequences_into_batches(
    sequences: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
) -> list[TokenBatch]:
    batches: list[TokenBatch] = []
    seq_len = len(sequences[0][0])
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        input_flat = [token for inputs, _ in chunk for token in inputs]
        target_flat = [token for _, targets in chunk for token in targets]
        input_ids = torch.tensor(input_flat, dtype=torch.long, device=device).reshape(len(chunk), seq_len)
        target_ids = torch.tensor(target_flat, dtype=torch.long, device=device).reshape(len(chunk), seq_len)
        batches.append(TokenBatch(input_ids=input_ids, target_ids=target_ids, token_count=len(chunk) * seq_len))
    return batches


def load_corpus_batches(
    train_path: Path,
    eval_path: Path,
    text_field: str,
    seq_len: int,
    window_stride: int,
    batch_size: int,
    device: "torch.device",
) -> tuple[dict[str, object], list[TokenBatch], list[TokenBatch]]:
    train_docs = load_jsonl_text_documents(train_path, text_field)
    eval_docs = load_jsonl_text_documents(eval_path, text_field)
    train_sequences = byte_sequences_from_documents(train_docs, seq_len, window_stride)
    eval_sequences = byte_sequences_from_documents(eval_docs, seq_len, window_stride)
    train_batches = sequences_into_batches(train_sequences, batch_size, device)
    eval_batches = sequences_into_batches(eval_sequences, batch_size, device)
    total_bytes = sum(len(doc) for doc in train_docs) + sum(len(doc) for doc in eval_docs)
    stats = {
        "files": [repo_relative(train_path), repo_relative(eval_path)],
        "total_bytes": total_bytes,
        "total_sequences": sum(batch.input_ids.shape[0] for batch in train_batches)
        + sum(batch.input_ids.shape[0] for batch in eval_batches),
        "train_sequences": sum(batch.input_ids.shape[0] for batch in train_batches),
        "eval_sequences": sum(batch.input_ids.shape[0] for batch in eval_batches),
        "seq_len": seq_len,
        "window_stride": window_stride,
    }
    return stats, train_batches, eval_batches


def process_peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def cuda_memory_stats(device: "torch.device") -> tuple[int, int]:
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)


def local_causal_mask(seq_len: int, local_window: int, device: "torch.device") -> "torch.Tensor":
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for query in range(seq_len):
        earliest_visible = max(0, query - (local_window - 1))
        if earliest_visible > 0:
            mask[query, :earliest_visible] = True
        if query + 1 < seq_len:
            mask[query, query + 1 :] = True
    return mask


class SimpleRmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class OfficialMamba3Block(nn.Module):
    def __init__(self, d_model: int, head_count: int, dtype_mode: str) -> None:
        super().__init__()
        from mamba_ssm import Mamba3

        self.input_norm = SimpleRmsNorm(d_model)
        self.output_norm = SimpleRmsNorm(d_model)
        chunk_size = 16 if dtype_mode == "bf16" else 8
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=DEFAULT_D_STATE,
            headdim=d_model // head_count,
            is_mimo=DEFAULT_IS_MIMO,
            mimo_rank=DEFAULT_MIMO_RANK,
            chunk_size=chunk_size,
            is_outproj_norm=False,
        )
        self.feedforward = PositionWiseFeedForward(d_model, d_model * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mixer(self.input_norm(x))
        residual = x + mixed
        return residual + self.feedforward(self.output_norm(residual))


class HybridAttentionMamba3Model(nn.Module):
    def __init__(self, dtype_mode: str) -> None:
        super().__init__()
        self.vocab_size = BYTE_LEVEL_VOCAB_SIZE
        self.d_model = DEFAULT_D_MODEL
        self.head_count = DEFAULT_HEADS
        self.local_window = DEFAULT_LOCAL_WINDOW
        self.total_layers = DEFAULT_LAYERS
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=BYTE_LEVEL_PAD_TOKEN)
        self.attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.head_count,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(4)
            ]
        )
        self.reference_layers = nn.ModuleList(
            [OfficialMamba3Block(self.d_model, self.head_count, dtype_mode) for _ in range(4)]
        )
        self.final_norm = SimpleRmsNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        seq_len = input_ids.shape[1]
        mask = local_causal_mask(seq_len, self.local_window, input_ids.device)
        attention_index = 0
        reference_index = 0
        for role_index in range(self.total_layers):
            if role_index % 2 == 0:
                hidden = self.attention_layers[attention_index](hidden, src_mask=mask)
                attention_index += 1
            else:
                hidden = self.reference_layers[reference_index](hidden)
                reference_index += 1
        return self.output(self.final_norm(hidden))


def evaluate_model(
    model: HybridAttentionMamba3Model,
    batches: list[TokenBatch],
    eval_batches: int,
    autocast_dtype: "torch.dtype | None",
) -> dict[str, float | int]:
    device = batches[0].input_ids.device
    selected = batches[: min(eval_batches, len(batches))]
    total_loss = 0.0
    total_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in selected:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model.forward_logits(batch.input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.target_ids.reshape(-1),
                    ignore_index=BYTE_LEVEL_PAD_TOKEN,
                )
            total_loss += float(loss.detach().float().item())
            total_batches += 1
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "batch_count": total_batches,
        "mean_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def warmup_model(
    model: HybridAttentionMamba3Model,
    optimizer: "torch.optim.Optimizer",
    train_batches: list[TokenBatch],
    eval_batches: list[TokenBatch],
    warmup_eval_batches: int,
    warmup_train_steps: int,
    autocast_dtype: "torch.dtype | None",
) -> None:
    if warmup_eval_batches > 0:
        evaluate_model(model, eval_batches, warmup_eval_batches, autocast_dtype)
    if warmup_train_steps > 0:
        model.train()
        for step in range(warmup_train_steps):
            batch = train_batches[step % len(train_batches)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model.forward_logits(batch.input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch.target_ids.reshape(-1),
                    ignore_index=BYTE_LEVEL_PAD_TOKEN,
                )
            loss.backward()
        optimizer.zero_grad(set_to_none=True)


def append_ledger_entry(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.backend != "cuda":
        raise SystemExit("official PyTorch Mamba3 hybrid runner currently supports only --backend cuda")

    try:
        global torch, nn, F
        import torch as torch_mod
        import torch.nn as nn_mod
        import torch.nn.functional as f_mod
        torch = torch_mod
        nn = nn_mod
        F = f_mod
        import mamba_ssm  # noqa: F401
    except Exception as exc:
        traceback.print_exception(exc, file=sys.stderr)
        raise SystemExit(
            "official Mamba3 import failed. Install requirements from scripts/requirements-v3a-python-mamba3.txt"
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{args.cuda_device}")
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None
    output_dir = args.output_dir / "python-reference-ssm-hybrid-siso"
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus, train_batches, eval_batches = load_corpus_batches(
        args.jsonl_train_path,
        args.jsonl_eval_path,
        args.corpus_text_field,
        args.seq_len,
        args.window_stride,
        args.batch_size,
        device,
    )
    train_steps = len(train_batches) if args.full_train_pass else args.steps
    eval_batch_count = len(eval_batches) if args.full_eval_pass else args.eval_batches

    model = HybridAttentionMamba3Model(dtype_mode=args.dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    warmup_model(
        model,
        optimizer,
        train_batches,
        eval_batches,
        max(0, min(args.warmup_eval_batches, len(eval_batches))),
        max(0, args.warmup_train_steps),
        autocast_dtype,
    )
    torch.cuda.synchronize(device)

    baseline_process_memory = process_peak_rss_bytes()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_cuda_used, _ = cuda_memory_stats(device)
    peak_process_memory = baseline_process_memory

    total_start = time.perf_counter()
    initial_eval_start = time.perf_counter()
    initial_eval = evaluate_model(model, eval_batches, eval_batch_count, autocast_dtype)
    torch.cuda.synchronize(device)
    initial_eval_wall_time_ms = (time.perf_counter() - initial_eval_start) * 1000.0
    peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())

    seen_tokens = 0
    train_step_reports: list[dict[str, float | int]] = []
    model.train()
    train_start = time.perf_counter()
    for step in range(train_steps):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model.forward_logits(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_ids.reshape(-1),
                ignore_index=BYTE_LEVEL_PAD_TOKEN,
            )
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().float().item())
        seen_tokens += batch.token_count
        train_step_reports.append(
            {
                "step": step + 1,
                "learning_rate": args.learning_rate,
                "train_loss": train_loss,
                "train_perplexity": math.exp(train_loss),
                "seen_tokens": seen_tokens,
            }
        )
        peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())
    torch.cuda.synchronize(device)
    train_wall_time_ms = (time.perf_counter() - train_start) * 1000.0

    final_eval_start = time.perf_counter()
    final_eval = evaluate_model(model, eval_batches, eval_batch_count, autocast_dtype)
    torch.cuda.synchronize(device)
    final_eval_wall_time_ms = (time.perf_counter() - final_eval_start) * 1000.0
    total_wall_time_ms = (time.perf_counter() - total_start) * 1000.0
    peak_process_memory = max(peak_process_memory, process_peak_rss_bytes())
    _, peak_cuda_used = cuda_memory_stats(device)
    device_props = torch.cuda.get_device_properties(device)

    selected_eval_tokens = sum(batch.token_count for batch in eval_batches[:eval_batch_count])
    report = {
        "model_label": "v3a_reference_ssm_python_mamba3_official_siso",
        "implementation_kind": "python_native",
        "note": DEFAULT_NOTE,
        "config": {
            "backend": args.backend,
            "benchmark_name": args.benchmark_name,
            "cuda_device_index": args.cuda_device,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "window_stride": args.window_stride,
            "batch_size": args.batch_size,
            "train_steps": train_steps,
            "eval_batches": eval_batch_count,
            "learning_rate": args.learning_rate,
            "run_label": args.run_label,
            "dtype": args.dtype,
            "warmup_eval_batches": args.warmup_eval_batches,
            "warmup_train_steps": args.warmup_train_steps,
            "corpus_name": args.corpus_name,
            "vocabulary": {
                "pad_token": BYTE_LEVEL_PAD_TOKEN,
                "vocab_size": BYTE_LEVEL_VOCAB_SIZE,
            },
            "schedule": ["attention", "mamba3"] * 4,
            "mamba3": {
                "d_model": DEFAULT_D_MODEL,
                "d_state": DEFAULT_D_STATE,
                "expand": DEFAULT_EXPAND,
                "headdim": DEFAULT_D_MODEL // DEFAULT_HEADS,
                "is_mimo": DEFAULT_IS_MIMO,
                "mimo_rank": DEFAULT_MIMO_RANK,
                "chunk_size": 16 if args.dtype == "bf16" else 8,
                "is_outproj_norm": False,
            },
        },
        "corpus": corpus,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "runtime": {
            "total_wall_time_ms": total_wall_time_ms,
            "initial_eval_wall_time_ms": initial_eval_wall_time_ms,
            "train_wall_time_ms": train_wall_time_ms,
            "final_eval_wall_time_ms": final_eval_wall_time_ms,
            "train_tokens_seen": seen_tokens,
            "eval_tokens_per_pass": selected_eval_tokens,
            "train_tokens_per_second": seen_tokens / (train_wall_time_ms / 1000.0),
            "overall_tokens_per_second": (seen_tokens + 2 * selected_eval_tokens)
            / (total_wall_time_ms / 1000.0),
            "process_memory_metric": "peak_rss",
            "peak_process_memory_bytes": peak_process_memory,
            "peak_process_memory_delta_bytes": max(0, peak_process_memory - baseline_process_memory),
            "cuda_device_memory": {
                "device_index": args.cuda_device,
                "memory_metric": "torch_max_memory_allocated",
                "peak_used_bytes": int(peak_cuda_used),
                "peak_used_delta_bytes": int(max(0, peak_cuda_used - baseline_cuda_used)),
                "total_bytes": int(device_props.total_memory),
                "note": "CUDA device memory metrics are sampled from torch.cuda.max_memory_allocated for the active device",
            },
            "memory_note": "process memory metrics are sampled around train/eval phases via getrusage(RUSAGE_SELF).ru_maxrss; CUDA memory uses torch.cuda.max_memory_allocated",
        },
        "train_steps": train_step_reports,
    }
    report_path = output_dir / "report.json"
    report["report_path"] = repo_relative(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.ledger_path:
        append_ledger_entry(
            args.ledger_path,
            {
                "run_label": args.run_label,
                "variant": "python-reference-ssm-hybrid-siso",
                "model_label": report["model_label"],
                "report_path": repo_relative(report_path),
                "corpus_name": args.corpus_name,
                "seed": args.seed,
                "final_loss": final_eval["mean_loss"],
                "train_tokens_per_second": report["runtime"]["train_tokens_per_second"],
                "peak_cuda_memory_bytes": report["runtime"]["cuda_device_memory"]["peak_used_bytes"],
            },
        )

    if args.output == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "python-reference-ssm-hybrid-siso"
            f"\tinitial_loss={initial_eval['mean_loss']:.4f}"
            f"\tfinal_loss={final_eval['mean_loss']:.4f}"
            f"\ttrain_tok_s={report['runtime']['train_tokens_per_second']:.2f}"
            f"\tcuda_peak_mb={report['runtime']['cuda_device_memory']['peak_used_bytes'] / (1024 * 1024):.2f}"
        )
        print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
