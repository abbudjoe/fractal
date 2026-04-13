from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from python.data.byte_corpus import load_byte_corpus
from python.models.path1 import build_path1_model
from python.reporting.render import render_path1_table
from python.reporting.schema import BenchmarkReport, append_ledger_entry, write_report
from python.runtime import (
    apply_runtime_policy,
    configure_reproducibility,
    resolve_autocast_dtype,
    resolve_torch_device,
    run_training_benchmark,
    warmup_model,
)
from python.specs.common import BenchmarkRunManifest, repo_relative, to_jsonable
from python.specs.path1 import BYTE_LEVEL_PAD_TOKEN, Path1VariantSpec


@dataclass(frozen=True)
class Path1RunnerRequest:
    manifest: BenchmarkRunManifest
    variant: Path1VariantSpec
    output_dir: Path
    output_format: str = "table"
    ledger_path: Path | None = None
    variant_output_name: str | None = None
    model_note: str = ""


def _variant_output_name(request: Path1RunnerRequest) -> str:
    if request.variant_output_name:
        return request.variant_output_name
    return request.variant.label


def _config_payload(request: Path1RunnerRequest, train_steps: int, eval_batches: int) -> dict[str, Any]:
    variant = request.variant
    return {
        "backend": request.manifest.runtime.backend,
        "compile_mode": request.manifest.runtime.compile_mode,
        "env_kind": request.manifest.runtime.env_kind,
        "primitive_runtime_backend": request.manifest.runtime.primitive_runtime_backend,
        "benchmark_name": request.manifest.benchmark_name,
        "seed": request.manifest.seed_spec.model_seed,
        "data_seed": request.manifest.seed_spec.data_seed,
        "seq_len": request.manifest.budget.seq_len,
        "window_stride": request.manifest.budget.window_stride,
        "batch_size": request.manifest.budget.batch_size,
        "train_steps": train_steps,
        "eval_batches": eval_batches,
        "learning_rate": request.manifest.budget.learning_rate,
        "run_label": request.manifest.run_label,
        "dtype": request.manifest.runtime.dtype,
        "warmup_eval_batches": request.manifest.budget.warmup_eval_batches,
        "warmup_train_steps": request.manifest.budget.warmup_train_steps,
        "schedule": [role.value for role in variant.layer_schedule],
        "variant": to_jsonable(variant),
    }


def run_path1_variant(request: Path1RunnerRequest) -> BenchmarkReport:
    request.manifest.validate()
    request.variant.validate()
    if request.output_format not in {"table", "json"}:
        raise ValueError(f"unsupported output format: {request.output_format}")

    configure_reproducibility(request.manifest.seed_spec, request.manifest.runtime)
    device = resolve_torch_device(request.manifest.runtime)
    autocast_dtype = resolve_autocast_dtype(request.manifest.runtime)
    corpus = load_byte_corpus(
        request.manifest.corpus,
        seq_len=request.manifest.budget.seq_len,
        window_stride=request.manifest.budget.window_stride,
        batch_size=request.manifest.budget.batch_size,
        data_seed=request.manifest.seed_spec.data_seed,
        shuffle_train=request.manifest.seed_spec.data_seed is not None,
        pin_memory=request.manifest.runtime.backend == "cuda",
    )
    train_steps = len(corpus.train_batches) if request.manifest.budget.full_train_pass else request.manifest.budget.train_steps
    eval_batch_count = len(corpus.eval_batches) if request.manifest.budget.full_eval_pass else request.manifest.budget.eval_batches

    model = build_path1_model(request.variant, dtype_mode=request.manifest.runtime.dtype).to(device)
    model = apply_runtime_policy(model, request.manifest.runtime)
    optimizer = torch.optim.Adam(
        model.optimizer_parameter_groups(request.manifest.budget.learning_rate),
        lr=request.manifest.budget.learning_rate,
    )
    warmup_model(
        model,
        optimizer,
        corpus.train_batches,
        corpus.eval_batches,
        min(request.manifest.budget.warmup_eval_batches, len(corpus.eval_batches)),
        request.manifest.budget.warmup_train_steps,
        autocast_dtype,
        pad_token=BYTE_LEVEL_PAD_TOKEN,
        device=device,
        device_type=device.type,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    report = run_training_benchmark(
        model=model,
        optimizer=optimizer,
        train_batches=corpus.train_batches,
        eval_batches=corpus.eval_batches,
        train_steps=train_steps,
        eval_batch_count=eval_batch_count,
        autocast_dtype=autocast_dtype,
        pad_token=BYTE_LEVEL_PAD_TOKEN,
        device=device,
        report_model_label=model.model_label,
        implementation_kind=request.manifest.implementation_kind,
        note=request.model_note or request.manifest.note,
        config_payload=_config_payload(request, train_steps, eval_batch_count),
        corpus_payload=corpus.corpus_stats,
    )
    diagnostic_payload = getattr(model, "diagnostic_payload", None)
    if callable(diagnostic_payload):
        report.diagnostics = diagnostic_payload()

    output_dir = request.output_dir / _variant_output_name(request)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    write_report(report, report_path)
    if request.ledger_path is not None:
        append_ledger_entry(
            request.ledger_path,
            {
                "run_label": request.manifest.run_label,
                "variant": request.variant.label,
                "model_label": report.model_label,
                "report_path": repo_relative(report_path),
                "corpus_name": request.manifest.corpus.corpus_name,
                "seed": request.manifest.seed_spec.model_seed,
                "data_seed": request.manifest.seed_spec.data_seed,
                "final_loss": report.final_eval.mean_loss,
                "train_tokens_per_second": report.runtime.train_tokens_per_second,
                "peak_cuda_memory_bytes": (
                    0
                    if report.runtime.cuda_device_memory is None
                    else report.runtime.cuda_device_memory["peak_used_bytes"]
                ),
            },
        )
    return report


def render_report(report: BenchmarkReport, request: Path1RunnerRequest) -> str | dict[str, Any]:
    if request.output_format == "json":
        return report.to_dict()
    return render_path1_table(report, request.variant.label)
