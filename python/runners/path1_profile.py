from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from python.data.byte_corpus import load_byte_corpus
from python.models.path1 import build_path1_model
from python.runtime import configure_reproducibility, materialize_batch, resolve_autocast_dtype, resolve_torch_device, warmup_model
from python.runners.path1 import Path1RunnerRequest
from python.runners.path1_cli import build_parser, build_request_from_args
from python.specs.common import ValidationError, repo_relative
from python.specs.path1 import BYTE_LEVEL_PAD_TOKEN


@dataclass(frozen=True)
class ProfileEventSummary:
    key: str
    count: int
    self_cpu_time_total_us: float
    cpu_time_total_us: float
    self_cuda_time_total_us: float
    cuda_time_total_us: float


@dataclass(frozen=True)
class ProfileDerivedSummary:
    metric_name: str
    dominant_named_event: str | None
    dominant_named_event_share: float
    scan_runtime_vs_prepare_runtime_ratio: float | None
    attention_vs_primitive_ratio: float | None
    native_mamba_share: float
    named_region_totals: dict[str, float]


@dataclass(frozen=True)
class Path1ProfileReport:
    run_label: str
    variant_label: str
    benchmark_name: str | None
    implementation_kind: str
    device: str
    dtype: str
    train_loss: float
    sort_by: str
    row_limit: int
    profile_table: str
    named_events: list[ProfileEventSummary]
    derived_summary: ProfileDerivedSummary
    output_dir: str
    profile_json_path: str
    profile_table_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_label": self.run_label,
            "variant_label": self.variant_label,
            "benchmark_name": self.benchmark_name,
            "implementation_kind": self.implementation_kind,
            "device": self.device,
            "dtype": self.dtype,
            "train_loss": self.train_loss,
            "sort_by": self.sort_by,
            "row_limit": self.row_limit,
            "profile_table": self.profile_table,
            "named_events": [
                {
                    "key": event.key,
                    "count": event.count,
                    "self_cpu_time_total_us": event.self_cpu_time_total_us,
                    "cpu_time_total_us": event.cpu_time_total_us,
                    "self_cuda_time_total_us": event.self_cuda_time_total_us,
                    "cuda_time_total_us": event.cuda_time_total_us,
                }
                for event in self.named_events
            ],
            "derived_summary": {
                "metric_name": self.derived_summary.metric_name,
                "dominant_named_event": self.derived_summary.dominant_named_event,
                "dominant_named_event_share": self.derived_summary.dominant_named_event_share,
                "scan_runtime_vs_prepare_runtime_ratio": self.derived_summary.scan_runtime_vs_prepare_runtime_ratio,
                "attention_vs_primitive_ratio": self.derived_summary.attention_vs_primitive_ratio,
                "native_mamba_share": self.derived_summary.native_mamba_share,
                "named_region_totals": self.derived_summary.named_region_totals,
            },
            "output_dir": self.output_dir,
            "profile_json_path": self.profile_json_path,
            "profile_table_path": self.profile_table_path,
        }


def _named_event_summaries(key_averages: list[Any]) -> list[ProfileEventSummary]:
    summaries: list[ProfileEventSummary] = []
    for event in key_averages:
        if not str(event.key).startswith("path1."):
            continue
        summaries.append(
            ProfileEventSummary(
                key=str(event.key),
                count=int(event.count),
                self_cpu_time_total_us=float(getattr(event, "self_cpu_time_total", 0.0)),
                cpu_time_total_us=float(getattr(event, "cpu_time_total", 0.0)),
                self_cuda_time_total_us=float(getattr(event, "self_cuda_time_total", 0.0)),
                cuda_time_total_us=float(getattr(event, "cuda_time_total", 0.0)),
            )
        )
    return summaries


def _event_metric(event: ProfileEventSummary, device_type: str) -> float:
    if device_type == "cuda":
        return event.self_cuda_time_total_us
    return event.self_cpu_time_total_us


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _derived_summary(named_events: list[ProfileEventSummary], device_type: str) -> ProfileDerivedSummary:
    metric_name = "self_cuda_time_total_us" if device_type == "cuda" else "self_cpu_time_total_us"
    totals: dict[str, float] = {}
    total_named = 0.0
    dominant_event: str | None = None
    dominant_value = 0.0

    for event in named_events:
        value = _event_metric(event, device_type)
        total_named += value
        if value > dominant_value:
            dominant_value = value
            dominant_event = event.key
        if event.key.startswith("path1.attention."):
            totals["attention"] = totals.get("attention", 0.0) + value
        elif event.key.startswith("path1.reference_ssm."):
            totals["reference_ssm"] = totals.get("reference_ssm", 0.0) + value
        elif event.key.startswith("path1.primitive."):
            totals["primitive"] = totals.get("primitive", 0.0) + value
        else:
            totals["other"] = totals.get("other", 0.0) + value
        totals[event.key] = value

    scan_runtime = totals.get("path1.primitive.scan_runtime", 0.0)
    prepare_runtime = totals.get("path1.primitive.prepare_runtime_plan", 0.0)
    attention_total = totals.get("attention", 0.0)
    primitive_total = totals.get("primitive", 0.0)
    native_mamba = totals.get("path1.reference_ssm.native_mamba3", 0.0)

    return ProfileDerivedSummary(
        metric_name=metric_name,
        dominant_named_event=dominant_event,
        dominant_named_event_share=0.0 if total_named <= 0.0 else dominant_value / total_named,
        scan_runtime_vs_prepare_runtime_ratio=_ratio_or_none(scan_runtime, prepare_runtime),
        attention_vs_primitive_ratio=_ratio_or_none(attention_total, primitive_total),
        native_mamba_share=0.0 if total_named <= 0.0 else native_mamba / total_named,
        named_region_totals=totals,
    )


def profile_path1_request(
    request: Path1RunnerRequest,
    *,
    output_dir: Path,
    row_limit: int = 30,
) -> Path1ProfileReport:
    request.manifest.validate()
    request.variant.validate()

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
    model = build_path1_model(request.variant, dtype_mode=request.manifest.runtime.dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=request.manifest.budget.learning_rate)
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

    batch = materialize_batch(corpus.train_batches[0], device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        activities.append(ProfilerActivity.CUDA)

    model.train()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=(device.type == "cuda"),
        with_stack=False,
    ) as prof:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            logits = model.forward_logits(batch.input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch.target_ids.reshape(-1),
                ignore_index=BYTE_LEVEL_PAD_TOKEN,
            )
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    key_averages = list(prof.key_averages())
    profile_table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    named_events = _named_event_summaries(key_averages)

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_json_path = output_dir / "profile.json"
    profile_table_path = output_dir / "profile.txt"

    report = Path1ProfileReport(
        run_label=request.manifest.run_label,
        variant_label=request.variant.label,
        benchmark_name=request.manifest.benchmark_name,
        implementation_kind=request.manifest.implementation_kind,
        device=str(device),
        dtype=request.manifest.runtime.dtype,
        train_loss=float(loss.detach().float().item()),
        sort_by=sort_by,
        row_limit=row_limit,
        profile_table=profile_table,
        named_events=named_events,
        derived_summary=_derived_summary(named_events, device.type),
        output_dir=repo_relative(output_dir),
        profile_json_path=repo_relative(profile_json_path),
        profile_table_path=repo_relative(profile_table_path),
    )
    profile_json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile_table_path.write_text(profile_table + "\n", encoding="utf-8")
    return report


def cli_main(argv: Sequence[str] | None = None, *, repo_root: Path) -> int:
    parser = build_parser()
    parser.description = "Profile one Python Path 1 train step on the shared research substrate."
    parser.add_argument("--profile-row-limit", type=int, default=30)
    parser.add_argument("--profile-output-dir", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        request = build_request_from_args(args, parser=parser, repo_root=repo_root)
        output_dir = args.profile_output_dir or (
            repo_root / "artifacts" / "v3a-python-path1-profile" / request.variant.label
        )
        report = profile_path1_request(request, output_dir=output_dir, row_limit=args.profile_row_limit)
    except ValidationError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "run_label": report.run_label,
                "variant_label": report.variant_label,
                "train_loss": report.train_loss,
                "sort_by": report.sort_by,
                "profile_json_path": report.profile_json_path,
                "profile_table_path": report.profile_table_path,
                "named_events": [event.__dict__ for event in report.named_events],
                "derived_summary": {
                    "metric_name": report.derived_summary.metric_name,
                    "dominant_named_event": report.derived_summary.dominant_named_event,
                    "dominant_named_event_share": report.derived_summary.dominant_named_event_share,
                    "scan_runtime_vs_prepare_runtime_ratio": report.derived_summary.scan_runtime_vs_prepare_runtime_ratio,
                    "attention_vs_primitive_ratio": report.derived_summary.attention_vs_primitive_ratio,
                    "native_mamba_share": report.derived_summary.native_mamba_share,
                    "named_region_totals": report.derived_summary.named_region_totals,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
