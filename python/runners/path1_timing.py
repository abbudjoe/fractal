from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from python.data import load_byte_corpus, load_tokenized_corpus
from python.models.path1 import build_path1_model
from python.runtime import (
    CudaEventTimingCollector,
    apply_runtime_policy,
    build_optimizer as build_runtime_optimizer,
    configure_reproducibility,
    cuda_timing_summary_to_dict,
    language_model_cross_entropy,
    materialize_batch,
    resolve_autocast_dtype,
    resolve_torch_device,
    timed_region,
    use_cuda_timing,
    warmup_model,
)
from python.runners.path1 import Path1RunnerRequest
from python.runners.path1_cli import build_parser, build_request_from_args
from python.specs.common import BenchmarkBudgetSpec, JsonlCorpusSpec, TokenIdCorpusSpec, ValidationError, repo_relative
from python.specs.path1 import BYTE_LEVEL_PAD_TOKEN


@dataclass(frozen=True)
class Path1CudaTimingReport:
    run_label: str
    variant_label: str
    implementation_kind: str
    device: str
    dtype: str
    primitive_runtime_backend: str | None
    optimizer_profile: str
    cuda_graph_step: bool
    timing_steps: int
    train_losses: list[float]
    timing_summary: dict[str, dict[str, Any]]
    wall_timing_summary: dict[str, dict[str, Any]]
    derived_summary: dict[str, Any]
    wall_derived_summary: dict[str, Any]
    diagnostics: dict[str, Any]
    output_dir: str
    timing_json_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_label": self.run_label,
            "variant_label": self.variant_label,
            "implementation_kind": self.implementation_kind,
            "device": self.device,
            "dtype": self.dtype,
            "primitive_runtime_backend": self.primitive_runtime_backend,
            "optimizer_profile": self.optimizer_profile,
            "cuda_graph_step": self.cuda_graph_step,
            "timing_steps": self.timing_steps,
            "train_losses": self.train_losses,
            "timing_summary": self.timing_summary,
            "wall_timing_summary": self.wall_timing_summary,
            "derived_summary": self.derived_summary,
            "wall_derived_summary": self.wall_derived_summary,
            "diagnostics": self.diagnostics,
            "output_dir": self.output_dir,
            "timing_json_path": self.timing_json_path,
        }


def _load_corpus_and_pad_token(request: Path1RunnerRequest):
    if isinstance(request.manifest.corpus, JsonlCorpusSpec):
        corpus = load_byte_corpus(
            request.manifest.corpus,
            seq_len=request.manifest.budget.seq_len,
            window_stride=request.manifest.budget.window_stride,
            batch_size=request.manifest.budget.batch_size,
            data_seed=request.manifest.seed_spec.data_seed,
            shuffle_train=request.manifest.seed_spec.data_seed is not None,
            pin_memory=request.manifest.runtime.backend == "cuda",
        )
        return corpus, BYTE_LEVEL_PAD_TOKEN
    if isinstance(request.manifest.corpus, TokenIdCorpusSpec):
        corpus = load_tokenized_corpus(
            request.manifest.corpus,
            seq_len=request.manifest.budget.seq_len,
            window_stride=request.manifest.budget.window_stride,
            batch_size=request.manifest.budget.batch_size,
            data_seed=request.manifest.seed_spec.data_seed,
            shuffle_train=request.manifest.seed_spec.data_seed is not None,
            pin_memory=request.manifest.runtime.backend == "cuda",
        )
        corpus_vocab_size = int(corpus.corpus_stats["vocab_size"])
        if request.variant.shape.vocab_size != corpus_vocab_size:
            raise ValidationError(
                "token-id corpus vocab_size must match Path1ModelShape.vocab_size: "
                f"corpus={corpus_vocab_size}, model={request.variant.shape.vocab_size}"
            )
        return corpus, int(corpus.corpus_stats.get("pad_token_id", -100))
    raise TypeError(f"unsupported corpus spec type: {type(request.manifest.corpus)!r}")


def _sum_prefix(timing_summary: dict[str, dict[str, Any]], prefix: str) -> float:
    return sum(
        float(payload["total_ms"])
        for name, payload in timing_summary.items()
        if name.startswith(prefix)
    )


def _region_total(timing_summary: dict[str, dict[str, Any]], name: str) -> float | None:
    payload = timing_summary.get(name)
    if payload is None:
        return None
    return float(payload["total_ms"])


def _sum_suffix(timing_summary: dict[str, dict[str, Any]], *, prefix: str, suffix: str) -> float | None:
    total = 0.0
    matched = False
    for name, payload in timing_summary.items():
        if name.startswith(prefix) and name.endswith(suffix):
            total += float(payload["total_ms"])
            matched = True
    return total if matched else None


def _coalesce_region_total(
    timing_summary: dict[str, dict[str, Any]],
    *,
    single_name: str,
    band_suffix: str,
) -> float | None:
    single_total = _region_total(timing_summary, single_name)
    band_total = _sum_suffix(timing_summary, prefix="path1.parcae.band", suffix=band_suffix)
    if single_total is None:
        return band_total
    if band_total is None:
        return single_total
    return single_total + band_total


def _derived_summary(timing_summary: dict[str, dict[str, Any]], *, timing_kind: str) -> dict[str, Any]:
    step_total = timing_summary.get("path1.train.step_total", {}).get("total_ms")
    forward_total = timing_summary.get("path1.train.forward", {}).get("total_ms")
    backward_total = timing_summary.get("path1.train.backward", {}).get("total_ms")
    optimizer_total = timing_summary.get("path1.train.optimizer_step", {}).get("total_ms")
    cuda_graph_replay_total = timing_summary.get("path1.cuda_graph.replay", {}).get("total_ms")
    loss_total = timing_summary.get("path1.train.loss", {}).get("total_ms")
    lm_head_loss_total = timing_summary.get("path1.lm_head.loss_total", {}).get("total_ms")
    lm_head_output_projection_total = timing_summary.get("path1.lm_head.output_projection", {}).get("total_ms")
    denominator = float(step_total or 0.0)

    def share(total: float | None) -> float | None:
        if total is None or denominator <= 0.0:
            return None
        return float(total) / denominator

    parcae_total = _sum_prefix(timing_summary, "path1.parcae.")
    attention_total = _sum_prefix(timing_summary, "path1.attention.")
    primitive_total = _sum_prefix(timing_summary, "path1.primitive.")
    parcae_regions = {
        "prelude_blocks": _region_total(timing_summary, "path1.parcae.prelude_blocks"),
        "wide_blocks_before": _sum_suffix(
            timing_summary,
            prefix="path1.parcae.band",
            suffix=".wide_blocks_before",
        ),
        "final_wide_blocks": _region_total(timing_summary, "path1.parcae.final_wide_blocks"),
        "band_prepare": _sum_suffix(timing_summary, prefix="path1.parcae.band", suffix=".prepare"),
        "loop_input_projection": _region_total(timing_summary, "path1.parcae.loop_input_projection"),
        "injection": _region_total(timing_summary, "path1.parcae.injection"),
        "injection_control_position": _region_total(timing_summary, "path1.parcae.injection_control_position"),
        "injection_p20_scan": _region_total(timing_summary, "path1.parcae.injection_p20_scan"),
        "injection_p20_post_compiled": _region_total(
            timing_summary,
            "path1.parcae.injection_p20_post_compiled",
        ),
        "loop_step": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.loop_step",
            band_suffix=".loop_step",
        ),
        "state_mix": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.state_mix",
            band_suffix=".state_mix",
        ),
        "recurrent_blocks": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.recurrent_blocks",
            band_suffix=".recurrent_blocks",
        ),
        "recurrent_block_forward": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.recurrent_block_forward",
            band_suffix=".recurrent_block_forward",
        ),
        "recurrent_residual_mix": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.recurrent_residual_mix",
            band_suffix=".recurrent_residual_mix",
        ),
        "triton_loop_update_forward": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.triton_loop_update_forward",
            band_suffix=".triton_loop_update_forward",
        ),
        "detach_truncated_history": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.detach_truncated_history",
            band_suffix=".detach_truncated_history",
        ),
        "loop_output_projection": _coalesce_region_total(
            timing_summary,
            single_name="path1.parcae.loop_output_projection",
            band_suffix=".loop_output_projection",
        ),
        "coda_blocks": _region_total(timing_summary, "path1.parcae.coda_blocks"),
    }
    attention_regions = {
        "qkv_projection": _region_total(timing_summary, "path1.attention.qkv_projection"),
        "flex_local": _region_total(timing_summary, "path1.attention.flex_local"),
        "flash_local": _region_total(timing_summary, "path1.attention.flash_local"),
        "sdpa": _region_total(timing_summary, "path1.attention.sdpa"),
        "output_projection": _region_total(timing_summary, "path1.attention.output_projection"),
        "input_norm": _region_total(timing_summary, "path1.attention.input_norm"),
        "feedforward": _region_total(timing_summary, "path1.attention.feedforward"),
        "feedforward_compiled": _region_total(timing_summary, "path1.attention.feedforward_compiled"),
        "full_block_compiled": _region_total(timing_summary, "path1.attention.full_block_compiled"),
    }
    native_target_totals = {
        "parcae_prepare_and_control": sum(
            total
            for total in (
                parcae_regions["band_prepare"],
                parcae_regions["loop_input_projection"],
                parcae_regions["injection"],
            )
            if total is not None
        ),
        "rgrp_scan_control": parcae_regions["injection_p20_scan"],
        "loop_update_glue": sum(
            total
            for total in (
                parcae_regions["state_mix"],
                parcae_regions["recurrent_residual_mix"],
                parcae_regions["triton_loop_update_forward"],
            )
            if total is not None
        ),
        "recurrent_block_region": parcae_regions["recurrent_blocks"],
        "recurrent_block_forward_calls": parcae_regions["recurrent_block_forward"],
        "attention_qkv": attention_regions["qkv_projection"],
        "attention_kernel": sum(
            total
            for total in (
                attention_regions["flex_local"],
                attention_regions["flash_local"],
                attention_regions["sdpa"],
            )
            if total is not None
        ),
        "attention_output_projection": attention_regions["output_projection"],
        "attention_ffn": sum(
            total
            for total in (
                attention_regions["feedforward"],
                attention_regions["feedforward_compiled"],
            )
            if total is not None
        ),
        "loop_output_projection": parcae_regions["loop_output_projection"],
    }
    native_target_totals = {
        name: total for name, total in native_target_totals.items() if total is not None and total > 0.0
    }
    native_target_step_shares = {
        name: share(total) for name, total in native_target_totals.items()
    }
    native_target_rank = sorted(
        (
            {
                "target": name,
                "total_ms": total,
                "step_share": native_target_step_shares.get(name),
            }
            for name, total in native_target_totals.items()
        ),
        key=lambda item: item["total_ms"],
        reverse=True,
    )
    parcae_region_step_shares = {
        name: share(total) for name, total in parcae_regions.items() if total is not None
    }
    attention_region_step_shares = {
        name: share(total) for name, total in attention_regions.items() if total is not None
    }
    return {
        "timing_kind": timing_kind,
        "step_total_ms": step_total,
        "forward_total_ms": forward_total,
        "backward_total_ms": backward_total,
        "optimizer_total_ms": optimizer_total,
        "cuda_graph_replay_total_ms": cuda_graph_replay_total,
        "loss_total_ms": loss_total,
        "lm_head_loss_total_ms": lm_head_loss_total,
        "lm_head_output_projection_total_ms": lm_head_output_projection_total,
        "forward_step_share": share(forward_total),
        "backward_step_share": share(backward_total),
        "optimizer_step_share": share(optimizer_total),
        "cuda_graph_replay_step_share": share(cuda_graph_replay_total),
        "loss_step_share": share(loss_total),
        "lm_head_loss_step_share": share(lm_head_loss_total),
        "lm_head_output_projection_step_share": share(lm_head_output_projection_total),
        "parcae_inclusive_total_ms": parcae_total,
        "attention_inclusive_total_ms": attention_total,
        "primitive_inclusive_total_ms": primitive_total,
        "parcae_step_share": share(parcae_total),
        "attention_step_share": share(attention_total),
        "primitive_step_share": share(primitive_total),
        "parcae_region_totals_ms": {
            name: total for name, total in parcae_regions.items() if total is not None
        },
        "parcae_region_step_shares": parcae_region_step_shares,
        "attention_region_totals_ms": {
            name: total for name, total in attention_regions.items() if total is not None
        },
        "attention_region_step_shares": attention_region_step_shares,
        "native_target_totals_ms": native_target_totals,
        "native_target_step_shares": native_target_step_shares,
        "native_target_rank": native_target_rank,
        "note": "Nested timing regions are inclusive; shares can exceed 1.0 when summed across nested scopes.",
    }


def _build_optimizer(
    model: torch.nn.Module,
    *,
    budget: BenchmarkBudgetSpec,
    device: torch.device,
    cuda_graph_step: bool,
) -> torch.optim.Optimizer:
    return build_runtime_optimizer(
        model,
        budget,
        capturable=cuda_graph_step and device.type == "cuda",
    )


def _run_eager_timed_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: Sequence[Any],
    timing_steps: int,
    autocast_dtype: torch.dtype | None,
    pad_token: int,
    device: torch.device,
) -> list[float]:
    train_losses: list[float] = []
    for step in range(timing_steps):
        with timed_region("path1.train.step_total"):
            with timed_region("path1.train.materialize_batch"):
                batch = materialize_batch(train_batches[step % len(train_batches)], device)
            with timed_region("path1.train.zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with timed_region("path1.train.forward"):
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    loss = language_model_cross_entropy(
                        model,
                        batch.input_ids,
                        batch.target_ids,
                        pad_token=pad_token,
                        loss_region_name="path1.train.loss",
                    )
            with timed_region("path1.train.backward"):
                loss.backward()
            with timed_region("path1.train.optimizer_step"):
                optimizer.step()
            if device.type == "cuda":
                with timed_region("path1.train.synchronize_step"):
                    torch.cuda.synchronize(device)
        train_losses.append(float(loss.detach().float().item()))
    return train_losses


def _run_cuda_graph_timed_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_batches: Sequence[Any],
    timing_steps: int,
    autocast_dtype: torch.dtype | None,
    pad_token: int,
    device: torch.device,
) -> list[float]:
    if device.type != "cuda":
        raise ValidationError("cuda_graph_step requires runtime.backend=cuda")
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise ValidationError("cuda_graph_step requires torch.cuda.CUDAGraph support")
    if not train_batches:
        raise ValidationError("cuda_graph_step requires at least one train batch")

    first_batch = materialize_batch(train_batches[0], device)
    static_input_ids = torch.empty_like(first_batch.input_ids, device=device)
    static_target_ids = torch.empty_like(first_batch.target_ids, device=device)
    static_input_ids.copy_(first_batch.input_ids, non_blocking=True)
    static_target_ids.copy_(first_batch.target_ids, non_blocking=True)

    capture_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device)

    # Initialize optimizer state and gradient buffers outside capture. Run the
    # warmup on the same non-default stream we use for capture so the graph does
    # not inherit accidental dependencies from the legacy/default stream.
    with use_cuda_timing(None):
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            optimizer.zero_grad(set_to_none=False)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                warmup_loss = language_model_cross_entropy(
                    model,
                    static_input_ids,
                    static_target_ids,
                    pad_token=pad_token,
                )
            warmup_loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)
        current_stream.wait_stream(capture_stream)
        current_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream.wait_stream(current_stream)
        with torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"):
            optimizer.zero_grad(set_to_none=False)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                static_loss = language_model_cross_entropy(
                    model,
                    static_input_ids,
                    static_target_ids,
                    pad_token=pad_token,
                )
            static_loss.backward()
            optimizer.step()
        current_stream.wait_stream(capture_stream)
    current_stream.synchronize()

    train_losses: list[float] = []
    for step in range(timing_steps):
        with timed_region("path1.train.step_total"):
            with timed_region("path1.train.materialize_batch"):
                batch = materialize_batch(train_batches[step % len(train_batches)], device)
                static_input_ids.copy_(batch.input_ids, non_blocking=True)
                static_target_ids.copy_(batch.target_ids, non_blocking=True)
            with timed_region("path1.cuda_graph.replay"):
                graph.replay()
            with timed_region("path1.train.synchronize_step"):
                torch.cuda.synchronize(device)
        train_losses.append(float(static_loss.detach().float().item()))
    return train_losses


def time_path1_request(
    request: Path1RunnerRequest,
    *,
    output_dir: Path,
    timing_steps: int,
    cuda_graph_step: bool = False,
) -> Path1CudaTimingReport:
    request.manifest.validate()
    request.variant.validate()
    if timing_steps <= 0:
        raise ValidationError(f"timing_steps must be positive, got {timing_steps}")
    if cuda_graph_step and request.manifest.runtime.backend != "cuda":
        raise ValidationError("cuda_graph_step requires runtime.backend=cuda")

    configure_reproducibility(request.manifest.seed_spec, request.manifest.runtime)
    device = resolve_torch_device(request.manifest.runtime)
    autocast_dtype = resolve_autocast_dtype(request.manifest.runtime)
    corpus, pad_token = _load_corpus_and_pad_token(request)
    model = build_path1_model(request.variant, dtype_mode=request.manifest.runtime.dtype).to(device)
    model = apply_runtime_policy(model, request.manifest.runtime)
    optimizer = _build_optimizer(
        model,
        budget=request.manifest.budget,
        device=device,
        cuda_graph_step=cuda_graph_step,
    )
    warmup_model(
        model,
        optimizer,
        corpus.train_batches,
        corpus.eval_batches,
        min(request.manifest.budget.warmup_eval_batches, len(corpus.eval_batches)),
        request.manifest.budget.warmup_train_steps,
        autocast_dtype,
        pad_token=pad_token,
        device=device,
        device_type=device.type,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    collector = CudaEventTimingCollector(enabled=device.type == "cuda")
    model.train()
    with use_cuda_timing(collector):
        train_losses = (
            _run_cuda_graph_timed_steps(
                model=model,
                optimizer=optimizer,
                train_batches=corpus.train_batches,
                timing_steps=timing_steps,
                autocast_dtype=autocast_dtype,
                pad_token=pad_token,
                device=device,
            )
            if cuda_graph_step
            else _run_eager_timed_steps(
                model=model,
                optimizer=optimizer,
                train_batches=corpus.train_batches,
                timing_steps=timing_steps,
                autocast_dtype=autocast_dtype,
                pad_token=pad_token,
                device=device,
            )
        )

    timing_summary = cuda_timing_summary_to_dict(collector.summarize())
    wall_timing_summary = cuda_timing_summary_to_dict(collector.summarize_wall())
    diagnostic_payload = getattr(model, "diagnostic_payload", None)
    diagnostics = diagnostic_payload() if callable(diagnostic_payload) else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    timing_json_path = output_dir / "cuda_timing.json"
    report = Path1CudaTimingReport(
        run_label=request.manifest.run_label,
        variant_label=request.variant.label,
        implementation_kind=request.manifest.implementation_kind,
        device=str(device),
        dtype=request.manifest.runtime.dtype,
        primitive_runtime_backend=request.manifest.runtime.primitive_runtime_backend,
        optimizer_profile=request.manifest.budget.optimizer_profile,
        cuda_graph_step=cuda_graph_step,
        timing_steps=timing_steps,
        train_losses=train_losses,
        timing_summary=timing_summary,
        wall_timing_summary=wall_timing_summary,
        derived_summary=_derived_summary(timing_summary, timing_kind="cuda_event_inclusive_regions"),
        wall_derived_summary=_derived_summary(wall_timing_summary, timing_kind="wall_clock_inclusive_regions"),
        diagnostics=diagnostics,
        output_dir=repo_relative(output_dir),
        timing_json_path=repo_relative(timing_json_path),
    )
    timing_json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def cli_main(argv: Sequence[str] | None = None, *, repo_root: Path) -> int:
    parser = build_parser()
    parser.description = "Run multi-step CUDA-event timing for one Python Path 1 variant."
    parser.add_argument("--timing-steps", type=int, default=20)
    parser.add_argument("--timing-output-dir", type=Path)
    parser.add_argument(
        "--cuda-graph-step",
        action="store_true",
        help="Capture and replay one fixed-shape CUDA training step with torch.cuda.CUDAGraph.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        request = build_request_from_args(args, parser=parser, repo_root=repo_root)
        output_dir = args.timing_output_dir or (
            repo_root / "artifacts" / "v3a-python-path1-cuda-timing" / request.variant.label
        )
        report = time_path1_request(
            request,
            output_dir=output_dir,
            timing_steps=args.timing_steps,
            cuda_graph_step=args.cuda_graph_step,
        )
    except ValidationError as exc:
        parser.error(str(exc))

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0
