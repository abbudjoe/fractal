#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.data import load_tokenized_corpus
from python.reporting.render import render_path1_table
from python.reporting.schema import append_ledger_entry, write_report
from python.runtime import (
    build_optimizer,
    configure_reproducibility,
    resolve_autocast_dtype,
    resolve_torch_device,
    run_training_benchmark,
    warmup_model,
)
from python.specs.common import (
    BenchmarkBudgetSpec,
    DeviceRuntimeSpec,
    SeedSpec,
    TokenIdCorpusSpec,
    ValidationError,
    repo_relative,
)


@dataclass(frozen=True)
class ExternalModelSpec:
    key: str
    model_id: str
    label: str
    implementation_kind: str
    note: str


EXTERNAL_MODEL_SPECS: dict[str, ExternalModelSpec] = {
    "gpt2-small": ExternalModelSpec(
        key="gpt2-small",
        model_id="openai-community/gpt2",
        label="hf-gpt2-small-architecture",
        implementation_kind="hf_transformers_gpt2_small_architecture",
        note=(
            "Hugging Face GPT-2 small architecture initialized from config and trained "
            "on the shared OpenLLaMA token-id cache; this is not pretrained GPT-2."
        ),
    ),
    "mamba-130m": ExternalModelSpec(
        key="mamba-130m",
        model_id="state-spaces/mamba-130m-hf",
        label="hf-mamba-130m-architecture",
        implementation_kind="hf_transformers_mamba_130m_architecture",
        note=(
            "Hugging Face Mamba-130M architecture initialized from config and trained "
            "on the shared OpenLLaMA token-id cache; this is not a pretrained checkpoint eval."
        ),
    ),
    "official-mamba-130m": ExternalModelSpec(
        key="official-mamba-130m",
        model_id="state-spaces/mamba-130m",
        label="official-mamba-130m-architecture",
        implementation_kind="official_mamba_ssm_130m_architecture",
        note=(
            "Official state-spaces/mamba Mamba-130M architecture initialized from config "
            "and trained on the shared OpenLLaMA token-id cache; this bypasses the HF "
            "Transformers Mamba wrapper and requires native mamba-ssm kernels."
        ),
    ),
}


class ExternalCausalLmWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        model_label: str,
        diagnostics: dict[str, Any],
        pass_use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_label = model_label
        self._diagnostics = diagnostics
        self._pass_use_cache = pass_use_cache

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._pass_use_cache:
            output = self.model(input_ids=input_ids, use_cache=False)
        else:
            output = self.model(input_ids=input_ids)
        logits = getattr(output, "logits", None)
        if logits is None:
            raise RuntimeError(f"{self.model_label} forward did not return logits")
        return logits

    def diagnostic_payload(self) -> dict[str, Any]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            **self._diagnostics,
            "parameter_count": int(total),
            "trainable_parameter_count": int(trainable),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an external Hugging Face causal LM architecture on the shared Path 1 token-cache harness."
    )
    parser.add_argument(
        "--external-model",
        required=True,
        choices=sorted(EXTERNAL_MODEL_SPECS),
        help="External architecture to instantiate from config and train on the shared token-id cache.",
    )
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--env-kind", default="requirements-only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--tokenized-manifest-path", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=513)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--warmup-eval-batches", type=int, default=0)
    parser.add_argument("--warmup-train-steps", type=int, default=0)
    parser.add_argument("--train-loss-record-interval", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--optimizer-profile",
        default="adam",
        choices=["adam", "adam-fused", "adam-triton-2d", "muon-reference"],
    )
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"])
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", choices=["table", "json"], default="table")
    return parser


def _load_manifest_vocab_size(manifest_path: Path) -> int:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"failed to parse tokenized manifest {manifest_path}: {exc}") from exc
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValidationError("tokenized manifest must contain tokenizer metadata")
    vocab_size = tokenizer.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValidationError("tokenized manifest tokenizer.vocab_size must be a positive integer")
    return int(vocab_size)


def _build_external_model(
    spec: ExternalModelSpec,
    *,
    vocab_size: int,
    max_position_embeddings: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if spec.key == "official-mamba-130m":
        return _build_official_mamba_model(
            spec,
            vocab_size=vocab_size,
        )
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for external LM baselines. "
            "Install transformers>=4.51.0 or use the SageMaker token-cache launcher."
        ) from exc

    config = AutoConfig.from_pretrained(spec.model_id, trust_remote_code=False)
    original_vocab_size = getattr(config, "vocab_size", None)
    config.vocab_size = vocab_size
    if hasattr(config, "use_cache"):
        config.use_cache = False
    if hasattr(config, "pad_token_id"):
        config.pad_token_id = None
    if hasattr(config, "bos_token_id"):
        config.bos_token_id = 1
    if hasattr(config, "eos_token_id"):
        config.eos_token_id = 2
    if spec.key == "gpt2-small":
        config.n_positions = max_position_embeddings
        config.n_ctx = max_position_embeddings
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    if spec.key == "mamba-130m":
        _assert_mamba_fast_path_available()
    diagnostics = {
        "external_model_key": spec.key,
        "external_model_id": spec.model_id,
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "original_vocab_size": original_vocab_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "use_cache": getattr(config, "use_cache", None),
        "architecture_note": spec.note,
    }
    return model, diagnostics


def _build_official_mamba_model(
    spec: ExternalModelSpec,
    *,
    vocab_size: int,
) -> tuple[nn.Module, dict[str, Any]]:
    _assert_official_mamba_fast_path_available()
    try:
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        from mamba_ssm.utils.hf import load_config_hf
    except ImportError as exc:
        raise SystemExit(
            "official Mamba baseline requires mamba-ssm. Install mamba-ssm and causal-conv1d "
            "or use the SageMaker token-cache launcher with the official Mamba lane."
        ) from exc
    config_data = load_config_hf(spec.model_id)
    original_vocab_size = config_data.get("vocab_size")
    config_data["vocab_size"] = vocab_size
    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config)
    diagnostics = {
        "external_model_key": spec.key,
        "external_model_id": spec.model_id,
        "config_class": type(config).__name__,
        "model_class": type(model).__name__,
        "original_vocab_size": original_vocab_size,
        "vocab_size": vocab_size,
        "d_model": config.d_model,
        "n_layer": config.n_layer,
        "d_intermediate": config.d_intermediate,
        "ssm_cfg": config.ssm_cfg,
        "rms_norm": config.rms_norm,
        "fused_add_norm": config.fused_add_norm,
        "residual_in_fp32": config.residual_in_fp32,
        "tie_embeddings": config.tie_embeddings,
        "architecture_note": spec.note,
    }
    return model, diagnostics


def _assert_official_mamba_fast_path_available() -> None:
    missing: list[str] = []
    try:
        from mamba_ssm.ops import selective_scan_interface
    except ImportError as exc:
        raise SystemExit(f"failed to import mamba_ssm selective-scan interface: {exc}") from exc
    try:
        from mamba_ssm.modules import mamba_simple
    except ImportError as exc:
        raise SystemExit(f"failed to import mamba_ssm Mamba module: {exc}") from exc
    for name in ("selective_scan_fn", "mamba_inner_fn"):
        if getattr(selective_scan_interface, name, None) is None:
            missing.append(f"mamba_ssm.ops.selective_scan_interface.{name}")
    if getattr(mamba_simple, "causal_conv1d_fn", None) is None:
        missing.append("mamba_ssm.modules.mamba_simple.causal_conv1d_fn")
    if missing:
        raise SystemExit(
            "official Mamba fast path is unavailable; refusing to run without native scan/conv kernels. "
            f"Missing kernels: {', '.join(missing)}"
        )


def _assert_mamba_fast_path_available() -> None:
    try:
        from transformers.models.mamba import modeling_mamba
    except ImportError as exc:
        raise SystemExit(f"failed to import transformers Mamba implementation: {exc}") from exc
    required = {
        "selective_state_update": getattr(modeling_mamba, "selective_state_update", None),
        "selective_scan_fn": getattr(modeling_mamba, "selective_scan_fn", None),
        "causal_conv1d_fn": getattr(modeling_mamba, "causal_conv1d_fn", None),
        "causal_conv1d_update": getattr(modeling_mamba, "causal_conv1d_update", None),
        "mamba_inner_fn": getattr(modeling_mamba, "mamba_inner_fn", None),
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise SystemExit(
            "HF Mamba fast path is unavailable; refusing to run invalid sequential fallback. "
            f"Missing kernels: {', '.join(missing)}"
        )


def _config_payload(
    *,
    args: argparse.Namespace,
    spec: ExternalModelSpec,
    budget: BenchmarkBudgetSpec,
    runtime: DeviceRuntimeSpec,
    train_steps: int,
    eval_batches: int,
) -> dict[str, Any]:
    return {
        "external_model": spec.key,
        "external_model_id": spec.model_id,
        "backend": runtime.backend,
        "env_kind": runtime.env_kind,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "seq_len": budget.seq_len,
        "window_stride": budget.window_stride,
        "batch_size": budget.batch_size,
        "train_steps": train_steps,
        "eval_batches": eval_batches,
        "learning_rate": budget.learning_rate,
        "optimizer_profile": budget.optimizer_profile,
        "muon_weight_decay": budget.muon_weight_decay,
        "muon_momentum": budget.muon_momentum,
        "muon_ns_steps": budget.muon_ns_steps,
        "muon_adjust_lr_fn": budget.muon_adjust_lr_fn,
        "run_label": args.run_label,
        "dtype": runtime.dtype,
        "warmup_eval_batches": budget.warmup_eval_batches,
        "warmup_train_steps": budget.warmup_train_steps,
        "train_loss_record_interval": budget.train_loss_record_interval,
        "max_position_embeddings": args.max_position_embeddings,
    }


def run_external_lm(args: argparse.Namespace) -> Any:
    spec = EXTERNAL_MODEL_SPECS[args.external_model]
    seed_spec = SeedSpec(model_seed=args.seed, data_seed=args.data_seed)
    runtime = DeviceRuntimeSpec(
        backend=args.backend,
        cuda_device=args.cuda_device,
        dtype=args.dtype,
        env_kind=args.env_kind,
        primitive_runtime_backend=None,
    )
    budget = BenchmarkBudgetSpec(
        seq_len=args.seq_len,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        train_steps=args.steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        warmup_eval_batches=args.warmup_eval_batches,
        warmup_train_steps=args.warmup_train_steps,
        train_loss_record_interval=args.train_loss_record_interval,
        optimizer_profile=args.optimizer_profile,
        muon_weight_decay=args.muon_weight_decay,
        muon_momentum=args.muon_momentum,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
    )
    seed_spec.validate()
    runtime.validate()
    budget.validate()

    configure_reproducibility(seed_spec, runtime)
    device = resolve_torch_device(runtime)
    autocast_dtype = resolve_autocast_dtype(runtime)
    corpus_spec = TokenIdCorpusSpec(manifest_path=args.tokenized_manifest_path)
    vocab_size = _load_manifest_vocab_size(args.tokenized_manifest_path)
    corpus = load_tokenized_corpus(
        corpus_spec,
        seq_len=budget.seq_len,
        window_stride=budget.window_stride,
        batch_size=budget.batch_size,
        data_seed=seed_spec.data_seed,
        shuffle_train=seed_spec.data_seed is not None,
        pin_memory=runtime.backend == "cuda",
    )
    model_impl, diagnostics = _build_external_model(
        spec,
        vocab_size=vocab_size,
        max_position_embeddings=args.max_position_embeddings,
    )
    model = ExternalCausalLmWrapper(
        model_impl,
        model_label=spec.label,
        diagnostics=diagnostics,
        pass_use_cache=spec.key != "official-mamba-130m",
    ).to(device)
    optimizer = build_optimizer(model, budget)
    pad_token = int(corpus.corpus_stats.get("pad_token_id", -100))
    warmup_model(
        model,
        optimizer,
        corpus.train_batches,
        corpus.eval_batches,
        min(budget.warmup_eval_batches, len(corpus.eval_batches)),
        budget.warmup_train_steps,
        autocast_dtype,
        pad_token=pad_token,
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
        train_steps=budget.train_steps,
        eval_batch_count=budget.eval_batches,
        autocast_dtype=autocast_dtype,
        pad_token=pad_token,
        device=device,
        report_model_label=spec.label,
        implementation_kind=spec.implementation_kind,
        note=spec.note,
        config_payload=_config_payload(
            args=args,
            spec=spec,
            budget=budget,
            runtime=runtime,
            train_steps=budget.train_steps,
            eval_batches=budget.eval_batches,
        ),
        corpus_payload=corpus.corpus_stats,
        train_loss_record_interval=budget.train_loss_record_interval,
    )
    diagnostic_payload = model.diagnostic_payload()
    report.diagnostics = diagnostic_payload

    output_dir = args.output_dir / spec.label
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    write_report(report, report_path)
    if args.ledger_path is not None:
        append_ledger_entry(
            args.ledger_path,
            {
                "run_label": args.run_label,
                "variant": spec.key,
                "model_label": report.model_label,
                "report_path": repo_relative(report_path),
                "corpus_name": corpus.corpus_stats["corpus_name"],
                "seed": seed_spec.model_seed,
                "data_seed": seed_spec.data_seed,
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


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        report = run_external_lm(args)
    except ValidationError as exc:
        parser.error(str(exc))
    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_path1_table(report, EXTERNAL_MODEL_SPECS[args.external_model].label))
        print(f"report_path={report.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
