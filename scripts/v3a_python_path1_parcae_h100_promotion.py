#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = "fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst"


def _cache_dir_name_from_artifact(artifact: str) -> str:
    if artifact.endswith(".tar.zst"):
        return artifact[: -len(".tar.zst")]
    if artifact.endswith(".tar"):
        return artifact[: -len(".tar")]
    return Path(artifact).stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hydrate the private Stage 0 token cache and run the Parcae H100 Path 1 promotion comparison."
    )
    parser.add_argument("--backend", default="cuda", choices=["cuda"], help="Accepted for runpod wrapper compatibility.")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--train-loss-record-interval", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=0)
    parser.add_argument("--warmup-eval-batches", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--optimizer-profile", choices=["adam", "adam-fused", "adam-triton-2d", "muon-reference"], default="adam")
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"])
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--head-count", type=int)
    parser.add_argument("--total-layers", type=int)
    parser.add_argument("--local-window", type=int)
    parser.add_argument("--attention-kernel", choices=["sdpa", "flex-local", "flash-local"], default="sdpa")
    parser.add_argument("--ffn-multiplier", type=int)
    parser.add_argument("--parcae-loop-count", type=int, default=2)
    parser.add_argument("--parcae-hourglass-pass-count", type=int, default=1)
    parser.add_argument("--parcae-hourglass-band-schedule")
    parser.add_argument("--parcae-backward-steps", type=int)
    parser.add_argument("--parcae-prelude-norm-kind", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--parcae-discretization", choices=["stable-exp", "zoh"], default="stable-exp")
    parser.add_argument("--parcae-dt-raw-init", type=float, default=0.54132485)
    parser.add_argument("--parcae-loop-d-model", type=int)
    parser.add_argument("--parcae-loop-head-count", type=int)
    parser.add_argument("--parcae-loop-ffn-multiplier", type=int)
    parser.add_argument("--parcae-loop-layer-count", type=int)
    parser.add_argument("--parcae-control-position-kind", choices=["none", "learned"], default="none")
    parser.add_argument("--parcae-control-position-scale-init", type=float, default=0.01)
    parser.add_argument("--parcae-control-stride", type=int, default=1)
    parser.add_argument(
        "--parcae-control-state-transform",
        choices=["trainable", "trainable-block-diagonal-8", "frozen-identity"],
        default="trainable",
    )
    parser.add_argument(
        "--parcae-recurrent-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--parcae-loop-update-backend",
        choices=[
            "eager",
            "lean-eager",
            "compiled",
            "manual-autograd",
            "triton-glue",
            "triton-loop-forward",
        ],
        default="eager",
    )
    parser.add_argument("--parcae-scaffold-backend", choices=["standard", "compiled"], default="standard")
    parser.add_argument("--parcae-band-block-contract", choices=["generic", "compiled-direct"], default="generic")
    parser.add_argument("--parcae-band-prepare-backend", choices=["standard", "compiled"], default="standard")
    parser.add_argument("--parcae-output-mix-backend", choices=["standard", "triton"], default="standard")
    parser.add_argument(
        "--parcae-fuse-first-state-mix",
        action="store_true",
        help="Skip the first Parcae band state-mix op when the recurrent state is known-zero.",
    )
    parser.add_argument("--position-encoding-kind", choices=["none", "learned"], default="none")
    parser.add_argument("--attention-position-contract", choices=["shared-input", "attention-only"], default="shared-input")
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    parser.add_argument("--final-norm-kind", choices=["identity", "layernorm", "rmsnorm"], default="identity")
    parser.add_argument("--token-cache-repo-id", default="joebud/fractal-fineweb-openllama-tokens")
    parser.add_argument("--token-cache-artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--token-cache-dir",
        help="Extracted token-cache directory name under --data-root. Defaults to the artifact basename.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ledger-path", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--lanes",
        default="attention-only,parcae-looped-attention,parcae-bx-looped-attention,parcae-p20-control-looped-attention",
        help="Comma-separated lane list.",
    )
    parser.add_argument(
        "--primitive-runtime-backend",
        default="triton",
        choices=["torch", "triton"],
        help="Runtime backend forwarded to all lanes; relevant for the P20-control lane.",
    )
    parser.add_argument(
        "--head-loss-backend",
        choices=["dense", "compiled", "streaming-kernel"],
        default="dense",
        help="LM-head/loss runtime backend forwarded to the Path 1 runner.",
    )
    parser.add_argument(
        "--ffn-backend",
        choices=["dense", "compiled", "manual-autograd", "triton-gelu", "recompute"],
        default="dense",
        help="Transformer FFN runtime backend forwarded to the Path 1 runner.",
    )
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--force-download", action="store_true")
    return parser


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _ensure_zstd() -> None:
    if shutil.which("zstd") is not None:
        return
    if shutil.which("apt-get") is None:
        raise SystemExit("zstd is required to extract the token cache, and apt-get is not available")
    _run(["apt-get", "update"])
    _run(["apt-get", "install", "-y", "zstd"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_sha(path: Path) -> str:
    first = path.read_text(encoding="utf-8").strip().split()[0]
    if len(first) != 64:
        raise SystemExit(f"invalid sha256 sidecar: {path}")
    return first


def hydrate_token_cache(args: argparse.Namespace) -> Path:
    cache_dir = args.data_root / (args.token_cache_dir or _cache_dir_name_from_artifact(args.token_cache_artifact))
    manifest = cache_dir / "manifest.json"
    if manifest.exists() and not args.force_download:
        print(f"using existing token cache: {manifest}", flush=True)
        return manifest

    if "HF_TOKEN" not in os.environ:
        raise SystemExit("HF_TOKEN is required to download the private token cache")
    _ensure_zstd()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; install scripts/requirements-v3a-python-tokenized-corpus.txt") from exc

    work_dir = args.data_root / ".hf-token-cache-download"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    artifact_path = Path(
        hf_hub_download(
            repo_id=args.token_cache_repo_id,
            repo_type="dataset",
            filename=args.token_cache_artifact,
            local_dir=work_dir,
        )
    )
    checksum_path = Path(
        hf_hub_download(
            repo_id=args.token_cache_repo_id,
            repo_type="dataset",
            filename=f"{args.token_cache_artifact}.sha256",
            local_dir=work_dir,
        )
    )

    expected = _expected_sha(checksum_path)
    actual = _sha256(artifact_path)
    if actual != expected:
        raise SystemExit(f"token-cache checksum mismatch: expected {expected}, got {actual}")

    shutil.rmtree(cache_dir, ignore_errors=True)
    _run(["tar", "-C", str(args.data_root), "-I", "zstd", "-xf", str(artifact_path)])
    shutil.rmtree(work_dir, ignore_errors=True)

    if not manifest.exists():
        raise SystemExit(f"token-cache extraction did not create manifest: {manifest}")
    return manifest


def _load_manifest_stats(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    train = payload["splits"]["train"]
    eval_split = payload["splits"]["eval"]
    return {
        "train_tokens": int(train["token_count"]),
        "eval_tokens": int(eval_split["token_count"]),
        "vocab_size": int(payload["tokenizer"]["vocab_size"]),
    }


def _nonoverlap_step_cap(train_tokens: int, *, seq_len: int, batch_size: int) -> int:
    stride = seq_len + 1
    required_len = seq_len + 1
    sequences = ((train_tokens - required_len) // stride) + 1
    return max(0, sequences // batch_size)


def _lane_args(
    lane: str,
    *,
    loop_count: int,
    hourglass_pass_count: int,
    hourglass_band_schedule: str | None,
    backward_steps: int | None,
    prelude_norm_kind: str,
    discretization: str,
    dt_raw_init: float,
    scaffold_backend: str,
    band_block_contract: str,
    band_prepare_backend: str,
    output_mix_backend: str,
    fuse_first_state_mix: bool,
) -> list[str]:
    if lane == "attention-only":
        return ["--variant", "attention-only"]
    if lane in {
        "parcae-looped-attention",
        "parcae-bx-looped-attention",
        "parcae-p20-control-looped-attention",
        "parcae-hourglass-looped-attention",
        "parcae-hourglass-bx-looped-attention",
        "parcae-hourglass-p20-control-looped-attention",
    }:
        args = [
            "--variant",
            "attention-only",
            "--scaffold-profile",
            lane,
            "--parcae-loop-count",
            str(loop_count),
            "--parcae-hourglass-pass-count",
            str(hourglass_pass_count),
            "--parcae-prelude-norm-kind",
            prelude_norm_kind,
            "--parcae-discretization",
            discretization,
            "--parcae-dt-raw-init",
            str(dt_raw_init),
            "--parcae-scaffold-backend",
            scaffold_backend,
            "--parcae-band-block-contract",
            band_block_contract,
            "--parcae-band-prepare-backend",
            band_prepare_backend,
            "--parcae-output-mix-backend",
            output_mix_backend,
        ]
        if fuse_first_state_mix:
            args.append("--parcae-fuse-first-state-mix")
        if hourglass_band_schedule:
            args.extend(["--parcae-hourglass-band-schedule", hourglass_band_schedule])
        if backward_steps is not None:
            args.extend(["--parcae-backward-steps", str(backward_steps)])
        return args
    raise SystemExit(f"unsupported lane: {lane}")


def _primitive_runtime_backend_for_lane(args: argparse.Namespace, lane: str) -> str:
    if lane == "attention-only":
        return "torch"
    return args.primitive_runtime_backend


def _env_kind_for_primitive_backend(primitive_runtime_backend: str) -> str:
    return "primitive-triton" if primitive_runtime_backend == "triton" else "requirements-only"


def _lane_uses_parcae_scaffold(lane: str) -> bool:
    return lane.startswith("parcae-")


def run_lane(args: argparse.Namespace, *, lane: str, manifest_path: Path, output_dir: Path, ledger_path: Path) -> dict[str, Any]:
    lane_output_dir = output_dir / lane
    lane_primitive_runtime_backend = _primitive_runtime_backend_for_lane(args, lane)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "v3a_python_path1.py"),
        "--backend",
        "cuda",
        "--cuda-device",
        str(args.cuda_device),
        "--dtype",
        args.dtype,
        "--env-kind",
        _env_kind_for_primitive_backend(lane_primitive_runtime_backend),
        "--primitive-runtime-backend",
        lane_primitive_runtime_backend,
        "--head-loss-backend",
        args.head_loss_backend,
        "--ffn-backend",
        args.ffn_backend,
        "--corpus-format",
        "token-ids",
        "--tokenized-manifest-path",
        str(manifest_path),
        "--seq-len",
        str(args.seq_len),
        "--window-stride",
        str(args.seq_len + 1),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--eval-batches",
        str(args.eval_batches),
        "--train-loss-record-interval",
        str(args.train_loss_record_interval),
        "--warmup-train-steps",
        str(args.warmup_train_steps),
        "--warmup-eval-batches",
        str(args.warmup_eval_batches),
        "--learning-rate",
        str(args.learning_rate),
        "--optimizer-profile",
        args.optimizer_profile,
        "--muon-weight-decay",
        str(args.muon_weight_decay),
        "--muon-momentum",
        str(args.muon_momentum),
        "--muon-ns-steps",
        str(args.muon_ns_steps),
        "--seed",
        str(args.seed),
        "--data-seed",
        str(args.data_seed),
        "--output-dir",
        str(lane_output_dir),
        "--ledger-path",
        str(ledger_path),
        "--run-label",
        f"{args.run_label}-{lane}",
        "--output",
        "table",
        "--position-encoding-kind",
        args.position_encoding_kind,
        "--attention-position-contract",
        args.attention_position_contract,
        "--max-position-embeddings",
        str(args.max_position_embeddings),
        "--final-norm-kind",
        args.final_norm_kind,
    ]
    if _lane_uses_parcae_scaffold(lane):
        command.extend(
            [
                "--parcae-recurrent-compile-mode",
                args.parcae_recurrent_compile_mode,
                "--parcae-loop-update-backend",
                args.parcae_loop_update_backend,
            ]
        )
    if args.compile_mode is not None and lane_primitive_runtime_backend != "triton":
        command.extend(["--compile-mode", args.compile_mode])
    elif args.compile_mode is not None:
        print(
            "skipping global compile mode for primitive-triton env; "
            "use --parcae-recurrent-compile-mode for recurrent compilation",
            flush=True,
        )
    if args.muon_adjust_lr_fn is not None:
        command.extend(["--muon-adjust-lr-fn", args.muon_adjust_lr_fn])
    for cli_name, value in (
        ("--d-model", args.d_model),
        ("--head-count", args.head_count),
        ("--total-layers", args.total_layers),
        ("--local-window", args.local_window),
        ("--attention-kernel", args.attention_kernel),
        ("--ffn-multiplier", args.ffn_multiplier),
    ):
        if value is not None:
            command.extend([cli_name, str(value)])
    if "hourglass" in lane:
        for cli_name, value in (
            ("--parcae-loop-d-model", args.parcae_loop_d_model),
            ("--parcae-loop-head-count", args.parcae_loop_head_count),
            ("--parcae-loop-ffn-multiplier", args.parcae_loop_ffn_multiplier),
            ("--parcae-loop-layer-count", args.parcae_loop_layer_count),
        ):
            if value is not None:
                command.extend([cli_name, str(value)])
    elif any(
        value is not None
        for value in (
            args.parcae_loop_d_model,
            args.parcae_loop_head_count,
            args.parcae_loop_ffn_multiplier,
            args.parcae_loop_layer_count,
        )
    ):
        print(f"skipping hourglass loop-width overrides for non-hourglass lane: {lane}", flush=True)
    if "p20-control" in lane:
        command.extend(
            [
                "--parcae-control-position-kind",
                args.parcae_control_position_kind,
                "--parcae-control-position-scale-init",
                str(args.parcae_control_position_scale_init),
                "--parcae-control-stride",
                str(args.parcae_control_stride),
                "--parcae-control-state-transform",
                args.parcae_control_state_transform,
            ]
        )
    elif args.parcae_control_position_kind != "none":
        print(f"skipping P20/RGRP control position features for non-P20 lane: {lane}", flush=True)
    command.extend(
        _lane_args(
            lane,
            loop_count=args.parcae_loop_count,
            hourglass_pass_count=args.parcae_hourglass_pass_count,
            hourglass_band_schedule=args.parcae_hourglass_band_schedule,
            backward_steps=args.parcae_backward_steps,
            prelude_norm_kind=args.parcae_prelude_norm_kind,
            discretization=args.parcae_discretization,
            dt_raw_init=args.parcae_dt_raw_init,
            scaffold_backend=args.parcae_scaffold_backend,
            band_block_contract=args.parcae_band_block_contract,
            band_prepare_backend=args.parcae_band_prepare_backend,
            output_mix_backend=args.parcae_output_mix_backend,
            fuse_first_state_mix=args.parcae_fuse_first_state_mix,
        )
    )
    _run(command, cwd=REPO_ROOT)

    reports = list(lane_output_dir.glob("*/report.json"))
    if len(reports) != 1:
        raise SystemExit(f"expected one report for {lane}, found {len(reports)} under {lane_output_dir}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    runtime = report["runtime"]
    diagnostics = report.get("diagnostics") or {}
    cuda_memory = runtime.get("cuda_device_memory") or {}
    return {
        "lane": lane,
        "report_path": str(reports[0]),
        "parameters": diagnostics.get("parameter_count"),
        "initial_loss": report["initial_eval"]["mean_loss"],
        "final_loss": report["final_eval"]["mean_loss"],
        "train_tokens_per_second": runtime["train_tokens_per_second"],
        "peak_cuda_memory_mb": (cuda_memory.get("peak_used_bytes") or 0) / (1024 * 1024),
        "cuda_device": cuda_memory.get("device_name"),
        "cuda_compute_capability": cuda_memory.get("compute_capability"),
        "primitive_runtime_backend": lane_primitive_runtime_backend,
    }


def write_summary(args: argparse.Namespace, *, output_dir: Path, manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    summary = {
        "run_label": args.run_label,
        "manifest_path": str(manifest_path),
        "seq_len": args.seq_len,
        "window_stride": args.seq_len + 1,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "eval_batches": args.eval_batches,
        "train_loss_record_interval": args.train_loss_record_interval,
        "warmup_train_steps": args.warmup_train_steps,
        "warmup_eval_batches": args.warmup_eval_batches,
        "optimizer_profile": args.optimizer_profile,
        "primitive_runtime_backend": args.primitive_runtime_backend,
        "head_loss_backend": args.head_loss_backend,
        "ffn_backend": args.ffn_backend,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "d_model": args.d_model,
        "head_count": args.head_count,
        "total_layers": args.total_layers,
        "local_window": args.local_window,
        "attention_kernel": args.attention_kernel,
        "ffn_multiplier": args.ffn_multiplier,
        "dtype": args.dtype,
        "parcae_loop_count": args.parcae_loop_count,
        "parcae_backward_steps": args.parcae_backward_steps,
        "parcae_prelude_norm_kind": args.parcae_prelude_norm_kind,
        "parcae_discretization": args.parcae_discretization,
        "parcae_dt_raw_init": args.parcae_dt_raw_init,
        "parcae_loop_d_model": args.parcae_loop_d_model,
        "parcae_loop_head_count": args.parcae_loop_head_count,
        "parcae_loop_ffn_multiplier": args.parcae_loop_ffn_multiplier,
        "parcae_loop_layer_count": args.parcae_loop_layer_count,
        "parcae_control_position_kind": args.parcae_control_position_kind,
        "parcae_control_position_scale_init": args.parcae_control_position_scale_init,
        "parcae_control_stride": args.parcae_control_stride,
        "parcae_control_state_transform": args.parcae_control_state_transform,
        "parcae_recurrent_compile_mode": args.parcae_recurrent_compile_mode,
        "parcae_loop_update_backend": args.parcae_loop_update_backend,
        "parcae_scaffold_backend": args.parcae_scaffold_backend,
        "parcae_band_block_contract": args.parcae_band_block_contract,
        "parcae_band_prepare_backend": args.parcae_band_prepare_backend,
        "parcae_output_mix_backend": args.parcae_output_mix_backend,
        "parcae_fuse_first_state_mix": args.parcae_fuse_first_state_mix,
        "position_encoding_kind": args.position_encoding_kind,
        "attention_position_contract": args.attention_position_contract,
        "max_position_embeddings": args.max_position_embeddings,
        "final_norm_kind": args.final_norm_kind,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Path 1 Parcae H100 Promotion",
        "",
        f"- run_label: `{args.run_label}`",
        f"- manifest: `{manifest_path}`",
        f"- seq_len/window_stride: `{args.seq_len}/{args.seq_len + 1}`",
        f"- batch_size: `{args.batch_size}`",
        f"- steps: `{args.steps}`",
        f"- eval_batches: `{args.eval_batches}`",
        f"- train_loss_record_interval: `{args.train_loss_record_interval}`",
        f"- warmup_train_steps: `{args.warmup_train_steps}`",
        f"- warmup_eval_batches: `{args.warmup_eval_batches}`",
        f"- optimizer_profile: `{args.optimizer_profile}`",
        f"- muon_weight_decay: `{args.muon_weight_decay}`",
        f"- muon_momentum: `{args.muon_momentum}`",
        f"- muon_ns_steps: `{args.muon_ns_steps}`",
        f"- muon_adjust_lr_fn: `{args.muon_adjust_lr_fn}`",
        f"- dtype: `{args.dtype}`",
        f"- primitive_runtime_backend: `{args.primitive_runtime_backend}`",
        f"- head_loss_backend: `{args.head_loss_backend}`",
        f"- ffn_backend: `{args.ffn_backend}`",
        f"- parcae_loop_count: `{args.parcae_loop_count}`",
        f"- parcae_hourglass_pass_count: `{args.parcae_hourglass_pass_count}`",
        f"- parcae_hourglass_band_schedule: `{args.parcae_hourglass_band_schedule}`",
        f"- parcae_backward_steps: `{args.parcae_backward_steps}`",
        f"- parcae_prelude_norm_kind: `{args.parcae_prelude_norm_kind}`",
        f"- parcae_discretization: `{args.parcae_discretization}`",
        f"- parcae_loop_d_model: `{args.parcae_loop_d_model}`",
        f"- parcae_loop_head_count: `{args.parcae_loop_head_count}`",
        f"- parcae_loop_ffn_multiplier: `{args.parcae_loop_ffn_multiplier}`",
        f"- parcae_loop_layer_count: `{args.parcae_loop_layer_count}`",
        f"- parcae_control_position_kind: `{args.parcae_control_position_kind}`",
        f"- parcae_control_position_scale_init: `{args.parcae_control_position_scale_init}`",
        f"- parcae_control_stride: `{args.parcae_control_stride}`",
        f"- parcae_control_state_transform: `{args.parcae_control_state_transform}`",
        f"- parcae_recurrent_compile_mode: `{args.parcae_recurrent_compile_mode}`",
        f"- parcae_loop_update_backend: `{args.parcae_loop_update_backend}`",
        f"- parcae_scaffold_backend: `{args.parcae_scaffold_backend}`",
        f"- parcae_band_block_contract: `{args.parcae_band_block_contract}`",
        f"- parcae_band_prepare_backend: `{args.parcae_band_prepare_backend}`",
        f"- parcae_output_mix_backend: `{args.parcae_output_mix_backend}`",
        f"- parcae_fuse_first_state_mix: `{args.parcae_fuse_first_state_mix}`",
        f"- position_encoding_kind: `{args.position_encoding_kind}`",
        f"- attention_position_contract: `{args.attention_position_contract}`",
        f"- max_position_embeddings: `{args.max_position_embeddings}`",
        f"- final_norm_kind: `{args.final_norm_kind}`",
        "",
        "| Lane | Primitive Backend | Params | Initial Loss | Final Loss | tok/s | Peak CUDA MB | CUDA Device |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        params = row["parameters"]
        cuda_device = row.get("cuda_device") or ""
        cuda_capability = row.get("cuda_compute_capability")
        if cuda_capability:
            cuda_device = f"{cuda_device} (cc {cuda_capability})" if cuda_device else f"cc {cuda_capability}"
        lines.append(
            f"| {row['lane']} | {row.get('primitive_runtime_backend') or ''} | "
            f"{params if params is not None else ''} | "
            f"{row['initial_loss']:.4f} | {row['final_loss']:.4f} | "
            f"{row['train_tokens_per_second']:.2f} | {row['peak_cuda_memory_mb']:.2f} | {cuda_device} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        artifact_root = Path(os.environ.get("FRACTAL_RUN_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
        run_id = os.environ.get("FRACTAL_RUN_ID", args.run_label)
        output_dir = artifact_root / "v3a-python-path1-parcae-h100" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.ledger_path or (output_dir / "ledger.jsonl")

    manifest_path = hydrate_token_cache(args)
    stats = _load_manifest_stats(manifest_path)
    step_cap = _nonoverlap_step_cap(
        stats["train_tokens"],
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    if args.steps > step_cap:
        raise SystemExit(
            f"requested steps={args.steps} would wrap the token cache; no-repeat cap is {step_cap} "
            f"for seq_len={args.seq_len}, batch_size={args.batch_size}"
        )
    print(
        "token cache ready: "
        f"train_tokens={stats['train_tokens']} eval_tokens={stats['eval_tokens']} "
        f"max_no_repeat_steps={step_cap}",
        flush=True,
    )

    lanes = [lane.strip() for lane in args.lanes.split(",") if lane.strip()]
    rows = [
        run_lane(args, lane=lane, manifest_path=manifest_path, output_dir=output_dir, ledger_path=ledger_path)
        for lane in lanes
    ]
    write_summary(args, output_dir=output_dir, manifest_path=manifest_path, rows=rows)
    print(output_dir / "summary.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
