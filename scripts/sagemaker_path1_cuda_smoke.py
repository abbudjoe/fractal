#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE_TYPE = "ml.g6.2xlarge"
DEFAULT_REGION = "us-east-1"
DEFAULT_PROFILE = "codex-eml"
DEFAULT_DLC_ACCOUNT = "763104351884"
DEFAULT_DLC_TAG = "2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
SMOKE_CORPUS_REL = Path("experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1")
LANES_WITH_PARCAE_SCAFFOLD = {
    "parcae-looped-attention",
    "parcae-bx-looped-attention",
    "parcae-p20-control-looped-attention",
    "parcae-hourglass-looped-attention",
    "parcae-hourglass-bx-looped-attention",
    "parcae-hourglass-p20-control-looped-attention",
}
EXTERNAL_LM_LANES = {
    "hf-gpt2-small": "gpt2-small",
    "hf-mamba-130m": "mamba-130m",
    "official-mamba-130m": "official-mamba-130m",
}
LANE_ALIASES = {
    "parcae-rgrp-control-looped-attention": "parcae-p20-control-looped-attention",
    "parcae-rgrp-control": "parcae-p20-control-looped-attention",
    "parcae-hourglass-rgrp-control-looped-attention": "parcae-hourglass-p20-control-looped-attention",
    "parcae-hourglass-rgrp-control": "parcae-hourglass-p20-control-looped-attention",
    "gpt2-small": "hf-gpt2-small",
    "hf-gpt2": "hf-gpt2-small",
    "mamba-130m": "hf-mamba-130m",
    "mamba130m": "hf-mamba-130m",
    "official-mamba": "official-mamba-130m",
    "mamba-130m-official": "official-mamba-130m",
    "mamba-official": "official-mamba-130m",
}
DEFAULT_TOKEN_CACHE_ARTIFACT = "fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst"
HF_ENV_CHANNEL_NAME = "hf_env"
HF_ENV_FILENAME = "hf.env"
SENSITIVE_ENV_KEYS = {"HF_TOKEN"}


ENTRYPOINT = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/opt/ml/model/path1-cuda-smoke")
CORPUS_DIR = ROOT / "experiments" / "stage0" / "assets" / "fineweb" / "stage0-local-bench-9row-v1"
PARCAE_LANES = {
    "parcae-looped-attention",
    "parcae-bx-looped-attention",
    "parcae-p20-control-looped-attention",
    "parcae-hourglass-looped-attention",
    "parcae-hourglass-bx-looped-attention",
    "parcae-hourglass-p20-control-looped-attention",
}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _primitive_backend_for_lane(lane: str) -> str:
    if lane == "attention-only":
        return "torch"
    return _env("FRACTAL_SMOKE_PRIMITIVE_RUNTIME_BACKEND", "torch")


def _load_hf_env_channel() -> None:
    if os.environ.get("HF_TOKEN", "").strip():
        return
    env_path = Path("/opt/ml/input/data") / "hf_env" / "hf.env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key == "HF_TOKEN" and value.strip():
            os.environ["HF_TOKEN"] = value.strip().strip("'\"")
            return


def _probe_cuda() -> None:
    import torch

    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    print("cuda_probe=" + json.dumps(payload, sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside the SageMaker training container")


def _lane_command(lane: str, *, output_dir: Path, ledger_path: Path) -> list[str]:
    seq_len = _env("FRACTAL_SMOKE_SEQ_LEN", "64")
    primitive_backend = _primitive_backend_for_lane(lane)
    base = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_python_path1.py"),
        "--backend",
        "cuda",
        "--cuda-device",
        "0",
        "--dtype",
        _env("FRACTAL_SMOKE_DTYPE", "fp32"),
        "--env-kind",
        "requirements-only",
        "--primitive-runtime-backend",
        primitive_backend,
        "--head-loss-backend",
        _env("FRACTAL_SMOKE_HEAD_LOSS_BACKEND", "dense"),
        "--ffn-backend",
        _env("FRACTAL_SMOKE_FFN_BACKEND", "dense"),
        "--jsonl-train-path",
        str(CORPUS_DIR / "train.jsonl"),
        "--jsonl-eval-path",
        str(CORPUS_DIR / "eval.jsonl"),
        "--seq-len",
        seq_len,
        "--window-stride",
        seq_len,
        "--batch-size",
        _env("FRACTAL_SMOKE_BATCH_SIZE", "4"),
        "--steps",
        _env("FRACTAL_SMOKE_STEPS", "5"),
        "--eval-batches",
        _env("FRACTAL_SMOKE_EVAL_BATCHES", "2"),
        "--train-loss-record-interval",
        _env("FRACTAL_SMOKE_TRAIN_LOSS_RECORD_INTERVAL", "1"),
        "--warmup-train-steps",
        "0",
        "--warmup-eval-batches",
        "0",
        "--learning-rate",
        _env("FRACTAL_SMOKE_LEARNING_RATE", "0.001"),
        "--optimizer-profile",
        _env("FRACTAL_SMOKE_OPTIMIZER_PROFILE", "adam"),
        "--muon-weight-decay",
        _env("FRACTAL_SMOKE_MUON_WEIGHT_DECAY", "0.0"),
        "--muon-momentum",
        _env("FRACTAL_SMOKE_MUON_MOMENTUM", "0.95"),
        "--muon-ns-steps",
        _env("FRACTAL_SMOKE_MUON_NS_STEPS", "5"),
        "--seed",
        _env("FRACTAL_SMOKE_SEED", "42"),
        "--data-seed",
        _env("FRACTAL_SMOKE_DATA_SEED", "42"),
        "--d-model",
        _env("FRACTAL_SMOKE_D_MODEL", "128"),
        "--head-count",
        _env("FRACTAL_SMOKE_HEAD_COUNT", "4"),
        "--total-layers",
        _env("FRACTAL_SMOKE_TOTAL_LAYERS", "4"),
        "--local-window",
        _env("FRACTAL_SMOKE_LOCAL_WINDOW", "64"),
        "--attention-kernel",
        _env("FRACTAL_SMOKE_ATTENTION_KERNEL", "sdpa"),
        "--ffn-multiplier",
        _env("FRACTAL_SMOKE_FFN_MULTIPLIER", "2"),
        "--parcae-loop-count",
        _env("FRACTAL_SMOKE_PARCAE_LOOP_COUNT", "1"),
        "--parcae-hourglass-pass-count",
        _env("FRACTAL_SMOKE_PARCAE_HOURGLASS_PASS_COUNT", "1"),
        "--parcae-hourglass-band-schedule",
        _env("FRACTAL_SMOKE_PARCAE_HOURGLASS_BAND_SCHEDULE", ""),
        "--parcae-prelude-norm-kind",
        _env("FRACTAL_SMOKE_PARCAE_PRELUDE_NORM_KIND", "layernorm"),
        "--parcae-discretization",
        _env("FRACTAL_SMOKE_PARCAE_DISCRETIZATION", "stable-exp"),
        "--parcae-dt-raw-init",
        _env("FRACTAL_SMOKE_PARCAE_DT_RAW_INIT", "0.54132485"),
        "--position-encoding-kind",
        _env("FRACTAL_SMOKE_POSITION_ENCODING_KIND", "none"),
        "--attention-position-contract",
        _env("FRACTAL_SMOKE_ATTENTION_POSITION_CONTRACT", "shared-input"),
        "--max-position-embeddings",
        _env("FRACTAL_SMOKE_MAX_POSITION_EMBEDDINGS", "1024"),
        "--final-norm-kind",
        _env("FRACTAL_SMOKE_FINAL_NORM_KIND", "identity"),
        "--output-dir",
        str(output_dir),
        "--ledger-path",
        str(ledger_path),
        "--run-label",
        f"{_env('FRACTAL_SMOKE_RUN_LABEL', 'sagemaker-path1-cuda-smoke')}-{lane}",
        "--output",
        "table",
    ]
    parcae_backward_steps = _env("FRACTAL_SMOKE_PARCAE_BACKWARD_STEPS", "")
    if parcae_backward_steps:
        base.extend(["--parcae-backward-steps", parcae_backward_steps])
    muon_adjust_lr_fn = _env("FRACTAL_SMOKE_MUON_ADJUST_LR_FN", "")
    if muon_adjust_lr_fn:
        base.extend(["--muon-adjust-lr-fn", muon_adjust_lr_fn])
    for env_name, cli_name in (
        ("FRACTAL_SMOKE_PARCAE_LOOP_D_MODEL", "--parcae-loop-d-model"),
        ("FRACTAL_SMOKE_PARCAE_LOOP_HEAD_COUNT", "--parcae-loop-head-count"),
        ("FRACTAL_SMOKE_PARCAE_LOOP_FFN_MULTIPLIER", "--parcae-loop-ffn-multiplier"),
        ("FRACTAL_SMOKE_PARCAE_LOOP_LAYER_COUNT", "--parcae-loop-layer-count"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value and "hourglass" in lane:
            base.extend([cli_name, value])
    if "p20-control" in lane:
        base.extend(
            [
                "--parcae-control-position-kind",
                _env("FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_KIND", "none"),
                "--parcae-control-position-scale-init",
                _env("FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_SCALE_INIT", "0.01"),
                "--parcae-control-stride",
                _env("FRACTAL_SMOKE_PARCAE_CONTROL_STRIDE", "1"),
                "--parcae-control-state-transform",
                _env("FRACTAL_SMOKE_PARCAE_CONTROL_STATE_TRANSFORM", "trainable"),
                "--parcae-recurrent-compile-mode",
                _env("FRACTAL_SMOKE_PARCAE_RECURRENT_COMPILE_MODE", "reduce-overhead"),
                "--parcae-loop-update-backend",
                _env("FRACTAL_SMOKE_PARCAE_LOOP_UPDATE_BACKEND", "eager"),
                "--parcae-scaffold-backend",
                _env("FRACTAL_SMOKE_PARCAE_SCAFFOLD_BACKEND", "standard"),
                "--parcae-band-block-contract",
                _env("FRACTAL_SMOKE_PARCAE_BAND_BLOCK_CONTRACT", "generic"),
                "--parcae-band-prepare-backend",
                _env("FRACTAL_SMOKE_PARCAE_BAND_PREPARE_BACKEND", "standard"),
                "--parcae-output-mix-backend",
                _env("FRACTAL_SMOKE_PARCAE_OUTPUT_MIX_BACKEND", "standard"),
            ]
        )
        if _env("FRACTAL_SMOKE_PARCAE_FUSE_FIRST_STATE_MIX", "false").lower() in {"1", "true", "yes"}:
            base.append("--parcae-fuse-first-state-mix")
    if lane == "attention-only":
        return base + ["--variant", "attention-only"]
    if lane in PARCAE_LANES:
        return base + [
            "--variant",
            "attention-only",
            "--scaffold-profile",
            lane,
        ]
    raise SystemExit(f"unsupported lane: {lane}")


def _load_lane_report(lane_output_dir: Path) -> dict:
    reports = sorted(lane_output_dir.glob("*/report.json"))
    if len(reports) != 1:
        raise SystemExit(f"expected one report under {lane_output_dir}, found {len(reports)}")
    return json.loads(reports[0].read_text(encoding="utf-8"))


def main() -> int:
    _probe_cuda()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lanes = [lane.strip() for lane in _env("FRACTAL_SMOKE_LANES", "attention-only,parcae-p20-control-looped-attention").split(",") if lane.strip()]
    rows = []
    ledger_path = OUT_ROOT / "ledger.jsonl"
    for lane in lanes:
        lane_output_dir = OUT_ROOT / lane
        command = _lane_command(lane, output_dir=lane_output_dir, ledger_path=ledger_path)
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        report = _load_lane_report(lane_output_dir)
        runtime = report["runtime"]
        diagnostics = report.get("diagnostics") or {}
        cuda_memory = runtime.get("cuda_device_memory") or {}
        rows.append(
            {
                "lane": lane,
                "parameters": diagnostics.get("parameter_count"),
                "initial_loss": report["initial_eval"]["mean_loss"],
                "final_loss": report["final_eval"]["mean_loss"],
                "train_tokens_per_second": runtime["train_tokens_per_second"],
                "overall_tokens_per_second": runtime["overall_tokens_per_second"],
                "peak_cuda_memory_mb": (cuda_memory.get("peak_used_bytes") or 0) / (1024 * 1024),
                "cuda_device": cuda_memory.get("device_name"),
                "cuda_compute_capability": cuda_memory.get("compute_capability"),
                "report_path": report["report_path"],
            }
        )

    summary = {
        "run_label": _env("FRACTAL_SMOKE_RUN_LABEL", "sagemaker-path1-cuda-smoke"),
        "lanes": lanes,
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SageMaker Path 1 CUDA Smoke",
        "",
        "| Lane | Params | Initial Loss | Final Loss | train tok/s | overall tok/s | Peak CUDA MB | CUDA Device |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        device = row.get("cuda_device") or ""
        capability = row.get("cuda_compute_capability")
        if capability:
            device = f"{device} (cc {capability})" if device else f"cc {capability}"
        lines.append(
            f"| {row['lane']} | {row.get('parameters') or ''} | "
            f"{row['initial_loss']:.4f} | {row['final_loss']:.4f} | "
            f"{row['train_tokens_per_second']:.2f} | {row['overall_tokens_per_second']:.2f} | "
            f"{row['peak_cuda_memory_mb']:.2f} | {device} |"
        )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((OUT_ROOT / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


TOKEN_CACHE_ENTRYPOINT = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/opt/ml/model/path1-cuda-scout")
DATA_ROOT = Path("/opt/ml/input/data/fractal-token-cache")
DEFAULT_TOKEN_CACHE_ARTIFACT = "fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst"
EXTERNAL_LM_LANES = {
    "hf-gpt2-small": "gpt2-small",
    "hf-mamba-130m": "mamba-130m",
    "official-mamba-130m": "official-mamba-130m",
}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _primitive_backend_for_lane(lane: str) -> str:
    if lane == "attention-only":
        return "torch"
    return _env("FRACTAL_SCOUT_PRIMITIVE_RUNTIME_BACKEND", "torch")


def _env_kind_for_primitive_backend(primitive_backend: str) -> str:
    return "primitive-triton" if primitive_backend == "triton" else "requirements-only"


def _load_hf_env_channel() -> None:
    if os.environ.get("HF_TOKEN", "").strip():
        return
    env_path = Path("/opt/ml/input/data") / "hf_env" / "hf.env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "HF_TOKEN" and value.strip():
            os.environ["HF_TOKEN"] = value.strip().strip("'\"")
            return


def _probe_cuda() -> None:
    import torch

    payload = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    print("cuda_probe=" + json.dumps(payload, sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside the SageMaker training container")


def _cache_dir_name_from_artifact(artifact: str) -> str:
    if artifact.endswith(".tar.zst"):
        return artifact[: -len(".tar.zst")]
    if artifact.endswith(".tar"):
        return artifact[: -len(".tar")]
    return Path(artifact).stem


def _token_cache_manifest(data_root: Path) -> Path:
    token_cache_dir = _env("FRACTAL_SCOUT_TOKEN_CACHE_DIR", "")
    if token_cache_dir:
        return data_root / token_cache_dir / "manifest.json"
    artifact = _env("FRACTAL_SCOUT_TOKEN_CACHE_ARTIFACT", DEFAULT_TOKEN_CACHE_ARTIFACT)
    return data_root / _cache_dir_name_from_artifact(artifact) / "manifest.json"


def _pip_install_if_needed(*, include_hf: bool, include_transformers: bool, include_mamba_kernels: bool) -> None:
    missing = []
    requirements = [
        ("pyarrow", "pyarrow>=23.0.0"),
        ("sentencepiece", "sentencepiece>=0.2.0"),
    ]
    if include_hf:
        requirements.append(("huggingface_hub", "huggingface_hub>=0.24.0"))
    if include_transformers:
        requirements.append(("transformers", "transformers>=4.51.0"))
    for module_name, requirement in requirements:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(requirement)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    if include_mamba_kernels:
        kernel_specs = [
            "kernels>=0.13.0",
            "causal-conv1d>=1.5.3.post1",
            "mamba-ssm>=2.2.6.post3",
        ]
        command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", *kernel_specs]
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


def _install_flash_attn_if_requested() -> None:
    if _env("FRACTAL_SCOUT_INSTALL_FLASH_ATTN", "false").lower() not in {"1", "true", "yes"}:
        return
    version = _env("FRACTAL_SCOUT_FLASH_ATTN_VERSION", "2.8.3")
    pip_cache_dir = ROOT / ".pip-cache"
    pip_tmp_dir = ROOT / ".pip-tmp"
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    pip_tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(pip_cache_dir)
    env["TMPDIR"] = str(pip_tmp_dir)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        f"flash-attn=={version}",
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=pip_cache_dir, env=env, check=True)


def _ensure_zstd() -> None:
    if shutil.which("zstd") is not None:
        return
    if shutil.which("apt-get") is None:
        raise SystemExit("zstd is required to extract the token cache, and apt-get is not available")
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "zstd"], check=True)


def _lane_args(
    lane: str,
    *,
    loop_count: str,
    hourglass_pass_count: str,
    hourglass_band_schedule: str,
    backward_steps: str,
    prelude_norm_kind: str,
    discretization: str,
    dt_raw_init: str,
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
            loop_count,
            "--parcae-hourglass-pass-count",
            hourglass_pass_count,
            "--parcae-prelude-norm-kind",
            prelude_norm_kind,
            "--parcae-discretization",
            discretization,
            "--parcae-dt-raw-init",
            dt_raw_init,
        ]
        if hourglass_band_schedule:
            args.extend(["--parcae-hourglass-band-schedule", hourglass_band_schedule])
        if backward_steps:
            args.extend(["--parcae-backward-steps", backward_steps])
        return args
    raise SystemExit(f"unsupported lane: {lane}")


def _external_lm_lane_command(lane: str, *, manifest: Path, output_dir: Path, ledger_path: Path) -> list[str]:
    external_model = EXTERNAL_LM_LANES[lane]
    seq_len = _env("FRACTAL_SCOUT_SEQ_LEN", "256")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_external_lm_baseline.py"),
        "--external-model",
        external_model,
        "--backend",
        "cuda",
        "--cuda-device",
        "0",
        "--dtype",
        _env("FRACTAL_SCOUT_DTYPE", "bf16"),
        "--env-kind",
        "requirements-only",
        "--seed",
        _env("FRACTAL_SCOUT_SEED", "42"),
        "--data-seed",
        _env("FRACTAL_SCOUT_DATA_SEED", "42"),
        "--tokenized-manifest-path",
        str(manifest),
        "--seq-len",
        seq_len,
        "--window-stride",
        str(int(seq_len) + 1),
        "--batch-size",
        _env("FRACTAL_SCOUT_BATCH_SIZE", "16"),
        "--steps",
        _env("FRACTAL_SCOUT_STEPS", "20"),
        "--eval-batches",
        _env("FRACTAL_SCOUT_EVAL_BATCHES", "4"),
        "--train-loss-record-interval",
        _env("FRACTAL_SCOUT_TRAIN_LOSS_RECORD_INTERVAL", "1"),
        "--warmup-train-steps",
        _env("FRACTAL_SCOUT_WARMUP_TRAIN_STEPS", "0"),
        "--warmup-eval-batches",
        _env("FRACTAL_SCOUT_WARMUP_EVAL_BATCHES", "0"),
        "--learning-rate",
        _env("FRACTAL_SCOUT_LEARNING_RATE", "0.001"),
        "--optimizer-profile",
        _env("FRACTAL_SCOUT_OPTIMIZER_PROFILE", "adam"),
        "--muon-weight-decay",
        _env("FRACTAL_SCOUT_MUON_WEIGHT_DECAY", "0.0"),
        "--muon-momentum",
        _env("FRACTAL_SCOUT_MUON_MOMENTUM", "0.95"),
        "--muon-ns-steps",
        _env("FRACTAL_SCOUT_MUON_NS_STEPS", "5"),
        "--max-position-embeddings",
        _env("FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS", "1024"),
        "--output-dir",
        str(output_dir),
        "--ledger-path",
        str(ledger_path),
        "--run-label",
        f"{_env('FRACTAL_SCOUT_RUN_LABEL', 'sagemaker-external-lm-scout')}-{lane}",
    ]
    muon_adjust_lr_fn = os.environ.get("FRACTAL_SCOUT_MUON_ADJUST_LR_FN", "").strip()
    if muon_adjust_lr_fn:
        command.extend(["--muon-adjust-lr-fn", muon_adjust_lr_fn])
    return command


def _write_combined_summary(*, lanes: list[str]) -> None:
    rows = []
    for report_path in sorted(OUT_ROOT.glob("**/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        runtime = report["runtime"]
        diagnostics = report.get("diagnostics") or {}
        cuda_memory = runtime.get("cuda_device_memory") or {}
        rows.append(
            {
                "model_label": report["model_label"],
                "implementation_kind": report["implementation_kind"],
                "parameters": diagnostics.get("parameter_count"),
                "initial_loss": report["initial_eval"]["mean_loss"],
                "final_loss": report["final_eval"]["mean_loss"],
                "train_tokens_per_second": runtime["train_tokens_per_second"],
                "overall_tokens_per_second": runtime["overall_tokens_per_second"],
                "peak_cuda_memory_mb": (cuda_memory.get("peak_used_bytes") or 0) / (1024 * 1024),
                "report_path": str(report_path),
            }
        )
    summary = {
        "run_label": _env("FRACTAL_SCOUT_RUN_LABEL", "sagemaker-path1-cuda-scout"),
        "lanes": lanes,
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SageMaker Path 1 CUDA Scout",
        "",
        "| Model | Kind | Params | Initial Loss | Final Loss | train tok/s | overall tok/s | Peak CUDA MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['implementation_kind']} | {row.get('parameters') or ''} | "
            f"{row['initial_loss']:.4f} | {row['final_loss']:.4f} | "
            f"{row['train_tokens_per_second']:.2f} | {row['overall_tokens_per_second']:.2f} | "
            f"{row['peak_cuda_memory_mb']:.2f} |"
        )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _profile_lane_command(lane: str, *, manifest: Path, data_root: Path) -> list[str]:
    seq_len = _env("FRACTAL_SCOUT_SEQ_LEN", "256")
    primitive_backend = _primitive_backend_for_lane(lane)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_python_path1_profile.py"),
        "--backend",
        "cuda",
        "--cuda-device",
        "0",
        "--dtype",
        _env("FRACTAL_SCOUT_DTYPE", "bf16"),
        "--env-kind",
        _env_kind_for_primitive_backend(primitive_backend),
        "--primitive-runtime-backend",
        primitive_backend,
        "--head-loss-backend",
        _env("FRACTAL_SCOUT_HEAD_LOSS_BACKEND", "dense"),
        "--ffn-backend",
        _env("FRACTAL_SCOUT_FFN_BACKEND", "dense"),
        "--seed",
        _env("FRACTAL_SCOUT_SEED", "42"),
        "--data-seed",
        _env("FRACTAL_SCOUT_DATA_SEED", "42"),
        "--corpus-format",
        "token-ids",
        "--tokenized-manifest-path",
        str(manifest),
        "--seq-len",
        seq_len,
        "--window-stride",
        str(int(seq_len) + 1),
        "--batch-size",
        _env("FRACTAL_SCOUT_BATCH_SIZE", "16"),
        "--steps",
        "1",
        "--eval-batches",
        _env("FRACTAL_SCOUT_EVAL_BATCHES", "1"),
        "--train-loss-record-interval",
        "1",
        "--warmup-train-steps",
        _env("FRACTAL_SCOUT_PROFILE_WARMUP_TRAIN_STEPS", "1"),
        "--warmup-eval-batches",
        _env("FRACTAL_SCOUT_PROFILE_WARMUP_EVAL_BATCHES", "0"),
        "--learning-rate",
        _env("FRACTAL_SCOUT_LEARNING_RATE", "0.001"),
        "--optimizer-profile",
        _env("FRACTAL_SCOUT_OPTIMIZER_PROFILE", "adam"),
        "--muon-weight-decay",
        _env("FRACTAL_SCOUT_MUON_WEIGHT_DECAY", "0.0"),
        "--muon-momentum",
        _env("FRACTAL_SCOUT_MUON_MOMENTUM", "0.95"),
        "--muon-ns-steps",
        _env("FRACTAL_SCOUT_MUON_NS_STEPS", "5"),
        "--d-model",
        _env("FRACTAL_SCOUT_D_MODEL", "128"),
        "--head-count",
        _env("FRACTAL_SCOUT_HEAD_COUNT", "4"),
        "--total-layers",
        _env("FRACTAL_SCOUT_TOTAL_LAYERS", "8"),
        "--local-window",
        _env("FRACTAL_SCOUT_LOCAL_WINDOW", "256"),
        "--attention-kernel",
        _env("FRACTAL_SCOUT_ATTENTION_KERNEL", "sdpa"),
        "--ffn-multiplier",
        _env("FRACTAL_SCOUT_FFN_MULTIPLIER", "4"),
        "--position-encoding-kind",
        _env("FRACTAL_SCOUT_POSITION_ENCODING_KIND", "none"),
        "--attention-position-contract",
        _env("FRACTAL_SCOUT_ATTENTION_POSITION_CONTRACT", "shared-input"),
        "--max-position-embeddings",
        _env("FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS", "1024"),
        "--output",
        "json",
        "--profile-row-limit",
        _env("FRACTAL_SCOUT_PROFILE_ROW_LIMIT", "40"),
        "--profile-output-dir",
        str(OUT_ROOT / f"profile-{lane}"),
        "--run-label",
        f"{_env('FRACTAL_SCOUT_RUN_LABEL', 'sagemaker-path1-cuda-profile')}-{lane}",
    ]
    for env_name, cli_name in (
        ("FRACTAL_SCOUT_PARCAE_LOOP_D_MODEL", "--parcae-loop-d-model"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_HEAD_COUNT", "--parcae-loop-head-count"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_FFN_MULTIPLIER", "--parcae-loop-ffn-multiplier"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_LAYER_COUNT", "--parcae-loop-layer-count"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value and "hourglass" in lane:
            command.extend([cli_name, value])
    if "p20-control" in lane:
        command.extend(
            [
                "--parcae-control-position-kind",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_KIND", "none"),
                "--parcae-control-position-scale-init",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_SCALE_INIT", "0.01"),
                "--parcae-control-stride",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_STRIDE", "1"),
                "--parcae-control-state-transform",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_STATE_TRANSFORM", "trainable"),
                "--parcae-recurrent-compile-mode",
                _env("FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE", "reduce-overhead"),
                "--parcae-loop-update-backend",
                _env("FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND", "eager"),
                "--parcae-scaffold-backend",
                _env("FRACTAL_SCOUT_PARCAE_SCAFFOLD_BACKEND", "standard"),
                "--parcae-band-block-contract",
                _env("FRACTAL_SCOUT_PARCAE_BAND_BLOCK_CONTRACT", "generic"),
                "--parcae-band-prepare-backend",
                _env("FRACTAL_SCOUT_PARCAE_BAND_PREPARE_BACKEND", "standard"),
                "--parcae-output-mix-backend",
                _env("FRACTAL_SCOUT_PARCAE_OUTPUT_MIX_BACKEND", "standard"),
            ]
        )
    compile_mode = os.environ.get("FRACTAL_SCOUT_COMPILE_MODE", "").strip()
    if compile_mode and primitive_backend != "triton":
        command.extend(["--compile-mode", compile_mode])
    elif compile_mode:
        print(
            "skipping global compile mode for primitive-triton env; "
            "use FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE for recurrent compilation",
            flush=True,
        )
    muon_adjust_lr_fn = os.environ.get("FRACTAL_SCOUT_MUON_ADJUST_LR_FN", "").strip()
    if muon_adjust_lr_fn:
        command.extend(["--muon-adjust-lr-fn", muon_adjust_lr_fn])
    if lane in PARCAE_LANES and _env("FRACTAL_SCOUT_PARCAE_FUSE_FIRST_STATE_MIX", "false").lower() in {"1", "true", "yes"}:
        command.append("--parcae-fuse-first-state-mix")
    command.extend(
        _lane_args(
            lane,
            loop_count=_env("FRACTAL_SCOUT_PARCAE_LOOP_COUNT", "2"),
            hourglass_pass_count=_env("FRACTAL_SCOUT_PARCAE_HOURGLASS_PASS_COUNT", "1"),
            hourglass_band_schedule=_env("FRACTAL_SCOUT_PARCAE_HOURGLASS_BAND_SCHEDULE", ""),
            backward_steps=os.environ.get("FRACTAL_SCOUT_PARCAE_BACKWARD_STEPS", "").strip(),
            prelude_norm_kind=_env("FRACTAL_SCOUT_PARCAE_PRELUDE_NORM_KIND", "layernorm"),
            discretization=_env("FRACTAL_SCOUT_PARCAE_DISCRETIZATION", "stable-exp"),
            dt_raw_init=_env("FRACTAL_SCOUT_PARCAE_DT_RAW_INIT", "0.54132485"),
        )
    )
    return command


def _timing_lane_command(lane: str, *, manifest: Path) -> list[str]:
    seq_len = _env("FRACTAL_SCOUT_SEQ_LEN", "256")
    primitive_backend = _primitive_backend_for_lane(lane)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_python_path1_timing.py"),
        "--backend",
        "cuda",
        "--cuda-device",
        "0",
        "--dtype",
        _env("FRACTAL_SCOUT_DTYPE", "bf16"),
        "--env-kind",
        _env_kind_for_primitive_backend(primitive_backend),
        "--primitive-runtime-backend",
        primitive_backend,
        "--head-loss-backend",
        _env("FRACTAL_SCOUT_HEAD_LOSS_BACKEND", "dense"),
        "--ffn-backend",
        _env("FRACTAL_SCOUT_FFN_BACKEND", "dense"),
        "--seed",
        _env("FRACTAL_SCOUT_SEED", "42"),
        "--data-seed",
        _env("FRACTAL_SCOUT_DATA_SEED", "42"),
        "--corpus-format",
        "token-ids",
        "--tokenized-manifest-path",
        str(manifest),
        "--seq-len",
        seq_len,
        "--window-stride",
        str(int(seq_len) + 1),
        "--batch-size",
        _env("FRACTAL_SCOUT_BATCH_SIZE", "16"),
        "--steps",
        "1",
        "--eval-batches",
        _env("FRACTAL_SCOUT_EVAL_BATCHES", "1"),
        "--train-loss-record-interval",
        "1",
        "--warmup-train-steps",
        _env(
            "FRACTAL_SCOUT_TIMING_WARMUP_TRAIN_STEPS",
            _env("FRACTAL_SCOUT_WARMUP_TRAIN_STEPS", "3"),
        ),
        "--warmup-eval-batches",
        _env(
            "FRACTAL_SCOUT_TIMING_WARMUP_EVAL_BATCHES",
            _env("FRACTAL_SCOUT_WARMUP_EVAL_BATCHES", "0"),
        ),
        "--learning-rate",
        _env("FRACTAL_SCOUT_LEARNING_RATE", "0.001"),
        "--optimizer-profile",
        _env("FRACTAL_SCOUT_OPTIMIZER_PROFILE", "adam"),
        "--muon-weight-decay",
        _env("FRACTAL_SCOUT_MUON_WEIGHT_DECAY", "0.0"),
        "--muon-momentum",
        _env("FRACTAL_SCOUT_MUON_MOMENTUM", "0.95"),
        "--muon-ns-steps",
        _env("FRACTAL_SCOUT_MUON_NS_STEPS", "5"),
        "--d-model",
        _env("FRACTAL_SCOUT_D_MODEL", "128"),
        "--head-count",
        _env("FRACTAL_SCOUT_HEAD_COUNT", "4"),
        "--total-layers",
        _env("FRACTAL_SCOUT_TOTAL_LAYERS", "8"),
        "--local-window",
        _env("FRACTAL_SCOUT_LOCAL_WINDOW", "256"),
        "--attention-kernel",
        _env("FRACTAL_SCOUT_ATTENTION_KERNEL", "sdpa"),
        "--ffn-multiplier",
        _env("FRACTAL_SCOUT_FFN_MULTIPLIER", "4"),
        "--position-encoding-kind",
        _env("FRACTAL_SCOUT_POSITION_ENCODING_KIND", "none"),
        "--attention-position-contract",
        _env("FRACTAL_SCOUT_ATTENTION_POSITION_CONTRACT", "shared-input"),
        "--max-position-embeddings",
        _env("FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS", "1024"),
        "--output",
        "json",
        "--timing-steps",
        _env("FRACTAL_SCOUT_TIMING_STEPS", "20"),
        "--timing-output-dir",
        str(OUT_ROOT / f"timing-{lane}"),
        "--run-label",
        f"{_env('FRACTAL_SCOUT_RUN_LABEL', 'sagemaker-path1-cuda-timing')}-{lane}",
    ]
    for env_name, cli_name in (
        ("FRACTAL_SCOUT_PARCAE_LOOP_D_MODEL", "--parcae-loop-d-model"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_HEAD_COUNT", "--parcae-loop-head-count"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_FFN_MULTIPLIER", "--parcae-loop-ffn-multiplier"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_LAYER_COUNT", "--parcae-loop-layer-count"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value and "hourglass" in lane:
            command.extend([cli_name, value])
    if "p20-control" in lane:
        command.extend(
            [
                "--parcae-control-position-kind",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_KIND", "none"),
                "--parcae-control-position-scale-init",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_SCALE_INIT", "0.01"),
                "--parcae-control-stride",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_STRIDE", "1"),
                "--parcae-control-state-transform",
                _env("FRACTAL_SCOUT_PARCAE_CONTROL_STATE_TRANSFORM", "trainable"),
                "--parcae-recurrent-compile-mode",
                _env("FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE", "reduce-overhead"),
                "--parcae-loop-update-backend",
                _env("FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND", "eager"),
                "--parcae-scaffold-backend",
                _env("FRACTAL_SCOUT_PARCAE_SCAFFOLD_BACKEND", "standard"),
                "--parcae-band-block-contract",
                _env("FRACTAL_SCOUT_PARCAE_BAND_BLOCK_CONTRACT", "generic"),
                "--parcae-band-prepare-backend",
                _env("FRACTAL_SCOUT_PARCAE_BAND_PREPARE_BACKEND", "standard"),
            ]
        )
    compile_mode = os.environ.get("FRACTAL_SCOUT_COMPILE_MODE", "").strip()
    if compile_mode and primitive_backend != "triton":
        command.extend(["--compile-mode", compile_mode])
    elif compile_mode:
        print(
            "skipping global compile mode for primitive-triton env; "
            "use FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE for recurrent compilation",
            flush=True,
        )
    muon_adjust_lr_fn = os.environ.get("FRACTAL_SCOUT_MUON_ADJUST_LR_FN", "").strip()
    if muon_adjust_lr_fn:
        command.extend(["--muon-adjust-lr-fn", muon_adjust_lr_fn])
    if _env("FRACTAL_SCOUT_CUDA_GRAPH_STEP", "false").lower() in {"1", "true", "yes"}:
        command.append("--cuda-graph-step")
    if lane in PARCAE_LANES and _env("FRACTAL_SCOUT_PARCAE_FUSE_FIRST_STATE_MIX", "false").lower() in {"1", "true", "yes"}:
        command.append("--parcae-fuse-first-state-mix")
    command.extend(
        _lane_args(
            lane,
            loop_count=_env("FRACTAL_SCOUT_PARCAE_LOOP_COUNT", "2"),
            hourglass_pass_count=_env("FRACTAL_SCOUT_PARCAE_HOURGLASS_PASS_COUNT", "1"),
            hourglass_band_schedule=_env("FRACTAL_SCOUT_PARCAE_HOURGLASS_BAND_SCHEDULE", ""),
            backward_steps=os.environ.get("FRACTAL_SCOUT_PARCAE_BACKWARD_STEPS", "").strip(),
            prelude_norm_kind=_env("FRACTAL_SCOUT_PARCAE_PRELUDE_NORM_KIND", "layernorm"),
            discretization=_env("FRACTAL_SCOUT_PARCAE_DISCRETIZATION", "stable-exp"),
            dt_raw_init=_env("FRACTAL_SCOUT_PARCAE_DT_RAW_INIT", "0.54132485"),
        )
    )
    return command


def _write_nsys_unavailable(lane: str, *, report_dir: Path, nsys_path: str | None) -> None:
    payload = {
        "lane": lane,
        "nsys_available": False,
        "nsys_path": nsys_path,
        "fallback": "cuda_event_timing",
        "reason": "nsys was not found in the SageMaker training image PATH",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "nsys_unavailable.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_nsys_lane(lane: str, *, manifest: Path) -> None:
    report_dir = OUT_ROOT / f"nsys-{lane}"
    report_dir.mkdir(parents=True, exist_ok=True)
    timing_command = _timing_lane_command(lane, manifest=manifest)
    nsys_path = shutil.which("nsys")
    if nsys_path is None:
        _write_nsys_unavailable(lane, report_dir=report_dir, nsys_path=nsys_path)
        print(f"warning: nsys not found; falling back to CUDA-event timing for {lane}", flush=True)
        print("+ " + " ".join(timing_command), flush=True)
        subprocess.run(timing_command, cwd=ROOT, check=True)
        return

    command = [
        nsys_path,
        "profile",
        "--trace",
        _env("FRACTAL_SCOUT_NSYS_TRACE", "cuda,nvtx,cublas,cudnn"),
        "--stats",
        _env("FRACTAL_SCOUT_NSYS_STATS", "true"),
        "--force-overwrite",
        "true",
        "-o",
        str(report_dir / "path1"),
        *timing_command,
    ]
    env = os.environ.copy()
    env["FRACTAL_ENABLE_NVTX_RANGES"] = "true"
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> int:
    _probe_cuda()
    data_root = Path(_env("FRACTAL_SCOUT_DATA_ROOT", str(DATA_ROOT)))
    manifest = _token_cache_manifest(data_root)
    has_mounted_token_cache = manifest.exists()
    print(
        "token_cache_probe="
        + json.dumps(
            {
                "data_root": str(data_root),
                "manifest": str(manifest),
                "manifest_exists": has_mounted_token_cache,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    lanes = [lane.strip() for lane in _env("FRACTAL_SCOUT_LANES", "attention-only,parcae-bx-looped-attention,parcae-p20-control-looped-attention").split(",") if lane.strip()]
    has_external_lanes = any(lane in EXTERNAL_LM_LANES for lane in lanes)
    has_external_mamba_lane = any("mamba" in lane for lane in lanes)
    _pip_install_if_needed(
        include_hf=not has_mounted_token_cache,
        include_transformers=has_external_lanes,
        include_mamba_kernels=has_external_mamba_lane,
    )
    _install_flash_attn_if_requested()
    if not has_mounted_token_cache:
        _ensure_zstd()
        _load_hf_env_channel()
        if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"].strip():
            raise SystemExit(
                "HF_TOKEN is required for token-cache SageMaker scout mode without a mounted token cache; "
                "provide the hf_env input channel or mount an S3 token cache"
            )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    profile_mode = _env("FRACTAL_SCOUT_PROFILE", "false").lower() in {"1", "true", "yes"}
    timing_mode = _env("FRACTAL_SCOUT_CUDA_TIMING", "false").lower() in {"1", "true", "yes"}
    nsys_mode = _env("FRACTAL_SCOUT_NSYS", "false").lower() in {"1", "true", "yes"}
    active_modes = sum(1 for enabled in (profile_mode, timing_mode, nsys_mode) if enabled)
    if active_modes > 1:
        raise SystemExit("FRACTAL_SCOUT_PROFILE, FRACTAL_SCOUT_CUDA_TIMING, and FRACTAL_SCOUT_NSYS are mutually exclusive")
    if profile_mode:
        if not has_mounted_token_cache:
            raise SystemExit("token-cache profile mode requires a mounted token cache manifest")
        if any(lane in EXTERNAL_LM_LANES for lane in lanes):
            raise SystemExit("token-cache profile mode does not support external Hugging Face LM lanes")
        for lane in lanes:
            command = _profile_lane_command(lane, manifest=manifest, data_root=data_root)
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return 0
    if timing_mode:
        if not has_mounted_token_cache:
            raise SystemExit("token-cache CUDA timing mode requires a mounted token cache manifest")
        if any(lane in EXTERNAL_LM_LANES for lane in lanes):
            raise SystemExit("token-cache CUDA timing mode does not support external Hugging Face LM lanes")
        for lane in lanes:
            command = _timing_lane_command(lane, manifest=manifest)
            print("+ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
        return 0
    if nsys_mode:
        if not has_mounted_token_cache:
            raise SystemExit("token-cache Nsight Systems mode requires a mounted token cache manifest")
        if any(lane in EXTERNAL_LM_LANES for lane in lanes):
            raise SystemExit("token-cache Nsight Systems mode does not support external Hugging Face LM lanes")
        for lane in lanes:
            _run_nsys_lane(lane, manifest=manifest)
        return 0
    external_lanes = [lane for lane in lanes if lane in EXTERNAL_LM_LANES]
    path1_lanes = [lane for lane in lanes if lane not in EXTERNAL_LM_LANES]
    ledger_path = OUT_ROOT / "ledger.jsonl"
    for lane in external_lanes:
        command = _external_lm_lane_command(
            lane,
            manifest=manifest,
            output_dir=OUT_ROOT / lane,
            ledger_path=ledger_path,
        )
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    if not path1_lanes:
        _write_combined_summary(lanes=lanes)
        print((OUT_ROOT / "summary.md").read_text(encoding="utf-8"), flush=True)
        return 0
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_python_path1_parcae_h100_promotion.py"),
        "--cuda-device",
        "0",
        "--dtype",
        _env("FRACTAL_SCOUT_DTYPE", "bf16"),
        "--seed",
        _env("FRACTAL_SCOUT_SEED", "42"),
        "--data-seed",
        _env("FRACTAL_SCOUT_DATA_SEED", "42"),
        "--seq-len",
        _env("FRACTAL_SCOUT_SEQ_LEN", "256"),
        "--batch-size",
        _env("FRACTAL_SCOUT_BATCH_SIZE", "16"),
        "--steps",
        _env("FRACTAL_SCOUT_STEPS", "20"),
        "--eval-batches",
        _env("FRACTAL_SCOUT_EVAL_BATCHES", "4"),
        "--train-loss-record-interval",
        _env("FRACTAL_SCOUT_TRAIN_LOSS_RECORD_INTERVAL", "1"),
        "--warmup-train-steps",
        _env("FRACTAL_SCOUT_WARMUP_TRAIN_STEPS", "0"),
        "--warmup-eval-batches",
        _env("FRACTAL_SCOUT_WARMUP_EVAL_BATCHES", "0"),
        "--learning-rate",
        _env("FRACTAL_SCOUT_LEARNING_RATE", "0.001"),
        "--optimizer-profile",
        _env("FRACTAL_SCOUT_OPTIMIZER_PROFILE", "adam"),
        "--muon-weight-decay",
        _env("FRACTAL_SCOUT_MUON_WEIGHT_DECAY", "0.0"),
        "--muon-momentum",
        _env("FRACTAL_SCOUT_MUON_MOMENTUM", "0.95"),
        "--muon-ns-steps",
        _env("FRACTAL_SCOUT_MUON_NS_STEPS", "5"),
        "--d-model",
        _env("FRACTAL_SCOUT_D_MODEL", "128"),
        "--head-count",
        _env("FRACTAL_SCOUT_HEAD_COUNT", "4"),
        "--total-layers",
        _env("FRACTAL_SCOUT_TOTAL_LAYERS", "8"),
        "--local-window",
        _env("FRACTAL_SCOUT_LOCAL_WINDOW", "256"),
        "--attention-kernel",
        _env("FRACTAL_SCOUT_ATTENTION_KERNEL", "sdpa"),
        "--ffn-multiplier",
        _env("FRACTAL_SCOUT_FFN_MULTIPLIER", "4"),
        "--parcae-loop-count",
        _env("FRACTAL_SCOUT_PARCAE_LOOP_COUNT", "2"),
        "--parcae-hourglass-pass-count",
        _env("FRACTAL_SCOUT_PARCAE_HOURGLASS_PASS_COUNT", "1"),
        "--parcae-hourglass-band-schedule",
        _env("FRACTAL_SCOUT_PARCAE_HOURGLASS_BAND_SCHEDULE", ""),
        "--parcae-prelude-norm-kind",
        _env("FRACTAL_SCOUT_PARCAE_PRELUDE_NORM_KIND", "layernorm"),
        "--parcae-discretization",
        _env("FRACTAL_SCOUT_PARCAE_DISCRETIZATION", "stable-exp"),
        "--parcae-dt-raw-init",
        _env("FRACTAL_SCOUT_PARCAE_DT_RAW_INIT", "0.54132485"),
        "--position-encoding-kind",
        _env("FRACTAL_SCOUT_POSITION_ENCODING_KIND", "none"),
        "--attention-position-contract",
        _env("FRACTAL_SCOUT_ATTENTION_POSITION_CONTRACT", "shared-input"),
        "--max-position-embeddings",
        _env("FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS", "1024"),
        "--final-norm-kind",
        _env("FRACTAL_SCOUT_FINAL_NORM_KIND", "identity"),
        "--primitive-runtime-backend",
        _env("FRACTAL_SCOUT_PRIMITIVE_RUNTIME_BACKEND", "torch"),
        "--head-loss-backend",
        _env("FRACTAL_SCOUT_HEAD_LOSS_BACKEND", "dense"),
        "--ffn-backend",
        _env("FRACTAL_SCOUT_FFN_BACKEND", "dense"),
        "--token-cache-repo-id",
        _env("FRACTAL_SCOUT_TOKEN_CACHE_REPO_ID", "joebud/fractal-fineweb-openllama-tokens"),
        "--token-cache-artifact",
        _env("FRACTAL_SCOUT_TOKEN_CACHE_ARTIFACT", "fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst"),
        "--data-root",
        str(data_root),
        "--lanes",
        ",".join(path1_lanes),
        "--output-dir",
        str(OUT_ROOT),
        "--run-label",
        _env("FRACTAL_SCOUT_RUN_LABEL", "sagemaker-path1-cuda-scout"),
    ]
    token_cache_dir = os.environ.get("FRACTAL_SCOUT_TOKEN_CACHE_DIR", "").strip()
    if token_cache_dir:
        command.extend(["--token-cache-dir", token_cache_dir])
    parcae_backward_steps = os.environ.get("FRACTAL_SCOUT_PARCAE_BACKWARD_STEPS", "").strip()
    if parcae_backward_steps:
        command.extend(["--parcae-backward-steps", parcae_backward_steps])
    for env_name, cli_name in (
        ("FRACTAL_SCOUT_PARCAE_LOOP_D_MODEL", "--parcae-loop-d-model"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_HEAD_COUNT", "--parcae-loop-head-count"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_FFN_MULTIPLIER", "--parcae-loop-ffn-multiplier"),
        ("FRACTAL_SCOUT_PARCAE_LOOP_LAYER_COUNT", "--parcae-loop-layer-count"),
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            command.extend([cli_name, value])
    command.extend(
        [
            "--parcae-control-position-kind",
            _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_KIND", "none"),
            "--parcae-control-position-scale-init",
            _env("FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_SCALE_INIT", "0.01"),
            "--parcae-control-stride",
            _env("FRACTAL_SCOUT_PARCAE_CONTROL_STRIDE", "1"),
            "--parcae-control-state-transform",
            _env("FRACTAL_SCOUT_PARCAE_CONTROL_STATE_TRANSFORM", "trainable"),
            "--parcae-recurrent-compile-mode",
            _env("FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE", "reduce-overhead"),
            "--parcae-loop-update-backend",
            _env("FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND", "eager"),
            "--parcae-scaffold-backend",
            _env("FRACTAL_SCOUT_PARCAE_SCAFFOLD_BACKEND", "standard"),
            "--parcae-band-block-contract",
            _env("FRACTAL_SCOUT_PARCAE_BAND_BLOCK_CONTRACT", "generic"),
            "--parcae-band-prepare-backend",
            _env("FRACTAL_SCOUT_PARCAE_BAND_PREPARE_BACKEND", "standard"),
            "--parcae-output-mix-backend",
            _env("FRACTAL_SCOUT_PARCAE_OUTPUT_MIX_BACKEND", "standard"),
        ]
    )
    compile_mode = os.environ.get("FRACTAL_SCOUT_COMPILE_MODE", "").strip()
    primitive_runtime_backend = _env("FRACTAL_SCOUT_PRIMITIVE_RUNTIME_BACKEND", "torch")
    if compile_mode and primitive_runtime_backend != "triton":
        command.extend(["--compile-mode", compile_mode])
    elif compile_mode:
        print(
            "skipping global compile mode for primitive-triton env; "
            "use FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE for recurrent compilation",
            flush=True,
        )
    muon_adjust_lr_fn = os.environ.get("FRACTAL_SCOUT_MUON_ADJUST_LR_FN", "").strip()
    if muon_adjust_lr_fn:
        command.extend(["--muon-adjust-lr-fn", muon_adjust_lr_fn])
    if _env("FRACTAL_SCOUT_PARCAE_FUSE_FIRST_STATE_MIX", "false").lower() in {"1", "true", "yes"}:
        command.append("--parcae-fuse-first-state-mix")
    if _env("FRACTAL_SCOUT_FORCE_DOWNLOAD", "false").lower() in {"1", "true", "yes"}:
        command.append("--force-download")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    if external_lanes:
        _write_combined_summary(lanes=lanes)
    summary = OUT_ROOT / "summary.md"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

MAMBA_WHEELHOUSE_ENTRYPOINT = r'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


OUT_ROOT = Path("/opt/ml/model/mamba-wheelhouse")
WHEEL_DIR = OUT_ROOT / "wheels"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def _cuda_contract() -> dict[str, object]:
    import torch

    payload: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return payload


def _assert_fast_path() -> dict[str, object]:
    from mamba_ssm.modules import mamba_simple
    from mamba_ssm.ops import selective_scan_interface

    flags = {
        "selective_scan_fn": selective_scan_interface.selective_scan_fn is not None,
        "mamba_inner_fn": selective_scan_interface.mamba_inner_fn is not None,
        "causal_conv1d_fn": getattr(mamba_simple, "causal_conv1d_fn", None) is not None,
    }
    missing = [name for name, available in flags.items() if not available]
    if missing:
        raise SystemExit("official Mamba fast path unavailable after wheelhouse install: " + ", ".join(missing))
    return flags


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)

    arch_list = _env("FRACTAL_MAMBA_WHEELHOUSE_TORCH_CUDA_ARCH_LIST", "8.9;9.0")
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = arch_list
    env.setdefault("MAX_JOBS", _env("FRACTAL_MAMBA_WHEELHOUSE_MAX_JOBS", "4"))

    native_specs = [
        f"kernels=={_env('FRACTAL_MAMBA_WHEELHOUSE_KERNELS_VERSION', '0.13.0')}",
        f"causal-conv1d=={_env('FRACTAL_MAMBA_WHEELHOUSE_CAUSAL_CONV1D_VERSION', '1.6.1')}",
        f"mamba-ssm=={_env('FRACTAL_MAMBA_WHEELHOUSE_MAMBA_SSM_VERSION', '2.3.1')}",
    ]
    runtime_specs = [
        f"transformers=={_env('FRACTAL_MAMBA_WHEELHOUSE_TRANSFORMERS_VERSION', '5.7.0')}",
        f"sentencepiece=={_env('FRACTAL_MAMBA_WHEELHOUSE_SENTENCEPIECE_VERSION', '0.2.1')}",
    ]
    forbidden_wheel_prefixes = ("torch-", "triton-", "nvidia_", "cuda_")

    contract = _cuda_contract()
    contract.update(
        {
            "native_package_specs": native_specs,
            "runtime_package_specs": runtime_specs,
            "wheel_build_dependency_policy": "native wheels are built with --no-deps against the container torch/CUDA",
            "forbidden_wheel_prefixes": forbidden_wheel_prefixes,
        }
    )
    (OUT_ROOT / "build-contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print("mamba_wheelhouse_build_contract=" + json.dumps(contract, sort_keys=True), flush=True)

    _run([sys.executable, "-m", "pip", "install", *runtime_specs], env=env)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(WHEEL_DIR),
            *native_specs,
        ],
        env=env,
    )
    wheel_names = sorted(path.name for path in WHEEL_DIR.glob("*.whl"))
    forbidden_wheels = [
        name
        for name in wheel_names
        if any(name.lower().startswith(prefix) for prefix in forbidden_wheel_prefixes)
    ]
    if forbidden_wheels:
        raise SystemExit(
            "wheelhouse dependency leak: native wheelhouse must not vendor torch/CUDA stack wheels: "
            + ", ".join(forbidden_wheels)
        )
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "wheels": wheel_names,
                "native_package_specs": native_specs,
                "runtime_package_specs": runtime_specs,
                "forbidden_wheel_prefixes": forbidden_wheel_prefixes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(WHEEL_DIR),
            *native_specs,
        ],
        env=env,
    )
    fast_path = _assert_fast_path()
    (OUT_ROOT / "preflight.json").write_text(json.dumps(fast_path, indent=2, sort_keys=True) + "\n")
    (OUT_ROOT / "README.md").write_text(
        "# Mamba wheelhouse\n\n"
        "Reusable SageMaker-built wheelhouse for official Mamba CUDA fast path.\n\n"
        f"- TORCH_CUDA_ARCH_LIST: `{arch_list}`\n"
        f"- Native package specs: `{', '.join(native_specs)}`\n"
        f"- Runtime package specs: `{', '.join(runtime_specs)}`\n"
        "- Native wheels are built with `--no-deps` and must not vendor torch/CUDA stack wheels.\n"
        "- Preflight requires selective scan, mamba inner, and causal-conv1d fast paths.\n",
        encoding="utf-8",
    )
    print("mamba_wheelhouse_ready=" + json.dumps({"wheel_count": len(wheel_names), **fast_path}, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _default_training_image(region: str) -> str:
    return (
        f"{DEFAULT_DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-training:{DEFAULT_DLC_TAG}"
    )


def _run_aws(
    args: argparse.Namespace,
    aws_args: Sequence[str],
    *,
    capture: bool = False,
) -> str:
    command = ["aws", *aws_args, "--region", args.region]
    if args.profile:
        command.extend(["--profile", args.profile])
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return completed.stdout if capture and completed.stdout is not None else ""


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def _stage_source_bundle(repo_root: Path, bundle_path: Path, *, runner: str = "smoke") -> None:
    with tempfile.TemporaryDirectory(prefix="fractal-sagemaker-src-") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        (stage / "scripts").mkdir()
        if runner == "mamba-wheelhouse":
            entrypoint_source = MAMBA_WHEELHOUSE_ENTRYPOINT
        else:
            _copy_tree(repo_root / "python", stage / "python")
            shutil.copy2(repo_root / "scripts" / "v3a_python_path1.py", stage / "scripts" / "v3a_python_path1.py")
            entrypoint_source = ENTRYPOINT
        if runner == "token-cache":
            shutil.copy2(
                repo_root / "scripts" / "v3a_python_path1_profile.py",
                stage / "scripts" / "v3a_python_path1_profile.py",
            )
            shutil.copy2(
                repo_root / "scripts" / "v3a_python_path1_timing.py",
                stage / "scripts" / "v3a_python_path1_timing.py",
            )
            shutil.copy2(
                repo_root / "scripts" / "v3a_python_path1_parcae_h100_promotion.py",
                stage / "scripts" / "v3a_python_path1_parcae_h100_promotion.py",
            )
            shutil.copy2(
                repo_root / "scripts" / "v3a_external_lm_baseline.py",
                stage / "scripts" / "v3a_external_lm_baseline.py",
            )
            entrypoint_source = TOKEN_CACHE_ENTRYPOINT
        elif runner == "smoke":
            corpus_src = repo_root / SMOKE_CORPUS_REL
            corpus_dst = stage / SMOKE_CORPUS_REL
            corpus_dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_tree(corpus_src, corpus_dst)
        entrypoint = stage / "sagemaker_path1_entrypoint.py"
        entrypoint.write_text(
            entrypoint_source,
            encoding="utf-8",
        )
        entrypoint.chmod(0o755)

        with tarfile.open(bundle_path, "w:gz") as tar:
            for path in sorted(stage.rglob("*")):
                tar.add(path, arcname=path.relative_to(stage))


def _env_or_arg(value: str | None, env_name: str) -> str | None:
    return value or os.environ.get(env_name)


def _lane_list(raw: str) -> list[str]:
    lanes = [LANE_ALIASES.get(lane.strip(), lane.strip()) for lane in raw.split(",") if lane.strip()]
    supported = {"attention-only", *LANES_WITH_PARCAE_SCAFFOLD, *EXTERNAL_LM_LANES}
    unknown = sorted(set(lanes) - supported)
    if unknown:
        raise SystemExit(f"unsupported lane(s): {', '.join(unknown)}")
    return lanes


def _validate_attention_head_dim_contract(
    *,
    width: int,
    head_count: int,
    label: str,
    attention_kernel: str,
    allow_slow_attention_head_dim: bool,
) -> None:
    if width % head_count != 0:
        raise SystemExit(f"{label} width {width} must be divisible by head_count {head_count}.")
    head_dim = width // head_count
    if attention_kernel == "flex-local" and head_dim & (head_dim - 1) != 0:
        raise SystemExit(
            f"{label} head_dim is {head_dim}, but attention_kernel=flex-local currently requires "
            "a power-of-two head_dim for PyTorch FlexAttention. Choose a width/head-count pair such "
            "as 448/7 or use --attention-kernel sdpa for this shape."
        )
    if head_dim % 8 == 0 or allow_slow_attention_head_dim:
        return
    raise SystemExit(
        f"{label} head_dim is {head_dim}, which is not a CUDA SDPA-friendly multiple of 8. "
        "This can silently force a much slower attention kernel. Choose a width/head-count pair "
        "with head_dim % 8 == 0, or pass --allow-slow-attention-head-dim for an intentional slow-path probe."
    )


def _validate_cuda_attention_shape_contract(args: argparse.Namespace) -> None:
    lanes = _lane_list(args.lanes)
    path1_lanes = [lane for lane in lanes if lane not in EXTERNAL_LM_LANES]
    if not path1_lanes:
        return
    _validate_attention_head_dim_contract(
        width=args.d_model,
        head_count=args.head_count,
        label="outer attention",
        attention_kernel=args.attention_kernel,
        allow_slow_attention_head_dim=args.allow_slow_attention_head_dim,
    )
    if not any("hourglass" in lane for lane in path1_lanes):
        return
    if args.parcae_loop_d_model is None:
        return
    _validate_attention_head_dim_contract(
        width=args.parcae_loop_d_model,
        head_count=args.parcae_loop_head_count or args.head_count,
        label="Parcae loop attention",
        attention_kernel=args.attention_kernel,
        allow_slow_attention_head_dim=args.allow_slow_attention_head_dim,
    )


def _load_local_hf_token(env_file: Path | None = None) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    if env_file is not None and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "HF_TOKEN" and value.strip():
                return value.strip().strip("'\"")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        return token or None
    return None


def _write_hf_env_file(path: Path, *, token: str) -> None:
    path.write_text(f"HF_TOKEN={token}\n", encoding="utf-8")
    path.chmod(0o600)


def _s3_join(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def _training_request(
    args: argparse.Namespace,
    *,
    source_s3_prefix: str,
    output_s3_path: str,
    hf_env_s3_prefix: str | None = None,
) -> dict[str, Any]:
    role_arn = _env_or_arg(args.role_arn, "FRACTAL_SAGEMAKER_ROLE_ARN")
    if not role_arn:
        raise SystemExit(
            "SageMaker execution role is required. Set FRACTAL_SAGEMAKER_ROLE_ARN "
            "or pass --role-arn arn:aws:iam::<account>:role/<role-name>."
        )

    training_image = args.training_image or _default_training_image(args.region)
    extract_and_run = (
        "set -euo pipefail; "
        "mkdir -p /opt/ml/code; "
        "ls -lah /opt/ml/input/data/source; "
        "tar -xzf /opt/ml/input/data/source/source.tar.gz -C /opt/ml/code; "
        "cd /opt/ml/code; "
        "python sagemaker_path1_entrypoint.py"
    )
    env = {
        "PYTHONUNBUFFERED": "1",
    }
    if args.runner == "mamba-wheelhouse":
        env.update(
            {
                "FRACTAL_MAMBA_WHEELHOUSE_TORCH_CUDA_ARCH_LIST": args.mamba_wheelhouse_torch_cuda_arch_list,
                "FRACTAL_MAMBA_WHEELHOUSE_MAX_JOBS": str(args.mamba_wheelhouse_max_jobs),
                "FRACTAL_MAMBA_WHEELHOUSE_KERNELS_VERSION": args.mamba_wheelhouse_kernels_version,
                "FRACTAL_MAMBA_WHEELHOUSE_CAUSAL_CONV1D_VERSION": args.mamba_wheelhouse_causal_conv1d_version,
                "FRACTAL_MAMBA_WHEELHOUSE_MAMBA_SSM_VERSION": args.mamba_wheelhouse_mamba_ssm_version,
                "FRACTAL_MAMBA_WHEELHOUSE_TRANSFORMERS_VERSION": args.mamba_wheelhouse_transformers_version,
                "FRACTAL_MAMBA_WHEELHOUSE_SENTENCEPIECE_VERSION": args.mamba_wheelhouse_sentencepiece_version,
            }
        )
    else:
        lane_csv = ",".join(_lane_list(args.lanes))
    if args.runner == "token-cache":
        token_cache_s3_uri = args.token_cache_s3_uri.rstrip("/") + "/" if args.token_cache_s3_uri else None
        if not token_cache_s3_uri and not hf_env_s3_prefix:
            raise SystemExit(
                "An hf_env input channel is required for --runner token-cache without --token-cache-s3-uri. "
                "Use main() so the local .env/HF token can be staged safely outside the SageMaker Environment."
            )
        scout_env = {
            "FRACTAL_SCOUT_RUN_LABEL": args.job_name,
            "FRACTAL_SCOUT_LANES": lane_csv,
            "FRACTAL_SCOUT_STEPS": str(args.steps),
            "FRACTAL_SCOUT_EVAL_BATCHES": str(args.eval_batches),
            "FRACTAL_SCOUT_TRAIN_LOSS_RECORD_INTERVAL": str(args.train_loss_record_interval),
            "FRACTAL_SCOUT_WARMUP_TRAIN_STEPS": str(args.warmup_train_steps),
            "FRACTAL_SCOUT_WARMUP_EVAL_BATCHES": str(args.warmup_eval_batches),
            "FRACTAL_SCOUT_SEQ_LEN": str(args.seq_len),
            "FRACTAL_SCOUT_BATCH_SIZE": str(args.batch_size),
            "FRACTAL_SCOUT_DTYPE": args.dtype,
            "FRACTAL_SCOUT_D_MODEL": str(args.d_model),
            "FRACTAL_SCOUT_HEAD_COUNT": str(args.head_count),
            "FRACTAL_SCOUT_TOTAL_LAYERS": str(args.total_layers),
            "FRACTAL_SCOUT_LOCAL_WINDOW": str(args.local_window),
            "FRACTAL_SCOUT_ATTENTION_KERNEL": args.attention_kernel,
            "FRACTAL_SCOUT_FFN_MULTIPLIER": str(args.ffn_multiplier),
            "FRACTAL_SCOUT_PARCAE_LOOP_COUNT": str(args.parcae_loop_count),
            "FRACTAL_SCOUT_PARCAE_HOURGLASS_PASS_COUNT": str(args.parcae_hourglass_pass_count),
            "FRACTAL_SCOUT_PARCAE_HOURGLASS_BAND_SCHEDULE": args.parcae_hourglass_band_schedule or "",
            "FRACTAL_SCOUT_PARCAE_BACKWARD_STEPS": (
                "" if args.parcae_backward_steps is None else str(args.parcae_backward_steps)
            ),
            "FRACTAL_SCOUT_PARCAE_PRELUDE_NORM_KIND": args.parcae_prelude_norm_kind,
            "FRACTAL_SCOUT_PARCAE_DISCRETIZATION": args.parcae_discretization,
            "FRACTAL_SCOUT_PARCAE_DT_RAW_INIT": str(args.parcae_dt_raw_init),
            "FRACTAL_SCOUT_PARCAE_LOOP_D_MODEL": (
                "" if args.parcae_loop_d_model is None else str(args.parcae_loop_d_model)
            ),
            "FRACTAL_SCOUT_PARCAE_LOOP_HEAD_COUNT": (
                "" if args.parcae_loop_head_count is None else str(args.parcae_loop_head_count)
            ),
            "FRACTAL_SCOUT_PARCAE_LOOP_FFN_MULTIPLIER": (
                "" if args.parcae_loop_ffn_multiplier is None else str(args.parcae_loop_ffn_multiplier)
            ),
            "FRACTAL_SCOUT_PARCAE_LOOP_LAYER_COUNT": (
                "" if args.parcae_loop_layer_count is None else str(args.parcae_loop_layer_count)
            ),
            "FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_KIND": args.parcae_control_position_kind,
            "FRACTAL_SCOUT_PARCAE_CONTROL_POSITION_SCALE_INIT": str(args.parcae_control_position_scale_init),
            "FRACTAL_SCOUT_PARCAE_CONTROL_STRIDE": str(args.parcae_control_stride),
            "FRACTAL_SCOUT_PARCAE_CONTROL_STATE_TRANSFORM": args.parcae_control_state_transform,
            "FRACTAL_SCOUT_PARCAE_RECURRENT_COMPILE_MODE": args.parcae_recurrent_compile_mode,
            "FRACTAL_SCOUT_PARCAE_LOOP_UPDATE_BACKEND": args.parcae_loop_update_backend,
            "FRACTAL_SCOUT_PARCAE_SCAFFOLD_BACKEND": args.parcae_scaffold_backend,
            "FRACTAL_SCOUT_PARCAE_BAND_BLOCK_CONTRACT": args.parcae_band_block_contract,
            "FRACTAL_SCOUT_PARCAE_BAND_PREPARE_BACKEND": args.parcae_band_prepare_backend,
            "FRACTAL_SCOUT_PARCAE_OUTPUT_MIX_BACKEND": args.parcae_output_mix_backend,
            "FRACTAL_SCOUT_PARCAE_FUSE_FIRST_STATE_MIX": "true" if args.parcae_fuse_first_state_mix else "false",
            "FRACTAL_SCOUT_POSITION_ENCODING_KIND": args.position_encoding_kind,
            "FRACTAL_SCOUT_ATTENTION_POSITION_CONTRACT": args.attention_position_contract,
            "FRACTAL_SCOUT_MAX_POSITION_EMBEDDINGS": str(args.max_position_embeddings),
            "FRACTAL_SCOUT_FINAL_NORM_KIND": args.final_norm_kind,
            "FRACTAL_SCOUT_LEARNING_RATE": str(args.learning_rate),
            "FRACTAL_SCOUT_OPTIMIZER_PROFILE": args.optimizer_profile,
            "FRACTAL_SCOUT_MUON_WEIGHT_DECAY": str(args.muon_weight_decay),
            "FRACTAL_SCOUT_MUON_MOMENTUM": str(args.muon_momentum),
            "FRACTAL_SCOUT_MUON_NS_STEPS": str(args.muon_ns_steps),
            "FRACTAL_SCOUT_MUON_ADJUST_LR_FN": args.muon_adjust_lr_fn or "",
            "FRACTAL_SCOUT_PRIMITIVE_RUNTIME_BACKEND": args.primitive_runtime_backend,
            "FRACTAL_SCOUT_HEAD_LOSS_BACKEND": args.head_loss_backend,
            "FRACTAL_SCOUT_FFN_BACKEND": args.ffn_backend,
            "FRACTAL_SCOUT_SEED": str(args.seed),
            "FRACTAL_SCOUT_DATA_SEED": str(args.data_seed),
            "FRACTAL_SCOUT_TOKEN_CACHE_REPO_ID": args.token_cache_repo_id,
            "FRACTAL_SCOUT_TOKEN_CACHE_ARTIFACT": args.token_cache_artifact,
            "FRACTAL_SCOUT_COMPILE_MODE": args.compile_mode or "",
            "FRACTAL_SCOUT_FORCE_DOWNLOAD": "true" if args.force_download else "false",
            "FRACTAL_SCOUT_PROFILE": "true" if args.profile_path1 else "false",
            "FRACTAL_SCOUT_PROFILE_ROW_LIMIT": str(args.profile_row_limit),
            "FRACTAL_SCOUT_CUDA_TIMING": "true" if args.cuda_timing_path1 else "false",
            "FRACTAL_SCOUT_TIMING_STEPS": str(args.timing_steps),
            "FRACTAL_SCOUT_TIMING_WARMUP_TRAIN_STEPS": str(args.warmup_train_steps),
            "FRACTAL_SCOUT_TIMING_WARMUP_EVAL_BATCHES": str(args.warmup_eval_batches),
            "FRACTAL_SCOUT_CUDA_GRAPH_STEP": "true" if args.cuda_graph_step else "false",
            "FRACTAL_SCOUT_NSYS": "true" if args.nsys_path1 else "false",
            "FRACTAL_SCOUT_NSYS_TRACE": args.nsys_trace,
            "FRACTAL_SCOUT_NSYS_STATS": "true" if args.nsys_stats else "false",
            "FRACTAL_SCOUT_INSTALL_FLASH_ATTN": "true" if args.install_flash_attn else "false",
            "FRACTAL_SCOUT_FLASH_ATTN_VERSION": args.flash_attn_version,
            "FRACTAL_P20_TRITON_ATOMIC_TRANSFORM_GRAD": (
                "true" if args.p20_triton_atomic_transform_grad else "false"
            ),
        }
        if token_cache_s3_uri:
            scout_env["FRACTAL_SCOUT_DATA_ROOT"] = "/opt/ml/input/data/token_cache"
            scout_env["FRACTAL_SCOUT_TOKEN_CACHE_DIR"] = args.token_cache_dir or "."
            scout_env["FRACTAL_SCOUT_FORCE_DOWNLOAD"] = "false"
        elif args.token_cache_dir:
            scout_env["FRACTAL_SCOUT_TOKEN_CACHE_DIR"] = args.token_cache_dir
        env.update(scout_env)
    elif args.runner == "smoke":
        env.update(
            {
                "FRACTAL_SMOKE_RUN_LABEL": args.job_name,
                "FRACTAL_SMOKE_LANES": lane_csv,
                "FRACTAL_SMOKE_STEPS": str(args.steps),
                "FRACTAL_SMOKE_EVAL_BATCHES": str(args.eval_batches),
                "FRACTAL_SMOKE_TRAIN_LOSS_RECORD_INTERVAL": str(args.train_loss_record_interval),
                "FRACTAL_SMOKE_SEQ_LEN": str(args.seq_len),
                "FRACTAL_SMOKE_BATCH_SIZE": str(args.batch_size),
                "FRACTAL_SMOKE_DTYPE": args.dtype,
                "FRACTAL_SMOKE_D_MODEL": str(args.d_model),
                "FRACTAL_SMOKE_HEAD_COUNT": str(args.head_count),
                "FRACTAL_SMOKE_TOTAL_LAYERS": str(args.total_layers),
                "FRACTAL_SMOKE_LOCAL_WINDOW": str(args.local_window),
                "FRACTAL_SMOKE_ATTENTION_KERNEL": args.attention_kernel,
                "FRACTAL_SMOKE_FFN_MULTIPLIER": str(args.ffn_multiplier),
                "FRACTAL_SMOKE_PARCAE_LOOP_COUNT": str(args.parcae_loop_count),
                "FRACTAL_SMOKE_PARCAE_HOURGLASS_PASS_COUNT": str(args.parcae_hourglass_pass_count),
                "FRACTAL_SMOKE_PARCAE_HOURGLASS_BAND_SCHEDULE": args.parcae_hourglass_band_schedule or "",
                "FRACTAL_SMOKE_PARCAE_BACKWARD_STEPS": (
                    "" if args.parcae_backward_steps is None else str(args.parcae_backward_steps)
                ),
                "FRACTAL_SMOKE_PARCAE_PRELUDE_NORM_KIND": args.parcae_prelude_norm_kind,
                "FRACTAL_SMOKE_PARCAE_DISCRETIZATION": args.parcae_discretization,
                "FRACTAL_SMOKE_PARCAE_DT_RAW_INIT": str(args.parcae_dt_raw_init),
                "FRACTAL_SMOKE_PARCAE_LOOP_D_MODEL": (
                    "" if args.parcae_loop_d_model is None else str(args.parcae_loop_d_model)
                ),
                "FRACTAL_SMOKE_PARCAE_LOOP_HEAD_COUNT": (
                    "" if args.parcae_loop_head_count is None else str(args.parcae_loop_head_count)
                ),
                "FRACTAL_SMOKE_PARCAE_LOOP_FFN_MULTIPLIER": (
                    "" if args.parcae_loop_ffn_multiplier is None else str(args.parcae_loop_ffn_multiplier)
                ),
                "FRACTAL_SMOKE_PARCAE_LOOP_LAYER_COUNT": (
                    "" if args.parcae_loop_layer_count is None else str(args.parcae_loop_layer_count)
                ),
                "FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_KIND": args.parcae_control_position_kind,
                "FRACTAL_SMOKE_PARCAE_CONTROL_POSITION_SCALE_INIT": str(args.parcae_control_position_scale_init),
                "FRACTAL_SMOKE_PARCAE_CONTROL_STRIDE": str(args.parcae_control_stride),
                "FRACTAL_SMOKE_PARCAE_CONTROL_STATE_TRANSFORM": args.parcae_control_state_transform,
                "FRACTAL_SMOKE_PARCAE_RECURRENT_COMPILE_MODE": args.parcae_recurrent_compile_mode,
                "FRACTAL_SMOKE_PARCAE_LOOP_UPDATE_BACKEND": args.parcae_loop_update_backend,
                "FRACTAL_SMOKE_PARCAE_SCAFFOLD_BACKEND": args.parcae_scaffold_backend,
                "FRACTAL_SMOKE_PARCAE_BAND_BLOCK_CONTRACT": args.parcae_band_block_contract,
                "FRACTAL_SMOKE_PARCAE_BAND_PREPARE_BACKEND": args.parcae_band_prepare_backend,
                "FRACTAL_SMOKE_PARCAE_OUTPUT_MIX_BACKEND": args.parcae_output_mix_backend,
                "FRACTAL_SMOKE_PARCAE_FUSE_FIRST_STATE_MIX": "true" if args.parcae_fuse_first_state_mix else "false",
                "FRACTAL_SMOKE_POSITION_ENCODING_KIND": args.position_encoding_kind,
                "FRACTAL_SMOKE_ATTENTION_POSITION_CONTRACT": args.attention_position_contract,
                "FRACTAL_SMOKE_MAX_POSITION_EMBEDDINGS": str(args.max_position_embeddings),
                "FRACTAL_SMOKE_FINAL_NORM_KIND": args.final_norm_kind,
                "FRACTAL_SMOKE_LEARNING_RATE": str(args.learning_rate),
                "FRACTAL_SMOKE_OPTIMIZER_PROFILE": args.optimizer_profile,
                "FRACTAL_SMOKE_MUON_WEIGHT_DECAY": str(args.muon_weight_decay),
                "FRACTAL_SMOKE_MUON_MOMENTUM": str(args.muon_momentum),
                "FRACTAL_SMOKE_MUON_NS_STEPS": str(args.muon_ns_steps),
                "FRACTAL_SMOKE_MUON_ADJUST_LR_FN": args.muon_adjust_lr_fn or "",
                "FRACTAL_SMOKE_PRIMITIVE_RUNTIME_BACKEND": args.primitive_runtime_backend,
                "FRACTAL_SMOKE_HEAD_LOSS_BACKEND": args.head_loss_backend,
                "FRACTAL_SMOKE_FFN_BACKEND": args.ffn_backend,
                "FRACTAL_SMOKE_SEED": str(args.seed),
                "FRACTAL_SMOKE_DATA_SEED": str(args.data_seed),
            }
        )
    input_data_config = [
        {
            "ChannelName": "source",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": source_s3_prefix,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "InputMode": "File",
            "CompressionType": "None",
            "ContentType": "application/x-tar",
        }
    ]
    if args.runner == "token-cache" and args.token_cache_s3_uri:
        input_data_config.append(
            {
                "ChannelName": "token_cache",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": args.token_cache_s3_uri.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": args.token_cache_input_mode,
                "CompressionType": "None",
                "ContentType": "application/x-fractal-token-cache",
            }
        )
    if args.runner == "token-cache" and hf_env_s3_prefix:
        input_data_config.append(
            {
                "ChannelName": HF_ENV_CHANNEL_NAME,
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": hf_env_s3_prefix.rstrip("/") + "/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "InputMode": "File",
                "CompressionType": "None",
                "ContentType": "text/plain",
            }
        )

    return {
        "TrainingJobName": args.job_name,
        "AlgorithmSpecification": {
            "TrainingImage": training_image,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["bash", "-lc"],
            "ContainerArguments": [extract_and_run],
        },
        "RoleArn": role_arn,
        "InputDataConfig": input_data_config,
        "OutputDataConfig": {
            "S3OutputPath": output_s3_path,
        },
        "ResourceConfig": {
            "InstanceType": args.instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": args.volume_size_gb,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": args.max_runtime_seconds,
        },
        "Environment": env,
        "EnableNetworkIsolation": False,
        "EnableManagedSpotTraining": False,
    }


def _wait_for_job(args: argparse.Namespace) -> dict[str, Any]:
    terminal = {"Completed", "Failed", "Stopped"}
    while True:
        raw = _run_aws(
            args,
            [
                "sagemaker",
                "describe-training-job",
                "--training-job-name",
                args.job_name,
                "--output",
                "json",
            ],
            capture=True,
        )
        payload = json.loads(raw)
        status = payload.get("TrainingJobStatus")
        secondary = payload.get("SecondaryStatus")
        print(f"status={status} secondary={secondary}", flush=True)
        if status in terminal:
            return payload
        time.sleep(args.poll_seconds)


def _model_subdir(args: argparse.Namespace) -> str:
    if args.runner == "token-cache":
        return "path1-cuda-scout"
    if args.runner == "mamba-wheelhouse":
        return "mamba-wheelhouse"
    return "path1-cuda-smoke"


def _try_download_model_artifact(args: argparse.Namespace, *, output_s3_path: str) -> None:
    if not args.download_output:
        return
    local_dir = REPO_ROOT / "experiments" / "aws_sagemaker" / _model_subdir(args).replace("-", "_") / args.job_name
    local_dir.mkdir(parents=True, exist_ok=True)
    artifact_s3 = f"{output_s3_path.rstrip('/')}/{args.job_name}/output/model.tar.gz"
    artifact_path = local_dir / "model.tar.gz"
    try:
        _run_aws(args, ["s3", "cp", artifact_s3, str(artifact_path)])
    except subprocess.CalledProcessError:
        print(f"warning: failed to download SageMaker model artifact from {artifact_s3}", flush=True)
        return
    extract_dir = local_dir / "extracted"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir()
    with tarfile.open(artifact_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")
    summary = extract_dir / _model_subdir(args) / "summary.md"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"), flush=True)
    print(f"local_output_dir={local_dir}", flush=True)


def _redact_request_for_display(request: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(request))
    env = redacted.get("Environment")
    if isinstance(env, dict):
        for key in SENSITIVE_ENV_KEYS:
            if key in env:
                env[key] = "<redacted>"
    return redacted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a tiny Path 1 CUDA smoke to SageMaker Training on a small GPU instance."
    )
    parser.add_argument(
        "--runner",
        choices=["smoke", "token-cache", "mamba-wheelhouse"],
        default="smoke",
        help=(
            "smoke uses the built-in JSONL corpus; token-cache hydrates/mounts a token cache and runs "
            "the matched scout; mamba-wheelhouse builds reusable official Mamba CUDA wheels."
        ),
    )
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--bucket", help="Writable S3 bucket. May also be set via FRACTAL_SAGEMAKER_BUCKET.")
    parser.add_argument("--prefix")
    parser.add_argument("--role-arn", help="SageMaker execution role ARN. May also be set via FRACTAL_SAGEMAKER_ROLE_ARN.")
    parser.add_argument("--training-image", help="Override the PyTorch SageMaker DLC image URI.")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--volume-size-gb", type=int, default=50)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--job-name")
    parser.add_argument(
        "--lanes",
        default="attention-only,parcae-p20-control-looped-attention",
        help="Comma-separated lanes: attention-only, parcae-looped-attention, parcae-bx-looped-attention, parcae-p20-control-looped-attention.",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--train-loss-record-interval", type=int, default=1)
    parser.add_argument("--warmup-train-steps", type=int, default=0)
    parser.add_argument("--warmup-eval-batches", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--total-layers", type=int, default=4)
    parser.add_argument("--local-window", type=int, default=64)
    parser.add_argument("--attention-kernel", choices=["sdpa", "flex-local", "flash-local"], default="sdpa")
    parser.add_argument("--ffn-multiplier", type=int, default=2)
    parser.add_argument(
        "--allow-slow-attention-head-dim",
        action="store_true",
        help=(
            "Allow CUDA scout shapes whose attention head_dim is not a multiple of 8. "
            "By default these are rejected because they can force very slow SDPA kernels."
        ),
    )
    parser.add_argument("--parcae-loop-count", type=int, default=1)
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
    parser.add_argument("--parcae-fuse-first-state-mix", action="store_true")
    parser.add_argument("--position-encoding-kind", choices=["none", "learned"], default="none")
    parser.add_argument("--attention-position-contract", choices=["shared-input", "attention-only"], default="shared-input")
    parser.add_argument("--max-position-embeddings", type=int, default=1024)
    parser.add_argument("--final-norm-kind", choices=["identity", "layernorm", "rmsnorm"], default="identity")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--optimizer-profile", choices=["adam", "adam-fused", "adam-triton-2d", "muon-reference"], default="adam")
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"])
    parser.add_argument("--primitive-runtime-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--head-loss-backend", choices=["dense", "compiled", "streaming-kernel"], default="dense")
    parser.add_argument("--ffn-backend", choices=["dense", "compiled", "manual-autograd", "triton-gelu", "recompute"], default="dense")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--profile-path1", action="store_true", help="Run the token-cache Path 1 profiler instead of training.")
    parser.add_argument("--profile-row-limit", type=int, default=40)
    parser.add_argument(
        "--cuda-timing-path1",
        action="store_true",
        help="Run multi-step CUDA-event Path 1 timings instead of training.",
    )
    parser.add_argument("--timing-steps", type=int, default=20)
    parser.add_argument(
        "--cuda-graph-step",
        action="store_true",
        help="When --cuda-timing-path1 is active, capture/replay a static CUDA train step.",
    )
    parser.add_argument(
        "--nsys-path1",
        action="store_true",
        help="Run Nsight Systems around token-cache CUDA timing instead of training.",
    )
    parser.add_argument("--nsys-trace", default="cuda,nvtx,cublas,cudnn")
    parser.add_argument("--no-nsys-stats", dest="nsys_stats", action="store_false")
    parser.set_defaults(nsys_stats=True)
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Install flash-attn inside the SageMaker scout job for attention_kernel=flash-local probes.",
    )
    parser.add_argument("--flash-attn-version", default="2.8.3")
    parser.add_argument(
        "--p20-triton-atomic-transform-grad",
        action="store_true",
        help=(
            "Enable the experimental P20/RGRP Triton scan backward path that atomically accumulates "
            "state-transform gradients instead of materializing per-batch gradient tensors."
        ),
    )
    parser.add_argument("--token-cache-repo-id", default="joebud/fractal-fineweb-openllama-tokens")
    parser.add_argument("--token-cache-artifact", default=DEFAULT_TOKEN_CACHE_ARTIFACT)
    parser.add_argument(
        "--token-cache-s3-uri",
        help=(
            "S3 prefix containing an extracted token cache. When set with --runner token-cache, "
            "the job mounts it as the token_cache channel instead of downloading from Hugging Face."
        ),
    )
    parser.add_argument(
        "--token-cache-input-mode",
        choices=["File", "FastFile"],
        default="FastFile",
        help="SageMaker input mode for --token-cache-s3-uri.",
    )
    parser.add_argument(
        "--token-cache-dir",
        help=(
            "Token-cache directory under the data root. Defaults to '.' for --token-cache-s3-uri "
            "and to the artifact basename for Hugging Face hydration."
        ),
    )
    parser.add_argument(
        "--hf-token-env-file",
        default=".env",
        help=(
            "Local-only env file used to stage HF_TOKEN into the private SageMaker hf_env input channel. "
            "The token is never placed in the SageMaker Environment map."
        ),
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--mamba-wheelhouse-torch-cuda-arch-list", default="8.9;9.0")
    parser.add_argument("--mamba-wheelhouse-max-jobs", type=int, default=4)
    parser.add_argument("--mamba-wheelhouse-kernels-version", default="0.13.0")
    parser.add_argument("--mamba-wheelhouse-causal-conv1d-version", default="1.6.1")
    parser.add_argument("--mamba-wheelhouse-mamba-ssm-version", default="2.3.1")
    parser.add_argument("--mamba-wheelhouse-transformers-version", default="5.7.0")
    parser.add_argument("--mamba-wheelhouse-sentencepiece-version", default="0.2.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--download-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.runner != "mamba-wheelhouse":
        _lane_list(args.lanes)
        _validate_cuda_attention_shape_contract(args)
    if not args.job_name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if args.runner == "token-cache":
            label = "scout"
        elif args.runner == "mamba-wheelhouse":
            label = "mamba-wheelhouse"
        else:
            label = "smoke"
        args.job_name = f"fractal-path1-cuda-{label}-{stamp}"

    bucket = _env_or_arg(args.bucket, "FRACTAL_SAGEMAKER_BUCKET")
    if not bucket:
        raise SystemExit(
            "Writable S3 bucket is required. Set FRACTAL_SAGEMAKER_BUCKET "
            "or pass --bucket <bucket-name>."
        )

    prefix = args.prefix or (
        "fractal/path1-cuda-scout"
        if args.runner == "token-cache"
        else "fractal/mamba-wheelhouse"
        if args.runner == "mamba-wheelhouse"
        else "fractal/path1-cuda-smoke"
    )
    source_s3_prefix = f"s3://{bucket}/{prefix.strip('/')}/{args.job_name}/source/"
    source_s3_uri = f"{source_s3_prefix}source.tar.gz"
    output_s3_path = f"s3://{bucket}/{prefix.strip('/')}/{args.job_name}/output"

    with tempfile.TemporaryDirectory(prefix="fractal-sagemaker-bundle-") as tmp:
        bundle_path = Path(tmp) / "source.tar.gz"
        _stage_source_bundle(REPO_ROOT, bundle_path, runner=args.runner)
        hf_env_s3_prefix = None
        hf_env_path = None
        if args.runner == "token-cache" and not args.token_cache_s3_uri:
            env_file = Path(args.hf_token_env_file).expanduser()
            if not env_file.is_absolute():
                env_file = REPO_ROOT / env_file
            hf_token = _load_local_hf_token(env_file)
            if not hf_token:
                raise SystemExit(
                    "HF_TOKEN is required for --runner token-cache without --token-cache-s3-uri. "
                    f"Put HF_TOKEN in {env_file}, export HF_TOKEN, or run `hf auth login` locally."
                )
            hf_env_s3_prefix = f"s3://{bucket}/{prefix.strip('/')}/{args.job_name}/{HF_ENV_CHANNEL_NAME}/"
            hf_env_path = Path(tmp) / HF_ENV_FILENAME
            _write_hf_env_file(hf_env_path, token=hf_token)
        request = _training_request(
            args,
            source_s3_prefix=source_s3_prefix,
            output_s3_path=output_s3_path,
            hf_env_s3_prefix=hf_env_s3_prefix,
        )
        request_path = Path(tmp) / "create-training-job.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if args.dry_run:
            print(f"source_bundle={bundle_path}", flush=True)
            print(json.dumps(_redact_request_for_display(request), indent=2, sort_keys=True), flush=True)
            return 0

        _run_aws(args, ["s3", "cp", str(bundle_path), source_s3_uri])
        if hf_env_path is not None and hf_env_s3_prefix is not None:
            _run_aws(args, ["s3", "cp", str(hf_env_path), _s3_join(hf_env_s3_prefix, HF_ENV_FILENAME)])
        _run_aws(args, ["sagemaker", "create-training-job", "--cli-input-json", f"file://{request_path}"])

    print(f"submitted_training_job={args.job_name}", flush=True)
    print(f"output_s3_path={output_s3_path}", flush=True)
    if args.no_wait:
        return 0

    payload = _wait_for_job(args)
    status = payload.get("TrainingJobStatus")
    if status != "Completed":
        failure = payload.get("FailureReason")
        if failure:
            print(f"failure_reason={failure}", flush=True)
        return 1
    _try_download_model_artifact(args, output_s3_path=output_s3_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
