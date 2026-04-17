#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPO_ROOT = Path("/workspace/fractal")
REMOTE_ARTIFACT_ROOT = Path("/workspace/artifacts")
REMOTE_DATA_ROOT = Path("/workspace/data")
APP_NAME = "fractal-v3a-path1-parcae-h100"
DATA_VOLUME_NAME = "fractal-fineweb-openllama-token-cache"
DEFAULT_LANES = (
    "attention-only,"
    "parcae-looped-attention,"
    "parcae-bx-looped-attention,"
    "parcae-p20-control-looped-attention"
)
DEFAULT_TOKEN_CACHE_ARTIFACT = "fineweb-cc-main-2024-10-openllama-tokens-750m-v1.tar.zst"


def _load_local_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text(encoding="utf-8").strip()
    return None


def _hf_secret() -> modal.Secret:
    token = _load_local_hf_token()
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("fractal-hf-token", required_keys=["HF_TOKEN"])


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("bash", "ca-certificates", "zstd")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("huggingface_hub>=0.24.0", "numpy>=2.0.0", "pyarrow>=23.0.0", "sentencepiece>=0.2.0")
    .add_local_dir(REPO_ROOT / "python", str(REMOTE_REPO_ROOT / "python"), copy=False)
    .add_local_dir(REPO_ROOT / "scripts", str(REMOTE_REPO_ROOT / "scripts"), copy=False)
)


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _collect_text_artifacts(output_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    allowed_names = {"summary.md", "summary.json", "ledger.jsonl", "report.json"}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name not in allowed_names:
            continue
        files[str(path.relative_to(output_dir))] = path.read_text(encoding="utf-8")
    return files


@app.function(
    image=image,
    secrets=[_hf_secret()],
    volumes={str(REMOTE_DATA_ROOT): data_volume},
    gpu="H100!",
    timeout=21_600,
    memory=32_768,
)
def run_promotion(
    *,
    run_label: str,
    seed: int,
    data_seed: int,
    seq_len: int,
    batch_size: int,
    steps: int,
    eval_batches: int,
    learning_rate: float,
    d_model: int,
    head_count: int,
    total_layers: int,
    local_window: int,
    ffn_multiplier: int,
    lanes: str,
    dtype: str,
    primitive_runtime_backend: str,
    parcae_loop_count: int,
    token_cache_artifact: str,
    compile_mode: str | None,
    force_download: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["FRACTAL_RUN_ARTIFACT_DIR"] = str(REMOTE_ARTIFACT_ROOT)
    env["FRACTAL_RUN_ID"] = run_label

    device_probe = (
        "import json, torch; "
        "p=torch.cuda.get_device_properties(0); "
        "print(json.dumps({"
        "'device_name': p.name, "
        "'compute_capability': f'{p.major}.{p.minor}', "
        "'total_memory': p.total_memory, "
        "'torch': torch.__version__, "
        "'cuda': torch.version.cuda"
        "}), flush=True)"
    )
    _run([sys.executable, "-c", device_probe], cwd=REMOTE_REPO_ROOT, env=env)

    output_dir = REMOTE_ARTIFACT_ROOT / "v3a-python-path1-parcae-h100-modal" / run_label
    command = [
        sys.executable,
        str(REMOTE_REPO_ROOT / "scripts" / "v3a_python_path1_parcae_h100_promotion.py"),
        "--cuda-device",
        "0",
        "--dtype",
        dtype,
        "--seed",
        str(seed),
        "--data-seed",
        str(data_seed),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--steps",
        str(steps),
        "--eval-batches",
        str(eval_batches),
        "--learning-rate",
        str(learning_rate),
        "--primitive-runtime-backend",
        primitive_runtime_backend,
        "--parcae-loop-count",
        str(parcae_loop_count),
        "--token-cache-repo-id",
        "joebud/fractal-fineweb-openllama-tokens",
        "--token-cache-artifact",
        token_cache_artifact,
        "--lanes",
        lanes,
        "--output-dir",
        str(output_dir),
        "--run-label",
        run_label,
    ]
    if compile_mode:
        command.extend(["--compile-mode", compile_mode])
    for cli_name, value in (
        ("--d-model", d_model),
        ("--head-count", head_count),
        ("--total-layers", total_layers),
        ("--local-window", local_window),
        ("--ffn-multiplier", ffn_multiplier),
    ):
        if value > 0:
            command.extend([cli_name, str(value)])
    if force_download:
        command.append("--force-download")
    _run(command, cwd=REMOTE_REPO_ROOT, env=env)
    data_volume.commit()

    summary_path = output_dir / "summary.md"
    summary_json_path = output_dir / "summary.json"
    return {
        "output_dir": str(output_dir),
        "summary_md": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        "summary_json": json.loads(summary_json_path.read_text(encoding="utf-8")) if summary_json_path.exists() else {},
        "files": _collect_text_artifacts(output_dir),
    }


@app.local_entrypoint()
def main(
    seed: int = 42,
    data_seed: int = 42,
    seq_len: int = 256,
    batch_size: int = 64,
    steps: int = 2000,
    eval_batches: int = 64,
    learning_rate: float = 1.0e-3,
    d_model: int = 0,
    head_count: int = 0,
    total_layers: int = 0,
    local_window: int = 0,
    ffn_multiplier: int = 0,
    lanes: str = DEFAULT_LANES,
    dtype: str = "bf16",
    primitive_runtime_backend: str = "triton",
    parcae_loop_count: int = 2,
    token_cache_artifact: str = DEFAULT_TOKEN_CACHE_ARTIFACT,
    compile_mode: str = "",
    run_label: str = "",
    force_download: bool = False,
) -> None:
    if not run_label:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_label = f"v3a-path1-parcae-h100-modal-s{seed}-steps{steps}-{stamp}"

    result = run_promotion.remote(
        run_label=run_label,
        seed=seed,
        data_seed=data_seed,
        seq_len=seq_len,
        batch_size=batch_size,
        steps=steps,
        eval_batches=eval_batches,
        learning_rate=learning_rate,
        d_model=d_model,
        head_count=head_count,
        total_layers=total_layers,
        local_window=local_window,
        ffn_multiplier=ffn_multiplier,
        lanes=lanes,
        dtype=dtype,
        primitive_runtime_backend=primitive_runtime_backend,
        parcae_loop_count=parcae_loop_count,
        token_cache_artifact=token_cache_artifact,
        compile_mode=compile_mode or None,
        force_download=force_download,
    )

    local_dir = REPO_ROOT / ".modal-local-logs" / "modal-results" / run_label
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for rel_path, text in result.get("files", {}).items():
        out_path = local_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

    print(f"local_result_dir={local_dir}", flush=True)
    if result.get("summary_md"):
        print(result["summary_md"], flush=True)
