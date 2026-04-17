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
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--head-count", type=int)
    parser.add_argument("--total-layers", type=int)
    parser.add_argument("--local-window", type=int)
    parser.add_argument("--ffn-multiplier", type=int)
    parser.add_argument("--parcae-loop-count", type=int, default=2)
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


def _lane_args(lane: str, *, loop_count: int) -> list[str]:
    if lane == "attention-only":
        return ["--variant", "attention-only"]
    if lane in {
        "parcae-looped-attention",
        "parcae-bx-looped-attention",
        "parcae-p20-control-looped-attention",
    }:
        return [
            "--variant",
            "attention-only",
            "--scaffold-profile",
            lane,
            "--parcae-loop-count",
            str(loop_count),
        ]
    raise SystemExit(f"unsupported lane: {lane}")


def run_lane(args: argparse.Namespace, *, lane: str, manifest_path: Path, output_dir: Path, ledger_path: Path) -> dict[str, Any]:
    lane_output_dir = output_dir / lane
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
        "primitive-triton" if args.primitive_runtime_backend == "triton" else "requirements-only",
        "--primitive-runtime-backend",
        args.primitive_runtime_backend,
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
        "--warmup-train-steps",
        "0",
        "--warmup-eval-batches",
        "0",
        "--learning-rate",
        str(args.learning_rate),
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
    ]
    if args.compile_mode is not None:
        command.extend(["--compile-mode", args.compile_mode])
    for cli_name, value in (
        ("--d-model", args.d_model),
        ("--head-count", args.head_count),
        ("--total-layers", args.total_layers),
        ("--local-window", args.local_window),
        ("--ffn-multiplier", args.ffn_multiplier),
    ):
        if value is not None:
            command.extend([cli_name, str(value)])
    command.extend(_lane_args(lane, loop_count=args.parcae_loop_count))
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
        "seed": args.seed,
        "data_seed": args.data_seed,
        "d_model": args.d_model,
        "head_count": args.head_count,
        "total_layers": args.total_layers,
        "local_window": args.local_window,
        "ffn_multiplier": args.ffn_multiplier,
        "dtype": args.dtype,
        "primitive_runtime_backend": args.primitive_runtime_backend,
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
        f"- dtype: `{args.dtype}`",
        f"- primitive_runtime_backend: `{args.primitive_runtime_backend}`",
        "",
        "| Lane | Params | Initial Loss | Final Loss | tok/s | Peak CUDA MB | CUDA Device |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        params = row["parameters"]
        cuda_device = row.get("cuda_device") or ""
        cuda_capability = row.get("cuda_compute_capability")
        if cuda_capability:
            cuda_device = f"{cuda_device} (cc {cuda_capability})" if cuda_device else f"cc {cuda_capability}"
        lines.append(
            f"| {row['lane']} | {params if params is not None else ''} | "
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
