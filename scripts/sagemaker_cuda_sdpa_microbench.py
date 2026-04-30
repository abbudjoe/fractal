#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
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


ENTRYPOINT = r'''#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path("/opt/ml/model/sdpa-microbench")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("FRACTAL_SDPA_INSTALL_FLASH_ATTN") == "1":
        version = os.environ.get("FRACTAL_SDPA_FLASH_ATTN_VERSION", "2.8.3")
        pip_cache_dir = ROOT / ".pip-cache"
        pip_tmp_dir = ROOT / ".pip-tmp"
        pip_cache_dir.mkdir(parents=True, exist_ok=True)
        pip_tmp_dir.mkdir(parents=True, exist_ok=True)
        install = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            f"flash-attn=={version}",
        ]
        print("+ " + " ".join(shlex.quote(part) for part in install), flush=True)
        install_env = os.environ.copy()
        install_env["PIP_CACHE_DIR"] = str(pip_cache_dir)
        install_env["TMPDIR"] = str(pip_tmp_dir)
        subprocess.run(install, cwd=pip_cache_dir, env=install_env, check=True)
    extra_args = shlex.split(os.environ.get("FRACTAL_SDPA_ARGS", ""))
    command = [
        sys.executable,
        str(ROOT / "scripts" / "v3a_cuda_sdpa_microbench.py"),
        "--device",
        "cuda",
        "--output-dir",
        str(OUT_ROOT),
        *extra_args,
    ]
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    summary = OUT_ROOT / "summary.md"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _default_training_image(region: str) -> str:
    return (
        f"{DEFAULT_DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/"
        f"pytorch-training:{DEFAULT_DLC_TAG}"
    )


def _env_or_arg(value: str | None, env_name: str) -> str | None:
    return value or os.environ.get(env_name)


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


def _microbench_cli_args(args: argparse.Namespace) -> list[str]:
    cli = [
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.seq_len),
        "--local-window",
        str(args.local_window),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--mode",
        args.mode,
        "--shape-specs",
        args.shape_specs,
        "--mask-modes",
        args.mask_modes,
        "--backends",
        args.backends,
        "--profile-row-limit",
        str(args.profile_row_limit),
    ]
    return cli


def _stage_source_bundle(repo_root: Path, bundle_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fractal-sdpa-src-") as tmp:
        stage = Path(tmp) / "source"
        stage.mkdir()
        (stage / "scripts").mkdir()
        shutil.copy2(
            repo_root / "scripts" / "v3a_cuda_sdpa_microbench.py",
            stage / "scripts" / "v3a_cuda_sdpa_microbench.py",
        )
        entrypoint = stage / "sagemaker_sdpa_entrypoint.py"
        entrypoint.write_text(ENTRYPOINT, encoding="utf-8")
        entrypoint.chmod(0o755)
        with tarfile.open(bundle_path, "w:gz") as tar:
            for path in sorted(stage.rglob("*")):
                tar.add(path, arcname=path.relative_to(stage))


def _training_request(args: argparse.Namespace, *, source_s3_prefix: str, output_s3_path: str) -> dict[str, Any]:
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
        "python sagemaker_sdpa_entrypoint.py"
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
        "InputDataConfig": [
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
        ],
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
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "FRACTAL_SDPA_ARGS": shlex.join(_microbench_cli_args(args)),
            "FRACTAL_SDPA_INSTALL_FLASH_ATTN": "1" if args.install_flash_attn else "0",
            "FRACTAL_SDPA_FLASH_ATTN_VERSION": args.flash_attn_version,
        },
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


def _try_download_model_artifact(args: argparse.Namespace, *, output_s3_path: str) -> None:
    if not args.download_output:
        return
    local_dir = REPO_ROOT / "experiments" / "aws_sagemaker" / "cuda_sdpa_microbench" / args.job_name
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
    summary = extract_dir / "sdpa-microbench" / "summary.md"
    if summary.exists():
        print(summary.read_text(encoding="utf-8"), flush=True)
    print(f"local_output_dir={local_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a focused CUDA SDPA/local-attention microbench to SageMaker."
    )
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", DEFAULT_REGION))
    parser.add_argument("--bucket", help="Writable S3 bucket. May also be set via FRACTAL_SAGEMAKER_BUCKET.")
    parser.add_argument("--prefix", default="fractal/cuda-sdpa-microbench")
    parser.add_argument("--role-arn", help="SageMaker execution role ARN. May also be set via FRACTAL_SAGEMAKER_ROLE_ARN.")
    parser.add_argument("--training-image", help="Override the PyTorch SageMaker DLC image URI.")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--volume-size-gb", type=int, default=30)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--job-name")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--local-window", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["forward", "forward-backward"], default="forward-backward")
    parser.add_argument(
        "--shape-specs",
        default=(
            "attn_d448_h7:448:7,"
            "rgrp_d448_h8:448:8,"
            "attn_d456_h19:456:19,"
            "rgrp_d480_h10:480:10,"
            "rgrp_d512_h8:512:8,"
            "slow_d472_h8:472:8"
        ),
    )
    parser.add_argument("--mask-modes", default="causal,local-additive,local-bool")
    parser.add_argument("--backends", default="auto,flash,efficient,math,flex-local,flash-local")
    parser.add_argument("--profile-row-limit", type=int, default=20)
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="Install flash-attn inside the SageMaker job before running flash-local benchmarks.",
    )
    parser.add_argument("--flash-attn-version", default="2.8.3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--download-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.job_name:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.job_name = f"fractal-cuda-sdpa-microbench-{stamp}"

    bucket = _env_or_arg(args.bucket, "FRACTAL_SAGEMAKER_BUCKET")
    if not bucket:
        raise SystemExit(
            "Writable S3 bucket is required. Set FRACTAL_SAGEMAKER_BUCKET "
            "or pass --bucket <bucket-name>."
        )

    source_s3_prefix = f"s3://{bucket}/{args.prefix.strip('/')}/{args.job_name}/source/"
    source_s3_uri = f"{source_s3_prefix}source.tar.gz"
    output_s3_path = f"s3://{bucket}/{args.prefix.strip('/')}/{args.job_name}/output"

    with tempfile.TemporaryDirectory(prefix="fractal-sdpa-sagemaker-") as tmp:
        bundle_path = Path(tmp) / "source.tar.gz"
        _stage_source_bundle(REPO_ROOT, bundle_path)
        request = _training_request(args, source_s3_prefix=source_s3_prefix, output_s3_path=output_s3_path)
        request_path = Path(tmp) / "create-training-job.json"
        request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if args.dry_run:
            print(f"source_bundle={bundle_path}", flush=True)
            print(json.dumps(request, indent=2, sort_keys=True), flush=True)
            return 0

        _run_aws(args, ["s3", "cp", str(bundle_path), source_s3_uri])
        _run_aws(args, ["sagemaker", "create-training-job", "--cli-input-json", f"file://{request_path}"])
        print(f"job_name={args.job_name}", flush=True)
        print(f"output_s3_path={output_s3_path}", flush=True)
        if args.no_wait:
            return 0
        payload = _wait_for_job(args)
        _try_download_model_artifact(args, output_s3_path=output_s3_path)
        status = payload.get("TrainingJobStatus")
        if status != "Completed":
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
