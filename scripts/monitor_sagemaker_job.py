#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_STATUSES = {"Completed", "Failed", "Stopped"}
SENSITIVE_ENV_KEYS = {"HF_TOKEN"}


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def _aws(profile: str, region: str, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["aws", *args, "--region", region, "--profile", profile], check=check)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _redact_sensitive_environment(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    env = redacted.get("Environment")
    if isinstance(env, dict):
        for key in SENSITIVE_ENV_KEYS:
            if key in env:
                env[key] = "<redacted>"
    return redacted


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now_iso()}] {message}\n")


def _extract_artifact(*, artifact_path: Path, extract_dir: Path) -> None:
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(artifact_path, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")


def _cleanup_cloud_job(*, profile: str, region: str, job_name: str, log_path: Path) -> str:
    result = _aws(
        profile,
        region,
        ["sagemaker", "delete-training-job", "--training-job-name", job_name],
        check=False,
    )
    if result.returncode == 0:
        _append_log(log_path, f"deleted SageMaker training job metadata for {job_name}")
        return "deleted"
    _append_log(
        log_path,
        "warning: delete-training-job failed for "
        f"{job_name}: {(result.stderr or result.stdout).strip()}",
    )
    return "delete-failed"


def _remove_launch_agent(*, plist_path: Path, label: str, log_path: Path) -> None:
    uid = os.getuid()
    try:
        plist_path.unlink()
        _append_log(log_path, f"removed launch agent plist {plist_path}")
    except FileNotFoundError:
        pass
    _run(["launchctl", "bootout", f"gui/{uid}", str(plist_path)], check=False)
    _run(["launchctl", "remove", label], check=False)


def _record_terminal_job(
    *,
    profile: str,
    region: str,
    job_name: str,
    describe_payload: dict[str, Any],
    local_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    status = str(describe_payload.get("TrainingJobStatus"))
    local_dir.mkdir(parents=True, exist_ok=True)
    sagemaker_output_dir = local_dir / "sagemaker-output"
    _write_json(
        sagemaker_output_dir / "describe-training-job.json",
        _redact_sensitive_environment(describe_payload),
    )

    record: dict[str, Any] = {
        "job_name": job_name,
        "status": status,
        "secondary_status": describe_payload.get("SecondaryStatus"),
        "recorded_at": _now_iso(),
        "local_dir": str(local_dir),
        "describe_path": str(sagemaker_output_dir / "describe-training-job.json"),
        "failure_reason": describe_payload.get("FailureReason"),
    }

    if status == "Completed":
        artifact_s3 = (
            describe_payload.get("ModelArtifacts", {}) or {}
        ).get("S3ModelArtifacts")
        if not artifact_s3:
            record["artifact_status"] = "missing-model-artifact"
            _write_json(local_dir / "monitor-status.json", record)
            _append_log(log_path, f"{job_name} completed but ModelArtifacts.S3ModelArtifacts is missing")
            return record
        artifact_path = local_dir / "model.tar.gz"
        download = _aws(profile, region, ["s3", "cp", artifact_s3, str(artifact_path)], check=False)
        if download.returncode != 0:
            record["artifact_status"] = "download-failed"
            record["artifact_s3"] = artifact_s3
            record["artifact_error"] = (download.stderr or download.stdout).strip()
            _write_json(local_dir / "monitor-status.json", record)
            _append_log(log_path, f"{job_name} artifact download failed: {record['artifact_error']}")
            return record
        extract_dir = local_dir / "extracted"
        _extract_artifact(artifact_path=artifact_path, extract_dir=extract_dir)
        record["artifact_status"] = "downloaded-and-extracted"
        record["artifact_s3"] = artifact_s3
        record["artifact_path"] = str(artifact_path)
        record["extract_dir"] = str(extract_dir)
        summary_path = extract_dir / "path1-cuda-scout" / "summary.json"
        if summary_path.exists():
            record["summary_json"] = str(summary_path)
    else:
        record["artifact_status"] = "not-applicable"

    record["cloud_cleanup"] = _cleanup_cloud_job(
        profile=profile,
        region=region,
        job_name=job_name,
        log_path=log_path,
    )
    _write_json(local_dir / "monitor-status.json", record)
    _append_log(log_path, f"recorded terminal job {job_name}: {status}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor one SageMaker training job and self-remove when terminal.")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--plist-path", required=True)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE", "codex-eml"))
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--local-root", default=str(REPO_ROOT / "experiments" / "aws_sagemaker" / "path1_cuda_scout"))
    parser.add_argument("--log-dir", default=str(REPO_ROOT / "experiments" / "aws_sagemaker" / "monitors"))
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_path = log_dir / f"{args.job_name}.log"
    local_dir = Path(args.local_root) / args.job_name
    plist_path = Path(args.plist_path)

    describe = _aws(
        args.profile,
        args.region,
        ["sagemaker", "describe-training-job", "--training-job-name", args.job_name, "--output", "json"],
        check=False,
    )
    if describe.returncode != 0:
        _append_log(log_path, f"describe-training-job failed: {(describe.stderr or describe.stdout).strip()}")
        return 1
    payload = json.loads(describe.stdout)
    status = str(payload.get("TrainingJobStatus"))
    secondary = payload.get("SecondaryStatus")
    _append_log(log_path, f"status={status} secondary={secondary}")

    if status not in TERMINAL_STATUSES:
        return 0

    record = _record_terminal_job(
        profile=args.profile,
        region=args.region,
        job_name=args.job_name,
        describe_payload=payload,
        local_dir=local_dir,
        log_path=log_path,
    )
    if record.get("artifact_status") in {"missing-model-artifact", "download-failed"}:
        _append_log(log_path, "terminal job not fully marked; keeping launch agent active for retry")
        return 1

    _remove_launch_agent(plist_path=plist_path, label=args.label, log_path=log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
