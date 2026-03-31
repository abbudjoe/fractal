#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/runpod-audit.sh <artifacts|pods> [options]

Audit preserved RunPod results under .runpod-local-logs/runpod-results.

Commands:
  artifacts            Check local artifact preservation health only.
  pods                 Check preservation health plus live pod state.

Options:
  --results-root PATH  Override runpod-results root. Default: <repo>/.runpod-local-logs/runpod-results
  --runpodctl PATH     Override runpodctl binary. Default: runpodctl
  --stale-hours N      Mark running pods stale after N hours without a preserved final state. Default: 2
  --help               Show this help.

Examples:
  scripts/runpod-audit.sh artifacts
  scripts/runpod-audit.sh pods --stale-hours 4
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"
RUNPODCTL_BIN="${RUNPODCTL_BIN:-runpodctl}"
STALE_HOURS="2"
MODE=""

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [ $# -gt 0 ]; do
    case "$1" in
        artifacts|pods)
            MODE="$1"
            shift
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --runpodctl)
            RUNPODCTL_BIN="$2"
            shift 2
            ;;
        --stale-hours)
            STALE_HOURS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [ -z "$MODE" ]; then
    MODE="artifacts"
fi

require_cmd python3

if [ "$MODE" = "pods" ]; then
    require_cmd "$RUNPODCTL_BIN"
fi

python3 - "$MODE" "$RESULTS_ROOT" "$RUNPODCTL_BIN" "$STALE_HOURS" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


mode = sys.argv[1]
results_root = Path(sys.argv[2])
runpodctl = sys.argv[3]
stale_hours = int(sys.argv[4])


@dataclass
class RunRecord:
    run_dir: Path
    run_id: str
    wrapper: dict[str, Any] | None
    wrapper_error: str | None
    live_pod: dict[str, Any] | None = None

    @property
    def pod_id(self) -> str:
        if self.wrapper:
            return str(self.wrapper.get("pod", {}).get("id") or "")
        return ""

    @property
    def pod_name(self) -> str:
        if self.wrapper:
            return str(self.wrapper.get("pod", {}).get("name") or "")
        return ""


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, "missing"
    except Exception as exc:  # noqa: BLE001
        return None, f"invalid-json: {exc}"


def run_dirs() -> list[RunRecord]:
    records: list[RunRecord] = []
    if not results_root.exists():
        return records
    for wrapper_path in sorted(results_root.glob("*/metadata/wrapper-manifest.json")):
        run_dir = wrapper_path.parent.parent
        wrapper, wrapper_error = load_json(wrapper_path)
        records.append(
            RunRecord(
                run_dir=run_dir,
                run_id=run_dir.name,
                wrapper=wrapper,
                wrapper_error=wrapper_error,
            )
        )
    return records


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except Exception:  # noqa: BLE001
        return None


def artifact_state(record: RunRecord) -> tuple[str, list[str]]:
    issues: list[str] = []
    wrapper_path = record.run_dir / "metadata" / "wrapper-manifest.json"
    remote_log = record.run_dir / "remote" / "logs" / "latest.log"
    remote_wrapper_run_manifest = record.run_dir / "remote" / "manifests" / "run-manifest.json"
    remote_tournament_manifest = record.run_dir / "remote" / "manifests" / "tournament-run-manifest.json"
    remote_tournament_artifact = record.run_dir / "remote" / "artifacts" / "tournament-run-artifact.json"
    wrapper_status = str((record.wrapper or {}).get("status") or record.wrapper_error or "unknown")

    if not wrapper_path.exists():
        issues.append("missing-wrapper-manifest")
    if not remote_log.exists():
        issues.append("missing-log")
    if not remote_wrapper_run_manifest.exists():
        issues.append("missing-wrapper-run-manifest")
    if wrapper_status != "running" and not remote_tournament_manifest.exists():
        issues.append("missing-tournament-manifest")
    if wrapper_status != "running" and not remote_tournament_artifact.exists():
        issues.append("missing-artifact")

    if record.wrapper_error:
        issues.append(record.wrapper_error)

    if not issues:
        return "complete", issues
    if wrapper_status == "running":
        return "in-progress", issues
    if "missing-wrapper-manifest" in issues:
        return "untracked", issues
    return "degraded", issues


def live_pods() -> dict[str, dict[str, Any]]:
    if mode != "pods":
        return {}
    payload = subprocess.check_output([runpodctl, "-o", "json", "pod", "list"], text=True)
    items = json.loads(payload)
    return {str(pod.get("id") or ""): pod for pod in items}


def summarize_artifacts(records: list[RunRecord]) -> int:
    if not records:
        print(f"No preserved run results found under {results_root}")
        return 0

    print(f"Artifact preservation audit: {results_root}")
    print()
    header = ["run_id", "pod_id", "wrapper_status", "log", "wrapper_manifest", "wrapper_run_manifest", "tournament_manifest", "artifact", "state", "notes"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    issue_count = 0
    for record in records:
        state, issues = artifact_state(record)
        wrapper_status = str((record.wrapper or {}).get("status") or record.wrapper_error or "unknown")
        remote_log = "yes" if (record.run_dir / "remote" / "logs" / "latest.log").exists() else "no"
        remote_manifest = "yes" if (record.run_dir / "remote" / "manifests" / "run-manifest.json").exists() else "no"
        wrapper_manifest = "yes" if (record.run_dir / "metadata" / "wrapper-manifest.json").exists() else "no"
        tournament_manifest = "yes" if (record.run_dir / "remote" / "manifests" / "tournament-run-manifest.json").exists() else "no"
        remote_artifact = "yes" if (record.run_dir / "remote" / "artifacts" / "tournament-run-artifact.json").exists() else "no"
        notes = ", ".join(issues) if issues else "ok"
        if state in {"degraded", "untracked"}:
            issue_count += 1
        print(
            " | ".join(
                [
                    record.run_id,
                    record.pod_id or "-",
                    wrapper_status,
                    remote_log,
                    wrapper_manifest,
                    remote_manifest,
                    tournament_manifest,
                    remote_artifact,
                    state,
                    notes,
                ]
            )
        )

    print()
    print(f"summary: {len(records)} runs scanned, {issue_count} with preservation issues")
    return 1 if issue_count else 0


def summarize_pods(records: list[RunRecord]) -> int:
    pods = live_pods()
    stale_after = timedelta(hours=stale_hours)
    now = datetime.now(timezone.utc)

    print(f"Pod hygiene audit: {results_root}")
    print()
    header = ["run_id", "pod_id", "pod_name", "wrapper_status", "live_status", "age", "notes"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))

    issues = 0
    seen_pods: set[str] = set()
    for record in records:
        wrapper = record.wrapper or {}
        pod = wrapper.get("pod", {}) if isinstance(wrapper, dict) else {}
        pod_id = str(pod.get("id") or "")
        pod_name = str(pod.get("name") or "")
        wrapper_status = str(wrapper.get("status") or record.wrapper_error or "unknown")
        live = pods.get(pod_id)
        live_status = str(live.get("desiredStatus") or "missing") if live else "missing"
        seen_pods.add(pod_id)
        started_at = parse_iso8601(str(wrapper.get("started_at") or ""))
        age = "-"
        stale_note = "ok"
        if started_at:
            delta = now - started_at
            age = f"{delta.total_seconds() / 3600:.1f}h"
            if wrapper_status == "running" and delta > stale_after and live_status != "RUNNING":
                stale_note = "stale-running-record"
            elif wrapper_status == "running" and live_status == "missing":
                stale_note = "stale-running-record"
            elif wrapper_status in {"success", "timeout", "failure"} and live_status == "RUNNING":
                stale_note = "pod-still-live-after-final-status"
        else:
            if wrapper_status == "running" and live_status == "missing":
                stale_note = "stale-running-record"

        if stale_note != "ok":
            issues += 1

        print(
            " | ".join(
                [
                    record.run_id,
                    pod_id or "-",
                    pod_name or "-",
                    wrapper_status,
                    live_status,
                    age,
                    stale_note,
                ]
            )
        )

    for pod_id, pod in sorted(pods.items()):
        if pod_id in seen_pods:
            continue
        issues += 1
        print(
            " | ".join(
                [
                    "-",
                    pod_id or "-",
                    str(pod.get("name") or "-"),
                    "untracked",
                    str(pod.get("desiredStatus") or "unknown"),
                    "-",
                    "live-pod-without-preserved-run",
                ]
            )
        )

    print()
    print(f"summary: {len(records)} preserved runs scanned, {issues} hygiene issues")
    return 1 if issues else 0


records = run_dirs()
if mode == "artifacts":
    raise SystemExit(summarize_artifacts(records))
if mode == "pods":
    raise SystemExit(summarize_pods(records))
raise SystemExit(f"unknown mode: {mode}")
PY
