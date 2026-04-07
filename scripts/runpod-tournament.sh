#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/runpod-tournament.sh [wrapper options] -- [command args]

Create or reuse a Runpod pod, sync the current repo snapshot, bootstrap Rust/CMake if
needed, and run the selected Rust target with the CUDA backend.

Wrapper options:
  --pod-id ID                     Reuse a specific existing pod by id.
  --pod-name NAME                 Pod name to reuse or create. Default: fractal-cuda
  --gpu-id GPU                    GPU id used when creating a new pod.
  --template-id ID                Runpod template id. Default: runpod-torch-v240
  --cloud-type TYPE               SECURE or COMMUNITY. Default: SECURE
  --gpu-count N                   GPU count for pod creation. Default: 1
  --container-disk-gb N           Container disk size. Default: 40
  --volume-gb N                   Persistent volume size. Default: 40
  --data-center-ids IDS           Comma-separated Runpod datacenter ids.
  --network-volume-id ID          Attach an existing network volume.
  --ssh-key PATH                  Private key path. Default: first matching local Runpod key, then ~/.ssh/id_ed25519
  --remote-dir PATH               Override remote worktree path. Default: <volumeMountPath>/fractal
  --state-dir PATH                Override remote state path. Default: <volumeMountPath>/.fractal-runpod
  --timeout-seconds N             Wait timeout for pod + SSH readiness. Default: 900
  --poll-seconds N                Poll interval while waiting. Default: 5
  --run-timeout-seconds N         Bound remote tournament runtime with timeout(1).
  --binary-kind KIND             Cargo target kind: example or bin. Default: example
  --binary-name NAME             Cargo target name. Default: tournament
  --no-compile                    Reuse a cached remote binary and fail if it is missing.
  --stop-after-run                Always stop the pod after the command finishes.
  --keep-pod                      Never stop the pod automatically.
  --dry-run                       Print the resolved actions without creating or running anything.
  --help                          Show this help.

Notes:
  - If no existing pod matches, the wrapper creates one and requires --gpu-id.
  - Newly created pods are stopped automatically after the run unless --keep-pod is set.
  - Existing pods are left running unless --stop-after-run is set.
  - After every run, the wrapper preserves the remote log plus any manifest/artifact files
    under .runpod-local-logs/runpod-results/<logical-id>/<attempt-id>/, with the synced remote
    tree in `remote/` and a wrapper manifest in `metadata/wrapper-manifest.json`.
  - Wrapper manifests now record Experiment Interface v1-style identity:
    logical experiment id/name stay stable across retries, while each attempt gets a new run id.
  - Command arguments after "--" are passed to the cached release binary as:
      <cached-binary> --backend cuda ...

Examples:
  scripts/runpod-tournament.sh \
    --gpu-id "NVIDIA GeForce RTX 4090" \
    -- --preset research-medium

  scripts/runpod-tournament.sh \
    --pod-name fractal-a100 \
    --gpu-id "NVIDIA A100 80GB PCIe" \
    --keep-pod \
    -- --sequence first-run

  scripts/runpod-tournament.sh \
    --pod-name fractal-v3a \
    --gpu-id "NVIDIA A100 80GB PCIe" \
    --binary-kind bin \
    --binary-name v3a-hybrid-attention-matrix \
    -- --variant all --primitive-profile p2-3 --steps 16 --eval-batches 4
EOF
}

log() {
    printf '[runpod-wrapper] %s\n' "$*"
}

die() {
    printf '[runpod-wrapper] error: %s\n' "$*" >&2
    exit 1
}

quote_cmd() {
    local quoted=()
    local arg
    for arg in "$@"; do
        quoted+=("$(printf '%q' "$arg")")
    done
    printf '%s ' "${quoted[@]}"
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

resolve_ssh_key() {
    if [ -n "${SSH_KEY:-}" ]; then
        return 0
    fi

if detected_key="$("$RUNPODCTL_BIN" ssh list-keys | python3 -c '
import glob
import json
import pathlib
import sys

def normalize_key(raw: str) -> str:
    parts = raw.split()
    return " ".join(parts[:2])

payload = json.load(sys.stdin)
registered = {
    normalize_key(entry.get("key") or "")
    for entry in payload.get("keys", [])
}

search_globs = [
    pathlib.Path.home() / ".runpod" / "ssh" / "*.pub",
    pathlib.Path.home() / ".ssh" / "*.pub",
]

for pattern in search_globs:
    for path in sorted(glob.glob(str(pattern))):
        candidate = normalize_key(pathlib.Path(path).read_text())
        if candidate in registered:
            print(path[:-4])
            sys.exit(0)

sys.exit(1)
')"; then
        SSH_KEY="$detected_key"
    else
        SSH_KEY="${HOME}/.ssh/id_ed25519"
    fi
}

json_value() {
    local path="$1"
    python3 -c '
import json
import sys

path = [part for part in sys.argv[1].split(".") if part]
value = json.load(sys.stdin)
for part in path:
    if isinstance(value, list):
        value = value[int(part)]
    else:
        value = value.get(part)
    if value is None:
        sys.exit(1)

if isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
' "$path"
}

resolve_pod_id_by_name() {
    local name="$1"
    "$RUNPODCTL_BIN" pod list | python3 -c '
import json
import sys

name = sys.argv[1]
pods = json.load(sys.stdin)
matches = [
    pod for pod in pods
    if pod.get("name") == name and pod.get("desiredStatus") != "TERMINATED"
]

if not matches:
    sys.exit(10)

if len(matches) > 1:
    names = ", ".join(pod.get("id", "<unknown>") for pod in matches)
    print(f"multiple reusable pods named {name}: {names}", file=sys.stderr)
    sys.exit(11)

print(matches[0]["id"])
' "$name"
}

load_pod_state() {
    local pod_json
    pod_json="$("$RUNPODCTL_BIN" pod get "$POD_ID")"

    POD_NAME_RESOLVED="$(printf '%s' "$pod_json" | json_value "name" || true)"
    POD_STATUS="$(printf '%s' "$pod_json" | json_value "desiredStatus" || true)"
    PUBLIC_IP="$(printf '%s' "$pod_json" | json_value "publicIp" || true)"
    if [ -z "${PUBLIC_IP:-}" ]; then
        PUBLIC_IP="$(printf '%s' "$pod_json" | json_value "ssh.ip" || true)"
    fi
    SSH_PORT="$(printf '%s' "$pod_json" | python3 -c '
import json
import sys

pod = json.load(sys.stdin)
port_mappings = pod.get("portMappings") or {}
value = port_mappings.get("22")
if value is None:
    value = port_mappings.get(22)
if value is None:
    ssh = pod.get("ssh") or {}
    value = ssh.get("port")
if value is not None:
    print(value)
'
)"
    VOLUME_MOUNT_PATH="$(printf '%s' "$pod_json" | json_value "volumeMountPath" || true)"

    if [ -z "${VOLUME_MOUNT_PATH:-}" ]; then
        VOLUME_MOUNT_PATH="/workspace"
    fi
    if [ -z "${REMOTE_DIR:-}" ]; then
        REMOTE_DIR="${VOLUME_MOUNT_PATH}/fractal"
    fi
    if [ -z "${STATE_DIR:-}" ]; then
        STATE_DIR="${VOLUME_MOUNT_PATH}/.fractal-runpod"
    fi
}

build_ssh_base() {
    SSH_BASE=(
        ssh
        -i "$SSH_KEY"
        -p "$SSH_PORT"
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o ServerAliveInterval=30
        -o ServerAliveCountMax=10
        "root@${PUBLIC_IP}"
    )
}

create_pod_if_needed() {
    if [ -n "$POD_ID" ]; then
        return 0
    fi

    local err_file
    local resolved_id
    local resolved_status
    err_file="$(mktemp)"

    set +e
    resolved_id="$(resolve_pod_id_by_name "$POD_NAME" 2>"$err_file")"
    resolved_status=$?
    set -e

    if [ "$resolved_status" -eq 0 ]; then
        POD_ID="$resolved_id"
        CREATED_POD=0
        rm -f "$err_file"
        return 0
    else
        case "$resolved_status" in
            10)
                ;;
            11)
                local message
                message="$(cat "$err_file")"
                rm -f "$err_file"
                die "$message"
                ;;
            *)
                rm -f "$err_file"
                die "failed to query existing pods"
                ;;
        esac
    fi
    rm -f "$err_file"

    [ -n "$GPU_ID" ] || die "no reusable pod named '$POD_NAME' was found; provide --gpu-id to create one"

    local create_cmd=(
        "$RUNPODCTL_BIN" pod create
        --name "$POD_NAME"
        --template-id "$TEMPLATE_ID"
        --gpu-id "$GPU_ID"
        --gpu-count "$GPU_COUNT"
        --cloud-type "$CLOUD_TYPE"
        --ports "22/tcp"
        --container-disk-in-gb "$CONTAINER_DISK_GB"
        --volume-in-gb "$VOLUME_GB"
    )

    if [ "$CLOUD_TYPE" = "COMMUNITY" ]; then
        create_cmd+=(--public-ip)
    fi
    if [ -n "$DATA_CENTER_IDS" ]; then
        create_cmd+=(--data-center-ids "$DATA_CENTER_IDS")
    fi
    if [ -n "$NETWORK_VOLUME_ID" ]; then
        create_cmd+=(--network-volume-id "$NETWORK_VOLUME_ID")
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "dry-run create command: $(quote_cmd "${create_cmd[@]}")"
        exit 0
    fi

    log "creating Runpod pod '$POD_NAME'"
    local create_json
    create_json="$("${create_cmd[@]}")"
    POD_ID="$(printf '%s' "$create_json" | json_value "id")"
    CREATED_POD=1
}

ensure_private_key_is_registered() {
    [ -f "$SSH_KEY" ] || die "ssh private key not found: $SSH_KEY"
    [ -f "${SSH_KEY}.pub" ] || die "ssh public key not found: ${SSH_KEY}.pub"

    local local_key
    local_key="$(tr -d '\n' < "${SSH_KEY}.pub")"
    if ! "$RUNPODCTL_BIN" ssh list-keys | python3 -c '
import json
import sys

def normalize_key(raw: str) -> str:
    parts = raw.split()
    return " ".join(parts[:2])

target = normalize_key(sys.argv[1])
payload = json.load(sys.stdin)
keys = payload.get("keys", [])
for entry in keys:
    key = normalize_key(entry.get("key") or "")
    if key == target:
        sys.exit(0)

sys.exit(1)
' "$local_key"
    then
        if [ "$DRY_RUN" -eq 1 ]; then
            log "warning: ${SSH_KEY}.pub is not registered with Runpod; a real run would require: runpodctl ssh add-key --key-file ${SSH_KEY}.pub"
            return 0
        fi
        die "public key ${SSH_KEY}.pub is not registered with Runpod; run: runpodctl ssh add-key --key-file ${SSH_KEY}.pub"
    fi
}

maybe_start_pod() {
    load_pod_state
    if [ "$POD_STATUS" = "EXITED" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            log "dry-run start command: $(quote_cmd "$RUNPODCTL_BIN" pod start "$POD_ID")"
            exit 0
        fi
        log "starting existing pod '$POD_NAME_RESOLVED' ($POD_ID)"
        "$RUNPODCTL_BIN" pod start "$POD_ID" >/dev/null
    fi
}

wait_for_ssh() {
    local deadline=$((SECONDS + TIMEOUT_SECONDS))
    while [ "$SECONDS" -lt "$deadline" ]; do
        load_pod_state
        if [ "$POD_STATUS" = "RUNNING" ] && [ -n "${PUBLIC_IP:-}" ] && [ -n "${SSH_PORT:-}" ]; then
            build_ssh_base
            if "${SSH_BASE[@]}" true >/dev/null 2>&1; then
                return 0
            fi
        fi
        sleep "$POLL_SECONDS"
    done

    die "pod '$POD_ID' did not become SSH-ready within ${TIMEOUT_SECONDS}s"
}

sync_worktree() {
    local remote_dir_q
    remote_dir_q="$(printf '%q' "$REMOTE_DIR")"

    log "syncing worktree to ${POD_NAME_RESOLVED:-$POD_ID}:${REMOTE_DIR}"
    if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git -C "$REPO_ROOT" ls-files --cached --others --exclude-standard -z \
            | while IFS= read -r -d '' path; do
                [ -e "${REPO_ROOT}/${path}" ] || continue
                printf '%s\0' "$path"
            done \
            | COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar \
                --no-mac-metadata \
                --no-xattrs \
                -C "$REPO_ROOT" \
                --null \
                -T - \
                -cf - \
            | "${SSH_BASE[@]}" "rm -rf ${remote_dir_q} && mkdir -p ${remote_dir_q} && tar --no-same-owner -xf - -C ${remote_dir_q}"
    else
        COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar \
            --no-mac-metadata \
            --no-xattrs \
            --exclude=.git \
            --exclude=target \
            --exclude=.cmake-venv \
            --exclude=.DS_Store \
            -C "$REPO_ROOT" \
            -cf - . \
            | "${SSH_BASE[@]}" "rm -rf ${remote_dir_q} && mkdir -p ${remote_dir_q} && tar --no-same-owner -xf - -C ${remote_dir_q}"
    fi
}

build_remote_tournament_args() {
    REMOTE_TOURNAMENT_ARGS=()
    local index=0
    local arg
    local manifest_path
    local rewritten
    while [ "$index" -lt "${#TOURNAMENT_ARGS[@]}" ]; do
        arg="${TOURNAMENT_ARGS[$index]}"
        if [ "$arg" = "--experiment-manifest" ] && [ $((index + 1)) -lt "${#TOURNAMENT_ARGS[@]}" ]; then
            manifest_path="${TOURNAMENT_ARGS[$((index + 1))]}"
            if [ "${manifest_path#${REPO_ROOT}/}" != "$manifest_path" ]; then
                rewritten="${REMOTE_DIR}/${manifest_path#${REPO_ROOT}/}"
            else
                rewritten="$manifest_path"
            fi
            REMOTE_TOURNAMENT_ARGS+=("$arg" "$rewritten")
            index=$((index + 2))
            continue
        fi
        REMOTE_TOURNAMENT_ARGS+=("$arg")
        index=$((index + 1))
    done
}

resolve_local_identity_context() {
    RUN_LOCAL_BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
    RUN_LOCAL_COMMIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
}

build_experiment_context() {
    python3 - "$RUN_RESULTS_ROOT" \
        "$RUN_STARTED_AT" \
        "$RUN_TIMEOUT_SECONDS" \
        "$RUN_LOCAL_BRANCH" \
        "$RUN_LOCAL_COMMIT_SHA" \
        "${POD_NAME_RESOLVED:-$POD_NAME}" \
        "${TOURNAMENT_ARGS[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


(
    results_root_raw,
    started_at,
    run_timeout_seconds_raw,
    branch,
    commit_sha,
    pod_name,
    *command_args,
) = sys.argv[1:]

results_root = Path(results_root_raw)
run_timeout_seconds = int(run_timeout_seconds_raw)

LANE_DEFAULT_PRESET = {
    "all": "default",
    "baseline": "research-medium",
    "challenger": "bullpen-polish",
    "proving-ground": "minimal-proving-ground",
    "leader": "generation-four",
}


def extract_arg(args: list[str], flag: str) -> str | None:
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sanitize(value: str) -> str:
    chars = []
    for ch in value:
        if ch.isalnum() or ch in "._-":
            chars.append(ch)
        elif ch in " /:":
            chars.append("_")
    cleaned = "".join(chars).strip("._-")
    return cleaned or "unknown"


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y%m%dT%H%M%S%z", "%Y%m%dT%H%M%SZ"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def format_timestamp(value: datetime | None, fallback: str) -> str:
    if value is None:
        return fallback
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_spec(
    args: list[str],
    *,
    branch_value: str,
    commit_value: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    manifest_path = extract_arg(args, "--experiment-manifest")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        species = manifest.get("species")
        preset = manifest.get("preset")
        seed = manifest.get("seed")
        execution_mode = manifest.get("execution_mode")
        parallelism = manifest.get("parallelism")
        backend = manifest.get("backend") or "cuda"
        comparison_contract = manifest.get("comparison") or "authoritative_same_preset"
        benchmark_mode = manifest.get("benchmark_mode") or "leaderboard"
        logical_name = manifest.get("logical_name")
        question_summary = manifest.get("question_summary")
        spec = {
            "question": {
                "lane": "manifest",
                "selection": {
                    "kind": "manifest",
                    "value": str(Path(manifest_path).name),
                },
                "summary": question_summary,
            },
            "variant": {
                "species": species,
            },
            "budget": {
                "seed": seed,
                "run_timeout_seconds": timeout_seconds,
                "perplexity_eval_batches": manifest.get("perplexity_eval_batches"),
                "arc_eval_batches": manifest.get("arc_eval_batches"),
            },
            "runtime": {
                "backend": backend,
                "execution_mode": execution_mode,
                "parallelism": parallelism,
                "benchmark_mode": benchmark_mode,
            },
            "comparison": {
                "contract": comparison_contract,
            },
            "execution": {
                "target": "runpod",
                "expected_branch": manifest.get("expected_branch"),
            },
            "build": {
                "branch": branch_value,
                "commit_sha": commit_value,
            },
        }
        if not logical_name:
            logical_name = sanitize(
                f"{benchmark_mode}-{species or 'unknown'}-seed-{seed if seed is not None else 'na'}-{(commit_value or 'unknown')[:8]}"
            )
        canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        logical_id = f"exp-{hashlib.sha256(canonical).hexdigest()[:16]}"
        return {
            "logical_id": logical_id,
            "logical_name": logical_name,
            "spec": spec,
            "resolved": {
                "lane": "manifest",
                "species": species,
                "preset": preset,
                "sequence": None,
                "seed": seed,
                "manifest_path": manifest_path,
            },
        }

    species = extract_arg(args, "--species")
    lane = extract_arg(args, "--lane") or "all"
    preset = extract_arg(args, "--preset")
    sequence = extract_arg(args, "--sequence")
    seed = parse_int(extract_arg(args, "--seed"))
    execution_mode = extract_arg(args, "--mode")
    parallelism = parse_int(extract_arg(args, "--parallelism"))
    perplexity_eval_batches = parse_int(extract_arg(args, "--perplexity-eval-batches"))
    arc_eval_batches = parse_int(extract_arg(args, "--arc-eval-batches"))
    if preset:
        selection = {"kind": "preset", "value": preset}
    elif sequence:
        selection = {"kind": "sequence", "value": sequence}
    elif species:
        selection = {"kind": "preset", "value": "candidate-stress"}
    else:
        selection = {"kind": "preset", "value": LANE_DEFAULT_PRESET.get(lane, "default")}

    spec = {
        "question": {
            "lane": lane,
            "selection": selection,
        },
        "variant": {
            "species": species,
        },
        "budget": {
            "seed": seed,
            "run_timeout_seconds": timeout_seconds,
            "perplexity_eval_batches": perplexity_eval_batches,
            "arc_eval_batches": arc_eval_batches,
        },
        "runtime": {
            "backend": "cuda",
            "execution_mode": execution_mode,
            "parallelism": parallelism,
        },
        "execution": {
            "target": "runpod",
        },
        "build": {
            "branch": branch_value,
            "commit_sha": commit_value,
        },
    }
    logical_name_parts = []
    if species:
        logical_name_parts.append(f"species-{species}")
    else:
        logical_name_parts.append(f"lane-{lane}")
    logical_name_parts.append(f"{selection['kind']}-{selection['value']}")
    if seed is not None:
        logical_name_parts.append(f"seed-{seed}")
    logical_name_parts.append(f"commit-{(commit_value or 'unknown')[:8]}")
    logical_name = "_".join(sanitize(part) for part in logical_name_parts)
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    logical_id = f"exp-{hashlib.sha256(canonical).hexdigest()[:16]}"
    return {
        "logical_id": logical_id,
        "logical_name": logical_name,
        "spec": spec,
        "resolved": {
            "lane": lane,
            "species": species,
            "preset": selection["value"] if selection["kind"] == "preset" else None,
            "sequence": selection["value"] if selection["kind"] == "sequence" else None,
            "seed": seed,
        },
    }


def infer_identity(manifest: dict[str, Any]) -> tuple[str, str, str | None]:
    experiment = manifest.get("experiment")
    if isinstance(experiment, dict):
        experiment_id = experiment.get("experiment_id")
        if isinstance(experiment_id, dict):
            logical_id = experiment_id.get("logical_id")
            logical_name = experiment_id.get("logical_name")
            created_at = experiment_id.get("created_at")
            if logical_id and logical_name:
                return str(logical_id), str(logical_name), str(created_at) if created_at else None
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    build = manifest.get("build") if isinstance(manifest.get("build"), dict) else {}
    args = runtime.get("tournament_args")
    if not isinstance(args, list):
        args = []
    inferred = resolve_spec(
        [str(arg) for arg in args],
        branch_value=str(build.get("branch") or ""),
        commit_value=str(build.get("commit_sha") or ""),
        timeout_seconds=parse_int(str(runtime.get("run_timeout_seconds"))) or 0,
    )
    created_at = manifest.get("started_at")
    if created_at:
        created_at = str(created_at)
    return inferred["logical_id"], inferred["logical_name"], created_at


identity = resolve_spec(
    tournament_args,
    branch_value=branch,
    commit_value=commit_sha,
    timeout_seconds=run_timeout_seconds,
)

earliest_created_at: datetime | None = parse_timestamp(started_at)
attempt_count = 0
if results_root.exists():
    for wrapper_path in sorted(results_root.glob("**/metadata/wrapper-manifest.json")):
        try:
            manifest = json.loads(wrapper_path.read_text())
        except Exception:
            continue
        logical_id, _, created_at_raw = infer_identity(manifest)
        if logical_id != identity["logical_id"]:
            continue
        attempt_count += 1
        candidate = parse_timestamp(created_at_raw) or parse_timestamp(manifest.get("started_at"))
        if candidate is not None and (
            earliest_created_at is None or candidate < earliest_created_at
        ):
            earliest_created_at = candidate

attempt_index = attempt_count + 1
started_at_slug = sanitize(started_at)
attempt_id = f"{started_at_slug}_a{attempt_index:02d}"

context = {
    "interface_version": "experiment-interface-v1-wrapper",
    "identity_source": "wrapper-derived",
    "experiment_id": {
        "logical_id": identity["logical_id"],
        "logical_name": identity["logical_name"],
        "generated_run_id": attempt_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "branch": branch,
        "commit_sha": commit_sha,
        "created_at": format_timestamp(earliest_created_at, started_at),
        "started_at": started_at,
    },
    "question": identity["spec"]["question"],
    "variant": identity["spec"]["variant"],
    "budget": identity["spec"]["budget"],
    "runtime": identity["spec"]["runtime"],
    "comparison": identity["spec"].get("comparison", {}),
    "execution": {
        "target": "runpod",
        "backend": "cuda",
        "pod_name": pod_name,
        "retry_policy": "manual-wrapper-retry",
    },
    "artifacts": {
        "wrapper_manifest_required": True,
        "wrapper_run_manifest_required": True,
        "tournament_manifest_required": True,
        "tournament_artifact_required": True,
        "final_log_required": True,
    },
    "resolved": identity["resolved"],
}
print(json.dumps(context, sort_keys=True))
PY
}

prepare_run_preservation() {
    local started_at
    started_at="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_STARTED_AT="$started_at"
    RUN_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"
    RUN_EXPERIMENT_CONTEXT_JSON="$(build_experiment_context)"
    RUN_LOGICAL_ID="$(printf '%s' "$RUN_EXPERIMENT_CONTEXT_JSON" | json_value "experiment_id.logical_id")"
    RUN_ID="$(printf '%s' "$RUN_EXPERIMENT_CONTEXT_JSON" | json_value "experiment_id.attempt_id")"
    RUN_RESULT_DIR="${RUN_RESULTS_ROOT}/${RUN_LOGICAL_ID}/${RUN_ID}"
    RUN_RESULT_REMOTE_DIR="${RUN_RESULT_DIR}/remote"
    RUN_RESULT_METADATA_DIR="${RUN_RESULT_DIR}/metadata"
    RUN_RESULT_MANIFEST="${RUN_RESULT_METADATA_DIR}/wrapper-manifest.json"

    mkdir -p "$RUN_RESULT_REMOTE_DIR" "$RUN_RESULT_METADATA_DIR"
}

write_local_wrapper_manifest() {
    local status="$1"
    local exit_code="$2"
    local finished_at="$3"
    local local_branch="$4"
    local local_build_key="$5"
    local local_commit_sha="$6"
    python3 - "$RUN_RESULT_MANIFEST" \
        "$status" \
        "$exit_code" \
        "$finished_at" \
        "$RUN_ID" \
        "$RUN_STARTED_AT" \
        "$POD_ID" \
        "${POD_NAME_RESOLVED:-$POD_NAME}" \
        "$POD_STATUS" \
        "$REMOTE_DIR" \
        "$STATE_DIR" \
        "$local_branch" \
        "$local_build_key" \
        "$local_commit_sha" \
        "$RUN_TIMEOUT_SECONDS" \
        "$RUN_EXPERIMENT_CONTEXT_JSON" \
        "${TOURNAMENT_ARGS[@]}" <<'PY'
import json
import pathlib
import sys

(
    path,
    status,
    exit_code,
    finished_at,
    run_id,
    started_at,
    pod_id,
    pod_name,
    pod_status,
    remote_dir,
    state_dir,
    local_branch,
    local_build_key,
    local_commit_sha,
    run_timeout_seconds,
    experiment_context_json,
    *tournament_args,
) = sys.argv[1:]

experiment = json.loads(experiment_context_json)
manifest = {
    "run_id": run_id,
    "started_at": started_at,
    "finished_at": finished_at,
    "status": status,
    "exit_code": int(exit_code),
    "pod": {
        "id": pod_id,
        "name": pod_name,
        "status": pod_status,
    },
    "paths": {
        "remote_dir": remote_dir,
        "state_dir": state_dir,
        "preservation_root": str(pathlib.Path(path).parent.parent),
    },
    "build": {
        "branch": local_branch,
        "build_key": local_build_key,
        "commit_sha": local_commit_sha,
    },
    "runtime": {
        "run_timeout_seconds": int(run_timeout_seconds),
        "command_args": command_args,
        "tournament_args": command_args,
        "backend": "cuda",
    },
    "experiment": experiment,
}

path_obj = pathlib.Path(path)
path_obj.parent.mkdir(parents=True, exist_ok=True)
path_obj.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

preserve_remote_results() {
    local preserve_status=0
    log "preserving remote results under ${RUN_RESULT_DIR}"
    if "${SSH_BASE[@]}" bash -s -- "$STATE_DIR" <<'REMOTE' | tar -x -C "$RUN_RESULT_REMOTE_DIR" -f -; then
set -euo pipefail

state_dir="$1"
cd "$state_dir"

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

for path in logs/latest.log last-sync.txt last-build-key.txt; do
    [ -f "$path" ] && printf '%s\0' "$path" >> "$tmp_list"
done

for dir in artifacts manifests; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -print0 | sort -z >> "$tmp_list"
    fi
done

if [ ! -s "$tmp_list" ]; then
    tar -cf - --files-from /dev/null
    exit 0
fi

tar -cf - --null -T "$tmp_list"
REMOTE
        preserve_status=0
    else
        preserve_status=$?
        log "warning: could not fully preserve remote results for ${POD_ID} (exit ${preserve_status})"
    fi
    return "$preserve_status"
}

bootstrap_remote() {
    log "bootstrapping remote toolchain"
    "${SSH_BASE[@]}" bash -s -- "$REMOTE_DIR" "$STATE_DIR" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
state_dir="$2"
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
fi

need_apt=0
for cmd in curl git cmake pkg-config g++ python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        need_apt=1
    fi
done

if [ "$need_apt" -eq 1 ]; then
    apt-get update
    apt-get install -y build-essential cmake curl git pkg-config libssl-dev python3
fi

if ! command -v cargo >/dev/null 2>&1 || ! cargo --version >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
fi

if command -v rustup >/dev/null 2>&1; then
    rustup default stable >/dev/null
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is not installed in this image; use an official CUDA-enabled Runpod template" >&2
    exit 1
fi

mkdir -p "$remote_dir" "$state_dir/logs" "$state_dir/target" "$state_dir/artifacts" "$state_dir/manifests"
REMOTE
}

run_remote_tournament() {
    local local_build_key
    local run_experiment_context_b64
    local_build_key="$(python3 - "$REPO_ROOT" <<'PY'
import hashlib
import os
import sys

root = sys.argv[1]
paths = [
    "Cargo.toml",
    "Cargo.lock",
    ".cargo",
    "src",
    "examples",
    "fractal-core",
    "fractal-primitives-private",
    "fractal-eval-private",
    "vendor",
]

digest = hashlib.sha256()
for rel in paths:
    full = os.path.join(root, rel)
    if not os.path.exists(full):
        continue
    if os.path.isfile(full):
        entries = [(rel, full)]
    else:
        entries = []
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames.sort()
            filenames.sort()
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                entries.append((os.path.relpath(path, root), path))

    for relpath, path in entries:
        digest.update(relpath.encode())
        digest.update(b"\0")
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        digest.update(b"\0")

print(digest.hexdigest())
PY
    )"

    RUN_LOCAL_BUILD_KEY="$local_build_key"
    build_remote_tournament_args
    run_experiment_context_b64="$(printf '%s' "$RUN_EXPERIMENT_CONTEXT_JSON" | base64 | tr -d '\n')"

    write_local_wrapper_manifest "running" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_LOCAL_BRANCH" "$local_build_key" "$RUN_LOCAL_COMMIT_SHA"

    log "running remote CUDA ${BINARY_KIND}:${BINARY_NAME}"
    "${SSH_BASE[@]}" bash -s -- \
        "$REMOTE_DIR" \
        "$STATE_DIR" \
        "$RUN_LOCAL_BRANCH" \
        "$RUN_LOCAL_COMMIT_SHA" \
        "$local_build_key" \
        "$BINARY_KIND" \
        "$BINARY_NAME" \
        "$NO_COMPILE" \
        "$RUN_TIMEOUT_SECONDS" \
        "$RUN_ID" \
        "$POD_ID" \
        "$run_experiment_context_b64" \
        "${REMOTE_TOURNAMENT_ARGS[@]}" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
state_dir="$2"
local_branch="$3"
local_commit_sha="$4"
local_build_key="$5"
binary_kind="$6"
binary_name="$7"
no_compile="$8"
run_timeout_seconds="$9"
run_id="${10}"
pod_id="${11}"
experiment_context_b64="${12}"
shift 12

if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -d "${CUDA_HOME}/bin" ]; then
    export PATH="${CUDA_HOME}/bin:$PATH"
fi
if [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$state_dir/target"
export FRACTAL_RUN_ARTIFACT_DIR="$state_dir/artifacts"
export FRACTAL_RUN_MANIFEST_DIR="$state_dir/manifests"
export FRACTAL_RUN_ID="$run_id"
export FRACTAL_RUN_POD_ID="$pod_id"
export FRACTAL_BRANCH="$local_branch"
export FRACTAL_COMMIT_SHA="$local_commit_sha"
export FRACTAL_WRAPPER_TIMEOUT_SECONDS="$run_timeout_seconds"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
binary_dir="$state_dir/bin"
binary_path="$binary_dir/$binary_name"
build_key_file="$state_dir/last-build-key.txt"
remote_manifest="$state_dir/manifests/run-manifest.json"

mkdir -p "$state_dir/logs" "$binary_dir"
mkdir -p "$(dirname "$remote_manifest")" "$state_dir/artifacts"
printf 'branch=%s\nbuild_key=%s\nrun_at=%s\n' \
    "$local_branch" \
    "$local_build_key" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state_dir/last-sync.txt"

cd "$remote_dir"
: > "$state_dir/logs/latest.log"

write_manifest() {
    local status="$1"
    local exit_code="$2"
    local finished_at="$3"
    python3 - "$remote_manifest" \
        "$status" \
        "$exit_code" \
        "$finished_at" \
        "$local_branch" \
        "$local_commit_sha" \
        "$local_build_key" \
        "$binary_kind" \
        "$binary_name" \
        "$run_timeout_seconds" \
        "$state_dir" \
        "$remote_dir" \
        "$experiment_context_b64" \
        "$@" <<'PY'
import base64
import json
import pathlib
import sys

(
    path,
    status,
    exit_code,
    finished_at,
    local_branch,
    local_commit_sha,
    local_build_key,
    binary_kind,
    binary_name,
    run_timeout_seconds,
    state_dir,
    remote_dir,
    experiment_context_b64,
    *command_args,
) = sys.argv[1:]

experiment = json.loads(base64.b64decode(experiment_context_b64).decode("utf-8"))
manifest = {
    "run_id": experiment["experiment_id"]["attempt_id"],
    "status": status,
    "exit_code": int(exit_code),
    "finished_at": finished_at,
    "build": {
        "branch": local_branch,
        "commit_sha": local_commit_sha,
        "build_key": local_build_key,
    },
    "runtime": {
        "backend": "cuda",
        "binary_kind": binary_kind,
        "binary_name": binary_name,
        "run_timeout_seconds": int(run_timeout_seconds),
        "command_args": command_args,
        "tournament_args": command_args,
    },
    "paths": {
        "state_dir": state_dir,
        "remote_dir": remote_dir,
        "manifest_path": str(pathlib.Path(path).as_posix()),
    },
    "experiment": experiment,
}

path_obj = pathlib.Path(path)
path_obj.parent.mkdir(parents=True, exist_ok=True)
path_obj.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

command_args=("$@")
write_manifest "running" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${command_args[@]}"

cleanup_manifest() {
    local exit_code="$1"
    local final_status="failure"
    if [ "$exit_code" -eq 0 ]; then
        final_status="success"
    elif [ "$exit_code" -eq 124 ]; then
        final_status="timeout"
    fi
    write_manifest "$final_status" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$@" || true
}
trap 'rc=$?; cleanup_manifest "$rc" "${command_args[@]}"' EXIT

needs_build=1
if [ -x "$binary_path" ] && [ -f "$build_key_file" ] && [ "$(cat "$build_key_file")" = "$local_build_key" ]; then
    needs_build=0
fi

if [ "$no_compile" = "1" ]; then
    if [ ! -x "$binary_path" ]; then
        echo "cached binary missing; run without --no-compile first" >&2
        exit 1
    fi
    needs_build=0
fi

if [ "$needs_build" -eq 1 ]; then
    echo "[runpod-wrapper] compiling release binary" | tee -a "$state_dir/logs/latest.log"
    build_rustflags="-Clink-self-contained=no -Clink-arg=-fuse-ld=bfd"
    if [ -n "${RUSTFLAGS:-}" ]; then
        export RUSTFLAGS="${RUSTFLAGS} ${build_rustflags}"
    else
        export RUSTFLAGS="$build_rustflags"
    fi
    echo "[runpod-wrapper] using system bfd linker for remote cargo build" \
        | tee -a "$state_dir/logs/latest.log"
    stdbuf -oL -eL cargo build --release --features cuda --"$binary_kind" "$binary_name" \
        2>&1 | tee -a "$state_dir/logs/latest.log"
    if [ "$binary_kind" = "example" ]; then
        built_binary_path="$CARGO_TARGET_DIR/release/examples/$binary_name"
    else
        built_binary_path="$CARGO_TARGET_DIR/release/$binary_name"
    fi
    cp "$built_binary_path" "$binary_path"
    chmod +x "$binary_path"
    printf '%s\n' "$local_build_key" > "$build_key_file"
else
    echo "[runpod-wrapper] reusing cached release binary" | tee -a "$state_dir/logs/latest.log"
fi

if [ -n "$run_timeout_seconds" ] && [ "$run_timeout_seconds" -gt 0 ]; then
    set +e
    timeout --signal=TERM "$run_timeout_seconds" \
        stdbuf -oL -eL "$binary_path" --backend cuda "$@" 2>&1 | tee -a "$state_dir/logs/latest.log"
    run_status=$?
    set -e
    if [ "$run_status" -eq 124 ]; then
        echo "[runpod-wrapper] tournament timed out after ${run_timeout_seconds}s" \
            | tee -a "$state_dir/logs/latest.log"
    fi
    exit "$run_status"
fi

stdbuf -oL -eL "$binary_path" --backend cuda "$@" 2>&1 | tee -a "$state_dir/logs/latest.log"
REMOTE
}

cleanup() {
    local should_stop=0
    case "$STOP_MODE" in
        always)
            should_stop=1
            ;;
        auto)
            if [ "$CREATED_POD" -eq 1 ]; then
                should_stop=1
            fi
            ;;
        never)
            should_stop=0
            ;;
        *)
            should_stop=0
            ;;
    esac

    if [ "$should_stop" -eq 1 ] && [ -n "${POD_ID:-}" ] && [ "$DRY_RUN" -eq 0 ]; then
        log "stopping pod $POD_ID"
        "$RUNPODCTL_BIN" pod stop "$POD_ID" >/dev/null || true
    fi
}

RUNPODCTL_BIN="${RUNPODCTL_BIN:-runpodctl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

POD_ID=""
POD_NAME="fractal-cuda"
GPU_ID=""
TEMPLATE_ID="runpod-torch-v240"
CLOUD_TYPE="SECURE"
GPU_COUNT="1"
CONTAINER_DISK_GB="40"
VOLUME_GB="40"
DATA_CENTER_IDS=""
NETWORK_VOLUME_ID=""
SSH_KEY="${SSH_KEY:-}"
REMOTE_DIR=""
STATE_DIR=""
TIMEOUT_SECONDS="900"
POLL_SECONDS="5"
RUN_TIMEOUT_SECONDS="0"
BINARY_KIND="example"
BINARY_NAME="tournament"
NO_COMPILE=0
STOP_MODE="auto"
DRY_RUN=0
CREATED_POD=0
RUN_STARTED_AT=""
RUN_ID=""
RUN_RESULTS_ROOT=""
RUN_RESULT_DIR=""
RUN_RESULT_REMOTE_DIR=""
RUN_RESULT_METADATA_DIR=""
RUN_RESULT_MANIFEST=""
RUN_EXPERIMENT_CONTEXT_JSON=""
RUN_LOCAL_BRANCH=""
RUN_LOCAL_BUILD_KEY=""
RUN_LOCAL_COMMIT_SHA=""
POD_NAME_RESOLVED=""
POD_STATUS=""
PUBLIC_IP=""
SSH_PORT=""
VOLUME_MOUNT_PATH=""
TOURNAMENT_ARGS=()
SSH_BASE=()

while [ $# -gt 0 ]; do
    case "$1" in
        --pod-id)
            POD_ID="$2"
            shift 2
            ;;
        --pod-name)
            POD_NAME="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --template-id)
            TEMPLATE_ID="$2"
            shift 2
            ;;
        --cloud-type)
            CLOUD_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --container-disk-gb)
            CONTAINER_DISK_GB="$2"
            shift 2
            ;;
        --volume-gb)
            VOLUME_GB="$2"
            shift 2
            ;;
        --data-center-ids)
            DATA_CENTER_IDS="$2"
            shift 2
            ;;
        --network-volume-id)
            NETWORK_VOLUME_ID="$2"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="$2"
            shift 2
            ;;
        --remote-dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --state-dir)
            STATE_DIR="$2"
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --poll-seconds)
            POLL_SECONDS="$2"
            shift 2
            ;;
        --run-timeout-seconds)
            RUN_TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --binary-kind)
            BINARY_KIND="$2"
            shift 2
            ;;
        --binary-name)
            BINARY_NAME="$2"
            shift 2
            ;;
        --no-compile)
            NO_COMPILE=1
            shift
            ;;
        --stop-after-run)
            STOP_MODE="always"
            shift
            ;;
        --keep-pod)
            STOP_MODE="never"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            TOURNAMENT_ARGS=("$@")
            break
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

require_cmd "$RUNPODCTL_BIN"
require_cmd python3
require_cmd ssh
require_cmd tar
require_cmd git

case "$BINARY_KIND" in
    example|bin)
        ;;
    *)
        die "invalid --binary-kind: $BINARY_KIND (expected example or bin)"
        ;;
esac

trap cleanup EXIT

resolve_ssh_key
ensure_private_key_is_registered
create_pod_if_needed
maybe_start_pod
load_pod_state

if [ "$DRY_RUN" -eq 1 ]; then
    load_pod_state
    log "dry-run pod id: $POD_ID"
    if [ -n "$REMOTE_DIR" ]; then
        log "dry-run remote dir: $REMOTE_DIR"
    fi
    if [ "$NO_COMPILE" -eq 1 ]; then
        log "dry-run command: <cached-binary> --backend cuda $(quote_cmd "${TOURNAMENT_ARGS[@]}")"
    else
        log "dry-run command: cargo build --release --features cuda --${BINARY_KIND} ${BINARY_NAME} && <cached-binary> --backend cuda $(quote_cmd "${TOURNAMENT_ARGS[@]}")"
    fi
    exit 0
fi

resolve_local_identity_context
prepare_run_preservation
wait_for_ssh
sync_worktree
bootstrap_remote
run_status=0
set +e
run_remote_tournament
run_status=$?
set -e

final_status="failure"
if [ "$run_status" -eq 0 ]; then
    final_status="success"
elif [ "$run_status" -eq 124 ]; then
    final_status="timeout"
fi
write_local_wrapper_manifest "$final_status" "$run_status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_LOCAL_BRANCH" "$RUN_LOCAL_BUILD_KEY" "$RUN_LOCAL_COMMIT_SHA" || true
preserve_remote_results || true

log "remote log saved at ${STATE_DIR}/logs/latest.log"
log "run artifacts preserved under ${RUN_RESULT_DIR}"
exit "$run_status"
