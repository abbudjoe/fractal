#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/runpod-tournament.sh [wrapper options] -- [tournament args]

Create or reuse a Runpod pod, sync the current repo snapshot, bootstrap Rust/CMake if
needed, and run the tournament example with the CUDA backend.

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
  --stop-after-run                Always stop the pod after the command finishes.
  --keep-pod                      Never stop the pod automatically.
  --dry-run                       Print the resolved actions without creating or running anything.
  --help                          Show this help.

Notes:
  - If no existing pod matches, the wrapper creates one and requires --gpu-id.
  - Newly created pods are stopped automatically after the run unless --keep-pod is set.
  - Existing pods are left running unless --stop-after-run is set.
  - Tournament arguments after "--" are passed to:
      cargo run --release --features cuda --example tournament -- --backend cuda ...

Examples:
  scripts/runpod-tournament.sh \
    --gpu-id "NVIDIA GeForce RTX 4090" \
    -- --preset research-medium

  scripts/runpod-tournament.sh \
    --pod-name fractal-a100 \
    --gpu-id "NVIDIA A100 80GB PCIe" \
    --keep-pod \
    -- --sequence first-run
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

for path in sorted(glob.glob(str(pathlib.Path.home() / ".ssh" / "*.pub"))):
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
    SSH_PORT="$(printf '%s' "$pod_json" | python3 -c '
import json
import sys

pod = json.load(sys.stdin)
port_mappings = pod.get("portMappings") or {}
value = port_mappings.get("22")
if value is None:
    value = port_mappings.get(22)
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
    COPYFILE_DISABLE=1 tar \
        --exclude=.git \
        --exclude=target \
        --exclude=.DS_Store \
        -C "$REPO_ROOT" \
        -cf - . | "${SSH_BASE[@]}" "rm -rf ${remote_dir_q} && mkdir -p ${remote_dir_q} && tar -xf - -C ${remote_dir_q}"
}

bootstrap_remote() {
    log "bootstrapping remote toolchain"
    "${SSH_BASE[@]}" bash -s -- "$REMOTE_DIR" "$STATE_DIR" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
state_dir="$2"
export DEBIAN_FRONTEND=noninteractive

need_apt=0
for cmd in curl git cmake pkg-config g++; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        need_apt=1
    fi
done

if [ "$need_apt" -eq 1 ]; then
    apt-get update
    apt-get install -y build-essential cmake curl git pkg-config libssl-dev
fi

if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is not installed in this image; use an official CUDA-enabled Runpod template" >&2
    exit 1
fi

mkdir -p "$remote_dir" "$state_dir/logs" "$state_dir/target"
REMOTE
}

run_remote_tournament() {
    local local_branch
    local local_commit
    local_branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
    local_commit="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

    log "running remote CUDA tournament"
    "${SSH_BASE[@]}" bash -s -- \
        "$REMOTE_DIR" \
        "$STATE_DIR" \
        "$local_branch" \
        "$local_commit" \
        "${TOURNAMENT_ARGS[@]}" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
state_dir="$2"
local_branch="$3"
local_commit="$4"
shift 4

if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.cargo/env"
fi

export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="$state_dir/target"

mkdir -p "$state_dir/logs"
printf 'branch=%s\ncommit=%s\nrun_at=%s\n' \
    "$local_branch" \
    "$local_commit" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$state_dir/last-sync.txt"

cd "$remote_dir"
cargo run --release --features cuda --example tournament -- --backend cuda "$@" \
    2>&1 | tee "$state_dir/logs/latest.log"
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
STOP_MODE="auto"
DRY_RUN=0
CREATED_POD=0
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

trap cleanup EXIT

resolve_ssh_key
ensure_private_key_is_registered
create_pod_if_needed
maybe_start_pod

if [ "$DRY_RUN" -eq 1 ]; then
    load_pod_state
    log "dry-run pod id: $POD_ID"
    if [ -n "$REMOTE_DIR" ]; then
        log "dry-run remote dir: $REMOTE_DIR"
    fi
    log "dry-run tournament command: cargo run --release --features cuda --example tournament -- --backend cuda $(quote_cmd "${TOURNAMENT_ARGS[@]}")"
    exit 0
fi

wait_for_ssh
sync_worktree
bootstrap_remote
run_remote_tournament

log "remote log saved at ${STATE_DIR}/logs/latest.log"
