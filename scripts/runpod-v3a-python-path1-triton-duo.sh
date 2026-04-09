#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
LABEL_PREFIX="${2:-v3a-python-path1-triton-duo}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-primitive-triton}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-torch}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
PRIMITIVE_STATE_TRANSFORM_PROFILE="${PRIMITIVE_STATE_TRANSFORM_PROFILE:-dense}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"

if [[ "${PYTHON_INSTALL_MODE}" != "primitive-triton" ]]; then
  echo "triton duo expects PYTHON_INSTALL_MODE=primitive-triton"
  exit 1
fi

already_recorded() {
  local run_label="$1"
  shift
  if [[ ! -d "${LOCAL_RESULTS_ROOT}" ]]; then
    return 1
  fi
  python3 - "${LOCAL_RESULTS_ROOT}" "${run_label}" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = sys.argv[2]
expected_args = sys.argv[3:]

for manifest_path in root.glob("**/metadata/wrapper-manifest.json"):
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        continue
    if manifest.get("status") != "success":
        continue
    runtime = manifest.get("runtime") or {}
    command_args = runtime.get("command_args") or []
    if command_args != expected_args:
        continue
    for index, value in enumerate(command_args):
        if value == "--run-label" and index + 1 < len(command_args) and command_args[index + 1] == target:
            sys.exit(0)
sys.exit(1)
PY
}

COMMON_ARGS=(
  --backend cuda
  --cuda-device "${CUDA_DEVICE}"
  --dtype "${DTYPE}"
  --env-kind "${PYTHON_INSTALL_MODE}"
  --primitive-runtime-backend "${PRIMITIVE_RUNTIME_BACKEND}"
  --seed "${SEED}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --benchmark-profile "${BENCHMARK_PROFILE}"
  --output table
  --full-train-pass
  --full-eval-pass
)

LABEL_SUFFIX="-env-${PYTHON_INSTALL_MODE}-primitive-${PRIMITIVE_RUNTIME_BACKEND}-state-${PRIMITIVE_STATE_TRANSFORM_PROFILE}"

run_lane() {
  local lifecycle_flag="$1"
  shift
  local run_label="$1"
  shift

  local expected_args=("$@" "${COMMON_ARGS[@]}" --run-label "${run_label}")
  if already_recorded "${run_label}" "${expected_args[@]}"; then
    echo "skip ${run_label}"
    return 0
  fi

  echo "run ${run_label}"
  "${WRAPPER}" \
    --pod-name "${POD_NAME}" \
    --gpu-id "${GPU_ID}" \
    --binary-kind python \
    --binary-name scripts/v3a_python_path1.py \
    --python-requirements scripts/requirements-v3a-python-mamba3.txt \
    --python-install-mode "${PYTHON_INSTALL_MODE}" \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    "${lifecycle_flag}" \
    -- \
    "${expected_args[@]}"
}

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-attention-only${LABEL_SUFFIX}" \
  --variant attention-only

run_lane \
  --stop-after-run \
  "${LABEL_PREFIX}-s${SEED}-p2-0-scaled-projected-pre-norm-only-standard-runtime${LABEL_SUFFIX}" \
  --variant primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-execution-profile runtime \
  --primitive-residual-profile scaled \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard \
  --primitive-state-transform-profile "${PRIMITIVE_STATE_TRANSFORM_PROFILE}"

echo "completed runpod v3a python path1 triton duo seed ${SEED}"
