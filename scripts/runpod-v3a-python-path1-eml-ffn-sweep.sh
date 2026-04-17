#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
LABEL_PREFIX="${2:-v3a-python-path1-eml-ffn-sweep}"

POD_NAME="${POD_NAME:-fractal-v3a-eml-ffn}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-requirements-only}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-torch}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
SEQ_LEN="${SEQ_LEN:-64}"
WINDOW_STRIDE="${WINDOW_STRIDE:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"

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
  --seq-len "${SEQ_LEN}"
  --window-stride "${WINDOW_STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --output table
  --full-train-pass
  --full-eval-pass
)

LABEL_SUFFIX="-s${SEED}-q${SEQ_LEN}-b${BATCH_SIZE}-env-${PYTHON_INSTALL_MODE}-primitive-${PRIMITIVE_RUNTIME_BACKEND}"

run_lane() {
  local lifecycle_flag="$1"
  shift
  local lane_slug="$1"
  shift
  local run_label="${LABEL_PREFIX}-${lane_slug}${LABEL_SUFFIX}"
  local expected_args=("${COMMON_ARGS[@]}" "$@" --run-label "${run_label}")
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

if [[ "${PRIMITIVE_RUNTIME_BACKEND}" != "torch" ]]; then
  echo "EML FFN sweep expects PRIMITIVE_RUNTIME_BACKEND=torch"
  exit 1
fi

run_lane --keep-pod baseline \
  --variant attention-only

run_lane --keep-pod initial-eml-tree-all \
  --variant attention-only \
  --feed-forward-profile eml-tree \
  --eml-slot-count 8 \
  --eml-tree-depth 3

run_lane --keep-pod initial-gated-eml-all \
  --variant attention-only \
  --feed-forward-profile mlp-eml-gated \
  --eml-slot-count 8 \
  --eml-tree-depth 3

run_lane --keep-pod surgical-gated-eml-layer4 \
  --variant attention-only \
  --feed-forward-profile mlp-eml-gated \
  --feed-forward-layer-indices 4 \
  --eml-slot-count 4 \
  --eml-tree-depth 2

run_lane --keep-pod surgical-gated-eml-layers3-4 \
  --variant attention-only \
  --feed-forward-profile mlp-eml-gated \
  --feed-forward-layer-indices 3,4 \
  --eml-slot-count 4 \
  --eml-tree-depth 2

run_lane --stop-after-run surgical-eml-tree-layers3-4 \
  --variant attention-only \
  --feed-forward-profile eml-tree \
  --feed-forward-layer-indices 3,4 \
  --eml-slot-count 4 \
  --eml-tree-depth 2

echo "completed runpod v3a python path1 EML FFN sweep seed ${SEED}"
