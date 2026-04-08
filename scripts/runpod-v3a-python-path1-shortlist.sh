#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
STEPS="${2:-16}"
EVAL_BATCHES="${3:-4}"
LABEL_PREFIX="${4:-v3a-python-path1-shortlist}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
RUN_P20_PLAIN="${RUN_P20_PLAIN:-1}"

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
  --seed "${SEED}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --benchmark-profile "${BENCHMARK_PROFILE}"
  --output table
)

if [[ "${BENCHMARK_PROFILE}" == "cuda-faithful-small-v1" ]]; then
  COMMON_ARGS+=(--full-train-pass --full-eval-pass)
else
  COMMON_ARGS+=(--steps "${STEPS}" --eval-batches "${EVAL_BATCHES}")
fi

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
    --python-install-mode official-mamba3 \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    "${lifecycle_flag}" \
    -- \
    "${expected_args[@]}"
}

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-attention-only" \
  --variant attention-only

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-reference-ssm-hybrid" \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile mamba3-siso-runtime

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-p2-incumbent-runtime" \
  --variant primitive-hybrid \
  --primitive-profile p2 \
  --primitive-execution-profile runtime

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-p2-3-gated-projected-norm-residual-renorm-standard-runtime" \
  --variant primitive-hybrid \
  --primitive-profile p2-3 \
  --primitive-execution-profile runtime \
  --primitive-residual-profile gated \
  --primitive-readout-profile projected-norm \
  --primitive-norm-profile residual-renorm \
  --primitive-wrapper-profile standard

run_lane \
  "$(
    if [[ "${RUN_P20_PLAIN}" == "1" ]]; then
      printf '%s' --keep-pod
    else
      printf '%s' --stop-after-run
    fi
  )" \
  "${LABEL_PREFIX}-s${SEED}-p2-0-scaled-projected-pre-norm-only-standard-runtime" \
  --variant primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-execution-profile runtime \
  --primitive-residual-profile scaled \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard

if [[ "${RUN_P20_PLAIN}" == "1" ]]; then
  run_lane \
    --stop-after-run \
    "${LABEL_PREFIX}-s${SEED}-p2-0-plain-projected-pre-norm-only-standard-runtime" \
    --variant primitive-hybrid \
    --primitive-profile p2-0 \
    --primitive-execution-profile runtime \
    --primitive-residual-profile plain \
    --primitive-readout-profile projected \
    --primitive-norm-profile pre-norm-only \
    --primitive-wrapper-profile standard
fi

echo "completed runpod v3a python path1 shortlist seed ${SEED}"
