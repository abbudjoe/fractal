#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
LABEL_PREFIX="${LABEL_PREFIX:-v3a-python-path1-schedule-sweep}"

POD_NAME_PREFIX="${POD_NAME_PREFIX:-fractal-v3a-schedule-sweep}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-primitive-triton}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-triton}"
COMPILE_MODE="${COMPILE_MODE:-}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
PRIMITIVE_PROFILE="${PRIMITIVE_PROFILE:-p2-0}"
PRIMITIVE_EXECUTION_PROFILE="${PRIMITIVE_EXECUTION_PROFILE:-runtime}"
PRIMITIVE_RESIDUAL_PROFILE="${PRIMITIVE_RESIDUAL_PROFILE:-scaled}"
PRIMITIVE_READOUT_PROFILE="${PRIMITIVE_READOUT_PROFILE:-projected}"
PRIMITIVE_NORM_PROFILE="${PRIMITIVE_NORM_PROFILE:-pre-norm-only}"
PRIMITIVE_WRAPPER_PROFILE="${PRIMITIVE_WRAPPER_PROFILE:-standard}"
PRIMITIVE_STATE_TRANSFORM_PROFILE="${PRIMITIVE_STATE_TRANSFORM_PROFILE:-block-diagonal-2}"
SCHEDULES="${SCHEDULES:-AAAAAAAAAAA,PAAAAAAAAAA,AAAPAAAAAAA,AAAAAPAAAAA,AAAAAAAAPAA,AAAAAAAAAAP,AAAAPPAAAAA,AAAAPAPAAAA,AAAAAPAAAAP}"

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

normalize_schedule() {
  printf '%s' "$1" | tr '[:lower:]' '[:upper:]' | tr -d ' ,-_'
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

if [[ -n "${COMPILE_MODE}" ]]; then
  COMMON_ARGS+=(--compile-mode "${COMPILE_MODE}")
fi

run_lane() {
  local raw_schedule="$1"
  local schedule
  schedule="$(normalize_schedule "${raw_schedule}")"
  if [[ -z "${schedule}" ]]; then
    echo "empty schedule entry"
    exit 1
  fi

  local variant_args=()
  local schedule_kind="attention-only"
  if [[ "${schedule}" == *P* && "${schedule}" == *R* ]]; then
    echo "schedule ${schedule} mixes P and R roles; unsupported in this sweep"
    exit 1
  elif [[ "${schedule}" == *P* ]]; then
    schedule_kind="primitive-hybrid"
    variant_args=(
      --variant primitive-hybrid
      --primitive-profile "${PRIMITIVE_PROFILE}"
      --primitive-execution-profile "${PRIMITIVE_EXECUTION_PROFILE}"
      --primitive-residual-profile "${PRIMITIVE_RESIDUAL_PROFILE}"
      --primitive-readout-profile "${PRIMITIVE_READOUT_PROFILE}"
      --primitive-norm-profile "${PRIMITIVE_NORM_PROFILE}"
      --primitive-wrapper-profile "${PRIMITIVE_WRAPPER_PROFILE}"
      --primitive-state-transform-profile "${PRIMITIVE_STATE_TRANSFORM_PROFILE}"
    )
  elif [[ "${schedule}" == *R* ]]; then
    echo "reference-SSM schedule sweeps are not wired in this runner yet"
    exit 1
  else
    variant_args=(--variant attention-only)
  fi

  local run_label="${LABEL_PREFIX}-s${SEED}-${schedule_kind}-${PRIMITIVE_PROFILE}-schedule-${schedule}"
  local pod_name="${POD_NAME_PREFIX}-${PRIMITIVE_PROFILE}-${schedule,,}"
  local expected_args=(
    "${variant_args[@]}"
    "${COMMON_ARGS[@]}"
    --layer-schedule "${schedule}"
    --run-label "${run_label}"
  )

  if already_recorded "${run_label}" "${expected_args[@]}"; then
    echo "skip ${run_label}"
    return 0
  fi

  echo "run ${run_label}"
  "${WRAPPER}" \
    --pod-name "${pod_name}" \
    --gpu-id "${GPU_ID}" \
    --binary-kind python \
    --binary-name scripts/v3a_python_path1.py \
    --python-requirements scripts/requirements-v3a-python-mamba3.txt \
    --python-install-mode "${PYTHON_INSTALL_MODE}" \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    --stop-after-run \
    -- \
    "${expected_args[@]}"
}

IFS=',' read -r -a schedules <<< "${SCHEDULES}"
for schedule in "${schedules[@]}"; do
  run_lane "${schedule}"
done

echo "completed path1 schedule sweep for ${PRIMITIVE_PROFILE} seed ${SEED}"
