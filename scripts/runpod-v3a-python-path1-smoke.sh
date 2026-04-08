#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
LABEL_PREFIX="${2:-v3a-python-path1-smoke}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
COMPILE_MODE="${COMPILE_MODE:-}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-official-mamba3}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
STEPS="${STEPS:-8}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"

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
  --seed "${SEED}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --benchmark-profile "${BENCHMARK_PROFILE}"
  --output table
)

LABEL_SUFFIX=""
if [[ "${PYTHON_INSTALL_MODE}" != "official-mamba3" ]]; then
  LABEL_SUFFIX="${LABEL_SUFFIX}-env-${PYTHON_INSTALL_MODE}"
fi
if [[ -n "${COMPILE_MODE}" ]]; then
  COMMON_ARGS+=(--compile-mode "${COMPILE_MODE}")
  LABEL_SUFFIX="${LABEL_SUFFIX}-compile-${COMPILE_MODE}"
fi

if [[ "${BENCHMARK_PROFILE}" == "cuda-faithful-small-v1" ]]; then
  COMMON_ARGS+=(--full-train-pass --full-eval-pass)
else
  COMMON_ARGS+=(--steps "${STEPS}" --eval-batches "${EVAL_BATCHES}")
fi

run_python_path1() {
  local variant="$1"
  shift
  local run_label="${LABEL_PREFIX}-s${SEED}-${variant}${LABEL_SUFFIX}"
  local expected_args=("${COMMON_ARGS[@]}" --variant "${variant}" "$@" --run-label "${run_label}")
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
    --stop-after-run \
    -- \
    "${expected_args[@]}"
}

if [[ "${PYTHON_INSTALL_MODE}" == "compile-safe" ]]; then
  echo "compile-safe mode does not provide official mamba; use the compile-specific runner instead of smoke"
  exit 1
fi

run_python_path1 attention-only
run_python_path1 reference-ssm-hybrid --reference-ssm-profile mamba3-siso-runtime
run_python_path1 primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-execution-profile runtime \
  --primitive-residual-profile scaled \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard

echo "completed runpod v3a python path1 smoke seed ${SEED}"
