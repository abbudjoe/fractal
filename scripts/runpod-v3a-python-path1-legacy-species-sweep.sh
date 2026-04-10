#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
LABEL_PREFIX="${2:-v3a-python-path1-legacy-species-sweep}"

POD_NAME_PREFIX="${POD_NAME_PREFIX:-fractal-v3a-legacy-species}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-compile-safe}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-torch}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
PRIMITIVE_PROFILES="${PRIMITIVE_PROFILES:-p1,p1-fractal-hybrid,p1-fractal-hybrid-composite,p1-fractal-hybrid-dyn-gate,p2-mandelbrot,p3-hierarchical,b1-fractal-gated,b2-stable-hierarchical,b3-fractal-hierarchical,b4-universal,ifs,generalized-mobius,logistic-chaotic-map,julia-recursive-escape,mandelbox-recursive}"

if [[ "${PYTHON_INSTALL_MODE}" != "compile-safe" ]]; then
  echo "legacy species sweep expects PYTHON_INSTALL_MODE=compile-safe"
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
  --compile-mode "${COMPILE_MODE}"
  --primitive-runtime-backend "${PRIMITIVE_RUNTIME_BACKEND}"
  --seed "${SEED}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --benchmark-profile "${BENCHMARK_PROFILE}"
  --output table
  --full-train-pass
  --full-eval-pass
  --variant primitive-hybrid
  --primitive-execution-profile runtime
  --primitive-residual-profile plain
  --primitive-readout-profile direct
  --primitive-norm-profile pre-norm-only
  --primitive-wrapper-profile standard
  --primitive-state-transform-profile dense
)

run_lane() {
  local primitive_profile="$1"
  shift

  local run_label="${LABEL_PREFIX}-s${SEED}-${primitive_profile}-plain-direct-pre-norm-only-standard-dense"
  local pod_name="${POD_NAME_PREFIX}-${primitive_profile}"
  local expected_args=(
    --primitive-profile "${primitive_profile}"
    "${COMMON_ARGS[@]}"
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

IFS=',' read -r -a primitive_profiles <<< "${PRIMITIVE_PROFILES}"
for primitive_profile in "${primitive_profiles[@]}"; do
  run_lane "${primitive_profile}"
done

echo "completed runpod v3a python path1 legacy species sweep seed ${SEED}"
