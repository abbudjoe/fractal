#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

# Use the already-frozen seed 42 as the baseline anchor and sweep four fresh seeds by default.
if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(7 13 123 256)
fi

LABEL_PREFIX="${LABEL_PREFIX:-v3a-python-path1-head2head}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
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

run_lane() {
  local pod_name="$1"
  local python_install_mode="$2"
  local run_label="$3"
  shift 3

  local expected_args=("$@" --backend cuda --cuda-device "${CUDA_DEVICE}" --dtype "${DTYPE}" --seed "${SEED}" --warmup-eval-batches "${WARMUP_EVAL_BATCHES}" --warmup-train-steps "${WARMUP_TRAIN_STEPS}" --benchmark-profile "${BENCHMARK_PROFILE}" --output table --full-train-pass --full-eval-pass --run-label "${run_label}")

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
    --python-install-mode "${python_install_mode}" \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    --stop-after-run \
    -- \
    "${expected_args[@]}"
}

for SEED in "${SEEDS[@]}"; do
  run_lane \
    "fractal-v3a-head2head-mamba-s${SEED}" \
    "official-mamba3" \
    "${LABEL_PREFIX}-s${SEED}-reference-ssm-hybrid-env-official-mamba3" \
    --env-kind official-mamba3 \
    --primitive-runtime-backend torch \
    --variant reference-ssm-hybrid \
    --reference-ssm-profile mamba3-siso-runtime

  run_lane \
    "fractal-v3a-head2head-p20-s${SEED}" \
    "primitive-triton" \
    "${LABEL_PREFIX}-s${SEED}-p20-block-diagonal-2-env-primitive-triton" \
    --env-kind primitive-triton \
    --primitive-runtime-backend triton \
    --variant primitive-hybrid \
    --primitive-profile p2-0 \
    --primitive-execution-profile runtime \
    --primitive-residual-profile scaled \
    --primitive-readout-profile projected \
    --primitive-norm-profile pre-norm-only \
    --primitive-wrapper-profile standard \
    --primitive-state-transform-profile block-diagonal-2
done

echo "completed path1 head-to-head seed sweep for seeds: ${SEEDS[*]}"
