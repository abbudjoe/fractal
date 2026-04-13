#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
STEPS="${2:-512}"
EVAL_BATCHES="${3:-1}"
LABEL_PREFIX="${4:-v3a-python-path1-gdn-head2head}"

GPU_ID="${GPU_ID:-NVIDIA H100 80GB HBM3}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-16}"
WINDOW_STRIDE="${WINDOW_STRIDE:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOCAL_WINDOW="${LOCAL_WINDOW:-256}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-}"
JSONL_TRAIN_PATH="${JSONL_TRAIN_PATH:-${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl}"
JSONL_EVAL_PATH="${JSONL_EVAL_PATH:-${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl}"
RUN_P20_TRITON="${RUN_P20_TRITON:-1}"
RUN_GDN_TOPOLOGY="${RUN_GDN_TOPOLOGY:-1}"
RUN_FLA_GDN_TOPOLOGY="${RUN_FLA_GDN_TOPOLOGY:-0}"
GDN_TOPOLOGY_SCHEDULE="${GDN_TOPOLOGY_SCHEDULE:-RRRRRARRRRRS}"
MAMBA_REQUIREMENTS="${MAMBA_REQUIREMENTS:-scripts/requirements-v3a-python-mamba3.txt}"
FLA_REQUIREMENTS="${FLA_REQUIREMENTS:-scripts/requirements-v3a-python-gdn-fla.txt}"
GDN_TOPOLOGY_SIGNATURE="$(printf '%s' "${GDN_TOPOLOGY_SCHEDULE}" | tr '[:upper:]' '[:lower:]')"

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

build_common_args() {
  local install_mode="$1"
  local primitive_backend="$2"
  COMMON_ARGS=(
    --backend cuda
    --cuda-device "${CUDA_DEVICE}"
    --dtype "${DTYPE}"
    --env-kind "${install_mode}"
    --primitive-runtime-backend "${primitive_backend}"
    --seed "${SEED}"
    --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
    --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
    --local-window "${LOCAL_WINDOW}"
    --output table
  )
  if [[ -n "${BENCHMARK_PROFILE}" ]]; then
    COMMON_ARGS+=(--benchmark-profile "${BENCHMARK_PROFILE}" --full-train-pass --full-eval-pass)
  else
    COMMON_ARGS+=(
      --jsonl-train-path "${JSONL_TRAIN_PATH}"
      --jsonl-eval-path "${JSONL_EVAL_PATH}"
      --seq-len "${SEQ_LEN}"
      --window-stride "${WINDOW_STRIDE}"
      --batch-size "${BATCH_SIZE}"
      --steps "${STEPS}"
      --eval-batches "${EVAL_BATCHES}"
    )
  fi
}

run_lane() {
  local pod_name="$1"
  local install_mode="$2"
  local primitive_backend="$3"
  local lifecycle_flag="$4"
  local requirements_file="$5"
  local run_label="$6"
  shift 6

  build_common_args "${install_mode}" "${primitive_backend}"
  local expected_args=("$@" "${COMMON_ARGS[@]}" --run-label "${run_label}")
  if already_recorded "${run_label}" "${expected_args[@]}"; then
    echo "skip ${run_label}"
    return 0
  fi

  echo "run ${run_label}"
  "${WRAPPER}" \
    --pod-name "${pod_name}" \
    --gpu-id "${GPU_ID}" \
    --cloud-type "${CLOUD_TYPE}" \
    --binary-kind python \
    --binary-name scripts/v3a_python_path1.py \
    --python-requirements "${requirements_file}" \
    --python-install-mode "${install_mode}" \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    "${lifecycle_flag}" \
    -- \
    "${expected_args[@]}"
}

OFFICIAL_POD="${POD_NAME_OFFICIAL:-fractal-v3a-gdn-head2head-official-s${SEED}}"
TRITON_POD="${POD_NAME_TRITON:-fractal-v3a-gdn-head2head-triton-s${SEED}}"
FLA_POD="${POD_NAME_FLA:-fractal-v3a-gdn-head2head-fla-s${SEED}}"

run_lane \
  "${OFFICIAL_POD}" \
  official-mamba3 \
  torch \
  --keep-pod \
  "${MAMBA_REQUIREMENTS}" \
  "${LABEL_PREFIX}-s${SEED}-attention-only-env-official-mamba3" \
  --variant attention-only

run_lane \
  "${OFFICIAL_POD}" \
  official-mamba3 \
  torch \
  --keep-pod \
  "${MAMBA_REQUIREMENTS}" \
  "${LABEL_PREFIX}-s${SEED}-reference-ssm-mamba3-env-official-mamba3" \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile mamba3-siso-runtime

run_lane \
  "${OFFICIAL_POD}" \
  official-mamba3 \
  torch \
  --keep-pod \
  "${MAMBA_REQUIREMENTS}" \
  "${LABEL_PREFIX}-s${SEED}-reference-ssm-gated-deltanet-torch-env-official-mamba3" \
  --variant reference-ssm-hybrid \
  --reference-ssm-profile gated-deltanet-torch

if [[ "${RUN_GDN_TOPOLOGY}" == "1" ]]; then
  run_lane \
    "${OFFICIAL_POD}" \
    official-mamba3 \
    torch \
    --keep-pod \
    "${MAMBA_REQUIREMENTS}" \
    "${LABEL_PREFIX}-s${SEED}-reference-ssm-gated-deltanet-torch-topology-${GDN_TOPOLOGY_SIGNATURE}-env-official-mamba3" \
    --variant reference-ssm-hybrid \
    --reference-ssm-profile gated-deltanet-torch \
    --layer-schedule "${GDN_TOPOLOGY_SCHEDULE}"
fi

run_lane \
  "${OFFICIAL_POD}" \
  official-mamba3 \
  torch \
  "$(
    if [[ "${RUN_P20_TRITON}" == "1" ]]; then
      printf '%s' --stop-after-run
    else
      printf '%s' --keep-pod
    fi
  )" \
  "${MAMBA_REQUIREMENTS}" \
  "${LABEL_PREFIX}-s${SEED}-p20-gdn-role-env-official-mamba3" \
  --variant primitive-hybrid \
  --primitive-profile p2-0-gdn-role \
  --primitive-execution-profile runtime \
  --primitive-residual-profile scaled \
  --primitive-readout-profile direct \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile mamba-rms \
  --primitive-state-transform-profile dense

if [[ "${RUN_P20_TRITON}" == "1" ]]; then
  run_lane \
    "${TRITON_POD}" \
    primitive-triton \
    triton \
    --stop-after-run \
    "${MAMBA_REQUIREMENTS}" \
    "${LABEL_PREFIX}-s${SEED}-p20-block-diagonal-2-env-primitive-triton" \
    --variant primitive-hybrid \
    --primitive-profile p2-0 \
    --primitive-execution-profile runtime \
    --primitive-residual-profile scaled \
    --primitive-readout-profile projected \
    --primitive-norm-profile pre-norm-only \
    --primitive-wrapper-profile standard \
    --primitive-state-transform-profile block-diagonal-2
fi

if [[ "${RUN_FLA_GDN_TOPOLOGY}" == "1" ]]; then
  run_lane \
    "${FLA_POD}" \
    primitive-triton \
    torch \
    --stop-after-run \
    "${FLA_REQUIREMENTS}" \
    "${LABEL_PREFIX}-s${SEED}-reference-ssm-gated-deltanet-fla-topology-${GDN_TOPOLOGY_SIGNATURE}-env-primitive-triton" \
    --variant reference-ssm-hybrid \
    --reference-ssm-profile gated-deltanet-fla \
    --layer-schedule "${GDN_TOPOLOGY_SCHEDULE}"
fi

echo "completed Path 1 GDN head-to-head seed ${SEED}"
