#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"

SEED="${1:-42}"
STEPS="${2:-128}"
EVAL_BATCHES="${3:-1}"
LABEL_PREFIX="${4:-v3a-python-path1-gdn-p20-diagnostic}"

GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 5090}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
POD_NAME="${POD_NAME:-fractal-v3a-gdn-p20-diagnostic-5090-s${SEED}}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-16}"
WINDOW_STRIDE="${WINDOW_STRIDE:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOCAL_WINDOW="${LOCAL_WINDOW:-256}"
SCHEDULE="${SCHEDULE:-RRRRRARRRRRS}"
SCHEDULE_SIGNATURE="$(printf '%s' "${SCHEDULE}" | tr '[:upper:]' '[:lower:]')"
REQUIREMENTS="${REQUIREMENTS:-scripts/requirements-v3a-python-mamba3.txt}"
INSTALL_MODE="${INSTALL_MODE:-primitive-triton}"
TRAIN_PATH="${TRAIN_PATH:-${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl}"
EVAL_PATH="${EVAL_PATH:-${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl}"

PROFILES=(
  gated-deltanet-torch
  p20-torch
  gated-deltanet-p20-torch
  gated-deltanet-p20-thin-torch
)

for index in "${!PROFILES[@]}"; do
  profile="${PROFILES[$index]}"
  lifecycle_flag="--keep-pod"
  if [[ "$index" == "$((${#PROFILES[@]} - 1))" ]]; then
    lifecycle_flag="--stop-after-run"
  fi
  run_label="${LABEL_PREFIX}-s${SEED}-${profile}-schedule-${SCHEDULE_SIGNATURE}"
  echo "run ${run_label}"
  "${WRAPPER}" \
    --pod-name "${POD_NAME}" \
    --gpu-id "${GPU_ID}" \
    --cloud-type "${CLOUD_TYPE}" \
    --binary-kind python \
    --binary-name scripts/v3a_python_path1.py \
    --python-requirements "${REQUIREMENTS}" \
    --python-install-mode "${INSTALL_MODE}" \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    "${lifecycle_flag}" \
    -- \
    --variant reference-ssm-hybrid \
    --reference-ssm-profile "${profile}" \
    --layer-schedule "${SCHEDULE}" \
    --backend cuda \
    --cuda-device 0 \
    --dtype "${DTYPE}" \
    --env-kind "${INSTALL_MODE}" \
    --primitive-runtime-backend torch \
    --seed "${SEED}" \
    --warmup-eval-batches 1 \
    --warmup-train-steps 1 \
    --local-window "${LOCAL_WINDOW}" \
    --jsonl-train-path "${TRAIN_PATH}" \
    --jsonl-eval-path "${EVAL_PATH}" \
    --seq-len "${SEQ_LEN}" \
    --window-stride "${WINDOW_STRIDE}" \
    --batch-size "${BATCH_SIZE}" \
    --steps "${STEPS}" \
    --eval-batches "${EVAL_BATCHES}" \
    --run-label "${run_label}" \
    --output table
done

echo "completed GDN/P20 diagnostic seed ${SEED}"
