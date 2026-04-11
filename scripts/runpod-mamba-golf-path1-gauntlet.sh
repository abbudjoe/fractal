#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(42)
fi

LABEL_PREFIX="${LABEL_PREFIX:-mamba-golf-path1-gauntlet}"
POD_NAME_PREFIX="${POD_NAME_PREFIX:-fractal-mamba-golf}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"
JSONL_TRAIN_PATH="${JSONL_TRAIN_PATH:-}"
JSONL_EVAL_PATH="${JSONL_EVAL_PATH:-}"
CORPUS_NAME="${CORPUS_NAME:-}"
CORPUS_TEXT_FIELD="${CORPUS_TEXT_FIELD:-text}"
SEQ_LENS_SPEC="${SEQ_LENS:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WINDOW_STRIDE="${WINDOW_STRIDE:-}"
TRAIN_STEPS="${TRAIN_STEPS:-8}"
EVAL_BATCHES="${EVAL_BATCHES:-2}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-3}"
FULL_TRAIN_PASS="${FULL_TRAIN_PASS:-1}"
FULL_EVAL_PASS="${FULL_EVAL_PASS:-1}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
MAMBA_PROFILE="${MAMBA_PROFILE:-mamba3-siso-runtime}"
P20_STATE_TRANSFORM_PROFILE="${P20_STATE_TRANSFORM_PROFILE:-block-diagonal-2}"

RUN_ATTENTION_OFFICIAL="${RUN_ATTENTION_OFFICIAL:-1}"
RUN_MAMBA="${RUN_MAMBA:-1}"
RUN_ATTENTION_TRITON="${RUN_ATTENTION_TRITON:-1}"
RUN_P20="${RUN_P20:-1}"

IFS=' ' read -r -a SEQ_LENS <<< "${SEQ_LENS_SPEC}"

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
  local lifecycle_flag="$1"
  local pod_name="$2"
  local python_install_mode="$3"
  local run_label="$4"
  shift 4

  local seq_len="${SEQ_LEN}"
  local window_stride="${WINDOW_STRIDE:-${SEQ_LEN}}"
  local corpus=()
  local passes=()
  if [[ -n "${BENCHMARK_PROFILE}" ]]; then
    corpus=(--benchmark-profile "${BENCHMARK_PROFILE}")
  else
    if [[ -z "${JSONL_TRAIN_PATH}" || -z "${JSONL_EVAL_PATH}" ]]; then
      echo "BENCHMARK_PROFILE is empty, so JSONL_TRAIN_PATH and JSONL_EVAL_PATH are required" >&2
      exit 1
    fi
    corpus=(--jsonl-train-path "${JSONL_TRAIN_PATH}" --jsonl-eval-path "${JSONL_EVAL_PATH}")
    if [[ -n "${CORPUS_NAME}" ]]; then
      corpus+=(--corpus-name "${CORPUS_NAME}")
    fi
    corpus+=(--corpus-text-field "${CORPUS_TEXT_FIELD}")
  fi
  if [[ "${FULL_TRAIN_PASS}" == "1" ]]; then
    passes+=(--full-train-pass)
  fi
  if [[ "${FULL_EVAL_PASS}" == "1" ]]; then
    passes+=(--full-eval-pass)
  fi

  local expected_args=(
    "$@"
    --backend cuda
    --cuda-device "${CUDA_DEVICE}"
    --dtype "${DTYPE}"
    --seed "${SEED}"
    --seq-len "${seq_len}"
    --window-stride "${window_stride}"
    --batch-size "${BATCH_SIZE}"
    --steps "${TRAIN_STEPS}"
    --eval-batches "${EVAL_BATCHES}"
    --learning-rate "${LEARNING_RATE}"
    --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
    --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
    "${corpus[@]}"
    --output table
    "${passes[@]}"
    --run-label "${run_label}"
  )

  if already_recorded "${run_label}" "${expected_args[@]}"; then
    echo "skip ${run_label}"
    return 0
  fi

  echo "run ${run_label}"
  local wrapper_args=(
    --pod-name "${pod_name}"
    --gpu-id "${GPU_ID}"
    --binary-kind python
    --binary-name scripts/v3a_python_path1.py
    --python-requirements scripts/requirements-v3a-python-mamba3.txt
    --python-install-mode "${python_install_mode}"
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}"
    "${lifecycle_flag}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    wrapper_args+=(--dry-run)
  fi
  "${WRAPPER}" \
    "${wrapper_args[@]}" \
    -- \
    "${expected_args[@]}"
}

for SEED in "${SEEDS[@]}"; do
  for SEQ_LEN in "${SEQ_LENS[@]}"; do
    official_pod="${POD_NAME_PREFIX}-official-s${SEED}-q${SEQ_LEN}"
    triton_pod="${POD_NAME_PREFIX}-triton-s${SEED}-q${SEQ_LEN}"

    if [[ "${RUN_ATTENTION_OFFICIAL}" == "1" ]]; then
      lifecycle="--keep-pod"
      if [[ "${RUN_MAMBA}" != "1" ]]; then
        lifecycle="--stop-after-run"
      fi
      run_lane \
        "${lifecycle}" \
        "${official_pod}" \
        "official-mamba3" \
        "${LABEL_PREFIX}-s${SEED}-seq${SEQ_LEN}-attention-env-official-mamba3" \
        --env-kind official-mamba3 \
        --primitive-runtime-backend torch \
        --variant attention-only
    fi

    if [[ "${RUN_MAMBA}" == "1" ]]; then
      run_lane \
        --stop-after-run \
        "${official_pod}" \
        "official-mamba3" \
        "${LABEL_PREFIX}-s${SEED}-seq${SEQ_LEN}-mamba-${MAMBA_PROFILE}" \
        --env-kind official-mamba3 \
        --primitive-runtime-backend torch \
        --variant reference-ssm-hybrid \
        --reference-ssm-profile "${MAMBA_PROFILE}"
    fi

    if [[ "${RUN_ATTENTION_TRITON}" == "1" ]]; then
      lifecycle="--keep-pod"
      if [[ "${RUN_P20}" != "1" ]]; then
        lifecycle="--stop-after-run"
      fi
      run_lane \
        "${lifecycle}" \
        "${triton_pod}" \
        "primitive-triton" \
        "${LABEL_PREFIX}-s${SEED}-seq${SEQ_LEN}-attention-env-primitive-triton" \
        --env-kind primitive-triton \
        --primitive-runtime-backend triton \
        --variant attention-only
    fi

    if [[ "${RUN_P20}" == "1" ]]; then
      run_lane \
        --stop-after-run \
        "${triton_pod}" \
        "primitive-triton" \
        "${LABEL_PREFIX}-s${SEED}-seq${SEQ_LEN}-p20-triton-${P20_STATE_TRANSFORM_PROFILE}" \
        --env-kind primitive-triton \
        --primitive-runtime-backend triton \
        --variant primitive-hybrid \
        --primitive-profile p2-0 \
        --primitive-execution-profile runtime \
        --primitive-residual-profile scaled \
        --primitive-readout-profile projected \
        --primitive-norm-profile pre-norm-only \
        --primitive-wrapper-profile standard \
        --primitive-state-transform-profile "${P20_STATE_TRANSFORM_PROFILE}"
    fi
  done
done

echo "completed Mamba Golf Path 1 gauntlet for seeds: ${SEEDS[*]} seq_lens: ${SEQ_LENS[*]}"
