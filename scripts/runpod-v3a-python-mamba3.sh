#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
STEPS="${2:-16}"
EVAL_BATCHES="${3:-4}"
LABEL_PREFIX="${4:-v3a-python-mamba3}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-28800}"
FULL_TRAIN_PASS="${FULL_TRAIN_PASS:-0}"
FULL_EVAL_PASS="${FULL_EVAL_PASS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-}"

CORPUS_TRAIN_JSONL="${CORPUS_TRAIN_JSONL:-}"
CORPUS_EVAL_JSONL="${CORPUS_EVAL_JSONL:-}"
CORPUS_NAME="${CORPUS_NAME:-}"
CORPUS_TEXT_FIELD="${CORPUS_TEXT_FIELD:-text}"

if [[ "${BENCHMARK_PROFILE}" == "cuda-faithful-small-v1" ]]; then
  if [[ -z "${CORPUS_TRAIN_JSONL}" && -z "${CORPUS_EVAL_JSONL}" ]]; then
    CORPUS_TRAIN_JSONL="${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl"
    CORPUS_EVAL_JSONL="${REPO_ROOT}/experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl"
    CORPUS_NAME="${CORPUS_NAME:-fineweb-stage0-local-bench-9row-v1}"
  fi
  FULL_TRAIN_PASS=1
  FULL_EVAL_PASS=1
fi

already_recorded() {
  local run_label="$1"
  if [[ ! -d "${LOCAL_RESULTS_ROOT}" ]]; then
    return 1
  fi
  python3 - "${LOCAL_RESULTS_ROOT}" "${run_label}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = sys.argv[2]

for manifest_path in root.glob("**/metadata/wrapper-manifest.json"):
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        continue
    if manifest.get("status") != "success":
        continue
    runtime = manifest.get("runtime") or {}
    command_args = runtime.get("command_args") or []
    for index, value in enumerate(command_args):
        if value == "--run-label" and index + 1 < len(command_args) and command_args[index + 1] == target:
            sys.exit(0)
sys.exit(1)
PY
}

COMMON_ARGS=(
  --seed "${SEED}"
  --cuda-device "${CUDA_DEVICE}"
  --dtype "${DTYPE}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --output table
)

if [[ -n "${BENCHMARK_PROFILE}" ]]; then
  COMMON_ARGS+=(--benchmark-name "${BENCHMARK_PROFILE}")
fi

if [[ "${FULL_TRAIN_PASS}" == "1" ]]; then
  COMMON_ARGS+=(--full-train-pass)
else
  COMMON_ARGS+=(--steps "${STEPS}")
fi

if [[ "${FULL_EVAL_PASS}" == "1" ]]; then
  COMMON_ARGS+=(--full-eval-pass)
else
  COMMON_ARGS+=(--eval-batches "${EVAL_BATCHES}")
fi

if [[ -n "${CORPUS_TRAIN_JSONL}" || -n "${CORPUS_EVAL_JSONL}" ]]; then
  if [[ -z "${CORPUS_TRAIN_JSONL}" || -z "${CORPUS_EVAL_JSONL}" ]]; then
    echo "CORPUS_TRAIN_JSONL and CORPUS_EVAL_JSONL must be set together" >&2
    exit 1
  fi
  COMMON_ARGS+=(
    --jsonl-train-path "${CORPUS_TRAIN_JSONL}"
    --jsonl-eval-path "${CORPUS_EVAL_JSONL}"
  )
  if [[ -n "${CORPUS_NAME}" ]]; then
    COMMON_ARGS+=(--corpus-name "${CORPUS_NAME}")
  fi
  if [[ "${CORPUS_TEXT_FIELD}" != "text" ]]; then
    COMMON_ARGS+=(--corpus-text-field "${CORPUS_TEXT_FIELD}")
  fi
else
  echo "CORPUS_TRAIN_JSONL and CORPUS_EVAL_JSONL are required for the python Mamba3 runner" >&2
  exit 1
fi

run_label="${LABEL_PREFIX}-s${SEED}-python-reference-ssm-hybrid"
if already_recorded "${run_label}"; then
  echo "skip ${run_label}"
  exit 0
fi

echo "run ${run_label}"
"${WRAPPER}" \
  --pod-name "${POD_NAME}" \
  --gpu-id "${GPU_ID}" \
  --binary-kind python \
  --binary-name scripts/v3a_pytorch_mamba3_hybrid.py \
  --python-requirements scripts/requirements-v3a-python-mamba3.txt \
  --python-install-mode official-mamba3 \
  --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
  --stop-after-run \
  -- \
  "${COMMON_ARGS[@]}" \
  --run-label "${run_label}"

echo "completed runpod v3a python mamba3 seed ${SEED}"
