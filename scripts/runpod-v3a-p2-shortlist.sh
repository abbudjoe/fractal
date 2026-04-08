#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"

SEED="${1:-42}"
STEPS="${2:-16}"
EVAL_BATCHES="${3:-4}"
LABEL_PREFIX="${4:-v3a-cuda-shortlist-canary}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-3600}"
FULL_TRAIN_PASS="${FULL_TRAIN_PASS:-0}"
FULL_EVAL_PASS="${FULL_EVAL_PASS:-0}"
RUN_P20_PLAIN="${RUN_P20_PLAIN:-1}"
BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-}"
REFERENCE_SSM_PROFILE="${REFERENCE_SSM_PROFILE:-rust-mimo-reference}"

CORPUS_TRAIN_JSONL="${CORPUS_TRAIN_JSONL:-}"
CORPUS_EVAL_JSONL="${CORPUS_EVAL_JSONL:-}"
CORPUS_NAME="${CORPUS_NAME:-}"
CORPUS_TEXT_FIELD="${CORPUS_TEXT_FIELD:-text}"

if [[ "${BENCHMARK_PROFILE}" == "cuda-faithful-small-v1" ]]; then
  if [[ -n "${CORPUS_TRAIN_JSONL}" || -n "${CORPUS_EVAL_JSONL}" ]]; then
    echo "BENCHMARK_PROFILE=cuda-faithful-small-v1 may not be combined with explicit CORPUS_TRAIN_JSONL/CORPUS_EVAL_JSONL for the Rust runner" >&2
    exit 1
  fi
  FULL_TRAIN_PASS=1
  FULL_EVAL_PASS=1
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

build_common_args() {
  COMMON_ARGS=(
    --seed "${SEED}"
    --output table
    --reference-ssm-profile "${REFERENCE_SSM_PROFILE}"
  )

  if [[ -n "${BENCHMARK_PROFILE}" ]]; then
    COMMON_ARGS+=(--benchmark-profile "${BENCHMARK_PROFILE}")
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
  fi
}

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
    --binary-kind bin \
    --binary-name v3a-hybrid-attention-matrix \
    --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
    "${lifecycle_flag}" \
    -- \
    "${expected_args[@]}"
}

build_common_args

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-attention-only" \
  --variant attention-only

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-reference-ssm-hybrid" \
  --variant reference-ssm-hybrid

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-p2-incumbent" \
  --variant primitive-hybrid \
  --primitive-profile p2

run_lane \
  --keep-pod \
  "${LABEL_PREFIX}-s${SEED}-p2-3-gated-projected-norm-residual-renorm-standard" \
  --variant primitive-hybrid \
  --primitive-profile p2-3 \
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
  "${LABEL_PREFIX}-s${SEED}-p2-0-scaled-projected-pre-norm-only-standard" \
  --variant primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-residual-profile scaled \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard

if [[ "${RUN_P20_PLAIN}" == "1" ]]; then
  run_lane \
    --stop-after-run \
    "${LABEL_PREFIX}-s${SEED}-p2-0-plain-projected-pre-norm-only-standard" \
    --variant primitive-hybrid \
    --primitive-profile p2-0 \
    --primitive-residual-profile plain \
    --primitive-readout-profile projected \
    --primitive-norm-profile pre-norm-only \
    --primitive-wrapper-profile standard
fi

echo "completed runpod v3a p2 shortlist seed ${SEED}"
