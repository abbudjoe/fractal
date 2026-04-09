#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"
LOCAL_RESULTS_ROOT="${REPO_ROOT}/.runpod-local-logs/runpod-results"
STATE_PATH="${STATE_PATH:-${REPO_ROOT}/artifacts/dreegmor-mini-moe-experiment/mps-autoresearch-d16-live/autoresearch_state.json}"

RUN_LABEL="${1:-mini-moe-cuda-shortlist}"

POD_NAME="${POD_NAME:-fractal-v3a}"
GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-14400}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
PYTHON_INSTALL_MODE="${PYTHON_INSTALL_MODE:-requirements-only}"
ENV_KIND="${ENV_KIND:-}"
COMPILE_MODE="${COMPILE_MODE:-}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-torch}"
PYTHON_BIN="${PYTHON_BIN:-/Users/joseph/fractal/.venv/bin/python}"
SEEDS="${SEEDS:-42,43}"
SEQ_LEN="${SEQ_LEN:-32}"
WINDOW_STRIDE="${WINDOW_STRIDE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEPS="${STEPS:-128}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-3}"
WARMUP_EVAL_BATCHES="${WARMUP_EVAL_BATCHES:-1}"
WARMUP_TRAIN_STEPS="${WARMUP_TRAIN_STEPS:-1}"
TOTAL_LAYERS="${TOTAL_LAYERS:-16}"
EXPERTS_PER_BLOCK="${EXPERTS_PER_BLOCK:-8}"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-0.95}"
DISPATCH_EXECUTION_STRATEGY="${DISPATCH_EXECUTION_STRATEGY:-dense_gather}"
ROUND2_EXECUTION_STRATEGY="${ROUND2_EXECUTION_STRATEGY:-dense_blend}"
CORPUS_TRAIN_JSONL="${CORPUS_TRAIN_JSONL:-experiments/stage0/assets/fineweb/stage0-canary/train.jsonl}"
CORPUS_EVAL_JSONL="${CORPUS_EVAL_JSONL:-experiments/stage0/assets/fineweb/stage0-canary/eval.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/dreegmor-mini-moe-experiment/runpod-cuda-shortlist}"
LEDGER_PATH="${LEDGER_PATH:-artifacts/dreegmor-mini-moe-experiment/runpod-cuda-shortlist/ledger.jsonl}"
TOP_K="${TOP_K:-2}"

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

build_mask_args() {
  "${PYTHON_BIN}" - "${REPO_ROOT}" "${STATE_PATH}" "${TOTAL_LAYERS}" "${TOP_K}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
state_path = Path(sys.argv[2])
total_layers = int(sys.argv[3])
top_k = int(sys.argv[4])

sys.path.insert(0, str(repo_root))
from python.runners.mini_moe_autoresearch import bitmask_to_mask, top_selective_mask_ids_from_state

mask_ids = top_selective_mask_ids_from_state(state_path, total_layers=total_layers, limit=top_k)
for mask_id in mask_ids:
    mask = bitmask_to_mask(mask_id, total_layers)
    print("--round2-mask")
    print(",".join(str(layer) for layer in mask))
PY
}

MASK_ARGS=()
while IFS= read -r line; do
  MASK_ARGS+=("${line}")
done < <(build_mask_args)

COMMON_ARGS=(
  --backend cuda
  --cuda-device "${CUDA_DEVICE}"
  --dtype "${DTYPE}"
  --jsonl-train-path "${CORPUS_TRAIN_JSONL}"
  --jsonl-eval-path "${CORPUS_EVAL_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --ledger-path "${LEDGER_PATH}"
  --run-label "${RUN_LABEL}"
  --benchmark-name "dreegmor-mini-moe-cuda-shortlist"
  --seq-len "${SEQ_LEN}"
  --window-stride "${WINDOW_STRIDE}"
  --batch-size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --eval-batches "${EVAL_BATCHES}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-eval-batches "${WARMUP_EVAL_BATCHES}"
  --warmup-train-steps "${WARMUP_TRAIN_STEPS}"
  --seeds "${SEEDS}"
  --total-layers "${TOTAL_LAYERS}"
  --experts-per-block "${EXPERTS_PER_BLOCK}"
  --entropy-threshold "${ENTROPY_THRESHOLD}"
  --dispatch-execution-strategy "${DISPATCH_EXECUTION_STRATEGY}"
  --round2-execution-strategy "${ROUND2_EXECUTION_STRATEGY}"
  --state-path "${STATE_PATH}"
  --top-k "${TOP_K}"
)

if [[ -n "${ENV_KIND}" ]]; then
  COMMON_ARGS+=(--env-kind "${ENV_KIND}")
fi
if [[ -n "${COMPILE_MODE}" ]]; then
  COMMON_ARGS+=(--compile-mode "${COMPILE_MODE}")
fi
if [[ -n "${PRIMITIVE_RUNTIME_BACKEND}" ]]; then
  COMMON_ARGS+=(--primitive-runtime-backend "${PRIMITIVE_RUNTIME_BACKEND}")
fi

EXPECTED_ARGS=("${COMMON_ARGS[@]}" "${MASK_ARGS[@]}")
if already_recorded "${RUN_LABEL}" "${EXPECTED_ARGS[@]}"; then
  echo "skip ${RUN_LABEL}"
  exit 0
fi

echo "run ${RUN_LABEL}"
"${WRAPPER}" \
  --pod-name "${POD_NAME}" \
  --gpu-id "${GPU_ID}" \
  --binary-kind python \
  --binary-name scripts/dreegmor_mini_moe_shortlist.py \
  --python-requirements scripts/requirements-mini-moe-python.txt \
  --python-install-mode "${PYTHON_INSTALL_MODE}" \
  --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
  --stop-after-run \
  -- \
  "${EXPECTED_ARGS[@]}"

echo "completed RunPod mini-MoE CUDA shortlist ${RUN_LABEL}"
