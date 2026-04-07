#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${1:-42}"
STEPS="${2:-64}"
EVAL_BATCHES="${3:-8}"
LABEL_PREFIX="${4:-v3a-p2-leader-longer-budget}"
LEDGER_PATH="${LEDGER_PATH:-default}"
BACKEND="${BACKEND:-cpu}"
CORPUS_TRAIN_JSONL="${CORPUS_TRAIN_JSONL:-}"
CORPUS_EVAL_JSONL="${CORPUS_EVAL_JSONL:-}"
CORPUS_NAME="${CORPUS_NAME:-}"
CORPUS_TEXT_FIELD="${CORPUS_TEXT_FIELD:-text}"
FULL_TRAIN_PASS="${FULL_TRAIN_PASS:-0}"
FULL_EVAL_PASS="${FULL_EVAL_PASS:-0}"

if [[ "${LEDGER_PATH}" == "default" ]]; then
  LEDGER_FILE="${REPO_ROOT}/docs/v3a-results-ledger.jsonl"
else
  LEDGER_FILE="${LEDGER_PATH}"
fi

run_if_needed() {
  local run_label="$1"
  shift
  if [[ -f "${LEDGER_FILE}" ]] && rg -q "\"run_label\":\"${run_label}\"" "${LEDGER_FILE}"; then
    echo "skip ${run_label}"
    return 0
  fi
  echo "run ${run_label}"
  local cmd=(
    cargo run --quiet --bin v3a-hybrid-attention-matrix --
    "$@"
    --backend "${BACKEND}"
    --seed "${SEED}"
    --ledger-path "${LEDGER_PATH}"
    --run-label "${run_label}"
    --output table
  )
  if [[ "${FULL_TRAIN_PASS}" == "1" ]]; then
    cmd+=(--full-train-pass)
  else
    cmd+=(--steps "${STEPS}")
  fi
  if [[ "${FULL_EVAL_PASS}" == "1" ]]; then
    cmd+=(--full-eval-pass)
  else
    cmd+=(--eval-batches "${EVAL_BATCHES}")
  fi
  "${cmd[@]}"
}

COMMON_ARGS=()
if [[ -n "${CORPUS_TRAIN_JSONL}" || -n "${CORPUS_EVAL_JSONL}" ]]; then
  if [[ -z "${CORPUS_TRAIN_JSONL}" || -z "${CORPUS_EVAL_JSONL}" ]]; then
    echo "CORPUS_TRAIN_JSONL and CORPUS_EVAL_JSONL must be set together" >&2
    exit 1
  fi
  COMMON_ARGS+=(--jsonl-train-path "${CORPUS_TRAIN_JSONL}" --jsonl-eval-path "${CORPUS_EVAL_JSONL}")
  if [[ -n "${CORPUS_NAME}" ]]; then
    COMMON_ARGS+=(--corpus-name "${CORPUS_NAME}")
  fi
  if [[ "${CORPUS_TEXT_FIELD}" != "text" ]]; then
    COMMON_ARGS+=(--corpus-text-field "${CORPUS_TEXT_FIELD}")
  fi
fi
run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-reference-ssm-hybrid" \
  "${COMMON_ARGS[@]}" \
  --variant reference-ssm-hybrid

run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-p2-incumbent" \
  "${COMMON_ARGS[@]}" \
  --variant primitive-hybrid \
  --primitive-profile p2

run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-p2-3-gated-projected-norm-residual-renorm-standard" \
  "${COMMON_ARGS[@]}" \
  --variant primitive-hybrid \
  --primitive-profile p2-3 \
  --primitive-residual-profile gated \
  --primitive-readout-profile projected-norm \
  --primitive-norm-profile residual-renorm \
  --primitive-wrapper-profile standard

run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-p2-0-scaled-projected-pre-norm-only-standard" \
  "${COMMON_ARGS[@]}" \
  --variant primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-residual-profile scaled \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard

run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-p2-0-plain-projected-pre-norm-only-standard" \
  "${COMMON_ARGS[@]}" \
  --variant primitive-hybrid \
  --primitive-profile p2-0 \
  --primitive-residual-profile plain \
  --primitive-readout-profile projected \
  --primitive-norm-profile pre-norm-only \
  --primitive-wrapper-profile standard

run_if_needed \
  "${LABEL_PREFIX}-s${SEED}-attention-only" \
  "${COMMON_ARGS[@]}" \
  --variant attention-only

echo "completed leader longer-budget run set"
