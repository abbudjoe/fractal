#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${1:-42}"
STEPS="${2:-16}"
EVAL_BATCHES="${3:-4}"
LABEL_PREFIX="${4:-v3a-p2-contender-grid}"
LEDGER_PATH="${LEDGER_PATH:-default}"

PROFILES=(p2-0 p2 p2-1 p2-2 p2-3)
RESIDUALS=(plain scaled gated)
READOUTS=(direct projected projected-norm)
NORMS=(pre-norm-only post-readout-norm residual-renorm)
WRAPPERS=(standard mamba-rms)

if [[ "${LEDGER_PATH}" == "default" ]]; then
  LEDGER_FILE="${REPO_ROOT}/docs/v3a-results-ledger.jsonl"
else
  LEDGER_FILE="${LEDGER_PATH}"
fi

total_runs=$(( ${#PROFILES[@]} * ${#RESIDUALS[@]} * ${#READOUTS[@]} * ${#NORMS[@]} * ${#WRAPPERS[@]} ))
current_run=0

echo "v3a contender sweep"
echo "repo=${REPO_ROOT}"
echo "seed=${SEED} steps=${STEPS} eval_batches=${EVAL_BATCHES}"
echo "ledger=${LEDGER_FILE}"
echo "label_prefix=${LABEL_PREFIX}"
echo "total_runs=${total_runs}"

for profile in "${PROFILES[@]}"; do
  for residual in "${RESIDUALS[@]}"; do
    for readout in "${READOUTS[@]}"; do
      for norm in "${NORMS[@]}"; do
        for wrapper in "${WRAPPERS[@]}"; do
          current_run=$(( current_run + 1 ))
          run_label="${LABEL_PREFIX}-s${SEED}-${profile}-${residual}-${readout}-${norm}-${wrapper}"
          if [[ -f "${LEDGER_FILE}" ]] && rg -q "\"run_label\":\"${run_label}\"" "${LEDGER_FILE}"; then
            echo "[${current_run}/${total_runs}] skip ${run_label}"
            continue
          fi
          echo "[${current_run}/${total_runs}] run ${run_label}"
          cargo run --quiet --bin v3a-hybrid-attention-matrix -- \
            --variant primitive-hybrid \
            --steps "${STEPS}" \
            --eval-batches "${EVAL_BATCHES}" \
            --seed "${SEED}" \
            --primitive-profile "${profile}" \
            --primitive-residual-profile "${residual}" \
            --primitive-readout-profile "${readout}" \
            --primitive-norm-profile "${norm}" \
            --primitive-wrapper-profile "${wrapper}" \
            --ledger-path "${LEDGER_PATH}" \
            --run-label "${run_label}" \
            --output table
        done
      done
    done
  done
done

echo "completed ${total_runs} contender configurations"
