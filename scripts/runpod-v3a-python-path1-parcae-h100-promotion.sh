#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/scripts/runpod-tournament.sh"

SEED="${1:-42}"
STEPS="${2:-2000}"
EVAL_BATCHES="${3:-64}"
LABEL_PREFIX="${4:-v3a-python-path1-parcae-h100-promotion}"

GPU_ID="${GPU_ID:-NVIDIA H100 80GB HBM3}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DTYPE="${DTYPE:-bf16}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_SEED="${DATA_SEED:-42}"
PRIMITIVE_RUNTIME_BACKEND="${PRIMITIVE_RUNTIME_BACKEND:-triton}"
TOKEN_CACHE_REPO_ID="${TOKEN_CACHE_REPO_ID:-joebud/fractal-fineweb-openllama-tokens}"
POD_NAME="${POD_NAME:-fractal-v3a-parcae-h100-s${SEED}}"

resolve_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s\n' "${HF_TOKEN}"
    return 0
  fi
  "${REPO_ROOT}/.venv/bin/python" - <<'PY'
from huggingface_hub import get_token
token = get_token()
if token:
    print(token)
PY
}

HF_TOKEN_VALUE="$(resolve_hf_token)"
if [[ -z "${HF_TOKEN_VALUE}" ]]; then
  echo "HF_TOKEN is required, or run huggingface-cli login locally before launching." >&2
  exit 1
fi

RUN_LABEL="${LABEL_PREFIX}-s${SEED}-steps${STEPS}-seq${SEQ_LEN}-bs${BATCH_SIZE}"

HF_TOKEN="${HF_TOKEN_VALUE}" "${WRAPPER}" \
  --pod-name "${POD_NAME}" \
  --gpu-id "${GPU_ID}" \
  --cloud-type "${CLOUD_TYPE}" \
  --binary-kind python \
  --binary-name scripts/v3a_python_path1_parcae_h100_promotion.py \
  --python-requirements scripts/requirements-v3a-python-tokenized-corpus.txt \
  --python-install-mode primitive-triton \
  --run-timeout-seconds "${RUN_TIMEOUT_SECONDS}" \
  --forward-env HF_TOKEN \
  --stop-after-run \
  -- \
  --cuda-device "${CUDA_DEVICE}" \
  --dtype "${DTYPE}" \
  --seed "${SEED}" \
  --data-seed "${DATA_SEED}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --eval-batches "${EVAL_BATCHES}" \
  --primitive-runtime-backend "${PRIMITIVE_RUNTIME_BACKEND}" \
  --token-cache-repo-id "${TOKEN_CACHE_REPO_ID}" \
  --run-label "${RUN_LABEL}"

