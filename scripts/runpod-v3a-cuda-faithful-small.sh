#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED="${1:-42}"
RUST_LABEL_PREFIX="${2:-v3a-cuda-faithful-small-rust}"
PYTHON_LABEL_PREFIX="${3:-v3a-cuda-faithful-small-python}"

BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-cuda-faithful-small-v1}"

(
  cd "${REPO_ROOT}"
  BENCHMARK_PROFILE="${BENCHMARK_PROFILE}" \
  ./scripts/runpod-v3a-p2-shortlist.sh "${SEED}" 64 8 "${RUST_LABEL_PREFIX}"
)

(
  cd "${REPO_ROOT}"
  BENCHMARK_PROFILE="${BENCHMARK_PROFILE}" \
  ./scripts/runpod-v3a-python-mamba3.sh "${SEED}" 64 8 "${PYTHON_LABEL_PREFIX}"
)
