#!/usr/bin/env bash
# Prepare regional GCS buckets for TPU training/eval data mirrors.
#
# This script intentionally creates only storage buckets. It does not download
# or upload datasets. Use DRY_RUN=false to create the buckets.

set -euo pipefail

PROJECT="${PROJECT:-project-81f2add4-9e80-4335-bb6}"
SUFFIX="${SUFFIX:-81f2add4}"
STORAGE_CLASS="${STORAGE_CLASS:-STANDARD}"
REGIONS="${REGIONS:-us-central1 us-east1 europe-west4}"
DRY_RUN="${DRY_RUN:-true}"

run() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

ensure_bucket() {
  local region="$1"
  local bucket="gs://fractal-llm-data-${region}-${SUFFIX}"

  if gcloud storage buckets describe "${bucket}" --project="${PROJECT}" >/dev/null 2>&1; then
    echo "exists: ${bucket}"
    return
  fi

  run gcloud storage buckets create "${bucket}" \
    --project="${PROJECT}" \
    --location="${region}" \
    --default-storage-class="${STORAGE_CLASS}" \
    --uniform-bucket-level-access \
    --public-access-prevention
}

for region in ${REGIONS}; do
  ensure_bucket "${region}"
done

cat <<EOF

Recommended mapping:
  us-central1 bucket    -> TPU v5e us-central1-a, TPU v4 us-central2-b
  us-east1 bucket       -> TPU v6e us-east1-d
  europe-west4 bucket   -> TPU v6e europe-west4-a, TPU v5e europe-west4-b

Dataset layout:
  gs://fractal-llm-data-<region>-${SUFFIX}/datasets/<dataset-id>/train/...
  gs://fractal-llm-data-<region>-${SUFFIX}/datasets/<dataset-id>/eval/...
  gs://fractal-llm-data-<region>-${SUFFIX}/tokenizers/<tokenizer-id>/...

Run with DRY_RUN=false to create the buckets.
EOF
