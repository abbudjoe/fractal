#!/usr/bin/env bash
# Create Storage Transfer Service URL-list jobs for each source in a Dolma
# transfer manifest. Use DRY_RUN=false to create jobs.

set -euo pipefail

MANIFEST_DIR="${1:?usage: scripts/start_dolma_raw_transfer_jobs.sh <manifest-dir> [bucket]}"
BUCKET="${2:-gs://fractal-llm-data-us-central1-81f2add4}"
TRANSFER_LIST_BUCKET="${TRANSFER_LIST_BUCKET:-gs://fractal-transfer-lists-81f2add4}"
PROJECT="${PROJECT:-project-81f2add4-9e80-4335-bb6}"
DRY_RUN="${DRY_RUN:-true}"

to_https_base() {
  local bucket_no_scheme="${TRANSFER_LIST_BUCKET#gs://}"
  printf 'https://storage.googleapis.com/%s' "${bucket_no_scheme%/}"
}

run() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

TRANSFER_LIST_HTTPS_BASE="$(to_https_base)"

python - "$MANIFEST_DIR" "$BUCKET" "$TRANSFER_LIST_HTTPS_BASE" <<'PY' | while IFS=$'\t' read -r source_tsv destination; do
import json
import sys
from pathlib import Path

manifest_dir = Path(sys.argv[1])
bucket = sys.argv[2].rstrip("/")
transfer_list_base = sys.argv[3].rstrip("/")
payload = json.loads((manifest_dir / "manifest.json").read_text())
dataset_id = payload["dataset_id"]
for source in payload["sources"]:
    source_name = source["source"]
    source_tsv = f"{transfer_list_base}/datasets/{dataset_id}/{source['transfer_tsv']}"
    destination = f"{bucket}/{source['destination_prefix']}"
    print(f"{source_tsv}\t{destination}")
PY
  desc="fractal ${source_tsv##*/} -> ${destination#${BUCKET}/}"
  run gcloud transfer jobs create "${source_tsv}" "${destination}" \
    --project="${PROJECT}" \
    --description="${desc}" \
    --custom-storage-class=STANDARD
done
