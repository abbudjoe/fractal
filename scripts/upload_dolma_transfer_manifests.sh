#!/usr/bin/env bash
# Upload Dolma transfer manifests to the canonical regional data bucket.

set -euo pipefail

MANIFEST_DIR="${1:?usage: scripts/upload_dolma_transfer_manifests.sh <manifest-dir> [bucket]}"
BUCKET="${2:-gs://fractal-llm-data-us-central1-81f2add4}"
TRANSFER_LIST_BUCKET="${TRANSFER_LIST_BUCKET:-gs://fractal-transfer-lists-81f2add4}"

DATASET_ID="$(python - "$MANIFEST_DIR" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
print(payload["dataset_id"])
PY
)"

DEST="${BUCKET}/datasets/${DATASET_ID}/manifests"
gcloud storage cp "${MANIFEST_DIR}/manifest.json" "${DEST}/manifest.json"
gcloud storage cp "${MANIFEST_DIR}/selected_urls.txt" "${DEST}/selected_urls.txt"
gcloud storage cp "${MANIFEST_DIR}/all_urls.txt" "${DEST}/all_urls.txt"
gcloud storage rsync -r "${MANIFEST_DIR}/transfer-url-lists" "${DEST}/transfer-url-lists"

TRANSFER_DEST="${TRANSFER_LIST_BUCKET}/datasets/${DATASET_ID}/transfer-url-lists"
gcloud storage rsync -r "${MANIFEST_DIR}/transfer-url-lists" "${TRANSFER_DEST}"

echo "${DEST}/manifest.json"
echo "${TRANSFER_DEST}"
