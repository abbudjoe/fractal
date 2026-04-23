#!/usr/bin/env bash
# Launch a short-lived GCE VM that recovers missing Dolma shards with whole-file
# HTTP downloads and GCS resumable uploads.

set -euo pipefail

MANIFEST_JSONL="${1:?usage: scripts/start_dolma_gce_recovery.sh <missing-recovery-manifest.jsonl> [vm-name]}"
PROJECT="${PROJECT:-project-81f2add4-9e80-4335-bb6}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-8}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-100GB}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-}"
VM_NAME="${2:-dolma-recovery-$(date -u +%Y%m%d-%H%M%S)}"
TRANSFER_LIST_BUCKET="${TRANSFER_LIST_BUCKET:-gs://fractal-transfer-lists-81f2add4}"
DATA_BUCKET="${DATA_BUCKET:-fractal-llm-data-us-central1-81f2add4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_PREFIX="${TRANSFER_LIST_BUCKET}/recovery/${RUN_ID}"
COMPUTE_SA="${COMPUTE_SA:-727124170067-compute@developer.gserviceaccount.com}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/recover_dolma_missing_shards.py"

gcloud storage buckets add-iam-policy-binding "gs://${DATA_BUCKET}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role=roles/storage.admin \
  --project="${PROJECT}" >/dev/null

gcloud storage cp "${RUNNER}" "${REMOTE_PREFIX}/recover_dolma_missing_shards.py"
gcloud storage cp "${MANIFEST_JSONL}" "${REMOTE_PREFIX}/missing_recovery_manifest.jsonl"

PUBLIC_BASE="https://storage.googleapis.com/${TRANSFER_LIST_BUCKET#gs://}/recovery/${RUN_ID}"
PUBLIC_BASE="${PUBLIC_BASE%/}"
LOG_OBJECT="datasets/dolma-v1_7-400b/recovery/${RUN_ID}/recovery_events.jsonl"
SUMMARY_OBJECT="datasets/dolma-v1_7-400b/recovery/${RUN_ID}/recovery_summary.json"

STARTUP_SCRIPT="$(mktemp)"
cat >"${STARTUP_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y ca-certificates curl python3

mkdir -p /opt/fractal-dolma-recovery /mnt/fractal-dolma-recovery
curl -fsSL "${PUBLIC_BASE}/recover_dolma_missing_shards.py" \
  -o /opt/fractal-dolma-recovery/recover_dolma_missing_shards.py
curl -fsSL "${PUBLIC_BASE}/missing_recovery_manifest.jsonl" \
  -o /opt/fractal-dolma-recovery/missing_recovery_manifest.jsonl
chmod +x /opt/fractal-dolma-recovery/recover_dolma_missing_shards.py

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

set +e
python3 /opt/fractal-dolma-recovery/recover_dolma_missing_shards.py \
  --manifest /opt/fractal-dolma-recovery/missing_recovery_manifest.jsonl \
  --workers "${WORKERS}" \
  --work-dir /mnt/fractal-dolma-recovery \
  --events-log /opt/fractal-dolma-recovery/recovery_events.jsonl \
  --final-log-bucket "${DATA_BUCKET}" \
  --final-log-object "${LOG_OBJECT}" \
  --summary-object "${SUMMARY_OBJECT}" \
  "\${LIMIT_ARGS[@]}"
STATUS=\$?
set -e

shutdown -h now
exit "\${STATUS}"
EOF

gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type=pd-balanced \
  --service-account="${COMPUTE_SA}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=run_id="${RUN_ID}" \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}"

rm -f "${STARTUP_SCRIPT}"

cat <<EOF
vm_name=${VM_NAME}
zone=${ZONE}
run_id=${RUN_ID}
manifest=${REMOTE_PREFIX}/missing_recovery_manifest.jsonl
logs=gs://${DATA_BUCKET}/${LOG_OBJECT}
summary=gs://${DATA_BUCKET}/${SUMMARY_OBJECT}
EOF
