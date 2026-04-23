# TPU Scale Data Kickoff Status

Date: 2026-04-22

## Storage

Created regional private data buckets:

| Bucket | Location |
| --- | --- |
| `gs://fractal-llm-data-us-central1-81f2add4` | `US-CENTRAL1` |
| `gs://fractal-llm-data-us-east1-81f2add4` | `US-EAST1` |
| `gs://fractal-llm-data-europe-west4-81f2add4` | `EUROPE-WEST4` |

Created public URL-list bucket:

| Bucket | Location | Contents |
| --- | --- | --- |
| `gs://fractal-transfer-lists-81f2add4` | `US-CENTRAL1` | public Dolma transfer-list TSVs only |

The Storage Transfer service agent
`project-727124170067@storage-transfer-service.iam.gserviceaccount.com` has
`roles/storage.admin` on the private data buckets so it can write transfer
outputs.

## Dolma Transfer

Source list: `/tmp/fractal-dolma-metadata/urls/v1_7.txt`

Full manifest:

```text
artifacts/dolma-v1_7-400b-transfer-manifest-v1/manifest.json
dataset_id: dolma-v1_7-400b
selected URLs: 332
estimated tokens: 411,740,388,590
selection basis: token-fraction-to-file-count
```

Note: HTTP `HEAD` requests to `olmo-data.org` returned 403s, so selected byte
counts are unknown. The current subset is an approximate file-count sample.

Smoke manifest:

```text
artifacts/dolma-v1_7-smoke-transfer-manifest-v1/manifest.json
dataset_id: dolma-v1_7-smoke
selected URLs: 1
estimated tokens: 1,240,181,893
```

Smoke result:

```text
job: transferJobs/3664254974449508683
operation: transferOperations/transferJobs-3664254974449508683-1395586673159838483
status: SUCCESS
objects copied: 1
bytes copied: 3,370,184,455
```

Full transfer launch snapshot:

```text
jobs: 19
operations: 19 IN_PROGRESS
objects found: 332
```

## Recovery

Storage Transfer completed only part of the raw transfer. The failure mode was
HTTP range validation against `olmo-data.org` / Cloudflare: Storage Transfer
requested nonzero byte ranges and sometimes received full-object `200 OK`
responses, causing `FAILED_PRECONDITION`.

Built a recovery manifest from exact ranged `Content-Range` sizes and current
GCS object state:

```text
artifacts/dolma-v1_7-400b-recovery-v1/recovery_summary.json
selected URLs: 332
present URLs: 139
missing URLs: 193
selected bytes: 659,979,581,992
present bytes: 191,958,477,792
missing bytes: 468,021,104,200
```

Launched a GCE whole-object recovery VM:

```text
vm: dolma-recovery-400b-v1
zone: us-central1-a
machine: e2-standard-8
workers: 8
run id: 20260422T150252Z
```

The recovery runner downloads whole objects with `curl`, verifies local byte
size, uploads via GCS resumable upload, and verifies the destination object
size. This avoids Storage Transfer's chunked HTTP source path.

Early recovery proof of life:

```text
status: recovered events observed in serial output
dataset objects after launch: 158 / 332
dataset bytes after launch: 227,070,089,934 / 659,979,581,992
```

Snapshot artifacts:

```text
artifacts/dolma-v1_7-400b-transfer-manifest-v1/transfer_jobs_after_launch.json
artifacts/dolma-v1_7-400b-transfer-manifest-v1/transfer_operations_after_launch.json
```

## Commands

```bash
DRY_RUN=false bash scripts/prepare_tpu_data_storage.sh

python scripts/build_dolma_transfer_manifests.py \
  --url-list /tmp/fractal-dolma-metadata/urls/v1_7.txt \
  --output-dir artifacts/dolma-v1_7-smoke-transfer-manifest-v1 \
  --dataset-id dolma-v1_7-smoke \
  --target-tokens 1000000000 \
  --include-sources books

python scripts/build_dolma_transfer_manifests.py \
  --url-list /tmp/fractal-dolma-metadata/urls/v1_7.txt \
  --output-dir artifacts/dolma-v1_7-400b-transfer-manifest-v1 \
  --dataset-id dolma-v1_7-400b \
  --target-tokens 400000000000

bash scripts/upload_dolma_transfer_manifests.sh \
  artifacts/dolma-v1_7-smoke-transfer-manifest-v1

bash scripts/upload_dolma_transfer_manifests.sh \
  artifacts/dolma-v1_7-400b-transfer-manifest-v1

DRY_RUN=false bash scripts/start_dolma_raw_transfer_jobs.sh \
  artifacts/dolma-v1_7-smoke-transfer-manifest-v1

DRY_RUN=false bash scripts/start_dolma_raw_transfer_jobs.sh \
  artifacts/dolma-v1_7-400b-transfer-manifest-v1
```
