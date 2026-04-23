# TPU Training/Eval Data Storage

The free TPU grant covers Cloud TPU chips in specific zones. It does not make
all supporting storage and network traffic free. Training and evaluation data
should therefore be staged deliberately instead of streamed repeatedly from the
public internet or from a far-away bucket.

## Current State

The project currently has one run bucket:

| Bucket | Location | Approx Contents | Recommended Use |
| --- | --- | ---: | --- |
| `gs://fractal-maxtext-runs-81f2add4` | `US-WEST4` | about 27 GB | existing logs/checkpoints only |

This bucket is not ideal as the main training-data bucket because none of the
granted TPU zones are in `us-west4`, and the grant includes both US and Europe
TPU regions.

Regional training-data buckets have now been created:

| Bucket | Location | Purpose |
| --- | --- | --- |
| `gs://fractal-llm-data-us-central1-81f2add4` | `US-CENTRAL1` | canonical prepared/raw dataset staging |
| `gs://fractal-llm-data-us-east1-81f2add4` | `US-EAST1` | mirror for `us-east1-d` TPU runs |
| `gs://fractal-llm-data-europe-west4-81f2add4` | `EUROPE-WEST4` | mirror for `europe-west4-a/b` TPU runs |

A separate public transfer-list bucket exists only for Storage Transfer URL-list
TSVs:

| Bucket | Location | Public contents |
| --- | --- | --- |
| `gs://fractal-transfer-lists-81f2add4` | `US-CENTRAL1` | Dolma public-source URL lists only |

The data buckets keep public access prevention enabled. The transfer-list bucket
is public because Storage Transfer's HTTP source mode requires an HTTP(S)
URL-list TSV. It must not contain private data, checkpoints, credentials, or
model artifacts.

## Recommended Layout

Use regional Standard GCS buckets as read-mostly mirrors:

| Bucket | Region | TPU zones served |
| --- | --- | --- |
| `gs://fractal-llm-data-us-central1-81f2add4` | `us-central1` | `us-central1-a`, nearby fallback for `us-central2-b` |
| `gs://fractal-llm-data-us-east1-81f2add4` | `us-east1` | `us-east1-d` |
| `gs://fractal-llm-data-europe-west4-81f2add4` | `europe-west4` | `europe-west4-a`, `europe-west4-b` |

Keep the object layout identical in each bucket:

```text
datasets/
  <dataset-id>/
    manifest.json
    train/
      shard-000000-of-...
    eval/
      shard-000000-of-...
tokenizers/
  <tokenizer-id>/
    ...
```

The runner should choose the nearest mirror based on TPU zone. If a zone loses
capacity and a run moves, only the data URI should change.

## Dataset Format

For short smokes, MaxText's HuggingFace pipeline is acceptable. For real runs,
prefer prepared GCS shards:

1. Store the frozen train/eval corpus once in a canonical prepared format.
2. Mirror it to the regional buckets.
3. Point MaxText at the regional files with `hf_train_files` and
   `hf_eval_files`, or move to the Grain pipeline for larger prepared corpora.

MaxText supports HuggingFace, Grain, and TFDS input pipelines. The Grain path is
the better long-run target because it supports large sharded datasets, including
ArrayRecord, Parquet, and TFRecord. MaxText's Grain documentation says `gs://`
paths can be used directly, but recommends Cloud Storage FUSE for better
ArrayRecord performance because metadata caching helps random access.

## Cost Guardrails

With about `$270` of remaining credits for storage/data movement, a practical
budget is:

- Keep hot data in Standard storage, not Nearline/Coldline/Archive. The colder
  classes can have retrieval or minimum-duration surprises.
- Mirror only the prepared dataset versions actually used in the scale ladder.
- Avoid multi-region or dual-region buckets for this use case; regional mirrors
  are cheaper and make data locality explicit.
- Avoid millions of tiny files. Use large shards to reduce operation overhead
  and metadata pressure.
- Delete failed-run scratch outputs and old transient checkpoints quickly.
- Prefer per-region dataset mirrors over reading a US bucket from Europe for
  every step.

As of the current Google Cloud pricing page, Standard regional storage in
`us-central1`, `us-east1`, and `europe-west4` is roughly `$0.020/GB-month`.
North America to Europe Cloud Storage transfer is materially more expensive than
same-region access, so one intentional mirror copy is better than repeated
cross-continent training reads.

## Prep Commands

Dry-run bucket creation:

```bash
bash scripts/prepare_tpu_data_storage.sh
```

Create the regional data buckets:

```bash
DRY_RUN=false bash scripts/prepare_tpu_data_storage.sh
```

Mirror a prepared dataset from the canonical US central bucket to the other
regions:

```bash
gcloud storage rsync -r \
  gs://fractal-llm-data-us-central1-81f2add4/datasets/<dataset-id> \
  gs://fractal-llm-data-us-east1-81f2add4/datasets/<dataset-id>

gcloud storage rsync -r \
  gs://fractal-llm-data-us-central1-81f2add4/datasets/<dataset-id> \
  gs://fractal-llm-data-europe-west4-81f2add4/datasets/<dataset-id>
```

Run Path1 against prepared HF-compatible GCS shards:

```bash
HF_PATH=json \
HF_TRAIN_FILES="gs://fractal-llm-data-us-east1-81f2add4/datasets/<dataset-id>/train/*.jsonl" \
HF_EVAL_FILES="gs://fractal-llm-data-us-east1-81f2add4/datasets/<dataset-id>/eval/*.jsonl" \
bash ~/fractal/scripts/run_maxtext_path1_scale_leaders_tpu.sh
```

For larger corpora, replace the HuggingFace file path with the MaxText Grain
pipeline once the prepared ArrayRecord/Parquet/TFRecord shards are available.

## Dolma v1.7 Raw Transfer Kickoff

Source metadata came from `allenai/dolma` URL list `urls/v1_7.txt`, cloned to
`/tmp/fractal-dolma-metadata`. The v1.7 list contains 2,419 public raw shard
URLs. HTTP `HEAD` size probing against `olmo-data.org` returned 403s, so the
first 400B-token subset is an approximate source-preserving file-count sample
rather than an exact byte-proportional sample.

Generated manifests:

| Manifest | Dataset id | Selection | URLs | Estimated tokens |
| --- | --- | --- | ---: | ---: |
| `artifacts/dolma-v1_7-smoke-transfer-manifest-v1/manifest.json` | `dolma-v1_7-smoke` | one `books` shard | 1 | about 1.24B |
| `artifacts/dolma-v1_7-400b-transfer-manifest-v1/manifest.json` | `dolma-v1_7-400b` | source-preserving file-count sample | 332 | about 411.7B |

The one-shard Storage Transfer smoke completed successfully:

```text
job: transferJobs/3664254974449508683
operation: transferOperations/transferJobs-3664254974449508683-1395586673159838483
objects copied: 1
bytes copied: 3,370,184,455
status: SUCCESS
```

The full raw transfer was then launched as 19 one-shot Storage Transfer jobs,
one per Dolma source family, into:

```text
gs://fractal-llm-data-us-central1-81f2add4/datasets/dolma-v1_7-400b/raw/<source>/
```

Launch snapshot:

```text
jobs: 19
operations: 19 IN_PROGRESS
objects found: 332
```

Storage Transfer did not finish the full dataset. It copied 139 of 332 shards
and then failed many larger HTTP objects because the source CDN sometimes
returned a full-object `200 OK` response to a nonzero byte-range request. The
remaining shards are being recovered by a GCE whole-object downloader:

```text
recovery VM: dolma-recovery-400b-v1
zone: us-central1-a
manifest: artifacts/dolma-v1_7-400b-recovery-v1/missing_recovery_manifest.jsonl
missing shards: 193
missing bytes: 468,021,104,200
```

The recovery path intentionally avoids ranged source downloads:

```text
curl whole object -> verify local byte size -> GCS resumable upload -> verify GCS byte size
```

Local launch artifacts:

```text
artifacts/dolma-v1_7-400b-transfer-manifest-v1/transfer_jobs_after_launch.json
artifacts/dolma-v1_7-400b-transfer-manifest-v1/transfer_operations_after_launch.json
```

Useful monitor commands:

```bash
gcloud transfer operations list --format=json --limit=80 \
  > artifacts/dolma-v1_7-400b-transfer-manifest-v1/transfer_operations_latest.json

gcloud storage du -s \
  gs://fractal-llm-data-us-central1-81f2add4/datasets/dolma-v1_7-400b
```
