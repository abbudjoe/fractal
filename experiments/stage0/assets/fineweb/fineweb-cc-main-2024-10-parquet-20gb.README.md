# FineWeb CC-MAIN-2024-10 Parquet Cache

Date: 2026-04-15

This is a local raw parquet cache for larger LM experiments. The shard files themselves are intentionally ignored by Git.

## Source

- dataset: `HuggingFaceFW/fineweb`
- revision: `refs/convert/parquet`
- subset/config: `CC-MAIN-2024-10`
- split: `train`
- files: `CC-MAIN-2024-10/train/0000.parquet` through `0008.parquet`
- planned size: about `19.4GB`

## Download Command

```bash
/opt/homebrew/bin/hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --revision refs/convert/parquet \
  --include 'CC-MAIN-2024-10/train/000[0-8].parquet' \
  --local-dir experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-parquet-20gb \
  --max-workers 4
```

## Harness Status

The current byte-level JSONL harness eagerly materializes byte windows into memory. That means the raw 19.4GB parquet cache is useful as the source of truth, but it should not be converted wholesale into one eager JSONL corpus without adding a streaming or sharded loader.

For immediate local MPS runs, use the materialized medium slice:

```text
experiments/stage0/assets/fineweb/stage0-local-bench-2krow-v1/
```

That slice contains 1,800 train rows and 200 eval rows, about `6.6MB` of raw text, and yields about 92k train byte-windows at `seq_len=64`.
