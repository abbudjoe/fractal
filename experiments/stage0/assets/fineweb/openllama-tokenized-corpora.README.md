---
license: apache-2.0
language:
  - en
task_categories:
  - text-generation
pretty_name: Fractal FineWeb OpenLLaMA Token Caches
---

# OpenLLaMA Tokenized FineWeb Caches

Date: 2026-04-16

These local token-ID corpora use the frozen Stage 0 tokenizer:

```text
experiments/stage0/assets/open_llama_3b_v2/tokenizer.model
```

The generated `.pt` shard files are intentionally ignored by Git. Each cache is described by a `manifest.json` in its output directory.

## Caches Materialized Locally

### `stage0-local-bench-2krow-openllama-tokens-v1`

- source: `stage0-local-bench-2krow-v1`
- source format: JSONL text
- tokenizer: OpenLLaMA SentencePiece
- vocab size: `32000`
- train docs: `1800`
- eval docs: `200`
- train tokens: `1,334,377`
- eval tokens: `146,435`
- shard size target: `1,000,000` tokens

### `fineweb-cc-main-2024-10-openllama-tokens-27m-v1`

- source: local FineWeb parquet cache, `CC-MAIN-2024-10/train/0000.parquet` through `0008.parquet`
- source format: parquet text
- tokenizer: OpenLLaMA SentencePiece
- vocab size: `32000`
- train docs: `32,477`
- eval docs: `2,531`
- train tokens: `25,007,703`
- eval tokens: `2,000,938`
- shard size target: `1,000,000` tokens
- split rule: deterministic eval holdout by document stride, `eval_doc_stride=100`

### `fineweb-cc-main-2024-10-openllama-tokens-250m-v1`

- source: local FineWeb parquet cache, `CC-MAIN-2024-10/train/0000.parquet` through `0008.parquet`
- source format: parquet text
- tokenizer: OpenLLaMA SentencePiece
- vocab size: `32000`
- train docs: `323,715`
- eval docs: `12,760`
- train tokens: `250,000,207`
- eval tokens: `10,000,169`
- shard size target: `1,000,000` tokens
- split rule: deterministic eval holdout by document stride, `eval_doc_stride=100`

### `fineweb-cc-main-2024-10-openllama-tokens-750m-v1`

- source: local FineWeb parquet cache, `CC-MAIN-2024-10/train/0000.parquet` through `0008.parquet`
- source format: parquet text
- tokenizer: OpenLLaMA SentencePiece
- vocab size: `32000`
- train docs: `950,553`
- eval docs: `12,533`
- train tokens: `750,000,544`
- eval tokens: `10,029,403`
- shard size target: `1,000,000` tokens
- split rule: deterministic eval holdout by document stride, `eval_doc_stride=100`
- purpose: supports 10-minute H100 lane runs at `seq_len=256`, `batch_size=64` without wrapping the train cache

## Rebuild Commands

```bash
/Users/joseph/fractal/.venv/bin/python scripts/tokenize_stage0_corpus.py \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-2krow-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-2krow-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-2krow-openllama-tokens-v1 \
  --output-dir experiments/stage0/assets/fineweb/stage0-local-bench-2krow-openllama-tokens-v1 \
  --shard-token-count 1000000
```

```bash
/Users/joseph/fractal/.venv/bin/python scripts/tokenize_stage0_corpus.py \
  --parquet-dir experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-parquet-20gb \
  --parquet-glob 'CC-MAIN-2024-10/train/*.parquet' \
  --corpus-name fineweb-cc-main-2024-10-openllama-tokens-27m-v1 \
  --output-dir experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1 \
  --max-train-tokens 25000000 \
  --max-eval-tokens 2000000 \
  --shard-token-count 1000000 \
  --eval-doc-stride 100 \
  --parquet-batch-size 512
```

```bash
/Users/joseph/fractal/.venv/bin/python scripts/tokenize_stage0_corpus.py \
  --parquet-dir experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-parquet-20gb \
  --corpus-name fineweb-cc-main-2024-10-openllama-750m \
  --output-dir experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-750m-v1 \
  --max-train-tokens 750000000 \
  --max-eval-tokens 10000000 \
  --shard-token-count 1000000 \
  --eval-doc-stride 100 \
  --parquet-batch-size 512 \
  --force
```

## Runner Usage

```bash
/Users/joseph/fractal/.venv/bin/python scripts/eml_ffn_expert_sweep.py \
  --backend mps \
  --dtype fp32 \
  --tokenized-manifest-path experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1/manifest.json \
  --corpus-name fineweb-cc-main-2024-10-openllama-tokens-27m-v1
```

The sweep infers `vocab_size=32000` from the tokenized manifest.
