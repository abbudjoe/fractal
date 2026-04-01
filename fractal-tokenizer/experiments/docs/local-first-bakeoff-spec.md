# Local-First Real-World Bakeoff Spec

## Purpose

This document defines the first real-world tokenizer bakeoff for the current
`NoveltyAware` + packaging + model-facing stack.

The bakeoff is intentionally local-first:

- use real operational text already present on this machine
- use real source/docs text from the local `fawx` codebase
- compare the fractal tokenizer against the official pretrained tokenizers we
  already have working locally

This phase is not a downstream model-quality experiment yet.
It is a real-world robustness and compression bakeoff.

## Goals

The bakeoff should answer these questions:

1. Does the tokenizer remain lossless on messy real inputs?
2. Do chunk packaging and native retokenization remain deterministic and UTF-8-safe?
3. Does the strong compression win survive outside the synthetic benchmark inputs?
4. Does `motif_reuse` stay high where repetition is real and low where reuse
   would be suspicious?

## Non-Goals

This phase does **not** attempt to prove:

- lower downstream perplexity
- better end-to-end model quality
- lower production inference cost
- final corpus diversity across every programming language or document family

Those come later.

## Data Sources

### Source A: Operational logs

Primary local paths:

- `/Users/joseph/.fawx/server.log`
- `/Users/joseph/.fawx/logs/*.log`

Current inventory on this machine:

- `server.log`: about `10,012,424` bytes
- rotated log files: `6`

Why this source matters:

- highly repetitive
- operationally noisy
- realistic timestamped/service text
- likely to expose both compression wins and tokenizer-path failures

### Source B: Structured JSONL/session text

Primary local paths:

- `/Users/joseph/.fawx/journal.jsonl`
- `/Users/joseph/.fawx/signals/headless.jsonl`

Current inventory on this machine:

- `journal.jsonl`: `9` lines
- `headless.jsonl`: `29,703` lines

Why this source matters:

- structured event text
- mixed natural language + machine fields
- good test for false-positive reuse and UTF-8 safety

### Source C: Local code

Primary local roots:

- `/Users/joseph/fawx/engine`
- `/Users/joseph/fawx/app`

Allowed file types:

- `*.rs`
- `*.swift`

Current inventory on this machine:

- Rust files: `443`
- Swift files: `146`

Why this source matters:

- realistic developer text
- indentation, symbols, identifiers, comments, strings
- useful code bucket even though the local language mix is mostly Rust + Swift

### Source D: Local prose/spec docs

Primary local root:

- `/Users/joseph/fawx/docs`

Allowed file types:

- `*.md`

Current inventory on this machine:

- Markdown docs: `354`

Why this source matters:

- natural prose
- technical explanation
- mixed structure with headings, lists, code fences, and design notes

## Sensitive Data Rules

This bakeoff must use an explicit allowlist, not a broad directory sweep.

Allowed local sources for this phase:

- `/Users/joseph/.fawx/server.log`
- `/Users/joseph/.fawx/logs/*.log`
- `/Users/joseph/.fawx/journal.jsonl`
- `/Users/joseph/.fawx/signals/headless.jsonl`
- `/Users/joseph/fawx/engine/**/*.rs`
- `/Users/joseph/fawx/app/**/*.swift`
- `/Users/joseph/fawx/docs/**/*.md`

Explicitly excluded:

- `/Users/joseph/.fawx/auth.db`
- `/Users/joseph/.fawx/credentials.db`
- `/Users/joseph/.fawx/fleet/tokens.json`
- `/Users/joseph/.fawx/devices.json`
- `/Users/joseph/.fawx/*.salt`
- database files, key material, credentials, binary blobs, screenshots, build
  artifacts, derived-data paths, and generated caches

All corpus artifacts produced from this bakeoff should stay local and untracked.

Recommended local artifact root:

- `/Users/joseph/fractal-tokenizer-checkout/fractal-tokenizer/benchmarks/.local/`

Nothing from that directory should be committed.

## Corpus Shape

The first local-first bakeoff should target about `120` documents.

Recommended bucket targets:

1. Operational logs: `36`
2. Structured JSONL/session text: `24`
3. Rust source: `24`
4. Swift source: `12`
5. Markdown docs/specs: `24`

This keeps the first pass balanced while still leaning into the local
operational-text advantage.

## Sampling Rules

### Operational logs

Sampling unit:

- contiguous line windows

Window target:

- `4 KiB` to `12 KiB` per document
- roughly `80` to `250` lines when possible

Rules:

- preserve original line order
- do not stitch unrelated files into one document
- keep each document from a single source file
- prefer windows with repeated operational structure
- dedupe obvious exact duplicates

Suggested sub-buckets:

- `logs.repetition_heavy`
- `logs.operational_mixed`

### Structured JSONL/session text

Sampling unit:

- contiguous line windows from a single `.jsonl` file

Window target:

- `2 KiB` to `10 KiB`
- roughly `25` to `150` lines

Rules:

- preserve original line order
- do not parse/reformat JSON before tokenization
- keep raw text exactly as stored on disk

Suggested sub-buckets:

- `jsonl.signals`
- `jsonl.journal`

### Rust and Swift source

Sampling unit:

- one whole file if the file is modest
- otherwise one contiguous source slice aligned to line boundaries

Window target:

- whole file if `<= 12,000` chars
- otherwise a contiguous slice between `4,000` and `12,000` chars

Rules:

- prefer hand-written source over generated/build output
- preserve comments and whitespace
- do not normalize indentation
- avoid test fixtures that are just synthetic benchmark text copies

Suggested sub-buckets:

- `code.rust`
- `code.swift`

### Markdown docs/specs

Sampling unit:

- one whole document if modest
- otherwise one contiguous section-aligned slice

Window target:

- whole file if `<= 12,000` chars
- otherwise `4,000` to `12,000` chars aligned to heading boundaries when feasible

Rules:

- preserve code fences and formatting
- prefer substantive docs over tiny stubs
- exclude archived HTML mockups and binary assets

Suggested sub-buckets:

- `docs.spec`
- `docs.prose`

## Corpus Record Format

The bakeoff corpus should be emitted as local JSONL.

One record per document:

```json
{
  "id": "logs-server-0001",
  "source_family": "local_fawx",
  "split": "induction",
  "bucket": "logs.repetition_heavy",
  "source_path": "/Users/joseph/.fawx/server.log",
  "start_line": 1200,
  "end_line": 1360,
  "byte_len": 8192,
  "char_len": 8140,
  "text": "..."
}
```

Required fields:

- `id`
- `source_family`
- `split`
- `bucket`
- `source_path`
- `start_line`
- `end_line`
- `byte_len`
- `char_len`
- `text`

Optional later fields:

- `source_sha256`
- `sampling_note`

Current local runner rule:

- every record carries `source_family=local_fawx`
- every record is assigned deterministically to `split=induction` or
  `split=evaluation`
- vocab induction uses only the induction split
- verdicts and scorecards are computed on the evaluation split only

## Bakeoff Comparisons

For each corpus document, run:

1. Fractal tokenizer
   - frontier policy: `NoveltyAware`
   - packaging: current typed chunking path
2. Native tokenizer retokenization for each local official tokenizer:
   - `llama31`
   - `mistral7`
   - `mixtral8x7b`
   - `qwen25`
   - `phi3mini`

## Metrics Per Document

Required fractal-side metrics:

- `input_bytes`
- `input_chars`
- `frontier_token_count`
- `chunk_count`
- `avg_chars_per_frontier_token`
- `motif_reuse_count`
- `fallback.motif_hits`
- `fallback.unknown_motifs`
- `fallback.recursed_to_children`
- `fallback.byte_fallback_tokens`
- `roundtrip_ok`
- `chunk_utf8_ok`
- `collation_ok`
- `wall_time_ms`

Required native-side metrics per model:

- `native_token_count`
- `avg_chars_per_native_token`
- `retokenize_ok`
- `wall_time_ms`

Derived metrics:

- `compression_ratio_vs_<model>` =
  `native_token_count / frontier_token_count`

## Output Format

The bakeoff results should also be JSONL and remain local.

Recommended path:

- `/Users/joseph/fractal-tokenizer-checkout/fractal-tokenizer/benchmarks/.local/local_bakeoff_results.jsonl`

One result record per document, with nested per-model native metrics.

The current runner also prints:

- induction/evaluation document counts
- evaluation-only hard-gate counts
- `FAMILY_SUMMARY` rows
- evaluation-only `BUCKET_SUMMARY` rows
- evaluation-only verdict reasons

## Pass/Fail Gates

The first local-first bakeoff passes if:

- `100%` exact round-trip
- `100%` UTF-8-safe chunks
- `100%` deterministic native retokenization
- `100%` deterministic collation
- `0` unexpected byte-fallback spikes on normal local text
- strong compression on `logs.repetition_heavy`
- restrained `motif_reuse` on code/docs/JSONL where reuse would be suspicious

Immediate stop conditions:

- any round-trip failure
- any UTF-8 chunk failure
- any collation-order failure
- repeated high byte-fallback on ordinary UTF-8 text

## Scorecard

This bakeoff should be judged in layers, not by one number, and the judgment
should be applied to the held-out evaluation split rather than the induction
split.

### Hard Gates

These are strict pass/fail checks.

| Metric | Pass | Fail |
|---|---:|---:|
| `roundtrip_ok` | `100%` | any failure |
| `chunk_utf8_ok` | `100%` | any failure |
| `collation_ok` | `100%` | any failure |
| `fallback.byte_fallback_tokens` | `0` on normal UTF-8 docs | any unexpected spike |

Any hard-gate failure means the run is a failure.

### Bucket Scorecard

The bucket-level read should use medians plus manual review of outliers.

| Bucket | Healthy compression | Healthy reuse | Worry signs |
|---|---|---|---|
| `logs.repetition_heavy` | strong, ideally `>= 5x` median vs native; top docs often `>= 10x` | clearly nonzero is fine | weak compression or fallback spikes |
| `logs.operational_mixed` | moderate to strong, ideally `>= 2x` median | some reuse is fine | chaotic reuse on unrelated spans |
| `jsonl.signals` / `jsonl.journal` | moderate, about `1.5x` to `3x` | low to moderate | high reuse on structurally different records |
| `code.rust` / `code.swift` | modest, about `1.1x` to `3x` | mostly low | very high held-out compression or high reuse |
| `docs.spec` / `docs.prose` | modest, about `1.1x` to `3x` | mostly low | overcollapse or unusually huge held-out ratios |

### Run-Level Verdict

The bakeoff should emit a simple red/yellow/green verdict.

- `RED`
  - any hard-gate break
- `YELLOW`
  - hard gates pass, but the run is not yet healthy enough to trust
  - examples:
    - weak compression on `logs.repetition_heavy`
    - suspiciously high reuse on held-out non-log buckets
    - extreme held-out compression on non-log buckets
    - byte fallback appears on ordinary local UTF-8 text
- `GREEN`
  - hard gates pass
  - `logs.repetition_heavy` compresses strongly
  - code/docs/JSONL stay restrained
  - review rows look intuitive instead of pathological

The runner should print enough summary detail to explain the verdict:

- hard-gate counts
- per-bucket median compression ratio
- per-bucket median reuse
- suspicious non-log overcollapse count
- verdict reasons

## Manual Review Set

After the run, manually inspect:

1. top `20` highest compression-ratio evaluation documents
2. top `20` lowest compression-ratio evaluation documents
3. all evaluation documents with non-zero `byte_fallback_tokens`
4. all evaluation documents with suspiciously high `motif_reuse_count` outside log-heavy buckets
5. all evaluation documents with unusually high chunk counts

## Expected Outcome

If the tokenizer is ready for real-world use in the narrow tokenizer-track
sense, we should see:

- operational logs compress dramatically
- JSONL signals compress moderately and remain clean
- code/docs compress modestly without overcollapse
- multilingual and Unicode-bearing text remain lossless and UTF-8-safe

## Current Runner Status

The local-only runner now exists and currently:

1. materializes the local corpus JSONL
2. assigns each document to `induction` or `evaluation`
3. induces vocab from induction docs only
4. runs the fractal + native tokenizer comparisons
5. writes local JSONL results
6. emits evaluation-scoped summary rows and verdicts

The next methodological step after this local held-out mode is the hybrid
bakeoff with external source families.
