# Hybrid Bakeoff Spec

## Purpose

This document defines the next bakeoff after the local-first run.

The local-first bakeoff was valuable for:

- contract hardening
- UTF-8 safety
- collation correctness
- native-tokenizer integration
- proving the runner works on real local data

But it is not yet a fair generalization test because the corpus is heavily
self-similar:

- one product family
- one logging style
- one codebase vocabulary
- one docs/spec writing voice

The hybrid bakeoff fixes that by adding:

1. explicit source families
2. held-out evaluation
3. external corpus diversity

## Goals

The hybrid bakeoff should answer these questions:

1. Does the tokenizer stay robust on both local and external corpora?
2. Does compression remain selective when the corpus is not dominated by one
   ecosystem?
3. Do code/docs/JSONL still look sane once the evaluation set is held out from
   vocab induction?
4. Does the current `NoveltyAware` frontier still deserve to be the
   model-facing default under a more honest corpus design?

## Non-Goals

This phase still does **not** prove:

- downstream perplexity wins
- downstream task wins
- production inference wins

Those require a later model experiment.

## Core Design Changes

Compared with the local-first bakeoff:

- vocab induction must happen on a separate induction split
- evaluation documents must not be used during vocab induction
- results must be reported by `source_family` as well as by bucket

## Source Families

The bakeoff should use two top-level source families.

### 1. Local Fawx

These are the same local sources already validated in the local-first bakeoff:

- `/Users/joseph/.fawx/server.log`
- `/Users/joseph/.fawx/logs/*.log`
- `/Users/joseph/.fawx/journal.jsonl`
- `/Users/joseph/.fawx/signals/headless.jsonl`
- `/Users/joseph/fawx/engine/**/*.rs`
- `/Users/joseph/fawx/app/**/*.swift`
- `/Users/joseph/fawx/docs/**/*.md`

Use these to retain the operational-text and local-code realism we already
proved the runner can handle.

### 2. External

The external family should intentionally break the local ecosystem similarity.

Recommended source groups:

- prose/web text
  - [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- code
  - [codeparrot/github-code-clean](https://huggingface.co/datasets/codeparrot/github-code-clean)
- multilingual text
  - [oscar-corpus/oscar](https://huggingface.co/datasets/oscar-corpus/oscar)

These are good candidates because they diversify:

- writing style
- domain
- language
- code vocabulary

The exact external subset can remain modest for the first hybrid pass.

## Corpus Shape

The first hybrid bakeoff should target about `240` documents total.

Recommended split:

- `120` local
- `120` external

Within each family, keep multiple buckets.

### Local family target

Use the same local-first bucket mix:

1. `logs.repetition_heavy`
2. `logs.operational_mixed`
3. `jsonl.signals`
4. `jsonl.journal`
5. `code.rust`
6. `code.swift`
7. `docs.spec`

### External family target

Recommended first external buckets:

1. `external.prose.web`
2. `external.code.python`
3. `external.code.js_ts`
4. `external.multilingual`

If the chosen external code source is not naturally language-labeled at sample
time, the runner may use file extensions to derive sub-buckets.

## Required Record Fields

The corpus JSONL should add a required `source_family` field.

Example:

```json
{
  "id": "external-prose-0001",
  "source_family": "external",
  "bucket": "external.prose.web",
  "source_path": "hf://HuggingFaceFW/fineweb-edu/.../sample-0001",
  "start_line": 1,
  "end_line": 1,
  "byte_len": 7120,
  "char_len": 7092,
  "text": "..."
}
```

Required values:

- `source_family=local_fawx`
- `source_family=external_hf`

If we later add another external provider, use a new explicit family name.

## Induction vs Evaluation Split

This is the most important methodological change.

### Rule

Vocab induction and evaluation must use different documents.

### Recommended first split

- induction set: `50%`
- evaluation set: `50%`

Apply the split within each source family and bucket so the held-out set stays
balanced.

### Constraints

- no evaluation document may appear in vocab induction
- near-duplicate windows from the same source file should not be split across
  induction and evaluation when avoidable
- if documents come from the same parent file, prefer assigning the whole file’s
  windows to one side

### Output fields

Each corpus record should carry:

- `split = induction | evaluation`
- `source_family`
- `bucket`

## Success Criteria

The hybrid bakeoff should still use the hard contract gates from the
local-first bakeoff:

- `roundtrip_ok = true` for every evaluation document
- `chunk_utf8_ok = true` for every evaluation document
- `collation_ok = true` for every evaluation document
- no unexpected byte-fallback spikes

But the interpretive thresholds should be stricter.

### Healthy result

- local logs still compress strongly
- external prose compresses moderately
- external multilingual text stays lossless and UTF-8-safe
- external code remains restrained
- non-log buckets do not show implausibly huge ratios on held-out data

### Warning result

- code/docs/external prose still show extremely large compression ratios on the
  held-out split
- or reuse becomes suspicious outside obviously repetitive buckets

That should trigger a `YELLOW` or `RED` verdict depending on severity.

## Updated Verdict Logic

The hybrid bakeoff should tighten the current scorecard.

### RED

- any hard-gate failure

### YELLOW

- hard gates pass, but one or more of these holds:
  - weak compression on repetitive local logs
  - suspicious reuse on non-log buckets
  - extreme held-out compression on non-log buckets

Suggested first caution thresholds for held-out non-log buckets:

- if a non-log bucket median best ratio exceeds `20x`, flag it for caution
- if a non-log bucket median motif reuse exceeds `2`, flag it for caution

These thresholds are intentionally conservative for the first honest held-out
pass.

### GREEN

- hard gates pass
- repetitive local logs remain strong
- non-log buckets stay restrained on held-out data
- review set looks intuitive across both source families

## Manual Review Set

The hybrid bakeoff should review documents across both source families.

Always inspect:

1. top `20` highest compression evaluation docs
2. top `20` lowest compression evaluation docs
3. every evaluation doc with non-zero byte fallback
4. every non-log evaluation doc above the caution threshold
5. every multilingual evaluation doc with surprising chunk counts

## Recommended Output Sections

The runner should print:

- global hard-gate counts
- per-model totals
- per-family summary
- per-bucket summary
- verdict
- review rows

New required summaries:

- `FAMILY_SUMMARY source_family=...`
- held-out-only bucket summaries

## Next Implementation Step

The next runner evolution should add:

1. `source_family`
2. `split`
3. held-out vocab induction
4. external-source ingestion
5. stricter verdict thresholds for non-log held-out buckets

The local-first runner should remain available as a separate mode because it is
still useful for contract and regression hardening.
