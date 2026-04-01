# Option Space Tracker

This tracker is the single place to see what the tokenizer program has already
tried, what remains untried, and where the remaining leverage appears to live.

The current evidence says the pipeline is no longer failing at the contract
floor. The hard gates are good, typed lexical fallback fixed byte collapse, and
the frontier/emission line has been exhausted. What remains is a missing
architectural layer for reusable held-out structure.

## Active Pivot

The active post-postmortem pivot is now:

- [canonical-tokenizer-recursive-overlay-spec.md](./canonical-tokenizer-recursive-overlay-spec.md)
- [shared-overlay-dictionary-spec.md](./shared-overlay-dictionary-spec.md)

This is intentionally **not** another attempt to rescue recursion as the
primary tokenizer substrate.

It reframes recursion as:

- a reversible structure and reuse overlay over a production-proven canonical
  tokenizer

The first experiment scope is narrow:

- document-local or record-local reuse on repetitive structured text
- no second token ABI
- no global fuzzy motif matching
- code, prose, and multilingual text are guardrail domains

Initial shadow-run read on `codex/post-tokenizer-pivot`:

- exact expansion back to canonical token ids is now clean
- local `jsonl.signals` is the clearest early win
  - median overlay ratio `1.27`
  - activation rate `1.00`
- local code is mildly positive
  - `code.rust` median overlay ratio `1.03`
  - `code.swift` median overlay ratio `1.05`
- local docs are effectively neutral
  - `docs.spec` median overlay ratio `1.00`
- hybrid guardrails stayed conservative
  - `external.prose.web` median overlay ratio `1.00`
  - `external.multilingual` median overlay ratio `1.00`
  - external code activated on some docs but stayed neutral at the median

Read:

- this is the first post-postmortem direction that is both exact and
  directionally useful
- the strongest current signal is still repetitive structured text, not
  universal code/prose improvement
- the clearest measured bottleneck is now transport overhead, not activation
  failure

Latest hybrid held-out read with `local-record-macro`:

- `jsonl.signals` median overlay ratio `2.27`
- `logs.operational_mixed` median overlay ratio `1.54`
- exact expansion failures `0`
- `OVERLAY_TRANSPORT_SUMMARY`
  - `transport_ratio = 1.27`
  - `definition_overhead_rate = 0.19`
  - `macro_definition_symbols = 32413`
  - `macro_ref_symbols = 4769`

Read:

- the overlay line is now earning its keep on the target buckets
- the next high-leverage layer is amortizing shared scaffold definitions across
  documents, not broadening activation heuristics

Latest `batch_local` shared-dictionary read on `codex/shared-overlay-dictionary-impl`:

- overall transport:
  - `document_local transport_ratio = 1.27`
  - `batch_local transport_ratio = 1.37`
  - `document_local definition_overhead_rate = 0.19`
  - `batch_local definition_overhead_rate = 0.12`
- target buckets:
  - `jsonl.signals` median transport ratio `2.27 -> 3.35`
  - `logs.operational_mixed` median transport ratio `1.54 -> 1.99`
- exact expansion remains `0` failures

Read:

- the shared-dictionary layer is the first clear transport-efficiency win after
  record-aware activation
- the remaining gap is now closer to the success bars in
  [shared-overlay-dictionary-spec.md](./shared-overlay-dictionary-spec.md)

Latest `batch_local + profitability gate` read on
`codex/shared-overlay-dictionary-impl`:

- default profitability floor: `min_net_gain_symbols = 8`
- overall transport:
  - `document_local transport_ratio = 1.27`
  - `batch_local transport_ratio = 1.38`
  - `document_local definition_overhead_rate = 0.19`
  - `batch_local definition_overhead_rate = 0.10`
- target buckets:
  - `jsonl.signals` median transport ratio `2.27 -> 3.34`
  - `logs.operational_mixed` median transport ratio `1.54 -> 1.99`
- exact expansion remains `0` failures

Read:

- profitability gating is worth keeping when it is tuned as a transport-layer
  floor rather than a blunt pruning pass
- the tuned gate improved overall transport efficiency without sacrificing the
  structured-text win buckets
- the next remaining overhead now looks more like scaffold factorization than
  dead-weight macro admission

## Tried By Layer

### Primitive

- `p1_fractal_hybrid_v1`
- `p1_fractal_hybrid_dyn-state-norm_v2`
- `b1_fractal_gated_v1`
- `p2_mandelbrot_v1`
- `b3_fractal_hierarchical_v1`
- `b4_universal_v1`
- `p1_contractive_v1`
- `p1_fractal_hybrid_composite_v1`
- `logistic_chaotic_map_v1`
- `p3_hierarchical_v1`
- `b2_stable_hierarchical_v1`
- `ifs_dyn-radius-depth_v1`
- `generalized_mobius_dyn-jitter-norm_v2`
- `julia_recursive_escape_v1`
- `mandelbox_recursive_dyn-escape-radius_v1`

Read:

- the tokenizer-local family found a stable leader in `p1`
- the broader primitive field did not break away in the honest held-out
  tokenizer bakeoff
- primitive choice is therefore not the main remaining lever

### Split / Segmentation

- `boundary-aware` split for `p1`
- `syntax-aware` split for lexical substrate

Read:

- this was the one serious rescue pass on span geometry
- syntax-aware improved the current lexical baseline slightly, but still did not
  materially move held-out code/docs
- segmentation alone is not the missing fix

### Motif Identity

- exact literal motif matching
- shape-based structural aliases
- clustered structural induction
- prototype-primary identity
- state-signature prototype induction
- adaptive prototype granularity
- prototype precision guardrails
- prototype emission gate

Read:

- state-signature induction finally increased held-out structural hits
- precision guardrails removed the overcollapse but erased most of the lift
- adaptive granularity kept some of the lift but still failed the gate
- selective runtime emission was a field no-op
- the current bottleneck is upstream of emission

### Fallback / OOV

- typed lexical fallback above bytes
- compositional recurring-submotif vocab
- motif-only ablation
- document-local motif cache

Read:

- this is the biggest hard win so far
- it fixed catastrophic byte collapse on held-out docs
- it did not by itself create enough reusable held-out structure for code/docs
- exact document-local cache produced a few honest contextual hits, but did not
  materially move the held-out medians

### Frontier / Emission Policy

- `GreedyKnown`
- `FinestKnown`
- `StateAware`
- `ReuseAware`
- `NoveltyAware`
- `SpanLengthAware`
- `Budgeted`
- `HybridStructural`

Read:

- `NoveltyAware` is the best frontier policy we found
- the other policies are useful comparators, but they do not change the
  held-out ceiling enough to be the next leverage point

### Packaging / Model-Facing

- chunking / model packaging
- native compatibility adapter
- HF-backed native tokenizer adapter
- file-backed pretrained tokenizer smoke tests
- padding, masking, and multi-document collation

Read:

- the packaging and model-facing stack is validated
- this layer is not currently the bottleneck

### Evaluation

- local-first bakeoff
- held-out induction/evaluation split
- scorecard and red/yellow/green verdicting
- real pretrained tokenizer integration against multiple model families

Read:

- evaluation is now honest enough to surface the real bottleneck
- the bakeoff pipeline is doing its job

## Remaining Untried By Layer

### Primitive

- new `v2` follow-ups from the broader primitive board
- a fresh `logistic_chaotic_map_v2`
- a fresh `p1_fractal_hybrid_composite_v2`
- harder confirmation lanes for `p3_hierarchical_v1` and `b2_stable_hierarchical_v1`

Read:

- these are still available as program-level options
- but the tokenizer evidence so far suggests they are lower leverage than a
  tokenizer-control-plane shift

### Motif Identity / Held-Out Matching

- signature-neighborhood matching
- prototype-neighborhood membership
- context-sensitive motif cache keys

Read:

- this is the clearest missing architectural opportunity
- the current system has exact identity, but not enough contextual reuse memory

### Fallback / OOV

- within-document reuse promotion before lexical fallback

Read:

- this layer still matters
- but the strict exact local-cache variant has already been tried and did not
  materially move the ceiling

### Packaging / Model-Facing

- learned embedding bridge into existing LMs
- actual training or fine-tuning on fractal-tokenized inputs

Read:

- this is important for downstream usefulness
- it is not the next tokenizer-internal rescue lever

### Evaluation

- hybrid external bakeoff on diversified corpora
- downstream LM comparison against native tokenization
- end-to-end task evaluation

Read:

- this is necessary for final truth
- it is less useful as the next architectural probe than a tokenizer-internal
  contextual-memory change

## Ranked Remaining Options

### 1. Signature-Neighborhood Matching

Layer:

- motif identity / held-out matching

Why it ranks first now:

- exact matching has been too brittle
- a neighborhood/prototype similarity layer could recover held-out structure
- but it is riskier because false positives can recreate the overcollapse we
  already fought off

Expected upside:

- more held-out structural hits without requiring literal identity
- a direct test of whether the missing problem is matching tolerance rather than
  substrate quality

Expected failure mode:

- false positives recreate the JSONL/code overcollapse line quickly

### 2. Hybrid External Bakeoff

Layer:

- evaluation

Why it ranks second now:

- the local held-out story is now fairly well understood
- if we keep spending cycles inside this architecture, we need to know whether
  the same ceiling holds outside the local `fawx` ecosystem

Expected upside:

- clearer truth signal about whether the tokenizer is fundamentally local-data
  coupled or generally limited

Expected failure mode:

- the same ceiling persists externally, which would strengthen the pivot case

Expected failure mode:

- selective matching becomes too permissive and collapses non-log buckets again

### 4. Hybrid External Bakeoff

Layer:

- evaluation

Why it ranks fourth:

- we need a diversified truth set before making final claims
- but this is mostly a measurement move, not a tokenizer rescue move

Expected upside:

- tells us whether the current ceiling is just local-fawx self-similarity

Expected failure mode:

- confirms the same ceiling on wider text without identifying a better internal
  mechanism

### 5. Learned Embedding Bridge

Layer:

- packaging / model-facing

Why it ranks fifth:

- this may be the right path if the tokenizer is only useful through an adapter
- but it moves us toward model integration, not tokenizer rescue

Expected upside:

- shows whether the tokenizer can be useful to existing LMs even if raw token
  counts are not the whole story

Expected failure mode:

- bridge complexity rises before we learn enough about the tokenizer itself

### 6. New Primitive `v2` Follow-Ups

Layer:

- primitive

Why it ranks sixth:

- the honest tokenizer field screen already washed out the current primitive
  family
- new variants are still worth keeping alive at the program level, but they are
  lower leverage than fixing contextual reuse

Expected upside:

- one of the new mutations might break the ceiling in a way the current family
  cannot

Expected failure mode:

- they cluster at the same held-out profile again, confirming the control plane
  is the real bottleneck

### 7. End-to-End Downstream LM Evaluation

Layer:

- evaluation

Why it ranks seventh:

- it is the final proof we ultimately need
- but it is downstream of the tokenizer question we are still trying to solve

Expected upside:

- tells us whether any tokenizer improvement is actually model-useful

Expected failure mode:

- expensive validation before the architecture question is settled

## Architectural Read

The evidence points to a missing contextual reuse plane.

The tokenizer already has:

- a real control plane
- exact round-trip correctness
- typed lexical fallback
- validated packaging and model-facing integration

What it still lacks is a strong way to turn held-out structure into reusable
document-local or syntax-local motifs without either:

- collapsing to bytes, or
- overcollapsing structured text

The hybrid external truth test sharpens that read:

- hard gates are strong
- external held-out structural hits are still zero
- external code remains below parity
- external prose remains below parity

That means the remaining option space is no longer “more small rescue tweaks.”
The honest remaining choices are:

- a much larger architecture pivot, or
- ending this tokenizer line as a robust but non-breakout structural attempt
