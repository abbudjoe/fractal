# Canonical Tokenizer + Recursive Overlay Spec

## Purpose

This spec defines the first post-postmortem architecture pivot for the
tokenizer program.

The recursive-tokenizer line reached a clear limit:

- recursion as the primary tokenizer substrate did not generalize on external
  held-out code and prose
- the engineering and evaluation stack did become robust and trustworthy

So the next architecture must preserve what mainstream tokenizers already do
well and only ask recursion to do the narrower job it showed evidence for:

- local structure
- repetition-sensitive reuse
- reversible compression on structured repetitive text

## Short Read

The new architecture is:

1. keep one production-proven canonical tokenizer as the always-valid base
2. tokenize all text through that base tokenizer first
3. build an optional recursive overlay on top of the canonical token stream
4. only activate the overlay when measured regularity is high enough
5. keep ordinary text on the plain canonical path

This is not:

- a tokenizer replacement
- a second incompatible token space
- a global fuzzy motif system

This is:

- a reversible structure and reuse layer over a stable substrate

## Why This Pivot Exists

The strongest empirical lessons from the previous line were:

- production-style tokenizers win by being stable first
- the recursive primitive had real signal on repetitive and templatic text
- the recursive primitive did not prove itself as a universal tokenizer
- the current opportunity is "stable canonical substrate plus structural
  memory", not "more aggressive raw-span tokenization"

See also:

- [recursive-tokenizer-postmortem.md](./recursive-tokenizer-postmortem.md)
- [sota-vs-fractal-tokenizer-architecture.md](./sota-vs-fractal-tokenizer-architecture.md)

## Hypothesis

If:

- all text is first mapped into a stable canonical token stream, and
- the recursive primitive is only used to detect and encode high-confidence
  local reuse over that stream,

then:

- we can preserve the robustness floor of mainstream tokenization, while
- recovering real wins on repetitive structured text, without
- regressing held-out code, prose, or multilingual behavior.

This is a different hypothesis from the failed one.

Old hypothesis:

- recursion should define the canonical tokenizer units

New hypothesis:

- recursion should augment a canonical tokenizer with reversible structural
  memory and local macro reuse

## Non-Goals

Phase 1 must **not** do any of the following:

- invent a second global tokenizer ABI
- replace the base tokenizer on general text
- learn a cross-document global recursive motif vocabulary
- add fuzzy neighborhood matching across documents
- change model input ids by default
- broaden immediately to universal code/prose optimization

Any of those would recreate the same scope creep that obscured the earlier
line.

## First Target

The first target is intentionally narrow:

- repetitive structured text
- operational logs
- JSONL or event records
- config-like repetitive text

This target is chosen because it is where the primitive showed the clearest
advantage in prior bakeoffs.

Code, prose, and multilingual text remain part of evaluation, but they are
guardrail domains in Phase 1, not optimization targets.

## Core Architecture

### 1. Canonical Tokenizer Is The Source Of Truth

One tokenizer provides the canonical base stream.

Phase 1 rules:

- choose a single production-proven base tokenizer implementation
- tokenize every document through it first
- preserve the exact base token ids and offsets
- treat the base stream as the irreversible contract floor

The first implementation should reuse the existing native tokenizer path in the
repo rather than introducing another tokenizer integration surface.

### 2. Recursive Overlay Operates On Base Tokens

The recursive system no longer operates on raw spans as the tokenizer
substrate.

Instead it operates on:

- contiguous spans of canonical token ids
- token-aligned record or line windows
- optional typed structure hints derived from stable text boundaries

This keeps the overlay aligned with a representation that already generalizes.

### 3. Overlay Is Optional And Reversible

The overlay is a reversible sidecar over the canonical token stream.

At a conceptual level:

```text
raw text
-> canonical tokenizer
-> canonical token ids
-> optional recursive overlay analysis
-> overlay document
```

And the overlay document must always be expandable back to the canonical token
ids exactly.

### 4. Activation Is Gated By Measured Regularity

The overlay must not activate because the system guessed a vague domain label.

It should activate only when measurable signals indicate reusable local
structure, such as:

- repeated token subsequences
- repeated line or record shapes
- repeated key orderings or delimiter patterns
- low-entropy local windows with high recurrence

If the gate is not met, the output is just the canonical path.

## Data Model

Phase 1 should use typed overlay records rather than implicit string
heuristics.

Illustrative shape:

```rust
struct CanonicalDocument {
    token_ids: Vec<u32>,
    offsets: Vec<TokenOffset>,
}

struct OverlayDocument {
    canonical: CanonicalDocument,
    mode: OverlayMode,
    segments: Vec<OverlaySegment>,
}

enum OverlayMode {
    Passthrough,
    LocalMacro,
}

enum OverlaySegment {
    BaseSlice { start: u32, len: u32 },
    MacroRef { macro_id: u32, span_len: u32 },
}

struct LocalMacro {
    macro_id: u32,
    token_ids: Vec<u32>,
    use_count: u32,
    kind: MacroKind,
}

enum MacroKind {
    RepeatedLine,
    RepeatedRecord,
    RepeatedFieldRun,
    RepeatedTokenSpan,
}
```

The exact names may change, but the invariants should not:

- overlay expansion is exact
- macros expand only to canonical token ids
- macro ids are document-local in Phase 1

## Scope Of Reuse

Phase 1 reuse should be:

- within-document
- exact over canonical token spans
- optionally record-aware

Phase 1 should not attempt:

- cross-document motif vocab
- approximate semantic matching
- global prototype identity

This is deliberate. The previous line already showed how quickly global
matching creates overcollapse risk.

## Gating Contract

The overlay gate should be explicit, typed, and measurable.

Possible signals:

- repeated span count above threshold
- repeated span byte or token mass above threshold
- repeated line-template ratio above threshold
- record-shape recurrence above threshold
- maximum span length and minimum reuse count guards

Illustrative config:

```rust
struct OverlayGateConfig {
    min_repeat_count: u32,
    min_repeated_token_mass: u32,
    min_template_reuse_ratio_bps: u16,
    max_macro_span_tokens: u16,
}
```

Important:

- these signals should route on regularity, not on domain names
- "logs" and "jsonl" are expected winners, but the gate should prove that from
  data

## Phase 1 Execution Plan

### Step 1. Shadow Overlay Only

Build the overlay generator as an offline sidecar.

It should:

- consume raw text
- produce canonical token ids
- optionally produce an overlay sidecar
- never alter the canonical ids used by the rest of the stack

This keeps the experiment narrow and low-risk.

### Step 2. Shadow Bakeoff

Extend the bakeoff runner to compare:

- canonical tokenizer alone
- canonical tokenizer plus recursive overlay sidecar

for the same held-out corpus.

The runner should report:

- overlay activation rate
- local macro hit docs
- average repeated token mass recovered
- compression ratio of overlay segments versus raw canonical stream
- expansion correctness
- domain bucket deltas

### Step 3. Guardrail Review

Inspect code, prose, and multilingual held-out buckets to confirm:

- overlay mostly stays off where it should
- no suspicious structure is being invented
- no regression is introduced by the gate

Only after that passes should we discuss model-facing consumption.

## Success Criteria

Phase 1 is a success if all of the following hold:

- exact expansion back to canonical token ids is 100%
- base tokenizer compatibility is unchanged when overlay is off
- overlay activates meaningfully on repetitive structured buckets
- repetitive structured buckets show material token-mass reduction over plain
  canonical token sequences
- code, prose, and multilingual buckets do not regress materially

Recommended success bars:

- structured repetitive buckets:
  - at least `1.5x` reduction in repeated token mass versus plain canonical
    stream
- guardrail buckets:
  - no more than `2%` median regression
- exactness:
  - `0` expansion failures
  - `0` UTF-8 or offset integrity failures

## Failure Criteria

This pivot should be treated as weak or failed if:

- overlay rarely activates even on structured repetitive buckets
- overlay only reproduces wins that could already be achieved by plain local
  deduplication with no recursive value
- guardrail domains regress materially
- the implementation drifts toward a second token ABI or broad fuzzy matching

In that case, the honest conclusion will be:

- recursion may still be useful above tokenization, but not as the overlay
  mechanism we hoped for here

## Why This Is The Right Next Experiment

This architecture is the first one that fully respects the main lessons of the
postmortem:

- stable canonical substrate first
- recursive structure only where it earned signal
- explicit gating instead of universal ambition
- narrow, falsifiable, reversible experiment surface

That makes it a better scientific probe than another rescue pass on the old
tokenizer formulation.
