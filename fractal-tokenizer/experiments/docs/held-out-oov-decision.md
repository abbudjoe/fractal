# Held-Out OOV Decision

## Decision Point

This document records the fork in the road exposed by the first honest
held-out local bakeoff.

The earlier local-first bakeoff was useful for:

- contract hardening
- UTF-8 safety
- collation correctness
- pretrained tokenizer integration
- proving the packaging and model-facing stack worked on real local text

But once the runner was changed to:

- split documents into `induction` vs `evaluation`
- induce vocab from `induction` docs only
- score verdicts on `evaluation` docs only

the old local-first `GREEN` result did not survive.

## Observed Result

Held-out local bakeoff result:

- `BAKEOFF_DOCUMENTS=120`
- `BAKEOFF_INDUCTION_DOCUMENTS=64`
- `BAKEOFF_EVALUATION_DOCUMENTS=56`
- `BAKEOFF_VERDICT=YELLOW`
- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=56`

Interpretation:

- the control-plane contracts are healthy
- the held-out tokenizer contract is not
- every evaluation document fell through to byte fallback
- the fractal path lost badly to native tokenizers on held-out local text

This means the current stable vocab/frontier contract is too
memorization-heavy.

## Root Cause Read

The current induction path learns exact motif digests from induction documents.

That works well when evaluation text matches induction text closely enough to
reuse those digests.
It fails when evaluation text is novel at the motif level, because the fallback
ladder is too shallow:

1. known motif
2. recurse to children
3. bytes

In practice, that means the contract has no strong model-facing representation
for novel but still structured held-out content.

The failure is therefore architectural, not cosmetic:

- not a scoring bug
- not a bakeoff-data problem
- not a reason to loosen the verdict

The missing primitive is a held-out-safe OOV contract.

## Decision

The next phase should focus on held-out OOV behavior, not broader bakeoff
coverage.

We are explicitly choosing two complementary fixes:

1. `Compositional recurring-submotif vocab`
2. `Typed lexical fallback above bytes`

And for lexical fallback we are explicitly choosing **Option B**:

- typed lexical fallback
- not a closed memorized lexical vocab

That means unknown structural spans should degrade to typed lexical atoms before
they degrade to raw bytes.

## Why These Two

### Compositional recurring-submotif vocab

This changes vocab induction from exact memorized motifs to reusable internal
substructure.

Expected effect:

- held-out documents can still reuse known internal building blocks even when
  the full parent span is novel

### Typed lexical fallback above bytes

This gives the tokenizer a deterministic non-byte path for novel content that
still has meaningful local structure.

Expected effect:

- held-out documents stop collapsing directly to bytes
- words, identifiers, numbers, punctuation, and whitespace survive as typed
  lexical units

Together, these two changes create the desired ladder:

1. known motif
2. known submotif cover
3. typed lexical atoms
4. bytes only as a final lossless floor

## Why Not Other Moves First

We are **not** choosing these as the next move:

- broader external bakeoff coverage first
- loosening the held-out verdict
- expanding induction data and hoping the issue disappears
- approximate/template matching first

Those may be useful later, but they do not address the revealed contract flaw.

## Success Criteria For The Next Phase

The next phase is successful if a rerun of the held-out bakeoff shows:

- `byte_fallback_docs` drops sharply
- held-out frontier token counts stop exploding
- non-log held-out documents no longer lose catastrophically to native
  tokenizers
- exact round-trip remains perfect
- UTF-8 and collation guarantees remain intact

## Follow-Up Inspection And New Read

After the first OOV hardening pass landed, deeper inspection showed a second
structural flaw:

- the so-called compositional vocab was still keyed by exact `record.text`
- that meant held-out generalization still depended mainly on exact literal
  recurrence
- typed lexical fallback prevented byte collapse, but could not create real
  structural reuse on its own

The next rescue hypothesis was therefore:

- keep typed lexical fallback as the safety floor
- change motif identity so that held-out spans can match recurring lexical
  shapes, not only exact induction literals

## First Result After Structural Shape Aliases

Held-out local bakeoff after adding shape-based structural aliases:

- `byte_fallback_docs=0`
- `motif_hit_docs=15/57`
- `shape_hit_docs=15/57`

Bucket read:

- `jsonl.signals` improved materially and became a strong held-out win
- `logs.operational_mixed` stayed a modest win
- `code.rust`, `code.swift`, and `docs.spec` remained below native-tokenizer
  parity and were still dominated by lexical fallback

Interpretation:

- structural aliasing is a real improvement
- it is not yet sufficient to save the primitive on general held-out code/docs
- the kill criterion remains active

## Follow-On Specs

This decision is carried forward in:

- [primitive-kill-criterion.md](./primitive-kill-criterion.md)
- [compositional-motif-vocab-spec.md](./compositional-motif-vocab-spec.md)
- [typed-lexical-fallback-spec.md](./typed-lexical-fallback-spec.md)

## Clustered Structural Induction Result

The next empirical step after the control-plane diagnosis was:

- keep runtime matching strict
- add clustered structural induction over coarse state bucket + length bucket +
  lexical shape
- treat cluster membership as an exact prototype hit
- do **not** add neighborhood or approximate matching yet

This was meant to test whether induction itself was the missing layer.

Focused regression result:

- a held-out shape-equivalent function example recovered real prototype hits in
  `motif-only` mode

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2`:

- `full` mode:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=5`
  - `lexical_only_docs=42`
- `motif-only` mode:
  - `exact_motif_hit_docs=1`
  - `prototype_hit_docs=5`
  - `lexical_only_docs=52`

Interpretation:

- clustered structural induction is real and not just a fake signal
- it reduced pure lexical collapse somewhat
- it did **not** materially improve held-out code/docs bucket medians
- the bottleneck remains the motif identity surface itself, not only induction

Updated read:

- `#2` worked as a partial lift
- `#1` is now justified as the next architectural move
- if stronger prototype-primary identity still fails to move held-out code/docs,
  the kill criterion gets much closer to firing

## Next Spec

The next decision is carried forward in:

- [prototype-primary-identity-spec.md](./prototype-primary-identity-spec.md)

## Prototype-Primary Identity Result

The next empirical step was to make prototype clusters the primary motif
identity surface while keeping matching strict:

- prototype membership first
- recurse to children
- typed lexical fallback
- no literal rescue
- no shape rescue
- no approximate neighborhood matching

Focused regression results:

- prototype-primary held-out shape-equivalent text recovered prototype hits
- prototype-primary vocab persistence round-tripped with explicit mode
- faceoff and model-facing suites stayed green

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2` in
`prototype-primary` mode:

- `exact_motif_hit_docs=0`
- `prototype_hit_docs=5`
- `lexical_only_docs=52`
- `code.rust=0.81`
- `code.swift=0.90`
- `docs.spec=0.76`
- `jsonl.signals=1.19`
- `logs.operational_mixed=1.07`
- hard-gate failures: `0`

Interpretation:

- the prototype-primary contract is real and technically sound
- it did **not** materially move the held-out code/docs ceiling
- clustered induction plus prototype-primary identity still leaves most
  evaluation documents effectively lexical-only

Updated read:

- the bottleneck is deeper than exact-vs-prototype identity surface alone
- the current tokenizer control plane still cannot turn primitive state into
  broadly reusable held-out motifs for code/docs
- further rescue work should be tightly limited
- primitive comparison or a more radical tokenizer-architecture pivot is now
  better justified than additional local rescue tuning

## State-Signature Prototype Induction Result

The next empirical step moved prototype induction onto a richer typed
`StateSignature` surface carried directly in `TokenRecord`.

This changed prototype clustering from:

- coarse state bucket
- coarse length bucket
- lexical shape

to:

- coarse length bucket
- coarse typed state signature

with exact matching still preserved.

Focused regression result:

- two records with the same typed state signature but different lexical shapes
  now induce one shared prototype cluster

Held-out legacy bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2`:

- `exact_motif_hit_docs=1`
- `prototype_hit_docs=29`
- `lexical_only_docs=27`
- `code.rust=0.83`
- `code.swift=0.93`
- `docs.spec=0.92`
- `jsonl.signals=113.26`
- verdict: `YELLOW`

Motif-only diagnostic:

- `prototype_hit_docs=29`
- `lexical_only_docs=28`
- `jsonl.signals=27.45`
- verdict: `YELLOW`

Interpretation:

- this is the first control-plane pass that materially increased held-out
  structural reuse
- it also moved docs/code upward in the right direction
- but it introduced obvious false-positive overcollapse on structured JSONL
- the problem is no longer "no structural hits"; it is now "structural hits
  are too permissive on some buckets"

Updated read:

- the typed state-signature surface is a real improvement and should stay
- the current line is not yet seaworthy
- if we continue on this path, the next fix must be prototype precision /
  anti-overcollapse guardrails, not looser matching

## Adaptive Signature Granularity Result

The next empirical step kept the typed state-signature surface, but changed the
precision mechanism:

- keep coarse prototypes when they pass admission
- refine overly broad coarse clusters into finer state-signature prototypes
- keep the stable default on coarse mode
- run adaptive granularity only as an explicit experiment mode

Focused regression result:

- a broad short-span cluster is rejected in `Coarse` mode
- the same source records refine into multiple `fine::` prototype clusters in
  `Adaptive` mode

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2`:

- `coarse`
  - `prototype_hit_docs=3`
  - `lexical_only_docs=42`
  - `code.rust=0.81`
  - `code.swift=0.90`
  - `docs.spec=0.76`
  - `jsonl.signals=3.45`
  - verdict: `GREEN`
- `adaptive`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`

Interpretation:

- adaptive refinement is meaningfully better than blunt guardrails
- it keeps a real share of the recall lift from state-signature induction
- it also avoids the catastrophic structured JSONL collapse from the raw
  state-signature pass
- but it still trips the non-log overcollapse gate and only modestly improves
  code/docs

Updated read:

- this line is now better instrumented and more explicit
- but it still does not clear the bakeoff gate
- the stable default should remain coarse
- adaptive refinement can stay available as an experiment mode
- the gate from candidate 2 to candidate 3 does not open

## Prototype Emission Gate Result

The last tightly scoped rescue attempt asked whether the remaining precision
problem lived at runtime emission rather than in induction or identity.

The experiment added a selective runtime gate that recurses to children when:

- a prototype hit is available
- the child frontier fully covers the parent
- every child is already structurally recoverable

Focused regression result:

- `Direct` emits one parent prototype
- `Selective` recurses and emits the two child prototypes

Held-out local bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2` with
adaptive granularity:

- `direct`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`
- `selective`
  - `prototype_hit_docs=19`
  - `lexical_only_docs=31`
  - `code.rust=0.83`
  - `code.swift=0.91`
  - `docs.spec=0.78`
  - `jsonl.signals=7.60`
  - verdict: `YELLOW`

Interpretation:

- the runtime emission gate works in isolation
- but it is a complete field no-op on the held-out corpus
- that means runtime prototype emission is not the active bottleneck

Updated read:

- the remaining ceiling is upstream of emission
- the rescue line of admission -> granularity -> emission is exhausted
- the next move should be a broader tokenizer-control-plane pivot or primitive
  comparison, not another local emission heuristic

## Prototype Precision Guardrails Result

The next empirical step added a dynamic prototype admission policy based on:

- distinct-text density
- repetition surplus
- span-size bucket

The goal was to keep the new state-signature prototype gains while filtering the
obvious false positives on structured JSONL.

Focused regression result:

- all-unique short clusters are rejected
- repeated short clusters are retained

Held-out legacy bakeoff result for `p1_fractal_hybrid_dyn-state-norm_v2`:

- `exact_motif_hit_docs=1`
- `prototype_hit_docs=3`
- `lexical_only_docs=42`
- `code.rust=0.81`
- `code.swift=0.90`
- `docs.spec=0.76`
- `jsonl.signals=3.45`
- verdict: `GREEN`

Interpretation:

- the guardrails successfully removed the overcollapse
- but they did so by erasing almost all of the new structural lift
- the system effectively fell back to the old pre-guardrail held-out profile

Updated read:

- the first precision gate was too blunt
- it does not earn progression to adaptive signature granularity
- if we continue on this control-plane line, the next candidate must be a more
  targeted precision function rather than a coarse admission cutoff

## Document-Local Motif Cache Result

After the atom-first substrate pass, the next contextual-reuse experiment added
an exact document-local motif cache:

- exact repeated UTF-8 spans only
- cache lifetime limited to a single document encode
- explicit mode, default off
- no global vocab persistence

Held-out local bakeoff on the lexical substrate:

- baseline `--local-cache off`
  - `prototype_hit_docs=2`
  - `local_cache_hit_docs=0`
  - `lexical_only_docs=21`
  - `code.rust=0.83`
  - `code.swift=0.96`
  - `docs.spec=0.77`
  - `jsonl.signals=5.16`
  - `logs.operational_mixed=1.10`
- experiment `--local-cache exact`
  - `prototype_hit_docs=2`
  - `local_cache_hit_docs=4`
  - `lexical_only_docs=20`
  - `code.rust=0.83`
  - `code.swift=0.96`
  - `docs.spec=0.77`
  - `jsonl.signals=5.16`
  - `logs.operational_mixed=1.10`

Hard gates stayed clean:

- `roundtrip_failures=0`
- `chunk_utf8_failures=0`
- `collation_failures=0`
- `byte_fallback_docs=0`

Interpretation:

- exact document-local reuse is real
- it produces a few honest contextual hits across logs, JSONL, Rust, and Swift
- but it does not materially move the held-out code/docs ceiling

Updated read:

- the missing opportunity is probably not a simple exact local cache
- local contextual memory may still matter, but not in this narrow form
- the next remaining high-leverage tokenizer-internal move is more likely a
  segmentation/substrate refinement than another local cache heuristic
