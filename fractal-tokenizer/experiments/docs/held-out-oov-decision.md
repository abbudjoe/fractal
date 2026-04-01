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
