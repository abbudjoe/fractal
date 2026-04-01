# Primitive Pivot Plan

## Purpose

This document makes the next fork explicit:

- one decisive rescue pass for `p1`
- then pivot to comparing other primitives if it fails

The goal is to avoid spending many more cycles locally patching a primitive
that may not hold up on general held-out text.

## Current Position

What is already true:

- the experiment pipeline is real
- held-out local bakeoff scoring is real
- model-facing integration is real
- OOV collapse is fixed
- structural aliasing helps `jsonl.signals`
- code/docs remain below parity

That means we now have enough infrastructure to compare primitives honestly.

## Allowed Remaining Rescue Budget For `p1`

Only one more serious rescue experiment should be attempted before pivoting:

- [boundary-aware-split-spec.md](./boundary-aware-split-spec.md)

Why this one is allowed:

- it targets the most plausible remaining structural flaw
- it changes the primitive contract, not just a threshold
- it is high signal

What is **not** allowed before the pivot:

- many small lexical tuning passes
- many shape alias variants
- repeated threshold fiddling
- long heuristic chains around split boundaries

## Pivot Trigger

If the boundary-aware split experiment fails to produce:

- materially more structural hits on held-out code/docs
- reduced dominance of lexical fallback on code/docs
- parity or better on at least one non-log code/docs bucket

then the next step is:

- run the same bakeoff pipeline on other available primitives

not:

- continue patching `p1`

This trigger has now fired for the first boundary-aware split pass:

- the held-out bucket medians were materially unchanged
- code/docs remained below parity
- lexical domination on code/docs remained effectively intact

## Primitive Comparison Plan

After the rescue budget is exhausted, run:

1. current `p1` baseline
2. next most plausible primitive candidates already available in the repo

Compare them on the same:

- held-out local bakeoff
- later hybrid bakeoff
- same packaging/model-facing stack where possible

## Success Standard For A Pivot

The pivot is justified if another primitive:

- preserves hard contracts
- produces more held-out structural hits
- reduces lexical domination on code/docs
- achieves parity or better on more non-log buckets

At that point, the right call is to move the pipeline forward with the better
primitive instead of defending `p1`.
