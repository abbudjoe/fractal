# Primitive Kill Criterion

## Purpose

This document records the point at which we should stop treating the current
recursive tokenizer primitive as a promising general tokenizer foundation.

The goal is not to be pessimistic.
The goal is to keep the experiment honest once the held-out local bakeoff has
become strict enough to falsify weak explanations.

## Current State

What the tokenizer primitive has already proved:

- it can produce a lossless recursive hierarchy
- it can expose real repetition on self-similar text
- it can support packaging, UTF-8-safe chunking, collation, and pretrained
  native-tokenizer integration
- it can avoid catastrophic held-out byte collapse after the OOV hardening pass

What it has **not** yet proved:

- that it learns reusable held-out structure strongly enough to beat native
  tokenizers on general real-world text
- that its structural motif vocabulary generalizes beyond exact or near-exact
  induction literals

## The Core Falsification Question

The primitive survives only if one more structural pass can produce held-out
structural reuse that is both:

- real
- and useful

That means:

- held-out documents should recover meaningful motif hits
- code/docs should move toward parity or better
- logs/structured operational text should remain wins
- the tokenizer should not depend mainly on lexical rescue layers

## Kill Criterion

After the next structural pass and a rerun of:

1. the held-out local bakeoff
2. the hybrid bakeoff with external variants

we should stop treating the primitive as a promising general tokenizer
foundation if the following remain true:

- held-out `fallback_motif_hits` remain near zero on most evaluation documents
- code/docs stay materially below native-tokenizer parity on token count
- the apparent wins are still concentrated in self-similar or highly repetitive
  local text
- the tokenizer is being carried primarily by lexical fallback rather than by
  reusable structural motifs
- hard contracts stay green, but the structural layer still fails to
  generalize

In that outcome, the honest conclusion is:

- the primitive is a useful hierarchical compression and analysis mechanism
- but not a seaworthy general tokenizer primitive

## Survival Criterion

The primitive remains alive if the next structural pass produces all of the
following:

- hard-gate contract success remains intact:
  - exact round-trip
  - UTF-8-safe chunks
  - collation correctness
  - no unexpected byte fallback
- held-out motif reuse becomes materially nontrivial
- at least one non-log held-out family moves to parity or better
- logs and structured operational text remain wins
- the improvement is not explained solely by lexical fallback

## Current Working Hypothesis

The current primitive has **not** been falsified yet.

The strongest remaining rescue path is:

1. fix motif identity so that it is structural rather than literal
2. keep typed lexical fallback as a safety net instead of the main carrier

If that still fails on held-out and hybrid evaluation, we should stop calling
this a promising general tokenizer primitive.
