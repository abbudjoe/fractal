# Recursive Tokenizer Postmortem

## Executive Summary

This tokenizer line produced a real engineering system, but it did not produce
a convincing real-world structural tokenizer.

What succeeded:

- exact round-trip correctness
- UTF-8-safe chunking
- stable packaging and model-facing integration
- honest held-out evaluation
- robust OOV handling through typed lexical fallback

What did not succeed:

- reusable held-out structural motif discovery on real external data
- competitive held-out performance on external code and prose
- evidence that recursive primitives, used as the primary tokenizer substrate,
  yield a breakout real-world tokenizer

The key conclusion is not that recursion is useless for language. The key
conclusion is that this specific formulation likely employed recursion in the
wrong role.

## Question We Tested

The operational thesis under test was:

- recursive primitives can serve as the primary tokenizer mechanism
- by recursively partitioning spans and inducing reusable motifs
- in a way that generalizes across held-out domains

This is narrower than the broader idea that recursion is a useful inductive
bias for language.

## What We Built

By the end of this line, we had:

- a real tokenizer control plane
- deterministic vocab induction and persistence
- typed lexical fallback above bytes
- packaging and chunking contracts
- model-facing document and batch contracts
- native tokenizer integration against official pretrained tokenizer artifacts
- honest bakeoff infrastructure with held-out scoring
- a hybrid local plus external truth-test runner

This matters because the negative result came from a mature enough
implementation to trust the evaluation.

## What The Evidence Said

Early results looked promising on:

- synthetic repetition-heavy inputs
- self-similar local corpora

But the picture changed once evaluation became honest:

- induction and evaluation were separated
- held-out local bakeoffs were added
- then a hybrid local plus external bakeoff was added

The most important final result came from the hybrid truth test on the stable
baseline:

- `primitive=p1_fractal_hybrid_dyn-state-norm_v2`
- `substrate=lexical`
- `split_policy=boundary-aware`
- `identity_mode=legacy`
- `prototype_granularity=coarse`
- `local_cache=off`

Result:

- hard gates all passed
  - `roundtrip_failures=0`
  - `chunk_utf8_failures=0`
  - `collation_failures=0`
  - `byte_fallback_docs=0`
- but structural generalization failed
  - `exact_motif_hit_docs=0`
  - `prototype_hit_docs=0`
  - `external_structural_hit_docs=0`

Held-out external medians:

- `external.code.python=0.89`
- `external.code.js_ts=0.80`
- `external.prose.web=0.66`
- `external.multilingual=1.08`

This means the tokenizer became robust, but not structurally convincing.

## What Was Falsified

The experiments strongly argue against this thesis:

- recursive primitives, used as the primary tokenizer substrate over spans,
  will discover reusable motifs that generalize well enough to act as a strong
  real-world tokenizer across code, docs, prose, and structured text

More concretely, the following formulation looks weak:

1. build a recursive span tree
2. induce motif vocab over those spans
3. match held-out spans back to those motifs
4. expect that representation to outperform or match strong native tokenizers

This is the formulation that reached the kill boundary.

## What Was Not Falsified

The experiments do **not** show that recursion itself is useless.

They do **not** show that recursive primitives cannot help with:

- hierarchical contextual memory
- compression of self-similar local structure
- model-side structural abstractions
- latent or adapter representations over a stable substrate

So the broader thesis:

- recursion is a useful inductive bias for language structure

is still plausible.

What looks specifically weak is:

- recursion as the mechanism that defines the canonical tokenizer units

## Square Peg, Round Hole

The evidence suggests that we were likely using recursion in the wrong place in
the stack.

Robust tokenization for real-world code and prose appears to want:

- a very stable canonical substrate
- conservative coverage
- low-risk segmentation

Our line asked recursion to do more than that:

- define the units
- define the reuse surface
- generalize those units across held-out domains

That appears to be the square peg.

The more promising role for recursion is likely:

- a structural layer on top of a stable substrate

rather than:

- the substrate itself

## What We Tried Before Reaching This Point

We did not stop after one or two weak results. We tried:

- multiple primitive families
- multiple frontier policies
- compositional vocab variants
- prototype identity variants
- state-signature induction
- precision guardrails
- adaptive granularity
- atom-first substrate
- document-local cache
- syntax-aware segmentation
- signature-neighborhood matching

Those experiments improved robustness and sharpened diagnosis, but they did not
change the held-out external conclusion.

## Final Read

The correct high-level reading is:

- the engineering work succeeded
- the evaluation infrastructure succeeded
- the specific recursive-tokenizer thesis did not

That means this line was valuable even if it does not become the production
tokenizer direction. It carried us far enough to discover where the idea stops
working.

## Recommendation

Do not continue spending cycles on narrow rescue heuristics for this tokenizer
formulation.

Keep:

- the bakeoff runner
- the held-out evaluation discipline
- the model-facing integration surface
- the lessons about substrate stability and contextual reuse

Retire:

- recursion as the primary tokenizer-unit discovery mechanism for this line

If recursion is revisited later, it should probably be in a new architecture
where it sits above a canonical token substrate rather than replacing it.
