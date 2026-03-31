# Promotion Policy

This policy defines how variants move between lanes.

It exists to keep the harness disciplined:
- one run answers one question
- one variant version changes one lever
- promotion is earned by explicit gates
- timeouts and `NaN` results count as outcomes

This policy applies prospectively. Historical labels in the tracker remain valid until the next qualifying run on the relevant lane.

## Lane Model

### Benchmark Lane

Purpose:
- keep a stable reference for comparison

Rules:
- benchmark variants do not mutate
- benchmark variants do not composite
- benchmark variants do not get promoted or retired

Current benchmark:
- `p1_contractive_v1`

### Bullpen Lane

Purpose:
- cheap proving-ground entry for new ideas

Default lane:
- `proving_ground_baseline`

### Validation Lane

Purpose:
- bounded stress for promising but unproven variants

Default lanes:
- `minimal_stress_lane`
- `lighter_intermediate_stress`

### Winner Lane

Purpose:
- confirm production-ready behavior under heavier load

Default lane:
- `full_medium_stress`

## Hard Gates

Metrics are evaluated in this order:
1. completion status
2. numeric health
3. fitness
4. stability
5. perplexity
6. ARC
7. throughput

`NaN` or `Inf` in stability or perplexity is an automatic numeric failure.

### 1. Bullpen -> Validation

A bullpen variant is promoted only if all of the following are true:
- completed within timeout
- no numeric failure
- `fitness >= 0.35`
- `stability <= 1.50`
- `perplexity <= 8.00`
- `ARC >= 0.15`

Bullpen variants get one bounded mutation pass instead of promotion if all of the following are true:
- completed within timeout
- no numeric failure
- `0.20 <= fitness < 0.35`
- `perplexity <= 12.00`

Bullpen variants are retired if any of the following are true:
- timeout before completion
- numeric failure
- `fitness < 0.20`
- `perplexity > 12.00`
- `ARC < 0.05`

### 2. Validation -> Upper Cohort

A validation-lane variant is promoted to Upper Cohort only if all of the following are true:
- completed within timeout
- no numeric failure
- `fitness >= 0.45`
- `stability <= 1.25`
- `perplexity <= 4.00`
- `ARC >= 0.25`

Validation-lane variants get one more bounded confirmation pass if any of the following are true:
- completed within timeout
- no numeric failure
- `0.35 <= fitness < 0.45`
- or `ARC < 0.25`
- or `stability > 1.25`

Validation-lane variants are retired or sidelined if any of the following are true:
- timeout during training
- numeric failure
- repeated timeout on the same lane

Special case:
- if training and stability complete but timeout occurs during perplexity or ARC evaluation, classify the variant as `eval-constrained` and allow one rerun with a reduced eval budget before retirement

### 3. Validation -> Co-Leader Contender

A validation-lane variant may skip directly to Co-Leader Contender only if all of the following are true:
- completed within timeout
- no numeric failure
- `fitness >= 0.55`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.40`

### 4. Winner Lane -> Co-Leader

A winner-lane variant is promoted to Co-Leader only if all of the following are true:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.57`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.45`

Tie-breakers between Co-Leaders:
1. higher ARC
2. lower perplexity
3. lower stability score
4. higher throughput

### 5. Fractal Core Co-Leader

A variant is classified as `Fractal Core Co-Leader` only if all of the following are true:
- already qualifies as Co-Leader
- `ARC >= 0.70`
- `stability <= 0.50`

This is the production-ready fractal designation.

## Failure Classifications

Every failed or sidelined run must be classified as exactly one of:
- `numeric-failure`
- `train-timeout`
- `eval-constrained`
- `low-signal`
- `runtime-cost`

These classifications should appear in the tracker notes.

## Versioning Rules

- every lever mutation creates a new versioned variant name
- a version gets at most one rerun on the same lane unless the rerun is solely to fix infrastructure
- retired variants return only through a new version with a new root-cause lever
- benchmark variants never fork in place

## Current Operating Policy

Based on current evidence:
- benchmark lane: `p1_contractive_v1`
- winner lane: `p1_fractal_hybrid_v1`, `p1_fractal_hybrid_composite_v1`, `logistic_chaotic_map_v1`
- validation lane: `p3_hierarchical_v1`, `b2_stable_hierarchical_v1`
- bullpen lane: no active additions until winner-lane questions narrow
- retired or tokenizer-track only: `ifs`, `generalized_mobius`, `julia`, `mandelbox`, retired squaring-family variants

## Discipline Rules

- do not widen the bullpen while winner-lane questions remain unresolved
- do not compare variants across different lanes as if they were apples-to-apples
- do not promote a variant on hope, only on completed metrics
- do not keep retrying the same version after two failures on the same question
- do not mutate benchmark variants
