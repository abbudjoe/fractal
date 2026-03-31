# Next Run Manifests

This file defines the next approved experiments on the shared `codex/next-run-manifests` branch.

The manifests below follow the current promotion policy:
- one run answers one question
- lever mutations get versioned names
- benchmark variants stay frozen
- same-preset comparisons are authoritative
- timeout and failure classes are outcomes, not noise

## 1. `p1_fractal_hybrid_composite_v2`

Exact question:
- Can the current fractal-core co-leader keep its ARC lead while a stability-focused mutation lowers the residual instability further without sacrificing fitness?

Variant / lever:
- Base variant: `p1_fractal_hybrid_composite_v1`
- Proposed execution name: `p1_fractal_hybrid_composite_v2`
- Lever type: stability-lever mutation on the compositing shell

Lane / preset:
- Lane: `winner`
- Preset: `full_medium_stress`

Backend / runtime:
- Backend: `cuda`
- Timeout: `none`
- Seeds: `42`, `43`, `44`

Success gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.57`
- `stability <= 0.50`
- `perplexity <= 2.20`
- `ARC >= 0.70`
- outcome class: `success`

Hold gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.57`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.45`
- but does not clear the fractal-core co-leader gate

Retire gate:
- timeout before completion
- numeric failure
- `fitness < 0.57`
- `ARC < 0.45`
- or repeated timeout on the same question

Why this is allowed:
- it is a bounded winner-lane mutation of an already proven co-leader
- it asks one question: whether the composited shell can be tightened without losing its current lead
- it respects the rule that benchmark variants stay frozen

## 2. `logistic_chaotic_map_v2`

Exact question:
- Can the newly promoted logistic contender keep its winner-lane fitness while being composed inside the hybrid shell without exposing hidden instability?

Variant / lever:
- Base variant: `logistic_chaotic_map_v1`
- Proposed execution name: `logistic_chaotic_map_v2`
- Lever type: compositing test inside the hybrid composite shell

Lane / preset:
- Lane: `winner`
- Preset: `full_medium_stress`

Backend / runtime:
- Backend: `cuda`
- Timeout: `none`
- Seeds: `42`, `43`, `44`

Success gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.57`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.45`
- outcome class: `success`

Hold gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.45`
- `stability <= 1.25`
- `perplexity <= 4.00`
- `ARC >= 0.25`
- but it does not clear the co-leader gate yet

Retire gate:
- timeout before completion
- numeric failure
- `fitness < 0.45`
- `ARC < 0.25`
- or repeated timeout on the same question

Why this is allowed:
- it is a winner-lane question about compositional robustness, not a new bullpen expansion
- the current evidence already justifies a head-to-head with the hybrid shell
- the run is same-lane, same-preset, and therefore authoritative for leaderboard use

## 3. `p3_hierarchical_v1`

Exact question:
- Does the hierarchical compressor hold up under a harder confirmation lane strongly enough to become a winner-lane candidate?

Variant / lever:
- Base variant: `p3_hierarchical_v1`
- Lever type: unchanged hierarchical compressor

Lane / preset:
- Lane: `validation`
- Preset: `full_medium_stress`

Backend / runtime:
- Backend: `cuda`
- Timeout: `none`
- Seeds: `42`, `43`, `44`

Success gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.57`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.45`
- outcome class: `success`

Hold gate:
- completed on `full_medium_stress`
- no numeric failure
- `fitness >= 0.55`
- `stability <= 0.80`
- `perplexity <= 2.20`
- `ARC >= 0.40`
- but it does not yet justify winner-lane promotion

Retire gate:
- timeout before completion
- numeric failure
- `fitness < 0.55`
- `ARC < 0.40`
- or repeated timeout on the same question

Why this is allowed:
- the tracker already marks `p3` as a contender, so a harder confirmation lane is the next disciplined step
- this run is meant to decide promotion readiness, not to mutate the primitive
- the policy explicitly allows bounded confirmation passes for validation-lane variants

## 4. `b2_stable_hierarchical_v1`

Exact question:
- Can a reduced eval budget recover the validation run from `eval-constrained` status without changing the primitive?

Variant / lever:
- Base variant: `b2_stable_hierarchical_v1`
- Lever type: unchanged stable hierarchical blend

Lane / preset:
- Lane: `validation`
- Preset: `minimal_stress_lane`

Backend / runtime:
- Backend: `cuda`
- Timeout: `20m`
- Seeds: `42`, `43`, `44`

Success gate:
- completed within timeout
- no numeric failure
- no eval-constrained failure
- `fitness >= 0.45`
- `stability <= 1.25`
- `perplexity <= 4.00`
- `ARC >= 0.25`
- outcome class: `success`

Hold gate:
- completed within timeout
- no numeric failure
- no eval-constrained failure
- `fitness >= 0.35`
- `ARC >= 0.25`
- but it does not yet clear the upper-cohort gate

Retire gate:
- repeated eval-constrained failure
- timeout during training
- numeric failure
- `fitness < 0.35`
- `ARC < 0.25`

Why this is allowed:
- the current tracker already classifies `b2` as `eval-constrained`
- the policy allows one rerun with a reduced eval budget before retirement
- this run changes the evaluation budget, not the primitive

## Shared Discipline Notes

- `p1_contractive_v1` remains frozen and is not part of this manifest set.
- The four manifests above are intentionally narrow and decision-complete.
- Any future run should be blocked until the current winner-lane questions narrow further.
- Tracker updates remain human-reviewed; these manifests are only the planned run contracts.
