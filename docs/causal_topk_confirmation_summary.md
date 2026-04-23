# Causal Top-K Confirmation Summary

## Question

Does the decode-safe causal prefix-top-k MoD lane still beat the local controls
when run through the standard Path 1 runner, and does it combine cleanly with
the Parcae/P20 recurrent scaffold?

## Artifact

- `artifacts/path1-causal-topk-confirmation-v1/summary.md`
- longer ladder: `artifacts/path1-causal-topk-longer-confirmation-v1/summary.md`

## Conditions

- seeds: `42,43,44,45,46`
- backend/dtype: `cpu` / `fp32`
- shape: `d_model=32`, `heads=4`, `layers=4`, `ffn_multiplier=2`
- budget: `train_steps=64`, `eval_batches=4`, `seq_len=16`, `batch_size=1`
- warmup: `warmup_eval_batches=1`, `warmup_train_steps=1`
- corpus: `fineweb-stage0-local-bench-9row-v1`

## Results

| Lane | Mean loss | Loss std | Train tok/s | Steps | Selected fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `causal-topk-route50-layer1` | 4.0076 | 0.0561 | 7738.7 | n/a | 0.512 |
| `causal-topk-route50-layer1-parcae-fixed3` | 4.0416 | 0.0857 | 4460.8 | 3.00 | 0.650 |
| `causal-topk-route50-layer1-p20-fixed5` | 4.0741 | 0.0701 | 2360.9 | 5.00 | 0.613 |
| `causal-topk-route50-layer1-p20-smart5` | 4.0773 | 0.0385 | 3235.2 | 2.00 | 0.600 |
| `p20-fixed5-proxy` | 4.1027 | 0.0855 | 3065.0 | 5.00 | n/a |
| `attention-control` | 4.1039 | 0.0843 | 9630.8 | n/a | n/a |
| `p20-smart5-proxy` | 4.1076 | 0.0527 | 3624.9 | 2.30 | n/a |
| `plain-parcae-fixed3` | 4.1163 | 0.0207 | 6435.6 | 3.00 | n/a |

## Interpretation

The decode-safe causal prefix-top-k MoD lane is confirmed as the strongest local
small-budget lane in this screen. It beats attention, plain Parcae, and the
local Parcae-P20 proxy controls while preserving autoregressive visibility.

The first hybrids are mixed:

- Adding causal top-k inside plain Parcae improves over plain Parcae, but does
  not beat standalone causal top-k.
- Adding causal top-k inside Parcae-P20 improves over the P20 controls, but does
  not beat standalone causal top-k and is much slower.
- The smart-halting P20 hybrid recovers throughput over the fixed P20 hybrid,
  but its quality is essentially tied or slightly worse here.

The useful conclusion is not "top-k plus everything." The current signal is:

- **Promote standalone causal top-k routing to the next confirmation/scaling
  lane.**
- Keep Parcae/P20 causal-top-k hybrids as secondary component experiments, not
  the lead form of the idea.

## Decision

- `causal-topk-route50-layer1`: `promote to ladder only`; do not scale as-is
  after the longer local confirmation
- `causal-topk-route50-layer1-parcae-fixed3`: `survives as hybrid control`
- `causal-topk-route50-layer1-p20-fixed5`: `component-only`
- `causal-topk-route50-layer1-p20-smart5`: `component-only`

## Longer Local Confirmation

The first longer local ladder increased the budget from `64` to `256` train
steps, increased eval batches from `4` to `8`, and compared the causal lane
against attention, paper train-time MoD, plain Parcae, and P20 proxy controls
over seeds `42,43,44`.

| Lane | Mean loss | Loss std | Train tok/s | Steps | Selected fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention-control` | 3.1623 | 0.0242 | 6739.5 | n/a | n/a |
| `p20-fixed5-proxy` | 3.1772 | 0.0508 | 2126.6 | 5.00 | n/a |
| `causal-topk-route50-layer1` | 3.2103 | 0.0418 | 5940.1 | n/a | 0.500 |
| `mod-train-topc` | 3.2206 | 0.0446 | 5797.4 | n/a | 0.500 |
| `p20-smart5-proxy` | 3.2313 | 0.0403 | 2642.8 | 2.08 | n/a |
| `plain-parcae-fixed3` | 3.2772 | 0.0078 | 4576.2 | 3.00 | n/a |

Interpretation:

- The short-run causal top-k win did not survive this longer local budget.
- `causal-topk-route50-layer1` still beat the paper train-time `mod-train-topc`
  reference, but lost to both attention and fixed P20 proxy.
- The lane remains interesting as a causal MoD adaptation, but `route50` at
  layer `1` should not be promoted directly to a scale run.
- The next disciplined move is a tiny routing ladder at the same longer budget:
  route fraction and placement only, with attention and P20 fixed controls.

## Hard-Skip / Prefix-Skew Test

Artifact:

- `artifacts/path1-causal-topk-bottleneck-ladder-v1/summary.md`

Question:

- Did `causal-topk-route50-layer1` fail because hard token skipping was too
  early, too aggressive, or position-skewed by the causal prefix rule?

Conditions:

- backend/dtype: `mps` / `fp32`
- shape: `d_model=192`, `heads=6`, `layers=8`, `ffn_multiplier=4`
- context/batch: `seq_len=256`, `batch_size=1`
- budget: `128` train steps, `4` eval batches
- seeds: `42,43,44`

Result:

| Lane | Mean loss | Loss std | Selected | First half | Second half | Mean pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `attention-control` | 2.6665 | 0.0118 | n/a | n/a | n/a | n/a |
| `mod-train-topc-route50-layer3` | 2.6674 | 0.0081 | 0.500 | 0.503 | 0.497 | 0.493 |
| `causal-topk-route50-layer3` | 2.6679 | 0.0078 | 0.488 | 0.469 | 0.508 | 0.513 |
| `causal-topk-route25-layer3` | 2.6679 | 0.0078 | 0.227 | 0.201 | 0.253 | 0.539 |
| `causal-topk-route75-layer3` | 2.6686 | 0.0068 | 0.749 | 0.763 | 0.734 | 0.495 |
| `causal-topk-route50-layer6` | 2.6703 | 0.0067 | 0.428 | 0.422 | 0.435 | 0.509 |
| `causal-topk-route50-layer1` | 2.6725 | 0.0087 | 0.499 | 0.510 | 0.487 | 0.497 |

Interpretation:

- Early hard routing is the worst placement tested. Moving the same route50
  rule from layer `1` to layer `3` improves mean loss from `2.6725` to `2.6679`.
- The best routed lanes still do not beat attention on mean loss; their deltas
  are small and inside seed noise, but the direction does not justify scale.
- The causal prefix rule does create some position skew at lower route fraction:
  route25 selected `20.1%` in the first half and `25.3%` in the second half.
  The skew is visible but not large enough to be the sole explanation.
- Softer routing did not rescue the lane: route75 layer3 was slightly worse than
  route50 and route25 at this rung.
- The most likely failure mode remains hard bypass / missed token refinement,
  with early placement making that bottleneck worse.

## Soft Partial-Update / Rotary Router Probe

Artifacts:

- `artifacts/path1-rgrp-soft-router-probe-v1/summary.md`
- `artifacts/path1-rgrp-soft-router-probe-v2/summary.md`

Question:

- Can the hard-bypass failure be avoided by replacing top-k selection with a
  continuous partial-update gate, and can a P20/RGRP-style rotary recurrent
  primitive act as a better causal gate controller?

Conditions:

- backend/dtype: `mps` / `fp32`
- shape: `d_model=192`, `heads=6`, `layers=8`, `ffn_multiplier=4`
- context/batch: `seq_len=256`, `batch_size=1`
- budget: `128` train steps, `4` eval batches
- seeds: `42,43,44`

Result:

| Lane | Mean loss | Loss std | Train tok/s | Gate / selected | Controller norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attention-control` | 2.6665 | 0.0118 | 18377.1 | n/a | n/a |
| `causal-topk-route50-layer3` | 2.6679 | 0.0078 | 1491.8 | 0.488 selected | n/a |
| `soft-gate-floor25-layer3` | 2.6681 | 0.0039 | 11498.5 | 0.309 gate | n/a |
| `soft-gate-floor50-layer3` | 2.6695 | 0.0037 | 12522.0 | 0.539 gate | n/a |
| `rotary-soft-gate-floor25-layer3` | 2.6780 | 0.0074 | 1594.1 | 0.267 gate | 14.475 |
| `rotary-soft-gate-floor50-layer3` | 2.6810 | 0.0065 | 1564.1 | 0.508 gate | 14.544 |

Interpretation:

- The first rotary pass exposed the right stability lesson: raw recurrent
  controller output can explode if used directly. Adding a Parcae-like output
  normalization made the controller numerically usable.
- A continuous MLP gate nearly matched hard top-k and attention while avoiding
  identity skips, but it did not beat attention on mean loss at this rung.
- The rotary recurrent gate was stable after normalization but materially worse
  than both attention and the simple MLP gate. As tested, P20/RGRP is not a
  better standalone token router.
- The useful idea to carry forward is not rotary top-k routing; it is the
  stability pattern from Parcae-RGRP: normalize recurrent controller state
  before using it to modulate loop injection or update magnitude.

## Next Tests

Use this family as a disciplined candidate, not a combinatorial branch:

- route fraction ladder: `0.25`, `0.5`, `0.75`
- layer placement: early, middle, late, and two-layer placement
- repeat only winners against attention and Parcae-P20/RGRP controls
- MaxText/TPU implementation only if a longer-budget routing configuration
  beats attention and fixed P20 locally
- optional later: distill soft router margins or hidden deltas, not binary masks
