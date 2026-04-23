# Parcae-RGRP Candidate Rule-Out Plan

## Purpose

The Parcae-RGRP-control lane is now the primary scaling lane. The proving
ground should stop broad promotion work and switch to falsifying remaining
non-Parcae-RGRP candidates. The goal is not to tune every primitive. The goal is
to identify whether any other paper family still has a credible claim to
displace or materially augment Parcae-RGRP-control.

## Current Lead

Primary lane:

- `parcae-rgrp-control-looped-attention`

Tuning candidates that belong in the main scaling lane:

- full RGRP-control, fixed recurrence budget
- full RGRP-control with vector-acceleration smart halting
- full RGRP-control with value scale `0.5` if value-ratio diagnostics are high
- thin RGRP-control with smart halting if width/throughput tradeoff survives TPU

These are no longer broad proving-ground candidates. They are calibration knobs
for the promoted lane.

## Rule-Out Criteria

A non-Parcae-RGRP candidate survives only if it satisfies at least one of:

- Quality: mean validation loss beats the local Parcae-RGRP proxy or is within a
  small tolerance while using materially less compute.
- Efficiency: loss is close to the Parcae-RGRP proxy while throughput or memory
  is clearly better.
- Orthogonality: the candidate solves a problem Parcae-RGRP does not address,
  such as long-context sparse attention, and should be evaluated in a separate
  harness rather than rejected by short-context loss.
- Mechanistic value: diagnostics reveal a reusable component that can improve
  Parcae-RGRP-control without carrying the full candidate.

A candidate should be ruled out or deferred if it:

- loses to attention or plain Parcae under matched conditions,
- is decode-unsafe for the target LM path,
- is far slower without a quality gain,
- requires a different benchmark axis, such as long context, to be meaningful,
- duplicates an already better Parcae-RGRP ablation.

## Already Deprioritized

| Candidate family | Reason |
| --- | --- |
| Value-only P20 control | Underperformed plain Parcae locally; useful only as a negative control. |
| Base-gate blend P20 control | Mostly neutral relative to non-blend variants. |
| Gate-only smart halting | Longer local screen underperformed gate-only fixed controls. |
| Tiny structured C1 path | Did not beat B2 depth-memory recurrent hybrid. |
| D4 P20 function-block proxy | Proxy underperformed and is not a faithful enough function-call architecture. |
| Strict fixed looped LM L1 | Paper-faithful recurrence scaffold but poor local quality. |
| Additive looped input L2 | Paper-faithful recurrence equation but unstable/poor local quality. |
| Huginn adapter recurrence L3 | Useful reference surface, but not a winner in local smoke. |
| Universal Transformer fixed/ACT | Implemented references, but local quality is far behind. |
| Ouro Stage 1 learned exit | Mechanistically useful, but local quality/latency do not justify promotion. |
| RRT strict CYCLE sharing | Useful compression baseline, but not competitive as current lead. |
| MoR expert-choice | Interesting adaptive compute primitive, but not currently competitive. |

## Still Plausible Enough To Falsify

These should get one disciplined medium screen before being ruled out.

| Candidate | Why it still deserves a falsification run | Minimum comparison |
| --- | --- | --- |
| B2-M depth-memory recurrent hybrid | Best non-Parcae-RGRP quality signal in earlier Path1 sweeps. | Compare against attention, plain Parcae, and local Parcae-P20 smart/fixed proxy. |
| D3 token-selective recurrence | Strong local loss signal, but Python-bound and possibly not hardware-realistic. | Compare quality and real throughput against Parcae-P20 smart. |
| D1+D3 routed recurrence hybrid | Best local D-family loss, but very slow. | Rule out unless quality gain is large enough to justify kernel work. |
| Paper MoDA depth-KV reference | Paper-faithful attention primitive; may be useful as component, not standalone. | Compare against B2-M and Parcae-P20 proxy. |
| MoD train-time top-C | Paper-faithful compute routing primitive but decode-unsafe. | Treat as training-time-only reference; rule out for decode path unless it transfers. |

## Orthogonal, Not Ruled Out By This Pass

| Candidate | Reason |
| --- | --- |
| Exact Flash MoDA | Kernel/hardware path, not represented by the slow PyTorch reference. |
| Native sparse attention | Long-context efficiency question, not short-context quality. |
| Stronger explicit function-call blocks | Current proxy is weak; a faithful implementation would be a new tranche. |
| SSM/RNN backbones | Different backbone family and should not be judged only by this short decoder harness. |

## Proposed Rule-Out Screen

Use a medium local run, not a giant sweep:

- seeds: `42,43,44`
- train steps: `64` or the nearest budget already used by smart-halting screens
- eval batches: `4`
- same corpus and shape as the current local proving-ground screens
- include diagnostics and throughput

Candidate lanes:

- attention control
- plain Parcae fixed
- local Parcae-P20 fixed or smart proxy
- B2-M depth-memory recurrent best prior setting
- D3 token-selective recurrence best prior setting
- D1+D3 routed recurrence best prior setting
- paper MoDA depth-KV reference
- MoD train-time top-C reference, labeled decode-unsafe

## Decision Output

For each family, record one of:

- `ruled-out`
- `deferred-orthogonal`
- `component-only`
- `survives`

The expected outcome is that most alternatives are either ruled out or reduced
to component ideas for Parcae-RGRP-control. Only a candidate that wins on a
clear quality/efficiency frontier should remain in contention as a primary
architecture.

## Rule-Out Screen Result

Artifact:

- `artifacts/path1-parcae-rgrp-candidate-ruleout-v1/summary.md`

Conditions:

- seeds: `42,43,44`
- backend/dtype: `cpu` / `fp32`
- shape: `d_model=32`, `heads=4`, `layers=4`, `ffn_multiplier=2`
- budget: `train_steps=64`, `eval_batches=4`, `seq_len=16`, `batch_size=1`
- corpus: `fineweb-stage0-local-bench-9row-v1`

Aggregate outcome:

| Lane | Mean loss | Mean train tok/s | Mean steps | Decision |
| --- | ---: | ---: | ---: | --- |
| `mod-train-topc` | 4.0222 | 8765.1 | n/a | `ruled-out-for-decode` |
| `p20-fixed5-proxy` | 4.0587 | 3233.0 | 5.00 | control |
| `d3-route25-accel` | 4.0591 | 5726.0 | 2.00 | `component-only` |
| `p20-smart5-proxy` | 4.0839 | 3903.7 | 2.22 | control |
| `b2-depthmem4-accel` | 4.0941 | 4674.4 | 2.00 | `component-only` |
| `attention-control` | 4.1108 | 9904.0 | n/a | control |
| `paper-moda-depth-kv` | 4.1219 | 7167.4 | n/a | `ruled-out` |
| `plain-parcae-fixed3` | 4.1320 | 6530.9 | 3.00 | control |
| `d1d3-route25-accel` | 4.7374 | 3599.7 | 2.00 | `ruled-out` |

Interpretation:

- `mod-train-topc` is the strongest local-loss result, but it is a
  full-sequence train-time top-C reference and is not decode safe. Keep it as a
  training-time diagnostic only unless a causal/decode-safe MoD path is designed.
- `d3-route25-accel` is not a replacement architecture, but the selected-token
  recurrence signal is credible enough to keep as a possible component idea for
  Parcae-RGRP-control or a future hardware-real routed recurrence path.
- `b2-depthmem4-accel` is close enough to remain a component idea. Depth memory
  may still be worth testing as an auxiliary lane inside the Parcae-RGRP
  scaffold, but not as a primary architecture on this evidence.
- `paper-moda-depth-kv` underperformed attention in this short local screen, so
  the current attention-side PyTorch reference is ruled out as a standalone
  candidate. Exact Flash MoDA remains an orthogonal kernel/hardware project.
- `d1d3-route25-accel` is ruled out for now. Combining block routing and
  recurrent token routing added fragility rather than synergy under matched
  local conditions.

This result strengthens `parcae-rgrp-control-looped-attention` as the promoted
scaling lane. The remaining useful work in this worktree is now component
falsification and diagnostics that can inform that lane, not broad promotion of
new primary architectures.

## MoD Oracle-Distillation Follow-Up

Artifact:

- `artifacts/path1-mod-oracle-distill-v1/summary.md`
- `artifacts/path1-mod-oracle-distill-w005-v1/summary.md`

Question:

- Can the decode-unsafe full-sequence MoD top-C router be used as an oracle
  teacher for a decode-safe causal prefix-top-k router?

Result:

| Lane | Weight | Mean loss | Oracle F1 | Oracle BCE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `causal-topk-scratch` | n/a | 3.9921 | 0.8515 | 0.6387 | control |
| `causal-topk-teacher-init` | n/a | 4.0136 | 0.8133 | 0.6195 | control |
| `causal-topk-oracle-distilled` | 0.2 | 4.0340 | 0.8451 | 0.5105 | not promoted |
| `causal-topk-oracle-distilled` | 0.05 | 4.0313 | 0.8444 | 0.5585 | not promoted |

Interpretation:

- The oracle-distillation loss works mechanically: it lowers the student's BCE
  to the full-sequence teacher mask.
- That routing agreement did not improve LM loss under this short local budget.
  The scratch causal router remained the strongest mean-loss lane.
- The decode-safe causal prefix-top-k router is therefore no longer ruled out.
  It is the strongest local small-budget non-Parcae result so far and should be
  promoted to a formal confirmation screen.
- Keep the routing-mask diagnostics and script as a reusable distillation
  harness, but do not promote binary MoD-mask distillation as a candidate
  architecture yet.

Corrected decision:

- `causal-topk-scratch`: `survives`
- `causal-topk-teacher-init`: `survives as a control`
- `causal-topk-oracle-distilled`: `not promoted as tested`
- `mod-train-topc-teacher`: still `ruled-out-for-decode`, but useful as a
  train-time teacher/reference

## Causal Top-K Confirmation

Artifact:

- `artifacts/path1-causal-topk-confirmation-v1/summary.md`

Question:

- Does the decode-safe causal prefix-top-k MoD lane still win when run through
  the standard Path 1 runner, and does it combine cleanly with Parcae/P20?

Result:

| Lane | Mean loss | Mean tok/s | Steps | Decision |
| --- | ---: | ---: | ---: | --- |
| `causal-topk-route50-layer1` | 4.0076 | 7738.7 | n/a | `promote` |
| `causal-topk-route50-layer1-parcae-fixed3` | 4.0416 | 4460.8 | 3.00 | `survives as hybrid control` |
| `causal-topk-route50-layer1-p20-fixed5` | 4.0741 | 2360.9 | 5.00 | `component-only` |
| `causal-topk-route50-layer1-p20-smart5` | 4.0773 | 3235.2 | 2.00 | `component-only` |
| `p20-fixed5-proxy` | 4.1027 | 3065.0 | 5.00 | control |
| `attention-control` | 4.1039 | 9630.8 | n/a | control |
| `p20-smart5-proxy` | 4.1076 | 3624.9 | 2.30 | control |
| `plain-parcae-fixed3` | 4.1163 | 6435.6 | 3.00 | control |

Interpretation:

- Standalone decode-safe causal top-k is now confirmed as the strongest local
  small-budget lane in this proving ground.
- The Parcae/P20 causal top-k hybrids improve over their controls, but they do
  not beat standalone causal top-k and are slower.
- This promotes causal prefix-top-k routing to a longer local/scaling
  confirmation lane. It does not yet displace the TPU Parcae-RGRP scaling lane,
  but it is no longer ruled out.

Longer local confirmation:

- `artifacts/path1-causal-topk-longer-confirmation-v1/summary.md`

Question:

- Does the standalone `route50-layer1` causal top-k lane still beat attention,
  paper train-time MoD, and local Parcae/P20 proxy controls when the local budget
  is increased from `64` to `256` train steps?

Result:

| Lane | Mean loss | Mean tok/s | Steps | Decision |
| --- | ---: | ---: | ---: | --- |
| `attention-control` | 3.1623 | 6739.5 | n/a | control |
| `p20-fixed5-proxy` | 3.1772 | 2126.6 | 5.00 | control |
| `causal-topk-route50-layer1` | 3.2103 | 5940.1 | n/a | ladder only |
| `mod-train-topc` | 3.2206 | 5797.4 | n/a | paper train-time reference |
| `p20-smart5-proxy` | 3.2313 | 2642.8 | 2.08 | control |
| `plain-parcae-fixed3` | 3.2772 | 4576.2 | 3.00 | control |

Interpretation:

- The short-run causal top-k win did not survive the first longer local
  confirmation.
- `causal-topk-route50-layer1` still beat the paper train-time MoD reference,
  but lost to attention and fixed P20 proxy.
- Do not promote `route50-layer1` directly to scale. If the family remains live,
  run a small longer-budget routing ladder over route fraction and layer
  placement only.

## MoD / Parcae-P20 Fusion Screen

Artifact:

- `artifacts/path1-mod-parcae-fusion-v1/summary.md`

Question:

- Can causal top-k be merged into the Parcae-RGRP/P20 scaffold as a control
  signal rather than as blunt hard routing inside the recurrent loop?

Result:

| Lane | Mean loss | Mean tok/s | Decision |
| --- | ---: | ---: | --- |
| `causal-topk-route50-layer1` | 4.0076 | 7703.4 | lead |
| `causal-topk-route50-layer1-parcae-fixed3` | 4.0416 | 4336.2 | hybrid control |
| `causal-topk-route50-layer1-p20-fixed5` | 4.0741 | 2169.8 | component-only |
| `causal-topk-route50-layer1-p20-smart5` | 4.0773 | 3150.8 | component-only |
| `p20-coda-topk-smart5` | 4.1211 | 3273.4 | ruled out locally |
| `p20-coda-topk-fixed5` | 4.1251 | 2791.5 | ruled out locally |
| `p20-mod-gate-bias-smart5` | 4.1297 | 3378.0 | ruled out locally |
| `p20-mod-value-scale-smart5` | 4.1304 | 3388.5 | ruled out locally |
| `p20-mod-value-scale-fixed5` | 4.1384 | 2780.8 | ruled out locally |
| `p20-mod-gate-bias-fixed5` | 4.1386 | 2862.4 | ruled out locally |

Interpretation:

- The elegant fusion attempts did not improve the Parcae/P20 scaffold in this
  local proxy.
- Gate-bias, value-scale, and coda-top-k all underperformed P20 controls.
- The best path remains standalone causal top-k. Do not expand the fusion grid
  until the standalone lane is tuned and longer-run confirmed.
