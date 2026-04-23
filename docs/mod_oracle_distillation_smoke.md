# MoD Oracle Distillation Smoke

## Question

Can the decode-unsafe full-sequence MoD top-C routing lane be used as an oracle
teacher for a decode-safe causal prefix-top-k router?

## Implementation

Added a narrow smoke harness:

- `scripts/path1_mod_oracle_distill.py`

The script uses the existing Path 1 model/data stack and runs:

- `mod-train-topc-teacher`: trains the full-sequence MoD top-C teacher.
- `causal-topk-scratch`: trains the decode-safe causal prefix-top-k router from
  scratch.
- `causal-topk-teacher-init`: initializes the causal router from the trained
  teacher weights, then trains with LM loss only.
- `causal-topk-oracle-distilled`: initializes from the trained teacher and adds
  BCE loss from causal student router scores to teacher full-sequence selected
  masks.

Routed transformer blocks now expose last-step routing tensors:

- router scores
- selected token mask

This keeps the experiment outside the normal trainer except for a small
diagnostic surface needed by any future router distillation work.

## Conditions

- seeds: `42,43,44`
- backend/dtype: `cpu` / `fp32`
- shape: `d_model=32`, `heads=4`, `layers=4`, `ffn_multiplier=2`
- budget: `train_steps=64`, `eval_batches=4`, `seq_len=16`, `batch_size=1`
- corpus: `fineweb-stage0-local-bench-9row-v1`

## Results

Primary artifact:

- `artifacts/path1-mod-oracle-distill-v1/summary.md`

Stabilization pass with lower distillation weight:

- `artifacts/path1-mod-oracle-distill-w005-v1/summary.md`

### Distillation Weight 0.2

| Lane | Mean loss | Mean tok/s | Oracle F1 | Oracle BCE |
| --- | ---: | ---: | ---: | ---: |
| `causal-topk-scratch` | 3.9921 | 7267.2 | 0.8515 | 0.6387 |
| `causal-topk-teacher-init` | 4.0136 | 7531.9 | 0.8133 | 0.6195 |
| `mod-train-topc-teacher` | 4.0222 | 7971.4 | n/a | n/a |
| `causal-topk-oracle-distilled` | 4.0340 | 5859.6 | 0.8451 | 0.5105 |

### Distillation Weight 0.05

| Lane | Mean loss | Mean tok/s | Oracle F1 | Oracle BCE |
| --- | ---: | ---: | ---: | ---: |
| `causal-topk-scratch` | 3.9921 | 7394.2 | 0.8515 | 0.6387 |
| `causal-topk-teacher-init` | 4.0136 | 7405.8 | 0.8133 | 0.6195 |
| `mod-train-topc-teacher` | 4.0222 | 7479.4 | n/a | n/a |
| `causal-topk-oracle-distilled` | 4.0313 | 5721.6 | 0.8444 | 0.5585 |

## Interpretation

The oracle-distillation objective is active: it substantially lowers the BCE
between student router scores and full-sequence teacher masks.

That did not translate into better LM loss in this short local proving-ground
screen. The scratch causal router was best on mean loss, and both distilled
settings were worse than the teacher-initialized causal control.

This creates two separate conclusions:

- The **decode-safe causal prefix-top-k router** is a real surviving candidate.
  It beat the prior local rule-out leaders under the same small Path 1
  conditions and should be promoted to a focused confirmation screen.
- The **binary full-sequence oracle-mask imitation objective** is not promoted as
  tested. It improves oracle BCE but does not improve LM loss over the causal
  scratch control.

This result does not invalidate oracle distillation generally; it says this
first mask-imitation version is probably optimizing the wrong target or
over-constraining the causal router.

## Decision

Status:

- `causal-topk-scratch`: `survives / promote to confirmation`
- `causal-topk-teacher-init`: `survives as control`
- `causal-topk-oracle-distilled`: `not promoted as tested`

Keep:

- routed-block mask/scores diagnostic surface
- oracle-overlap metrics
- the script as a reusable test harness
- causal prefix-top-k MoD routing as a live non-Parcae candidate

Do not promote yet:

- full-sequence MoD mask BCE as a primary causal router training objective

Immediate next tests:

- rerun causal prefix-top-k routing through the formal matched manifest path
- run a longer 3-seed or 5-seed confirmation against Parcae-P20/RGRP controls
- test causal top-k as a component inside the Parcae-RGRP scaffold

More plausible oracle-distillation tests, if this subfamily is revisited:

- distill soft teacher router ranks or margins, not only binary selected masks
- distill teacher hidden deltas on selected tokens
- train the teacher longer before distillation
- use Parcae-RGRP-control as the causal student rather than plain causal top-k
- add a cheap prefix predictor/SSM only after a stronger oracle target is found

## Follow-Up Confirmation

The standard Path 1 runner confirmation is stored at:

- `artifacts/path1-causal-topk-confirmation-v1/summary.md`

That run confirmed the decode-safe causal top-k signal:

| Lane | Mean loss | Decision |
| --- | ---: | --- |
| `causal-topk-route50-layer1` | 4.0076 | `promote` |
| `attention-control` | 4.1039 | control |
| `p20-fixed5-proxy` | 4.1027 | control |
| `plain-parcae-fixed3` | 4.1163 | control |

The Parcae/P20 causal-top-k hybrids improved over their controls but did not
beat standalone causal top-k. The current lead form of this family is therefore
the simple decode-safe causal prefix-top-k routed block, not the hybrid.
