# MoD / Parcae-P20 Fusion Summary

## Question

Can the MoD-inspired decode-safe causal top-k signal be merged elegantly into
the Parcae-RGRP/P20 scaffold rather than treated as a standalone routed block?

This local screen uses the Path 1 Parcae-P20 scaffold as the fast proxy for the
Parcae-RGRP control lane. It is not a MaxText TPU RGRP result.

## Implemented Fusion Variants

Three fusion styles were tested:

- **Hard top-k inside loop**: existing causal top-k routed middle block inside
  Parcae/P20 recurrence.
- **Router-conditioned gate bias**: causal router logits bias the Parcae-P20
  control gate before sigmoid.
- **Router-conditioned value scale**: causal router salience scales the
  Parcae-P20 control value update.
- **Top-k coda**: dense Parcae-P20 loop followed by a causal top-k coda block.

Artifact:

- `artifacts/path1-mod-parcae-fusion-v1/summary.md`

## Conditions

- seeds: `42,43,44,45,46`
- backend/dtype: `cpu` / `fp32`
- shape: `d_model=32`, `heads=4`, `layers=4`, `ffn_multiplier=2`
- budget: `train_steps=64`, `eval_batches=4`, `seq_len=16`, `batch_size=1`
- warmup: `warmup_eval_batches=1`, `warmup_train_steps=1`
- corpus: `fineweb-stage0-local-bench-9row-v1`

## Results

| Lane | Mean loss | Loss std | Train tok/s | Steps | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `causal-topk-route50-layer1` | 4.0076 | 0.0561 | 7703.4 | n/a | lead |
| `causal-topk-route50-layer1-parcae-fixed3` | 4.0416 | 0.0857 | 4336.2 | 3.00 | hybrid control |
| `causal-topk-route50-layer1-p20-fixed5` | 4.0741 | 0.0701 | 2169.8 | 5.00 | component-only |
| `causal-topk-route50-layer1-p20-smart5` | 4.0773 | 0.0385 | 3150.8 | 2.00 | component-only |
| `p20-fixed5-proxy` | 4.1027 | 0.0855 | 2936.1 | 5.00 | control |
| `attention-control` | 4.1039 | 0.0843 | 9490.9 | n/a | control |
| `p20-smart5-proxy` | 4.1076 | 0.0527 | 3540.0 | 2.30 | control |
| `plain-parcae-fixed3` | 4.1163 | 0.0207 | 6248.4 | 3.00 | control |
| `p20-coda-topk-smart5` | 4.1211 | 0.0731 | 3273.4 | 2.24 | ruled out locally |
| `p20-coda-topk-fixed5` | 4.1251 | 0.0571 | 2791.5 | 5.00 | ruled out locally |
| `p20-mod-gate-bias-smart5` | 4.1297 | 0.0756 | 3378.0 | 2.32 | ruled out locally |
| `p20-mod-value-scale-smart5` | 4.1304 | 0.0767 | 3388.5 | 2.31 | ruled out locally |
| `p20-mod-value-scale-fixed5` | 4.1384 | 0.0984 | 2780.8 | 5.00 | ruled out locally |
| `p20-mod-gate-bias-fixed5` | 4.1386 | 0.0975 | 2862.4 | 5.00 | ruled out locally |

## Interpretation

The standalone decode-safe causal top-k routed block remains the clear local
leader. It is both better and faster than the Parcae/P20 fusion variants.

The naive hard-routing hybrids still have the strongest fusion signal:

- Plain Parcae + hard loop top-k improves over plain Parcae but loses to
  standalone top-k.
- P20 + hard loop top-k improves over P20 controls but is much slower and still
  loses to standalone top-k.

The more elegant control-path fusions did not help in this implementation:

- Gate-bias and value-scale conditioning both underperformed the P20 controls.
- Coda top-k after dense P20 recurrence also underperformed the P20 controls.

This suggests that, at least in the current local Path 1 proxy, causal top-k is
not naturally improved by attaching it to the Parcae/P20 control path. The
systems may still be conceptually compatible, but the evidence says the current
best use of MoD causal routing is standalone.

## Decision

- Promote standalone `causal-topk-route50-layer1`.
- Keep `causal-topk-route50-layer1-parcae-fixed3` only as a hybrid control.
- Treat P20 hard-loop top-k as component-only.
- Rule out the first gate-bias, value-scale, and coda-top-k P20 fusions locally.

## Next Step

Do not keep widening the Parcae/P20 fusion grid yet. Tune the standalone causal
top-k lane first:

- route fraction: `0.25`, `0.5`, `0.75`
- layer placement: early, middle, late
- two-layer routing
- longer local run

Only revisit Parcae-RGRP fusion if standalone causal top-k survives the longer
screen or if TPU-scale Parcae-RGRP diagnostics suggest a specific control
failure that causal routing can address.
