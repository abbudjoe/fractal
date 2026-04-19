# MaxText RGRP vs Attention Baseline Scorecard

## Matched Contract

Both runs used:

- Date: 2026-04-19
- Hardware: one `v5litepod-1` spot TPU VM in `us-west4-a`
- Runtime: `v2-tpuv5-litepod`
- Dataset: `roneneldan/TinyStories`
- Tokenizer: Hugging Face `gpt2`
- Sequence length: `512`
- Per-device batch size: `4`
- Training steps: `200`
- Eval interval: `100`
- Eval steps: `5`
- Dtype: `bfloat16`
- Shape: `base_emb_dim=256`, `base_mlp_dim=1024`,
  `base_num_decoder_layers=4`, `base_num_query_heads=4`,
  `base_num_kv_heads=4`, `head_dim=64`
- Checkpointing disabled

Difference:

- Baseline: unchanged MaxText `decoder_block=default`
- RGRP: patched MaxText FFN seam using the rotary gated recurrent state update
  primitive with `jax.lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence projection, and trig precompute

## Results

| Lane | Params | Memory After Init | Eval Loss @99 | Final Eval Loss @199 | Final Eval PPL | Final Train Loss | Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention baseline | `0.030B` | `0.35 GB` | `4.030` | `3.758` | `42.874` | `3.817` | `375,987` |
| RGRP FFN seam | `0.028B` | `0.33 GB` | `4.400` | `4.066` | `58.317` | `4.102` | `105,763` |

Delta, RGRP relative to baseline:

| Metric | Delta |
|---|---:|
| Parameters | `-0.002B` |
| Memory after init | `-0.02 GB` |
| Final eval loss | `+0.308` worse |
| Final eval perplexity | `+15.443` worse |
| Final train loss | `+0.285` worse |
| Median throughput | `0.28x` baseline, or baseline is `3.55x` faster |

## Interpretation

This rung does not support a claim that the current RGRP MaxText `lax.scan`
FFN-seam variant beats the default transformer baseline. Attention wins quality
and speed under the matched short-run contract.

What the result does prove:

- The patched RGRP MaxText integration is real and trainable on TPU.
- The unchanged MaxText baseline is now available as the required control.
- The current speed bottleneck is not just a toy-shell artifact; it persists in
  MaxText training.

Most likely explanation:

- The default MaxText MLP/attention path lowers extremely well through XLA at
  this small shape.
- The RGRP seam saves a small amount of parameter/memory budget but pays a large
  sequential `lax.scan` tax.
- The current RGRP FFN-seam contract is not extracting enough quality benefit
  over `200` steps to justify that sequential tax.

Recommended next step:

- Do not repeat this exact run.
- If continuing TPU work, use the baseline as the control and change one
  architectural or contract axis at a time: residual/ramp scaling, hybrid
  FFN+RGRP rather than full FFN replacement, or a larger/longer rung where
  recurrent state might plausibly matter.
- If the goal is speed competitiveness, do not invest further in this plain
  `lax.scan` lowering. The current evidence says we need a different recurrent
  contract or a custom lowering that changes the sequential work profile.
