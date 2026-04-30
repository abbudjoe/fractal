# Parcae RGRP Next Rungs

Date: 2026-04-23

This note captures the next architecture ladder for the Parcae/RGRP lane after the recent CUDA long-context reads.

It is a sequencing note, not a claim note. The goal is to preserve the next decisions in the order that best matches the current evidence.

## Current Read

What the current runs suggest:

- At short context with learned global positions, the hourglass RGRP lane can approach loss parity with attention, but does not clearly beat it.
- At longer context with reduced local attention window, RGRP regains a quality edge.
- RGRP appears to prefer weaker, more local positional injection than heavy global learned position.
- The main unresolved blocker is systems cost: long-context RGRP is materially slower and heavier than attention on the current CUDA runtime path.

What that means:

- The position contract should be cleaned up before we spend time on larger architectural rewrites.
- Dynamic depth should be treated as the next compute-allocation lever, not as the first repair for the current position mismatch.
- Cache-transfer / YOCO-style ideas remain promising, but belong to a later rung where the architecture is explicitly designed around long-context state transfer rather than full-sequence training activations.

## Step 1: Split The Position Contract

Primary goal:

- attention keeps normal explicit positional features
- RGRP gets only seam-local weak position or time-step control

Why first:

- This is the smallest architectural change directly supported by the current evidence.
- It addresses the likely hidden contract bug: one global positional mechanism is currently serving two subsystems that appear to want different treatment.
- It preserves a fair attention shell while letting the recurrent controller use a lighter-order signal.

Implementation direction:

- Add an explicit attention-position contract in the Path 1 spec and runtime.
- Stop relying on the single shared residual-stream position injection as the only mechanism.
- Feed attention-local position features into the prelude/coda attention path.
- Keep the existing controller-local weak-position mechanism for RGRP, or evolve it into a cleaner time-step/local-order signal.

Success gate:

- cleaner ablations between shared global position, attention-only position, and weak controller-local position
- no regression in shell correctness or runner parity
- a fairer explanation of whether RGRP gains come from recurrence rather than from starving attention of position

Deferred option to keep in reserve:

- a small loop-to-coda seam signal at the first coda block

This is not part of the base position split. It stays deferred until the simpler split is tested cleanly.

## Step 2: Add Bounded Recurrent Depth

Primary goal:

- let the controller choose between coarse recurrent budgets such as one versus two loop passes
- keep the decision at sequence or chunk granularity so the kernel shape stays dense

Why second:

- Once the position contract is sane, the next pressure point is compute allocation.
- The current long-context tax suggests we should not force every chunk to pay the same recurrent-depth cost.
- A bounded mixture-of-depths style mechanism is more likely to survive real runtime constraints than token-level dynamic branching.

Implementation direction:

- start with fixed, bounded choices only
- prefer chunk-level or whole-sequence routing over token-level routing
- record depth-choice diagnostics from the start

Success gate:

- reduced average recurrent compute without collapsing quality
- dense execution paths that still map well to CUDA kernels
- diagnostics showing that the controller is using depth selectively rather than saturating to one choice

Initial implementation slice:

- Added explicit `parcae_loop_layer_count` plumbing to the Path 1 spec, CLI, and
  SageMaker launcher.
- Default remains the existing centered middle-band behavior. For an 8-layer
  hourglass model, that is `3 prelude + 2 recurrent + 3 coda`.
- Setting `parcae_loop_layer_count=1` keeps the same dense execution style but
  changes the 8-layer hourglass to a centered loop band:
  `3 prelude + 1 recurrent + 4 coda`.
- This is a fixed-depth ablation, not dynamic MoD yet. The first CUDA retry
  showed that this is not automatically cheaper, because fixed total depth
  reassigns the removed recurrent layer into a wide coda block.

First CUDA attempt:

- Submitted `fractal-lctx-rgrp448h7-flex-looplayers1-timing-0424` on
  SageMaker `ml.g6.2xlarge`.
- AWS left the job in `Pending` with status message `Training job waiting for
  capacity`.
- The job was stopped before `TrainingStartTime`, so it did not produce timing
  results and should not be interpreted as a model result.

Current CUDA retry:

- EC2 could not run the smoke yet: `g6.2xlarge` launch failed with
  `VcpuLimitExceeded` because the active EC2 G/VT on-demand quota in
  `us-east-1` is still `0` vCPUs.
- SageMaker retry completed on `ml.g6.2xlarge`.

| Lane | Params | Schedule | Dense block execs | Step ms | Tok/s | Read |
|---|---:|---|---:|---:|---:|---|
| 8L default loop band | 45.35M | `3 + 2x2 + 3` | 10 | 206.952 | 79,168 | current baseline timing |
| 8L loop band 1 | 46.97M | `3 + 1x2 + 4` | 9 | 212.369 | 77,149 | bad contract, extra wide coda |
| 7L loop band 1 | 44.56M | `3 + 1x2 + 3` | 8 | 200.590 | 81,679 | true reduced-virtual-depth smoke |

Decision:

- Do not promote `8L loop band 1`; it is slower and larger because the coda
  absorbs the removed recurrent layer.
- If this lever is tested for quality, use the true reduced-virtual-depth shape
  (`3 + 1x2 + 3`) or add explicit prelude/loop/coda ownership knobs before
  running a larger sweep.
- The next fair quality question is whether the speed gain from `3 + 1x2 + 3`
  survives parameter rebalancing, not whether the naive 8-layer loop-band change
  works.

8192-step quality check:

- Run: `fractal-lctx-rgrp448h7-flex-7l-looplayers1-8192-s42-0424`
- Corpus: `fineweb-cc-main-2024-10-openllama-750m`
- Tokens seen: `134,217,728`
- Shape: `d_model=448`, `head_count=7`, `seq_len=512`, `batch_size=32`,
  `local_window=128`, `flex-local`, `loop_d_model=256`, learned controller
  position, attention-only position contract.

| Lane | Params | Schedule | Final loss | Tok/s | Peak CUDA |
|---|---:|---|---:|---:|---:|
| attention-only d448/h7 flex | 48.44M | plain 8L | 4.2756 | 81,376 | 8.47 GB |
| RGRP d448/h7 flex | 45.61M | `3 + 2x2 + 3` | 4.2449 | 76,972 | 8.51 GB |
| RGRP d448/h7 7L loop band 1 | 44.82M | `3 + 1x2 + 3` | 4.2658 | 82,301 | 8.34 GB |

Interpretation:

- The reduced-virtual-depth lane preserved a small loss win over attention while
  becoming faster than both anchors and using less CUDA memory.
- It gave back about two thirds of the original RGRP quality edge over
  attention, so this is not a straight replacement for the 8L RGRP shape.
- This is the best evidence so far that the next problem is shape allocation,
  not simply a kernel rewrite. The next rung should rebalance parameters around
  the cheaper `3 + 1x2 + 3` contract before revisiting custom kernels.

## Step 3: Prototype Cache-Transfer / Prefix-Compression

Primary goal:

- use exact recent attention for the active window
- use RGRP as a compact state-transfer machine for older prefix context
- refresh with periodic exact seams when needed

Why third:

- This is the first rung that meaningfully targets the long-context memory story.
- It is architecturally bigger than the first two steps and should not be used to paper over an unclear position or compute contract.
- YOCO-style reuse is more relevant once the model is explicitly treating recurrence as a prefix-summary mechanism rather than only as a training-time middle-loop controller.

Implementation direction:

- preserve an exact recent KV window
- summarize evicted prefix context into recurrent state
- explore periodic exact seam refresh rather than all-prefix exact storage

Success gate:

- meaningful long-context memory reduction
- a credible inference-side story, not only training-side activation cost
- a clear contract for when exact attention is retained versus summarized

## Why This Order

The order is intentional:

1. fix the likely hidden contract bug first
2. then improve compute allocation
3. only then redesign long-context state ownership

That keeps us from using a large cache-transfer rewrite to compensate for a smaller position-contract problem, and it avoids turning bounded dynamic depth into another hidden workaround.

## Immediate Practical Read

For the next rung, the default stance should be:

- document the split position contract as the current primary architecture change
- keep bounded recurrent depth as the next promoted lever
- keep loop-to-coda seam signaling and YOCO-style cache transfer explicitly deferred, not forgotten

This preserves the current evidence trail and gives the next experiments a clean architectural order.

## Recurrent Transformer Extraction: Native Runtime Addendum

Date: 2026-04-29

The Recurrent Transformer paper changes how we should think about the next
systems rung. It is a nearby example of temporal recurrence inside a
transformer-like architecture: quality can improve through greater effective
depth, but naive execution becomes launch-heavy and memory-traffic-heavy.

Plan update:

1. Keep the current matched attention control first. We should not draw more
   conclusions from RGRP lanes without a clean attention run under the same
   fixed launcher contracts.
2. Promote loop-region timing to a hard gate before more shape search. The
   paper's CUDA Graph result is a warning that launch overhead can masquerade
   as architecture weakness.
3. Add a tiled future-accumulator kernel target after typed loop/control layout.
   The paper's exact tiling schedule suggests we should not process recurrent
   loop effects as isolated tiny updates when future query/control tensors are
   already available.
4. Add full-loop CUDA Graph capture/replay as a promotion candidate, but only
   after the loop-region tensor contract is stable enough that capture does not
   encode accidental PyTorch layout choices.
5. Treat custom backward/recompute as part of the native loop contract. Forward
   fusion alone is not enough if backward re-enters PyTorch reductions,
   materialization, or in-place-incompatible autograd paths.
6. Defer an output-derived persistent memory ablation. A faithful Recurrent
   Transformer-style block is interesting, but it is a new architecture branch.
   It should come after the current Parcae/RGRP champion is fairly controlled
   and timed.

Practical next-kernel order:

```text
matched attention control
-> stable loop-region timing buckets
-> typed loop/control tensor layout
-> tiled future-accumulator microkernel
-> full-loop CUDA Graph capture/replay
-> custom backward/recompute
-> output-derived persistent memory ablation
```

Success criterion:

- A kernel/runtime change must move a named timing bucket while preserving the
  matched-loss story. We should not promote a faster recurrent path if it wins
  only by silently changing the recurrence, attention window, or position
  contract.

## Frozen CUDA Promotion Baseline: Faithful 8192-Step Rerun

Date: 2026-04-25

Purpose:

- Restore the exact high-signal 8192-step RGRP contract before scaling.
- Preserve the old quality-winning architecture while keeping the newer runtime
  improvements that do not intentionally change model semantics.
- Use this result as the current promotion baseline for future seed replication,
  loop-region fusion, and shape-scaling work.

Shared run surface:

- SageMaker `ml.g6.2xlarge` / NVIDIA L4
- Corpus: `fineweb-cc-main-2024-10-openllama-750m`
- Token cache: S3 FastFile
- Steps: `8192`
- Batch/sequence: `32 x 512`
- Tokens seen: `134,217,728`
- Eval batches: `8`
- Shape shell: `d_model=448`, `head_count=7`, `total_layers=8`
- Attention: `flex-local`, `local_window=128`
- Runtime: `bf16`, compiled FFN, compiled head/loss, fused Adam,
  Triton primitive backend

Attention control:

- Job: `fractal-faithful-attn448h7-flex-8192-0425a`
- Position contract: learned shared-input position
- Params: `48.44M`
- Final eval loss: `4.1293`
- Train throughput: `104,034 tok/s`
- Peak CUDA memory: `1321 MB`

RGRP lane:

- Job: `fractal-faithful-rgrp448h7-flex-8192-0425a`
- Position contract: no shared residual position; learned controller-local
  position
- Hourglass: `3 prelude + 2 recurrent + 3 coda`
- Loop config: `loop_count=2`, `backward_steps=1`, `loop_d_model=256`,
  `loop_head_count=4`, `loop_ffn_multiplier=4`
- Prelude norm: `rmsnorm`
- Control state transform: `trainable`
- Params: `45.61M`
- Final eval loss: `4.0962`
- Train throughput: `102,995 tok/s`
- Peak CUDA memory: `1263 MB`

Comparison:

| Lane | Params | Final eval loss | Tok/s | Peak CUDA |
|---|---:|---:|---:|---:|
| attention d448/h7 learned-pos | 48.44M | 4.1293 | 104,034 | 1321 MB |
| RGRP hourglass d448/h7 loopd256 | 45.61M | 4.0962 | 102,995 | 1263 MB |

Read:

- RGRP restored the quality edge: `-0.0331` eval loss versus attention.
- RGRP used `2.84M` fewer parameters.
- RGRP used about `58 MB` less peak CUDA memory.
- RGRP paid about a `1.0%` throughput tax.
- RGRP train loss was lower than attention at every recorded checkpoint.

Recorded train-loss deltas, RGRP minus attention:

| Step | Attention train loss | RGRP train loss | Delta |
|---:|---:|---:|---:|
| 1 | 10.7654 | 10.5767 | -0.1887 |
| 1024 | 5.2702 | 5.0588 | -0.2115 |
| 2048 | 4.8629 | 4.7563 | -0.1066 |
| 3072 | 4.5344 | 4.4683 | -0.0661 |
| 4096 | 4.3981 | 4.3761 | -0.0220 |
| 5120 | 4.1499 | 4.1126 | -0.0373 |
| 6144 | 4.3382 | 4.3109 | -0.0273 |
| 7168 | 4.1588 | 4.1293 | -0.0296 |
| 8192 | 4.2851 | 4.2684 | -0.0168 |

Decision:

- Treat this as the current CUDA promotion baseline.
- Before scaling model size, replicate this exact contract with at least one
  additional seed.
- Runtime work should preserve this math first. Exact loop-region fusion,
  custom backward, projection/norm fusion, and step-capture attempts belong
  before new architecture changes.
- Shape changes such as loopd variants, deeper/narrower allocation, dynamic
  depth, or cache-transfer should be compared against this frozen baseline, not
  against the lower-signal 2k runs.

## Promoted CUDA Fast Lane: Triton Loop Glue

Date: 2026-04-26

Purpose:

- Preserve the first kernel-side Parcae/RGRP optimization that improved the
  actual quality/speed/memory surface.
- Replace the earlier "RGRP wins quality but pays a small speed tax" read with
  a stronger promoted lane: RGRP wins quality, speed, memory, and parameter
  count against the matched attention control.
- Keep the change explicit behind `parcae_loop_update_backend=triton-glue`
  rather than silently changing all Parcae defaults. The default remains useful
  for non-Triton/local paths; the fast lane must request the Triton backend
  deliberately.

Implementation:

- Added `parcae_loop_update_backend=triton-glue`.
- The backend keeps the existing compiled recurrent transformer full-block path.
- Native Triton kernels now cover the Parcae scalar loop-glue forward path:
  `state = decay * state + injection` and
  `mixed = mixed + nonlinear * (block_out - mixed)`.
- Backward is not yet a fully fused native loop-region backward. This is a
  first low-level native fusion boundary, not the final custom kernel stack.

Shared confirmation surface:

- SageMaker `ml.g6.2xlarge` / NVIDIA L4
- Corpus: `fineweb-cc-main-2024-10-openllama-750m`
- Token cache: S3 FastFile
- Steps: `8192`
- Batch/sequence: `32 x 512`
- Tokens seen: `134,217,728`
- Eval batches: `8`
- Shape shell: `d_model=448`, `head_count=7`, `total_layers=8`
- Attention: `flex-local`, `local_window=128`
- Runtime: `bf16`, compiled FFN, compiled head/loss, fused Adam,
  Triton primitive backend
- RGRP hourglass: `3 prelude + 2 recurrent + 3 coda`
- RGRP loop config: `loop_count=2`, `backward_steps=1`,
  `loop_d_model=256`, `loop_head_count=4`, `loop_ffn_multiplier=4`
- RGRP position contract: no shared residual position; learned
  controller-local position

Confirmed run:

- Job: `fractal-rgrp448h7-tritonglue-8192-0426a`
- `parcae_loop_update_backend`: `triton-glue`
- Params: `45.61M`
- Final eval loss: `4.0948`
- Train throughput: `109,672 tok/s`
- Peak CUDA memory: `1268 MB`

Comparison:

| Lane | Params | Final eval loss | Tok/s | Peak CUDA |
|---|---:|---:|---:|---:|
| attention d448/h7 learned-pos | 48.44M | 4.1293 | 104,034 | 1321 MB |
| RGRP hourglass eager-mix | 45.61M | 4.0962 | 102,995 | 1263 MB |
| RGRP hourglass eager-mix confirmation | 45.61M | 4.0956 | 104,259 | 1292 MB |
| RGRP hourglass Triton-glue | 45.61M | 4.0948 | 109,672 | 1268 MB |

Read:

- Versus attention, Triton-glue RGRP improved final eval loss by `0.0345`.
- Versus attention, Triton-glue RGRP improved throughput by about `5.4%`.
- Versus attention, Triton-glue RGRP used about `53 MB` less peak CUDA memory.
- Versus attention, Triton-glue RGRP used about `2.84M` fewer parameters.
- Versus eager RGRP, Triton-glue preserved quality while removing the speed tax.

Decision:

- Promote `parcae_loop_update_backend=triton-glue` as the current CUDA RGRP
  fast lane.
- Keep `eager`, `compiled`, `manual-autograd`, and `lean-eager` as ablation
  knobs, not promoted defaults.
- Future scale or shape changes should compare against this Triton-glue run,
  not the older eager baseline.
- The next systems rung should target a larger native loop-region boundary:
  fused Parcae/RGRP backward and eventually a custom loop-region kernel. The
  current Triton-glue result proves that some loop-region overhead is
  recoverable, but it does not exhaust the native-fusion opportunity.
