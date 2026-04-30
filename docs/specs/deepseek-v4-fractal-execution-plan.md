# DeepSeek-V4 Extraction Plan For Fractal/RGRP

Date: 2026-04-29

## Purpose

This note translates the useful architectural and systems ideas from
DeepSeek-V4 into a staged Fractal/RGRP execution plan.

The goal is not to copy DeepSeek-V4. Their full system depends on trillion-scale
MoE training, custom compressed-attention kernels, FP4/FP8 infrastructure,
large-scale Muon, and production KV-cache management. The useful question for
this repo is narrower:

```text
Which DeepSeek-V4 ideas can help RGRP become a useful long-context memory and
control mechanism, and when does each idea become worth implementing?
```

## Core Read

DeepSeek-V4 and the Recurrent Transformer paper point in the same direction:
KV/cache structure is part of the model architecture, not a mere serving detail.

The Recurrent Transformer keeps layerwise persistent KV and redesigns execution
around exact tiled recurrence. DeepSeek-V4 takes a different route: keep recent
tokens exact, compress older KV into learned memory entries, sparsely retrieve
the relevant compressed entries, and own the kernels deeply enough that this
does not collapse into hundreds of fine-grained framework calls.

For Fractal/RGRP, the strongest implication is:

```text
RGRP should not be treated only as an attention replacement.
It can be the controller/compressor/router for long-context memory.
```

## Applicability By Scale

| Idea | Current scale relevance | First sensible rung | Why |
|---|---|---|---|
| Exact local attention plus compressed old memory | High | Current 50M-100M long-context work | We already see RGRP's relative value improve when local attention is constrained. |
| HCA-lite dense compressed memory | High | Current 50M-100M | Simpler than sparse retrieval and directly tests whether compressed old context helps. |
| CSA-lite top-k compressed memory | Medium | After HCA-lite survives | Top-k indexer adds complexity; only worth it after compressed memory has signal. |
| RGRP as compression/controller path | High | Current 50M-100M | This is the cleanest role for RGRP: write/update/select state, not replace every block. |
| Output-derived memory writes | Medium | After compressed-memory scaffold works | Borrowed from Recurrent Transformer; useful, but changes memory provenance and needs controls. |
| mHC-style residual highway | Medium-high | 100M+ or unstable loop-depth rungs | Useful for recurrent stability, but adds residual-state machinery that should be tested after current controls. |
| Attention sink | Medium | Long-context compressed attention | Cheap and stabilizing when attention can choose to ignore poor compressed entries. |
| Query/KV RMSNorm before compressed attention | High when compressed attention exists | First compressed-attention ablation | Stability guardrail, not optional once logits involve compressed memory. |
| Partial RoPE on compressed memory | Medium | Long-context compressed-memory ablation | Relevant when compressed entries need relative-position behavior. |
| Grouped output projection | Low-medium now | 100M+ if attention projection dominates | Mostly a scaling/runtime optimization for large head dimensions. |
| FP8 KV and FP4 indexer path | Low now | 250M+ or true long-context serving | Premature before the memory architecture proves useful. |
| On-disk KV cache | Not relevant now | Inference/product serving, not training proof ladder | Useful for repeated-prefix serving, not current training experiments. |
| Full DeepSeek CSA/HCA cache manager | Not relevant now | Only after a successful long-context compressed-memory lane | Too much machinery before signal. |
| DeepSeek-style Muon assignment | Medium | After architecture/control is stable | Our earlier Muon tests were not paper-faithful; revisit as an optimizer contract, not a quick swap. |
| TileLang-like host codegen/deterministic kernel stack | Strategic | Native kernel track | Reinforces our anti-hidden-framework-contract direction, but not a single model ablation. |

## Phase 0: Keep Current Proof Ladder Honest

Status: must remain active before new claims.

Actions:

1. Keep matched attention controls for every promoted RGRP run.
2. Keep `attention-only` inert with respect to RGRP/Triton primitive backend flags.
3. Keep timing buckets stable enough that kernel changes are attributable.
4. Do not promote compressed memory until the current attention-control contract is clean.

Success gate:

```text
attention-only runs are uncontaminated
RGRP runs report comparable timing buckets
loss/speed/memory comparisons are matched by shape, data, seed, and runtime knobs
```

## Phase 1: HCA-Lite Compressed Memory

Current-scale relevance: high.

This is the first DeepSeek-inspired model experiment worth doing at our current
scale. It avoids top-k sparse retrieval and tests the simpler proposition:

```text
Can RGRP-controlled compressed old-context memory improve long-context modeling
while preserving exact recent attention?
```

Architecture sketch:

```text
tokens
-> wide prelude attention
-> exact local attention window
-> token/block compressor every m' tokens
-> compressed old-memory entries
-> Parcae/RGRP loop band
-> coda reads exact recent window + dense compressed old memory
-> LM head
```

Initial knobs:

| Knob | First values |
|---|---|
| Local exact window | 128 |
| HCA compression rate `m'` | 64, then 128 |
| Compressed memory dimension | match loop/control width first |
| Compressor source | normalized hidden state |
| RGRP role | compression gate/value/controller |
| Retrieval | dense over compressed entries |
| Stabilizers | RMSNorm on query and compressed entries; attention sink |

Controls:

1. Attention-only local window.
2. Current RGRP hourglass champion.
3. HCA-lite without RGRP controller, using learned weighted compression.
4. HCA-lite with RGRP controller.

Success gate:

```text
HCA-lite + RGRP improves long-context loss or memory at tolerable speed tax
and beats HCA-lite without RGRP on at least one matched run.
```

Stop condition:

```text
dense compressed memory is slower and no better than exact local attention
or RGRP does not improve over learned compression control.
```

## Phase 2: CSA-Lite Sparse Compressed Memory

Current-scale relevance: medium. Implement only after Phase 1 earns it.

CSA-lite adds a learned indexer and top-k selection over compressed memory.
This is closer to the DeepSeek-V4 mechanism, but it introduces a much larger
kernel and correctness surface.

Architecture delta from HCA-lite:

```text
compressed memory entries
-> compressed indexer keys
-> query-side lightweight indexer
-> top-k compressed entries
-> core attention over exact recent window + selected compressed entries
```

Initial knobs:

| Knob | First values |
|---|---|
| CSA compression rate `m` | 4 or 8 |
| Top-k | small fixed values first, e.g. 32/64 at our scale |
| Indexer precision | BF16/FP32 first, no FP4 yet |
| Query compression | low-rank query projection |
| Attention sink | enabled |
| Query/KV RMSNorm | enabled |

Controls:

1. HCA-lite winner.
2. CSA-lite learned indexer without RGRP.
3. CSA-lite with RGRP controlling compression or indexer weights.

Success gate:

```text
CSA-lite recovers quality over HCA-lite when old context grows,
without introducing a speed/memory cliff.
```

Deferred until later:

- FP4 indexer QK.
- Production top-k fused selector.
- Full cache manager.

## Phase 3: Output-Derived Memory Writes

Current-scale relevance: medium.

This is the bridge back to the Recurrent Transformer paper. The important idea
is that memory written from post-block output may carry processed information
that raw input-derived compression misses.

Two ablations:

```text
input-derived compressed memory:
  memory_t = compressor(norm(hidden_before_block))

output-derived compressed memory:
  memory_t = compressor(norm(hidden_after_loop_or_block))
```

Why not first:

- It changes what the cache means.
- It can create circular dependencies if the current token reads memory that
  depends on itself.
- It belongs after the HCA/CSA scaffold has a clear causality contract.

Success gate:

```text
output-derived memory improves loss at same compression rate and does not
increase instability or leak future-token information.
```

## Phase 4: mHC-Style Residual Highway Around The Loop

Current-scale relevance: medium-high, but not first.

DeepSeek-V4 uses manifold-constrained hyper-connections to expand the residual
stream into multiple lanes while constraining residual mixing to a stable
doubly-stochastic matrix. For us, this is most relevant around the Parcae/RGRP
loop band, where repeated latent computation needs a cleaner stability contract.

Minimal Fractal version:

```text
residual lanes X: [n_hc, d_model], n_hc = 4
layer input = A X
loop output = F(layer input)
next lanes = B X + C F(layer input)
```

Start simple:

| Component | First implementation |
|---|---|
| `A` | static sigmoid-constrained weights |
| `B` | static doubly-stochastic matrix |
| `C` | static sigmoid-constrained output weights |
| Dynamic generation | disabled or tiny alpha initialized near zero |
| Projection | around loop band only |

Why it matters:

- It gives RGRP/Parcae a stable multi-lane residual highway.
- It may reduce our reliance on ad hoc residual ramps and gates.
- It creates a principled place for recurrent state/control to route information.

Success gate:

```text
mHC around the loop improves stability or loss without more than a small
runtime/memory tax.
```

Do not implement full DeepSeek dynamic mHC until:

- 100M+ shapes show residual instability or loop-depth sensitivity.
- Static mHC shows a positive signal.

## Phase 5: DeepSeek-Style Optimizer Contract

Current-scale relevance: medium, but depends on stable architecture.

Our earlier Muon attempts should not be treated as a final verdict. DeepSeek's
optimizer contract is selective:

```text
AdamW:
  embeddings
  prediction head
  RMSNorm weights
  mHC static biases/gates/scalars

Muon:
  matrix-heavy projections
  attention matrices
  FFN matrices
  RGRP/control matrices, if stable
```

Implementation notes:

- Use Nesterov Muon with hybrid Newton-Schulz.
- Use update RMS rescaling, not a naive LR swap.
- Keep AdamW for scalar and normalization parameters.
- Treat native/fused Muon as a later kernel track, not a prerequisite for HCA-lite.

Success gate:

```text
selective Muon improves convergence or stability under matched wall-clock,
without degrading the attention control.
```

## Phase 6: Full DeepSeek-Style Runtime

Current-scale relevance: not yet.

This is the scale architecture, not the first implementation plan.

Full version includes:

1. Interleaved CSA and HCA layers.
2. Exact sliding-window branch in every compressed attention layer.
3. Learned compressed KV entries with overlap for CSA.
4. Lightning indexer with top-k sparse selection.
5. Shared KV MQA and grouped output projection.
6. Query/KV RMSNorm and partial RoPE on compressed entries.
7. Attention sink.
8. FP8 KV storage with BF16 RoPE dims.
9. FP4 indexer QK path.
10. Heterogeneous KV cache manager for SWA, CSA, HCA, and uncompressed tails.
11. On-disk KV cache for repeated-prefix inference.
12. TileLang/CUTLASS-class fused kernels and host codegen.
13. Deterministic/batch-invariant kernel libraries.
14. Selective Muon plus QAT where relevant.

Do not build this until:

```text
HCA-lite or CSA-lite wins a matched long-context proof rung
and the native kernel surface is stable enough to support compressed memory
without falling back into framework launch overhead.
```

## Execution Order

Recommended order from here:

1. Finish current matched attention-control cleanup.
2. Keep native loop-region timing and kernel-contract work active.
3. Implement HCA-lite as the first compressed-memory experiment.
4. Test HCA-lite with and without RGRP control.
5. If positive, add CSA-lite top-k compressed retrieval.
6. If compressed memory works, test output-derived writes.
7. Add static mHC around the loop band if loop depth/stability remains the bottleneck.
8. Revisit selective Muon after architecture stabilizes.
9. Only then consider full DeepSeek-style precision/cache/runtime machinery.

## Bottom Line

At our current scale, the useful DeepSeek-V4 extraction is not FP4, trillion-scale
MoE, or production KV storage.

The useful extraction is:

```text
exact recent attention
+ learned compressed old-context memory
+ RGRP as controller/compressor/router
+ stable multi-lane residual highways if loop depth gets unstable
+ native kernels that own the compressed-memory contract
```

That is a credible next research lane because it directly addresses the problem
we keep circling: RGRP becomes more interesting when attention is no longer
allowed to brute-force all context, but it needs a cleaner role than "replace
attention everywhere."
