# Parcae/RGRP Execution Lanes

Date: 2026-04-30

## Purpose

This note keeps the next work split into three attributable lanes:

1. main proof scaling
2. DeepSeek-inspired ablations
3. native kernel work

The rule is simple: do not mix a new model concept into the main proof lane
until it has earned promotion in a bounded ablation.

## Current Empirical Spine

Corrected 100M-ish comparison:

| Lane | Job | Params | Final loss | Tok/s | Peak CUDA |
|---|---|---:|---:|---:|---:|
| attention fixed control | `attn704h11-flex8192-s45-lr3e4-layernorm-control-0429a` | 117.26M | 3.9747 | 38,024 | 11.10 GB |
| RGRP quality | `rgrp768-b32322-loop256-stable8192-s45-0430a` | 111.09M | 3.9946 | 38,352 | 10.77 GB |
| RGRP efficient | `rgrp704-b42123-loop320-stable8192-s45-0430a` | 100.05M | 4.0341 | 39,349 | 10.49 GB |
| official Mamba 130M | `official-mamba130-wheelhouse8192-s45-0430a` | 115.10M | 4.0787 | 25,350 | 11.77 GB |

Interpretation:

- Attention still leads quality at this rung.
- RGRP quality is close while being slightly faster, smaller, and lower-memory.
- RGRP efficient is the Pareto lane.
- RGRP beats official Mamba in this harness.

## Lane A: Main Proof Scaling

Goal:

```text
Scale clean RGRP vs matched attention to 250M-300M without adding new concepts.
```

Starting candidates:

| Candidate | Role | Starting shape idea | Why |
|---|---|---|---|
| attention control | quality baseline | GPT-like attention-only, matched token/data/runtime contract | Must exist for every promoted scale rung. |
| RGRP quality | proof challenger | wider outer shell, loop width held below full model width | Best 100M RGRP quality lane. |
| RGRP efficient | Pareto challenger | `4,2,1,2,3`-style outer-shell allocation | Smaller/faster 100M lane. |

### Active 300M Scout

Submitted 2026-04-30:

```text
job: rgrp300q-gpt2medgeo-fit1024-s45-0430a
instance: ml.g6e.24xlarge
max_runtime_seconds: 7200
token cache: fineweb-cc-main-2024-10-openllama-750m
steps: 1024
seq_len: 512
batch_size: 32
seed/data_seed: 45/45
dtype: bf16
attention kernel: flex-local
position contract: learned positions, attention-only
final norm: layernorm
optimizer/lr: adam, 3e-4
MTP: off
```

Lanes:

| Lane | Shape | Expected role |
|---|---|---|
| GPT-2-medium-geometry attention | `24L d1024 h16` | Stress baseline, close to GPT-2 Medium geometry but larger than the 300M target in this harness. |
| RGRP 5-band quality | `24L d1024 h16`, bands `6,3,6,3,6`, loop `384x2`, loop heads `6` | First scaled 5-band quality challenger. |

This is a fit/speed scout, not the final 300M proof control. If it fits and
the RGRP curve is live, follow with a trimmed GPT-2-medium-like attention
control around `24L d960 h15` and distributed 7-band RGRP candidates.

First non-mutating prep:

1. Derive two 250M-300M candidate shapes before launch.
2. Estimate parameter count and activation memory for each.
3. Run fit/speed scouts only after explicit launch authorization.
4. Promote to longer runs only with matched attention controls.

Do not add MTP, compressed memory, MoD, or cache-transfer ideas to this lane
until those ideas have their own positive ablation.

## Lane B: DeepSeek-Inspired Ablations

Goal:

```text
Test small, attributable DeepSeek-inspired additions at 100M before promotion.
```

### Deferred MTP Scout

MTP is now deferred until the model is much larger, likely `>1B`, to avoid
mixing a new training objective into the current 300M proof lane.

Four 1024-step jobs were submitted on 2026-04-30 and failed immediately because
the SageMaker token-cache entrypoint passed `--mtp-*` args into the Parcae
promotion runner before that runner accepted them. The launcher contract was
fixed in commit `97af085`, but the jobs should not be interpreted as model
results and should not be relaunched at this scale.

| Job | Shape | Lanes | MTP |
|---|---|---|---|
| `mtp-d704-off1024-s45-0430a` | d704/h11, bands `4,2,1,2,3`, loop320x2 | attention + RGRP efficient | off |
| `mtp-d704-on1024-h3w005-s45-0430a` | d704/h11, bands `4,2,1,2,3`, loop320x2 | attention + RGRP efficient | `weight=0.05`, `horizon=3` |
| `mtp-d768-off1024-s45-0430a` | d768/h12, bands `3,2,3,2,2`, loop256x2 | RGRP quality | off |
| `mtp-d768-on1024-h3w005-s45-0430a` | d768/h12, bands `3,2,3,2,2`, loop256x2 | RGRP quality | `weight=0.05`, `horizon=3` |

Promotion read:

```text
MTP promotes only if final next-token eval loss improves or the early loss
slope improves without material speed/memory regression.
```

Important: MTP training loss includes auxiliary future-token CE, but eval/final
loss remains next-token CE only.

Next DeepSeek-inspired candidates after MTP:

1. HCA-lite dense compressed memory.
2. RGRP-controlled compression gate/value.
3. CSA-lite sparse compressed memory only after HCA-lite earns it.

## Lane C: Native Kernel Lane

Goal:

```text
Reduce PyTorch-owned overhead around the recurrent loop without changing model
semantics.
```

Dependency order:

1. Lock timing names so every run tells us where time moved.
2. Type and pack loop/control tensors so native kernels consume a stable layout.
3. Replace the first backward boundary with explicit native reductions.
4. Fuse recurrent block pieces that depend on the stable layout.
5. Move into full loop-region forward/backward/recompute.
6. Revisit local/sparse attention and optimizer only after the loop contract is stable.

Current doctrine:

- PyTorch fallback is acceptable only as a named control, not as an invisible
  dependency in the promoted native path.
- Every native kernel candidate must pass isolated parity before cloud training.
- A kernel candidate promotes only if it preserves loss under matched controls.

First work item:

```text
Add a concrete loop/control packed-layout spec and parity fixture before the
next native backward or fusion attempt.
```
