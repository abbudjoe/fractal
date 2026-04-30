# Custom Windowed Attention Kernel Contract

Timestamp: 2026-04-24

## Motivation

The CUDA SDPA investigation showed that our current "local attention" path is
not a true windowed CUDA kernel. It is dense SDPA with a local additive mask.
That preserves math, but it changes the CUDA backend:

- full causal attention can use FlashAttention;
- dense local masks fall back to efficient attention for friendly head dims;
- unfriendly head dims can fall all the way to math attention;
- PyTorch FlexAttention can express a local block mask, but this runtime rejects
  non-power-of-two head dimensions such as `48`.

The architecture should not be forced to contort around incidental PyTorch
kernel constraints. If the quality-optimal shape wants `head_dim=48`, the kernel
surface should support it directly.

## Required Semantics

Inputs:

- `q`, `k`, `v` shaped `[batch, heads, seq, head_dim]`
- dtype: bf16 first; fp16 optional; fp32 reference only
- causal local window size `W`

For each query position `t`, visible key positions are:

```text
max(0, t - W + 1) <= j <= t
```

Output:

```text
softmax(q_t @ k_j / sqrt(head_dim)) @ v_j
```

over only the visible local band.

The kernel must match the current dense masked SDPA semantics within normal bf16
training tolerance.

## Shape Targets

The first target is the current Path 1 long-context surface:

| Shape | Value |
|---|---:|
| batch | 32 |
| seq_len | 512 first, then 1024/2048 |
| local_window | 128 first |
| outer d_model/head_count | `480/10`, `448/7`, `512/8` |
| head_dim targets | `48`, `64` |
| loop d_model/head_count | `256/8`, `256/4` |
| loop head_dim targets | `32`, `64` |

Support for `head_dim=48` is the main reason to own this kernel.

## Performance Targets

Baseline numbers on NVIDIA L4:

| Path | Shape | Attention-call mean |
|---|---|---:|
| dense local SDPA | d480/h10, head_dim 48 | 0.660 ms |
| full causal flash | d480/h10, head_dim 48 | 0.303 ms |
| Flex local | d448/h7, head_dim 64 | 0.250 ms |

Initial success threshold:

- beat dense local SDPA for `head_dim=48`;
- approach full-causal flash per-call timing enough to matter end-to-end;
- avoid materially worse peak memory than dense local SDPA;
- preserve autograd correctness for training.

## Implementation Ladder

1. Reference parity harness.

Implement a small isolated test that compares:

- dense masked SDPA local attention;
- candidate custom windowed attention;
- forward outputs;
- gradients for q/k/v.

Use small CPU/GPU-testable shapes first, then L4/H100 shape tests.

2. Triton forward kernel.

Compute one block of query positions and a bounded key window. Do not materialize
the full `[seq, seq]` mask. Use online softmax accumulation across the local
window.

3. Triton backward kernel.

Add training support. The forward-only kernel is useful for inference, but this
research lane needs backward.

4. Model integration.

Add a third explicit attention kernel profile:

```text
attention_kernel = sdpa | flex-local | flash-local | triton-window
```

Do not hide this behind environment variables.
`flex-local` is intentionally restricted to power-of-two head dimensions because
that is the current PyTorch FlexAttention contract in our CUDA runtime. A custom
`triton-window` lane should remove that architecture leak and support the
quality-favored `head_dim=48` geometry directly.
`flash-local` is an optional dependency probe, not the final ownership boundary:
if the package is available and fast for `head_dim=48`, it becomes the teacher
kernel for parity and timing; if not, `triton-window` remains the owned path.

5. Runtime comparison.

Run model-level timing on:

- d480/h10 `sdpa` local;
- d480/h10 `triton-window`;
- d448/h7 `flex-local`;
- d480 full causal flash diagnostic.

6. Quality comparison.

Only after timing/parity holds, run 8192-step quality at the same data/context
surface.

## Non-Goals

- Do not rewrite the whole transformer block first.
- Do not fuse QKV projection into the windowed kernel until the windowed
  attention primitive itself is correct and faster.
- Do not optimize for one shape by making shape support implicit. Shape
  constraints must be explicit and tested.

## Promotion Gate

Promote `triton-window` only if it passes:

- forward parity against dense local SDPA;
- q/k/v gradient parity against dense local SDPA;
- model-level timing win on `head_dim=48`;
- no unexplained memory blow-up;
- short quality run stays within expected numerical noise.
