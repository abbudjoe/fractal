# EML-Inspired FFN Codex Work Transcript

This document records the working conversation and decisions that began with the
request to evaluate an EML-inspired transformer variant from
<https://arxiv.org/html/2603.21852v2>.

Note: the
implementation and discussion consistently describe the paper-inspired module as
`EML-inspired`, after the paper's Exp-Minus-Log framing. The implementation is
not a faithful complex-valued EML reproduction; it is a stable real-valued,
tree-style approximation for experiments.

## 1. Original Request

The user asked to add and evaluate a new model variant:

- Keep the existing transformer attention and language-model setup.
- Add a feed-forward alternative inspired by a uniform binary tree of repeated
  nodes.
- Have the module derive scalar slots/features from token hidden states.
- Let the leaves select from those slots plus constants.
- Combine leaves through a differentiable binary tree.
- Use a real-valued approximation if the exact complex EML function is
  impractical.
- Add a model/config option that can replace or augment the standard FFN/MLP.
- Prefer clean integration into the existing config, registry, runner, logging,
  and evaluation harness.
- Run a focused comparison against the existing baseline.
- If the first variant underperformed badly, make one reasonable adjustment and
  rerun once.

The desired deliverables were:

- Code integrated into the current repo.
- Config/experiment entries needed to run it.
- Actual experiment runs.
- A concise summary of changes, commands, metrics, and interpretation.

## 2. Extension Point Inspection

Codex inspected the Path 1 model surface and found the right seam:

- `python/models/transformer.py` already supports custom FFN modules through
  `LocalCausalTransformerBlock(..., ffn_module=...)`.
- `python/models/path1.py` builds the Path 1 model from a typed
  `Path1VariantSpec`.
- `python/specs/path1.py` owns the variant/config contract.
- `python/runners/path1_cli.py` is the CLI entrypoint behind
  `scripts/v3a_python_path1.py`.
- `python/runners/path1.py` already handles training, evaluation, reports, and
  ledgers.

This meant the new EML FFN could be added without a new training stack.

## 3. First Implementation: EML-Inspired FFN Profiles

Codex added real-valued EML-inspired modules in `python/models/common.py`:

- `RealEmlNodeOperator`
- `EmlInspiredTreeFeedForward`
- `GatedEmlFeedForward`

The real-valued node uses a bounded `tanh` combination:

```text
tanh(
    left_weight * left
  + right_weight * right
  + product_weight * left * right
  + difference_weight * (left - right)
  + bias
)
```

The EML tree feed-forward module:

- Projects each hidden state into a small scalar slot basis.
- Adds constants `[-1, 0, 1]`.
- Learns soft leaf selectors over slots plus constants.
- Builds `2 ** tree_depth` leaves.
- Repeatedly applies the same binary node until one root remains per tree.
- Projects the tree roots back to `d_model`.
- Raises loudly on non-finite output.

The initial gated hybrid:

```text
y_mlp = MLP(x)
y_eml = EMLTree(x)
mix = sigmoid(channel_mix)
y = (1 - mix) * y_mlp + mix * y_eml
```

The initial mix was set to 10% EML and 90% standard MLP.

## 4. Config and CLI Integration

Codex added a typed feed-forward profile enum:

```text
standard
eml-tree
mlp-eml-gated
```

The attention-only Path 1 variant gained:

- `feed_forward_profile`
- `eml_slot_count`
- `eml_tree_depth`

The CLI gained:

```bash
--feed-forward-profile standard|eml-tree|mlp-eml-gated
--eml-slot-count
--eml-tree-depth
```

Reference SSM and primitive hybrids were explicitly kept on the standard FFN
profile at this stage. This avoided accidentally applying EML to GDN/Mamba/P20
lanes before there was a clean contract.

## 5. Initial Local Validation

Codex first tried the default `python3`, which failed because that interpreter
did not have Torch installed. The repo venv was then checked:

```text
/Users/joseph/fractal/.venv/bin/python --version
Python 3.12.12

torch 2.11.0
cuda False
mps True
```

Targeted tests passed:

```bash
/Users/joseph/fractal/.venv/bin/python -m unittest \
  python.tests.test_models.Path1ModelTests.test_attention_only_forward_cpu \
  python.tests.test_models.Path1ModelTests.test_attention_only_eml_tree_feed_forward_cpu \
  python.tests.test_models.Path1ModelTests.test_attention_only_gated_eml_feed_forward_cpu
```

The Path 1 spec tests also passed:

```bash
/Users/joseph/fractal/.venv/bin/python -m unittest python.tests.test_specs
```

## 6. Initial CPU Experiment

The runner module itself did not have a `python -m` entrypoint, so Codex switched
to the intended wrapper:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/v3a_python_path1.py
```

The initial CPU experiment used:

- corpus: `fineweb-stage0-local-bench-9row-v1`
- `seq_len=64`
- `window_stride=64`
- `batch_size=4`
- `steps=32`
- `eval_batches=4`
- backend: CPU
- dtype: fp32
- seed: 42

Shared command shape:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/v3a_python_path1.py \
  --variant attention-only \
  --backend cpu \
  --dtype fp32 \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-9row-v1 \
  --seq-len 64 \
  --window-stride 64 \
  --batch-size 4 \
  --steps 32 \
  --eval-batches 4 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --output table \
  --output-dir artifacts/eml-inspired-tree-ffn \
  --ledger-path artifacts/eml-inspired-tree-ffn/ledger.jsonl
```

Initial CPU results:

| Variant | Params | Initial Loss | Final Loss | Tok/s | Interpretation |
|---|---:|---:|---:|---:|---|
| baseline attention MLP | 1.65M | 5.8957 | 2.7535 | 20,773.6 | control |
| EML-tree FFN replacement | 1.49M | 5.9962 | 3.4691 | 4,630.0 | clear miss |
| gated MLP+EML hybrid | 2.55M | 5.8052 | 2.7550 | 4,097.3 | loss recovered, too expensive |

Interpretation:

- Full EML replacement hurt badly.
- Full-stack gated EML recovered quality but was too slow and too large.
- The next reasonable move was a cheaper, surgical insertion.

## 7. Surgical EML Layer Selection

The user asked to try a cheaper surgical version.

Codex added an explicit `feed_forward_layer_indices` contract so EML FFNs could
target only selected attention layers. This was intentionally typed and explicit
rather than inferred from labels.

New CLI option:

```bash
--feed-forward-layer-indices 4
```

or:

```bash
--feed-forward-layer-indices 3,4
```

Layer indices are zero-based.

Examples:

```text
layer 4 = 5th transformer block
layers 3,4 = 4th and 5th transformer blocks
```

Surgical shape:

```text
8-layer attention-only local causal transformer
vocab=257
d_model=128
heads=4
d_ff=512
local_window=256
```

Surgical EML settings:

```text
slot_count=4
tree_depth=2
leaf_count=4
```

## 8. Surgical CPU Results

Codex ran:

1. One middle gated EML layer.
2. Two middle gated EML layers.
3. Two middle EML-tree replacement layers as a sanity check.

Results:

| Variant | Params | Final Loss | Tok/s | Interpretation |
|---|---:|---:|---:|---|
| baseline attention MLP | 1.65M | 2.7535 | 20,773.6 | control |
| all-layer gated EML, slots8/depth3 | 2.55M | 2.7550 | 4,097.3 | too expensive |
| 1-layer surgical gated EML, layer 4, slots4/depth2 | 1.73M | 2.7325 | 16,492.5 | mild positive |
| 2-layer surgical gated EML, layers 3-4, slots4/depth2 | 1.81M | 2.6941 | 13,593.2 | best CPU result |
| 2-layer surgical EML replacement, layers 3-4, slots4/depth2 | 1.55M | 2.7819 | 14,393.1 | worse than baseline |

Interpretation:

- The useful role was not EML as a replacement.
- The useful role was EML as a small gated augmentation.
- The CPU-best candidate was two middle gated EML layers.

## 9. GPU Sweep Request

The user asked whether the variants could be run on GPU:

> can we run all of these variants on GPU and see how the results compare?
> those 5 above, and the two initial runs

Codex noted that one variant was duplicated across those lists, so there were
six unique lanes:

1. Baseline attention MLP.
2. Initial all-layer EML-tree replacement.
3. Initial all-layer gated EML.
4. Surgical gated EML, layer 4.
5. Surgical gated EML, layers 3-4.
6. Surgical EML-tree replacement, layers 3-4.

## 10. GPU Sweep Script

Codex created:

```text
scripts/runpod-v3a-python-path1-eml-ffn-sweep.sh
```

The sweep used:

- RunPod RTX 4090
- `PYTHON_INSTALL_MODE=requirements-only`
- `cuda-faithful-small-v1`
- `bf16`
- `seq_len=64`
- `window_stride=64`
- `batch_size=4`
- seed 42
- full train/eval pass

Launch command:

```bash
GPU_ID="NVIDIA GeForce RTX 4090" \
PYTHON_INSTALL_MODE=requirements-only \
SEQ_LEN=64 \
BATCH_SIZE=4 \
WINDOW_STRIDE=64 \
scripts/runpod-v3a-python-path1-eml-ffn-sweep.sh 42 eml-ffn-gpu
```

RunPod created a new GPU pod named:

```text
fractal-v3a-eml-ffn
```

The sweep script kept the pod alive through intermediate lanes and stopped it
after the final lane.

## 11. GPU Sweep Results

The GPU sweep completed successfully. The GPU pod was stopped afterward.

Each run used:

```text
120 train steps
12 eval batches
```

Results:

| Variant | Params | Final Loss | Delta vs Baseline | Tok/s | Peak CUDA MB | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| baseline attention MLP | 1.65M | 2.5294 | 0.0000 | 34,674 | 48.68 | control |
| initial EML-tree all-layer replacement | 1.49M | 2.7099 | +0.1805 | 16,672 | 101.92 | clear miss |
| initial all-layer gated EML | 2.55M | 2.5473 | +0.0179 | 14,258 | 121.44 | close loss, too slow/heavy |
| surgical gated EML, layer 4 | 1.73M | 2.5160 | -0.0134 | 32,925 | 53.27 | best GPU result |
| surgical gated EML, layers 3-4 | 1.81M | 2.5362 | +0.0068 | 28,351 | 57.86 | CPU win did not reproduce |
| surgical EML-tree replacement, layers 3-4 | 1.55M | 2.5812 | +0.0518 | 35,591 | 52.96 | fast but worse loss |

Interpretation:

- The all-layer variants are effectively ruled out for this surface.
- The CPU-winning two-layer surgical gated EML did not reproduce on GPU.
- The single layer-4 surgical gated EML was the best GPU result.
- It improved final loss by `0.0134` while adding only about `4.6 MB` peak
  CUDA memory and a roughly 5% throughput cost.

The strongest candidate became:

```text
8L attention-only transformer
one gated EML side branch in layer 4
slot_count=4
tree_depth=2
```

## 12. Quantization Clarification

The user asked:

> you didn't run a quantization experiment did you?

Codex clarified that no quantization experiment had been run.

The GPU sweep measured only:

- bf16 training/eval loss
- throughput
- CUDA peak memory
- parameter count

It did not include:

- GPTQ
- int6
- zstd
- artifact size accounting
- calibration
- post-quant BPB

Conclusion:

```text
Layer-4 surgical gated EML looks promising on float/bf16 loss and runtime.
It has not yet been shown to help post-quant BPB or artifact score.
```

## 13. Shape Explanation

The user asked for the shape of the top surgical variants.

Codex summarized:

```text
Base:
8-layer attention-only local causal transformer
vocab=257
d_model=128
heads=4
d_ff=512
local_window=256
seq_len=64
batch_size=4
bf16 CUDA
seed=42
```

Layer index convention:

```text
layer 4 = 5th transformer block
```

Architectural shorthand:

```text
surgical gated EML, layer 4:
A A A A A* A A A

surgical gated EML, layers 3-4:
A A A A* A* A A A

surgical EML-tree replacement, layers 3-4:
A A A E E A A A
```

Where:

```text
A  = standard attention block with standard MLP
A* = attention block with standard MLP plus gated EML side branch
E  = attention block whose standard MLP is replaced by EML-tree FFN
```

## 14. EML vs GDN

The user asked:

> how much does our EML differ from a GDN?

Codex explained:

| Aspect | EML-inspired FFN | Gated DeltaNet / GDN |
|---|---|---|
| Primary role | Feed-forward computation | Sequence mixer |
| State across tokens | None | Yes |
| Long-range mechanism | None by itself | Recurrent matrix/KV-like state |
| Computation | Binary tree over scalar slots/constants | Gated targeted recurrent updates |
| Replaces | MLP/FFN path | Attention or recurrent mixer path |
| Best role so far | Tiny gated MLP augmentation | Main non-attention sequence mixer |

Short version:

```text
EML asks: can a structured tree improve per-token nonlinear transformation?
GDN asks: can a gated recurrent memory update replace/complement attention?
```

This helped reframe EML as an FFN/expert candidate rather than a GDN/Mamba/P20
competitor.

## 15. EML as Sparse Expert

The user suggested:

> so the EML should almost be treated like a sparse "expert"?

Codex agreed. The best mental model became:

```text
Transformer block:
attention -> normal MLP -> residual

Surgical EML expert:
attention -> normal MLP + small structured-tree expert -> residual
```

Codex noted that the current gated EML is expert-like but not a true sparse
expert because it is always on. A true sparse EML expert would need:

- Token-dependent routing.
- Top-k activation.
- Capacity or duty-cycle control.
- Zero/ramp initialization.
- Quantization-aware gating and selector stability.

## 16. Routed EML Expert Implementation

The user then said they wanted to do that and also consider hybrid models with
Mamba, GDN, or P20.

Codex inspected the existing MiniMoE implementation and found that it computes
all expert outputs before selecting them. That is useful for correctness but not
for a low-duty-cycle EML expert.

Codex implemented a purpose-built routed EML FFN:

```text
mlp-eml-routed
```

The routed profile:

- Keeps the standard MLP as the owner.
- Uses a router to score tokens.
- Selects a top fraction of tokens.
- Gathers only those selected token hidden states.
- Runs the EML tree only on selected tokens.
- Scatters the EML contribution back into the MLP output.

Conceptual form:

```text
y = MLP(x)
selected_tokens = topk(router(x), route_fraction)
y[selected_tokens] += sigmoid(router_score) * channel_mix * EML(x[selected_tokens])
```

New CLI option:

```bash
--feed-forward-profile mlp-eml-routed
--eml-route-fraction 0.25
```

## 17. Routed EML Local Tests

Codex added targeted tests for the routed profile:

- Selected layer only.
- Non-selected layers remain standard MLP.
- Selected layer uses `RoutedEmlFeedForward`.
- Diagnostics expose `route_fraction`.

The broader local model/spec suite passed:

```text
81 tests passed
```

## 18. Routed EML CPU Smoke Results

Codex ran local CPU smokes for:

- Routed layer-4 EML, 25% token duty cycle.
- Routed layer-4 EML, 50% token duty cycle.

Same CPU surface:

- `fineweb-stage0-local-bench-9row-v1`
- `seq_len=64`
- `batch_size=4`
- `steps=32`
- `eval_batches=4`
- fp32 CPU

Results:

| Variant | Params | Final Loss | Tok/s | Route Fraction | Interpretation |
|---|---:|---:|---:|---:|---|
| baseline | 1.65M | 2.7535 | 20,773.6 | n/a | control |
| dense gated EML, layer 4 | 1.73M | 2.7325 | 16,492.5 | n/a | best local loss |
| routed EML, layer 4 | 1.73M | 2.7410 | 17,878.3 | 0.25 | faster, still beats baseline |
| routed EML, layer 4 | 1.73M | 2.7413 | 16,989.8 | 0.50 | no extra gain |

Interpretation:

- Routed EML is alive.
- It improves over baseline locally.
- It gives up some loss improvement relative to dense gated EML.
- Raising route fraction from 25% to 50% did not recover the dense-gated loss.
- The first hard top-k router likely does not learn useful token selection
  quickly on this short run.

## 19. Hybrid Model Strategy

The user wanted to think about a hybrid model with Mamba, GDN, or P20 that could
win loss while remaining speed competitive, with quantization as a bonus.

Codex's recommendation:

Do not treat EML as a sequence mixer.

Instead:

```text
Mamba/GDN/P20 = sequence mixing / cross-token information movement
EML = per-token structured nonlinear expert
```

The clean hybrid structure is:

```text
sequence mixer backbone:
A/Mamba, A/GDN, or A/P20

plus:
one small EML expert in a seam-adjacent MLP block
```

Candidate lanes proposed:

| Lane | Purpose |
|---|---|
| `A + dense gated EML layer4` | Current best float/loss EML candidate |
| `A + routed EML layer4 route25` | Speed/quantization candidate |
| `A + GDN/P20/Mamba best current schedule + dense gated EML near seam` | Test whether EML stacks with sequence mixers |
| Same with routed EML | Test sparse EML speed/quality tradeoff |
| Post-quant probe on top 1-2 | Only after float/speed looks alive |

## 20. Quantization Hypothesis

The user believed EML should benefit quantization.

Codex agreed directionally but framed it as unproven:

Potential quantization advantages:

- EML tree is bounded by `tanh`.
- Inputs are low-dimensional scalar slots plus constants.
- The best candidate is surgical, not full-stack.
- Routed EML can reduce active expert usage.
- A structured expert might reduce reliance on large dense MLP capacity.

Potential quantization risks:

- Leaf selector logits may be sensitive.
- Router logits may be sensitive.
- Hard top-k routing can be brittle.
- Extra branch weights may increase artifact size.
- Gated additive paths can shift post-quant behavior even when float loss looks
  good.

Conclusion:

```text
EML may help quantization, but only a GPTQ/int6/post-quant BPB experiment can
confirm it.
```

## 21. Current Best Understanding

Current EML findings:

1. Full EML replacement is not promising.
2. Full-stack gated EML is too slow and too large.
3. Surgical dense gated EML at layer 4 is the best float/bf16 candidate.
4. Two-layer surgical EML was best on CPU but did not reproduce on GPU.
5. Routed EML is alive and somewhat faster locally, but the first hard top-k
   router gives up some loss quality.
6. EML should be treated as an expert-like FFN augmentation, not as a competitor
   to GDN/Mamba/P20.
7. The next high-value test is whether a single EML expert improves an already
   strong sequence-mixer hybrid.

## 22. Current Best Candidate

Best GPU float candidate:

```text
attention-only transformer
8 layers
single dense gated EML side branch at layer 4
slot_count=4
tree_depth=2
```

GPU result:

```text
baseline final loss: 2.5294
surgical gated EML layer 4 final loss: 2.5160
delta: -0.0134
throughput: 32,925 tok/s vs 34,674 tok/s baseline
peak memory: 53.27 MB vs 48.68 MB baseline
```

Best routed candidate:

```text
attention-only transformer
8 layers
single routed EML side branch at layer 4
slot_count=4
tree_depth=2
route_fraction=0.25
```

CPU result:

```text
baseline final loss: 2.7535
routed EML final loss: 2.7410
dense gated EML final loss: 2.7325
```

## 23. Recommended Next Steps

The next sequence should be:

1. Run a small CUDA seed sweep:
   - baseline
   - dense gated EML layer 4
   - routed EML layer 4 route 25%

2. If dense layer-4 EML remains positive, test it inside the current best
   Mamba/GDN/P20 hybrid schedules as a seam-adjacent FFN expert.

3. If routed EML remains close enough, improve the router:
   - add soft/ramped routing before hard top-k
   - add route regularization or entropy/load objective
   - test route fractions 10%, 25%, 50%

4. Only after a float/speed winner exists, run quantization:
   - GPTQ/int6
   - post-quant BPB
   - artifact size
   - calibration sensitivity

5. Preserve the conceptual separation:

```text
sequence mixers move information across tokens
EML expert adds structured per-token transformation
quantization decides whether the structured expert is useful in the final scoring surface
```
