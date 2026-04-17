# Parcae Control Local Comparison

Date: 2026-04-15

This note records the first local comparison between the stable Parcae reference scaffold and two tighter control variants:

- `parcae-looped-attention`: stable Parcae-inspired middle-loop scaffold.
- `parcae-bx-looped-attention`: same scaffold with learned `B(x)` value and gate projections for the recurrent injection.
- `parcae-p20-control-looped-attention`: same scaffold with a P20 runtime controller generating the injection value/gate path.

This is not a faithful reproduction of Parcae. It is a local architecture probe to decide whether to keep the Parcae reference lane and the P20-as-Parcae-controller lane separate.

## Implementation

The Parcae-family scaffolds preserve the Path 1 attention-only backbone and alter only the attention-block execution:

```text
prelude attention blocks -> looped middle attention blocks -> coda attention blocks
```

For the default 8-layer model, the split is:

```text
3 prelude layers -> 2 recurrent middle layers -> 3 coda layers
```

Shared stable controls:

- recurrent state starts at zero
- recurrent decay is bounded by `exp(-softplus(raw_decay))`
- injection and nonlinear update magnitudes are sigmoid-bounded
- loop count is explicit through `parcae_loop_count`

Variant-specific controls:

- `parcae-looped-attention` injects the normalized prelude stream directly.
- `parcae-bx-looped-attention` learns a value projection and channel gate for `B(x)` before injection.
- `parcae-p20-control-looped-attention` runs a P20 block-diagonal-4 runtime scan over the prelude stream and uses the P20 output to control the injected value/gate.

The P20-control lane is intentionally heavier. It tests whether the learned recurrent primitive can serve as a loop controller, not whether it is already the right production implementation.

## Corpus

All runs used the local OpenLLaMA-tokenized FineWeb cache:

```text
experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1/manifest.json
```

Corpus stats:

- vocab size: `32000`
- train tokens: `25,007,703`
- eval tokens: `2,000,938`
- tokenizer: OpenLLaMA SentencePiece

## Validation

```bash
/Users/joseph/fractal/.venv/bin/python -m py_compile \
  python/models/path1.py \
  python/specs/path1.py \
  python/runners/path1_cli.py \
  scripts/eml_ffn_expert_sweep.py \
  python/tests/test_models.py \
  python/tests/test_specs.py

/Users/joseph/fractal/.venv/bin/python -m unittest \
  python.tests.test_models \
  python.tests.test_specs \
  python.tests.test_data
```

Result:

```text
Ran 89 tests in 2.954s
OK
```

## 32-Step Three-Seed Results

Settings:

- backend: `mps`
- dtype: `fp32`
- seq len: `64`
- batch size: `4`
- eval batches: `16`
- train steps: `32`
- seeds: `42, 43, 44`
- data seed: `42`
- Parcae loops: `2`

Artifacts:

```text
artifacts/parcae-control-compare-steps32-clean
```

| Variant | Params | Mean Initial Loss | Mean Final Loss | Final Std | Mean tok/s | tok/s Min | tok/s Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention-only | 9,778,176 | 10.6437 | 7.9585 | 0.0177 | 8,406.75 | 3,460.70 | 11,357.41 |
| parcae-looped-attention | 9,778,816 | 10.4707 | 7.9645 | 0.0033 | 7,009.85 | 2,659.81 | 9,385.12 |
| parcae-bx-looped-attention | 9,811,712 | 10.4762 | 7.9148 | 0.0090 | 5,189.54 | 2,709.11 | 9,507.89 |
| parcae-p20-control-looped-attention | 9,873,856 | 10.5008 | 7.9428 | 0.0048 | 2,254.21 | 1,540.26 | 3,608.18 |

The 32-step result favors `parcae-bx-looped-attention` on validation loss. P20-control is also ahead of baseline on mean final loss, but it is much slower.

MPS throughput was noisy across seed order, so quality should be weighted more heavily than tok/s in this local sweep.

## 128-Step Seed-42 Results

Settings:

- backend: `mps`
- dtype: `fp32`
- seq len: `64`
- batch size: `4`
- eval batches: `32`
- train steps: `128`
- seed: `42`
- data seed: `42`
- Parcae loops: `2`

Artifacts:

```text
artifacts/parcae-control-compare-steps128-clean
```

| Variant | Params | Initial Loss | Final Loss | tok/s | Peak Process MB |
|---|---:|---:|---:|---:|---:|
| attention-only | 9,778,176 | 10.6211 | 7.4965 | 10,093.22 | 1,131.56 |
| parcae-looped-attention | 9,778,816 | 10.4438 | 7.5360 | 9,299.96 | 1,150.22 |
| parcae-bx-looped-attention | 9,811,712 | 10.4514 | 7.5503 | 9,263.57 | 1,150.52 |
| parcae-p20-control-looped-attention | 9,873,856 | 10.4897 | 7.4884 | 3,546.74 | 1,183.25 |

At 128 steps, P20-control is the only Parcae-family lane to edge the baseline on final loss, but the margin is tiny and the runtime cost is large.

## Interpretation

The Parcae reference and P20-control lanes should stay separate for now.

The `B(x)` projection is the better mathematical-wrapper direction for the Parcae reference lane. It produced the clearest short-run multi-seed quality signal, but it did not hold up in the longer seed-42 check.

The P20-control lane is not dead. It is stable and produced the best 128-step seed-42 loss in this comparison. However, it is not promotable as-is because it pays a roughly 3x local runtime cost for a very small loss improvement.

The practical read is:

- keep `parcae-bx-looped-attention` as the clean Parcae reference lane
- keep `parcae-p20-control-looped-attention` as a speculative control lane
- do not merge the lanes until P20-control gets a thinner controller or a stronger quality edge
- do not tune loop count blindly before fixing the compute contract

## Next Step

The next useful Parcae-side experiment is a bounded controller ablation:

- run `parcae-bx-looped-attention` across 3 seeds at 128 steps to verify whether the 32-step seam was transient
- test a thinner P20-control path, ideally bottlenecked or shared, to cut the scan tax
- only then test stochastic per-sequence loop depth, because dynamic loop depth is a training contract change and should not be mixed with controller-shape changes

## H100 Promotion

Date: 2026-04-16

The first H100 promotion used a private Hugging Face token-cache artifact:

```text
joebud/fractal-fineweb-openllama-tokens
```

The pod downloaded and verified:

```text
fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst
sha256: 10608c6929fdf1bb6bdd70d9492f94ff3df3d726001fb38ecb26c39659e02a5f
```

Run settings:

- GPU: `NVIDIA H100 80GB HBM3`
- backend: `cuda`
- dtype: `bf16`
- primitive runtime backend: `triton`
- seq len: `256`
- window stride: `257`
- batch size: `64`
- train steps: `2000`
- eval batches: `64`
- seed: `42`
- data seed: `42`
- real-data warmup: `0`
- max no-repeat steps at this shape: `15199`

Each lane saw:

```text
2000 * 64 * 256 = 32,768,000 train tokens
```

This is below the no-repeat cap, so no training-window wraparound occurred.

Artifacts:

```text
/Users/joseph/fractal/.runpod-local-logs/runpod-results/exp-53677925087aeb1c/20260416T043842Z_a01/remote/artifacts/v3a-python-path1-parcae-h100/20260416T043842Z_a01
```

Results:

| Variant | Params | Initial Loss | Final Loss | tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|
| attention-only | 9,778,176 | 10.5876 | 5.2365 | 908,609.25 | 6,774.79 |
| parcae-looped-attention | 9,778,816 | 10.4271 | 5.2624 | 648,338.17 | 6,983.55 |
| parcae-bx-looped-attention | 9,811,712 | 10.4533 | 5.2308 | 627,228.53 | 6,991.98 |
| parcae-p20-control-looped-attention | 9,873,856 | 10.5012 | 5.1813 | 584,542.58 | 7,046.88 |

H100 interpretation:

- `parcae-looped-attention` did not beat baseline.
- `parcae-bx-looped-attention` beat baseline by `0.0057` final loss, but cost about `31%` throughput.
- `parcae-p20-control-looped-attention` beat baseline by `0.0552` final loss, and beat `B(x)` by `0.0495`, while costing about `36%` throughput versus baseline.
- The P20-control lane is now alive enough to justify a confirmatory seed run and a thinner-controller ablation.

Operational notes:

- The H100 pod was stopped after the run.
- `runpodctl pod list` returned empty after completion.
- The HF token was forwarded as an environment variable and not recorded in wrapper manifests or run artifacts.
