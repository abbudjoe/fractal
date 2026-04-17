# Parcae-Inspired Local Smoke

Date: 2026-04-15

This note records a local MPS prototype of a Parcae-inspired stable middle-loop scaffold for Path 1. It is not an exact reproduction of the Parcae paper. The prototype tests whether a small, stable, looped-depth transformer scaffold is worth promoting into the larger architecture track.

## Implementation

Added `Path1ScaffoldProfile.PARCAE_LOOPED_ATTENTION`.

The scaffold keeps the existing Path 1 attention-only model surface and changes only the execution of the attention blocks:

- split attention blocks into prelude / recurrent middle / coda
- run prelude once
- normalize the prelude output before injection
- initialize recurrent state at zero
- update the recurrent state with bounded diagonal decay and bounded input injection
- reuse the middle block weights for `parcae_loop_count` passes
- run coda once

For an 8-layer model the split is:

```text
3 prelude layers -> 2 recurrent middle layers -> 3 coda layers
```

The stable controls are:

- `decay = exp(-softplus(raw_decay))`, initialized around `0.88`
- `injection = sigmoid(logit)`, initialized around `0.10`
- `nonlinear_delta_scale = sigmoid(logit)`, initialized around `0.50`

The diagnostic payload records loop count, split sizes, decay/injection statistics, and recurrent-state norms.

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
/Users/joseph/fractal/.venv/bin/python -m unittest \
  python.tests.test_models \
  python.tests.test_specs \
  python.tests.test_data
```

Result:

```text
Ran 87 tests in 4.117s
OK
```

## Commands

Baseline:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/v3a_python_path1.py \
  --variant attention-only \
  --backend mps \
  --dtype fp32 \
  --run-label parcae-local-baseline-s42-steps128 \
  --corpus-format token-ids \
  --tokenized-manifest-path experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1/manifest.json \
  --seq-len 64 \
  --window-stride 64 \
  --batch-size 4 \
  --steps 128 \
  --eval-batches 32 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --seed 42 \
  --data-seed 42 \
  --output-dir artifacts/parcae-local-compare-steps128/baseline-s42
```

Parcae loop2:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/v3a_python_path1.py \
  --variant attention-only \
  --scaffold-profile parcae-looped-attention \
  --parcae-loop-count 2 \
  --backend mps \
  --dtype fp32 \
  --run-label parcae-local-loop2-s42-steps128 \
  --corpus-format token-ids \
  --tokenized-manifest-path experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1/manifest.json \
  --seq-len 64 \
  --window-stride 64 \
  --batch-size 4 \
  --steps 128 \
  --eval-batches 32 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --seed 42 \
  --data-seed 42 \
  --output-dir artifacts/parcae-local-compare-steps128/loop2-s42
```

Parcae loop4:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/v3a_python_path1.py \
  --variant attention-only \
  --scaffold-profile parcae-looped-attention \
  --parcae-loop-count 4 \
  --backend mps \
  --dtype fp32 \
  --run-label parcae-local-loop4-s42-steps128 \
  --corpus-format token-ids \
  --tokenized-manifest-path experiments/stage0/assets/fineweb/fineweb-cc-main-2024-10-openllama-tokens-27m-v1/manifest.json \
  --seq-len 64 \
  --window-stride 64 \
  --batch-size 4 \
  --steps 128 \
  --eval-batches 32 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --seed 42 \
  --data-seed 42 \
  --output-dir artifacts/parcae-local-compare-steps128/loop4-s42
```

## Results

| Variant | Loops | Initial Loss | Final Loss | tok/s | Peak MB | Params |
|---|---:|---:|---:|---:|---:|---:|
| attention-only | 0 | 10.6211 | 7.4965 | 8496.79 | 1172.14 | 9,778,176 |
| parcae-looped-attention | 2 | 10.4438 | 7.5360 | 6742.89 | 1151.75 | 9,778,816 |
| parcae-looped-attention | 4 | 10.5129 | 7.5795 | 5381.04 | 1150.94 | 9,778,816 |

Loop2 three-seed, 32-step aggregate:

| Variant | Seeds | Mean Final Loss | Mean tok/s | Mean Peak MB |
|---|---:|---:|---:|---:|
| attention-only | 3 | 7.9585 | 7704.35 | 1216.95 |
| parcae-looped-attention loop2 | 3 | 7.9645 | 6797.86 | 1264.92 |

## Interpretation

The prototype is stable, but it does not beat the plain attention-only baseline locally.

The useful signal is diagnostic, not promotable:

- loop2 is close to baseline at 32 steps but slightly worse on mean final loss
- loop2 remains worse at 128 steps for seed 42
- loop4 is worse and slower
- recurrent-state norms grow with loop depth but do not explode in this short run

This suggests the exact local scaffold is too conservative or not faithful enough to Parcae to win. It should not be promoted as a new champion lane yet.

## Next Step

If we continue this direction, do not tune loop count blindly. The more principled next change is to make the Parcae update closer to the paper:

- initialize the recurrent state from the prelude stream instead of zero, or add an explicit learned `B(x)` injection projection
- add stochastic per-sequence loop depth during training
- test with a model where looping replaces depth, rather than adding compute on top of the same 8-layer shape

Until then, keep this as a stable local scaffold prototype, not a result-bearing architecture.
