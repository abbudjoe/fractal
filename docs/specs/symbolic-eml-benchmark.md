# Symbolic EML Benchmark Track

This track evaluates EML-style trees on the paper-native problem: fitting and
recovering shallow elementary functions, hardening them into readable formulas,
and compiling those formulas into lightweight callables.

The paper-aligned reference is "All elementary functions from a single binary
operator" (arXiv:2603.21852v2). The relevant primitive is:

```text
eml(x, y) = exp(x) - log(y)
```

The paper requires complex internal arithmetic with the principal logarithm
branch. The implementation here keeps that distinction explicit.

## Model Families

- `paper-complex-eml`: closest practical surrogate in this repo. It uses complex
  EML internally, terminal choices over `1` and `x`, soft simplex selectors, and
  hard argmax snapping. Numeric execution clamps the real part of exponential
  inputs and guards logarithms near zero. Those guards make it a surrogate, not a
  proof-faithful replica.
- `stable-real-eml`: real-valued gated square tree patterned after the repo's
  stable P1-style primitive. This is the practical LM-track approximation, not
  the paper's complex EML operator.
- `generic-tree`: same broad binary-tree harness with generic primitive choices
  such as add, subtract, multiply, protected divide, exp, and log-abs. It carries
  no EML-specific claim.
- `small-mlp`: dense tanh baseline. It exports executable weights, but it is not
  a symbolic recovery path.

## Dataset

The default compact suite has two formula families per difficulty depth, depths
1 through 4. Each task records:

- source formula
- declared difficulty depth
- AST depth and complexity
- train, validation, and extrapolation ranges
- train, validation, and extrapolation sample counts

Domains are chosen to avoid branch and division singularities in the generated
targets. Extrapolation samples use a disjoint positive range beyond the training
interval.

## Training And Hardening

The active local Python in this worktree is Python 3.14 without PyTorch. To keep
the track runnable immediately, the benchmark uses only the standard library:

- tree families default to a small forward-mode autodiff engine with Adam-style
  moments over soft selectors, readout parameters, and tree-local operator
  parameters
- `--tree-optimizer torch-autodiff` uses PyTorch autograd and can run on CUDA or
  MPS via `--backend cuda` / `--backend mps`; use
  `uv run --python 3.12 --with torch --with numpy ...` because the repo's
  default Python is 3.14
- the previous deterministic SPSA optimizer is still available through
  `--tree-optimizer spsa` for before/after comparisons
- `small-mlp` uses an analytic tanh MLP gradient with Adam-style moments
- hardening snaps selectors by argmax, sharpens selectors after autodiff runs,
  and snaps near-simple scalar values
- compiled execution uses a restricted Python lambda over safe helper functions

The Torch CUDA/MPS path reuses the same dataset, hardening, export, and report
contracts as the CPU path. For the paper-complex arm, the Torch path also
supports deterministic EML-native restarts and scores candidate restarts after
hardening/readout refit on the train split, because the paper claim is about
recoverable hardened structure rather than lowest soft loss alone.

The paper-complex hardening path also includes deliberately small search layers:

- depth-2 EML nesting search over the 36 unique roots reachable with terminals
  `1`, `x`, and one nested EML level
- log-lift product/ratio search over 24 additional roots, using
  `log(z) = e - eml(1, z)` for `z in {x, x + 2}` and small integer log
  combinations inside `eml(log_combo, 1)`
- offset-scheduled log-lift plus sparse readout search, using offsets
  `{-0.3, 0, 0.5, 1.2, 1.5, 2, 2.5, 3}` and selecting up to two features with
  a fitted linear readout

These layers refit only scalar or sparse linear readouts on the train split and
then let the existing validation/extrapolation metrics decide exactness. This is
a hardening-time symbolic search, not a claim that the soft optimizer alone
discovered every selected expression.

## Commands

Smoke:

```bash
python scripts/symbolic_benchmark.py \
  --preset smoke \
  --run-label symbolic-smoke-check \
  --output-dir artifacts/symbolic-benchmark/symbolic-smoke-check \
  --seeds 1 \
  --steps 8
```

Compact result run:

```bash
python scripts/symbolic_benchmark.py \
  --preset compact \
  --run-label symbolic-compact-v3 \
  --output-dir artifacts/symbolic-benchmark/symbolic-compact-v3 \
  --seeds 2 \
  --tasks-per-depth 2 \
  --steps 180
```

Tier-0 exact recovery run:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_benchmark.py \
  --preset tier0-exact \
  --run-label symbolic-tier0-torch-mps-sparse-loglift-v2 \
  --output-dir artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-sparse-loglift-v2 \
  --seeds 2 \
  --steps 240 \
  --tree-learning-rate 0.05 \
  --tree-optimizer torch-autodiff \
  --backend mps \
  --paper-restarts 6
```

Compact Torch/MPS run:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_benchmark.py \
  --preset compact \
  --run-label symbolic-compact-torch-mps-sparse-loglift-v2 \
  --output-dir artifacts/symbolic-benchmark/symbolic-compact-torch-mps-sparse-loglift-v2 \
  --seeds 2 \
  --tasks-per-depth 2 \
  --steps 180 \
  --tree-learning-rate 0.05 \
  --tree-optimizer torch-autodiff \
  --backend mps \
  --paper-restarts 6
```

Compact optimizer depth slice:

```bash
python scripts/symbolic_benchmark.py \
  --preset compact \
  --run-label symbolic-compact-spsa-depthslice-v1 \
  --output-dir artifacts/symbolic-benchmark/symbolic-compact-spsa-depthslice-v1 \
  --seeds 1 \
  --tasks-per-depth 1 \
  --steps 30 \
  --tree-optimizer spsa

uv run --python 3.12 --with torch --with numpy python scripts/symbolic_benchmark.py \
  --preset compact \
  --run-label symbolic-compact-torch-mps-depthslice-v1 \
  --output-dir artifacts/symbolic-benchmark/symbolic-compact-torch-mps-depthslice-v1 \
  --seeds 1 \
  --tasks-per-depth 1 \
  --steps 80 \
  --tree-learning-rate 0.05 \
  --tree-optimizer torch-autodiff \
  --backend mps
```

Focused tests:

```bash
python -m unittest python.tests.test_symbolic
python -m py_compile \
  python/specs/symbolic.py \
  python/symbolic/formulas.py \
  python/symbolic/models.py \
  python/symbolic/autodiff.py \
  python/symbolic/torch_backend.py \
  python/symbolic/runner.py \
  scripts/symbolic_benchmark.py

uv run --python 3.12 --with torch --with numpy python -m unittest python.tests.test_symbolic
```

Bridge probe:

```bash
python scripts/symbolic_bridge.py \
  --symbolic-summary artifacts/symbolic-benchmark/symbolic-compact-torch-mps-sparse-loglift-v2/summary.json \
  --run-label symbolic-bridge-compact-both-v3 \
  --output-dir artifacts/symbolic-bridge/symbolic-bridge-compact-both-v3 \
  --token-bins 32
```

## Initial Results

Artifact bundle:

```text
artifacts/symbolic-benchmark/symbolic-compact-v3/
```

Summary over 64 runs, two formula families per depth, depths 1-4, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | harden | export | compile | latency us/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.0987 | 0.3417 | 0.8127 | 0.00 | 0.75 | 1.00 | 1.00 | 0.0639 |
| `paper-complex-eml` | 0.2596 | 0.3702 | 1.3748 | 0.00 | 0.94 | 1.00 | 1.00 | 0.5670 |
| `small-mlp` | 0.0170 | 0.0512 | 0.3929 | 0.00 | 0.81 | 0.00 | 1.00 | 0.4600 |
| `stable-real-eml` | 0.0586 | 0.4585 | 0.9210 | 0.00 | 0.50 | 1.00 | 1.00 | 0.5092 |

Best soft fit: `small-mlp`.

Best hardening success rate: `paper-complex-eml`.

Best hardened extrapolation: `small-mlp`.

## Torch/MPS Results

Tier-0 artifact bundle without hardening-aware paper restarts:

```text
artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-v1/
```

Summary over 40 runs, five shallow tasks, four model families, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.5630 | 0.5630 | 1.3467 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.1783 | 0.1783 | 0.3294 | 0.20 | 0.20 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0275 | 0.0892 | 0.9706 | 0.00 | 0.00 | 0.90 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0415 | 0.0557 | 0.8325 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

`paper-complex-eml` exactly recovered `exp(x)` in both seeds:

```text
(1 * real(eml(x, 1)) + 0)
```

That is the first nonzero exact recovery result for the paper-aligned arm in this
track.

Tier-0 artifact bundle with six paper-complex restarts and hardening-aware
restart selection:

```text
artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-restarts-v2/
```

Summary over 40 runs, five shallow tasks, four model families, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.5630 | 0.5630 | 1.3467 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.1258 | 0.0000 | 9.930e-17 | 0.60 | 0.60 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0275 | 0.0892 | 0.9706 | 0.00 | 0.00 | 0.90 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0415 | 0.0557 | 0.8325 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

`paper-complex-eml` exactly recovered all depth-1 EML-native tier-0 targets in
both seeds:

```text
exp(x)                  -> (1 * real(eml(x, 1)) + 0)
2.7182818 - log(x)      -> (1 * real(eml(1, x)) + 0)
exp(x) - log(x)         -> (1 * real(eml(x, x)) + 0)
```

It did not exactly recover the non-EML-native depth-2 controls `x * x` or
`x / (x + 2)` under this budget.

Tier-0 artifact bundle with depth-2 hardening search:

```text
artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-depth2-search-v1/
```

Summary over 64 runs, eight shallow tasks, four model families, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.5630 | 0.5630 | 1.5234 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.1473 | 1.241e-17 | 6.450e-17 | 0.75 | 0.75 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0275 | 0.0892 | 0.9706 | 0.00 | 0.00 | 0.88 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0355 | 0.0453 | 0.9171 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

The recovered depth-2 EML-native targets were:

```text
exp(exp(x))                  -> (1 * real(eml(eml(x, 1), 1)) + 0)
e - log(e - log(x))          -> (1 * real(eml(1, eml(1, x))) + 0)
exp(e - log(x))              -> (1 * real(eml(eml(1, x), 1)) + 0)
```

Depth-2 exact recovery count for `paper-complex-eml` was 6/10 runs: the three
EML-native depth-2 tasks recovered in both seeds, while the polynomial and
safe-ratio controls still failed exactness.

Tier-0 artifact bundle with depth-2 plus log-lift hardening search:

```text
artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-loglift-search-v1/
```

Summary over 64 runs, eight shallow tasks, four model families, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.5630 | 0.5630 | 1.5234 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.1473 | 1.241e-17 | 5.269e-17 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0275 | 0.0892 | 0.9706 | 0.00 | 0.00 | 0.88 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0355 | 0.0453 | 0.9171 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

The two previously failed tier-0 controls now recover through log-lift
identities:

```text
x * x       -> exp(2 log(x))
x / (x+2)   -> 1 - 2 exp(-log(x+2))
```

The exported formulas are still EML-form expressions. The `x*x` case is exact
under the benchmark tolerance, with small residual error from guarded complex
log evaluation near zero and the affine readout refit.

Tier-0 artifact bundle with offset-scheduled log-lift plus sparse two-feature
readout:

```text
artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-sparse-loglift-v2/
```

Summary over 64 runs, eight shallow tasks, four model families, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.5630 | 0.5630 | 1.5234 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.1473 | 1.241e-17 | 5.269e-17 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0275 | 0.0892 | 0.9706 | 0.00 | 0.00 | 0.88 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0355 | 0.0453 | 0.9171 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |

This preserves the 16/16 paper-complex exact recovery result while preferring
simpler expressions when train RMSE is already within exact-level tolerance.

Compact Torch/MPS artifact bundle before sparse readout:

```text
artifacts/symbolic-benchmark/symbolic-compact-torch-mps-restarts-v2/
```

Summary over 64 runs, two formula families per depth, depths 1-4, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.3198 | 0.3198 | 0.5828 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.0946 | 0.0470 | 0.5338 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0170 | 0.0512 | 0.3929 | 0.00 | 0.00 | 0.81 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0180 | 0.0203 | 0.0832 | 0.00 | 0.06 | 1.00 | 1.00 | 1.00 |

Compact before/after:

| run | paper soft val RMSE | paper hard extrap RMSE | stable soft val RMSE | stable hard extrap RMSE | symbolic recovery leader |
| --- | ---: | ---: | ---: | ---: | --- |
| SPSA compact v3 | 0.2596 | 1.3748 | 0.0586 | 0.9210 | `paper-complex-eml` by hardening only |
| Torch/MPS compact v1 | 0.1717 | 0.7173 | 0.0180 | 0.0832 | `stable-real-eml` |
| Torch/MPS compact restarts v2 | 0.0946 | 0.5338 | 0.0180 | 0.0832 | `stable-real-eml` |
| Torch/MPS paper depth-2 search v1 | 0.0946 | 0.4012 | n/a | n/a | no exact paper recovery |
| Torch/MPS paper log-lift search v1 | 0.0946 | 0.4169 | n/a | n/a | no exact paper recovery |
| Torch/MPS sparse log-lift v2 | 0.0946 | 2.317e-05 | 0.0180 | 0.0832 | `paper-complex-eml` |

Compact Torch/MPS artifact bundle with sparse log-lift:

```text
artifacts/symbolic-benchmark/symbolic-compact-torch-mps-sparse-loglift-v2/
```

Summary over 64 runs, two formula families per depth, depths 1-4, two seeds:

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.3198 | 0.3198 | 0.5828 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.0946 | 2.033e-05 | 2.317e-05 | 0.62 | 0.62 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0170 | 0.0512 | 0.3929 | 0.00 | 0.00 | 0.81 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0180 | 0.0203 | 0.0832 | 0.00 | 0.06 | 1.00 | 1.00 | 1.00 |

Paper-complex exact recoveries on compact:

```text
d1_affine
d2_quadratic_mix
d2_reciprocal_shift
d3_log_quadratic
d3_ratio_product
```

Depth 4 remains unrecovered under this search layer.

Tier-0 before/after:

| run | paper soft val RMSE | paper hard val RMSE | paper hard extrap RMSE | paper exact | paper near exact | symbolic recovery leader |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU forward autodiff v1 | 0.1882 | 0.1882 | 0.6980 | 0.20 | 0.20 | `paper-complex-eml` |
| Torch/MPS v1 | 0.1783 | 0.1783 | 0.3294 | 0.20 | 0.20 | `paper-complex-eml` |
| Torch/MPS restarts v2 | 0.1258 | 0.0000 | 9.930e-17 | 0.60 | 0.60 | `paper-complex-eml` |
| Torch/MPS depth-2 search v1 | 0.1473 | 1.241e-17 | 6.450e-17 | 0.75 | 0.75 | `paper-complex-eml` |
| Torch/MPS log-lift search v1 | 0.1473 | 1.241e-17 | 5.269e-17 | 1.00 | 1.00 | `paper-complex-eml` |
| Torch/MPS sparse log-lift v2 | 0.1473 | 1.241e-17 | 5.269e-17 | 1.00 | 1.00 | `paper-complex-eml` |

Compact depth-slice before/after:

| optimizer | paper soft val RMSE | paper hard extrap RMSE | stable soft val RMSE | stable hard extrap RMSE | symbolic recovery leader |
| --- | ---: | ---: | ---: | ---: | --- |
| SPSA | 0.7157 | 2.1771 | 0.3169 | 1.5708 | `small-mlp` |
| Torch/MPS autodiff | 0.3309 | 1.0137 | 0.0389 | 0.0631 | `stable-real-eml` |

## Interpretation

The first CPU/SPSA implementation proved the harness path, not the EML claim.
The Torch/MPS autodiff upgrade plus hardening-aware restarts gave exact recovery
for the root-level EML identities `eml(x, 1)`, `eml(1, x)`, and `eml(x, x)`.
The depth-2 hardening search extends that result: `paper-complex-eml` now
recovers nested EML-native formulas such as `eml(eml(x, 1), 1)` and
`eml(1, eml(1, x))` in both seeds.

The evidence is now substantially stronger. The log-lift search recovers the
non-EML-native tier-0 controls `x * x` and `x / (x + 2)` through exact EML/log
identities, bringing paper-complex tier-0 exact recovery to 16/16. The
offset-scheduled sparse readout then extends recovery to the compact suite:
paper-complex exactly recovers the affine task, both depth-2 tasks, and both
depth-3 tasks, for 10/16 compact exact recoveries.

The result is still bounded. Depth-4 compact compositions are not recovered, and
the MLP still wins soft approximation. But the compact symbolic-recovery leader
has moved from the stable real surrogate to the paper-complex arm. This is the
first result in the track that makes the paper-aligned line look bridge-worthy,
provided the next bridge is still a symbolic/LM-adjacent probe rather than a
full language-model integration.

## Bridge Probe

The first bridge probe consumes frozen symbolic benchmark artifacts and converts
each formula stream into a tokenized next-value prediction problem with 32 bins.
It does not retrain the symbolic models. It asks whether the compiled hardened
formula is useful as a token predictor under held-out extrapolation.

Compact bridge artifact bundle:

```text
artifacts/symbolic-bridge/symbolic-bridge-compact-both-v3/
```

| model | validation token accuracy | extrap token accuracy | validation RMSE | extrap RMSE | source exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.104 | 0.062 | 0.3198 | 0.5828 | 0.00 |
| `paper-complex-eml` | 1.000 | 1.000 | 2.033e-05 | 2.317e-05 | 0.62 |
| `small-mlp` | 0.401 | 0.000 | 0.0512 | 0.3929 | 0.00 |
| `stable-real-eml` | 0.620 | 0.245 | 0.0203 | 0.0832 | 0.00 |
| `token-majority` | 0.172 | 0.000 | 0.3710 | 1.3650 | 0.00 |

Tier-0 bridge artifact bundle:

```text
artifacts/symbolic-bridge/symbolic-bridge-tier0-both-v3/
```

| model | validation token accuracy | extrap token accuracy | validation RMSE | extrap RMSE | source exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.081 | 0.000 | 0.5630 | 1.5230 | 0.00 |
| `paper-complex-eml` | 1.000 | 1.000 | 1.241e-17 | 5.269e-17 | 1.00 |
| `small-mlp` | 0.525 | 0.000 | 0.0892 | 0.9706 | 0.00 |
| `stable-real-eml` | 0.263 | 0.000 | 0.0453 | 0.9171 | 0.00 |
| `token-majority` | 0.138 | 0.000 | 0.5281 | 2.4330 | 0.00 |

This is a bridge signal, not an LM result. It says that hardened paper-complex
formulas transfer cleanly into a token-prediction framing on formula-generated
streams, especially out of range. A real LM bridge should now test these
compiled symbolic experts as frozen side-channel features or router targets in
a tiny byte-level canary, with attention-only and current primitive-hybrid
baselines held fixed.

Two bridge variants were tried:

1. Frozen side-channel: fit a tiny linear readout over `x` plus each frozen
   compiled expert prediction on the train split, then score tokenized
   validation/extrapolation outputs.
2. Router target: choose the best compiled expert per task/seed from train-token
   accuracy, then route validation/extrapolation tokens through that expert.
   An oracle token-router upper bound is also recorded.

The bridge now writes a feature table for the next LM-adjacent canary:

```text
artifacts/symbolic-bridge/symbolic-bridge-compact-both-v3/feature_table.jsonl
artifacts/symbolic-bridge/symbolic-bridge-tier0-both-v3/feature_table.jsonl
```

Each row records `task_id`, `seed`, split, `x`, target value/token, per-expert
compiled predictions/tokens/residuals, a `safe_expert_mask`, and
`best_expert_id`. Compact contains 3,840 rows with 0.911 safe-expert coverage.
Tier-0 contains 3,200 rows with 1.000 safe-expert coverage.

Compact bridge variant summary:

| probe | validation token accuracy | extrap token accuracy | mean extrap token accuracy | leader |
| --- | ---: | ---: | ---: | --- |
| direct compiled expert | 1.000 | 1.000 | n/a | `paper-complex-eml` |
| frozen side-channel | n/a | 1.000 | n/a | `paper-complex-eml` |
| task router target | 1.000 | 1.000 | 0.772 | `paper-complex-eml` |
| oracle token router | n/a | 1.000 | 0.783 | upper bound |

The router median is perfect because most compact tasks are recovered, but the
mean exposes the remaining tail: `d1_exp_soft` and `d4_nested_ratio_exp` still
route to paper-complex from train accuracy and fail out of range. That is useful
for the next LM-facing canary: router supervision should include extrapolation
or uncertainty features, not only in-range train fit.

## Bridge Canary

The second bridge pass trains a tiny Torch classifier on the bridge feature
table. It is still not a language model. It is a canary for two LM-facing
integration contracts:

1. Frozen side-channel: predict target tokens from `x`, task identity, and
   frozen expert predictions/tokens.
2. Router target: predict which compiled expert to call, then score the token
   produced by the selected expert.

The canary is intentionally small: a single hidden layer with 32 units, 800
epochs, and MPS when available. It also records deterministic baselines:
majority token, best single expert by train split, and a target-leaking
row-oracle router ceiling.

Compact canary artifact bundle:

```text
artifacts/symbolic-bridge-canary/symbolic-bridge-canary-compact-mlp32-v1/
```

| run | validation token accuracy | extrap token accuracy | router validation accuracy | router extrap accuracy |
| --- | ---: | ---: | ---: | ---: |
| `token-majority` | 0.195 | 0.000 | n/a | n/a |
| `best-single-expert-by-train` | 0.990 | 0.772 | n/a | n/a |
| `oracle-row-router` | 0.996 | 0.783 | 1.000 | 1.000 |
| `token-x-task-mlp32` | 0.762 | 0.014 | n/a | n/a |
| `token-frozen-side-channel-mlp32` | 0.851 | 0.014 | n/a | n/a |
| `router-x-task-mlp32` | 0.990 | 0.772 | 0.984 | 0.887 |
| `router-expert-signal-mlp32` | 0.990 | 0.780 | 0.987 | 0.895 |

Tier-0 canary artifact bundle:

```text
artifacts/symbolic-bridge-canary/symbolic-bridge-canary-tier0-mlp32-v1/
```

| run | validation token accuracy | extrap token accuracy | router validation accuracy | router extrap accuracy |
| --- | ---: | ---: | ---: | ---: |
| `token-majority` | 0.156 | 0.050 | n/a | n/a |
| `best-single-expert-by-train` | 1.000 | 1.000 | n/a | n/a |
| `oracle-row-router` | 1.000 | 1.000 | 1.000 | 1.000 |
| `token-x-task-mlp32` | 0.753 | 0.014 | n/a | n/a |
| `token-frozen-side-channel-mlp32` | 0.847 | 0.042 | n/a | n/a |
| `router-x-task-mlp32` | 1.000 | 1.000 | 1.000 | 1.000 |
| `router-expert-signal-mlp32` | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

- The router-target path carries forward cleanly. On tier-0 it is perfect; on
  compact it reaches 0.780 extrapolation token accuracy, close to the
  row-oracle ceiling of 0.783 and slightly above the best-single-expert
  baseline of 0.772.
- The frozen side-channel helps in validation but does not solve extrapolation
  when trained as a direct token classifier over numeric bins. This is not a
  contradiction of the earlier side-channel result: the earlier probe used a
  continuous readout and then tokenized the result, while this canary asks a
  learned token head to emit out-of-range bins that are sparse or absent in the
  train split.
- The bridge should therefore move forward first as router supervision plus
  direct compiled expert calls. A frozen side-channel is still worth testing,
  but it should be represented as a continuous/value feature or textual numeric
  annotation, not only as a learned token head over extrapolation bins.

## Sequence Bridge

The third bridge pass moves one step closer to an LM interface while still using
the controlled formula stream. It groups bridge rows by `(task_id, seed, split)`
and trains tiny GRU next-token heads from the previous target token plus
side-channel features. This is still a canary, not a full LM, but it tests the
shape that matters next: sequence state plus optional symbolic expert calls.

Variants:

1. Token heads: predict the next formula token directly.
2. Continuous heads: predict the next real value, then quantize it.
3. Router-call heads: predict which compiled expert to call, then use that
   expert's token as the output.

Compact sequence bridge artifact bundle:

```text
artifacts/symbolic-sequence-bridge/symbolic-sequence-bridge-compact-gru32-v1/
```

| run | validation token accuracy | extrap token accuracy | extrap RMSE | router validation accuracy | router extrap accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seq-best-single-expert-by-train` | 0.990 | 0.772 | n/a | n/a | n/a |
| `seq-oracle-row-router` | 0.996 | 0.783 | n/a | 1.000 | 1.000 |
| `seq-token-x-task-gru32` | 0.193 | 0.013 | n/a | n/a | n/a |
| `seq-token-side-channel-gru32` | 0.374 | 0.010 | n/a | n/a | n/a |
| `seq-continuous-x-task-gru32` | 0.198 | 0.000 | 1.174 | n/a | n/a |
| `seq-continuous-side-channel-gru32` | 0.413 | 0.020 | 0.816 | n/a | n/a |
| `seq-router-x-task-gru32` | 0.990 | 0.772 | n/a | 0.986 | 0.878 |
| `seq-router-expert-signal-gru32` | 0.975 | 0.772 | n/a | 0.962 | 0.887 |

Tier-0 sequence bridge artifact bundle:

```text
artifacts/symbolic-sequence-bridge/symbolic-sequence-bridge-tier0-gru32-v1/
```

| run | validation token accuracy | extrap token accuracy | extrap RMSE | router validation accuracy | router extrap accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `seq-best-single-expert-by-train` | 1.000 | 1.000 | n/a | n/a | n/a |
| `seq-oracle-row-router` | 1.000 | 1.000 | n/a | 1.000 | 1.000 |
| `seq-token-x-task-gru32` | 0.047 | 0.016 | n/a | n/a | n/a |
| `seq-token-side-channel-gru32` | 0.136 | 0.041 | n/a | n/a | n/a |
| `seq-continuous-x-task-gru32` | 0.100 | 0.013 | 2.685 | n/a | n/a |
| `seq-continuous-side-channel-gru32` | 0.358 | 0.080 | 0.650 | n/a | n/a |
| `seq-router-x-task-gru32` | 1.000 | 1.000 | n/a | 1.000 | 1.000 |
| `seq-router-expert-signal-gru32` | 1.000 | 1.000 | n/a | 1.000 | 1.000 |

Interpretation:

- Router-call sequence training is the only bridge shape that preserves the
  symbolic win. On tier-0 it is perfect. On compact it matches the global
  paper-complex expert's 0.772 extrapolation token accuracy but cannot exceed
  the current safe-expert ceiling of 0.783.
- Continuous frozen side-channel features reduce RMSE substantially
  (`0.816` vs `1.174` on compact extrapolation and `0.650` vs `2.685` on
  tier-0 extrapolation), but token-bin extrapolation remains weak. This says
  the side-channel is useful value information, not a complete symbolic control
  plane by itself.
- The next true LM bridge should expose compiled experts as callable tools or
  router-selected adapters, and should log abstention/failure when no safe
  symbolic expert exists. Feeding expert values as plain numeric side-channel
  features may help calibration, but it should not be the primary success path.

## LM Contract Confirmation

The fourth bridge pass tests the actual control-plane contract proposed for a
later LM integration. It uses a tiny GRU LM-style backbone with a typed symbolic
batch:

- previous target tokens as `input_ids`
- target formula tokens
- per-position `x`
- compiled expert values/tokens
- expert valid/safe masks
- router targets over `E experts + ABSTAIN`

Variants:

1. `lm-token-only`: previous-token LM, no symbolic context.
2. `lm-x-task`: LM plus non-symbolic `x` and task identity.
3. `lm-frozen-side-channel`: LM plus frozen expert values/tokens as features.
4. `lm-router-hard-call`: LM plus a router head. If the router selects a valid
   expert, the compiled expert token is used; otherwise the LM fallback is used.

Tier-0 contract artifact bundle:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-tier0-contract-v3/
```

| run | validation final accuracy | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.017 | 0.006 | 0.006 | n/a | 0.000 | 0.000 |
| `lm-x-task` | 0.036 | 0.008 | 0.008 | n/a | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.055 | 0.017 | 0.017 | n/a | 0.000 | 0.000 |
| `lm-router-hard-call` | 1.000 | 1.000 | 0.047 | 1.000 | 1.000 | 0.000 |

Tier-0 verdict:

- Capability gain confirmed: true.
- Safe abstention confirmed: true. No abstain targets exist in tier-0, but the
  router makes no unsafe expert calls.
- Contract confirmed: true.

Compact contract artifact bundle:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-contract-v2/
```

| run | validation final accuracy | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate | abstain recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.033 | 0.000 | 0.000 | n/a | 0.000 | 0.000 | n/a |
| `lm-x-task` | 0.089 | 0.005 | 0.005 | n/a | 0.000 | 0.000 | n/a |
| `lm-frozen-side-channel` | 0.292 | 0.004 | 0.004 | n/a | 0.000 | 0.000 | n/a |
| `lm-router-hard-call` | 0.976 | 0.772 | 0.002 | 0.766 | 1.000 | 0.228 | 0.000 |

Compact verdict:

- Capability gain confirmed: true. The router contract lifts extrapolation from
  `0.004` for frozen side-channel LM to `0.772`, matching the best global
  paper-complex expert.
- Safe abstention confirmed: false. The train split has no abstain targets
  (`0.000`), while extrapolation has `0.217`. The trained router therefore
  learns to always call an expert, yielding a `0.228` unsafe-call rate and
  `0.000` abstain recall.
- Contract confirmed: false for compact as currently generated. The contract is
  representationally right, but the training data must include abstain/safety
  calibration examples before bridging into a real LM.

Recommendation:

- Promote the router-call contract, not direct token prediction from side
  channels.
- Add a safety-calibration split to the symbolic bridge feature table before
  the LM bridge: generated widened-range rows that include both safe and no-safe
  expert cases, separate from validation and extrapolation.
- Treat `ABSTAIN` as a first-class route in the LM loss and report unsafe-call
  rate as a promotion gate. The bridge is not ready for broad LM tasks until the
  compact run confirms both capability and safe abstention.

## Safety-Calibration Follow-Up

The bridge feature table now includes a fourth split, `safety_calibration`,
sampled from widened formula ranges and selected after expert scoring so that
safe and no-safe expert cases are both represented when available. The LM
contract runner trains on `train + safety_calibration` but still reports
validation and extrapolation separately.

Updated bridge artifacts:

```text
artifacts/symbolic-bridge/symbolic-bridge-tier0-both-v4/
artifacts/symbolic-bridge/symbolic-bridge-compact-both-v7/
```

| preset | train rows | safety rows | validation rows | extrap rows | safety safe-expert coverage | extrap safe-expert coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tier-0 | 640 | 640 | 1280 | 1280 | 1.000 | 1.000 |
| compact v5 | 768 | 768 | 1536 | 1536 | 0.785 | 0.783 |
| compact v7 | 768 | 1536 | 1536 | 1536 | 0.786 | 0.783 |

The compact v7 table keeps the deliberately pessimistic no-safe calibration
sample, expands safety calibration to the validation/extrapolation sequence
length, and prefers out-of-training-range safe rows nearest to no-safe rows. The
LM batch contract now masks variable-length sequences, so the expanded safety
split can train together with the shorter train split without treating padding
as data. This moves the fit split closer to extrapolation:
`fit_abstain_target_rate=0.142` versus `0.217` on extrapolation. The earlier
compact v5 table had only `fit_abstain_target_rate=0.107`.

Safety-calibrated bridge commands:

```bash
python scripts/symbolic_bridge.py \
  --symbolic-summary artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-sparse-loglift-v2/summary.json \
  --run-label symbolic-bridge-tier0-both-v4 \
  --output-dir artifacts/symbolic-bridge/symbolic-bridge-tier0-both-v4 \
  --token-bins 32

python scripts/symbolic_bridge.py \
  --symbolic-summary artifacts/symbolic-benchmark/symbolic-compact-torch-mps-sparse-loglift-v2/summary.json \
  --run-label symbolic-bridge-compact-both-v7 \
  --output-dir artifacts/symbolic-bridge/symbolic-bridge-compact-both-v7 \
  --token-bins 32
```

Tier-0 safety-calibrated LM artifact:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-tier0-safety-v1/
```

| run | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.205 | 0.205 | n/a | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.069 | 0.069 | n/a | 0.000 | 0.000 |
| `lm-router-hard-call` | 1.000 | 0.042 | 1.000 | 1.000 | 0.000 |

Compact safety-calibrated LM artifact:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-v7-c5-t99999-v1/
```

| run | call/abstain loss | threshold | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate | abstain recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | n/a | n/a | 0.599 | 0.599 | n/a | 0.000 | 0.000 | n/a |
| `lm-x-task` | n/a | n/a | 0.555 | 0.555 | n/a | 0.000 | 0.000 | n/a |
| `lm-frozen-side-channel` | n/a | n/a | 0.600 | 0.600 | n/a | 0.000 | 0.000 | n/a |
| `lm-router-hard-call` | 5.0 | 0.99999 | 0.865 | 0.647 | 0.792 | 0.588 | 0.007 | 0.970 |

Safety-calibrated LM command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-both-v7/summary.json \
  --run-label symbolic-bridge-lm-compact-v7-c5-t99999-v1 \
  --output-dir artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-v7-c5-t99999-v1 \
  --epochs 1800 \
  --learning-rate 0.004 \
  --hidden-units 96 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 5.0 \
  --router-call-threshold 0.99999 \
  --device auto \
  --output table
```

Compact calibration sweep:

| artifact | router loss | abstain class | unsafe-prob loss | call/abstain loss | unsafe margin loss | threshold | extrap final accuracy | unsafe call rate | abstain recall | contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `symbolic-bridge-lm-compact-safety-v5-threshold80-v1` | 10.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.831 | 0.020 | 0.913 | false |
| `symbolic-bridge-lm-compact-v7-control-t80-v1` | 10.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.884 | 0.053 | 0.763 | false |
| `symbolic-bridge-lm-compact-v7-control-t9999-v1` | 10.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.9999 | 0.859 | 0.015 | 0.934 | false |
| `symbolic-bridge-lm-compact-v7-control-t99999-v1` | 10.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.99999 | 0.859 | 0.008 | 0.967 | true |
| `symbolic-bridge-lm-compact-v7-c5-t99999-v1` | 10.0 | 1.0 | 0.0 | 5.0 | 0.0 | 0.99999 | 0.865 | 0.007 | 0.970 | true |
| `symbolic-bridge-lm-compact-v7-c5-m5-t80-v1` | 10.0 | 1.0 | 0.0 | 5.0 | 5.0 | 0.8 | 0.887 | 0.079 | 0.647 | false |
| `symbolic-bridge-lm-compact-safety-rw30-v2` | 30.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.855 | 0.124 | 0.452 | false |
| `symbolic-bridge-lm-compact-safety-threshold999-v1` | 10.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.999 | 0.840 | 0.049 | 0.775 | false |
| `symbolic-bridge-lm-compact-calibrated-u1-a2-t80-v1` | 10.0 | 2.0 | 1.0 | 0.0 | 0.0 | 0.8 | 0.835 | 0.066 | 0.707 | false |
| `symbolic-bridge-lm-compact-calibrated-u10-a4-t80-v1` | 10.0 | 4.0 | 10.0 | 0.0 | 0.0 | 0.8 | 0.859 | 0.088 | 0.611 | false |

Safety-calibration verdict:

- The data fix worked: compact fit now contains more abstain examples
  (`fit_abstain_target_rate=0.142`) and extrapolation no-safe cases are no
  longer completely unseen.
- The contract is now clearly capability-improving on compact: routed final
  accuracy reaches `0.865`, above the frozen-side-channel LM (`0.600`) and
  above the best global expert under the same safe-call accounting (`0.772`).
  This happens because the LM fallback can answer some no-safe cases while
  expert calls handle recovered formulas.
- Safe abstention is now confirmed on compact. The best compact run cuts unsafe
  calls from `0.228` before calibration to `0.007`, raises abstain recall to
  `0.970`, and passes the strict `<=0.01` unsafe-call gate.
- The useful recipe is expanded near-boundary safety data plus a calibrated call
  threshold. A small call/abstain loss helps slightly at that strict threshold
  (`0.865` vs `0.859` extrapolation, `0.007` vs `0.008` unsafe calls). The
  unsafe-margin loss did not help in this setup and should remain experimental.

Updated recommendation:

- Keep the router-call bridge. The capability case is now stronger, not weaker.
- Promote the compact bridge only behind this safety contract: expanded
  near-boundary calibration rows, a high calibrated call threshold, and
  unsafe-call reporting as a hard gate.
- The next bridge experiment should carry the same contract into a real LM-side
  adapter/router harness and keep the no-safe abstention cases in the training
  mixture. Do not relax the unsafe-call gate just because the router is more
  capable.

## Larger Compact Scaling Run

The next scaling pass lifted the compact generator to four tasks per depth and
ran three seeds on Torch/MPS. This keeps the same depth bins and benchmark
contracts, but doubles the formula families per depth and adds a third seed.

Large symbolic artifact:

```text
artifacts/symbolic-benchmark/symbolic-compact-torch-mps-large-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_benchmark.py \
  --preset compact \
  --run-label symbolic-compact-torch-mps-large-v1 \
  --output-dir artifacts/symbolic-benchmark/symbolic-compact-torch-mps-large-v1 \
  --seeds 3 \
  --tasks-per-depth 4 \
  --steps 240 \
  --tree-learning-rate 0.05 \
  --tree-optimizer torch-autodiff \
  --backend mps \
  --paper-restarts 6 \
  --output table
```

| model | soft val RMSE | hard val RMSE | hard extrap RMSE | exact | near exact | harden | export | compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.3427 | 0.3428 | 0.4480 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| `paper-complex-eml` | 0.0963 | 6.557e-05 | 8.859e-04 | 0.44 | 0.56 | 1.00 | 1.00 | 1.00 |
| `small-mlp` | 0.0109 | 0.0518 | 0.1941 | 0.00 | 0.02 | 0.71 | 0.00 | 1.00 |
| `stable-real-eml` | 0.0161 | 0.0209 | 0.0879 | 0.00 | 0.15 | 1.00 | 1.00 | 1.00 |

Large bridge artifact:

```text
artifacts/symbolic-bridge/symbolic-bridge-compact-large-v1/
```

| model | validation token accuracy | extrap token accuracy | validation RMSE | extrap RMSE | source exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `generic-tree` | 0.062 | 0.062 | 0.3428 | 0.4480 | 0.00 |
| `paper-complex-eml` | 1.000 | 0.969 | 6.557e-05 | 8.859e-04 | 0.44 |
| `small-mlp` | 0.297 | 0.010 | 0.05184 | 0.1941 | 0.00 |
| `stable-real-eml` | 0.599 | 0.099 | 0.02085 | 0.08788 | 0.00 |

Large bridge table:

```text
feature rows: 16128
split counts: train=2304, safety_calibration=4608, validation=4608, extrapolation=4608
safe coverage: train=0.988, safety_calibration=0.714, validation=0.991, extrapolation=0.747
fit_abstain_target_rate: 0.195
```

Large bridge-LM artifacts:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-large-control-t99999-v1/
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-large-c5-t99999-v1/
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-large-control-t80-v1/
```

| condition | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate | abstain recall | contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict threshold | 0.864 | 0.760 | 0.807 | 0.573 | 0.008 | 0.986 | true |
| call/abstain loss + strict threshold | 0.877 | 0.773 | 0.817 | 0.582 | 0.008 | 0.985 | true |
| threshold `0.8` negative control | 0.872 | 0.756 | 0.887 | 0.710 | 0.038 | 0.901 | false |

Scaling verdict:

- The paper-complex arm scales better than the practical surrogate on symbolic
  recovery. On the larger run it is no longer just the hardening leader; it is
  also the hardened extrapolation leader.
- The strict calibrated router contract holds on the larger table even though
  extrapolation has more no-safe cases (`0.253` abstain target rate).
- The call/abstain loss is now the best candidate for the next bridge run:
  it improves extrapolation from `0.864` to `0.877` while preserving the strict
  unsafe-call gate at `0.008`.
- The `0.8` threshold remains a useful negative control: it is capable, but
  unsafe. The bridge should continue with the strict calibrated call boundary.

## Path1 LM-Side Bridge

The LM-side integration replaces the tiny GRU contract model with the repo's
Path1 attention LM backbone. The integration exposes a hidden-state surface on
`Path1HybridLanguageModel`:

```text
Path1 hidden state + frozen compiled-expert features -> router over experts/ABSTAIN
```

The normal token head still trains as the fallback LM path. The symbolic expert
path is a hard call only when the calibrated router crosses the call threshold;
otherwise the Path1 token head decides.

Path1 bridge artifacts:

```text
artifacts/symbolic-bridge-path1/symbolic-bridge-path1-compact-large-c5-t99999-v1/
artifacts/symbolic-bridge-path1/symbolic-bridge-path1-compact-large-c5-t9999-v1/
artifacts/symbolic-bridge-path1/symbolic-bridge-path1-compact-large-c5-t80-v1/
```

Primary command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_path1.py \
  --bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-large-v1/summary.json \
  --run-label symbolic-bridge-path1-compact-large-c5-t9999-v1 \
  --output-dir artifacts/symbolic-bridge-path1/symbolic-bridge-path1-compact-large-c5-t9999-v1 \
  --epochs 900 \
  --learning-rate 0.003 \
  --d-model 96 \
  --total-layers 4 \
  --head-count 4 \
  --ffn-multiplier 2 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 5.0 \
  --router-call-threshold 0.9999 \
  --device auto \
  --output table
```

| condition | threshold | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate | abstain recall | contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strict Path1 | 0.99999 | 0.772 | 0.729 | 0.503 | 0.257 | 0.002 | 0.995 | false |
| calibrated Path1 | 0.9999 | 0.802 | 0.739 | 0.609 | 0.365 | 0.002 | 0.992 | true |
| loose Path1 negative control | 0.8 | 0.845 | 0.763 | 0.782 | 0.655 | 0.062 | 0.814 | false |

Path1 bridge verdict:

- The router-gated compiled expert path works inside the Path1 LM surface. It
  improves extrapolation over token-only, x-task, and frozen side-channel Path1
  baselines while keeping the base LM fallback intact.
- The best safe Path1 setting is less aggressive than the tiny GRU bridge:
  `0.9999` is the right threshold on this run. `0.99999` is too conservative
  and falls just under the capability-gain gate.
- The loose threshold control confirms the safety boundary: it reaches higher
  accuracy but violates the unsafe-call gate (`0.062`).
- This is the first actual LM-side bridge contract in this track. It is not yet
  a broad language-corpus result; it is a Path1-backed symbolic-token bridge
  showing that the compiled expert bank can be called safely from a real repo LM
  backbone.

## Softmax Fusion A/B

The LM bridge now tests two softmax-native expert integrations in parallel with
the existing hard-call router:

```text
logit-fusion: z_final = z_lm + scale * z_expert
prob-mixture: p_final = (1 - expert_mass) * softmax(z_lm) + p_expert
```

Both modes use the same compiled expert token bank and router head. Expert
tokens are injected as sparse vocabulary-aligned distributions; invalid expert
tokens contribute no mass. For fusion modes, final NLL is measured on the fused
distribution, not on deterministic hard-call accuracy.

Compact A/B artifact:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-fusion-ab-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label symbolic-bridge-lm-compact-fusion-ab-v1 \
  --output-dir artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-fusion-ab-v1 \
  --epochs 1800 \
  --learning-rate 0.004 \
  --hidden-units 96 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 5.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device auto \
  --output table
```

| run | mode | extrap final accuracy | extrap final NLL | extrap LM accuracy | router extrap accuracy | expert mass/call | unsafe mass/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-frozen-side-channel` | LM side-channel | 0.602 | 3.508 | 0.602 | n/a | 0.000 | 0.000 |
| `lm-router-hard-call` | hard call | 0.703 | 2.394 | 0.646 | 0.571 | 0.354 | 0.000 |
| `lm-router-logit-fusion` | logit fusion | 0.714 | 1.945 | 0.573 | 0.855 | 0.830 | 0.058 |
| `lm-router-prob-mixture` | probability mixture | 0.865 | 1.117 | 0.223 | 0.953 | 0.806 | 0.034 |

Softmax fusion verdict:

- Probability mixture is the best first softmax-native integration. It improves
  compact extrapolation from `0.602` to `0.865` accuracy and lowers extrapolation
  NLL from `3.508` to `1.117` versus the frozen side-channel LM.
- Logit fusion is alive but less convincing on this surface: it improves NLL and
  accuracy over side-channel, but trails probability mixture and assigns more
  soft mass to unsafe experts.
- The fusion unsafe columns are soft probability mass, not hard unsafe calls.
  Probability mixture therefore needs a calibration follow-up before it replaces
  the strict hard-call router in safety-sensitive runs.
- This supports the user's proposed architecture: an EML expert can contribute
  directly to the final softmax distribution, and loss comparison is meaningful
  when measured on the fused token distribution.

## Decoder-Only Transformer Control

The first softmax-fusion run above used the bridge harness's tiny recurrent LM.
That was useful for the contract, but it was not a pure decoder-only transformer
control. The bridge LM harness now accepts `--backbone transformer`, which swaps
the recurrent backbone for a small causal Transformer encoder used decoder-only:
previous tokens plus causal mask, learned positional embeddings, zero dropout,
and a bias-free feature projection. The bias-free projection matters for the
`lm-token-only` control because its side feature is exactly zero, so no learned
feature bias can leak into the pure token baseline.

Transformer-control artifact:

```text
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-transformer-fusion-ab-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label symbolic-bridge-lm-compact-transformer-fusion-ab-v1 \
  --output-dir artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-transformer-fusion-ab-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 1800 \
  --learning-rate 0.004 \
  --hidden-units 96 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 5.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device auto \
  --output table
```

Run environment:

- `torch==2.11.0`, MPS available and selected by `--device auto`.
- Same compact bridge feature table as the RunPod CUDA symbolic benchmark:
  `artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json`.
- Backbone: 2-layer, 4-head causal transformer, `hidden_units=96`,
  dropout `0.0`.

| run | mode | extrap final accuracy | extrap final NLL | extrap LM accuracy | router extrap accuracy | expert mass/call | unsafe mass/call |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | pure transformer control | 0.624 | 3.518 | 0.624 | n/a | 0.000 | 0.000 |
| `lm-frozen-side-channel` | transformer side-channel | 0.574 | 4.304 | 0.574 | n/a | 0.000 | 0.000 |
| `lm-router-hard-call` | hard call | 0.866 | 1.087 | 0.589 | 0.930 | 0.732 | 0.009 |
| `lm-router-logit-fusion` | logit fusion | 0.691 | 2.302 | 0.551 | 0.904 | 0.801 | 0.033 |
| `lm-router-prob-mixture` | probability mixture | 0.866 | 1.567 | 0.135 | 0.938 | 0.793 | 0.025 |

Transformer-control verdict:

- The EML hybrid beats the pure decoder-only transformer control on the same
  symbolic-token task. The hard-call router improves extrapolation accuracy from
  `0.624` to `0.866`, and the probability-mixture softmax path improves it from
  `0.624` to `0.866`.
- The loss comparison should use the softmax-native rows, not hard-call's
  accuracy-derived pseudo-NLL. On that apples-to-apples basis, probability
  mixture lowers extrapolation NLL from the pure transformer's `3.518` to
  `1.567`.
- The hard-call router now satisfies the bridge contract on this run:
  capability gain is confirmed, unsafe hard calls are below the `0.01` gate
  (`0.009`), abstain recall is `0.961`, and the artifact reports
  `contract_confirmed=True`.
- The frozen side-channel alone does not explain the gain. With the same
  transformer backbone it underperforms the token-only control (`0.574` versus
  `0.624` extrapolation accuracy), while explicit expert routing/fusion produces
  the jump.
- This is a real positive control for the bridge idea, but still only on the
  symbolic-token benchmark. It does not yet prove broad language modeling
  improvement on natural text; it shows that, under the bridge contract, a
  decoder-only transformer can use the compiled EML expert bank to improve
  extrapolative symbolic prediction.

## Bridge Corpus V1

The bridge now has a three-rung corpus ladder:

1. `pure-language`: synthetic prose only. There are no safe expert calls.
2. `language-math`: prose-wrapped formula examples with a single math answer
   token per example. Expert calls should happen on `math_answer` tokens and
   abstain on prose/context tokens.
3. `math-only`: the existing symbolic-token bridge corpus, annotated as
   `math_only`.

Corpus artifacts:

```text
artifacts/bridge-corpus-v1/bridge-corpus-v1-pure-language/
artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math/
artifacts/bridge-corpus-v1/bridge-corpus-v1-math-only/
```

LM artifacts:

```text
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-pure-language-transformer/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-language-math-transformer/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-math-only-transformer/
```

All three LM runs used the same 2-layer causal transformer control surface:
`hidden_units=64`, `transformer_heads=4`, `epochs=900`, `learning_rate=0.004`,
MPS selected by `--device auto`.

Corpus sizes:

| corpus | rows | token bins | extrap roles | extrap safe expert coverage |
| --- | ---: | ---: | --- | ---: |
| pure language | 5,760 | 37 | prose: 1,440 | 0.000 |
| language + math | 11,968 | 109 | prose: 2,240; math_context: 960; math_answer: 320 | overall 0.075; math_answer 0.825 |
| math only | 5,376 | 32 | math_only: 1,536 | 0.783 |

Extrapolation headline metrics:

| corpus | run | extrap acc | extrap NLL | expert call | unsafe call/mass |
| --- | --- | ---: | ---: | ---: | ---: |
| pure language | `lm-token-only` | 0.461 | 3.773 | 0.000 | 0.000 |
| pure language | `lm-router-prob-mixture` | 0.460 | 4.600 | 0.000 | 0.000 |
| language + math | `lm-token-only` | 0.808 | 0.998 | 0.000 | 0.000 |
| language + math | `lm-router-logit-fusion` | 0.942 | 0.578 | 0.080 | 0.008 |
| language + math | `lm-router-prob-mixture` | 0.947 | 0.617 | 0.079 | 0.007 |
| math only | `lm-token-only` | 0.615 | 3.208 | 0.000 | 0.000 |
| math only | `lm-router-hard-call` | 0.801 | 1.610 | 0.526 | 0.001 |
| math only | `lm-router-prob-mixture` | 0.869 | 1.457 | 0.815 | 0.043 |

Language+math answer-span metrics:

| run | math-answer acc | math-answer NLL | answer expert call | answer unsafe mass |
| --- | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.394 | 4.325 | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.559 | 2.741 | 0.000 | 0.000 |
| `lm-router-logit-fusion` | 0.769 | 1.971 | 0.875 | 0.090 |
| `lm-router-prob-mixture` | 0.863 | 1.638 | 0.866 | 0.075 |

Bridge Corpus V1 verdict:

- The pure-language rung is a useful negative control. The router abstains
  everywhere and makes zero unsafe calls. The hybrid does not invent a fake
  language gain; token-only remains the best NLL.
- The language+math rung is the first actual bridge-shaped win. Probability
  mixture improves overall extrapolation accuracy from `0.808` to `0.947`; on
  math-answer tokens it improves from `0.394` to `0.863`, while prose stays at
  `1.000` accuracy and near-zero NLL.
- The math-only rung reproduces the symbolic-token story under the shared
  900-epoch transformer setting. Probability mixture reaches `0.869`
  extrapolation accuracy versus `0.615` for token-only.
- The current failure mode is calibration on soft fusion. Language+math
  probability mixture assigns `0.075` unsafe expert mass on answer tokens, and
  math-only assigns `0.043` unsafe mass overall. Hard-call is much safer on
  math-only (`0.001`) but less accurate than probability mixture.
- The next bridge step should not be a larger model first. It should tighten
  soft-fusion calibration on answer spans: unsafe-mass penalty by role, explicit
  prose/context abstention loss, and a loss report that gates on math-answer NLL
  plus unsafe answer mass.

## Role-Calibrated Soft Fusion

The bridge LM now supports two role-aware router penalties:

- `answer_unsafe_loss_weight`: penalizes unsafe expert probability mass on
  `math_answer` and `math_only` positions.
- `non_answer_abstain_loss_weight`: pushes the router to abstain on `prose` and
  `math_context` positions.

These losses default to zero, so previous runs remain comparable. The first
calibrated bridge-corpus setting used:

```text
answer_unsafe_loss_weight=5.0
non_answer_abstain_loss_weight=3.0
router_loss_weight=10.0
call_abstain_loss_weight=5.0
```

Calibrated artifacts:

```text
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-pure-language-transformer-calibrated-v1/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-language-math-transformer-calibrated-v2/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-math-only-transformer-calibrated-v1/
```

Calibration results:

| corpus | run | base acc | calibrated acc | base NLL | calibrated NLL | base unsafe | calibrated unsafe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pure language | `lm-router-prob-mixture` | 0.460 | 0.455 | 4.600 | 4.573 | 0.000 | 0.000 |
| language + math | `lm-router-prob-mixture` overall | 0.947 | 0.942 | 0.617 | 0.657 | 0.007 | 0.005 |
| language + math | `lm-router-prob-mixture` math_answer | 0.863 | 0.872 | 1.638 | 1.639 | 0.075 | 0.058 |
| language + math | `lm-router-logit-fusion` math_answer | 0.769 | 0.756 | 1.971 | 1.881 | 0.090 | 0.029 |
| math only | `lm-router-prob-mixture` | 0.869 | 0.874 | 1.457 | 1.441 | 0.043 | 0.027 |
| math only | `lm-router-hard-call` | 0.801 | 0.829 | 1.610 | 1.380 | 0.001 | 0.002 |

Role-calibration verdict:

- The role-aware objective works in the direction intended. It reduces unsafe
  expert mass on answer/math positions without introducing prose calls.
- On `language + math`, probability mixture keeps the answer-span capability
  gain and slightly improves answer accuracy (`0.863 -> 0.872`) while reducing
  answer unsafe mass (`0.075 -> 0.058`). Its answer NLL is essentially flat.
- On `math only`, probability mixture improves on all three key axes:
  extrapolation accuracy (`0.869 -> 0.874`), extrapolation NLL
  (`1.457 -> 1.441`), and unsafe mass (`0.043 -> 0.027`).
- Logit fusion is more safety-responsive than probability mixture on
  `language + math` answer tokens (`0.090 -> 0.029` unsafe mass), but loses
  answer accuracy. It is a useful calibration control, not the current best
  bridge candidate.
- The remaining issue is not abstention on prose/context; that is already clean.
  The remaining issue is answer-token expert selection quality. The next run
  should add an answer-role objective that prefers safe expert mass over merely
  reducing unsafe mass, so the router does not solve safety by becoming timid.

## Expert-Bank Ablation And Shuffle Controls

The next null test keeps the calibrated `language + math` corpus and transformer
bridge recipe fixed, then changes only the available expert bank:

- single-expert ablations: `paper-complex-eml`, `stable-real-eml`,
  `generic-tree`, `small-mlp`
- grouped ablations: symbolic trees only, non-EML controls only
- shuffled-all control: same four experts and token distribution, but expert
  payloads are shuffled within split and role before safety metadata is
  recomputed

This tests whether the bridge gain follows the paper-aligned expert payloads, or
whether any extra predictor/shuffled token prior can reproduce it.

Ablation corpus artifacts:

```text
artifacts/bridge-corpus-v1-ablation/
```

Representative corpus-build commands:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind expert-ablation \
  --source-corpus-summary artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math/summary.json \
  --experts paper-complex-eml \
  --run-label language-math-paper-complex-only \
  --output-dir artifacts/bridge-corpus-v1-ablation/language-math-paper-complex-only

python scripts/symbolic_bridge_corpus.py \
  --corpus-kind expert-shuffle \
  --source-corpus-summary artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math/summary.json \
  --run-label language-math-shuffled-all \
  --output-dir artifacts/bridge-corpus-v1-ablation/language-math-shuffled-all \
  --shuffle-seed 20260418
```

Ablation LM artifacts:

```text
artifacts/bridge-corpus-v1-ablation-lm/
```

All LM ablations used the same calibrated transformer recipe as
`bridge-corpus-v1-language-math-transformer-calibrated-v2`:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary <ablation-summary.json> \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 900 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 5.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 3.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

Language+math extrapolation, math-answer role only:

| condition | experts | safe answer coverage | token-only acc | token-only NLL | prob-mixture acc | prob-mixture NLL | logit-fusion acc | logit-fusion NLL | prob unsafe mass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all-four calibrated baseline | generic, paper-complex, small-mlp, stable-real | 0.825 | 0.375 | 4.321 | 0.872 | 1.639 | 0.756 | 1.881 | 0.058 |
| paper-complex only | paper-complex | 0.812 | 0.375 | 4.317 | 0.881 | 1.515 | 0.769 | 1.660 | 0.082 |
| stable-real only | stable-real | 0.394 | 0.375 | 4.361 | 0.444 | 4.323 | 0.494 | 3.159 | 0.040 |
| generic-tree only | generic-tree | 0.016 | 0.394 | 4.222 | 0.503 | 4.181 | 0.463 | 3.519 | 0.012 |
| small-mlp only | small-mlp | 0.069 | 0.394 | 4.298 | 0.447 | 3.980 | 0.497 | 3.985 | 0.010 |
| symbolic trees | generic, paper-complex, stable-real | 0.825 | 0.394 | 4.294 | 0.753 | 1.791 | 0.819 | 1.477 | 0.019 |
| non-EML control | generic, small-mlp | 0.084 | 0.375 | 4.297 | 0.450 | 4.795 | 0.422 | 3.666 | 0.015 |
| shuffled all | generic, paper-complex, small-mlp, stable-real | 0.206 | 0.375 | 4.287 | 0.362 | 3.883 | 0.394 | 3.865 | 0.039 |

Null-test verdict:

- The effect follows `paper-complex-eml`. Paper-complex-only is at least as good
  as the all-four calibrated bank on math-answer accuracy (`0.881` vs `0.872`)
  and better on answer NLL (`1.515` vs `1.639`).
- The practical `stable-real-eml` surrogate preserves some structure but not the
  bridge result. Its answer accuracy is only `0.444`, close to the token-only
  baseline, and its NLL remains high.
- The non-EML controls do not explain the gain. `generic-tree`, `small-mlp`, and
  their combined control all trail the x/task transformer and do not produce a
  useful probability-mixture answer head.
- The shuffled-all control collapses. Keeping the same expert ids and token
  distribution while breaking per-example alignment drops probability-mixture
  answer accuracy below token-only (`0.362` vs `0.375`). That is strong evidence
  against a token-prior or router-leak explanation.
- The remaining risk is calibration, not capability. Paper-complex-only has the
  best answer NLL and accuracy, but assigns more unsafe answer mass (`0.082`)
  than the all-four calibrated bank (`0.058`). The next bridge should optimize
  safe expert mass on answer tokens directly rather than only penalizing unsafe
  mass.

## Null-Hypothesis Gate Ledger

The bridge result is now tracked through an explicit gate ledger:

```text
docs/specs/symbolic-bridge-null-gates.md
```

Each gate in that ledger records a fixed resolution block:

```text
Resolution needed:
Promotion condition:
Current blocker:
Next action:
```

Current status:

| gate | status | short verdict |
| --- | --- | --- |
| 1. Expert-bank ablation and shuffle controls | passed | The effect follows aligned `paper-complex-eml`; non-EML and shuffled controls do not reproduce it. |
| 2. Held-out formula/language templates | mixed, safety failed | Probability mixture keeps math-answer capability on held-out templates, but unsafe soft mass is too high. |
| 3. Target/random-label and wrong-expert controls | pending | Not run yet. |
| 4. Seed/template variance | pending | Not run yet. |
| 5. More natural mixed corpus | pending | Not run yet. |

Gate 2 artifact bundle:

```text
artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math-heldout-templates/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-language-math-heldout-templates-transformer-calibrated-v1/
```

Gate 2 held out formula families `d1_exp_soft`, `d2_reciprocal_shift`,
`d3_ratio_product`, and `d4_nested_ratio_exp`; held out all validation and
extrapolation language wrappers; and varied math-answer positions across
indices `1`, `6`, `8`, and `9`.

Language+math held-out-template extrapolation, math-answer role:

| run | math-answer acc | math-answer NLL | expert mass/call | unsafe mass/call |
| --- | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.000 | 11.094 | 0.000 | 0.000 |
| `lm-x-task` | 0.013 | 8.926 | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.031 | 8.410 | 0.000 | 0.000 |
| `lm-router-hard-call` | 0.006 | 9.350 | 0.000 | 0.000 |
| `lm-router-logit-fusion` | 0.163 | 5.972 | 0.747 | 0.279 |
| `lm-router-prob-mixture` | 0.625 | 6.703 | 1.000 | 0.375 |

Gate 2 verdict:

- Capability partially survives. The probability-mixture hybrid recovers many
  held-out-template math answers (`0.625`) while the token-only transformer is
  at `0.000`.
- The safety contract does not survive. Unsafe answer mass rises to `0.375`,
  and the hard-call route stays safe only by abstaining away the gain.
- This blocks promotion to broader LM runs until the answer-token objective
  learns safe expert mass under held-out-style template/formula diversity.
