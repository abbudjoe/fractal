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
- `--tree-optimizer torch-autodiff` uses PyTorch autograd and can run on MPS via
  `--backend mps`; use `uv run --python 3.12 --with torch --with numpy ...`
  because the repo's default Python is 3.14
- the previous deterministic SPSA optimizer is still available through
  `--tree-optimizer spsa` for before/after comparisons
- `small-mlp` uses an analytic tanh MLP gradient with Adam-style moments
- hardening snaps selectors by argmax, sharpens selectors after autodiff runs,
  and snaps near-simple scalar values
- compiled execution uses a restricted Python lambda over safe helper functions

The Torch/MPS path reuses the same dataset, hardening, export, and report
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
artifacts/symbolic-bridge/symbolic-bridge-compact-both-v5/
```

| preset | train rows | safety rows | validation rows | extrap rows | safety safe-expert coverage | extrap safe-expert coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tier-0 | 640 | 640 | 1280 | 1280 | 1.000 | 1.000 |
| compact | 768 | 768 | 1536 | 1536 | 0.785 | 0.783 |

The compact v5 table uses a deliberately pessimistic safety-calibration sample:
when no-safe candidates exist, it targets an abstain-heavy safety split and then
prefers out-of-training-range safe examples for the remaining rows. This makes
the fit split closer to extrapolation: `fit_abstain_target_rate=0.107` versus
`0.217` on extrapolation. The earlier compact v4 table had only
`fit_abstain_target_rate=0.063`, which was too mild for the extrapolation
safety boundary.

Safety-calibrated bridge commands:

```bash
python scripts/symbolic_bridge.py \
  --symbolic-summary artifacts/symbolic-benchmark/symbolic-tier0-torch-mps-sparse-loglift-v2/summary.json \
  --run-label symbolic-bridge-tier0-both-v4 \
  --output-dir artifacts/symbolic-bridge/symbolic-bridge-tier0-both-v4 \
  --token-bins 32

python scripts/symbolic_bridge.py \
  --symbolic-summary artifacts/symbolic-benchmark/symbolic-compact-torch-mps-sparse-loglift-v2/summary.json \
  --run-label symbolic-bridge-compact-both-v5 \
  --output-dir artifacts/symbolic-bridge/symbolic-bridge-compact-both-v5 \
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
artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-safety-v5-threshold80-v1/
```

| run | threshold | extrap final accuracy | extrap LM accuracy | router extrap accuracy | expert call rate | unsafe call rate | abstain recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | n/a | 0.656 | 0.656 | n/a | 0.000 | 0.000 | n/a |
| `lm-x-task` | n/a | 0.680 | 0.680 | n/a | 0.000 | 0.000 | n/a |
| `lm-frozen-side-channel` | n/a | 0.712 | 0.712 | n/a | 0.000 | 0.000 | n/a |
| `lm-router-hard-call` | 0.8 | 0.831 | 0.680 | 0.889 | 0.711 | 0.020 | 0.913 |

Safety-calibrated LM command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-both-v5/summary.json \
  --run-label symbolic-bridge-lm-compact-safety-v5-threshold80-v1 \
  --output-dir artifacts/symbolic-bridge-lm/symbolic-bridge-lm-compact-safety-v5-threshold80-v1 \
  --epochs 1800 \
  --learning-rate 0.004 \
  --hidden-units 96 \
  --router-loss-weight 10.0 \
  --router-call-threshold 0.8 \
  --device auto \
  --output table
```

Additional compact calibration sweeps did not improve the safety frontier:

| artifact | router loss weight | abstain class weight | unsafe loss weight | threshold | extrap final accuracy | unsafe call rate | abstain recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `symbolic-bridge-lm-compact-safety-rw30-v2` | 30.0 | 1.0 | 0.0 | 0.0 | 0.855 | 0.124 | 0.452 |
| `symbolic-bridge-lm-compact-safety-threshold999-v1` | 10.0 | 1.0 | 0.0 | 0.999 | 0.840 | 0.049 | 0.775 |
| `symbolic-bridge-lm-compact-calibrated-u1-a2-t80-v1` | 10.0 | 2.0 | 1.0 | 0.8 | 0.835 | 0.066 | 0.707 |
| `symbolic-bridge-lm-compact-calibrated-u10-a4-t80-v1` | 10.0 | 4.0 | 10.0 | 0.8 | 0.859 | 0.088 | 0.611 |

Safety-calibration verdict:

- The data fix worked: compact fit now contains abstain examples
  (`fit_abstain_target_rate=0.107`) and extrapolation no-safe cases are no
  longer completely unseen.
- The contract is now clearly capability-improving on compact: routed final
  accuracy reaches `0.831`, above the frozen-side-channel LM (`0.712`) and
  above the best global expert under the same safe-call accounting (`0.772`).
  This happens because the LM fallback can answer some no-safe cases while
  expert calls handle recovered formulas.
- Safe abstention is much better but still not fully confirmed. The best compact
  run cuts unsafe calls from `0.228` before calibration to `0.020`, and raises
  abstain recall to `0.913`, but it still fails the strict `<=0.01`
  unsafe-call gate.
- The attempted router-loss calibration terms were not enough. Increasing router
  loss weight, adding abstain class weight, or penalizing unsafe expert
  probability all either worsened unsafe calls or traded away too much
  abstention precision. The main useful safety gain came from the typed
  safety-calibration data, not from the auxiliary loss.

Updated recommendation:

- Keep the router-call bridge. The capability case is now stronger, not weaker.
- Do not yet promote the compact bridge to broad LM tasks. Tier-0 is fully
  confirmed, but compact still misses the strict unsafe-call gate.
- The next bridge step should make the call boundary explicit rather than
  asking a single router head to infer it indirectly: a separate safety head,
  task-conditioned thresholds, or calibrated per-expert call margins are the
  most direct candidates.
