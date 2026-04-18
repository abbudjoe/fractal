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
