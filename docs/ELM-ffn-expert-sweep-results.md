# EML-Inspired FFN Expert Sweep Results

Date: 2026-04-15

This note summarizes the local MPS sweep for treating EML as a small FFN-side expert, not as a sequence mixer or all-layer transformer replacement. The goal was to determine whether the earlier single-layer gated EML signal is real, whether it is EML-specific, and whether the best expert survives as a seam-adjacent add-on for P20 or GDN backbones.

## Implementation

The sweep added a reusable expert surface to the existing Path 1 language-model harness.

- Added dense gated EML, routed EML, tiny MLP, tiny GLU, and generic binary-tree gated expert profiles.
- Kept experts as selected-layer FFN-side branches attached to existing attention FFNs.
- Added diagnostics for gate magnitude, selector entropy, routing utilization, and activation norms.
- Added MPS runtime support for the Path 1 CLI and sweep runner.
- Fixed macOS peak RSS accounting so local memory summaries are reported in bytes correctly.
- Added tests for profile plumbing, selected-layer behavior, controls, routed EML, and MPS validation.

Primary entrypoint:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/eml_ffn_expert_sweep.py
```

## Validation

```bash
/Users/joseph/fractal/.venv/bin/python -m unittest python.tests.test_models python.tests.test_specs
```

Result:

```text
Ran 83 tests in 3.135s
OK
```

Syntax check:

```bash
/Users/joseph/fractal/.venv/bin/python -m py_compile \
  python/models/common.py python/models/path1.py python/specs/path1.py \
  python/specs/common.py python/runtime/seeding.py python/runtime/train_eval.py \
  python/runners/path1_cli.py scripts/eml_ffn_expert_sweep.py \
  python/tests/test_models.py python/tests/test_specs.py
```

## Phase 1: Attention-Only Seam Search

Command:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/eml_ffn_expert_sweep.py \
  --backend mps --dtype fp32 \
  --seeds 42,43,44 \
  --layers all \
  --steps 16 \
  --eval-batches 4 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --run-id phase1-mps-s42-44-steps16
```

Artifacts:

- `artifacts/eml-ffn-expert-sweep/phase1-mps-s42-44-steps16/summary.md`
- `artifacts/eml-ffn-expert-sweep/phase1-mps-s42-44-steps16/summary.json`
- `artifacts/eml-ffn-expert-sweep/phase1-mps-s42-44-steps16/ledger.jsonl`

Top aggregate results:

| Variant | Layer | Seeds | Mean Loss | Delta vs Baseline | tok/s | Params |
|---|---:|---:|---:|---:|---:|---:|
| baseline | all | 3 | 2.9819 | +0.0000 | 16521.94 | 1.65M |
| dense EML | 1 | 3 | 2.9332 | -0.0487 | 13777.59 | 1.73M |
| generic tree | 1 | 3 | 2.9333 | -0.0485 | 12614.29 | 1.73M |
| dense EML | 5 | 3 | 2.9377 | -0.0441 | 13620.17 | 1.73M |
| generic tree | 5 | 3 | 2.9383 | -0.0435 | 13916.83 | 1.73M |
| tiny MLP | 6 | 3 | 2.9399 | -0.0419 | 14907.19 | 1.72M |
| dense EML | 4 | 3 | 2.9434 | -0.0384 | 13298.48 | 1.73M |

Phase 1 read:

- The single-expert seam is real across three seeds in this local MPS budget.
- The old layer-4 signal remains positive, but it was not the best seam here.
- Layers 1 and 5 were stronger than layer 4 in this run.
- EML-specific structure did not separate from the matched generic tree control.

## Phase 2: Local Expansion

Command:

```bash
/Users/joseph/fractal/.venv/bin/python scripts/eml_ffn_expert_sweep.py \
  --phase phase2 \
  --backend mps --dtype fp32 \
  --seeds 42,43,44 \
  --layers 1,5 \
  --steps 16 \
  --eval-batches 4 \
  --warmup-train-steps 1 \
  --warmup-eval-batches 1 \
  --run-id phase2-mps-l1-l5-s42-44-steps16
```

Artifacts:

- `artifacts/eml-ffn-expert-sweep/phase2-mps-l1-l5-s42-44-steps16/summary.md`
- `artifacts/eml-ffn-expert-sweep/phase2-mps-l1-l5-s42-44-steps16/summary.json`
- `artifacts/eml-ffn-expert-sweep/phase2-mps-l1-l5-s42-44-steps16/ledger.jsonl`

Pareto-relevant aggregate results:

| Variant | Layer | Shape | Seeds | Mean Loss | Delta vs Baseline | tok/s | Peak MB | Params |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | all | standard | 3 | 2.9819 | +0.0000 | 17379.69 | 454.71 | 1.65M |
| dense EML | 1 | slots 4, depth 2 | 3 | 2.9332 | -0.0487 | 13744.39 | 471.84 | 1.73M |
| generic tree | 1 | slots 4, depth 2 | 3 | 2.9333 | -0.0485 | 14579.28 | 472.73 | 1.73M |
| dense EML | 5 | slots 4, depth 2 | 3 | 2.9377 | -0.0441 | 13648.51 | 499.09 | 1.73M |
| generic tree | 5 | slots 4, depth 2 | 3 | 2.9383 | -0.0435 | 14276.61 | 499.14 | 1.73M |
| generic tree | 5 | slots 8, depth 2 | 3 | 2.9422 | -0.0397 | 14579.71 | 499.93 | 1.74M |
| routed EML | 1 | slots 4, depth 2, route 0.10 | 3 | 2.9587 | -0.0232 | 13056.96 | 476.53 | 1.73M |
| tiny GLU | 5 | control | 3 | 2.9607 | -0.0211 | 15675.97 | 498.96 | 1.75M |
| tiny MLP | 5 | control | 3 | 2.9667 | -0.0152 | 15704.03 | 498.86 | 1.72M |

Phase 2 read:

- The best shape stayed small: `slot_count=4`, `tree_depth=2`.
- Larger slot counts and deeper trees did not improve loss in this bounded budget.
- Routed EML was worse than dense EML and generic tree at matched seams.
- The generic binary-tree control effectively tied dense EML and was usually faster.

Representative diagnostics from Phase 2 seed 42:

| Variant | Layer | Mean Gate/Mix | Selector Entropy | Routed Utilization | Mean Expert Norm | Mean Output Norm |
|---|---:|---:|---:|---:|---:|---:|
| dense EML | 1 | 0.0999 | 1.9457 | n/a | 11.84 | 3.51 |
| generic tree | 1 | 0.0999 | 1.9457 | n/a | 11.49 | 3.54 |
| routed EML r0.10 | 1 | 0.0998 | 1.9457 | 0.1016 | 0.66 | 3.49 |

The dense tree branches are doing measurable work. However, the EML-specific node math did not produce a clear advantage over a generic real-valued binary-tree node.

## Phase 3: P20 / GDN Survival Check

Phase 3 was intentionally short because MPS recurrent lanes are slow. It should be read as a gate, not as a final benchmark.

Settings:

- Seeds: `42,43,44`
- Steps: `4`
- Eval batches: `2`
- Batch size: `2`
- Sequence length: `64`
- P20 schedule: `AAAAPPAAAAA`
- GDN schedule: `RRRRRARRRRRS`
- Expert shape: slots 4, depth 2

Artifact:

- `artifacts/eml-ffn-expert-sweep/phase3-mps-survival-s42-44-steps4-rerun/ledger.jsonl`

Aggregate results:

| Backbone | Variant | Expert Layer | Seeds | Mean Loss | Delta vs Backbone | tok/s |
|---|---|---:|---:|---:|---:|---:|
| P20 | baseline | none | 3 | 3.8455 | +0.0000 | 1262.44 |
| P20 | dense EML | 1 | 3 | 4.0297 | +0.1843 | 1204.19 |
| P20 | generic tree | 1 | 3 | 4.0311 | +0.1856 | 1177.17 |
| GDN | baseline | none | 3 | 4.1278 | +0.0000 | 301.33 |
| GDN | dense EML | 5 | 3 | 3.9591 | -0.1687 | 293.77 |
| GDN | generic tree | 5 | 3 | 3.9591 | -0.1688 | 277.63 |

Phase 3 read:

- The expert did not survive the P20 seam in this short MPS check.
- The expert did improve the GDN seam check, but dense EML and generic tree were tied.
- This supports “tree-style FFN-side expert near a GDN attention seam” more than it supports “EML-specific math.”

## Answers To The Track Questions

1. Is the earlier positive layer signal real across seeds?

Yes, but not uniquely at the old layer-4 seam. A single selected-layer tree expert improved the attention-only baseline across three seeds, with the strongest local MPS results at layers 1 and 5.

2. Is the gain EML-specific?

Not yet. The matched generic binary-tree expert tied dense EML within noise and was often faster. The evidence supports a structured tree expert branch, not the current EML-specific node operator.

3. Dense or routed?

Dense won this bounded search. Routed EML preserved the machinery and reported plausible utilization, but it lost too much quality at route fractions 0.10, 0.25, and 0.50.

4. Does the winner survive on P20 or GDN?

It survived on GDN as a seam-adjacent FFN-side expert in a short run. It did not survive on P20. Since GDN and generic tree tied, any GDN follow-up should include both controls.

## Pareto Frontier

For the attention-only Phase 2 surface:

| Candidate | Why It Is On The Frontier |
|---|---|
| dense EML, layer 1, slots 4, depth 2 | Best loss: 2.9332 |
| generic tree, layer 1, slots 4, depth 2 | Same quality within 0.0001, faster than dense EML |
| generic tree, layer 5, slots 4, depth 2 | Strong second seam, faster than dense EML layer 5 |
| tiny GLU, layer 5 | Faster small-expert control with smaller quality gain |
| baseline | Speed anchor |

## Interpretation

The strongest result is not “EML wins.” The stronger and cleaner claim is:

> A tiny selected-layer structured tree expert can improve this small attention-only LM surface, but the current EML-inspired node math has not beaten a matched generic tree control.

This keeps the EML-inspired direction alive, but only weakly as an EML-specific hypothesis. The more robust hypothesis is that a constrained tree-shaped FFN-side expert is a useful small-capacity branch at particular seams.

## Recommended Next Experiment

Run one longer GPU confirmation with only the Pareto candidates:

- baseline
- dense EML, layer 1, slots 4, depth 2
- generic tree, layer 1, slots 4, depth 2
- generic tree, layer 5, slots 4, depth 2
- tiny GLU, layer 5

Use more train steps and include post-quant checks if the target surface is Parameter Golf. Do not scale routed EML or all-layer EML until the EML-specific operator separates from generic-tree controls.
