# Symbolic Bridge Null-Hypothesis Gates

This ledger tracks the staged attempts to disprove the EML bridge result before
promoting it toward broader LM work. Each gate has a local artifact trail plus a
short verdict. The high-level summary is updated as gates complete; detailed
records stay below it.

## High-Level Status

| gate | question | status | current answer |
| --- | --- | --- | --- |
| 1. Expert-bank ablation and shuffle controls | Does the gain follow aligned EML expert predictions, or can any/shuffled expert bank fake it? | passed after repaired audit | The repaired gain follows `paper-complex-eml` / symbolic-tree experts; non-EML and shuffled controls do not reproduce it. |
| 2. Held-out formula/language templates | Does the bridge survive unseen formula families, unseen wrappers, and varied answer positions? | repaired after audit | Multi-split calibration plus answer-token call/abstain loss keeps most capability while reducing unsafe mass below the role-aware gate. |
| 3. Target/random-label and wrong-expert controls | Does the harness leak target identity through labels, routing, or feature construction? | passed after repaired audit | Broken labels and wrong expert pairings still collapse the repaired hybrid gain. |
| 4. Seed/template variance | Is the positive result stable across seeds and template draws? | repaired after audit | Multi-split calibration turns the unstable variance result into stable math-answer gains with low unsafe mass. |
| 5. More natural mixed corpus | Does the contract hold beyond the synthetic grammar? | partial pass | The repaired probability-mixture bridge still wins strongly on held-out natural-wrapper math-answer tokens, but the full mixed-corpus contract is not confirmed because prose accuracy/NLL does not improve over controls. |

Current recommendation: do not promote to broad LM claims yet. The repaired
recipe has now survived the bridge-critical math-answer slice through Gate 5,
including held-out formulas, held-out wrappers, varied answer positions, and
naturalistic distractor prose. However, Gate 5 also shows that this is still a
role-local improvement: the hybrid improves answer tokens but does not improve
the surrounding prose distribution. Frozen-teacher KL helps prose NLL, and a
prose-only KL mask reduces the math-context damage, but neither is a finished
general-decoder repair. The next step should keep the symbolic bridge isolated
behind a calibrated answer/action interface, while testing only narrow
composition repairs before making broad LM claims.

## Metric Glossary

The bridge tables use compact metric names. Read them this way:

| shorthand | meaning | direction |
| --- | --- | --- |
| `acc` | Accuracy: fraction of evaluated tokens whose final predicted token equals the target token. | Higher is better. |
| `NLL` | Negative log likelihood assigned to the correct token. This measures calibrated probability, not just argmax correctness. | Lower is better. |
| `token-only` | Pure decoder-only transformer path with no symbolic expert signal. | Control baseline. |
| `side-channel` | Transformer sees frozen expert features as input, but expert predictions are not fused into the final distribution. | Control baseline. |
| `hard-call` | Router either selects one compiled expert token or abstains back to the LM. | Safety-oriented control. |
| `logit-fusion` | Expert token mass is added as a logit bonus before softmax. | Hybrid variant. |
| `prob-mixture` | Final probability is a mixture of the LM softmax and expert token probabilities. | Main positive bridge variant. |
| `expert mass` / `call mass` | Average probability mass assigned to valid expert predictions. In hard-call rows this is a call rate; in soft-fusion rows this is soft probability mass. | Needs balance: too low means timid, too high can be unsafe. |
| `unsafe mass` / `unsafe call` | Average probability mass or hard calls assigned to experts whose token does not match the target. | Lower is better. |
| `safe answer coverage` | Fraction of math-answer rows where at least one expert has the correct token available. | Upper bound context. |
| `math-answer role` | Only the token positions containing the numeric/formula answer, excluding prose and context tokens. | The bridge-critical slice. |
| `role-aware contract` | Pass/fail for the math-answer probability-mixture gate: useful gain over token-only/side-channel controls with unsafe answer mass at or below the safety threshold used here. | `true` is better. |

## Resolution Block Contract

Each gate records the same operational block:

```text
Resolution needed:
Promotion condition:
Current blocker:
Next action:
```

`Resolution needed` is the scientific or engineering uncertainty that must be
closed before the gate can be considered done. `Promotion condition` is the
minimum result needed to move the bridge forward. `Current blocker` is the active
failure mode, if any. `Next action` is the next bounded experiment or code change
that should address the blocker.

## Gate 1: Expert-Bank Ablation And Shuffle Controls

Question:

> Does the bridge gain actually come from aligned EML expert predictions, or can
> a generic/shuffled extra predictor explain it?

Artifacts:

```text
artifacts/bridge-corpus-v1-ablation/
artifacts/bridge-corpus-v1-ablation-lm/
```

Caption: Gate 1 compares expert-bank variants on only the extrapolation
`math_answer` tokens. The purpose is to see whether the probability-mixture gain
comes from aligned `paper-complex-eml` experts or from generic/shuffled expert
capacity.

| condition | safe answer coverage | token-only acc | prob-mixture acc | prob-mixture NLL | unsafe mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-four calibrated baseline | 0.825 | 0.375 | 0.872 | 1.639 | 0.058 |
| paper-complex only | 0.812 | 0.375 | 0.881 | 1.515 | 0.082 |
| stable-real only | 0.394 | 0.375 | 0.444 | 4.323 | 0.040 |
| generic-tree only | 0.016 | 0.394 | 0.503 | 4.181 | 0.012 |
| small-mlp only | 0.069 | 0.394 | 0.447 | 3.980 | 0.010 |
| symbolic trees | 0.825 | 0.394 | 0.753 | 1.791 | 0.019 |
| non-EML control | 0.084 | 0.375 | 0.450 | 4.795 | 0.015 |
| shuffled all | 0.206 | 0.375 | 0.362 | 3.883 | 0.039 |

Verdict:

- `paper-complex-eml` carries the effect. Paper-complex-only improves answer
  accuracy and NLL over the all-four bank.
- The stable-real surrogate preserves some structure, but does not reproduce the
  bridge result.
- Non-EML controls and shuffled expert payloads do not explain the gain.
- Remaining risk after this gate: calibration. Paper-complex-only is strong, but
  has higher unsafe answer mass than the all-four calibrated bank.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Decide whether the bridge gain is genuinely tied to aligned paper-complex expert predictions rather than generic expert capacity, expert-token priors, or shuffled payload artifacts. |
| Promotion condition | Aligned `paper-complex-eml` or EML-family experts outperform token-only and non-EML controls on math-answer accuracy/NLL, while shuffled experts fail to reproduce the gain. |
| Current blocker | Gate 1 itself is not blocked; it passed. Residual issue is that paper-complex-only has higher unsafe answer mass (`0.082`) than the all-four calibrated bank (`0.058`). |
| Next action | Carry the paper-complex signal forward, but require later gates to improve answer-token safe-mass calibration before any broad LM promotion. |

## Gate 2: Held-Out Formula And Language Templates

Question:

> Does the bridge survive when formula families, language wrappers, and answer
> positions are held out from train/safety calibration?

Corpus contract:

| split | formula families | language wrappers |
| --- | --- | --- |
| train | seen formulas | seen wrappers |
| safety_calibration | seen formulas | seen wrappers |
| validation | held-out formulas | held-out wrappers |
| extrapolation | held-out formulas | held-out wrappers |

Seen formula families:

```text
d1_affine
d2_quadratic_mix
d3_log_quadratic
d4_exp_log_product
```

Held-out formula families:

```text
d1_exp_soft
d2_reciprocal_shift
d3_ratio_product
d4_nested_ratio_exp
```

Answer positions are no longer fixed. The generated corpus has answer tokens at
positions `1`, `6`, `8`, and `9`.

Artifacts:

```text
artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math-heldout-templates/
artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-language-math-heldout-templates-transformer-calibrated-v1/
```

Corpus command:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind language-math-heldout-templates \
  --source-bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label bridge-corpus-v1-language-math-heldout-templates \
  --output-dir artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math-heldout-templates \
  --seed 20260418 \
  --language-train-per-group 12 \
  --language-safety-per-group 16 \
  --language-eval-per-group 20
```

LM command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math-heldout-templates/summary.json \
  --run-label bridge-corpus-v1-language-math-heldout-templates-transformer-calibrated-v1 \
  --output-dir artifacts/bridge-corpus-v1-lm/bridge-corpus-v1-language-math-heldout-templates-transformer-calibrated-v1 \
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
  --device mps
```

Caption: Gate 2 original held-out-template result on the frozen extrapolation
`math_answer` slice. `final acc`/`final NLL` are the model's fused output;
`LM-only acc` is the same model's decoder prediction before expert fusion;
`expert mass/call` and `unsafe mass/call` are soft mass for fusion rows and hard
call rates for hard-call rows.

| run | final acc | final NLL | LM-only acc | expert mass/call | unsafe mass/call |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.000 | 11.094 | 0.000 | 0.000 | 0.000 |
| `lm-x-task` | 0.013 | 8.926 | 0.013 | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.031 | 8.410 | 0.031 | 0.000 | 0.000 |
| `lm-router-hard-call` | 0.006 | 9.350 | 0.006 | 0.000 | 0.000 |
| `lm-router-logit-fusion` | 0.163 | 5.972 | 0.069 | 0.747 | 0.279 |
| `lm-router-prob-mixture` | 0.625 | 6.703 | 0.000 | 1.000 | 0.375 |

Held-out extrapolation safe answer coverage is `0.650`.

Verdict:

- Capability partially survives the held-out-template gate. Probability mixture
  reaches `0.625` math-answer accuracy while token-only is `0.000`.
- Safety does not pass. Probability mixture assigns unsafe expert mass on
  `0.375` of held-out math-answer positions, and logit fusion also assigns too
  much unsafe mass (`0.279`).
- The hard-call route stays safe by abstaining, but it gives up the capability
  gain. That means the current calibrated soft-fusion objective does not
  generalize its safety decision to unseen formula/template combinations.
- Gate 2 blocks promotion to a larger/broader LM run until answer-token safe
  mass is trained more directly on held-out-style diversity.

Next useful run:

- Add a held-out-template training mixture or cross-template safety calibration
  that includes seen formulas under held-out wrappers and held-out-like no-safe
  cases, then rerun Gate 2.
- Keep the same Gate 2 eval split frozen so improvement is measured against this
  failure, not against an easier corpus.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Determine whether the bridge generalizes beyond the fixed language wrapper, fixed answer position, and train-seen formula families. |
| Promotion condition | On the frozen held-out-template eval, the hybrid must materially beat token-only and side-channel controls on math-answer accuracy/NLL while keeping unsafe answer mass/calls below the existing safety gate. |
| Current blocker | Initial pass failed safety (`0.375` unsafe mass). Repair pass lowers probability-mixture unsafe answer mass to `0.049` while retaining `0.562` math-answer accuracy, so the role-aware probability-mixture blocker is resolved. |
| Next action | Carry the repaired probability-mixture calibration into Gate 5, while keeping hard-call and overall prose behavior as separate controls. |

## Gate 3: Target/Random-Label And Wrong-Expert Controls

Question:

> Does the harness leak target identity through labels, routing targets, feature
> construction, or expert-token priors?

Artifacts:

```text
artifacts/bridge-corpus-v1-gate3/
artifacts/bridge-corpus-v1-gate3-lm/
```

Controls:

- Randomize math-answer target tokens while preserving input/template structure.
- Pair rows with wrong expert payloads from different tasks.
- Recompute safety metadata after each transform.
- Run the same calibrated transformer recipe used by the positive
  `language + math` result.

Corpus commands:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind target-randomized \
  --source-corpus-summary artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math/summary.json \
  --run-label bridge-corpus-v1-language-math-target-randomized \
  --output-dir artifacts/bridge-corpus-v1-gate3/bridge-corpus-v1-language-math-target-randomized \
  --shuffle-seed 20260418

python scripts/symbolic_bridge_corpus.py \
  --corpus-kind wrong-expert \
  --source-corpus-summary artifacts/bridge-corpus-v1/bridge-corpus-v1-language-math/summary.json \
  --run-label bridge-corpus-v1-language-math-wrong-expert \
  --output-dir artifacts/bridge-corpus-v1-gate3/bridge-corpus-v1-language-math-wrong-expert \
  --shuffle-seed 20260418
```

The target-randomized corpus changed `0.819` of math-answer labels. The
wrong-expert corpus paired answer rows with a different task `1.000` of the
time.

Caption: Gate 3 leakage controls on the extrapolation `math_answer` slice.
Target-randomized changes answer labels; wrong-expert pairs rows with expert
payloads from other tasks. If the bridge still won here, the benchmark would be
leaky.

| condition | safe answer coverage | token-only acc | side-channel acc | prob-mixture acc | prob-mixture NLL | prob unsafe mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline all-four | 0.825 | 0.375 | 0.606 | 0.872 | 1.639 | 0.058 |
| target-randomized | 0.263 | 0.125 | 0.113 | 0.134 | 6.689 | 0.161 |
| wrong-expert | 0.250 | 0.375 | 0.562 | 0.328 | 3.738 | 0.272 |

Verdict:

- The positive hybrid effect does not survive broken labels. Probability-mixture
  math-answer accuracy falls from `0.872` to `0.134`, essentially the same
  low-accuracy regime as token-only (`0.125`).
- The positive hybrid effect does not survive wrong expert pairing.
  Probability-mixture accuracy falls to `0.328`, below token-only (`0.375`) and
  well below side-channel (`0.562`).
- This is evidence against target-label leakage, task-level expert-token priors,
  or feature-construction leakage as the source of the original gain.
- The controls still produce residual safe coverage through token collisions,
  which is expected for quantized token outputs. That residual does not preserve
  the bridge effect.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Prove that the bridge cannot succeed when target labels or expert payload alignment are deliberately broken. |
| Promotion condition | Token-only and hybrid variants should not show a meaningful math-answer gain on randomized labels or wrong-expert pairings; shuffled/wrong experts should perform at or below baseline. |
| Current blocker | Gate 3 itself is not blocked; it passed. Broken-label and wrong-expert controls collapse the original probability-mixture gain. Gate 2 safety remains the active blocker for promotion. |
| Next action | Continue to Gate 4 seed/template variance, while separately planning a Gate 2 safety repair pass with explicit safe-expert-mass calibration. |

## Gate 4: Seed And Template Variance

Question:

> Is the positive bridge result stable across seeds, template choices, and
> formula split choices?

Artifacts:

```text
artifacts/bridge-corpus-v1-gate4/
artifacts/bridge-corpus-v1-gate4-lm/
```

Controls:

- Repeat the held-out-template setup over multiple corpus seeds.
- Rotate which formula families are seen vs held out within each depth by seed.
- Vary held-out wrapper assignment and answer positions.
- Report mean, variance, and worst-case safety metrics.

Corpus command shape:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind language-math-heldout-variance \
  --source-bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label bridge-corpus-v1-language-math-heldout-variance-s<seed> \
  --output-dir artifacts/bridge-corpus-v1-gate4/bridge-corpus-v1-language-math-heldout-variance-s<seed> \
  --seed <seed> \
  --language-train-per-group 12 \
  --language-safety-per-group 16 \
  --language-eval-per-group 20
```

LM command shape:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate4/bridge-corpus-v1-language-math-heldout-variance-s<seed>/summary.json \
  --run-label language-math-heldout-variance-s<seed>-transformer-calibrated-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate4-lm/language-math-heldout-variance-s<seed>-transformer-calibrated-v1 \
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
  --device mps
```

Caption: Gate 4 original variance sweep on the extrapolation `math_answer`
slice. Each variance row rotates which formulas are seen during fit versus held
out during eval. `contract` is the pre-repair gate verdict for that run.

| condition | held-out formulas | safe answer coverage | token-only acc | side-channel acc | prob-mixture acc | prob-mixture NLL | prob call/mass | prob unsafe mass | contract |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Gate 2 fixed seed | `d1_exp_soft`, `d2_reciprocal_shift`, `d3_ratio_product`, `d4_nested_ratio_exp` | 0.650 | 0.000 | 0.031 | 0.625 | 6.703 | 1.000 | 0.375 | false |
| variance `20260419` | `d1_exp_soft`, `d2_quadratic_mix`, `d3_ratio_product`, `d4_exp_log_product` | 0.825 | 0.000 | 0.000 | 0.225 | 4.338 | 0.320 | 0.055 | false |
| variance `20260420` | `d1_affine`, `d2_reciprocal_shift`, `d3_log_quadratic`, `d4_nested_ratio_exp` | 0.825 | 0.000 | 0.000 | 0.087 | 4.554 | 0.042 | 0.010 | false |
| variance `20260421` | `d1_exp_soft`, `d2_quadratic_mix`, `d3_ratio_product`, `d4_exp_log_product` | 0.825 | 0.000 | 0.000 | 0.394 | 3.866 | 0.511 | 0.090 | false |

Caption: Aggregate of the three original Gate 4 variance rows above. This
summarizes stability: the mean tells us typical behavior, while min/max show the
worst and best split.

| metric | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| token-only math-answer acc | 0.000 | 0.000 | 0.000 | 0.000 |
| side-channel math-answer acc | 0.000 | 0.000 | 0.000 | 0.000 |
| prob-mixture math-answer acc | 0.235 | 0.125 | 0.087 | 0.394 |
| prob-mixture math-answer NLL | 4.253 | 0.287 | 3.866 | 4.554 |
| prob-mixture expert mass/call | 0.291 | 0.192 | 0.042 | 0.511 |
| prob-mixture unsafe mass | 0.052 | 0.033 | 0.010 | 0.090 |

Verdict:

- Gate 4 fails the promotion condition. The positive Gate 2 capability signal is
  not stable across rotated formula/template splits.
- Probability mixture remains better than token-only and side-channel on the
  variance runs, but the accuracy range is wide (`0.087` to `0.394`) and far
  below the fixed Gate 2 split (`0.625`).
- The router often solves safety by becoming timid: unsafe mass is much lower
  than Gate 2, but expert mass also falls sharply. That preserves safety in some
  splits by giving up the bridge.
- No variance run confirms the full contract. This blocks broader LM promotion
  until Gate 2 safety and Gate 4 robustness are repaired together.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Establish whether the bridge result is robust or an accident of one formula/template split. |
| Promotion condition | The capability gain and safety behavior must be stable enough across seeds/templates that the mean result is useful and the worst-case unsafe mass/call rate is acceptable. |
| Current blocker | Initial pass failed. Repair pass resolves the math-answer probability-mixture blocker: variance mean accuracy rises from `0.235` to `0.685`, worst-case accuracy rises from `0.087` to `0.606`, and worst-case unsafe mass falls to `0.021`. Overall hard-call contract remains weaker and should stay a control, not the primary bridge claim. |
| Next action | Proceed to Gate 5 with the repaired probability-mixture recipe and preserve token-only, side-channel, hard-call, and logit-fusion controls. |

## Gate 2/4 Repair: Multi-Split Calibration

Question:

> Can broader synthetic pretraining repair the Gate 2 safety failure and Gate 4
> variance failure without changing the core bridge architecture?

Code changes:

- The LM runner now accepts extra fit-only bridge summaries. Their
  `train`/`safety_calibration` rows are used for fitting, while the primary
  gate corpus remains the frozen evaluation target.
- Extra fit rows are sequence-namespaced before batching so multiple rotations
  cannot accidentally merge sequences with the same source IDs.
- The router objective now has an answer-token call/abstain term, separate from
  global call/abstain and non-answer abstention. This avoids letting prose and
  context tokens dominate the decision that matters for math answers.
- Reports now expose a role-aware probability-mixture math-answer contract
  instead of burying it inside per-role metrics.

Artifacts:

```text
artifacts/bridge-corpus-v1-repair/
artifacts/bridge-corpus-v1-repair-lm/
```

Calibration corpus command shape:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind language-math-heldout-variance \
  --source-bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label bridge-corpus-v1-language-math-calibration-s<seed> \
  --output-dir artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s<seed> \
  --seed <seed> \
  --language-train-per-group 32 \
  --language-safety-per-group 96 \
  --language-eval-per-group 4
```

Repair LM command shape:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary <frozen-gate-summary.json> \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps
```

Caption: Gate 2 repair on the same frozen held-out-template extrapolation
`math_answer` slice as the original Gate 2 table. The first repair adds
multi-split calibration data only; the second also adds the answer-token
call/abstain objective.

| condition | prob-mixture acc | prob-mixture NLL | prob expert mass | prob unsafe mass | role-aware contract |
| --- | ---: | ---: | ---: | ---: | --- |
| original calibrated | 0.625 | 6.703 | 1.000 | 0.375 | false |
| multi-split calibration only | 0.419 | 4.159 | 0.362 | 0.054 | false |
| multi-split + answer call/abstain | 0.562 | 2.859 | 0.468 | 0.049 | true |

Caption: Gate 4 repair on the same frozen variance extrapolation `math_answer`
slices. Rows compare original versus repaired probability-mixture behavior for
each held-out split seed.

| condition | prob-mixture acc | prob-mixture NLL | prob expert mass | prob unsafe mass | role-aware contract |
| --- | ---: | ---: | ---: | ---: | --- |
| original variance `20260419` | 0.225 | 4.338 | 0.320 | 0.055 | false |
| original variance `20260420` | 0.087 | 4.554 | 0.042 | 0.010 | false |
| original variance `20260421` | 0.394 | 3.866 | 0.511 | 0.090 | false |
| repaired variance `20260419` | 0.719 | 1.607 | 0.643 | 0.011 | true |
| repaired variance `20260420` | 0.606 | 1.875 | 0.493 | 0.021 | true |
| repaired variance `20260421` | 0.731 | 1.471 | 0.691 | 0.013 | true |

Caption: Aggregate of the original and repaired Gate 4 variance runs. The key
question is whether the repair improves both average behavior and worst-case
behavior without increasing unsafe expert mass.

| metric | original mean | original min | original max | repaired mean | repaired min | repaired max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prob-mixture math-answer acc | 0.235 | 0.087 | 0.394 | 0.685 | 0.606 | 0.731 |
| prob-mixture math-answer NLL | 4.253 | 3.866 | 4.554 | 1.651 | 1.471 | 1.875 |
| prob-mixture expert mass/call | 0.291 | 0.042 | 0.511 | 0.609 | 0.493 | 0.691 |
| prob-mixture unsafe mass | 0.052 | 0.010 | 0.090 | 0.015 | 0.011 | 0.021 |

Verdict:

- The repair materially improves the paper-aligned bridge question. Gate 4 is
  no longer an instability story on the math-answer probability-mixture metric.
- Gate 2 safety is substantially repaired: unsafe mass drops from `0.375` to
  `0.049`, with accuracy still far above token-only and side-channel controls.
- The improvement comes from calibration data and objective shape, not from a
  bigger LM or a new expert architecture.
- Caveat: the top-level hard-call contract is still weaker and sometimes passes
  by abstaining. The bridge claim should therefore stay attached to the
  probability-mixture path plus role-aware safety metrics.

## Repaired Gates 1-4 Audit

Question:

> After repairing the calibration objective, do the original null gates still
> hold, or did the repair open a new loophole?

Artifacts:

```text
artifacts/bridge-corpus-v1-repair-audit/
artifacts/bridge-corpus-v1-repair-audit-lm/
```

Note: the original Gate 1/3 language+math corpora used a smaller `109`-token
vocabulary. The repaired calibration rotations use the held-out-template
`123`-token vocabulary, and those vocabularies are not prefix-compatible. The
repair audit therefore regenerates Gate 1/3 controls from the frozen
held-out-template surface and applies the same transform to all four calibration
rotations. This keeps the control-plane contract explicit: each expert-bank or
leakage condition is matched between eval rows and fit-only calibration rows.

Caption: Repaired Gate 1 audit on extrapolation `math_answer` tokens. `all-four`
uses the repaired Gate 2 all-expert run. Other rows use matched transformed eval
and calibration corpora.

| condition | token-only acc | side-channel acc | prob-mixture acc | prob-mixture NLL | prob expert mass | prob unsafe mass | role-aware contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all-four | 0.000 | 0.000 | 0.562 | 2.859 | 0.468 | 0.049 | true |
| paper-complex only | 0.000 | 0.000 | 0.194 | 4.295 | 0.100 | 0.001 | true |
| stable-real only | 0.000 | 0.000 | 0.019 | 9.267 | 0.000 | 0.000 | false |
| generic-tree only | 0.000 | 0.013 | 0.013 | 10.707 | 0.000 | 0.000 | false |
| small-MLP only | 0.000 | 0.025 | 0.025 | 7.289 | 0.001 | 0.001 | false |
| symbolic trees | 0.000 | 0.000 | 0.256 | 3.895 | 0.283 | 0.030 | true |
| non-EML control | 0.000 | 0.013 | 0.006 | 9.658 | 0.001 | 0.001 | false |
| shuffled all | 0.000 | 0.037 | 0.019 | 10.077 | 0.017 | 0.016 | false |

Caption: Repaired Gate 3 audit on extrapolation `math_answer` tokens. Both
controls are regenerated on the repaired-compatible held-out-template surface
and matched across calibration rows.

| condition | token-only acc | side-channel acc | prob-mixture acc | prob-mixture NLL | prob expert mass | prob unsafe mass | role-aware contract |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| target-randomized | 0.000 | 0.000 | 0.000 | 8.001 | 0.005 | 0.005 | false |
| wrong-expert | 0.000 | 0.062 | 0.006 | 9.279 | 0.007 | 0.006 | false |

Caption: Repaired Gate 2/4 audit summary on extrapolation `math_answer` tokens.

| gate | condition | prob-mixture acc | prob-mixture NLL | prob expert mass | prob unsafe mass | role-aware contract |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2 | frozen held-out-template | 0.562 | 2.859 | 0.468 | 0.049 | true |
| 4 | variance `20260419` | 0.719 | 1.607 | 0.643 | 0.011 | true |
| 4 | variance `20260420` | 0.606 | 1.875 | 0.493 | 0.021 | true |
| 4 | variance `20260421` | 0.731 | 1.471 | 0.691 | 0.013 | true |

Audit verdict:

- The repair survives the full Gates 1-4 audit on the role-aware
  probability-mixture metric.
- Gate 1 remains meaningful after repair: paper-complex-only and symbolic-tree
  conditions pass, while stable-real-only, generic-tree-only, small-MLP-only,
  non-EML, and shuffled controls do not.
- Gate 3 remains meaningful after repair: randomized labels and wrong expert
  pairings do not reproduce the bridge gain.
- Gates 2 and 4 remain repaired on their frozen eval splits.
- The claim is still bounded. This audit supports moving to Gate 5, not claiming
  a broad LM improvement.

## Gate 5: More Natural Mixed Corpus

Question:

> Does the bridge contract hold beyond this synthetic grammar?

Status: partial pass.

Controls:

- Mix pure-language examples with less templated math prompts.
- Include multiple prompt phrasings and distractor prose around math answers.
- Preserve role labels so math-answer and prose/context behavior remain
  separately measurable.
- Keep pure transformer, side-channel, hard-call, logit-fusion, and
  probability-mixture comparisons.
- Preserve the repaired `123`-token held-out-template vocabulary so the same
  multi-split calibration runs can be reused.

Artifacts:

```text
artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/
artifacts/bridge-corpus-v1-gate5-lm/gate5-repair-language-math-natural-v1/
```

Corpus command:

```bash
python scripts/symbolic_bridge_corpus.py \
  --corpus-kind language-math-natural \
  --source-bridge-summary artifacts/symbolic-bridge/symbolic-bridge-compact-runpod-cuda-v1/summary.json \
  --run-label bridge-corpus-v1-language-math-natural-gate5 \
  --output-dir artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5 \
  --seed 20260418 \
  --language-train-per-group 12 \
  --language-safety-per-group 16 \
  --language-eval-per-group 20 \
  --output table
```

LM command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-repair-language-math-natural-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-repair-language-math-natural-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

Corpus facts:

| field | value |
| --- | ---: |
| token bins | 123 |
| total token rows | 9,752 |
| math-answer rows | 544 |
| math-context rows | 1,632 |
| prose rows | 7,576 |
| extrapolation math-answer rows | 160 |
| prose-only sequences per split | 20 |

Held-out formulas were the same frozen formula families used by Gate 2:
`d1_exp_soft`, `d2_reciprocal_shift`, `d3_ratio_product`, and
`d4_nested_ratio_exp`. Held-out natural wrappers placed answer tokens at
indices `1`, `14`, `15`, `16`, `17`, and `18`.

Caption: Gate 5 whole-corpus extrapolation results. This table includes prose,
math context, and math answer tokens together, so it tests whether the bridge
improves the full mixed sequence distribution rather than only answer slots.

| run | extrap acc | extrap NLL | expert mass/call | unsafe mass/call |
| --- | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.166 | 6.378 | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.284 | 6.471 | 0.000 | 0.000 |
| `lm-router-hard-call` | 0.219 | 7.349 | 0.000 | 0.000 |
| `lm-router-logit-fusion` | 0.190 | 6.631 | 0.029 | 0.008 |
| `lm-router-prob-mixture` | 0.256 | 5.992 | 0.031 | 0.004 |

Caption: Gate 5 extrapolation `math_answer` role only. This is the
bridge-critical slice: the answer token has a symbolic expert prediction
available, while surrounding prose and context tokens should mostly abstain.

| run | math-answer acc | math-answer NLL | expert mass/call | unsafe mass/call |
| --- | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.062 | 6.395 | 0.000 | 0.000 |
| `lm-frozen-side-channel` | 0.106 | 7.009 | 0.000 | 0.000 |
| `lm-router-hard-call` | 0.113 | 8.352 | 0.000 | 0.000 |
| `lm-router-logit-fusion` | 0.400 | 4.777 | 0.517 | 0.134 |
| `lm-router-prob-mixture` | 0.544 | 2.301 | 0.540 | 0.041 |

Caption: Gate 5 extrapolation non-answer behavior. This table separates the
surrounding language and formula-context tokens so the answer-token gain cannot
hide prose degradation.

| run | prose acc | prose NLL | math-context acc | math-context NLL |
| --- | ---: | ---: | ---: | ---: |
| `lm-token-only` | 0.172 | 6.529 | 0.173 | 5.687 |
| `lm-frozen-side-channel` | 0.232 | 7.136 | 0.577 | 3.271 |
| `lm-router-hard-call` | 0.156 | 7.943 | 0.542 | 4.318 |
| `lm-router-logit-fusion` | 0.157 | 7.233 | 0.269 | 4.512 |
| `lm-router-prob-mixture` | 0.167 | 7.214 | 0.565 | 1.677 |

Gate 5 verdict:

- The math-answer bridge survives the more natural mixed corpus. Probability
  mixture beats token-only by `+0.481` accuracy and frozen side-channel by
  `+0.438` on extrapolation answer tokens, with unsafe answer mass `0.041`.
- Relative to the repaired Gate 2 answer slice, capability is essentially
  preserved: accuracy is `0.544` versus `0.562`, NLL improves from `2.859` to
  `2.301`, and unsafe mass improves from `0.049` to `0.041`.
- The full mixed-corpus contract is not confirmed. Frozen side-channel has
  better whole-corpus extrapolation accuracy (`0.284` vs `0.256`), and
  probability-mixture prose accuracy/NLL is slightly worse than token-only
  (`0.167`/`7.214` vs `0.172`/`6.529`).
- The positive claim should stay role-local: calibrated symbolic experts can
  materially help answer tokens when a safe expert is available, but this run
  does not show a general language-model improvement.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Determine whether the EML bridge is useful outside the controlled synthetic language/math grammar. |
| Promotion condition | The hybrid must improve math-answer accuracy/NLL without degrading pure-language/prose behavior and without violating unsafe-call or unsafe-mass gates. |
| Current blocker | Math-answer transfer passes, but prose retention does not. The fusion path is useful as an answer/action bridge, not yet as a general next-token improvement. |
| Next action | Run the ablation ladder one step at a time. Ablation 1 preserved answer gains but did not solve prose NLL. Ablation 2 retention-only worsened prose and broke answer safety. Ablation 3 showed teacher KL is the right prose-retention direction, and the weight sweep found `0.5` as the current best safe all-non-answer KL point. Ablation 3c showed prose-only KL preserves math-context behavior better but gives up some prose-NLL repair. The next clean test can combine prose-only `0.5` KL with answer-span fusion, but should still be treated as a composition test rather than assumed solved. |

### Gate 5 Ablation 1: Answer-Span-Only Fusion

Question:

> Is the prose degradation caused by stray soft expert fusion on non-answer
> tokens?

Change:

- Add `fusion_allowed_roles`.
- Rerun the Gate 5 recipe with soft expert fusion allowed only on
  `math_answer` and `math_only` roles.
- Keep model size, training data, calibration rotations, loss weights, and MPS
  backend otherwise unchanged.

Artifact:

```text
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-answer-span-fusion-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-ablation-answer-span-fusion-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-answer-span-fusion-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --fusion-allowed-roles math_answer,math_only \
  --device mps \
  --output table
```

Caption: Gate 5 Ablation 1 compares the original repaired probability-mixture
run with answer-span-only fusion. The intervention is narrow: expert mass is
forced to zero on prose and math-context tokens, while answer tokens can still
receive the symbolic expert mixture.

| run | whole acc | whole NLL | math-answer acc | math-answer NLL | prose acc | prose NLL | context acc | context NLL | answer unsafe mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired Gate 5 | 0.256 | 5.992 | 0.544 | 2.301 | 0.167 | 7.214 | 0.565 | 1.677 | 0.041 |
| answer-span fusion | 0.273 | 6.094 | 0.550 | 2.092 | 0.178 | 7.327 | 0.615 | 1.827 | 0.045 |

Interpretation:

- Answer-span fusion does not remove the symbolic answer benefit. The
  math-answer result is essentially preserved and slightly improves on this
  run (`0.544 -> 0.550`, NLL `2.301 -> 2.092`).
- The gate successfully removes non-answer expert mass: prose and math-context
  expert mass/unsafe mass go to `0.000`.
- Prose is not repaired. Prose accuracy moves up a little (`0.167 -> 0.178`),
  but prose NLL gets slightly worse (`7.214 -> 7.327`) and remains worse than
  the pure token-only prose baseline from the original Gate 5 run.
- Therefore the main failure is probably not stray softmax mass on prose. The
  next ablation should target the shared training objective, such as explicit
  prose-retention/KL against the token-only LM on non-answer roles.

### Gate 5 Ablation 2: Non-Answer LM Retention

Question:

> Is prose degradation caused by router/fusion training pulling the shared LM
> head away from non-answer language modeling?

Change:

- Add `non_answer_lm_retention_loss_weight`.
- Leave answer-span-only fusion off.
- Add extra pre-fusion LM-head NLL on `prose` and `math_context` roles for
  router/fusion variants.
- Keep model size, training data, calibration rotations, safety losses, and MPS
  backend otherwise unchanged.

Artifact:

```text
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-non-answer-retention-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-ablation-non-answer-retention-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-non-answer-retention-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --non-answer-lm-retention-loss-weight 2.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

Caption: Gate 5 Ablation 2 adds only a non-answer LM-retention objective and
does not use answer-span fusion. This tests the shared-training hypothesis
without stacking the previous inference-time span gate.

| run | whole acc | whole NLL | math-answer acc | math-answer NLL | prose acc | prose NLL | context acc | context NLL | answer unsafe mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired Gate 5 | 0.256 | 5.992 | 0.544 | 2.301 | 0.167 | 7.214 | 0.565 | 1.677 | 0.041 |
| answer-span fusion | 0.273 | 6.094 | 0.550 | 2.092 | 0.178 | 7.327 | 0.615 | 1.827 | 0.045 |
| retention-only | 0.256 | 6.537 | 0.575 | 2.525 | 0.136 | 7.928 | 0.694 | 1.557 | 0.153 |

Interpretation:

- Retention-only is not a valid repair at weight `2.0`. It improves
  math-answer accuracy (`0.575`) and math-context accuracy (`0.694`), but
  worsens whole-corpus NLL and prose behavior.
- It breaks the safety part of the answer contract: unsafe answer mass rises
  from `0.041` to `0.153`, above the `<= 0.05` gate.
- The extra self-NLL objective appears to redirect capacity toward context
  tokens while weakening answer safety calibration. It is not equivalent to a
  true frozen-token-only teacher or KL-retention objective.
- Do not stack this exact retention-only setting into the bridge. A better next
  ablation is either a lower retention weight with answer-span fusion, or a
  frozen token-only teacher KL on prose roles.

### Gate 5 Ablation 3: Frozen Token-Only Teacher KL

Question:

> Can a frozen token-only teacher preserve the hybrid's non-answer language
> distribution better than self-retention, while keeping the answer bridge
> intact?

Change:

- Add `non_answer_teacher_kl_loss_weight`.
- Train a frozen token-only teacher inside the same LM runner.
- Add masked `KL(p_teacher || p_hybrid_final)` on `prose` and `math_context`
  roles for router/fusion variants.
- Leave answer-span-only fusion off.
- Keep model size, training data, calibration rotations, safety losses, and MPS
  backend otherwise unchanged.

Artifact:

```text
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-ablation-teacher-kl-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --non-answer-teacher-kl-loss-weight 1.0 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

Caption: Gate 5 Ablation 3 uses a frozen token-only teacher KL on non-answer
roles. It is more principled than self-retention because the teacher
distribution is fixed before hybrid training.

| run | whole acc | whole NLL | math-answer acc | math-answer NLL | prose acc | prose NLL | context acc | context NLL | answer unsafe mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| repaired Gate 5 | 0.256 | 5.992 | 0.544 | 2.301 | 0.167 | 7.214 | 0.565 | 1.677 | 0.041 |
| answer-span fusion | 0.273 | 6.094 | 0.550 | 2.092 | 0.178 | 7.327 | 0.615 | 1.827 | 0.045 |
| self-retention | 0.256 | 6.537 | 0.575 | 2.525 | 0.136 | 7.928 | 0.694 | 1.557 | 0.153 |
| teacher KL | 0.213 | 5.165 | 0.556 | 3.165 | 0.162 | 5.891 | 0.333 | 2.535 | 0.106 |

Interpretation:

- Teacher KL confirms that a frozen language teacher is the right kind of
  pressure for prose NLL: prose NLL improves from `7.214` to `5.891`.
- It does not solve the full bridge. Whole-corpus NLL improves, but whole
  accuracy drops, math-context accuracy drops, and answer NLL worsens.
- Most importantly, the safety contract fails: unsafe answer mass rises from
  `0.041` to `0.106`.
- This is a partial/unsafe result, not a repair. The next ablation should lower
  teacher-KL weight or pair teacher KL with stronger answer-safety calibration
  before considering it for the bridge recipe.

### Gate 5 Ablation 3b: Teacher-KL Weight Tuning

Question:

> Can a smaller frozen-teacher KL weight preserve the prose-NLL improvement
> while keeping answer unsafe mass under the `<= 0.05` safety gate?

Artifacts:

```text
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-w025-v1/
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-w050-v1/
```

Commands:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-ablation-teacher-kl-w025-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-w025-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --non-answer-teacher-kl-loss-weight 0.25 \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

The `0.5` run used the same command with
`--run-label gate5-ablation-teacher-kl-w050-v1`, matching output directory, and
`--non-answer-teacher-kl-loss-weight 0.5`.

Caption: Gate 5 teacher-KL weight sweep. This table tunes only the frozen
teacher KL weight; answer-span fusion remains off.

| teacher KL weight | whole acc | whole NLL | math-answer acc | math-answer NLL | prose acc | prose NLL | context acc | context NLL | answer unsafe mass | answer contract |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.256 | 5.992 | 0.544 | 2.301 | 0.167 | 7.214 | 0.565 | 1.677 | 0.041 | true |
| 0.25 | 0.227 | 5.364 | 0.569 | 2.526 | 0.162 | 6.238 | 0.408 | 2.340 | 0.035 | true |
| 0.5 | 0.244 | 5.120 | 0.562 | 2.502 | 0.185 | 5.984 | 0.408 | 2.069 | 0.043 | true |
| 1.0 | 0.213 | 5.165 | 0.556 | 3.165 | 0.162 | 5.891 | 0.333 | 2.535 | 0.106 | false |

Interpretation:

- `0.5` is the current best safe teacher-KL setting. It improves whole NLL
  (`5.992 -> 5.120`) and prose NLL (`7.214 -> 5.984`) while preserving the
  answer safety gate (`0.043 <= 0.05`).
- `0.25` is also safe but weaker on prose NLL and whole NLL.
- `1.0` buys only a little more prose NLL and breaks answer safety.
- Teacher KL still trades off math-context behavior versus repaired Gate 5, so
  this is not a finished bridge recipe. It is the best single-knob
  prose-retention candidate before composition with answer-span fusion.

### Gate 5 Ablation 3c: Prose-Only Teacher-KL Mask

Question:

> Is the math-context regression caused by applying frozen-teacher KL to both
> prose and math-context roles?

Artifact:

```text
artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-prose-w050-v1/
```

Command:

```bash
uv run --python 3.12 --with torch --with numpy python scripts/symbolic_bridge_lm.py \
  --bridge-summary artifacts/bridge-corpus-v1-gate5/bridge-corpus-v1-language-math-natural-gate5/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260430/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260431/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260432/summary.json \
  --extra-fit-bridge-summary artifacts/bridge-corpus-v1-repair/bridge-corpus-v1-language-math-calibration-s20260433/summary.json \
  --run-label gate5-ablation-teacher-kl-prose-w050-v1 \
  --output-dir artifacts/bridge-corpus-v1-gate5-lm/gate5-ablation-teacher-kl-prose-w050-v1 \
  --backbone transformer \
  --transformer-layers 2 \
  --transformer-heads 4 \
  --transformer-ffn-multiplier 2 \
  --epochs 600 \
  --learning-rate 0.004 \
  --hidden-units 64 \
  --router-loss-weight 10.0 \
  --call-abstain-loss-weight 1.0 \
  --answer-call-abstain-loss-weight 8.0 \
  --answer-unsafe-loss-weight 5.0 \
  --non-answer-abstain-loss-weight 2.0 \
  --non-answer-teacher-kl-loss-weight 0.5 \
  --non-answer-teacher-kl-roles prose \
  --router-call-threshold 0.99999 \
  --expert-logit-scale 6.0 \
  --device mps \
  --output table
```

Caption: Gate 5 Ablation 3c keeps the frozen-teacher KL weight at `0.5`, but
applies it only to `prose`, not `math_context`. Answer-span fusion remains off.

| condition | KL roles | whole acc | whole NLL | math-answer acc | math-answer NLL | prose acc | prose NLL | context acc | context NLL | answer unsafe mass | answer contract |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| repaired Gate 5 | none | 0.256 | 5.992 | 0.544 | 2.301 | 0.167 | 7.214 | 0.565 | 1.677 | 0.041 | true |
| teacher KL `0.5` | prose, math_context | 0.244 | 5.120 | 0.562 | 2.502 | 0.185 | 5.984 | 0.408 | 2.069 | 0.043 | true |
| prose-only teacher KL `0.5` | prose | 0.265 | 5.250 | 0.550 | 1.877 | 0.179 | 6.241 | 0.558 | 1.877 | 0.037 | true |

Interpretation:

- Prose-only KL confirms that the math-context drop was mostly caused by
  constraining math-context tokens to the frozen token-only teacher.
- It is the cleaner safety/answer tradeoff than all-non-answer KL: answer NLL
  improves (`2.301 -> 1.877`), answer unsafe mass stays under the gate
  (`0.037`), and math-context accuracy nearly returns to baseline
  (`0.558` vs `0.565`).
- It gives up some prose-NLL repair versus all-non-answer KL (`6.241` vs
  `5.984`) and still does not beat the best whole-corpus control on accuracy.
- This is a better component for the next composition test than the old
  all-non-answer mask, but the null hypothesis is not yet disproven for broad
  language modeling.
