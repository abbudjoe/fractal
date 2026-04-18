# Symbolic Bridge Null-Hypothesis Gates

This ledger tracks the staged attempts to disprove the EML bridge result before
promoting it toward broader LM work. Each gate has a local artifact trail plus a
short verdict. The high-level summary is updated as gates complete; detailed
records stay below it.

## High-Level Status

| gate | question | status | current answer |
| --- | --- | --- | --- |
| 1. Expert-bank ablation and shuffle controls | Does the gain follow aligned EML expert predictions, or can any/shuffled expert bank fake it? | passed | The gain follows `paper-complex-eml`; non-EML and shuffled controls do not reproduce it. |
| 2. Held-out formula/language templates | Does the bridge survive unseen formula families, unseen wrappers, and varied answer positions? | mixed, safety failed | Capability partially survives on math answers, but soft unsafe expert mass is too high. |
| 3. Target/random-label and wrong-expert controls | Does the harness leak target identity through labels, routing, or feature construction? | passed | Broken labels and wrong expert pairings collapse the hybrid gain. |
| 4. Seed/template variance | Is the positive result stable across seeds and template draws? | pending | Not run yet. |
| 5. More natural mixed corpus | Does the contract hold beyond the synthetic grammar? | pending | Not run yet. |

Current recommendation: continue null testing, but do not promote the bridge as a
general LM improvement yet. Gate 1 supports the paper-complex signal. Gate 2
shows that held-out-template capability is real enough to keep testing, but the
calibration contract does not generalize safely to unseen formulas/templates.
Gate 3 does not show evidence of target-label or expert-pairing leakage.

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

Language+math extrapolation, math-answer role:

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

Extrapolation, math-answer role:

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
| Current blocker | Capability partially survives (`0.625` probability-mixture math-answer accuracy), but unsafe answer mass is too high (`0.375`), so the safety contract fails. |
| Next action | Add held-out-style safety calibration or an explicit safe-expert-mass objective, then rerun Gate 2 against the same frozen held-out formula/template split. |

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

Extrapolation, math-answer role:

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

Status: pending.

Planned controls:

- Repeat Gates 1 and 2 over multiple corpus seeds.
- Rotate which formula families are seen vs held out within each depth.
- Vary held-out wrapper assignment and answer positions.
- Report mean, variance, and worst-case safety metrics.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Establish whether the bridge result is robust or an accident of one formula/template split. |
| Promotion condition | The capability gain and safety behavior must be stable enough across seeds/templates that the mean result is useful and the worst-case unsafe mass/call rate is acceptable. |
| Current blocker | Only one Gate 2 held-out split has been run; no seed/template variance estimate exists yet. |
| Next action | Add a manifest or loop for several held-out-template seeds/splits, then summarize mean/std/min/max by math-answer role. |

## Gate 5: More Natural Mixed Corpus

Question:

> Does the bridge contract hold beyond this synthetic grammar?

Status: pending.

Planned controls:

- Mix pure-language examples with less templated math prompts.
- Include multiple prompt phrasings and distractor prose around math answers.
- Preserve role labels so math-answer and prose/context behavior remain
  separately measurable.
- Keep pure transformer, side-channel, hard-call, logit-fusion, and
  probability-mixture comparisons.

Resolution block:

| field | value |
| --- | --- |
| Resolution needed | Determine whether the EML bridge is useful outside the controlled synthetic language/math grammar. |
| Promotion condition | The hybrid must improve math-answer accuracy/NLL without degrading pure-language/prose behavior and without violating unsafe-call or unsafe-mass gates. |
| Current blocker | Gate 2 already exposed safety generalization failure under held-out templates, so a more natural corpus is premature until that is addressed. |
| Next action | After Gate 2 safety is repaired and Gate 3 leakage controls pass, build a small mixed naturalistic math-language corpus with frozen eval prompts. |
