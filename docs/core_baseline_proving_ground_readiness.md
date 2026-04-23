# Core Baseline Proving Ground Readiness

Generated: 2026-04-20

This is the control-plane checklist for turning the paper teachback into an empirically sound proving ground. It covers the 11 core architecture baselines from `docs/paper_primitive_research_teachback.md`.

Verdict: the repo has a useful Path 1 experiment scaffold, but it is not yet a paper-faithful proving ground for all core baselines. Several current rows are approximations that are good for scouting and bad for promotion unless labeled and controlled.

## Readiness Levels

| Level | Meaning |
| --- | --- |
| Ready | Paper-faithful implementation, contract tests, diagnostics, and matched benchmark manifest exist. |
| Partial | A useful scaffold or approximation exists, but at least one core paper contract is missing. |
| Absent | No meaningful implementation path is wired into the harness yet. |
| Deferred | Intentionally postponed because it is orthogonal or too expensive for the current tranche. |

## Existing Harness Assets

These pieces are already useful and should be reused:

- Path 1 CLI/config surface: `python/runners/path1_cli.py`.
- Path 1 variant/config types: `python/specs/path1.py`.
- Path 1 model construction and diagnostics: `python/models/path1.py`.
- Transformer, depth-attention approximation, paper MoDA reference attention, causal MoD approximation, paper MoD train-time top-C block, and looped-transformer scaffolds: `python/models/transformer.py`, `python/models/path1.py`.
- Shared train/eval loop with loss, throughput, process memory, and CUDA memory: `python/runtime/train_eval.py`.
- Report schema diagnostics slot: `python/reporting/schema.py`.
- Current smoke tests for attention, approximate depth attention, token routing, and Parcae-style recurrence: `python/tests/test_models.py`, `python/tests/test_specs.py`.
- Existing architecture search matrix: `docs/depth_recurrence_architecture_matrix.md`.

## Readiness Matrix

| Baseline | Current Implementation | Paper-Faithful Contract Tests | Diagnostics | Benchmark Manifest | Readiness | Blocking Gaps | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MoDA | `AttentionProfile.MODA_DEPTH_KV` remains an approximation. `AttentionProfile.PAPER_MODA_DEPTH_KV` and `PaperMoDACausalSelfAttention` now provide a slow same-token prior-depth KV reference path. | Contract tests now cover same-token depth visibility, no cross-token depth leakage, joint sequence/depth softmax behavior, causality for the approximation, and profile validation. | Approx and paper paths report sequence/depth attention mass, joint-softmax flag, memory scope, and same-token flag. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | Paper reference covers attention-side same-token prior-depth sequence KV, but does not implement Flash MoDA kernels or the optional FFN-side depth KV lane. Approx path is still intentionally named and diagnosed as approximate. | Use matched-v1 as local evidence only; add promotion-scale manifest and, if still needed, an FFN-side depth KV reference lane. Do not promote the old `moda-depth-kv` approximation as paper-faithful. |
| MoD | `TokenRoutedLocalCausalTransformerBlock` remains the causal decode-safe approximation. `TokenRoutingProfile.MOD_TRAIN_TOPC_BLOCK` and `PaperMoDTrainTopCTransformerBlock` now provide a separate full-sequence top-C training primitive. | Contract tests cover full-sequence capacity, selected-only attention in the paper lane, skipped-token identity, router-weighted selected update, causal approximation behavior, and profile validation. | Reports selected/skipped counts, selected fraction, router score stats, gate means, selected attention scope, selected attention token count, and causal-decode-safety flag. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | Paper train lane is intentionally non-causal and should not be used as a decode path. Matched-v1 now pairs paper train and causal decode controls under identical local budget. | Add promotion-scale paired train/decode manifests before making quality/efficiency claims. |
| Native Sparse Attention | Not implemented. | None. | None. | Deferred row exists only. | Absent | Need three separate attention branches: learned compressed global blocks, compression-derived selected fine blocks, local sliding window, gated sum. | Implement slow reference NSA first. Test branch separation, selection source, causal masks, and gated sum. |
| Fixed Looped LM | `Path1ScaffoldProfile.FIXED_LOOPED_LM` now runs `embed -> shared k-layer block group repeated L times -> output`. | Contract tests cover parameter count independent of loop count, stored vs effective layer count, embedding-once/head-once diagnostics, and forward finiteness. | Reports loop count, stored block count, effective layer count, shared-parameter flag, embedding/head application flags, hidden norms, and step norms. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | This is a strict looped decoder-LM scaffold, but not yet paired with long enough tasks or repeated-depth benchmark manifests to evaluate learning-algorithm behavior. | Use matched-v1 to choose loop-depth follow-ups, then compare loop counts under fixed stored parameter count at promotion scale. |
| Input-Injected Loop | `Path1ScaffoldProfile.LOOPED_ADDITIVE_INPUT` implements `Y <- M(Y + P)`. `Path1ScaffoldProfile.HUGINN_ADAPTER_RECURRENCE` implements a deterministic zero-state concat-adapter `R(e, s)` reference. | Contract tests cover zero initial state, repeated prompt injection, adapter prompt dependency, adapter diagnostics, and forward finiteness. | Reports injection mode, prompt-injected-each-loop flag, injection norms, adapter input/output norms, hidden norms, and step norms. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | Additive loop captures the 2311 input-injection equation. Huginn adapter captures the concat adapter surface but not sampled initial latent state, lognormal-Poisson depth sampling, sandwich blocks, or truncated BPTT. | Defer full Huginn recurrent-depth training recipe to recurrent-dynamics baseline. |
| Universal Transformer | `Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER` now runs a tied recurrent transition with per-step sinusoidal position/time coordinate signals. | Contract tests cover coordinate changes across recurrence, shared transition diagnostics, effective depth, and forward finiteness. | Reports coordinate modes/norms, loop count, stored/effective layer counts, hidden norms, step norms, and shared-step flag. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | This is decoder-lite UT with sinusoidal coordinate signals, not a full encoder-decoder reproduction or learned coordinate ablation. | Compare recurrence depths against fixed-looped LM before adding coordinate variants. |
| UT/ACT | `Path1ScaffoldProfile.UNIVERSAL_TRANSFORMER_ACT` now wraps UT with per-token ACT halting and weighted state interpolation. | Contract tests cover update weight sums, remainder behavior, weighted interpolation, forced final halt, threshold validation, and ponder loss surface. | Reports halt probability mean, remainders, update-count stats, update-weight-sum min/max, forced-final-halt fraction, ponder loss, and loss contract. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | Training now adds a separate ponder loss, while reported CE remains comparable. Still needs longer matched runs and possibly learned coordinate/transition variants before promotion. | Sweep ACT thresholds only after fixed UT baseline is stable under matched-v1 and promotion-scale budgets. |
| Recurrent-Depth Dynamics | Parcae middle-loop recurrence and acceleration/step-norm halting exist as approximations. | Loop/halting tests exist and now cover trajectory diagnostic lengths. No paper-specific group layout or sampled-depth tests. | Reports steps, step norms, cosine history, acceleration history, drift norms, average steps, halting metric, and exit count. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | Missing paper group layout, lognormal-Poisson train depth sampling, explicit per-group recurrent state contracts, and paper-faithful latent-dynamics baseline. | Add a recurrent-dynamics profile with explicit group definitions and sampled depth schedule. Keep Parcae labeled as an approximation. |
| Ouro Learned Exit | `Path1ScaffoldProfile.OURO_LEARNED_EXIT` now runs a tied looped decoder stack with per-step logits and a per-token learned exit distribution. | Contract tests cover `exit_pdf` normalization, final-step survival mass, Q-exit CDF crossing, forward finiteness, and expected-CE weighting. | Reports exit probability mass by step, CDF Q-exit depth, expected exit depth, gate probabilities, per-step CE, entropy, entropy auxiliary term, hidden norms, and shared-step flags. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | This implements the Stage 1 expected-CE plus entropy objective and deterministic Q-exit diagnostics. It does not yet implement Stage 2 frozen-LM continuation-benefit gate training. | If Stage 1 remains stable at promotion scale, add a separate Stage 2 gate-training runner rather than folding it into the generic LM loop. |
| RRT | `Path1ScaffoldProfile.RRT_CYCLE` now stores `K = total_layers / recursion_count` blocks and applies absolute depth `ell` through stored block `ell % K`. | Contract tests cover divisibility, true shared block object identity, shared norm parameters, CYCLE mapping, absolute-depth cache-key surface, and forward finiteness. | Reports stored/effective layer counts, recursion count, cycle map, last shared-layer indices, absolute-depth cache keys, hidden norms, step norms, strict-norm-sharing flag, and LoRA rank 0. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | This is strict recursive sharing only. Relaxed depth-specific LoRA, average layer initialization, and SVD residual initialization remain deferred. Incremental KV cache is not implemented in this harness, but diagnostics reserve absolute-depth cache keys. | Add relaxed LoRA only after strict CYCLE has a stable promotion-scale comparison. |
| MoR | `Path1ScaffoldProfile.MOR_EXPERT_CHOICE` now runs a unique first block, shared middle stack, per-recursion expert-choice routers, active-token shrinkage, and unique final block. | Contract tests cover minimum layer count, route/update controls, active set shrinkage, sorted selected indices, unselected-token halt semantics, router aux loss, KV policy diagnostics, and forward finiteness. | Reports active/selected counts per recursion, selected fractions, selected positions, gate means, subset flags, router aux loss, KV policy, decode-safety flag, stored/effective layer counts, hidden norms, and step norms. | Matched local manifest exists in `artifacts/core-baseline-matched-v1`; promotion-scale manifest is still missing. | Partial | This is a full-sequence expert-choice train/eval reference and is explicitly not decode-safe. It uses selected-subsequence causal attention and a simple BCE top-k router auxiliary loss. Token-choice MoR and recursive KV-sharing variants remain deferred. | Compare against RRT CYCLE plus token-selective recurrence controls before considering token-choice or KV-sharing variants. |

## Minimum Promotion Gates

A core baseline is not promotion-grade until all gates below pass:

1. Implementation gate: the primitive is wired as an explicit named profile, not a hidden combination of unrelated flags.
2. Contract-test gate: tests prove the paper-specific mathematical behavior, not only shape/causality.
3. Diagnostics gate: report JSON exposes the primitive-specific state needed to audit compute allocation.
4. Benchmark gate: a manifest records data, seed, token budget, optimizer, batch size, sequence length, eval cadence, dtype, backend, and output path.
5. Control gate: a matched pure-attention baseline exists under the same manifest.
6. Approximation gate: any deviation from the paper appears in the profile name, diagnostics label, and matrix notes.
7. Stability gate: NaNs, non-finite outputs, broken masks, and invalid routing/halting states fail loudly.

## Required Contract Tests By Baseline

| Baseline | Required Tests |
| --- | --- |
| MoDA | Same-token depth visibility; no cross-token depth leakage; sequence and depth logits in one softmax; causal sequence mask preserved; optional FFN-side depth KV appends slots without changing sequence KV. |
| MoD | Full-sequence top-C train routing; selected tokens sorted back into original order; skipped tokens are identity; selected update is router-weighted; selected attention excludes skipped tokens. |
| NSA | Compressed branch uses learned compression; selected branch derives choices from compressed attention scores; local branch respects window; branches use separate softmaxes and gated sum. |
| Fixed Looped LM | Same block parameters reused each loop; embedding applied once; output head applied once after final loop; loop count changes effective depth but not parameter count. |
| Input-Injected Loop | Initial loop state contract; original prompt/embedding injected every loop; additive input injection and Huginn-style concat adapter are separate profiles. |
| Universal Transformer | Step/time coordinate embedding applied every step; block parameters shared; decoder mask remains causal if decoder profile is used. |
| UT/ACT | Per-token halt probability; remainders; update counts; weighted interpolation; ACT loss/ponder term separated from LM loss. |
| Recurrent-Depth Dynamics | Group layout is explicit; sampled train depth schedule reproducible; step norm, cosine, drift, and acceleration are logged per group. |
| Ouro | `exit_pdf` sums to one; final step gets survival mass; expected CE is probability-weighted; entropy term sign is correct; Q-exit chooses first CDF crossing. |
| RRT | CYCLE layer mapping; true shared parameter objects; strict recursion shares norms; absolute-depth cache indexing; LoRA rank 0 equals strict recursion. |
| MoR | Middle-Cycle sharing; expert-choice capacity schedule; hierarchical filtering; selected tokens sorted; unselected tokens halt; router aux losses included. |

## Required Diagnostics By Baseline

| Baseline | Diagnostics |
| --- | --- |
| MoDA | Depth slot count, same-token depth mask assertion, depth attention probability mass, sequence attention probability mass, FFN depth slot count if enabled. |
| MoD | Selected token count/fraction by block, router score stats, selected gate stats, skipped-token count, selected-only attention token count. |
| NSA | Branch gate means, compressed block count, selected block histograms, local window size, branch FLOP estimates. |
| Fixed Looped LM | Loop count, shared parameter count, effective depth, per-loop hidden norms. |
| Input-Injected Loop | Injection norm/gate, prompt-injection mode, loop-state norm, adapter norm if Huginn-style. |
| Universal Transformer | Step count, coordinate mode, per-step hidden norm. |
| UT/ACT | Average updates, halt histogram, remainder stats, ponder/ACT loss. |
| Recurrent-Depth Dynamics | Average recurrent steps, step norm history, cosine history, drift, acceleration, exit count. |
| Ouro | Exit probabilities, CDF crossing step, average exit depth, entropy, per-step CE. |
| RRT | Stored layer count, effective layer count, shared group map, absolute-depth cache keys, LoRA rank if enabled. |
| MoR | Token-depth histogram, active tokens per recursion, router aux loss, selected gate stats, KV policy. |

## Benchmark Manifests To Add

The harness should create one manifest per baseline family. Each manifest should name both the paper-faithful profile and the approximation profile if both exist.

| Manifest | Purpose |
| --- | --- |
| `path1-core-baseline-attention-control` | Pure attention reference. |
| `path1-core-baseline-moda-paper-ref` | Slow paper-faithful MoDA reference. |
| `path1-core-baseline-mod-paper-train` | Paper-faithful MoD training-time top-C. |
| `path1-core-baseline-mod-causal-decode` | Causal MoD approximation control. |
| `path1-core-baseline-nsa-ref` | Slow NSA reference. |
| `path1-core-baseline-looped-lm` | Fixed looped-block LM. |
| `path1-core-baseline-input-injected-loop` | Additive prompt injection and Huginn-style adapter variants. |
| `path1-core-baseline-ut-act` | UT fixed and ACT variants. |
| `path1-core-baseline-recurrent-dynamics` | Recurrent-depth group dynamics with sampled/fixed depths. |
| `path1-core-baseline-ouro-exit` | Ouro exit distribution and Q-exit. |
| `path1-core-baseline-rrt-cycle` | Strict CYCLE sharing and later relaxed LoRA. |
| `path1-core-baseline-mor-expert-choice` | MoR expert-choice routing. |

## Implementation Progress 2026-04-20

This pass completed the first three items in the immediate build order at smoke-test depth:

- Existing approximations now expose stronger diagnostics:
  - MoDA approximation reports sequence/depth attention mass, memory scope, joint-softmax status, and same-token flag.
  - Causal MoD approximation reports skipped-token count, selected attention token count, attention scope, and decode-safety flag.
  - Parcae recurrence reports step cosine, acceleration, and drift histories.
- New paper-faithful reference lanes:
  - `paper-moda-depth-kv`: slow attention-side MoDA reference using current sequence KV plus same-token prior-depth sequence KV in one softmax.
  - `mod-train-topc-block`: training-time MoD reference using full-sequence top-C capacity, selected-only attention, router-gated selected updates, and identity skipped tokens.
- Guardrails:
  - Paper MoDA is currently restricted to the standard scaffold until a recurrent-depth memory contract exists.
  - Token block routing is currently restricted to standard attention so MoD and MoDA semantics do not silently collide.
  - Looped-transformer scaffolds currently reject MoDA, MoD routing, shared-attention schedules, and non-standard FFNs to keep this tranche primitive-only.
  - Universal Transformer scaffolds use the same primitive-only guardrails and keep ACT separate from Parcae acceleration halting.
  - Ouro learned-exit uses the same primitive-only guardrails and keeps Stage 1 expected-CE training separate from the deferred Stage 2 gate-training protocol.
  - RRT CYCLE uses the same primitive-only guardrails and requires `total_layers % recursion_count == 0`.
  - MoR expert-choice uses the same primitive-only guardrails, requires unique first/middle/last layers, and is labeled as a non-decode-safe full-sequence routing primitive.

Smoke reports written under `artifacts/core-baseline-smoke`:

| Variant | Params | Initial Loss | Final Loss | Train Tok/s | Overall Tok/s | Peak RSS Bytes | Key Diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `attention-only` | 14,896 | 6.1290 | 6.1241 | 847.26 | 2174.50 | 290,635,776 | control |
| `attention-only-moda-depth-kv-depthmem2` | 16,627 | 5.6109 | 5.6083 | 745.56 | 1718.51 | 292,257,792 | approx depth mass 0.1794 |
| `attention-only-paper-moda-depth-kv-depthmem2` | 14,896 | 6.0765 | 6.0715 | 433.51 | 581.80 | 291,504,128 | same-token depth mass 0.3529 |
| `attention-only-causal-topk-block-route50pct-layers1` | 14,945 | 5.8304 | 5.8398 | 855.52 | 1918.71 | 292,798,464 | selected fraction 0.625 |
| `attention-only-mod-train-topc-block-route50pct-layers1` | 14,945 | 5.8515 | 5.8426 | 841.55 | 2050.53 | 292,847,616 | selected fraction 0.500 |
| `attention-only-parcae-looped-attention-loops3-halt-acceleration-min2-t0p6` | 21,648 | 5.5575 | 5.5640 | 683.36 | 1410.80 | 292,257,792 | average steps 2.0 |
| `attention-only-fixed-looped-lm-loops3` | 12,672 | 5.9443 | 5.9223 | 407.83 | 582.96 | 291,160,064 | effective layers 6, injection none |
| `attention-only-looped-additive-input-loops3` | 12,672 | 7.0606 | 7.0379 | 797.03 | 1906.01 | 291,045,376 | effective layers 6, additive injection |
| `attention-only-huginn-adapter-recurrence-loops3` | 13,264 | 5.6514 | 5.6422 | 740.13 | 1703.20 | 291,225,600 | effective layers 6, concat adapter |
| `attention-only-universal-transformer-loops3` | 10,448 | 7.5362 | 7.4211 | 869.21 | 1959.32 | 291,553,280 | effective layers 3, coordinate signals |
| `attention-only-universal-transformer-act-loops3` | 10,465 | 7.0140 | 6.9636 | 737.55 | 1530.89 | 292,323,328 | mean updates 2.0, ponder loss 0.0233 |
| `attention-only-ouro-learned-exit-loops3` | 25,025 | 5.6875 | 5.6571 | 333.39 | 430.38 | 293,470,208 | Q-exit mean 1.5, expected exit step 1.77, entropy 0.9932 |
| `attention-only-rrt-cycle-loops2` | 33,536 | 5.3690 | 5.3259 | 744.72 | 1861.07 | 291,520,512 | stored layers 2, effective layers 4, cycle 0-1-0-1 |
| `attention-only-mor-expert-choice-loops3-route50pct` | 42,179 | 5.5118 | 5.4958 | 689.87 | 1348.33 | 293,437,440 | active counts 8-4-2, selected counts 4-2-1 |

These are reachability smokes only: 1 train step, 1 eval batch, CPU fp32, 8-token sequences, tiny shape. They prove harness integration and diagnostics, not model quality.

Matched local reports written under `artifacts/core-baseline-matched-v1`:

Common contract: CPU fp32, seed 42, byte-level FineWeb stage0 local corpus, `d_model=32`, `head_count=4`, `total_layers=4`, `ffn_multiplier=2`, `seq_len=16`, batch size 1, 8 train steps, 2 eval batches, 1 eval warmup batch, and 1 train warmup step. This is still local proving-ground evidence, not a promotion-scale run.

| Lane | Params | Initial Loss | Final Loss | Train Tok/s | Overall Tok/s | Key Diagnostic |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `attention-control` | 50,624 | 6.0928 | 5.5396 | 8510.47 | 11217.57 | control |
| `moda-paper-depth-kv` | 50,624 | 6.0473 | 5.5186 | 6985.25 | 8904.73 | paper depth attention |
| `mod-train-topc` | 50,721 | 5.7591 | 5.4112 | 7931.63 | 10272.55 | full-sequence top-C routing |
| `mod-causal-decode-control` | 50,721 | 5.7620 | 5.4022 | 7366.54 | 9467.55 | causal prefix-top-k routing |
| `parcae-fixed-recurrent-depth` | 50,784 | 5.5723 | 4.9238 | 5692.78 | 7258.04 | avg steps 3.0 |
| `parcae-acceleration-exit` | 50,784 | 5.6006 | 4.9895 | 6408.88 | 8233.89 | avg steps 2.0 |
| `fixed-looped-lm` | 50,624 | 6.7478 | 5.2168 | 3981.04 | 5027.94 | effective layers 12 |
| `looped-additive-input` | 50,624 | 8.5810 | 6.7320 | 4251.44 | 5388.84 | effective layers 12 |
| `huginn-adapter-recurrence` | 52,832 | 5.7638 | 5.2618 | 3660.91 | 4644.18 | effective layers 12 |
| `universal-transformer` | 50,624 | 10.0554 | 5.5587 | 3999.99 | 5004.66 | effective layers 12 |
| `universal-transformer-act` | 50,657 | 7.7548 | 5.3591 | 3601.17 | 4524.05 | ACT mean updates 2.0 |
| `ouro-stage1-learned-exit` | 50,657 | 6.0225 | 5.3300 | 3844.44 | 4822.84 | q-exit 1.19, expected exit 1.44 |
| `rrt-cycle` | 33,536 | 5.7559 | 5.3830 | 9666.21 | 12241.58 | stored/effective layers 2/4 |
| `mor-expert-choice` | 50,723 | 5.7328 | 5.3968 | 4603.90 | 5874.92 | active 16-8-4, selected 8-4-2 |

Artifacts:

- Suite manifest: `artifacts/core-baseline-matched-v1/suite_manifest.json`
- Ledger: `artifacts/core-baseline-matched-v1/ledger.jsonl`
- Summary: `artifacts/core-baseline-matched-v1/summary.md`

## Immediate Build Order

Use this order to avoid combinatorial chaos:

1. Done at smoke-test depth: add missing diagnostics/tests for existing approximations: MoDA approximation, MoD causal routing, Parcae recurrence.
2. Done at slow-reference depth: add paper-faithful MoDA slow reference. This corrects the largest current naming/fidelity gap.
3. Done at smoke-test depth: add paper-faithful MoD train-time top-C as a separate lane from causal decode.
4. Done at smoke-test depth: add strict fixed looped LM and input-injected loop profiles.
5. Done at smoke-test depth: add UT fixed recurrence, then ACT.
6. Done at smoke-test depth: add Ouro learned exit after fixed looped LM diagnostics are stable.
7. Done at smoke-test depth: add RRT strict CYCLE sharing before LoRA.
8. Done at smoke-test depth: add MoR expert-choice after RRT and MoD contracts exist.
9. Conditional next: add NSA when the experiment phase moves to longer context or when attention efficiency becomes the primary bottleneck.

## Current Bottom Line

The repo now has contract-tested, harness-reachable reference lanes for attention-side MoDA, train-time MoD, strict fixed looped LM, additive input-injected loops, a Huginn-style adapter recurrence surface, Universal Transformer fixed recurrence, UT/ACT, Ouro Stage 1 learned exit, strict RRT CYCLE sharing, and MoR expert-choice recurrence. It is still not ready for evidence-based promotion across the full core baseline set because the benchmark manifests are smoke-level, Ouro Stage 2 gate training is deferred, RRT relaxed LoRA is deferred, MoR token-choice and KV-sharing variants are deferred, NSA is absent, and recurrent-depth dynamics are still represented by a Parcae approximation plus deterministic recurrent references rather than the full sampled-depth paper primitive. The next tranche should be matched manifests for the completed core primitives, with NSA held until longer-context efficiency matters.
