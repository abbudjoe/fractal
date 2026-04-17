# Parcae P20-Control H100 Proof Ladder

Date: 2026-04-16

This note records the first bounded H100 proof ladder for the Parcae/P20-control lane on the 750M-token OpenLLaMA-tokenized FineWeb cache.

## Contract

- Hardware: Modal 1x NVIDIA H100 80GB HBM3.
- Torch/CUDA: Torch 2.10.0+cu128, CUDA 12.8.
- Corpus: `fineweb-cc-main-2024-10-openllama-750m`.
- Train tokens in cache: 750,000,544.
- Eval tokens in cache: 10,029,403.
- Context: 256 tokens.
- Batch size: 64.
- Eval batches: 64.
- Dtype: bf16.
- Runtime backend: Triton primitive backend.
- Loop count: 2.

Each 30k-step lane sees 491,520,000 train tokens, about 65.5% of the train split, so these runs stay below one pass through the train cache.

## Initial Four-Lane Promotion

Run label: `modal-h100-750m-parcae-promotion-s42-steps30000-bs64`

| Lane | Params | Final Loss | tok/s | Peak CUDA MB |
|---|---:|---:|---:|---:|
| attention-only | 9,778,176 | 5.9347 | 1,067,936.77 | 6,774.79 |
| parcae-looped-attention | 9,778,816 | 4.4132 | 878,842.40 | 6,983.55 |
| parcae-bx-looped-attention | 9,811,712 | 4.3842 | 866,259.73 | 6,991.98 |
| parcae-p20-control-looped-attention | 9,873,856 | 4.3012 | 793,614.94 | 7,046.88 |

Interpretation: plain looped middle-depth accounts for most of the improvement over the 8-layer attention baseline. B(x) improves the looped scaffold slightly. P20-control improves the same looped scaffold further, at a throughput cost.

## Equal-Step Finalist Repeat

This is the first clean claim test for P20-control: same data budget, same optimizer/settings, same scaffold class, comparing the strongest non-P20 looped control against P20-control across three seeds.

| Seed | B(x) Loss | P20-Control Loss | P20 Delta | B(x) tok/s | P20 tok/s | P20 tok/s % |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4.3842 | 4.3012 | -0.0830 | 866,260 | 793,615 | 91.6% |
| 43 | 4.4658 | 4.2972 | -0.1687 | 908,138 | 832,650 | 91.7% |
| 44 | 4.3518 | 4.2965 | -0.0553 | 863,253 | 792,434 | 91.8% |

Summary:

| Lane | Mean Loss | Loss Stdev | Mean tok/s | Mean Train Min |
|---|---:|---:|---:|---:|
| B(x), 30k steps | 4.4006 | 0.0588 | 879,217 | 9.32 |
| P20-control, 30k steps | 4.2983 | 0.0026 | 806,233 | 10.17 |

Mean P20 delta vs B(x): -0.1023 loss.

Mean perplexity ratio from the mean delta: 0.9027.

Claim supported: P20-control beat the strongest non-P20 looped scaffold across seeds at equal step/token budget.

## Speed-Compensated B(x) Pressure Check

This check gives B(x) extra steps to spend its speed advantage against the existing P20-control 30k-step runs. It is a speed-compensated pressure check rather than an exact stopwatch-identical comparison; actual H100 lane throughput varied between runs.

| Seed | B(x) Extra Steps | B(x) Loss | P20-Control 30k Loss | P20 Delta | B(x) Train Min | P20 Train Min |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 32,750 | 4.4223 | 4.3012 | -0.1211 | 9.39 | 10.32 |
| 43 | 32,725 | 4.5352 | 4.2972 | -0.2380 | 9.45 | 9.84 |
| 44 | 32,685 | 4.4055 | 4.2965 | -0.1090 | 10.33 | 10.34 |

Summary:

| Lane | Mean Loss | Loss Stdev | Mean tok/s | Mean Train Min |
|---|---:|---:|---:|---:|
| B(x), speed-compensated extra steps | 4.4543 | 0.0705 | 920,533 | 9.72 |
| P20-control, 30k steps | 4.2983 | 0.0026 | 806,233 | 10.17 |

Mean P20 delta vs speed-compensated B(x): -0.1560 loss.

Mean perplexity ratio from the mean delta: 0.8555.

Claim supported: P20-control survived the first speed-compensated pressure check against B(x). The faster B(x) control did not catch P20-control when given extra steps.

## Tuned Non-P20 Control Check

This check gives the non-P20 looped controls a small learning-rate sweep on seed 42 while keeping the training budget fixed. P20-control remains fixed at the incumbent LR `1e-3`.

Seed 42 LR sweep:

| LR | Parcae Looped Loss | B(x) Looped Loss |
|---:|---:|---:|
| 3e-4 | 4.5403 | 4.5320 |
| 6e-4 | 4.3712 | 4.3890 |
| 1e-3 | 4.4132 | 4.3842 |
| 1.5e-3 | 7.7104 | 5.7993 |

The best non-P20 setting was plain Parcae looped attention at LR `6e-4`. That tuned control was then repeated across seeds.

| Seed | Tuned Parcae Looped LR 6e-4 | P20-Control LR 1e-3 | P20 Delta | Tuned tok/s | P20 tok/s |
|---:|---:|---:|---:|---:|---:|
| 42 | 4.3712 | 4.3012 | -0.0700 | 935,472 | 793,615 |
| 43 | 4.3771 | 4.2972 | -0.0800 | 913,677 | 832,650 |
| 44 | 4.3781 | 4.2965 | -0.0816 | 856,470 | 792,434 |

Summary:

| Lane | Mean Loss | Loss Stdev | Mean tok/s |
|---|---:|---:|---:|
| Tuned Parcae looped, LR 6e-4 | 4.3755 | 0.0037 | 901,873 |
| P20-control, LR 1e-3 | 4.2983 | 0.0026 | 806,233 |

Mean P20 delta vs tuned non-P20 control: -0.0772 loss.

Mean perplexity ratio from the mean delta: 0.9257.

Claim supported: P20-control survives the first tuned-control rung against the best non-P20 looped control from the bounded LR sweep.

## Stronger Attention Depth Check

This rung tests whether a stronger pure-attention model catches the P20-control lane. The attention controls keep the same width, heads, context, corpus, and batch size, but vary physical layer depth. These are not all same-size controls; 12L, 14L, and 16L attention use materially more parameters and compute than the 8L P20-control lane.

Seed 42 attention depth screen:

| Lane | Params | Steps | Final Loss | tok/s | Train Min | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|
| 8L attention, LR 1e-3 | 9,778,176 | 30,000 | 5.9347 | 1,067,937 | 7.67 | 6,774.79 |
| 9L attention, LR 6e-4 | 9,976,448 | 30,000 | 4.4459 | 1,091,606 | 7.50 | 6,849.93 |
| 10L attention, LR 6e-4 | 10,174,720 | 30,000 | 4.3649 | 981,974 | 8.34 | 6,925.08 |
| 12L attention, LR 6e-4 | 10,571,264 | 30,000 | 4.3304 | 897,467 | 9.13 | 7,075.37 |
| 14L attention, LR 6e-4 | 10,967,808 | 30,000 | 4.3044 | 773,922 | 10.59 | 7,225.28 |
| 16L attention, LR 6e-4 | 11,364,352 | 30,000 | 4.2758 | 721,719 | 11.35 | 7,375.57 |
| P20-control, LR 1e-3 | 9,873,856 | 30,000 | 4.3012 | 793,615 | 10.32 | 7,046.88 |

Interpretation:

- P20-control beats 9L, 10L, and 12L pure attention on this seed.
- P20-control narrowly beats 14L pure attention on this seed, while using fewer parameters and slightly less memory.
- 16L pure attention beats P20-control at equal steps, but uses about 15% more parameters, more memory, and lower throughput.

14L boundary repeat:

| Seed | 14L Attention Loss | P20-Control Loss | P20 Delta | 14L tok/s | P20 tok/s |
|---:|---:|---:|---:|---:|---:|
| 42 | 4.3044 | 4.3012 | -0.0032 | 773,922 | 793,615 |
| 43 | 4.2960 | 4.2972 | +0.0011 | 772,807 | 832,650 |
| 44 | 4.3061 | 4.2965 | -0.0097 | 847,395 | 792,434 |

Summary:

| Lane | Mean Loss | Loss Stdev | Mean tok/s | Mean Train Min | Params |
|---|---:|---:|---:|---:|---:|
| 14L attention, LR 6e-4 | 4.3022 | 0.0054 | 798,041 | 10.28 | 10,967,808 |
| P20-control, LR 1e-3 | 4.2983 | 0.0026 | 806,233 | 10.17 | 9,873,856 |

Mean P20 delta vs 14L attention: -0.0039 loss.

Mean perplexity ratio from the mean delta: 0.9961.

Interpretation: 14L attention is the current same-budget boundary. P20-control has a tiny mean edge while using about 10% fewer parameters, but the margin is small enough that this should be called a tie-band, not a decisive quality win.

P20 speed-spend pressure against 16L attention:

| Lane | Params | Steps | Final Loss | tok/s | Train Min | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|
| 16L attention, LR 6e-4 | 11,364,352 | 30,000 | 4.2758 | 721,719 | 11.35 | 7,375.57 |
| P20-control, LR 1e-3 | 9,873,856 | 33,000 | 4.2835 | 820,523 | 10.98 | 7,046.88 |
| P20-control, LR 1e-3 | 9,873,856 | 34,000 | 4.2812 | 835,389 | 11.11 | 7,046.88 |

Interpretation:

- Letting P20-control spend its speed advantage narrows the 16L gap, but did not catch 16L attention on seed 42.
- The 34k-step P20 run was still slightly shorter than the 16L run by train-wall-clock, but the remaining gap is small enough that this should be treated as a boundary, not a decisive universal loss.

Claim supported: P20-control beats parameter-near and compute-near pure attention controls in this screen, is in a tie-band with 14L attention across three seeds, and stays competitive with much deeper attention. Claim not supported: P20-control does not beat arbitrarily deeper attention; 16L attention passes it on seed 42.

## Responsible Claim Boundary

Supported:

- The Parcae looped scaffold beats the plain 8-layer attention baseline in this tiny-LM H100 contract.
- Within the looped scaffold family, P20-control beats B(x) across three seeds at equal step/token budget.
- P20-control survives a first speed-compensated B(x) pressure check.
- P20-control survives a bounded tuned-control check against the best non-P20 looped control found in the LR sweep.
- P20-control beats pure attention up through the 12L control on seed 42 and sits in a three-seed tie-band with 14L attention while using fewer parameters.

Not yet proven:

- P20 generally beats attention.
- P20 beats a tuned Transformer baseline.
- P20 scales to larger LMs.
- P20 wins at longer context.
- P20 wins after broad LR/optimizer tuning of all controls.
- P20 beats much deeper pure-attention baselines; 16L attention beat P20-control on seed 42.

## Local Artifacts

- `.modal-local-logs/modal-results/modal-h100-750m-parcae-promotion-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-finalists-s43-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-finalists-s44-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-bx-equaltime-s42-steps32750-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-bx-equaltime-s43-steps32725-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-bx-equaltime-s44-steps32685-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-controls-s42-lr3e-4-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-controls-s42-lr6e-4-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-controls-s42-lr1p5e-3-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-looped-lr6e-4-s43-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-parcae-looped-lr6e-4-s44-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn9-lr6e-4-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn10-lr6e-4-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn12-lr6e-4-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn14-lr6e-4-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn14-lr6e-4-s43-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn14-lr6e-4-s44-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-attn16-lr6e-4-s42-steps30000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-p20-equaltime-vs-attn16-s42-steps33000-bs64/`
- `.modal-local-logs/modal-results/modal-h100-750m-p20-equaltime-vs-attn16-s42-steps34000-bs64/`

## Next Proof Rung

The next lowest-risk rung is a longer-context or scaling check, but only after deciding which claim to pursue:

- If the goal is a same-size architectural claim, repeat P20-control vs 10L/12L/14L attention across seeds.
- If the goal is a practical quality frontier, compare P20-control against 16L attention at larger token budget and/or longer context.
- If the goal is mechanism understanding, inspect Parcae/P20 diagnostics and loss curves before adding more architecture.

Only after that should this lane move to longer context or larger model size.
