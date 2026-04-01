# Tokenizer Face-Off

## Goal

Determine whether `p1_fractal_hybrid_dyn-state-norm_v2` is a better tokenizer in the model-quality sense, not only a better structural compressor.

The decisive question is:

- can the fractal tokenizer improve or preserve modeling quality while increasing effective context and compression on repetition-heavy text?

## Current Evidence

The tokenizer track already established:

- exact balanced recursive hierarchy: `d0:1, d1:2, d2:4, d3:8, d4:16, d5:32`
- controlled cross-depth motif reuse on repetition-heavy inputs
- zero false-positive motif reuse on mixed-domain inputs
- exact round-trip reconstruction on benchmark inputs
- dramatic compression vs `tiktoken cl100k_base` and native OSS model tokenizers on repeated text

These results are strong tokenizer-track signals, but they do not yet prove improved end-to-end language modeling quality.

## Hypothesis

A self-regulating hierarchical tokenizer can:

- compress repeated structure more effectively than flat subword tokenizers
- preserve modeling quality on diverse natural text
- increase effective raw-text coverage per context window
- avoid spurious reuse on mixed-domain inputs

## Hard Prerequisite: Stable Token Contract

Before any fair model comparison, `v2` must expose a deterministic model-facing token contract.

Required properties:

- vocabulary is induced from the training split only
- every motif digest receives a stable integer token ID
- recursion depth is represented as a separate feature embedding, not fused into the token string
- encoding is deterministic
- decoding is exact
- unknown motifs fall back deterministically:
  - unseen parent motif -> recurse to children
  - unseen child motif -> recurse deeper
  - terminal fallback -> byte-level token path
- fallback rate is logged on train, validation, and test splits

Without this layer, the benchmark would compare a usable tokenizer against a dynamic span analyzer, which is not a fair face-off.

## Experimental Arms

Primary arms:

1. Baseline tokenizer
   - `cl100k_base` or an OSS-native tokenizer with stable IDs
2. Fractal tokenizer
   - `p1_fractal_hybrid_dyn-state-norm_v2`
   - stable vocabulary
   - deterministic fallback
   - depth embedding

Optional ablations after the primary comparison:

- `p1_fractal_hybrid_v1` static baseline
- `v2` without depth embedding
- `v2` with byte fallback only
- `v2` with additional span-length embedding

## Data Design

Build one raw-text corpus with three buckets:

1. Repetition-heavy text
   - repeated paragraphs
   - duplicated boilerplate
   - repeated logs/templates
   - synthetic repeated phrase blocks like the current stress benchmark
2. Mixed-domain natural text
   - news
   - literature
   - encyclopedia or general prose
3. Code-like text
   - source files
   - code comments
   - stack traces
   - config snippets

Rules:

- split on raw text, not tokenized text
- keep train, validation, and test splits identical across tokenizers
- induce tokenizer vocabulary from train split only
- never leak validation/test motifs into training-time vocab construction

Recommended first-pass scale:

- `50M` to `200M` raw characters
- enough to expose real modeling differences while keeping training affordable

## Model Setup

Train the same decoder-only transformer for every tokenizer arm.

Recommended first-pass model:

- `80M` to `150M` parameters
- same layer count, hidden size, heads, optimizer, schedule, dropout, and context length
- identical training code and hardware
- only the tokenizer path changes

For the fractal tokenizer arm:

- token ID embedding
- depth embedding
- span-length embedding is explicitly excluded from the first main comparison and reserved for later ablation

## Fairness Regimes

Run both fairness regimes.

### Fixed Compute Budget

Match:

- optimizer steps
- wall-clock budget
- or approximate FLOPs

Purpose:

- test whether fractal tokenization improves efficiency under the same training budget

### Fixed Raw-Text Budget

Match:

- total raw characters seen

Purpose:

- test whether the model learns more effectively from the same source information

Only running one regime leaves too much room for interpretation.

## Evaluation Metrics

Primary metrics:

- validation bits-per-byte or nats-per-byte
- exact round-trip rate
- effective raw characters covered per context window
- throughput in raw chars/sec
- fallback rate
- motif reuse selectivity
  - high on repetition-heavy evaluation
  - near zero on mixed-domain evaluation
- training stability
  - no divergence
  - no NaN spikes
  - healthy loss curves

Secondary metrics:

- token count
- avg chars/token
- wall-time throughput
- memory usage

Important rule:

- do not treat loss-per-token as the primary comparison metric
- normalize quality to raw text, not tokenizer-specific token boundaries

## Success Criteria

Call `v2` a genuine tokenizer win if it satisfies all of the following:

- matches or beats the baseline on repetition-heavy validation bits-per-byte
- does not materially regress on mixed-domain bits-per-byte
- increases effective raw-text context at fixed sequence length
- preserves exact round-trip
- keeps fallback low enough to be practical
- maintains near-zero false-positive motif reuse on mixed-domain text

## Phased Experiment

### Phase 1: Tokenizer Contract Validation

- build stable vocab from train split
- verify deterministic encoding
- verify exact decoding
- measure fallback rate by bucket
- measure motif reuse selectivity by bucket

### Phase 2: Small-Model Face-Off

- baseline tokenizer under fixed-compute regime
- fractal tokenizer under fixed-compute regime
- baseline tokenizer under fixed-raw-bytes regime
- fractal tokenizer under fixed-raw-bytes regime

### Phase 3: Evaluation

Evaluate all trained checkpoints on:

- repetition-heavy test set
- mixed-domain test set
- code-like test set

### Phase 4: Optional Bridge Experiment

If the small-model study is promising:

- freeze a pretrained OSS model
- learn a small adapter from fractal token stream to model embedding space
- compare against native tokenizer on the same evaluation buckets

This bridge experiment is faster to run, but the from-scratch small-model study is the cleaner scientific proof.

## Artifacts To Save

- raw-text split manifest
- tokenizer vocab file
- fallback statistics
- encode/decode contract test results
- training configs
- training and validation curves
- bits-per-byte tables
- chars/sec tables
- motif reuse selectivity tables
- hardware and software environment snapshot

## Expected Outcome

If the current tokenizer-track signals hold under model training, the likely outcome is:

- strong win on repetition-heavy text
- neutral to mild win on mixed-domain text
- materially larger effective context coverage
- no loss of reversibility

If the fractal tokenizer fails, the most likely failure modes are:

- high fallback rate wipes out the compression advantage
- model struggles to learn from hierarchical token IDs
- compression improves while bits-per-byte worsens on mixed-domain text

That failure would still be informative because it would isolate the missing contract between hierarchical token structure and model consumption.
