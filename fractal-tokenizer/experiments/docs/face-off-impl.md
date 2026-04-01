# Face-Off Implementation Plan

## Objective

Implement the minimum clean infrastructure needed to run a fair tokenizer face-off between:

- a flat baseline tokenizer
- `p1_fractal_hybrid_dyn-state-norm_v2`

The plan is intentionally staged so we prove the tokenizer contract before spending GPU time on model training.

## Scope

In scope:

- stable vocabulary induction for `v2`
- deterministic encode/decode contract
- fallback instrumentation
- shared raw-text dataset pipeline
- small decoder-only training harness
- evaluation harness normalized by raw text

Out of scope for the first implementation:

- large-model full fine-tuning
- speculative new tokenizer levers
- span-length embedding experiments
- production deployment plumbing

## Directory Plan

Create and keep all face-off work inside the tokenizer crate:

- `fractal-tokenizer/experiments/docs/`
- `fractal-tokenizer/experiments/data/`
- `fractal-tokenizer/experiments/results/`
- `fractal-tokenizer/src/faceoff/`

Suggested source layout:

- `fractal-tokenizer/src/faceoff/mod.rs`
- `fractal-tokenizer/src/faceoff/vocab.rs`
- `fractal-tokenizer/src/faceoff/encode.rs`
- `fractal-tokenizer/src/faceoff/decode.rs`
- `fractal-tokenizer/src/faceoff/fallback.rs`
- `fractal-tokenizer/src/faceoff/dataset.rs`
- `fractal-tokenizer/src/faceoff/metrics.rs`

Suggested scripts:

- `fractal-tokenizer/experiments/train_small_lm.py`
- `fractal-tokenizer/experiments/eval_faceoff.py`

## Milestone 1: Stable Vocabulary

### Deliverable

A deterministic vocab builder for `p1_fractal_hybrid_dyn-state-norm_v2`.

### Tasks

1. Define a typed vocab entry:
   - token ID
   - motif digest
   - canonical depth
   - stats needed for analysis only
2. Build vocab from train split only.
3. Persist vocab in a simple deterministic format:
   - JSONL or compact JSON
4. Ensure repeated runs with identical input produce identical IDs.

### Acceptance Checks

- same train split -> same vocab file
- no validation/test leakage
- stable ID assignment confirmed by regression test

## Milestone 2: Encode/Decode Contract

### Deliverable

Model-facing tokenizer API with exact round-trip.

### Tasks

1. Encode raw text into:
   - token IDs
   - depth IDs
   - spans
2. Decode back into raw text.
3. Add deterministic fallback:
   - unseen motif -> recurse deeper
   - unresolved terminal -> byte token
4. Track fallback reasons and counts.

### Acceptance Checks

- exact round-trip on stress, mixed-domain, and streaming corpora
- fallback path covered by regression tests
- no `.unwrap()` outside tests

## Milestone 3: Dataset Pipeline

### Deliverable

One shared raw-text dataset pipeline consumed by both tokenizer arms.

### Tasks

1. Define dataset manifest format.
2. Build train/validation/test splits by raw text.
3. Tag records with bucket:
   - repetition-heavy
   - mixed-domain
   - code-like
4. Materialize tokenized views for:
   - baseline tokenizer
   - fractal tokenizer

### Acceptance Checks

- splits are identical across tokenizers
- bucket labels are preserved
- tokenization artifacts are reproducible

## Milestone 4: Tokenizer-Only Evaluation

### Deliverable

A lightweight tokenizer-only report before model training.

### Tasks

1. Measure:
   - token count
   - avg chars/token
   - fallback rate
   - motif reuse selectivity
   - round-trip success
2. Compare by bucket.
3. Save results under `experiments/results/`.

### Acceptance Checks

- repetition-heavy bucket shows elevated motif reuse
- mixed-domain bucket stays near zero motif reuse
- no bucket fails round-trip

## Milestone 5: Small LM Training Harness

### Deliverable

A shared training harness that can train the same small decoder model on both tokenizers.

### Tasks

1. Pick framework:
   - PyTorch + Transformers-style custom decoder
   - or a minimal native training loop
2. Define config file for:
   - model size
   - context length
   - optimizer
   - budget regime
3. Add tokenizer adapters:
   - baseline token stream
   - fractal token IDs + depth IDs

### Acceptance Checks

- both arms train with identical model hyperparameters
- loss decreases normally
- checkpoint save/load works

## Milestone 6: Evaluation Harness

### Deliverable

A raw-text-normalized evaluation report.

### Tasks

1. Compute:
   - bits-per-byte
   - chars/sec
   - effective chars/context window
   - fallback rate
   - motif reuse selectivity
2. Report by bucket and overall.
3. Compare fixed-compute and fixed-raw-bytes regimes.

### Acceptance Checks

- metrics are computed identically for both arms
- raw-text normalization is explicit in code and report output
- result table is reproducible from saved checkpoints

## Milestone 7: Optional OSS Bridge

### Deliverable

An optional adapter experiment against a frozen OSS model.

### Tasks

1. Keep the OSS base model frozen.
2. Learn a small projection from fractal token embeddings into the model input space.
3. Compare against native tokenizer on the same evaluation buckets.

### Acceptance Checks

- adapter training is isolated from tokenizer-contract work
- no changes to the core primitive are required

## Regression Tests To Add

At minimum:

- vocab determinism test
- train-only vocab induction test
- unknown motif fallback test
- exact round-trip test for encoded ID stream
- mixed-domain false-positive reuse test
- repetition-heavy positive reuse test

## First Execution Slice

The smallest meaningful implementation slice is:

1. add stable vocab builder
2. add encode/decode contract with fallback
3. add tokenizer-only evaluation report

Do not start model training until that slice passes.

## Compute Plan

Start small:

- tokenizer-only eval on local CPU/GPU
- small-model pilot on a single cloud GPU
- full face-off only after contract stability is proven

Recommended first training budget:

- one small model
- one baseline tokenizer
- one fractal tokenizer
- one fixed-compute run each
- one fixed-raw-bytes run each

## Decision Gates

Proceed to the next milestone only if:

- round-trip remains exact
- fallback rate stays manageable
- mixed-domain false-positive reuse stays near zero
- tokenizer-only compression advantage survives the stable-ID contract

If any of those fail, pause and fix the contract before adding more machinery.

## Definition Of Done

The face-off is complete when we can publish one table that shows, for both tokenizer arms:

- bits-per-byte
- chars/sec
- effective chars/context
- fallback rate
- motif reuse selectivity
- round-trip success

across repetition-heavy, mixed-domain, and code-like test buckets under both fairness regimes.
