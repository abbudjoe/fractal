# v3A Hybrid Attention Plan

## Purpose

Path 1 asks one focused question:

* can we establish a working Rust hybrid-attention baseline that plays the same
  role as modern Mamba-style hybrids, then evaluate our primitives against that
  frozen in-repo baseline?

This is not a memory/index experiment.
This is a predictive-core experiment.

---

## Thesis

Attention is still the strongest known exact token-interaction primitive.

The immediate problem is no longer “can our primitives beat Mamba?”

The immediate problem is:

* **can we build and validate a real Rust hybrid-attention baseline first**

Only after that baseline exists do we ask whether our primitives can serve the
same role that Mamba-2, Mamba-3, or related state-space blocks serve in models
like Jamba or Nemotron-H:

* cheaper background sequence processing
* better efficiency at fixed predictive quality
* reduced reliance on dense exact attention everywhere

So Path 1 now has two claims, in order:

1. **we can build a validated Rust Mamba-style hybrid baseline in this repo and
   environment**
2. **our primitives can later compete with that baseline on a matched budget**

---

## Scope

Included:

* causal self-attention
* local-window or other fixed exact-attention hot path
* Mamba-style reference hybrids for comparison
* our recurrent/selective primitives inside the predictive stack, but only
  after the reference baseline is real
* normal LM head prediction

Excluded:

* fractal tree retrieval
* sealed-leaf memory/index sidecars
* routed remote token gathering
* direct memory-to-logit fusion
* external memory writes as part of the proving version

Path 1 must stand on predictive-core merit alone.

---

## Fixed Discipline

Path 1 must compare architecture families fairly.

Hold fixed unless an ablation explicitly changes them:

* tokenizer
* corpus slice
* training budget
* parameter budget or active-parameter budget
* model width and depth envelope
* local exact-attention window
* evaluation suites
* precision / quantization regime for a given proving round

Do not let one variant win by silently receiving:

* more exact-attention layers
* a larger local window
* a larger parameter budget
* easier benchmarks

Do not let our primitive line define the benchmark it is supposed to beat.

The reference hybrid must become real first.

---

## RunPod Benchmark Gate

RunPod benchmarking is **not** currently allowed for the Rust Mamba-3 baseline.

We now have a credible Rust baseline for local CPU and Metal development, but
that does **not** yet close the CUDA gap.

Before any Path 1 benchmark is run on RunPod or any other CUDA-only surface, we
must first:

* implement a CUDA execution path for the Rust reference lane
* validate CUDA correctness against the existing parity ladder
* run at least one seeded CUDA smoke benchmark on the shared Path 1 surface
* confirm report/schema parity with the local CPU/Metal control plane

Until those are done, RunPod results would mix architecture questions with a
missing-backend question, and that would break the apples-to-apples discipline
for Path 1.

---

## Baseline Ladder

The ladder must be run in order.

1. attention-only baseline
   A clean decoder-style baseline with exact causal self-attention and no
   fractal retrieval.

2. attention + reference SSM hybrid in Rust
   A fair comparison line using a strong modern state-space reference such as a
   Mamba-3-style hybrid schedule, implemented in this environment and validated
   on the shared Path 1 surface.

3. freeze the reference hybrid as the Path 1 baseline
   Once step `2` is real and validated, it becomes the baseline our contender
   must beat.

4. attention + our primitive hybrid
   Replace the reference SSM contribution with our primitive-based sequence
   mixing path under the same budget envelope.

5. schedule ablations inside our primitive line
   Only after step `4` exists:
   * layer placement
   * ratio of exact-attention to primitive layers
   * projection width
   * recurrence/state update form

No step `5` before steps `1` through `4` are all real.

### Phase-1 matched baseline matrix

The first concrete matrix is fixed as the build order for Path 1:

1. `attention-only`
   * `8` exact-attention layers
   * no SSM or primitive layers

2. `reference-ssm-hybrid`
   * `8` total layers
   * interleaved `A-S-A-S-A-S-A-S`
   * `S` is a Mamba-family reference block
   * first executable pass may use a typed proxy lane for bring-up
   * the real Path 1 baseline must graduate to a faithful Rust Mamba-3-style
     implementation before contender claims are taken seriously

3. `primitive-hybrid`
   * `8` total layers
   * interleaved `A-P-A-P-A-P-A-P`
   * `P` is our primitive sequence-mixing block

All three variants must share:

* hidden width
* head count
* local attention window
* training budget
* eval suites

For the first proving round, the typed control plane for this matrix lives in:

* `fractal-core/src/hybrid_attention/`
* `fractal-eval-private/src/hybrid_attention.rs`
* planned runner: `src/bin/v3a-hybrid-attention-matrix.rs`

The concrete baseline-build checklist lives in:

* [`v3a-rust-mamba-baseline-checklist.md`](./v3a-rust-mamba-baseline-checklist.md)

---

## Initial Control Plane

The first proving version should stay narrow.

Recommended defaults:

* single predictive stream
* no MoE in the first pass
* local exact attention only
* periodic hybrid blocks rather than a fully novel stack
* no external memory
* no multi-root recurrence

This keeps the question sharp:

* can we build a trustworthy Rust hybrid baseline first?
* and then, is our primitive useful inside that hybrid attention backbone?

---

## Required Metrics

Measure both predictive quality and systems value.

Predictive quality:

* train loss
* eval loss
* perplexity
* `MQAR`
* copy
* induction
* retrieval-heavy probes that do not depend on external memory

Systems value:

* training throughput
* decode throughput
* KV-cache size
* activation memory
* effective exact-attention fraction

Hybrid-specific diagnostics:

* token-to-token comparison quality
* copy/retrieval preservation relative to attention-only baseline
* stability under longer context

---

## Success Criteria

Path 1 succeeds only if all of the following are true:

Phase 1A succeeds only if all of the following are true:

* the Rust reference SSM hybrid is stable and reproducible
* it matches or beats the attention-only baseline on core predictive quality at
  the chosen budget
* it gives us a credible in-repo Path 1 benchmark surface for later contender
  work

Phase 1B succeeds only if all of the following are true:

* our primitive hybrid matches or beats the attention-only baseline on core
  predictive quality at the chosen budget
* our primitive hybrid is competitive with or better than the frozen Rust
  reference SSM hybrid on the same budget
* efficiency improves in a way that matters:
  * less memory
  * faster decode
  * or meaningfully less exact-attention usage
* copy, induction, and comparison behavior remain healthy

If predictive quality drops while only efficiency improves, Path 1 has not yet
validated.

---

## Failure Criteria

Phase 1A fails if:

* we cannot produce a stable Rust reference hybrid with a coherent training and
  evaluation story
* the reference lane remains a toy proxy without a credible path to a faithful
  implementation

Phase 1B fails if:

* our primitive hybrid underperforms the attention-only baseline on core
  predictive quality
* our primitive hybrid is clearly worse than the frozen Rust reference SSM
  hybrid
* efficiency gains are only marginal
* copy and comparison behavior degrade materially

At that point, do not rescue Path 1 by importing Path 2 memory features.
That would break the path split.

---

## Relation To Mamba

Path 1 treats models like Jamba and Nemotron-H as the relevant comparison
family:

* Jamba-style interleaved attention + SSM hybrids
* Nemotron-H-style mostly-SSM backbones with occasional attention

The goal is not to imitate them exactly at first.
The goal is to build a faithful-enough Rust reference for their SSM role, then
beat that role using our own primitives.

---

## Immediate Build Order

1. define the attention-only baseline
2. define the reference SSM-hybrid baseline
3. validate the Rust reference lane and freeze it as the Path 1 benchmark
4. define the primitive-hybrid contender
5. run the matched contender matrix
6. only then decide whether Path 1 deserves deeper investment
