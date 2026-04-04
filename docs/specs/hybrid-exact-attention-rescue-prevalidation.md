# Hybrid Exact-Attention Rescue Prevalidation

## Purpose

Before building the full hybrid v3 line, run one narrow falsification test:

**can a tiny exact-attention rescue path recover the retrieval behaviors that
the tree-only v2 line could not?**

This document defines that pre-validation.

It exists to answer a concrete question:

* if the tree memory is used only to narrow candidates
* and a real exact token-to-token attention step is restored over those
  candidates
* does the architecture immediately recover copy and retrieval behavior?

If yes, then the next serious line should be a hybrid.
If no, then the problem is deeper than the missing exact-interaction primitive.

---

## Why this test matters

The v2 line already falsified several simpler explanations:

* router leaf selection alone did not rescue the probe tasks
* oracle exact-token selection alone did not rescue the probe tasks
* supervised synthetic retrieval did not rescue held-out behavior
* multi-root specialization was not the main blocker
* direct read-fusion to logits was part of the problem, but fixing that did
  not save the thesis

So the highest-probability remaining explanation is:

**the architecture is missing a true exact token-interaction primitive in the
active predictive path**

This rescue experiment tests exactly that and nothing broader.

---

## Working hypothesis

The tree system may still be useful if it is demoted to:

* long-range candidate generator
* cheap searchable memory index

and exact attention is restored as:

* the final binder between the active token and retrieved evidence

In other words:

* tree memory proposes
* exact attention decides

That is the whole point of the test.

---

## Minimal rescue architecture

Add a tiny exact-attention rescue block near the top of the stack.

At each token step:

1. the existing local trunk processes the recent sequence
2. the existing tree router proposes the top `8` distant sealed spans
3. the model gathers raw token-state snapshots from those routed spans
4. the active hidden state runs exact causal attention over:
   * a local recent window
   * the gathered remote tokens
   * optional learned sink tokens
5. the updated hidden state continues through the normal residual stream and LM
   head

Important:

* the tree does **not** become the final predictive mechanism
* the tree does **not** vote directly on logits
* retrieval is advisory
* exact attention performs the final content binding

---

## Scope constraints

This must stay deliberately small.

### Required constraints

* single-root only
* one rescue block first
* top-of-stack placement only
* local attention window only, not dense global attention
* routed remote token gathering only from sealed leaves
* top `8` remote spans only
* no side-memory bank
* no learned early stop
* no merge-policy changes
* no other long-range integration path during the experiment
* begin with a frozen-backbone control where only the rescue block and its
  projections are trainable

### Suggested starting values

* local window: `256`
* remote routed spans: `8`
* leaf size: `16`
* remote tokens per span: all `16` tokens in the sealed span for the first test
* sink tokens: `0` initially
* fixed total exact-attention token budget: `384`

Matched controls:

* local-only baseline: `384` local tokens
* rescue run: `256` local tokens + `128` remote tokens

Do not let the rescue variant silently win by seeing more exact tokens overall.

Keep the proving version as small as possible.

---

## Retrieval contract

The rescue block must operate on **raw token-level representations**, not leaf
summaries.

For this pre-validation, the stored remote representation is exactly:

* the sealed-time token-state snapshot taken at the output of the existing
  local trunk
* before any rescue-attention block
* with one stored row per token and explicit absolute positions

The rescue block itself owns the projection from these raw token states into
query/key/value space for exact attention.

That means:

* router selects candidate spans
* runtime gathers sealed-time raw token states from those spans
* exact attention consumes those gathered token states directly after the rescue
  block projects them into K/Vs

Do **not** use:

* precomputed memory-only projections
* v2-style exact-read output vectors
* blended leaf summaries as a substitute for gathered token states

Do **not** repeat the v2 mistake of collapsing retrieved evidence into a
summary-like side-channel before exact interaction happens.

This contract must be visible in code.

---

## Causality and masking contract

The rescue block must obey all of these rules:

* routed remote spans may come only from sealed spans whose end position is
  strictly less than the current query position
* local-window tokens use the normal causal mask over absolute positions
* gathered remote tokens are concatenated into the attention context only if
  their absolute positions are strictly earlier than the current query position
* no live unsealed remote tokens may appear in the gathered set
* the concatenated local + remote attention mask must be constructed from
  absolute token positions, not from array order alone

Sealed leaf status is necessary but not sufficient.
The rescue block must still enforce causal visibility explicitly.

---

## Recommended implementation seam

Implement the pre-validation directly under the future hybrid line.
Do **not** create a second experimental control plane elsewhere.

Use:

* `fractal-core/src/hybrid/rescue_attention.rs`
* `fractal-core/src/hybrid/retrieval_gather.rs`
* `fractal-core/src/hybrid/model.rs`

Suggested type ownership:

* `RescueAttentionConfig`
  - owns local-window size, routed-leaf count, and sink-token count
* `GatheredRetrievalContext`
  - owns gathered raw token states, span metadata, absolute positions, and
    candidate-recall diagnostics
* `RescueAttentionBlock`
  - owns exact attention over `local window + remote gathered tokens`

---

## Forward pass sketch

At token step `t`:

1. compute the current hidden state from the existing local path
2. build the local recent window keys and values
3. route over sealed leaves
4. gather raw token states from the selected sealed spans
5. concatenate:
   * local-window K/Vs projected from live local states
   * remote gathered K/Vs projected from gathered raw token states
   * optional sink K/Vs
6. run exact causal attention from the active token representation
7. write the updated hidden state back into the residual stream
8. project logits from the updated hidden state

This is a real exact-attention step, not a soft diagnostic reweighting trick.

---

## Required ablations

Run these in order and record all of them in the ledger.

1. local-only exact attention with matched total token budget
2. local + routed remote attention
3. local + oracle remote attention
4. local + oracle remote + oracle exact-token subset
5. local + routed remote attention, no tree summaries anywhere else
6. local + routed remote attention, no warm recurrent contribution
7. frozen-backbone run where only the rescue block and its projections train
8. unfrozen follow-up run only if step `7` shows oracle usefulness

These are not optional storytelling tools.
They are the point of the pre-validation.

Oracle exact-token subset means:

* all tokens whose absolute positions fall inside the labeled evidence span
* capped at the same remote token budget as the learned run
* if the evidence span is longer than the budget, keep the earliest contiguous
  prefix that fits

That rule must stay fixed across tasks and runs.

---

## Required metrics

Track the following for the rescue block:

* probe accuracy on:
  * `MQAR`
  * copy
  * induction
  * retrieval-heavy probes
* target token rank
* target logit lift relative to local-only baseline
* evidence-span recall in routed candidate spans
* evidence-token recall in the gathered remote token set
* local vs remote attention mass
* attention mass on ground-truth evidence span
* routed span distance
* latency overhead
* KV or gathered-token memory overhead

For oracle runs, track separately:

* oracle leaf selected
* oracle token contained
* attention mass on oracle evidence token

If we cannot see whether remote exact attention actually touched the evidence,
the test is under-instrumented.

A failed learned run must let us distinguish:

* router miss
* gather miss
* attention miss

---

## Success criteria

The pre-validation is successful if **both** of these become true:

### Behavioral success

* on the held-out `MQAR + copy + induction + retrieval-heavy` suite, oracle
  remote attention improves mean target rank by at least `25%` relative to the
  matched local-only control
* on that same suite, oracle remote attention improves aggregate probe accuracy
  by at least `0.10` absolute over the matched local-only control
* learned routed remote attention recovers at least `33%` of the oracle
  target-rank gain
* learned routed remote attention improves aggregate probe accuracy by at least
  `0.05` absolute over the matched local-only control

### Mechanistic success

* evidence-token recall in the gathered set is reported for every run
* under oracle remote attention, mean attention mass on evidence tokens is at
  least `0.50`
* under learned routed remote attention, mean attention mass on evidence tokens
  is greater than mean attention mass on non-evidence remote tokens
* when evidence-token recall is true, target rank improves over local-only by
  at least `8` positions on average

The best case is:

* local-only is decent
* local + routed remote is better
* local + oracle remote is clearly better still

That would strongly support the hybrid direction.

---

## Failure criteria

The pre-validation fails if any of these remain true after a fair run:

* oracle remote exact attention improves aggregate probe accuracy by less than
  `0.10` absolute over the matched local-only control
* oracle remote exact attention improves mean target rank by less than `25%`
  over the matched local-only control
* attention over gathered tokens stays diffuse or irrelevant
* gathered remote exact interaction is active but not helpful

If oracle remote exact attention still fails, then the issue is not just the
missing attention primitive.

That would point toward:

* deeper representation misalignment
* incompatible training setup
* or a more fundamental design failure

In that case, do **not** proceed to a full hybrid implementation without a
stronger new theory.

---

## Non-goals

This pre-validation is **not**:

* a final hybrid architecture
* a license to add dense global attention back everywhere
* a place to reintroduce multi-root complexity
* a benchmark-optimized production implementation
* a rescue attempt with hidden knobs

It is one clean test of one specific missing primitive.

The task surface for this pre-validation is limited to:

* `MQAR`
* `copy`
* `induction`
* retrieval-heavy probes

Do not broaden the task mix until this gate is resolved.

---

## Decision rule

After the pre-validation, choose exactly one of these paths:

1. **Proceed to hybrid v3**
   - if oracle and learned rescue attention improve retrieval behavior
2. **Stop and reconsider**
   - if even oracle rescue attention does not help materially

Do not blur the outcome with partial optimism.

If the rescue path works, it gives a concrete design direction.
If it does not, it saves a much larger implementation cycle.
