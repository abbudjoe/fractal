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
2. the existing tree router proposes top-`k` distant candidate leaves
3. the model gathers raw token-level hidden states or token K/Vs from those
   routed leaves
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
* no side-memory bank
* no learned early stop
* no merge-policy changes

### Suggested starting values

* local window: `256`
* remote routed leaves: `4`
* remote tokens per leaf: all tokens in the selected leaf for the first test
* sink tokens: `0` initially

Keep the proving version as small as possible.

---

## Retrieval contract

The rescue block must operate on **raw token-level representations**, not leaf
summaries.

That means:

* router selects candidate leaves
* runtime gathers token-level hidden states or token K/Vs from those leaves
* exact attention consumes those token-level representations directly

Do **not** repeat the v2 mistake of collapsing retrieved evidence into a
summary-like side-channel before exact interaction happens.

This contract must be visible in code.

---

## Recommended implementation seam

Add a small new module family under the future hybrid line or as a narrow
pre-validation surface:

* `fractal-core/src/hybrid/rescue_attention.rs`
* `fractal-core/src/hybrid/retrieval_gather.rs`
* `fractal-core/src/hybrid/model.rs`

Suggested type ownership:

* `RescueAttentionConfig`
  - owns local-window size, routed-leaf count, and sink-token count
* `GatheredRetrievalContext`
  - owns gathered token-level remote K/Vs and span metadata
* `RescueAttentionBlock`
  - owns exact attention over `local window + remote gathered tokens`

If this is prototyped first inside an experiment module, preserve these
ownership boundaries so the behavior can be promoted cleanly.

---

## Forward pass sketch

At token step `t`:

1. compute the current hidden state from the existing local path
2. build the local recent window keys and values
3. route over sealed leaves
4. gather token-level remote K/Vs from the selected leaves
5. concatenate:
   * local-window K/Vs
   * remote gathered K/Vs
   * optional sink K/Vs
6. run exact causal attention from the active token representation
7. write the updated hidden state back into the residual stream
8. project logits from the updated hidden state

This is a real exact-attention step, not a soft diagnostic reweighting trick.

---

## Required ablations

Run these in order and record all of them in the ledger.

1. local-only exact attention
2. local + routed remote attention
3. local + oracle remote attention
4. local + oracle remote + oracle exact-token subset

Optional but useful:

5. local + routed remote attention, no tree summaries anywhere else
6. local + routed remote attention, no warm recurrent contribution

These are not optional storytelling tools.
They are the point of the pre-validation.

---

## Required metrics

Track the following for the rescue block:

* probe accuracy on:
  * copy
  * associative recall
  * noisy retrieval
  * far-token comparison
* target token rank
* target logit lift relative to local-only baseline
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

---

## Success criteria

The pre-validation is successful if **both** of these become true:

### Behavioral success

* copy and retrieval probes improve materially over local-only baseline
* oracle remote attention improves sharply
* learned routed remote attention closes a meaningful fraction of that gap

### Mechanistic success

* remote attention becomes sharp on evidence tokens
* attention mass over the evidence span is interpretable
* target token rank rises substantially when oracle remote evidence is present

The best case is:

* local-only is decent
* local + routed remote is better
* local + oracle remote is clearly better still

That would strongly support the hybrid direction.

---

## Failure criteria

The pre-validation fails if any of these remain true after a fair run:

* oracle remote exact attention barely improves probe accuracy
* oracle remote exact attention does not materially improve target rank
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
