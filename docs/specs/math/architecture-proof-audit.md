# Architecture Proof Audit

This document audits the current theorem surface in
[`architecture-proof-notes.tex`](./architecture-proof-notes.tex).

It does not claim machine-checked correctness.
It classifies each theorem-like statement into one of four buckets:

- `Definition only`
  - a contract or notation declaration, not a truth-apt claim
- `Verified sketch`
  - the proof idea is materially sound under the stated architecture model
- `Needs stronger assumptions`
  - directionally plausible, but the written proof does not yet justify the
    exact statement as written
- `Conjectural / empirical`
  - intentionally outside the present theorem surface
- `Boundary remark`
  - commentary that marks a theorem gap without pretending the gap is closed

Remarks are not counted as formal claims in the summary below.

## Summary

- `9` definitions
- `28` verified sketches
- `0` statements that need stronger assumptions
- `4` conjectural or empirical claims

The current file is materially cleaner than the first draft.
The most important caution points now live in:

- explicit assumptions written into theorem statements
- explicit cost-model definitions
- remarks that mark known theorem gaps instead of pretending they are proved

## Section Audit

### Common Machine and Proof Discipline

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Causal update | `109-113` | `Definition only` | Declares the meaning of causality for later proofs. | None. |
| Causal composition | `115-124` | `Verified sketch` | Standard closure-under-composition result. | None beyond routine polish. |
| Incremental-decode equivalence | `126-133` | `Definition only` | Declares what equivalence means. | None. |
| Reduction to a simpler class | `135-139` | `Definition only` | Contract definition. | None. |
| Strict structural generalization | `141-148` | `Definition only` | Contract definition used later. | None. |

### Standard Decoder LLMs

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Decoder language model | `175-182` | `Definition only` | Defines the family. | None. |
| Causal decoder correctness | `184-196` | `Verified sketch` | Masked attention, pointwise FFN, and residual composition are enough for causality. | Optional: state positional-encoding assumptions explicitly. |
| KV-cache decode equivalence | `198-212` | `Verified sketch` | The theorem now explicitly assumes fixed weights, fixed masks, and deterministic inference. | Optional: mention immutable cached K/Vs in the statement too. |
| Approximate copy capacity | `214-226` | `Verified sketch` | The statement now matches the proof: attention can be made arbitrarily concentrated on one visible prior token. | None unless we want a hard-attention limit theorem. |
| Token comparison capacity | `228-238` | `Verified sketch` | As an expressivity sketch, attention plus FFN can implement comparison features over selected tokens. | Tighten what “comparison” means if we want a narrower theorem. |
| Complexity bound | `240-253` | `Verified sketch` | Standard asymptotic statement for dense train-time attention and cached decode. | Add explicit big-O notation and constants if desired. |

### Sparse Mixture of Experts

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Dense reduction for sparse MoE | `265-277` | `Verified sketch` | Degenerate routing with identical experts reduces to a dense FFN. | None. |
| Expert budget invariant | `279-288` | `Verified sketch` | Immediate from the router contract. | None. |
| Control-plane separation | `290-299` | `Verified sketch` | Causality is preserved if routing and experts consume only causal state. | None. |
| Parameter-efficiency benefit | `301-304` | `Conjectural / empirical` | Correctly marked as non-theorem. | Leave empirical. |

### Linear-Attention Families

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Associative recurrent form | `315-326` | `Verified sketch` | Prefix summaries `S_t` and `z_t` yield the standard recurrent update form. | Optional: write the exact recurrence explicitly. |
| Softmax exactness boundary | `328-333` | `Boundary remark` | Now carried as a boundary remark rather than a proved proposition. | Leave as a remark unless we add a real separation theorem. |
| Decode complexity bound | `335-345` | `Verified sketch` | Follows from bounded-size recurrent summaries replacing full cache scans. | Add explicit state-dimension assumptions if desired. |

### Hybrid Recurrent-Attention Families

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Hybrid layer cost model | `359-371` | `Definition only` | Introduces the explicit compute and memory model needed by the hybrid cost propositions. | None. |
| Hybrid causal composition | `373-382` | `Verified sketch` | Standard causal-composition argument. | None. |
| Exact-attention budget decomposition | `384-402` | `Verified sketch` | The earlier loose fraction claim is now a direct decomposition under the explicit cost model. | None. |
| Reduction edge cases | `404-412` | `Verified sketch` | Immediate by zeroing one family count. | None. |
| Kimi-style periodic exact-attention exposure | `414-425` | `Verified sketch` | The fixed `3:1` schedule guarantees periodic exact-attention layers structurally. | None. |
| Hybrid memory decomposition | `427-441` | `Verified sketch` | The earlier cache-pressure claim is now a direct memory decomposition under the explicit cost model. | None. |

### `A + P2`

The full section is still conditional on the expected contract class in
`469-473`, which is the right posture for now.

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Abstract `P1`-style contract | `475-483` | `Definition only` | Declares the direct-readout single-state contract used by the later `P2` reduction and enlargement claims. | None. |
| Step/scan consistency for `P2` | `485-490` | `Definition only` | Declares the equivalence contract; it is not yet proved here. | Add a future implementation theorem once `step` and `scan` are formalized. |
| Causal `P2` update correctness | `492-502` | `Verified sketch` | Causal if all submaps consume only `(s_{t-1}, x_t)`. | None beyond making the dependency graph explicit. |
| `A + P2` hybrid causality | `504-511` | `Verified sketch` | Immediate from causal composition. | None. |
| `P1`-style reduction | `513-528` | `Verified sketch` | The note now defines the target `P1` contract explicitly and proves reduction by parameter restriction. | None. |
| Structural enlargement over the abstract `P1` contract | `530-544` | `Verified sketch` | The separation now relies on an explicit direct-readout `P1` contract, which makes the non-representability claim clean. | None. |
| Matched slot complexity | `546-562` | `Verified sketch` | The statement now carries the bounded-state sequence-primitive assumption it needs. | None. |
| Efficiency-quality advantage | `564-567` | `Conjectural / empirical` | Correctly marked as empirical. | Leave empirical. |

### Graph-of-Experts / Thought-Channel Model

| Statement | Source lines | Status | Reason | Next step |
| --- | --- | --- | --- | --- |
| Channel-separated state | `607-611` | `Definition only` | Declares the separation contract. | None. |
| Single-channel reduction | `613-622` | `Verified sketch` | With `K_max = 1` and trivial merge/prune, the branching control collapses to a single-stream hybrid. | None. |
| Channel budget invariant | `624-633` | `Verified sketch` | Straight induction over the controller cap. | None. |
| Internal search budget bound | `635-646` | `Verified sketch` | Follows from bounded rounds plus bounded channel count. | Optional: add a symbolic cost function per round. |
| No architectural forced collapse under separated slots | `648-659` | `Verified sketch` | The claim now says exactly what the proof shows: separation prevents forced collapse between declared interaction points. | None. |
| Bounded beam-style control skeleton | `661-673` | `Verified sketch` | The earlier emulation wording is now weakened to a structural control correspondence that the proof justifies. | None. |
| Shared-trunk amortization | `675-685` | `Verified sketch` | Common token processing is performed once before channel-local work. | None. |
| Token-serialization savings | `687-697` | `Verified sketch` | Intermediate branches can remain latent instead of being serialized as text. | Optional: qualify the claim as a lower-bound statement on token overhead. |
| Search-efficiency advantage over external GoT | `699-703` | `Conjectural / empirical` | Correctly kept out of the theorem surface. | Leave empirical. |
| Native-search advantage over plain hybrids | `705-708` | `Conjectural / empirical` | Also empirical. | Leave empirical. |

## Recommended Next Pass

If we want this note to become substantially more trustworthy without changing
its research direction, the best sequence is:

1. formalize step/scan consistency for `P2`
   - this is still only a definition, not yet a theorem
2. add a true softmax-separation theorem if we want one
   - the current note correctly carries that boundary as a remark
3. sharpen token-level complexity statements into explicit big-O notation
   - especially if we want comparison across families in one table
4. keep empirical claims out of theorem environments
   - the current conjecture labels are good and should stay

## Bottom Line

The core causality, reduction, and budget-invariant claims are now in good
shape as proof sketches.
The remaining open work is mostly about:

- implementation-equivalence theorems such as `P2` step/scan consistency
- optional stronger separation theorems for linear attention
- tighter asymptotic notation, not basic logical soundness

That is a good sign.
It means the document is strongest where it should be strongest:
on control-plane correctness, architectural reduction structure, and bounded
search bookkeeping.
