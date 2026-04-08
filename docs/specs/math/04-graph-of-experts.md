# Graph-of-Experts / Thought-Channel Model

This note defines the theorem surface for the proposed search-native
Graph-of-Experts family, also described informally as a thought-channel hybrid.

This is the most speculative track in the proof set.
Accordingly:

- the boundary between theorem, proof sketch, and conjecture must stay sharp
- no empirical strength claim should be smuggled in as a formal result

## Informal Architecture

The proposed family contains:

- a shared exact-attention trunk
- `K` active thought channels
- per-channel recurrent or SSM workspaces
- explicit compare, score, prune, merge, and halt control

The goal is not sparse expert routing in the classic MoE sense.
The goal is bounded internal multi-hypothesis search.

## Abstract State

At step `t`, define:

- `h_t`
  - shared trunk state
- `F_t = {f_t^(1), ..., f_t^(K_t)}`
  - active channel frontier
- `K_t`
  - number of live channels at step `t`
- `c_t`
  - controller state

where each channel state `f_t^(i)` may contain:

- channel-local recurrent or SSM memory
- a channel score
- optional branch metadata

The transition can be written abstractly as:

```text
h_t = SharedAttentionTheta(h_{t-1}, x_t)
F_t^seed = SeedTheta(h_t, F_{t-1}, c_{t-1})
F_t^expand = ExpandTheta(F_t^seed, h_t)
F_t^retrieve = RetrieveTheta(F_t^expand, h_t)
F_t^cmp = CompareMergeTheta(F_t^retrieve, h_t)
(F_t, c_t) = ControlTheta(F_t^cmp, c_{t-1})
y_t = ReadoutTheta(h_t, F_t, c_t)
```

## Budget Assumptions

Assume explicit bounds:

- `K_t <= K_max`
- at most `B` internal refinement rounds per output step
- controller operations preserve the declared channel cap

These are not implementation details.
They are required for a clean control-plane theorem surface.

## Definition 4.1: Channel-Separated State

A thought-channel system has channel-separated state if each live hypothesis is
assigned a distinct explicit state slot and any cross-channel interaction occurs
only through declared compare or merge operators.

This excludes the fully blurry case where all hypotheses are permanently mixed
inside one undifferentiated latent state.

## Proposition 4.2: Single-Channel Reduction

When:

- `K_max = 1`
- the controller never branches
- merge and prune are trivial

the Graph-of-Experts family reduces to a single-stream hybrid predictor.

Proof sketch:

1. with one channel, there is no branching
2. compare and merge become vacuous
3. the controller reduces to a trivial pass-through
4. the model becomes a standard shared trunk plus one recurrent workspace

So the search-native family contains a single-stream hybrid as a special case.

## Proposition 4.3: Channel Budget Invariant

If the controller is defined to keep at most `K_max` live channels after every
prune step, then the live channel count remains bounded by `K_max` at every
step.

Proof sketch:

- the controller is the only operator allowed to change live-channel count
- by contract it outputs no more than `K_max` channels
- therefore the invariant holds by induction over steps

This is the first control-plane theorem the family must satisfy.

## Proposition 4.4: Internal Search Budget Bound

If the controller allows at most `B` internal refinement rounds per output step
and each round preserves the channel budget invariant, then per-token internal
search compute is bounded by a function of:

- `K_max`
- `B`
- one shared-trunk pass
- one compare / control pass per round

This is the formal reason the architecture can be discussed as a bounded latent
search process rather than an unbounded inner loop.

## Lemma 4.5: Channel Identity Preservation Under Separated Slots

If channel state is separated by explicit slots and cross-channel communication
occurs only through declared compare or merge operators, then branch identity
can be preserved across refinement rounds.

Proof sketch:

1. each live hypothesis occupies a distinct state slot
2. local channel updates modify only that slot
3. cross-channel interaction occurs only at explicit operators
4. therefore identity loss is not forced by the architecture itself

This does not prove the model will use the slots well.
It proves the architecture does not require immediate hypothesis collapse.

## Proposition 4.6: Bounded Beam-Style Search Emulation

A Graph-of-Experts controller with explicit:

- seed
- expand
- score
- prune
- merge
- halt

can emulate a bounded beam-style search process in latent space.

Proof sketch:

1. `seed` initializes the starting candidate set
2. `expand` creates successor candidate states
3. `score` ranks candidates
4. `prune` enforces a bounded beam width
5. `merge` collapses compatible hypotheses when desired
6. `halt` terminates the loop when the controller decides the frontier is ready

This is not a proof of good search quality.
It is a proof of formal search-process expressiveness under bounded control.

## Proposition 4.7: Shared-Trunk Amortization

Relative to an external GoT harness that re-runs a full model for each branch,
a Graph-of-Experts family with one shared trunk amortizes common token
processing work across all live channels.

Proof sketch:

1. the shared token-context processing occurs once in the trunk
2. channel-local branching happens after that shared computation
3. therefore common prefix processing is not repeated per branch inside the same
   output step

This is a structural efficiency proposition, not an end-to-end performance
guarantee.

## Proposition 4.8: Token-Serialization Savings

A latent channel model can avoid explicit token serialization of intermediate
branches that an external GoT harness would represent as text or extra model
calls.

Proof sketch:

1. intermediate branch states remain inside the frontier object `F_t`
2. they do not need to be emitted as visible text tokens
3. therefore the architecture has a lower token-overhead floor than an external
   text-serialized harness

Again, this is a structural lower-bound claim, not a benchmark claim.

## Proposition 4.9: Soft-Structured Dominance Over Pure Blur

A channel-separated design with explicit compare and merge operators strictly
adds control-plane structure relative to a fully blurry single-state latent
mixture.

Proof sketch:

1. setting `K_max = 1` collapses back toward the blurry single-state case
2. allowing `K_max > 1` plus explicit channel-preserving operators adds new
   representational and control structure
3. therefore the structured design contains the blurrier case as a restriction

This is a structural statement.
It does not prove the more structured system always trains better.

## Conjecture 4.10: Better Search Efficiency Than External GoT

The family may be more token-efficient or more compute-efficient than an
external GoT harness because of:

- latent branch storage
- shared-trunk amortization
- bounded internal compare / prune loops

But whether that becomes an actual win in accuracy, throughput, or stability is
empirical.

## Conjecture 4.11: Better Native Search Than Plain Hybrids

The family may outperform single-stream hybrids on tasks that require keeping
multiple candidate hypotheses alive across several reasoning steps.

This is exactly the kind of claim that belongs in a synthetic POC or benchmark,
not in a theorem.

## What This Note Gives Us

This note gives the clean formal core of the search-native idea:

- reduction to a single-stream special case
- explicit channel-budget and search-budget invariants
- a proof surface for channel identity preservation
- a latent beam-search emulation result
- structural amortization and token-overhead propositions

That is the right theorem surface for this family at the current stage.
