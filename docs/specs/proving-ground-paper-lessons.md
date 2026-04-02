# Proving-Ground Paper Lessons

This note captures the immediate lessons from two comparison papers:

- [fn.pdf](/Users/joseph/papers/fn.pdf)
- [arXiv-2507.10524v3.tar.gz](/Users/joseph/Downloads/arXiv-2507.10524v3.tar.gz)

Its purpose is not to decide whether `fractal` should stop.

Its purpose is to sharpen the proving-ground questions:

- what is already validated at the family level
- what remains uncertain in the current kernel
- what we must measure before changing semantics or scaling spend

## Straight Appraisal

The broad family is already validated.

What is already established by prior work:

- FractalNet showed that deep self-similar networks without residual connections are not inherently doomed.
- MoR showed that a language-model architecture with parameter sharing, adaptive token-level recursion depth, and efficiency-oriented routing/caching can work in practice.

What is not yet answered by those papers:

- whether the current `fractal` executable kernel is the right primitive
- whether the current router/control path learns useful adaptive depth
- whether the observed failures are mathematical, architectural, or backend/layout artifacts

So the current program is not:

- replaying a known winner exactly
- replaying a known dead end exactly

It is testing a different kernel inside a family that already has external evidence of viability.

## What MoR Is, Precisely

MoR is not the current `fractal` runtime.

MoR is:

- a Llama-based Transformer
- with shared recursive transformer blocks
- with token-level routing over recursion depth
- with recursion-aware KV handling

The current `fractal` Stage 0 executable path is:

- `embedding -> recursive rule -> router -> output`

and the recursive core is a custom recurrent state-update rule, not a shared transformer block.

That distinction matters. It means the proving ground should focus less on whether recursive adaptive depth is legitimate at all, and more on whether this specific recurrent kernel is a strong instantiation of that family.

## Seven Lessons

### 1. The family is validated, not the core

MoR removes the need to prove that shared recursive adaptive-depth language modeling is a crazy idea.

That question is already answered well enough to move on.

The unresolved question is narrower:

- is the current `fractal` recurrent kernel competitive inside that family?

### 2. Routing is a first-class mechanism, not plumbing

MoR reports materially different outcomes across routing choices.

That means the router is part of the scientific object, not just an implementation detail.

For `fractal`, router behavior should be treated as a primary proving-ground target:

- useful adaptive allocation
- collapse detection
- trivial always-deep or always-shallow behavior

### 3. The current router is less controlled than MoR's

MoR uses explicit routing formulations, capacity logic, and balancing/auxiliary machinery.

The current `fractal` runtime uses a simpler thresholded early-exit path.

That does not mean the current router is wrong.

It does mean the proving ground should assume the router is a serious uncertainty until it is measured directly.

### 4. Caching and memory behavior are part of the science

MoR treats recursion-aware memory behavior as part of the architecture, not merely a systems optimization.

That lesson transfers directly.

For `fractal`, throughput, memory traffic, and backward-path stability are not side concerns. They are part of whether the architecture is viable.

### 5. Shared recursive cores can bottleneck at small capacity

MoR explicitly reports a recursive capacity bottleneck at smaller scales.

That matters for interpretation:

- if the current kernel underperforms, the likely question is not just "does the idea fail?"
- it may instead be "is the shared rule underpowered for the recursion budget?"

### 6. Forced-depth ablations are mandatory

MoR analyzes recursion depth directly, and FractalNet benefited from comparing paths of different effective depths.

For `fractal`, adaptive depth should not be evaluated in isolation.

The proving ground should compare:

- fixed shallow depth
- fixed deep depth
- adaptive depth

under the same data and budget rules.

If adaptive depth does not beat the fixed-depth controls, that is a meaningful result.

### 7. Lack of residual structure is not, by itself, a fatal objection

FractalNet matters here mainly as a rebuttal to a lazy dismissal.

It does not validate the current language-model kernel directly.

But it does show that "this is not a standard residual stack" is not enough to conclude the design is unsound.

That means skepticism should target the actual primitive, routing, and training behavior, not the absence of familiar residual form alone.

## Proving-Ground Checklist

The proving ground should answer these questions before larger spend or semantic changes.

### A. Is adaptive depth real or collapsing?

Measure:

- recursion-depth histogram over tokens
- fraction of tokens exiting at each depth
- router entropy or confidence shape
- dead-token or never-deep-selected behavior
- stability of routing behavior across checkpoints

Success signal:

- the router uses multiple depths in a nontrivial, stable way
- it does not collapse immediately to a trivial policy

### B. Does extra recursion improve token prediction?

Compare under the same training setup:

- forced depth `1`
- forced depth `k`
- forced max depth
- adaptive router-enabled depth

Success signal:

- adaptive or deeper recursion produces a measurable quality gain
- not just additional compute with no benefit

### C. Is failure mathematical or systems-induced?

Track:

- forward completion boundaries
- loss materialization boundaries
- backward boundaries
- optimizer-step boundaries
- memory snapshots at those points

Success signal:

- a failure can be localized to a real boundary
- we can distinguish kernel weakness from backend/layout failure

### D. Is the shared rule underpowered?

Probe:

- smaller vs larger hidden width under fixed recursion
- smaller vs larger recursion budget under fixed width
- contractive vs hybrid variants under the same outer runtime

Success signal:

- performance differences reveal whether the rule family has headroom
- not just whether the current Stage 0 default wins

### E. Does adaptive depth pay for itself operationally?

Measure:

- throughput
- memory use
- tokens per second
- failure rate
- checkpoint/restart reliability

Success signal:

- adaptive depth improves quality, systems efficiency, or both
- without introducing unacceptable instability

### F. Is the run contract trustworthy?

Require:

- typed diagnostics
- typed export/load semantics
- typed failure snapshots
- preserved lineage from manifest through artifact

Success signal:

- failures are inspectable without guesswork
- future fixes can be chosen from evidence rather than speculation

## Training Playbook

This section translates the paper lessons into concrete pretraining and proving-ground behavior.

### What To Copy From MoR's Experimental Method

#### 1. Treat pretraining as the primary arena

Do:

- judge the architecture on real pretraining, not only toy tasks
- keep the proving ground tied to real corpus ingestion, tokenizer behavior, and training runtime
- treat successful pretraining as the minimum bar for architectural credibility

Reason:

- MoR was validated through real pretraining runs, not just clever routing diagrams

#### 2. Evaluate under both equal-token and equal-compute views

Do:

- compare models at equal tokens seen
- compare models at equal compute when measurement is available
- keep both comparisons in reports and decision documents

Reason:

- equal-token answers "does it learn better under the same data budget?"
- equal-compute answers "does the architecture buy more progress per unit of work?"

#### 3. Always include fixed-depth controls

Do:

- run a fixed shallow-depth control
- run a fixed deep-depth control
- run the adaptive-depth version under the same outer contract

Reason:

- adaptive depth only means something if it beats a fixed-depth baseline under the same budget

#### 4. Treat routing as a first-class ablation axis

Do:

- report depth-allocation histograms
- report exit-rate behavior over training
- report collapse or trivial-policy behavior
- compare router-enabled versus force-depth modes

Reason:

- routing is part of the architecture, not just a switch

#### 5. Treat efficiency as part of the architecture

Do:

- record throughput
- record memory behavior
- record restart/checkpoint reliability
- record the runtime cost of adaptive depth

Reason:

- MoR's claim is not only quality, but quality relative to compute and memory behavior

#### 6. Probe scaling behavior before making big claims

Do:

- vary model width or recursion budget in small controlled studies
- check whether the architecture improves or bottlenecks as capacity changes
- keep these runs smaller than flagship runs but large enough to be meaningful

Reason:

- some recursive/shared designs only look good at one narrow scale

#### 7. Make adaptive depth earn its keep

Do:

- require adaptive depth to improve quality, efficiency, or both
- remove or narrow it if it only adds instability and complexity

Reason:

- "interesting behavior" is not enough

### What To Deliberately Do Differently For `fractal`

#### 1. Keep the kernel question open

Do:

- preserve the distinction between the validated family and the current kernel
- ask whether the custom recurrent rule is good, not whether recursion in general is good

Do not:

- treat MoR's success as proof that the current primitive is already validated

#### 2. Do not inherit transformer assumptions

Do:

- keep the proving ground focused on the current executable path
- evaluate the recurrent state-update rule on its own terms

Do not:

- drift toward transformer-style claims just because MoR is transformer-based

#### 3. Keep semantic changes separate from observability changes

Do:

- improve diagnostics, failure snapshots, and exports before changing training semantics
- rerun the same manifest after observability upgrades

Do not:

- use architecture changes to hide systems uncertainty

#### 4. Treat backend/layout failures as architecture-adjacent, not mere noise

Do:

- localize failures to forward, loss, backward, optimizer, or export boundaries
- preserve enough state to determine whether failure is mathematical or systems-induced

Do not:

- jump straight from a backend failure to a model-theory conclusion

#### 5. Prefer control-plane discipline over benchmark theater

Do:

- keep manifest, diagnostics, export, and failure semantics typed
- require every proving-ground result to be inspectable and reproducible

Do not:

- let wrapper-only behavior or hand-reconstructed run state drive scientific conclusions

### Recommended Proving-Ground Run Shapes

The next proving-ground sequence should look like this:

1. Same-manifest rerun with improved diagnostics and failure snapshots.
2. Fixed-depth shallow control under the same data contract.
3. Fixed-depth deeper control under the same data contract.
4. Adaptive-depth run under the same outer contract.
5. Small capacity sweep to test whether the shared rule is bottlenecked.

Each run should preserve:

- the exact manifest
- the exact commit
- diagnostics output
- failure or export artifacts
- a short comparative summary against the other control runs

### Minimum Decision Questions After Each Run

After every proving-ground run, answer:

1. Did the router behave nontrivially?
2. Did additional recursion improve prediction quality?
3. Did the failure, if any, localize to a real runtime boundary?
4. Did the run suggest kernel weakness, router weakness, or backend weakness?
5. Did adaptive depth pay for itself in quality, efficiency, or both?

If those questions cannot be answered from the artifact set, the run is operationally incomplete even if it technically finished.

## What Not To Do Yet

Until the proving-ground questions above are answered, do not:

- change training semantics speculatively
- change sequence length just to hide failures
- change batch size just to hide failures
- replace the router or core because of intuition alone
- interpret one systems failure as a verdict on the entire model family

## Program Implication

The proving ground should now focus on:

- whether the current recurrent kernel is a good member of the recursive adaptive-depth family
- whether the router learns meaningful token-level allocation
- whether the observed failures come from backend/layout seams or from the kernel itself

That is the real decision surface.

Not:

- "has anyone ever tried anything recursive before?"
