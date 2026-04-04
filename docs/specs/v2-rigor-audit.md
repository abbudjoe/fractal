## V2 Rigor Audit

This note records where `fractal-v2` currently satisfies the narrowed v1 process rules, where it only satisfies them partially, and what must be added retroactively before further architectural debugging.

It is intentionally descriptive, not aspirational.

### Current status

The branch has a real narrowed v2 implementation:

- multi-root local trunk
- live leaf and sealed-leaf path
- causal dyadic tree
- sparse router
- exact leaf read
- read fusion and LM head wiring
- causal memory auditor
- synthetic probe harness
- benchmark runner
- smoke training
- checkpoint load/eval runners

That means the architecture is implemented and falsifiable.

It does **not** mean the architecture is validated.

Current learned runs are still negative:

- roots collapse strongly
- tree and exact-read paths are active but not helpful
- oracle routing and oracle exact-read do not rescue probe accuracy
- supervised synthetic training also stays negative on held-out probes

### What we followed correctly

These rules were followed well:

- deferred features stayed deferred
- tree construction stayed regular and deterministic
- early stop stayed disabled in v1
- exact read remained explicitly ablatable
- causal auditing was added instead of inferred from prose
- the architecture can now be falsified cleanly

### Where rigor was only partially followed

#### Required ablations

The implementation added an equal-budget ablation sweep in:

- [v2_ablation.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_ablation.rs)
- [v2-ablation-sweep.rs](/Users/joseph/fractal/src/bin/v2-ablation-sweep.rs)

That sweep compares:

- `single_root`
- `multi_root`

against these memory modes:

- `NoMemory`
- `SummariesOnly`
- `TreeOnly`
- `TreePlusExactRead`

This captures the main factorization, but it is still a compressed `2 x 4` sweep rather than a first-class maintained nine-case proving matrix.

More importantly, the learned-checkpoint loop has not yet been enforced through that same matrix as a required release gate.

So the ablation rule was followed in spirit, but not yet with full process rigor.

#### Observability

The implementation computes the following diagnostics internally:

- root collapse / similarity
- routing depth histogram
- candidate entropy per head
- selected-span distance histogram
- head agreement
- leaf usage
- dead / unused tree-node behavior

The main public observability surface in:

- [v2_benchmark.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_benchmark.rs)
- [v2-benchmark-suite.rs](/Users/joseph/fractal/src/bin/v2-benchmark-suite.rs)

currently exposes only a summarized subset:

- routing sparsity
- root collapse mean pairwise cosine similarity
- exact-read usage
- mean retrieval distance
- tree depth reached
- head agreement rate
- dead / unused tree nodes
- selected leaf usage bins

This is useful, but it does not yet satisfy the stricter requirement that routing depth histograms, candidate entropy per head, and selected-span distance histograms be surfaced as first-class benchmark / ledger outputs.

So observability is real, but only partially surfaced.

### Definition-of-done reality check

The branch satisfies the implementation-side prerequisites:

- multi-root local trunk works
- leaf sealing is causal and correct
- dyadic tree is stable and incremental
- sparse routing exists
- exact leaf read exists and is ablatable
- synthetic tasks run
- scaling benchmarks exist
- collapse and dead-weight behavior are measurable
- causal auditing exists
- the architecture is falsifiable

But the branch does **not** satisfy the spirit of a validated first serious v1, because the learned behavior is still negative:

- roots are nearly interchangeable
- memory paths are active but not helpful
- learned retrieval does not separate from no-memory controls
- oracle and supervised probes still do not rescue held-out behavior

The correct reading is:

- implementation foundation: present
- architecture validation: not achieved

### Retroactive rigor work that now becomes mandatory

Before spending more cycles on architectural debugging, the following should be treated as required:

1. Make the required ablation matrix explicit and durable.
   This means a first-class learned-checkpoint sweep that records the full single-root / multi-root by memory-mode matrix as a tracked output, not only an ad hoc live run.

2. Surface the full routing observability contract publicly.
   Promote these from internal diagnostics to benchmark / ledger outputs:
   - routing depth histogram
   - candidate entropy per head
   - selected-span distance histogram

3. Treat learned ablation and observability outputs as release gates.
   New architectural diagnosis work should not skip these artifacts or replace them with prose interpretation.

4. Keep deferred complexity deferred.
   Do not add:
   - side-memory bank
   - learned eviction
   - learned merge scheduling
   - routing early stop
   - dense fallback attention
   as a substitute for missing rigor.

### Immediate next rigor slices

The next corrective steps should be:

1. extend the learned ablation runner so the single-root / multi-root matrix is a required checkpoint evaluation surface
2. extend the benchmark and ledger schema so the full routing histogram / entropy surfaces are exported directly
3. rerun the learned checkpoint loop only after those outputs are recorded automatically

That is the cleanest way to reduce technical debt and stop paying repeated debugging costs for missing observability and incomplete ablation discipline.
