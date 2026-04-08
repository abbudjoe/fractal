# V2 KPIs

## Purpose

This document defines the stability and usefulness metrics for the narrowed `fractal-v2` proving architecture.

Implementation of the tree-only v2 is largely complete.
These KPIs exist to answer the next question:

Is the foundation stable and useful enough to justify more training and later complexity?

The metrics below are grouped into six families.
Each family should be tracked across repeated checkpoints, not judged from a single run.

## Data sources

Primary measurement surfaces:

* [/Users/joseph/fractal/src/bin/v2-smoke-train.rs](/Users/joseph/fractal/src/bin/v2-smoke-train.rs)
* [/Users/joseph/fractal/src/bin/v2-synthetic-probe.rs](/Users/joseph/fractal/src/bin/v2-synthetic-probe.rs)
* [/Users/joseph/fractal/src/bin/v2-benchmark-suite.rs](/Users/joseph/fractal/src/bin/v2-benchmark-suite.rs)
* [/Users/joseph/fractal/fractal-core/src/v2/auditor.rs](/Users/joseph/fractal/fractal-core/src/v2/auditor.rs)

Supporting logic:

* [/Users/joseph/fractal/fractal-eval-private/src/v2_training.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_training.rs)
* [/Users/joseph/fractal/fractal-eval-private/src/v2_synthetic.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_synthetic.rs)
* [/Users/joseph/fractal/fractal-eval-private/src/v2_benchmark.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_benchmark.rs)
* [/Users/joseph/fractal/fractal-eval-private/src/v2_ablation.rs](/Users/joseph/fractal/fractal-eval-private/src/v2_ablation.rs)

## 1. Learning Signal

### What we track

* train loss
* eval loss
* perplexity
* loss delta over fixed token budgets

### What it means

This is the first gate.
It answers whether the model is actually learning rather than merely executing.

If this family is unhealthy, the rest of the architecture metrics are mostly noise.

### Current read

Current smoke runs show a real but still small improvement in eval loss.
That is enough to say the path is trainable at smoke scale, but not enough to claim strong architectural quality.

## 2. Synthetic Probe Utility

### What we track

For each probe family:

* accuracy
* mean target logit
* mean loss

Across each memory mode:

* `NoMemory`
* `SummariesOnly`
* `TreeOnly`
* `TreePlusExactRead`

Probe families:

* copy
* associative recall
* induction
* noisy retrieval
* far-token comparison

### What it means

This is the clearest direct test of the v2 thesis.
It asks whether tree memory and exact local reads improve the behaviors they were added to support.

### Current read

The probe surface is working, but the latest learned checkpoint is still too small and undertrained to produce decision-grade separation.
Right now the results are valid, but not yet informative enough to declare the architecture healthy or broken.

## 3. Causal Memory Auditor Deltas

### What we track

Counterfactual performance drops when intervening on the memory path:

* no tree read
* no exact read
* next-best substitution
* root drop

### What it means

This family answers whether the memory paths are causally useful in the live graph.
It is not enough for the tree and exact-read paths to exist.
Removing them should measurably hurt the tasks they are supposed to help.

### Current read

The auditor is implemented and runnable, but the present checkpoint is too weak for strong utility claims.
This family becomes much more meaningful after a larger trained checkpoint exists.

## 4. Root Specialization / Collapse

### What we track

* root similarity / collapse cosine
* per-root norm statistics
* root-drop utility deltas

### What it means

Multi-root only matters if roots specialize at least somewhat.
If the roots remain near-identical, the architecture is paying complexity for width without real functional separation.

### Practical interpretation

There is no single magic number, but these ranges are useful:

* `> 0.95`: probably collapsed
* `0.7 - 0.95`: weak specialization, still redundant
* `0.3 - 0.7`: healthier range if utility also improves
* `< 0.2`: may indicate strong diversity or fragmentation; interpret with care

### Current read

The current learned benchmark runs still show very high root similarity, around `0.997`.
That is a concern, but not yet a final verdict given how small the trained checkpoint is.

## 5. Routing Health

### What we track

* routing depth histogram
* candidate entropy per head
* head agreement rate
* selected span distance distribution
* selected leaf usage distribution

### What it means

This family answers whether routing is real, sparse, and query-dependent.
The routing system should not collapse into one trivial path or one default leaf regardless of input.

### Current read

Routing is structurally working and observable.
We do not yet have enough trained signal to say the chosen paths are semantically useful.

## 6. Systems / Scaling Health

### What we track

* forward throughput
* tree update time
* routing time
* exact read time
* dead / unused tree nodes
* causal correctness invariants

### What it means

This family answers whether the implementation is stable enough to scale training further.
It is the strongest current evidence that the architecture is mechanically sound.

### Current read

This is currently the healthiest category:

* the incremental tree path is stable
* benchmark runs complete end to end
* `dead_nodes=false`
* tree, routing, and exact-read costs stay bounded and interpretable

## Current summary

The current state of `fractal-v2` is:

* implementation foundation: largely complete
* systems health: good
* smoke-scale training viability: real
* root specialization: currently weak
* memory usefulness: not proven yet

The next meaningful checkpoint is not another architecture feature.
It is a stronger trained run followed by probe, auditor, and benchmark reruns on that learned checkpoint.
