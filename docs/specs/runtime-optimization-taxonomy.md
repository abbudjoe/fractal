# Runtime Optimization Taxonomy

The shared Python research stack now distinguishes runtime families explicitly so
backend policy and optimization work do not leak across incompatible model
shapes.

## Families

- `pure-transformer`
  - no recurrent scan path
  - attention and dense feed-forward kernels dominate
  - current Path 1 `attention-only` belongs here

- `recurrent-scan-hybrid`
  - recurrent or SSM scan path is part of the model's hot runtime
  - scan kernels, packed projections, and recurrent runtime backends matter
  - current Path 1 `reference-ssm-hybrid` and `primitive-hybrid` belong here

- `transformer-moe-routing`
  - transformer backbone with routing, dispatch, and expert execution in the hot path
  - routing and dispatch execution strategy matter more than scan kernels
  - current mini-MoE experiments belong here

## Why This Exists

This taxonomy is a control-plane guardrail.

It lets us say, explicitly:

- Triton scan optimizations target `recurrent-scan-hybrid`
- MoE dispatch and round-2 execution strategies target `transformer-moe-routing`
- generic compile/runtime claims should be compared within a runtime family first

Without this shared surface, optimization policy gets inferred from variant
names and benchmark labels, which is exactly how hidden contracts drift.
