# Runtime Optimization Taxonomy

This note records which runtime enhancement families are generic sequence-model
work and which are specific to recurrent / scan-based hybrids.

The point is to keep later optimization work honest. A target that is useful
for a transformer+scan hybrid is not automatically a good target for a pure
transformer backbone, and vice versa.

## Universal Sequence Targets

These help many sequence architectures:

* packed projections
* layout/stride contracts
* autotuned kernel launch choices
* chunked execution when the model admits it
* better backward/workspace design

These belong to the shared Python runtime substrate, not to one experiment.

## Recurrent Scan Targets

These are specific to recurrent / SSM-like / scan-based hybrids:

* chunked state passing
* sequence scan kernels
* recurrence-aware fusion boundaries
* structured state transforms
* latent-state update kernels

These are the right optimization family for Path 1 primitives and future
recurrent-router-style models that expose an explicit latent scan.

## Pure Transformer Targets

These are much more specific to pure transformer backbones:

* attention kernels
* KV cache layout
* MLP fusion
* sequence parallelism
* memory-efficient attention/backward

These should not be presented as the main optimization frontier for a recurrent
primitive line just because both models process sequences.

## Typed Surface

The shared typed surface for this taxonomy lives in:

* `/Users/joseph/fractal/python/specs/runtime.py`

Current supported architecture families:

* `recurrent-scan-hybrid`
* `pure-transformer`

That surface is intended to guide future reusable runtime work in:

* `/Users/joseph/fractal/python/runtime/`

instead of hiding architecture-specific optimization assumptions inside:

* `/Users/joseph/fractal/python/models/`
