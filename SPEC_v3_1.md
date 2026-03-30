# fractal-rust-evolver v3.1

## Goal

Implement exactly 7 interchangeable fractal primitives and run them through a deterministic parallel lifecycle tournament in Rust.

This `v3.1` spec is the buildable successor to `v3`. It preserves the doctrine, but fixes the broken contracts that would otherwise make the crate internally inconsistent or non-compiling.

## Non-Negotiable Doctrine

- One single reusable transition rule per primitive: `apply`.
- All recurrent state evolution starts with `rule.apply(...)`.
- No attention, no transformers, no external REPL, no meta-loss hacks.
- Pure Rust with Burn using the Candle backend.
- The control plane must be explicit and typed. Primitive state layout may not be inferred from ad hoc shape heuristics.

## Why v3.1 Exists

The original `v3` draft had four architectural problems:

1. It mixed raw Candle tensors in the primitive trait with Burn-owned training in the lifecycle.
2. It pinned `candle-core = 0.8.x` while `burn-candle = 0.20.0` pulls in `candle-core = 0.9.1`.
3. It claimed a single interchangeable `apply(state, x)` contract while actually requiring several incompatible state shapes.
4. The Mandelbrot and hierarchical pseudocode was not mathematically complete enough to implement directly.

`v3.1` fixes those issues by making Burn the only tensor/control-plane surface, using Candle only as the backend through `burn-candle`, and introducing an explicit typed state contract.

## Workspace Layout

```text
fractal-rust-evolver/
├── Cargo.toml
├── fractal-core/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── error.rs
│       ├── state.rs
│       ├── rule_trait.rs
│       ├── primitives/
│       │   ├── mod.rs
│       │   ├── p1_contractive.rs
│       │   ├── p2_mandelbrot.rs
│       │   ├── p3_hierarchical.rs
│       │   ├── b1_fractal_gated.rs
│       │   ├── b2_stable_hierarchical.rs
│       │   ├── b3_fractal_hierarchical.rs
│       │   └── b4_universal.rs
│       ├── data_generator.rs
│       ├── router.rs
│       ├── model.rs
│       ├── fitness.rs
│       ├── lifecycle.rs
│       └── tests.rs
├── examples/
│   └── tournament.rs
└── README.md
```

## Root Cargo.toml

```toml
[workspace]
members = ["fractal-core"]
resolver = "2"

[workspace.package]
edition = "2021"

[workspace.dependencies]
burn = "0.20.0"
burn-candle = "0.20.0"
rand = "0.8"
```

## Dependency Rules

- Do not depend on a second standalone `candle-core` version unless it is pinned to exactly the same version transitively used by `burn-candle`.
- For `v3.1`, the public tensor API is Burn only.
- Candle is present only as the execution backend.

## Core Tensor and State Contract

### Backend Ownership

All tensors in the public crate API are Burn tensors:

```rust
use burn::tensor::{backend::Backend, Tensor};
```

### Explicit State Layout

All primitive state travels through a typed enum.

```rust
pub enum FractalState<B: Backend> {
    Flat(Tensor<B, 2>),                // [batch, dim]
    Complex(Tensor<B, 2>),             // [batch, 2 * dim]
    Hierarchical(Tensor<B, 3>),        // [batch, levels, dim]
    HierarchicalComplex(Tensor<B, 3>), // [batch, levels, 2 * dim]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateLayout {
    Flat,
    Complex,
    Hierarchical { levels: usize },
    HierarchicalComplex { levels: usize },
}
```

### Primitive Trait

```rust
pub trait FractalRule<B: Backend>: Send + Sync {
    fn apply(
        &self,
        state: &FractalState<B>,
        x: &Tensor<B, 2>,
    ) -> Result<FractalState<B>, FractalError>;

    fn name(&self) -> &'static str;
    fn hidden_dim(&self) -> usize;
    fn state_layout(&self) -> StateLayout;
    fn clone_box(&self) -> Box<dyn FractalRule<B>>;
}
```

### Trait Invariants

- `apply` must preserve the state layout declared by `state_layout()`.
- `hidden_dim()` is the base hidden width `dim`.
- Complex layouts store width `2 * dim`.
- Hierarchical layouts store `levels >= 2`.
- Input `x` always has shape `[batch, dim]`.
- `apply` returns a full new state and never mutates state in place.

## Recurrent Doctrine Enforcement

The only reusable transition operator is `apply`.

The model may contain:

- token embedding
- output projection to logits
- an early-exit router

The model may not contain any second recurrent transition primitive. No extra recurrent cell, no residual controller, no attention block, and no hidden transition outside `rule.apply`.

This must be enforced by tests with a counting mock rule.

## Shared Helper Definitions

### Flat Input Projection

Each primitive may contain learned linear projections from `x` into its own state width.

- For flat rules, projected width is `dim`.
- For complex rules, projected width is `2 * dim`.

### Complex Square

The Mandelbrot family uses real and imaginary channels packed side by side. `Complex` and `HierarchicalComplex` states must use a real implementation of complex square:

```text
If z = [re || im], then square(z) = [re^2 - im^2 || 2 * re * im]
```

Elementwise `state * state` is not valid for complex squaring and must not be used.

### Hierarchical Update Order

All hierarchical primitives update levels bottom-up for every token step.

For level `k`:

- level `0` has no lower contribution
- level `k > 0` uses the already updated `k - 1` level from the current token step

This makes the hierarchy explicit and deterministic.

### Compression Operator

Each hierarchical primitive owns a learned `compressor` projection:

```text
compress : R^[batch, width] -> R^[batch, width]
```

where `width = dim` for real-valued hierarchy and `width = 2 * dim` for complex hierarchy.

## Exact Primitive Definitions

All 7 primitives must be implemented exactly as defined below.

### Common Notation

- `x_t`: current token embedding, shape `[batch, dim]`
- `h_{t-1}`: previous flat real state, shape `[batch, dim]`
- `z_{t-1}`: previous flat complex state, shape `[batch, 2 * dim]`
- `h_{t-1}^{(k)}`: previous real state at hierarchy level `k`
- `z_{t-1}^{(k)}`: previous complex state at hierarchy level `k`
- `u_t = U x_t`
- `c_t = C x_t`
- `g_t = sigmoid(G x_t)`
- `alpha_t = sigmoid(A x_t)`
- `beta_t = sigmoid(B x_t)`
- `gamma_t = sigmoid(H x_t)`

All gates broadcast elementwise over batch and hidden width.

### 1. `p1_contractive`

Minimal contractive gated recurrence.

State layout:

- `Flat`

Formula:

```text
g_t = sigmoid(G x_t)
m_t = W_h h_{t-1} + U x_t
h_t = g_t ⊙ m_t + (1 - g_t) ⊙ h_{t-1}
```

### 2. `p2_mandelbrot`

Mandelbrot-inspired flat complex oscillator.

State layout:

- `Complex`

Formula:

```text
g_t = clamp(sigmoid(G x_t), 0.0, 0.9)
c_t = C x_t
z_t = g_t ⊙ square(z_{t-1}) + c_t
```

`G x_t` and `C x_t` project into width `2 * dim`.

### 3. `p3_hierarchical`

Hierarchical selective state compressor.

State layout:

- `Hierarchical { levels }`

Formula:

```text
u_t = U x_t
alpha_t = sigmoid(A x_t)
beta_t = sigmoid(B x_t)
gamma_t = sigmoid(H x_t)

h_t^(0) = alpha_t ⊙ h_{t-1}^(0) + beta_t ⊙ u_t
h_t^(k) = alpha_t ⊙ h_{t-1}^(k)
         + beta_t ⊙ u_t
         + gamma_t ⊙ compress(h_t^(k-1))
```

for `k = 1 .. levels - 1`.

### 4. `b1_fractal_gated`

Blend of contractive gating and Mandelbrot dynamics.

State layout:

- `Complex`

Formula:

```text
g_t = sigmoid(G x_t)
c_t = C x_t
z_t = g_t ⊙ (square(z_{t-1}) + c_t) + (1 - g_t) ⊙ z_{t-1}
```

### 5. `b2_stable_hierarchical`

Blend of contractive gating and hierarchical compression.

State layout:

- `Hierarchical { levels }`

Formula:

```text
g_t = sigmoid(G x_t)
u_t = U x_t
base_t^(k) = g_t ⊙ (W_h h_{t-1}^(k) + u_t) + (1 - g_t) ⊙ h_{t-1}^(k)

h_t^(0) = base_t^(0)
h_t^(k) = base_t^(k) + gamma_t ⊙ compress(h_t^(k-1))
```

for `k = 1 .. levels - 1`, with `gamma_t = sigmoid(H x_t)`.

### 6. `b3_fractal_hierarchical`

Blend of Mandelbrot dynamics and hierarchical compression.

State layout:

- `HierarchicalComplex { levels }`

Formula:

```text
g_t = clamp(sigmoid(G x_t), 0.0, 0.9)
c_t = C x_t

z_t^(0) = g_t ⊙ square(z_{t-1}^(0)) + c_t
z_t^(k) = g_t ⊙ square(z_{t-1}^(k))
         + c_t
         + gamma_t ⊙ compress(z_t^(k-1))
```

for `k = 1 .. levels - 1`, with `gamma_t = sigmoid(H x_t)`.

### 7. `b4_universal`

Blend of contractive gating, Mandelbrot dynamics, and hierarchical compression.

State layout:

- `HierarchicalComplex { levels }`

Formula:

```text
g_t = sigmoid(G x_t)
c_t = C x_t
base_t^(k) = g_t ⊙ (square(z_{t-1}^(k)) + c_t) + (1 - g_t) ⊙ z_{t-1}^(k)

z_t^(0) = base_t^(0)
z_t^(k) = base_t^(k) + gamma_t ⊙ compress(z_t^(k-1))
```

for `k = 1 .. levels - 1`, with `gamma_t = sigmoid(H x_t)`.

## Default Hyperparameters

These values are the required defaults for a first successful build and tournament run:

- `dim = 128`
- `levels = 4` for all hierarchical species
- `vocab_size = 64`
- `max_seq_len = 128`
- `max_recursion_depth = 20`
- `router_threshold = 0.90`
- `batch_size = 16`
- `train_steps_per_species = 50`
- `eval_batches_per_family = 8`
- `device = Candle CPU backend`
- `seed = 42`

These defaults are intentionally conservative so `cargo run --example tournament` can complete on a typical CPU-only machine.

## Data Generator

`data_generator.rs` must implement `SimpleHierarchicalGenerator`.

### Output Contract

The generator returns integer token batches ready for Burn training:

```rust
pub struct TokenBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, burn::tensor::Int>,
    pub target_ids: Tensor<B, 2, burn::tensor::Int>,
    pub family: TaskFamily,
}
```

`target_ids` is next-token supervision, aligned by left shift from `input_ids`.

### Task Families

#### 1. Recursive Sentence Nesting

Generate deterministic synthetic sentences with nesting depth `1..=8`.

Required properties:

- nested relative-clause style recursion
- bounded vocabulary
- variable length
- depth label recoverable from token structure

Example shape:

```text
BOS ROOT CLAUSE_OPEN NOUN REL CLAUSE_OPEN NOUN VERB CLAUSE_CLOSE VERB CLAUSE_CLOSE EOS
```

#### 2. ARC-Style Fractal Grid Patterns

Generate tokenized recursive tile-expansion sequences.

Required properties:

- start from a small base tile
- recursively expand via repeat and substitution
- serialize the resulting grid into a token sequence
- include at least two recursion depths in training and a deeper held-out split for evaluation

This is an ARC-style synthetic family, not a full ARC benchmark loader.

### Generator Requirements

- deterministic under seed
- no file I/O
- no external datasets
- no tokenizer training
- generate train and eval splits entirely in code

## Router

`router.rs` implements a simple per-token early-exit controller.

### Router Contract

The router receives a readout view of the current state after each recursive `apply`.

```text
exit_score = sigmoid(W_r readout(state) + b_r)
exit if exit_score >= threshold
```

The router may only decide whether to stop recursion for the current token. It may not alter the state or inject new transition logic.

### Readout View

The model uses a deterministic state readout:

- `Flat`: use the flat tensor directly
- `Complex`: use the complex tensor directly
- `Hierarchical`: use the top level `levels - 1`
- `HierarchicalComplex`: use the top level `levels - 1`

The output projection head consumes this readout view.

## Model

`model.rs` implements the sequence model and is the only location where token unrolling happens.

### Model Structure

- token embedding
- recurrent unroll using only `rule.apply`
- early-exit router
- output projection head

### Per-Token Step

For each token `x_t`:

1. Embed token to `x_t`.
2. Recurse from the current state using only repeated `rule.apply`.
3. After each application, query the router.
4. Stop at the first router exit or at `max_recursion_depth`.
5. Emit logits from the final state readout.

No additional recurrent operator is allowed.

## Lifecycle Tournament

`lifecycle.rs` defines `Tournament`.

### Species Set

The tournament always instantiates exactly these 7 species:

1. `p1_contractive`
2. `p2_mandelbrot`
3. `p3_hierarchical`
4. `b1_fractal_gated`
5. `b2_stable_hierarchical`
6. `b3_fractal_hierarchical`
7. `b4_universal`

### Parallel Execution

`run_generation()` must run the 7 species in parallel using standard Rust threads.

Required properties:

- fixed seed derivation per species
- isolated model instance per thread
- isolated optimizer state per thread
- no shared mutable model parameters across species

### Training Loop

Each species receives:

- the same train split
- the same eval split
- the same step budget
- the same optimizer configuration

Initial tournament target:

- one short training phase per species
- one evaluation pass per species
- one ranked output table

`v3.1` does not require mutation or offspring generation. It requires a parallel survival competition and ranked survivors after a generation.

## Fitness

`fitness.rs` defines four required metrics and one aggregate score.

### 1. Stability

Measure gradient norm at recursion depth `20` on a fixed held-out batch.

Score:

```text
stability_score = 1 / (1 + grad_norm_depth_20)
```

If the norm is NaN or infinite, the score is `0`.

### 2. Long-Context Perplexity

Measure perplexity on held-out recursive sentence sequences with depth `8` and long lengths.

Score contribution uses inverse perplexity:

```text
perplexity_score = 1 / max(perplexity, 1.0)
```

### 3. ARC-Style Accuracy

Measure exact next-token accuracy on the held-out grid-pattern family.

```text
arc_score = correct_tokens / total_tokens
```

### 4. Throughput

Measure tokens processed per second during evaluation.

```text
speed_score = species_tokens_per_sec / best_tokens_per_sec_in_generation
```

### Aggregate Fitness

```text
fitness =
    0.35 * stability_score +
    0.30 * perplexity_score +
    0.25 * arc_score +
    0.10 * speed_score
```

Higher is better.

## Hot-Swap and Cloning

All species must support hot-swap through `clone_box`.

Required behavior:

- cloning preserves parameter values
- cloning preserves layout metadata
- cloned rules can be inserted into a fresh model instance without manual shape repair

## Acceptance Criteria

The implementation is accepted only when all of the following are true:

1. `cargo build --release` passes.
2. `cargo test` passes.
3. `cargo run --example tournament` runs one full generation with all 7 species.
4. The example prints ranked results including all four metrics and total fitness.
5. All recurrent state transitions in the model occur through repeated `rule.apply`.
6. No crate introduces attention, transformers, external datasets, or external services.

## Required Tests

The crate must include at least these tests:

1. `complex_square` correctness against hand-computed values.
2. Every primitive preserves its declared state layout and shape.
3. `clone_box` preserves rule identity and metadata.
4. Hierarchical rules update all levels without shape collapse.
5. The router never changes state width or layout.
6. `Tournament::run_generation()` returns exactly 7 ranked results.
7. A counting mock verifies that the recurrent model path uses only `rule.apply` for state transitions.

## Example Output Contract

`cargo run --example tournament` must print a table similar to:

```text
rank  species                  stability  perplexity  arc_acc  tok/s   fitness
1     b4_universal             0.81       5.42        0.74     18200   0.71
2     b2_stable_hierarchical   0.88       6.10        0.69     20500   0.69
3     p1_contractive           0.93       7.40        0.61     24800   0.67
...
7     p2_mandelbrot            0.22       21.90       0.31      9700   0.24
```

Exact numbers need not match, but all columns must be present.

## Implementation Notes

- Start with CPU-only Candle backend through `burn-candle`.
- Keep `dim`, `levels`, and `max_recursion_depth` configurable through constructors.
- Prefer explicit helper functions for layout unpacking over stringly typed branching.
- Do not hide shape repair in model code. State conversion is a bug, not a convenience.
- If a primitive needs a different width, express it through `StateLayout`, not an implicit tensor reshape.

## Summary

`v3.1` keeps the 7 requested primitives and the survival-tournament objective, but makes the system buildable by:

- using Burn tensors everywhere
- using Candle only as Burn's backend
- defining an explicit state enum
- fixing complex squaring
- fixing hierarchical update semantics
- making tournament fitness and output deterministic enough to test
