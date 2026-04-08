# DREEGMOR Mini-MoE API Sketch

## Status

Design contract for implementation.

This document turns the mini-MoE design into explicit typed Rust-facing boundaries so the control plane remains honest as the architecture evolves.

Related:

- [Mini-MoE design](/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-mini-moe-design.md)
- [Experiment rubric](/Users/joseph/fractal-worktrees/goe-a-am-experiments/docs/specs/dreegmor-experiment-rubric.md)

## Objectives

- Preserve a faithful scaled-MoE structure in miniature
- Keep the reference and thesis models aligned except for router behavior
- Make routing, dispatch, and expert execution explicit typed surfaces
- Support controlled extensibility without turning the code into an unstructured feature pile

## Top-Level Pattern

The implementation should separate:

- config/control plane
- block/runtime plane
- router/controller plane
- dispatch plane
- reporting/observability plane

The key architectural rule is:

- routers produce a typed `RoutePlan`
- dispatchers consume that `RoutePlan`
- experts do not decide routing
- models do not hide routing decisions inside opaque blocks

## Core Types

### Surface spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeSurfaceSpec {
    pub architecture: MiniMoeArchitectureSpec,
    pub runtime: MiniMoeRuntimeSpec,
    pub observability: MiniMoeObservabilitySpec,
}
```

### Architecture spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeArchitectureSpec {
    pub schema_version: u32,
    pub preset: Option<MiniMoePreset>,
    pub label: String,
    pub backbone: MiniMoeBackboneSpec,
    pub moe: MiniMoeStackSpec,
    pub router: MiniMoeRouterSpec,
}
```

### Backbone spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeBackboneSpec {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub head_count: usize,
    pub total_layers: usize,
    pub local_window: usize,
    pub ffn_multiplier: usize,
}
```

### MoE stack spec

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeStackSpec {
    pub experts_per_block: usize,
    pub active_experts_per_token: usize,
    pub moe_layer_schedule: MiniMoeLayerSchedule,
    pub expert_ffn_multiplier: usize,
    pub load_balance_loss_weight: f64,
}
```

### Layer schedule

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeLayerSchedule {
    AllLayers,
    EveryN { n: usize },
    Explicit(Vec<usize>),
}
```

`EveryN { n }` must mean:

- select layers `0, n, 2n, 3n, ...`
- stop before `total_layers`

The starting offset must not be implied anywhere else in the codebase.

### Layer schedule helper preset

For human-facing convenience and common ablations, the control plane should also support a named helper layer:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeLayerSchedulePreset {
    AllLayers,
    AlternatingFromZero,
    AlternatingFromOne,
}
```

This helper should be lowered into `MiniMoeLayerSchedule` before model build.

Recommended lowering:

- `AllLayers` -> `MiniMoeLayerSchedule::AllLayers`
- `AlternatingFromZero` -> `MiniMoeLayerSchedule::Explicit(vec![0, 2, 4, ...])`
- `AlternatingFromOne` -> `MiniMoeLayerSchedule::Explicit(vec![1, 3, 5, ...])`

The main experiment line should still use the explicit schedule contract, not depend on helper presets at runtime.

### Resolved layout

The raw schedule enum should not be consulted after build-time resolution.

Instead, the builder should materialize a canonical resolved layout:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedMiniMoeLayout {
    pub moe_layers: Vec<usize>,
    pub dense_layers: Vec<usize>,
}
```

`moe_layers` must be:

- sorted
- deduplicated
- in ascending order
- non-empty

`dense_layers` should be derived from `total_layers - moe_layers`.

### Router spec

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeRouterSpec {
    OneShot(OneShotRouterSpec),
    RecurrentPreExpert(RecurrentPreExpertRouterSpec),
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneShotRouterSpec {}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecurrentPreExpertRouterSpec {
    pub round_count: usize,
    pub state_dim: usize,
}
```

### Runtime spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeRuntimeSpec {
    pub dispatch: MiniMoeDispatchSpec,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeDispatchSpec {
    pub mode: MiniMoeDispatchMode,
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeDispatchMode {
    SparseTopK,
    DenseDebug,
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TieBreakPolicy {
    LowestIndex,
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchCapacityPolicy {
    Unlimited,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDispatchContract {
    pub mode: MiniMoeDispatchMode,
    pub active_experts_per_token: usize,
    pub tie_break: TieBreakPolicy,
    pub capacity: DispatchCapacityPolicy,
}
```

### Preset identity

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoePreset {
    Phase1Reference,
    Phase1Recurrent,
}
```

## Validation Contract

`MiniMoeSurfaceSpec::validate()` should delegate to the component specs and enforce cross-spec consistency.

`MiniMoeArchitectureSpec::validate()` should enforce:

- `schema_version` supported
- `label` non-empty
- `vocab_size > 0`
- `hidden_dim > 0`
- `head_count > 0`
- `hidden_dim % head_count == 0`
- `total_layers > 0`
- `local_window > 0`
- `ffn_multiplier > 0`
- `experts_per_block >= 2`
- `active_experts_per_token >= 1`
- `active_experts_per_token <= experts_per_block`
- `expert_ffn_multiplier > 0`
- `load_balance_loss_weight >= 0.0`
- `moe_layer_schedule` resolves successfully into a canonical `ResolvedMiniMoeLayout`
- all explicit layer indices are `< total_layers`
- `ResolvedMiniMoeLayout.moe_layers` is non-empty
- `ResolvedMiniMoeLayout.dense_layers.len() + ResolvedMiniMoeLayout.moe_layers.len() == total_layers`
- `OneShotRouterSpec` has no extra fields
- `RecurrentPreExpertRouterSpec.round_count >= 2`
- `RecurrentPreExpertRouterSpec.state_dim > 0`

`MiniMoeRuntimeSpec::validate()` should enforce:

- `dispatch.mode` is explicitly chosen

`MiniMoeSurfaceSpec::validate()` should also enforce:

- `DenseDebug` is allowed for smoke and debugging only
- benchmark surfaces must use `SparseTopK`
- `ResolvedDispatchContract.active_experts_per_token == architecture.moe.active_experts_per_token`
- `ResolvedDispatchContract.tie_break` is explicit
- `ResolvedDispatchContract.capacity` is explicit

The schedule should normalize into a canonical resolved layout:

```rust
impl MiniMoeLayerSchedule {
    pub fn resolve(&self, total_layers: usize) -> Result<ResolvedMiniMoeLayout, FractalError>;
}
```

The helper preset should also have an explicit lowering path:

```rust
impl MiniMoeLayerSchedulePreset {
    pub fn lower(self, total_layers: usize) -> MiniMoeLayerSchedule;
}
```

## Runtime Types

### Model

```rust
#[derive(Module, Debug)]
pub struct MiniMoeModel<B: Backend, F: FfnSublayer<B>> {
    embedding: Embedding<B>,
    blocks: Vec<MiniTransformerBlock<B, F>>,
    output: LanguageModelHead<B>,
    architecture: Ignored<MiniMoeArchitectureSpec>,
    runtime: Ignored<MiniMoeRuntimeSpec>,
    resolved_layout: Ignored<ResolvedMiniMoeLayout>,
}
```

### Block

```rust
#[derive(Module, Debug)]
pub struct MiniTransformerBlock<B: Backend, F: FfnSublayer<B>> {
    pre_attention_norm: SimpleNorm<B>,
    attention: SharedAttentionSublayer<B>,
    pre_ffn_norm: SimpleNorm<B>,
    ffn: F,
}
```

This block is the main reusable abstraction.

- attention path remains shared
- FFN slot remains explicit
- the FFN implementation can be swapped without rewriting the block

The block contract should treat residual structure as explicit:

- input -> pre-attention norm -> attention -> residual add
- intermediate hidden -> pre-FFN norm -> `FfnSublayer` -> residual add

Recommended block forward output:

```rust
pub struct BlockForwardOutput<B: Backend, T> {
    pub hidden: Tensor<B, 3>,
    pub ffn_trace: T,
}
```

### Shared attention

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedAttentionSpec {
    pub hidden_dim: usize,
    pub head_count: usize,
    pub local_window: usize,
}
```

```rust
#[derive(Module, Debug)]
pub struct SharedAttentionSublayer<B: Backend> {
    spec: Ignored<SharedAttentionSpec>,
    inner: TransformerEncoder<B>,
}
```

```rust
pub trait AttentionSublayer<B: Backend>: Module<B> + Clone {
    fn forward(
        &self,
        hidden: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> Result<AttentionForwardOutput<B>, FractalError>;
}
```

```rust
pub struct AttentionForwardOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub trace: Option<AttentionTrace<B>>,
}
```

```rust
pub struct AttentionTrace<B: Backend> {
    pub attention_scores: Option<Tensor<B, 4>>,
}
```

Phase 1 implementation note:

- it is acceptable to wrap Burn’s attention-capable internals here initially
- but the FFN seam must remain explicit at the block level
- the attention contract itself must be ours, not Burn's
- the attention sublayer should receive the mask explicitly rather than implicitly rebuilding it
- `trace` may remain `None` in phase 1, but the contract should not make later attention diagnostics impossible

### FFN sublayer trait

```rust
pub trait FfnSublayer<B: Backend>: Module<B> + Clone {
    type Trace;

    fn forward(
        &self,
        hidden: Tensor<B, 3>,
    ) -> Result<FfnForwardOutput<B, Self::Trace>, FractalError>;
}
```

```rust
pub struct FfnForwardOutput<B: Backend, T> {
    pub hidden: Tensor<B, 3>,
    pub trace: T,
}
```

### Expert FFN

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertId {
    pub layer_index: usize,
    pub expert_index: usize,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertActivationKind {
    Gelu,
    Silu,
    Relu,
    SquaredRelu,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertFfnSpec {
    pub hidden_dim: usize,
    pub expansion_dim: usize,
    pub activation: ExpertActivationKind,
    pub gated: bool,
}
```

```rust
#[derive(Module, Debug)]
pub struct ExpertFfn<B: Backend> {
    id: Ignored<ExpertId>,
    spec: Ignored<ExpertFfnSpec>,
    up_projection: StructuredProjection<B>,
    gate_projection: Option<StructuredProjection<B>>,
    down_projection: StructuredProjection<B>,
}
```

```rust
impl<B: Backend> ExpertFfn<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>) -> Result<Tensor<B, 3>, FractalError>;
}
```

Phase 1:

- standard FFN-style routed expert
- homogeneous expert shape within a block
- homogeneous activation policy across the model
- one routed expert instance per expert slot

The expert contract should be explicit about shape:

- input shape: `[batch, seq, hidden_dim]`
- output shape: `[batch, seq, hidden_dim]`
- expansion happens internally
- the expert is position-wise, not sequence-mixing

This keeps the expert layer aligned with scaled MoE FFN practice while preserving room for future gated or alternative FFN variants.

## Router / Dispatch Boundary

### Router trait

```rust
#[derive(Debug)]
pub struct RouteSiteId {
    pub layer_index: usize,
}
```

```rust
pub struct PreExpertRouterInput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub site: RouteSiteId,
}
```

```rust
pub trait PreExpertRouterController<B: Backend>: Module<B> + Clone {
    fn route(
        &self,
        input: PreExpertRouterInput<B>,
    ) -> Result<RoutePlan<B>, FractalError>;
}
```

This is the key abstraction boundary.

- routers do not execute experts
- routers do not return mixed outputs
- routers produce dense routing intent
- routers are explicitly scoped to pre-expert routing in phase 1

### Route plan

```rust
#[derive(Debug)]
pub struct RoutePlan<B: Backend> {
    pub expert_logits: Tensor<B, 3>,
    pub expert_weights: Tensor<B, 3>,
    pub round_summaries: Vec<RouteRoundSummary<B>>,
}
```

```rust
#[derive(Debug)]
pub struct RouteRoundSummary<B: Backend> {
    pub expert_logits: Tensor<B, 3>,
    pub expert_weights: Tensor<B, 3>,
}
```

`RoutePlan` should represent dense routing intent:

- all per-expert logits
- all per-expert weights
- per-round controller evolution

It should not commit to sparse selection policy.

### Dispatch plan

```rust
#[derive(Debug)]
pub struct DispatchPlan<B: Backend> {
    pub site: RouteSiteId,
    pub mode: MiniMoeDispatchMode,
    pub selected_expert_indices: Tensor<B, 3, Int>,
    pub selected_expert_weights: Tensor<B, 3>,
}
```

### Dispatcher

```rust
#[derive(Debug, Clone)]
pub struct MoeDispatcher {
    contract: Ignored<ResolvedDispatchContract>,
}
```

```rust
impl MoeDispatcher {
    pub fn compile<B: Backend>(
        &self,
        site: RouteSiteId,
        route_plan: &RoutePlan<B>,
    ) -> Result<DispatchPlan<B>, FractalError>;

    pub fn dispatch<B: Backend>(
        &self,
        hidden: Tensor<B, 3>,
        routed_experts: &[ExpertFfn<B>],
        dispatch_plan: &DispatchPlan<B>,
    ) -> Result<Tensor<B, 3>, FractalError>;
}
```

The dispatcher owns:

- execution-policy compilation
- dispatch-plan compilation
- sparse combine semantics
- output mixing semantics

The router owns:

- dense logits
- dense weights
- per-round state evolution
- final routing intent

Phase 1 execution semantics should be explicit:

- `SparseTopK`: compile sparse top-k selections from `RoutePlan`
- `DenseDebug`: execute all routed experts using dense `expert_weights`

The dispatcher must not choose `active_experts_per_token` independently.

That value belongs to the architecture contract and should be carried in `ResolvedDispatchContract`.

Tie behavior must also be explicit and deterministic.

Phase 1 recommendation:

- `TieBreakPolicy::LowestIndex`
- `DispatchCapacityPolicy::Unlimited`

## Concrete Routers

### One-shot router

```rust
#[derive(Module, Debug)]
pub struct OneShotRouter<B: Backend> {
    route_projection: StructuredProjection<B>,
}
```

Behavior:

- single pass from hidden state to expert logits
- `round_count == 1`
- no controller state
- produces dense route intent, not top-k indices

### Recurrent pre-expert router

```rust
#[derive(Module, Debug)]
pub struct RecurrentPreExpertRouter<B: Backend> {
    token_state_projection: StructuredProjection<B>,
    state_logits_projection: StructuredProjection<B>,
    route_feedback_projection: StructuredProjection<B>,
    reset_gate_projection: StructuredProjection<B>,
    update_gate_projection: StructuredProjection<B>,
    candidate_input_projection: StructuredProjection<B>,
    candidate_state_projection: StructuredProjection<B>,
    spec: Ignored<RecurrentPreExpertRouterSpec>,
}
```

Behavior:

- routes iteratively before expert execution
- maintains controller state
- emits final dense routing intent after the last round
- no expert-feedback in phase 1

## MoE FFN Module

```rust
#[derive(Module, Debug)]
pub struct MoeFfnSublayer<B: Backend, R: PreExpertRouterController<B>> {
    router: R,
    routed_experts: Vec<ExpertFfn<B>>,
    dispatcher: Ignored<MoeDispatcher>,
}
```

```rust
impl<B: Backend, R: PreExpertRouterController<B>> MoeFfnSublayer<B, R> {
    pub fn forward(
        &self,
        input: PreExpertRouterInput<B>,
    ) -> Result<MoeFfnForwardOutput<B>, FractalError>;
}
```

```rust
pub struct MoeFfnForwardOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub route_plan: RoutePlan<B>,
    pub dispatch_plan: DispatchPlan<B>,
}
```

Returning both the `RoutePlan` and the `DispatchPlan` from forward is important for observability.

The route plan tells us:

- what the router wanted
- how recurrent routing evolved across rounds

The dispatch plan tells us:

- what sparse selection policy actually executed

`MoeFfnSublayer` should implement `FfnSublayer`.

Future FFN implementations should also satisfy the same trait:

```rust
pub struct DenseFfnSublayer<B: Backend> { ... }
pub struct OneShotMoeFfnSublayer<B: Backend> { ... }
pub struct RecurrentMoeFfnSublayer<B: Backend> { ... }
```

The block should be generic over the FFN slot, not the router specifically.

## Observability Contract

The observability surface must support three explanation levels:

- routing mechanics
- functional specialization
- controller reasoning trace

We should be able to explain the model in simple terms later only if the runtime preserves enough structured trace information now.

### Level 1: routing mechanics

Phase 1 reports must be able to compute:

- expert usage counts
- mean routing weights
- top-k selection histogram
- route entropy
- recurrent round deltas
- load-balance auxiliary loss
- layer-by-layer active expert counts

Recommended reduced summary type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRoutingSummary {
    pub sampled_tokens: usize,
    pub layer_count: usize,
    pub round_count: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_expert_weights: Vec<f64>,
    pub winner_counts: Vec<usize>,
    pub active_expert_count: usize,
    pub mean_round_adjustment_l1: Vec<f64>,
}
```

### Level 2: functional specialization

The contract should preserve enough information to summarize what experts appear to specialize in.

This does not mean we must solve semantic interpretability in phase 1. It means the trace surface should make later summary passes possible.

Recommended reduced per-layer expert usage type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertUsageSummary {
    pub site: RouteSiteId,
    pub expert_id: usize,
    pub selection_count: usize,
    pub mean_weight: f64,
    pub representative_token_examples: Vec<String>,
}
```

Recommended reduced layer summary type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerRouteSummary {
    pub site: RouteSiteId,
    pub sampled_tokens: usize,
    pub expert_usage: Vec<ExpertUsageSummary>,
    pub route_entropy_bits: f64,
    pub reroute_fraction: f64,
}
```

### Level 3: controller reasoning trace

For recurrent routing, we need to preserve how decisions changed across rounds.

This is the closest thing we will have to a simple description of the model's abstract "thinking process."

Recommended reduced per-token trace type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenRouteTrace {
    pub token_text: String,
    pub site: RouteSiteId,
    pub round_expert_indices: Vec<Vec<usize>>,
    pub round_expert_weights: Vec<Vec<f64>>,
    pub final_expert_indices: Vec<usize>,
    pub confidence_margin: f64,
    pub rerouted: bool,
}
```

Recommended reduced recurrent round summary type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerRoundSummary {
    pub site: RouteSiteId,
    pub round_index: usize,
    pub mean_route_entropy_bits: f64,
    pub mean_winner_margin: f64,
    pub mean_route_adjustment_l1: Option<f64>,
    pub rerouted_token_fraction: f64,
}
```

### Dispatch execution summary

Reduced reporting should also preserve what actually executed:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DispatchSummary {
    pub site: RouteSiteId,
    pub mode: MiniMoeDispatchMode,
    pub selected_expert_counts: Vec<usize>,
    pub dropped_token_fraction: Option<f64>,
}
```

Phase 1:

- `dropped_token_fraction` should be `None`
- this field exists so later capacity-limited dispatch does not require an observability redesign

### Runtime events

The runtime should emit bounded typed events rather than only report-shaped outputs.

Recommended event surface:

```rust
#[derive(Debug)]
pub enum MiniMoeTraceEvent<B: Backend> {
    RoutePlanned {
        site: RouteSiteId,
        route_plan: RoutePlan<B>,
    },
    DispatchCompiled {
        dispatch_plan: DispatchPlan<B>,
    },
    TokenExampleObserved {
        site: RouteSiteId,
        token_text: String,
        final_expert_indices: Vec<usize>,
        final_expert_weights: Vec<f64>,
    },
}
```

### Runtime trace bundle

The forward/runtime surface should preserve a bounded retained bundle that can later be reduced into human-readable summaries.

Recommended retained runtime bundle:

```rust
#[derive(Debug)]
pub struct MiniMoeTraceBundle {
    pub layer_summaries: Vec<LayerRouteSummary>,
    pub dispatch_summaries: Vec<DispatchSummary>,
    pub controller_round_summaries: Vec<ControllerRoundSummary>,
    pub sampled_token_traces: Vec<TokenRouteTrace>,
}
```

### Reduced report summary

Reporting should lower the retained runtime bundle into a compact report-facing summary.

Recommended report-facing root type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeReportSummary {
    pub routing: MiniMoeRoutingSummary,
    pub layers: Vec<LayerRouteSummary>,
    pub dispatch: Vec<DispatchSummary>,
    pub controller_rounds: Vec<ControllerRoundSummary>,
    pub sampled_tokens: Vec<TokenRouteTrace>,
}
```

### Observability sink

The model should not own report policy directly.

Recommended sink boundary:

```rust
pub trait MiniMoeObservabilitySink<B: Backend> {
    fn record_route_plan(&mut self, site: RouteSiteId, route_plan: &RoutePlan<B>);
    fn record_dispatch_plan(&mut self, dispatch_plan: &DispatchPlan<B>);
    fn record_token_example(
        &mut self,
        site: RouteSiteId,
        token_text: &str,
        final_expert_indices: &[usize],
        final_expert_weights: &[f64],
    );
    fn finalize(self) -> MiniMoeTraceBundle;
}
```

### Reporting rule

The runtime should emit structured events into a bounded sink.

The reporting layer should compress the retained bundle into:

- aggregate metrics for benchmark tables
- short natural-language summaries for human understanding

Example future summaries:

- "Layer 3 routes punctuation-heavy tokens mostly to expert 1."
- "The recurrent router mainly sharpens confidence rather than flipping expert choices."
- "Most reroutes happen in later layers on uncertain tokens."

### Sampling policy

We should not try to retain full token traces for every token in every batch.

Instead, the contract should support:

- aggregate metrics over all sampled eval tokens
- bounded token trace sampling for inspection
- representative token example extraction per expert

Recommended sampling controls:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeObservabilitySpec {
    pub sampling: TraceSamplingPolicy,
    pub capture_round_summaries: bool,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceSamplingPolicy {
    pub token_trace_budget_per_layer: usize,
    pub expert_example_budget_per_layer: usize,
    pub deterministic: bool,
    pub sample_seed: u64,
}
```

Phase 1 recommendation:

- deterministic sampling should be enabled
- `sample_seed` should default from the main experiment seed unless explicitly overridden

### Route plan implication

Because observability is part of the architectural contract, `RoutePlan` should be treated as a first-class trace surface, not just a dispatch input.

That means:

- preserve dense logits
- preserve dense per-expert weights
- preserve per-round controller evolution
- keep the structure easy to serialize into summaries

`DispatchPlan` should also remain observable, but only for execution-policy interpretation:

- selected expert indices
- selected expert weights
- any future capacity overflow or dropped-token metadata

## Train / Eval Harness Contract

The harness must preserve the same control-plane discipline as the model.

That means:

- train, eval, backend, and isolation policy must be explicit typed surfaces
- requested specs and resolved contracts must both be recorded
- smoke and benchmark semantics must be enforceable in code
- memory / throughput benchmark claims must require isolated process execution

### Run manifest

Recommended root run contract:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRunManifest {
    pub surface: MiniMoeSurfaceSpec,
    pub resolved_layout: ResolvedMiniMoeLayout,
    pub resolved_dispatch: ResolvedDispatchContract,
    pub train: MiniMoeTrainSpec,
    pub eval: MiniMoeEvalSpec,
    pub backend: MiniMoeBackendSpec,
    pub benchmark_policy: BenchmarkPolicy,
    pub isolation_mode: ExecutionIsolationMode,
}
```

The manifest should be the canonical record of what actually ran.

### Train spec

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeTrainSpec {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub model_seed: u64,
    pub data_seed: Option<u64>,
}
```

The harness should keep model initialization randomness and data-order randomness separate.

### Eval spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeEvalSpec {
    pub eval_batches: usize,
    pub full_eval_pass: bool,
}
```

### Backend spec

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MiniMoeBackendSpec {
    pub backend: MiniMoeBackendKind,
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMoeBackendKind {
    Cpu,
    Metal,
}
```

### Benchmark policy

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPolicy {
    Smoke,
    Benchmark,
}
```

### Isolation mode

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionIsolationMode {
    SharedProcess,
    IsolatedProcess,
}
```

### Harness validation rules

The harness contract should enforce:

- `MiniMoeTrainSpec.steps > 0`
- `MiniMoeTrainSpec.batch_size > 0`
- `MiniMoeTrainSpec.learning_rate > 0.0`
- `MiniMoeEvalSpec.eval_batches > 0` unless `full_eval_pass == true`
- `BenchmarkPolicy::Benchmark` requires `ExecutionIsolationMode::IsolatedProcess`
- `BenchmarkPolicy::Benchmark` forbids `runtime.dispatch.mode = DenseDebug`
- `BenchmarkPolicy::Benchmark` should record throughput and process memory metrics

These should be enforced in code, not only in documentation.

### Run artifact

Recommended per-run artifact root:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeRunArtifact {
    pub manifest: MiniMoeRunManifest,
    pub summary: MiniMoeReportSummary,
    pub train_metrics: MiniMoeTrainMetrics,
    pub eval_metrics: MiniMoeEvalMetrics,
    pub system_metrics: MiniMoeSystemMetrics,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeTrainMetrics {
    pub initial_loss: f64,
    pub final_loss: f64,
    pub load_balance_aux_loss: Option<f64>,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeEvalMetrics {
    pub final_loss: f64,
    pub perplexity: f64,
}
```

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MiniMoeSystemMetrics {
    pub train_tokens_per_second: Option<f64>,
    pub eval_tokens_per_second: Option<f64>,
    pub overall_tokens_per_second: Option<f64>,
    pub process_memory_metric: Option<String>,
    pub peak_process_memory_mb: Option<f64>,
}
```

### Harness behavior

Recommended behavior:

- a shared-process multi-variant runner may exist for smoke only
- benchmark runs should execute one variant per process
- manifests should record resolved layout, resolved dispatch contract, and resolved sampling policy
- summary and artifact output should live under isolated per-run directories

Recommended run-level defaults:

- smoke runners default to `BenchmarkPolicy::Smoke`
- smoke runners may use `ExecutionIsolationMode::SharedProcess`
- benchmark runners default to `BenchmarkPolicy::Benchmark`
- benchmark runners must use `ExecutionIsolationMode::IsolatedProcess`
- resolved dispatch defaults should use `TieBreakPolicy::LowestIndex`
- resolved dispatch defaults should use `DispatchCapacityPolicy::Unlimited`

This keeps benchmark comparisons honest while preserving fast smoke loops.

## Default Constructors

These should exist to lock the main comparison line:

```rust
impl MiniMoeSurfaceSpec {
    pub fn phase1_reference_default() -> Self;
    pub fn phase1_recurrent_default() -> Self;
}
```

Recommended defaults:

- `architecture.schema_version = 1`
- `architecture.preset = Some(Phase1Reference | Phase1Recurrent)`
- `total_layers = 8`
- `hidden_dim = 128`
- `head_count = 4`
- `ffn_multiplier = 4`
- `experts_per_block = 4`
- `active_experts_per_token = 1`
- `moe_layer_schedule = AllLayers`
- `expert_ffn_multiplier = 4`
- `load_balance_loss_weight > 0`
- `runtime.dispatch.mode = SparseTopK`
- `observability.sampling.token_trace_budget_per_layer` bounded but non-zero
- `observability.sampling.expert_example_budget_per_layer` bounded but non-zero
- `observability.sampling.deterministic = true`
- reference router:
  - `router = OneShot(OneShotRouterSpec {})`
- thesis router:
  - `router = RecurrentPreExpert(RecurrentPreExpertRouterSpec { round_count: 2, state_dim: 64 })`

## Concrete Type Aliases

```rust
pub type ReferenceMiniMoeModel<B> = MiniMoeModel<B, OneShotMoeFfnSublayer<B>>;
pub type RecurrentMiniMoeModel<B> = MiniMoeModel<B, RecurrentMoeFfnSublayer<B>>;
```

These aliases should be the main experiment surfaces.

## Explicit Non-Goals

Phase 1 should not include:

- external whole-model routers
- whole-backbone expert duplication
- graph priors
- expert-feedback recurrent routing
- shared experts
- top-2 or top-k greater than 1
- attention-expert MoE

## Immediate Implementation Order

1. `MiniMoeSurfaceSpec` and validation
2. `MiniMoeArchitectureSpec`, `MiniMoeRuntimeSpec`, and `MiniMoeObservabilitySpec`
3. `MiniMoeLayerSchedulePreset::lower`
4. `MiniMoeLayerSchedule::resolve`
5. `ResolvedDispatchContract`
6. `ExpertFfn`
7. `OneShotRouter`
8. `RecurrentPreExpertRouter`
9. `MoeDispatcher`
10. `DenseFfnSublayer`, `OneShotMoeFfnSublayer`, and `RecurrentMoeFfnSublayer`
11. `MiniTransformerBlock`
12. `MiniMoeModel`
13. `MiniMoeRunManifest`, `BenchmarkPolicy`, and `ExecutionIsolationMode`
14. observability sink and report reduction
15. train/eval/report harness

This order keeps the control plane explicit and the experiment surface narrow.
