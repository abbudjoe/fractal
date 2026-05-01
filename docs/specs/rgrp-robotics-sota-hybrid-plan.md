# RGRP Robotics SOTA Hybrid Plan

Date: 2026-04-30

Status: draft research plan, revised after external review

## Purpose

This plan defines a robotics architecture track that is parallel to, but not allowed to outrun, validating the legal ACT input surface. The ACT force/torque fix is not a side quest. It is the first control gate. If corrected ACT plus wrist force/torque, normalization, causal history, and auxiliary physics heads solves most contact failures, then RGRP becomes a research add-on rather than the competition-critical center.

The broader goal remains ambitious: build a state-of-the-art robot control stack around a compact recurrent physical belief-state bridge, then hybridize only the pieces that earn their place through matched controls.

The core hypothesis is:

> Contact-rich robot control needs a persistent physical belief state, not only more attention over observations. RGRP/Parcae may provide that missing recurrent latent state: compact, updateable, loop-refined, and efficient enough to sit between high-level world/future reasoning and low-level action execution.

The first serious hypothesis is narrower:

```text
corrected ACT with wrist force/torque
+ causal action/proprio/contact history
+ auxiliary physics prediction
+ compact recurrent physical belief-state bridge
```

Do not start with:

```text
Cosmos + DreamZero + Hyperloop + DeepSeek + RGRP + ACT
```

That full hybrid is too many moving parts and would be nearly uninterpretable.

## Current Trigger

The robot lane discovered that wrist force/torque was legally available but excluded from the ACT input projection. That likely explains many force/contact failures. However, this plan is not merely "add force/torque to ACT." That corrected baseline is a necessary control. The research track is to design a stronger hybrid controller that uses force/torque, proprioception, vision, and action history as inputs to a shared physical state.

External review sharpened the immediate risk:

```text
If contact failures are primarily caused by omitted force/torque, then
corrected ACT + force/torque + normalization + auxiliary heads may solve much
of the problem without RGRP.
```

Therefore the plan must prove RGRP adds value beyond the corrected sensor stack and beyond matched standard recurrent bridges.

## Ingredients

### 1. RGRP/Parcae

RGRP is our rotary gated recurrent state update primitive inside a Parcae loop scaffold.

At a high level:

```text
prelude attention / multimodal encoders
-> loop_input = normalized latent stream
-> RGRP updates compact recurrent state
-> Parcae reuses/refines a middle loop band
-> recurrent readout/injection returns to residual stream
-> coda/action decoder consumes refined latent
```

RGRP is not just a GRU and not a plain attention replacement. It is a compact recurrent controller that computes gated state updates, rotary/transition controls, candidate state values, and readout/injection values. The key value is persistent latent state:

```text
state_t = update(state_{t-1}, observation_t, action_{t-1}, force_t, tactile_t, proprio_t)
```

For robotics, RGRP should represent a physical belief state:

- current task phase;
- contact state;
- resistance/jam/slip evidence;
- object response to action;
- latent estimates of mass/friction/compliance;
- recent action consequences;
- uncertainty or mismatch between expected and observed state.

### 2. ACT

ACT remains valuable as a deterministic action chunk decoder. It should not be discarded. The likely role is:

```text
physical belief state + current observation
-> ACT action chunk
```

The corrected ACT baseline must include legal wrist force/torque, proprioception, and any other legal robot-state channels. But in the SOTA hybrid, ACT is the executor, not the whole brain.

### 3. Cosmos Reason2 8B / Qwen3-Instruct Backbone

Cosmos Reason2 8B provides high-level semantic reasoning, task interpretation, scene understanding, and future-state proposal/judgment. It should not directly output every low-level action.

Likely role:

```text
language/task + visual scene + candidate future
-> semantic plan / phase / constraint hints
```

Those hints condition the physical belief-state bridge and action decoder.

### 4. DreamZero / World Action Model / DiT

DreamZero-like DiT/WAM ideas are useful as a future/action predictor. It can propose or evaluate likely future visual states under actions.

Likely role:

```text
current observation + candidate action/phase
-> predicted future frames/states
-> RGRP compares predicted vs observed consequences
```

DreamZero is not a replacement for RGRP. It is a forward dynamics/future imagination axis. RGRP is the compact recurrent state and correction axis.

### 5. Hyperloop-Style Parallel Residual Streams

Hyperloop-style architectures report that a smaller looped model can match a larger standard Transformer on perplexity by using a looped begin/middle/end block structure plus hyper-connections over multiple residual streams.

The relevant recipe is:

```text
begin block
-> duplicate residual stream into n parallel streams
-> loop a middle block several times
-> after each loop, use hyper-connection mixing:
   pre: read/select from residual streams
   post: write update into streams
   res: mix/update streams
-> average or project streams
-> end block
```

The important missing ingredient for our stack is not simply "looping." We already have looped Parcae/RGRP bands. The missing ingredient is **multiple residual streams plus loop-level hyper-connections**. This may prevent repeated loop passes from collapsing into near-identical representations and may give RGRP a richer latent scratchpad to update.

More precisely, Hyperloop is not one trick. It is a bundle. We should not assume a single ingredient explains the claimed 1B-vs-2B parameter efficiency. The recipe should be isolated before it is imported into robotics:

- begin/middle/end loop topology;
- loop-position embedding;
- multiple residual streams;
- pre/post/res hyper-connection mixing;
- data-dependent diagonal stream transition;
- stream averaging or learned stream merge;
- paper-faithful training and optimizer assumptions;
- perplexity-first reporting.

Our current stack overlaps with the looped-middle idea, but does not yet isolate the rest of the recipe.

For robotics, the residual streams can map naturally onto parallel physical beliefs:

```text
stream 1: visual/object geometry
stream 2: force/contact state
stream 3: action phase/intention
stream 4: uncertainty/error correction
```

This should be tested as a structured extension of the RGRP bridge:

```text
multimodal latent
-> split into parallel physical residual streams
-> RGRP controls recurrent updates
-> loop-level hyper-connections mix streams
-> ACT consumes the merged physical belief state
```

Do not treat Hyperloop as a replacement for RGRP. Treat it as a residual-state topology that may make RGRP/Parcae scale better.

### 6. What RGRP Adds Beyond Hyperloop

RGRP is not "recurrence invented from scratch." Its ingredients overlap with known families:

- GRU/LSTM-style gated updates;
- Mamba/SSM/RWKV-style compact sequence state;
- rotary or complex transition dynamics;
- Gated DeltaNet-style targeted memory/update intuition;
- looped Transformer / Ouroboros / Hyperloop-style repeated latent computation.

The defensible novelty claim is compositional:

> RGRP is a rotary gated recurrent state update primitive integrated into a looped Parcae hourglass, where compact recurrent state controls latent refinement. In robotics, it becomes a physical belief-state bridge over force/torque, proprioception, action history, and visual latents.

What RGRP may add that Hyperloop alone does not:

- a persistent compact recurrent state across time;
- gated state transitions conditioned on action/observation consequences;
- a natural interface for force/torque and contact history;
- a controller/readout path that can update physical belief without requiring full attention over all prior tokens;
- a bridge between high-level semantic/future planning and low-level action chunking.

Hyperloop improves depth recurrence and residual routing. RGRP targets time-evolving belief-state update. For robotics, both may matter, but they should be tested separately.

### 7. DeepSeek V4-Inspired Innovations

DeepSeek V4 is not something to copy wholesale. The useful ideas are modular:

- Manifold-Constrained Hyper-Connections (mHC): stable residual/state mixing.
- Multi-Token Prediction (MTP): auxiliary prediction over short futures.
- Muon optimizer: possible faster convergence for newly trained modules.
- Compressed/sparse attention: useful if long-context robot traces become the bottleneck.
- MoE and FP4/quant infra: scale-only, not first-pass.

Most relevant now:

```text
mHC-style stable residual mixing
short-horizon auxiliary future/action prediction
Muon for trainable bridge modules
```

CSA/HCA, MoE, and quantized cache infrastructure belong to later scale lanes.

## Architectural North Star

The near-term architecture is a physical belief-state bridge, not a full VLA/DiT hybrid:

```text
inputs at control rate:
  RGB/depth or visual latent
  proprioception
  wrist force/torque
  tactile/contact if available
  previous action chunk
  previous gripper/contact state

        ↓

sensor-specific encoders:
  vision encoder
  proprio encoder
  force/torque encoder
  tactile/contact encoder
  action-history encoder

        ↓

shared physical latent z_t

        ↓

bridge candidate:
  corrected ACT baseline: identity / MLP
  recurrent controls: GRU/LSTM, Mamba/SSM, small Transformer
  research candidate: RGRP/Parcae

        ↓

same ACT action chunk decoder

        ↓

robot action chunk

auxiliary heads:
  next force/torque
  next proprio state
  contact / slip / jam
  object delta
  phase / failure label
  action residual
```

This is "one shared physical state, many auxiliary heads," not many separate ACT policies fighting each other.

Cosmos Reason2, DreamZero/WAM-DiT, DeepSeek-style residual machinery, and Hyperloop streams are later additions. They should not be placed in the inner control loop until the bridge candidate wins under corrected sensors.

Slow/medium/fast timescale split:

```text
fast inner loop:
  sensors -> encoders -> physical belief state -> ACT action chunk

medium loop:
  physical state -> phase/failure/contact predictor -> recovery adjustment

slow planner loop:
  Cosmos or 2-3B specialist -> phase/constraint/subgoal hints

world-model loop:
  DreamZero/WAM-DiT -> candidate futures or future-state teacher
```

## Why Not Multiple ACT Heads First?

Multiple ACT heads with role-specific sensors can fragment the policy:

```text
vision head says continue
force head says retreat
tactile head says regrip
proprio head says compensate
```

Then the system needs an arbitration layer, which becomes another control problem.

The preferred first SOTA design is:

```text
one shared latent state
one action policy / ACT decoder
multiple auxiliary heads that shape the latent
```

Multiple policy heads can be revisited later as MoE/action-expert routing after the shared-state design is understood.

## Experiment Lanes

### Lane A: Corrected ACT Control

Purpose: establish a sane baseline with legal inputs.

```text
vision + proprio + pose + wrist force/torque -> ACT
```

Variants:

- raw force/torque only;
- normalized force/torque;
- force/torque history window;
- force/torque derivatives or short-window max/mean if legal.

Audit requirements:

- units;
- sensor frame versus world frame;
- gravity compensation;
- bias/offset calibration;
- force saturation and clipping;
- torque scaling;
- contact threshold labels;
- temporal alignment with images, proprioception, and actions;
- low-pass filtering and sensor delay;
- normalization statistics per task versus global.

This lane is mandatory. Do not promote RGRP, Hyperloop, Cosmos, or DreamZero until corrected ACT and corrected ACT plus auxiliary heads are measured.

### Lane B: Shared Physics Latent Control

Purpose: test whether auxiliary physics prediction improves action quality.

```text
multimodal encoders -> shared latent -> ACT
                         + auxiliary physics heads
```

Controls:

- no auxiliary heads;
- next proprio only;
- next force/torque only;
- contact/jam/slip head;
- all auxiliary heads.

Metrics:

- sim task success;
- force/contact violations;
- jam recovery;
- trajectory smoothness;
- action latency;
- peak force/torque;
- success after perturbation;
- sim-to-real gap when available.

### Lane C: RGRP Physical Belief-State Bridge

Purpose: test the secret ingredient directly.

```text
multimodal encoders
-> RGRP/Parcae physical belief-state update
-> ACT action chunk decoder
-> auxiliary heads
```

Matched controls:

- MLP bridge;
- GRU/LSTM bridge;
- Mamba/SSM bridge if practical;
- small Transformer bridge;
- RGRP bridge.

The comparison must be matched on:

- inputs;
- train/eval data;
- action decoder;
- parameter budget;
- latency budget;
- auxiliary heads;
- sim randomization.

### Lane D: Hyperloop/RGRP Residual-Stream Bridge

Purpose: test whether parallel residual streams and loop-level hyper-connections are the missing scaling recipe for RGRP/Parcae in physical control.

```text
multimodal encoders
-> split shared latent into n residual streams
-> RGRP/Parcae loop update
-> hyper-connection pre/post/res stream mixing after each loop
-> merged physical belief state
-> ACT action chunk decoder
-> auxiliary heads
```

First-pass knobs:

- residual stream count: `2`, `4`;
- loop count: `1`, `2`;
- stream merge: average, learned projection;
- stream transition: static, data-dependent diagonal, mHC-style constrained;
- stream assignment: generic learned streams first; semantic stream labels only as diagnostics.

Matched controls:

- RGRP bridge with one residual stream;
- RGRP bridge with parallel streams but no hyper-connection mixing;
- Hyperloop-style streams with MLP/Transformer bridge but no RGRP;
- GRU/LSTM bridge with matched parameter count.

Go signal:

- better contact recovery or offline physics prediction at similar latency;
- lower force/contact violations than single-stream RGRP;
- improved stability under longer action horizons.

### Lane E: Recipe Isolation Ladder

Purpose: identify which recurrence/looping ingredients actually help before moving them into the robotics/world-model stack.

This ladder should be run on the cheapest trustworthy proxy first:

- language-model/perplexity proxy if testing abstract architecture efficiency;
- offline robot trace proxy if testing physical belief-state value;
- short sim rollout only after the proxy is stable.

Core ablations:

| ID | Candidate | Purpose |
|---|---|---|
| `R0` | baseline attention / corrected ACT | control |
| `R1` | RGRP single-stream bridge | tests recurrent state update alone |
| `R2` | RGRP + loop-position embedding | tests loop/time-step awareness |
| `R3` | RGRP + data-dependent diagonal or block-diagonal transition | tests transition structure |
| `R4` | RGRP + parallel residual streams, no hyper-connection mixing | tests extra scratchpad capacity |
| `R5` | RGRP + hyper-connection stream mixing | tests loop-level stream routing |
| `R6` | full Hyperloop/RGRP stream contract | tests combined recipe |
| `R7` | Hyperloop-style streams without RGRP | separates residual-stream benefit from RGRP |
| `R8` | GRU/LSTM/Mamba bridge with matched params | recurrence family controls |
| `R9` | small Transformer bridge with matched params | attention-style bridge control |

Do not jump from `R1` to `R6` and then claim victory. The key attribution questions are:

```text
Is the win from recurrence?
from looped depth?
from loop-position conditioning?
from residual streams?
from hyper-connection mixing?
from structured transitions?
from all of them together?
```

Metrics for language-model proxy:

- loss;
- perplexity;
- tokens/sec;
- peak memory;
- parameter count;
- longer-run curve stability;
- seed replication.

Metrics for robotics proxy:

- next force/torque prediction;
- next proprio prediction;
- contact/jam/slip classification;
- action residual prediction;
- success/failure phase prediction;
- latency and memory.

Promotion rule:

- promote an ingredient only if it improves a matched proxy metric without unacceptable latency/memory cost;
- promote a bundle only after at least one ingredient-level ablation explains why it should help.

### Lane F: Cosmos Reason2 Planner Conditioning

Purpose: add high-level semantic reasoning without making it responsible for low-level control.

```text
Cosmos Reason2 / Qwen3-instruct backbone
-> plan/phase/constraint tokens
-> RGRP physical belief state
-> ACT
```

Controls:

- ACT + force/torque only;
- ACT + text/phase conditioning only;
- Cosmos conditioning without RGRP;
- Cosmos conditioning with RGRP.

### Lane G: DreamZero/WAM-DiT Future Predictor

Purpose: test future-state imagination and action-conditioned dynamics.

```text
current observation + candidate action/phase
-> DreamZero-like future predictor
-> predicted future latent
-> RGRP compares/refines physical state
-> ACT
```

Controls:

- no future predictor;
- future predictor only;
- RGRP only;
- future predictor + RGRP.

### Lane H: DeepSeek-Inspired Stability and Prediction

Purpose: add attributable innovations, not a giant bundle.

First-pass additions:

- mHC-style residual mixing around the physical belief-state latent;
- mHC-style constraints for Hyperloop residual-stream transitions;
- MTP-style short-horizon auxiliary prediction for action/state/force;
- Muon optimizer for the trainable bridge/action modules.

Deferred:

- compressed sparse attention;
- MoE;
- FP4/quant/cache infra;
- million-token-style context tricks.

## Competition-Specific 2-3B Variant

The competition may be resource constrained. A full 8B planner may be too heavy. The 2-3B lane should be a specialist, not a full replacement for the stack.

Proposed shape:

```text
2-3B Qwen/Cosmos-derived specialist
-> semantic plan / phase / constraints
-> RGRP physical belief-state bridge
-> ACT action decoder
```

The 2-3B model should:

- interpret task language;
- encode scene/task phase;
- produce plan hints and constraints;
- condition the RGRP/ACT stack;
- not directly output dense low-level motor commands.

Training path:

1. Use Cosmos Reason2 8B or stronger teacher traces to generate high-level plan/phase labels.
2. Distill into a 2-3B specialist.
3. Freeze or lightly tune the 2-3B model while training RGRP/ACT.
4. Evaluate resource-constrained sim performance.

Possible variants:

```text
2-3B only -> ACT
2-3B -> ACT + auxiliary heads
2-3B -> RGRP -> ACT
2-3B -> Hyperloop/RGRP -> ACT
2-3B -> DreamZero future predictor -> RGRP -> ACT
```

## Staged Execution Plan

### Stage 0: Correct Inputs and Instrumentation

- Confirm legal observation channels.
- Add wrist force/torque to the ACT input projection.
- Normalize force/torque.
- Add logging for force peaks, contact violations, slip/jam events, and action smoothness.
- Preserve old no-force baseline as diagnostic-only.
- Verify strict causal slicing:
  - allowed: actions up to `t-1`, sensors up to `t`, instruction, current proprio/F/T/tactile;
  - disallowed: target action chunk `t:t+k`, future proprio, future contact, future object state.
- Verify auxiliary heads use deploy-time sensor latents as inputs and privileged sim quantities only as targets.

### Stage 1: Offline Trace Proxies

Before expensive sim rollouts, train on logged trajectories.

Tasks:

- next action chunk;
- next proprio;
- next force/torque;
- contact/jam/slip;
- action residual;
- phase/success label.

Goal: determine whether RGRP improves physical-state prediction before it controls a robot.

### Stage 2: Recipe Isolation

Before mixing Cosmos, DreamZero, Hyperloop, and RGRP into a large stack, isolate recurrence/loop ingredients:

```text
baseline
RGRP
RGRP + loop position
RGRP + structured transition
RGRP + residual streams
RGRP + hyper-connection mixing
full Hyperloop/RGRP
Hyperloop without RGRP
GRU/LSTM/Mamba controls
```

Run this first on offline trace proxies and, if useful, a short sim rollout.

The ladder must isolate axes one at a time:

| Ingredient | Required Control | What It Tests |
|---|---|---|
| Recurrence | MLP vs GRU/LSTM vs Mamba/SSM vs RGRP | whether RGRP beats standard memory |
| Looped depth | tied loop `K=1/2/4` vs untied extra layers | whether looping helps beyond more depth |
| Loop position | no loop embedding vs learned loop embedding vs shuffled loop index | whether loop identity matters |
| Rotary gated update | RGRP vs non-rotary gated recurrence vs dense transition | whether rotary gating matters |
| Structured transition | diagonal/block-diagonal vs dense vs fixed transition | whether state transition structure matters |
| Residual streams | single stream vs matched-parameter multi-stream | whether scratchpad streams help |
| Hyper-connection mixing | unconstrained HC vs mHC-style constrained mixing | whether constrained mixing stabilizes streams |
| Optimizer | AdamW vs Muon, architecture fixed | whether optimization is being mistaken for architecture |
| Auxiliary heads | bridge with/without short-horizon heads | whether auxiliary supervision drives gains |

### Stage 3: Minimal Sim A/B

Run short sim evals:

```text
ACT + force/torque
ACT + force/torque history
ACT + auxiliary physics heads
ACT + RGRP bridge
ACT + Hyperloop/RGRP bridge
ACT + GRU/LSTM bridge
ACT + Mamba bridge if available
```

Do not add Cosmos/DreamZero yet unless the corrected baseline still fails badly.

### Stage 4: RGRP + Planner/Future Hybrid

If RGRP or Hyperloop/RGRP wins Stage 2/3:

```text
Cosmos conditioning -> RGRP -> ACT
DreamZero future predictor -> RGRP -> ACT
Hyperloop/RGRP residual streams -> ACT
Cosmos + DreamZero -> RGRP -> ACT
```

Run one factor at a time.

### Stage 5: Competition Specialist

Build the 2-3B specialist if:

- Cosmos 8B is too slow/large for eval;
- RGRP/ACT bridge is stable;
- high-level plan/phase conditioning is useful;
- teacher traces or distillation data exist.

### Stage 6: Full SOTA Candidate

Only after attribution is clear:

```text
2-3B specialist or Cosmos Reason2
-> DreamZero/WAM-DiT future predictor
-> Hyperloop/RGRP physical belief-state bridge with mHC-style mixing
-> ACT action chunk decoder
-> auxiliary physics heads
```

## Evaluation Metrics

Primary:

- task success rate;
- eval score;
- force/contact violation rate;
- insertion/jam recovery;
- completion time;
- latency per control step.

Secondary:

- action smoothness;
- peak force/torque;
- cumulative force exposure;
- retreat/retry behavior;
- trajectory consistency;
- perturbation recovery;
- sim-to-real degradation.

Model/system:

- parameter count;
- memory;
- tokens/steps per second;
- inference latency;
- training stability;
- ablation attribution.

## Failure Modes to Watch

- RGRP adds latency but no sim success gain.
- Hyperloop streams add representational capacity but no control benefit.
- Hyperloop streams fragment the physical state instead of improving it.
- Auxiliary heads improve prediction but not control.
- Cosmos/DreamZero introduces high-level hallucination or delayed correction.
- 2-3B specialist loses important teacher reasoning.
- Force/torque overfits simulator contact dynamics and hurts sim-to-real.
- Multiple modules create attribution fog.
- Reward/eval improvements come from safety clamps rather than policy competence.

## Go / No-Go Gates

### Gate 0: Sensor Repair

Proceed only if:

- corrected ACT with force/torque is reproducible;
- normalization and timing are audited;
- contact labels align with force events;
- old missing-F/T ACT is treated as diagnostic-only.

Pause broader architecture work if force/torque preprocessing, temporal alignment, or corrected ACT baselining is unresolved.

### RGRP Bridge Go

Proceed only if RGRP beats corrected ACT plus auxiliary heads and the best matched MLP/GRU/LSTM/Mamba/small-Transformer bridge on at least two of:

- lower force/contact failures;
- better recovery after jam;
- higher task success;
- smoother actions;
- better offline physics prediction;
- acceptable latency.

Suggested first threshold:

- at least a `5-10` percentage-point closed-loop success gain; or
- at least `15%` reduction in contact/slip/jam failures;
- no worse latency or action smoothness beyond the predefined budget.

Do not promote RGRP if it only improves offline auxiliary prediction loss.

### Causal State Go

RGRP state interventions must matter:

- zero state mid-episode;
- swap states between two episodes;
- delay state by `N` steps;
- reset state at phase boundary;
- freeze state after contact.

If closed-loop behavior barely changes, the recurrent state is not doing useful causal work.

### Robustness Go

RGRP must improve under held-out perturbations:

- object poses;
- mass/friction/contact parameters;
- force/torque sensor noise and bias;
- visual occlusion;
- perturbed initial states.

If gains vanish outside the narrow training distribution, redesign or kill the bridge for the competition stack.

### Hyperloop/RGRP Bridge Go

Proceed if parallel residual streams improve over single-stream RGRP on at least one primary control metric and one diagnostic metric:

- lower force/contact violations;
- better jam/slip recovery;
- higher success under perturbation;
- better next force/torque or next proprio prediction;
- stable or improved latency/memory relative to the value gained.

### Cosmos/DreamZero Hybrid Go

Proceed if the base RGRP/ACT stack is stable and either:

- semantic task errors dominate;
- future-state prediction errors dominate;
- high-level phase conditioning improves eval score.

Before DreamZero/WAM-DiT:

- oracle future latents must improve action performance.

Before Cosmos or a 2-3B specialist planner:

- ground-truth phase/constraint hints must improve action performance.

If oracle futures or ground-truth hints do not help, skip the model that predicts them.

### 2-3B Specialist Go

Proceed if:

- 8B model is too expensive for competition eval;
- planner conditioning helps;
- distillation data exists;
- the specialist can preserve most of the planner benefit at much lower latency.

## Recommended Immediate Next Step

Run the corrected ACT + force/torque baseline, but in parallel design the RGRP bridge contract:

```text
obs_t = [vision_latent, proprio_t, wrist_force_torque_t, previous_action_t]
physics_state_t = RGRP(physics_state_{t-1}, obs_t)
action_chunk_t = ACT(obs_t, physics_state_t)
auxiliary_predictions_t = heads(physics_state_t)
```

Then test the Hyperloop/RGRP extension only after the single-stream RGRP bridge is alive:

```text
streams_t = hyperloop_rgrp_update(streams_{t-1}, obs_t)
physics_state_t = merge(streams_t)
action_chunk_t = ACT(obs_t, physics_state_t)
auxiliary_predictions_t = heads(physics_state_t, streams_t)
```

The key research claim to test:

> RGRP provides a compact recurrent physical belief state that improves contact-rich robot control beyond attention-only or feed-forward action chunking, especially when legal force/torque signals are available.

The Hyperloop extension claim is:

> Parallel residual streams with loop-level hyper-connections give RGRP a richer latent scratchpad, allowing recurrent physical-state refinement to scale without collapsing into repeated similar updates.

## 2026-04-30 LM Scout Implementation Surface

The first bounded Hyperloop-inspired LM scout should isolate ingredients rather
than jump straight to the full paper recipe. The implemented toggles are:

- `parcae_loop_position_kind=learned`: adds a loop-local learned position signal
  at the wide-to-loop projection seam before RGRP control and recurrent-band
  attention.
- `parcae_stream_count > 1`: runs multiple residual streams through the same
  existing rank-3 Parcae loop-region boundary, once per stream, so the current
  Triton loop kernel contract is not silently changed to a rank-4 tensor.
- `parcae_stream_merge_mode=average|static|dynamic-diagonal`: merges stream
  states before the loop-to-wide output projection.
- `attention_position_profile=rope`: applies RoPE to attention q/k heads,
  separate from learned additive position embeddings.
- `transformer_ffn_kind=swiglu`: swaps standard GELU MLP blocks for a
  near-parameter-matched SwiGLU FFN.

Kernel-contract hedge:

- The current champion path remains unchanged when all new knobs are default.
- Multi-stream lanes intentionally reuse the existing loop-region function per
  stream instead of pretending the Triton kernel accepts a new stream dimension.
- RoPE/SwiGLU change attention-block math and should be reported as a separate
  attention-layer recipe permutation, not as a pure RGRP kernel optimization.

Recommended 1024-step scout order:

1. Champion control: current `5,3,4,3,4,3,2`, loop `384x2`, learned positions,
   RGRP control position, GELU/additive attention profile.
2. Loop-position only: add `parcae_loop_position_kind=learned`.
3. Static two-stream: add `parcae_stream_count=2`,
   `parcae_stream_merge_mode=static`.
4. Dynamic two-stream: add `parcae_stream_count=2`,
   `parcae_stream_merge_mode=dynamic-diagonal`.
5. RoPE/SwiGLU attention recipe: use `attention_position_profile=rope`,
   `transformer_ffn_kind=swiglu`, and disable additive learned token positions
   unless the explicit experiment asks to stack both.
