# Depth/Recurrence Architecture-Family Probe Summary

Status: first pass over previously untested architecture families completed
locally.

## Scope

This pass intentionally did not promote B2. It opened the next practical
families from the matrix inside the existing Python Path 1 harness:

- D1: causal token-level MoD-style block routing.
- D3: token-selective recurrent latent refinement.
- D4: lightweight function/control-block proxy via the existing
  Parcae P20-control loop.

D2 exact Flash MoDA and D5 NSA-style sparse attention remain deferred because
they are kernel/long-context families and would dominate this short local pass.

## Repo Integration Summary

- Added typed `TokenRoutingProfile` and `RecurrentTokenRoutingProfile` controls
  to `Path1VariantSpec`.
- Added CLI flags for token-routed blocks and recurrent token routing.
- Added `TokenRoutedLocalCausalTransformerBlock`, which routes selected tokens
  through attention/FFN while skipped tokens carry their hidden state unchanged.
- Added a causal prefix-top-k routing primitive. This avoids the causal bug
  where full-sequence top-k lets future tokens change whether earlier tokens
  execute a block.
- Extended the Parcae loop with token-selective recurrence using the same causal
  prefix-top-k routing principle.
- Added diagnostics for selected-token fractions, selected gates, routed layers,
  and recurrent token routing.
- Reused the existing `parcae-p20-control-looped-attention` scaffold as the D4
  function/control-block proxy; no new symbolic runtime was introduced.

## Exact vs Approximate

- D1 is a practical MoD approximation. It uses causal prefix-top-k routing, not
  full-sequence top-k capacity routing. That keeps autoregressive behavior
  correct but means actual selected-token fractions may differ from the request.
- D1 currently uses Python gather/scatter loops and dense K/V projection, so the
  implementation tests architecture behavior more than hardware efficiency.
- D3 is a practical token-selective recurrent approximation. Selected recurrent
  tokens run block updates; unselected tokens retain state for that recurrent
  step.
- D4 is only a lightweight function/control proxy via an existing P20 controller,
  not an explicit function-call interpreter or scratchpad computer.

## Benchmark Surface

- Backend: local `mps`, fp32
- Corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- Seed: 42
- Shape: default Path 1 shape, `d_model=128`, `heads=4`, `layers=8`
- Budget: `seq_len=32`, `batch_size=1`, `steps=8`, `eval_batches=2`
- Warmup: 1 eval batch, 1 train step
- Reports: `artifacts/depth-recurrence-architecture-families/*/*/report.json`
- Ledger: `artifacts/depth-recurrence-architecture-families/ledger.jsonl`

## Benchmark Results

| Rank | Variant | Final Loss | Train tok/s | Overall tok/s | Peak RSS MiB | Avg Steps | Exit | Token Actual | Recur Actual |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | D1+D3 route25 fixed | 4.0122 | 159.5 | 174.1 | 687.7 | 3.0 | 0/14 | 0.328 | 0.295 |
| 2 | D1+D3 route25 accel `t0.6` | 4.0550 | 183.4 | 207.9 | 683.8 | 2.0 | 14/14 | 0.359 | 0.286 |
| 3 | D1+D3 route50 fixed | 4.0887 | 173.1 | 197.2 | 585.0 | 3.0 | 0/14 | 0.555 | 0.519 |
| 4 | D3 route50 fixed | 4.1229 | 183.5 | 194.1 | 733.5 | 3.0 | 0/14 | n/a | 0.585 |
| 5 | D3 route25 fixed | 4.1384 | 246.8 | 273.3 | 663.0 | 3.0 | 0/14 | n/a | 0.359 |
| 6 | D3 route25 accel `t0.6` | 4.1389 | 323.0 | 376.9 | 640.3 | 2.0 | 14/14 | n/a | 0.346 |
| 7 | D1 mid route50 | 4.5321 | 221.3 | 259.9 | 690.4 | n/a | n/a | 0.523 | n/a |
| 8 | D1 mid route25 | 4.5946 | 153.9 | 164.3 | 685.5 | n/a | n/a | 0.367 | n/a |
| 9 | D1 all route50 | 4.6781 | 134.7 | 124.4 | 665.3 | n/a | n/a | 0.500 | n/a |
| 10 | A1 control | 4.8464 | 1793.3 | 2515.5 | 365.3 | n/a | n/a | n/a | n/a |
| 11 | D4 P20-control loop proxy | 4.8789 | 713.7 | 943.4 | 414.3 | 3.0 | 0/14 | n/a | n/a |

For context, the prior local B2-depthmem4 leader from the next-tranche pass had
seed-42 final loss 4.3278 at 791.6 train tok/s. D1/D3 variants beat that loss on
this tiny budget, but they are much slower in the current implementation.

## Interpretation

- D3 is the most interesting new family. Token-selective recurrence produced a
  large loss improvement over A1 and over the prior B2-depthmem4 seed-42 result.
- D1 alone is mixed. Routing all layers is not useful. Routing only middle
  layers improves loss over A1, but the Python-loop implementation is slow and
  memory-heavy.
- D1+D3 has the best local loss. The route25 fixed variant reached 4.0122, and
  adding acceleration recovered some throughput with only moderate quality loss.
- Actual causal prefix-top-k fractions were higher than the requested fractions,
  especially at route25. This is expected from the causal capacity rule and must
  be treated as part of the contract.
- D4 P20-control loop is not promising in this surface. It was slower than A1
  and worse on loss. Keep it as a control/proxy, not as a lead branch.
- The current D1/D3 implementation is not hardware efficient yet. It has the
  right typed control plane and causal semantics, but still uses Python loops and
  dense projections. The architecture signal and runtime signal point in opposite
  directions.

## Commands Run

Verification:

```bash
python3 -m compileall python
uv run --with torch --with numpy --with pytest python -m unittest python.tests.test_specs.Path1SpecTests python.tests.test_models.Path1ModelTests
uv run --with black black python/models/transformer.py python/models/path1.py python/specs/path1.py python/specs/__init__.py python/runners/path1_cli.py python/tests/test_models.py python/tests/test_specs.py
```

Family sweep common arguments:

```bash
uv run --with torch --with numpy python scripts/v3a_python_path1.py \
  --backend mps --dtype fp32 \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-9row-v1 \
  --benchmark-name depth-recurrence-architecture-families-v1 \
  --seq-len 32 --window-stride 32 --batch-size 1 \
  --steps 8 --eval-batches 2 \
  --warmup-eval-batches 1 --warmup-train-steps 1 \
  --seed 42 \
  --output-dir artifacts/depth-recurrence-architecture-families/seed42 \
  --ledger-path artifacts/depth-recurrence-architecture-families/ledger.jsonl \
  --output table
```

Variant suffixes included:

```bash
--variant attention-only --run-label depthrec-families-a1-control-s42
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.5 --run-label depthrec-families-d1-token-mod-all-route50-s42
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.5 --token-routing-layer-indices 2,3,4,5 --run-label depthrec-families-d1-token-mod-mid-route50-s42
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.5 --run-label depthrec-families-d3-token-selective-recur-route50-s42
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.5 --token-routing-layer-indices 0,1,6,7 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.5 --run-label depthrec-families-d1d3-token-mod-recur-route50-s42
--variant attention-only --scaffold-profile parcae-p20-control-looped-attention --parcae-loop-count 3 --run-label depthrec-families-d4-p20-control-loop-s42
```

Stabilization suffixes:

```bash
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.25 --token-routing-layer-indices 2,3,4,5 --run-label depthrec-families-d1-token-mod-mid-route25-s42
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.25 --run-label depthrec-families-d3-token-selective-recur-route25-s42
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.25 --token-routing-layer-indices 0,1,6,7 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.25 --run-label depthrec-families-d1d3-token-mod-recur-route25-s42
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.25 --run-label depthrec-families-d3-token-selective-recur-route25-accel-t0p6-s42
--variant attention-only --token-routing-profile causal-topk-block --token-route-fraction 0.25 --token-routing-layer-indices 0,1,6,7 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --recurrent-token-routing-profile causal-topk-state --recurrent-token-route-fraction 0.25 --run-label depthrec-families-d1d3-token-mod-recur-route25-accel-t0p6-s42
```

## Next Recommendation

1. Do not promote D1/D3 yet. First optimize the selected-token execution path or
   test it in a vectorized form so runtime is not dominated by Python loops.
2. Run a 3-seed sanity sweep for only three rows: B2-depthmem4, D3 route25 accel,
   and D1+D3 route25 accel. This is not promotion; it is to check whether the
   new family signal survives seeds.
3. Add a typed causal-routing calibration option if route25 continues selecting
   closer to 30-36% of tokens. The requested budget and actual selected budget
   should be explicit in every report.
4. Defer D4 unless a stronger function-block design emerges from the existing
   primitive stack.
5. Consider D5 only with a longer-context benchmark profile. The current 32-token
   local sweep is the wrong surface for sparse attention.
