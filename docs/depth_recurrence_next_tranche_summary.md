# Depth/Recurrence Next-Tranche Summary

Status: ablation and seed-promotion pass completed locally. A subsequent
architecture-family probe is recorded in
`docs/depth_recurrence_architecture_families_summary.md`.

## Run Contract

- Backend: local `mps`, fp32
- Corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- Shape: default Path 1 shape, `d_model=128`, `heads=4`, `layers=8`
- Budget: `seq_len=32`, `batch_size=1`, `steps=8`, `eval_batches=2`
- Warmup: 1 eval batch, 1 train step
- Search rule: sequential runs only; no concurrent benchmark jobs
- Ablation reports: `artifacts/depth-recurrence-next-tranche/ablations/*/report.json`
- Promotion reports: `artifacts/depth-recurrence-next-tranche/promotion-s43/*/report.json`
  and `artifacts/depth-recurrence-next-tranche/promotion-s44/*/report.json`
- Ledger: `artifacts/depth-recurrence-next-tranche/ledger.jsonl`

## Ablation Results

Seed 42 was used for the ablation pass. The previously completed B2
`depthmem2/t0.6` run from the initial tranche is included as the center point.

| Variant | Final Loss | Train tok/s | Overall tok/s | Peak RSS MiB | Avg Steps | Exit | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| B2 `depthmem2`, acceleration `t0.6` | 4.4415 | 568.2 | 692.2 | 411.6 | 2.00 | 14/14 | Initial-tranche center point. |
| B2 fixed `depthmem2`, no halting | 4.5509 | 620.0 | 707.5 | 411.8 | 3.00 | 0/14 | Worse than the 2-step adaptive path on this seed. |
| B2 `depthmem2`, acceleration `t0.4` | 4.5509 | 621.7 | 696.1 | 410.8 | 3.00 | 0/14 | Behaves like fixed recurrence; threshold too strict. |
| B2 `depthmem2`, acceleration `t0.8` | 4.4415 | 700.5 | 805.0 | 410.6 | 2.00 | 14/14 | Same exit behavior/loss as `t0.6` here, faster in this run. |
| B2 `depthmem2`, normalized-step `t0.5` | 4.4999 | 253.8 | 305.7 | 338.8 | 2.36 | 9/14 | Mixed exits and poor observed throughput; not preferred yet. |
| B2 `depthmem1`, acceleration `t0.6` | 4.4826 | 598.4 | 713.1 | 401.7 | 2.00 | 14/14 | Slightly cheaper memory, worse quality than `depthmem2`. |
| B2 `depthmem4`, acceleration `t0.6` | 4.3278 | 791.6 | 947.4 | 428.4 | 2.00 | 14/14 | Best seed-42 result; promoted immediately. |

## Promotion Results

Promotion used seeds 42, 43, and 44. Seed 42 comes from the initial tranche for
A1/A2/B1/B2-depthmem2, and from the ablation pass for B2-depthmem4.

| Variant | Seeds | Mean Loss | Std Loss | Mean Train tok/s | Mean Overall tok/s | Mean Peak RSS MiB | Mean Steps | Exit Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 attention-only | 42,43,44 | 4.7058 | 0.1032 | 1517.5 | 2112.9 | 365.6 | n/a | n/a |
| A2 fixed recurrence | 42,43,44 | 4.7233 | 0.1907 | 1003.7 | 1315.5 | 374.6 | 3.00 | 0.00% |
| B1 depthmem2 | 42,43,44 | 4.5826 | 0.0933 | 1085.4 | 1316.8 | 406.4 | n/a | n/a |
| B2 depthmem2, acceleration `t0.6` | 42,43,44 | 4.3329 | 0.0771 | 671.3 | 782.5 | 410.4 | 2.00 | 100.00% |
| B2 depthmem4, acceleration `t0.6` | 42,43,44 | 4.2751 | 0.0384 | 768.2 | 893.1 | 427.6 | 2.00 | 100.00% |

Per-seed final losses:

| Variant | s42 | s43 | s44 |
| --- | ---: | ---: | ---: |
| A1 attention-only | 4.8464 | 4.6696 | 4.6014 |
| A2 fixed recurrence | 4.5365 | 4.6483 | 4.9851 |
| B1 depthmem2 | 4.6423 | 4.4509 | 4.6546 |
| B2 depthmem2, acceleration `t0.6` | 4.4415 | 4.2706 | 4.2866 |
| B2 depthmem4, acceleration `t0.6` | 4.3278 | 4.2372 | 4.2602 |

## Interpretation

- The main hybrid signal strengthened. B2 with adaptive 2-step recurrence beats
  A1, A2, and B1 across the three-seed promotion surface.
- `depthmem4` is the new local leader. It improves mean loss from 4.3329 to
  4.2751 versus `depthmem2`, with higher memory use but also better observed
  throughput in this short MPS run.
- Fixed B2 and acceleration `t0.4` are not worth emphasizing next. They both use
  3 recurrent steps and landed at the same worse seed-42 loss.
- Acceleration `t0.6` and `t0.8` collapse to the same behavior on this budget:
  both exit at step 2 for every measured forward. Treat them as one policy until
  longer runs expose a difference.
- Normalized-step halting is not competitive in this pass. It had mixed exits,
  worse loss than acceleration, and a suspiciously low throughput observation.
  Keep it as a diagnostic policy, not as the lead path.
- A2 fixed recurrence was unstable across seeds here. Its seed-44 result was
  worse than baseline, so recurrence alone is less convincing than the
  depth+recurrence combination.

## Commands Run

Ablation common arguments:

```bash
uv run --with torch --with numpy python scripts/v3a_python_path1.py \
  --backend mps --dtype fp32 \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-9row-v1 \
  --benchmark-name depth-recurrence-local-next-tranche-v1 \
  --seq-len 32 --window-stride 32 --batch-size 1 \
  --steps 8 --eval-batches 2 \
  --warmup-eval-batches 1 --warmup-train-steps 1 \
  --seed 42 \
  --output-dir artifacts/depth-recurrence-next-tranche/ablations \
  --ledger-path artifacts/depth-recurrence-next-tranche/ledger.jsonl \
  --output table
```

Ablation variant suffixes:

```bash
# B2 fixed depth recurrence
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --run-label depthrec-next-b2-fixed-depthmem2-s42

# B2 acceleration threshold ladder
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.4 --run-label depthrec-next-b2-accel-t0p4-depthmem2-s42
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.8 --run-label depthrec-next-b2-accel-t0p8-depthmem2-s42

# B2 normalized-step comparison
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile normalized-step-norm --recurrent-min-steps 2 --recurrent-halting-threshold 0.5 --run-label depthrec-next-b2-normstep-t0p5-depthmem2-s42

# B2 depth-memory width ablation
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 1 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-next-b2-accel-t0p6-depthmem1-s42
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 4 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-next-b2-accel-t0p6-depthmem4-s42
```

Promotion common arguments used for seeds 43 and 44:

```bash
uv run --with torch --with numpy python scripts/v3a_python_path1.py \
  --backend mps --dtype fp32 \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-9row-v1 \
  --benchmark-name depth-recurrence-local-next-tranche-promotion-v1 \
  --seq-len 32 --window-stride 32 --batch-size 1 \
  --steps 8 --eval-batches 2 \
  --warmup-eval-batches 1 --warmup-train-steps 1 \
  --seed <43-or-44> \
  --output-dir artifacts/depth-recurrence-next-tranche/promotion-s<seed> \
  --ledger-path artifacts/depth-recurrence-next-tranche/ledger.jsonl \
  --output table
```

Promotion variant suffixes:

```bash
--variant attention-only --run-label depthrec-promote-a1-s<seed>
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --run-label depthrec-promote-a2-s<seed>
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --run-label depthrec-promote-b1-depthmem2-s<seed>
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-promote-b2-depthmem2-accel-t0p6-s<seed>
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 4 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-promote-b2-depthmem4-accel-t0p6-s<seed>
```

## Next Recommendation

1. Promote `B2-depthmem4-accel-t0.6` to the next larger harness profile before
   touching kernel work.
2. Compare `depthmem4` against `depthmem6` only if memory headroom remains; avoid
   a full width sweep until the larger run confirms the signal.
3. Add one longer-budget B2-depthmem4 run with fixed train/eval protocol to see
   whether the early seed advantage survives beyond 8 steps.
4. Do not expand C1/structured experts yet. The hybrid attention/recurrence
   branch is now clearer than the scratchpad branch.
5. Keep exact Flash MoDA deferred until `depthmem4` wins on a larger profile,
   because the current PyTorch approximation is already enough to identify the
   architecture signal.
