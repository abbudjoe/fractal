# Depth/Recurrence Initial Tranche Summary

Status: implemented and benchmarked locally. A follow-on ablation/promotion pass
is recorded in `docs/depth_recurrence_next_tranche_summary.md`.

## Repo Integration Summary

- Reused the Python Path 1 harness: `Path1VariantSpec`, `LocalCausalTransformerBlock`,
  `phase1_attention_only_variant`, `scripts/v3a_python_path1.py`, report JSONs,
  and ledger writing.
- Added typed variant controls:
  - `AttentionProfile`: `standard`, `moda-depth-kv`
  - `RecurrentHaltingProfile`: `fixed`, `acceleration`, `normalized-step-norm`
  - explicit depth-memory, recurrent-min-step, and halting-threshold fields.
- Added `DepthAugmentedCausalSelfAttention`, a MoDA-style approximation that
  appends projected prior-depth hidden-state KV to current causal sequence KV.
- Extended the existing Parcae loop scaffold with hidden-state halting diagnostics:
  average steps, last step norms, halting metric, early-exit count, and forward
  count.
- Reused existing tiny structured FFN paths for C1; no new symbolic runtime was
  introduced.
- Fixed one stale Rust test fixture by adding `data_seed: None` to a `CliArgs`
  literal so the bin-local test matches the current typed CLI shape.

## Exact vs Approximate

- A1 is the existing attention-only baseline.
- A2/A3 are practical recurrent-depth approximations using the repo's existing
  Parcae-inspired middle-loop scaffold.
- B1/B2/C1 are not exact Flash MoDA implementations. They approximate the paper
  concept by projecting prior layer hidden states into depth KV and preserving
  causal visibility over the sequence dimension.
- C1 is a compact structured-compute probe using the existing `tiny-glu-gated`
  FFN on recurrent layer 4, not a full scratchpad or symbolic computer.

## Benchmark Surface

- Backend: local `mps`, fp32
- Corpus: `experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1`
- Seed: 42
- Shape: default Path 1 shape, `d_model=128`, `heads=4`, `layers=8`
- Budget: `seq_len=32`, `batch_size=1`, `steps=8`, `eval_batches=2`
- Warmup: 1 eval batch, 1 train step
- Reports: `artifacts/depth-recurrence-initial-tranche/*/report.json`
- Ledger: `artifacts/depth-recurrence-initial-tranche/ledger.jsonl`

## Benchmark Results

| Variant | Label | Params | Initial Loss | Final Loss | Train tok/s | Overall tok/s | Peak RSS MiB | Avg Recurrent Steps | Exit Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | `attention-only` | 1,651,968 | 5.9607 | 4.8464 | 1143.5 | 1611.6 | 365.6 | n/a | n/a |
| A2 | `parcae-looped-attention-loops3` | 1,652,608 | 5.6344 | 4.5365 | 802.0 | 1091.3 | 374.9 | 3.0 | 0/14 |
| A3 | `parcae-looped-attention-loops3-halt-acceleration-min2-t0p6` | 1,652,608 | 5.6061 | 4.6411 | 886.6 | 1165.2 | 375.0 | 2.0 | 14/14 |
| B1 | `moda-depth-kv-depthmem2` | 1,918,216 | 5.8349 | 4.6423 | 956.3 | 1205.9 | 406.4 | n/a | n/a |
| B2 | `moda-depth-kv-depthmem2-parcae-looped-attention-loops3-halt-acceleration-min2-t0p6` | 1,918,856 | 5.5331 | 4.4415 | 568.2 | 692.2 | 411.6 | 2.0 | 14/14 |
| C1 | `B2 + tiny-glu-gated layer4` | 2,017,928 | 5.5566 | 4.5319 | 560.0 | 676.7 | 415.0 | 2.0 | 14/14 |

## Interpretation

- Fixed recurrent depth (A2) was the strongest simple change versus baseline:
  much better final loss than A1 at about 70% of A1 train throughput.
- Acceleration halting (A3) recovered some throughput by exiting at 2 steps, but
  gave back quality versus fixed 3-step recurrence. It is useful, but the
  threshold needs a more formal policy.
- Depth-KV attention alone (B1) improved loss over A1 and was less expensive
  than recurrent hybrids, but did not beat A2 on this small budget.
- B2 was the best quality result in the tranche. The depth+recurrence interaction
  looks promising, but it is slower and should be promoted carefully before
  multiplying variants.
- C1 did not beat B2. The tiny GLU side path added parameters and cost without
  enough quality gain in this first run.

## Commands Run

Verification:

```bash
python3 -m compileall python
uv run --with torch --with numpy --with pytest python -m unittest python.tests.test_specs.Path1SpecTests python.tests.test_models.Path1ModelTests
uv run --with black black python/models/transformer.py python/models/path1.py python/specs/path1.py python/runners/path1_cli.py python/tests/test_models.py python/tests/test_specs.py python/specs/__init__.py
uv run --with black black --check python/models/transformer.py python/models/path1.py python/specs/path1.py python/runners/path1_cli.py python/tests/test_models.py python/tests/test_specs.py python/specs/__init__.py
uv run --with torch --with numpy --with pytest python -m unittest discover python/tests
cargo fmt --all
cargo fmt --all --check
cargo test -p fractal --bin v3a-hybrid-attention-matrix
```

Full `cargo test --workspace` was also attempted. After the stale `data_seed`
fixture was fixed, the run still fails in pre-existing Mamba3 parity tests
because the Rust harness calls `scripts/mamba3_pytorch_reference.py` with the
legacy input/output-file contract while the Python script now expects
`<command> <bundle.json>`. That is outside this tranche. One transient v2
tolerance failure disappeared on rerun.

Smoke command:

```bash
uv run --with torch --with numpy python scripts/v3a_python_path1.py --backend cpu --dtype fp32 --variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 1000000000 --feed-forward-profile tiny-glu-gated --feed-forward-layer-indices 2 --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl --seq-len 16 --window-stride 16 --batch-size 1 --steps 1 --eval-batches 1 --warmup-eval-batches 0 --warmup-train-steps 0 --d-model 32 --head-count 4 --total-layers 4 --ffn-multiplier 2 --run-label smoke-depth-recurrent-c1 --output-dir artifacts/depth-recurrence-smoke --output json
```

Benchmark common arguments:

```bash
uv run --with torch --with numpy python scripts/v3a_python_path1.py \
  --backend mps --dtype fp32 \
  --jsonl-train-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/train.jsonl \
  --jsonl-eval-path experiments/stage0/assets/fineweb/stage0-local-bench-9row-v1/eval.jsonl \
  --corpus-name fineweb-stage0-local-bench-9row-v1 \
  --benchmark-name depth-recurrence-local-sweep-v1 \
  --seq-len 32 --window-stride 32 --batch-size 1 \
  --steps 8 --eval-batches 2 \
  --warmup-eval-batches 1 --warmup-train-steps 1 \
  --seed 42 \
  --output-dir artifacts/depth-recurrence-initial-tranche \
  --ledger-path artifacts/depth-recurrence-initial-tranche/ledger.jsonl \
  --output table
```

Variant-specific suffixes:

```bash
# A1
--variant attention-only --run-label depthrec-a1-mps-s42

# A2
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --run-label depthrec-a2-mps-s42

# A3
--variant attention-only --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-a3-mps-s42-stabilized

# B1
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --run-label depthrec-b1-mps-s42

# B2
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --run-label depthrec-b2-mps-s42-stabilized

# C1
--variant attention-only --attention-profile moda-depth-kv --depth-memory-layers 2 --scaffold-profile parcae-looped-attention --parcae-loop-count 3 --recurrent-halting-profile acceleration --recurrent-min-steps 2 --recurrent-halting-threshold 0.6 --feed-forward-profile tiny-glu-gated --feed-forward-layer-indices 4 --run-label depthrec-c1-mps-s42-stabilized
```

## Next-Tranche Recommendation

1. B2-fixed-depth-recurrence: depth-KV plus fixed 3-step recurrence, no halting.
   This isolates whether B2's gain comes from depth+recurrence or the 2-step
   adaptive path.
2. B2 halting policy ladder: compare acceleration thresholds `0.4`, `0.6`,
   `0.8` plus `normalized-step-norm` at one calibrated threshold.
3. B2 depth-memory width ablation: `depthmem1`, `depthmem2`, `depthmem4` with
   the same recurrent/halting policy.
4. B2 promotion run: A1/A2/B1/B2 across at least 3 seeds or the full
   `cuda-faithful-small-v1` pass before investing in kernels.
5. Defer more structured experts unless B2 remains strong after promotion; C1
   was not enough evidence to expand the scratchpad branch yet.
