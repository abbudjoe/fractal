# Canonical Benchmarks

These manifests define the frozen benchmark suites for the current winner lane.

## Discipline

- These files are the canonical run requests for the benchmark suite.
- They are checked in as manifest v2.
- They intentionally pin the stable `main` branch, not a transient feature
  branch.
- The frozen commit for a suite is the commit that launches the suite, and that
  commit SHA is preserved in the emitted artifacts/manifests.

Why not store `expected_commit_sha` here?

- A checked-in manifest cannot self-pin the commit that contains it without
  creating a self-reference loop.
- So the runner supports commit pinning, but canonical checked-in manifests use:
  - `expected_branch: main`
  - authoritative comparison contract
  - preserved launch commit in run artifacts

## Suites

- `leaderboard/`
  Frozen scientific leaderboard suite for the top three winner-lane primitives.
- `systems-speed/`
  Separate systems-speed suite for the same top three primitives.

## Runner

Launch a single manifest with:

```bash
cargo run --release --features cuda --example tournament -- \
  --experiment-manifest /Users/joseph/fractal/experiments/canonical/leaderboard/seed42-p1_contractive.json
```

RunPod wrapper launches should pass the same `--experiment-manifest` path through
to the tournament example so every attempt preserves the same logical experiment contract.
