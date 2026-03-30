# fractal

Fractal is a Rust harness for mutation-based learning experiments.

This workspace currently implements a seven-species fractal primitive tournament.

## Read first

- `ENGINEERING.md`
- `docs/harness-doctrine.md`
- `SPEC_v3_1.md`

## Commands

```bash
cargo build --release
cargo test
cargo run --example tournament
cargo run --example tournament -- --preset fast-test
cargo run --release --example tournament -- --preset research-medium
cargo run --release --example tournament -- --sequence first-run
cargo run --release --features cuda --example tournament -- --backend cuda --preset research-medium
scripts/runpod-tournament.sh --gpu-id "NVIDIA GeForce RTX 4090" -- --preset research-medium
```

## Runpod

`scripts/runpod-tournament.sh` creates or reuses a Runpod pod, syncs the current worktree snapshot, bootstraps the Rust toolchain if needed, and runs the CUDA tournament remotely.

Example:

```bash
scripts/runpod-tournament.sh \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  -- --preset research-medium
```

Behavior:

- Uses `runpodctl` and your registered SSH key to reach the pod over exposed TCP port `22`.
- Defaults to the official `runpod-torch-v240` template and syncs a clean copy of the repo without `.git` or `target`.
- Persists remote build artifacts under `<volumeMountPath>/.fractal-runpod/target` so repeated runs avoid full rebuilds.
- Stops pods it created after the run by default; use `--keep-pod` to leave them running.

## Goal

Keep the harness small, deterministic, extensible, and useful for comparative experiments.

## Notes

- `cargo run --example tournament` now defaults to sequential Burn Metal execution on Apple Silicon.
- Fast tests use CPU Candle execution for deterministic, low-overhead validation.
- `research-medium` is the first meaningful single-GPU leaderboard preset for Apple Silicon.
- `first-run` sequences `fast-test -> research-medium -> pressure-test` as the staged initial tournament path.
- Parallel execution remains available through `TournamentConfig::with_execution_mode(ExecutionMode::Parallel)`.
- The heavier spec-aligned configuration is available as `TournamentConfig::pressure_test()`.
- All recurrent state transitions in the model go through `rule.apply(...)`.
