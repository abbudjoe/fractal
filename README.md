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
```

## Goal

Keep the harness small, deterministic, extensible, and useful for comparative experiments.

## Notes

- `cargo run --example tournament` now defaults to sequential Burn Metal execution on Apple Silicon.
- Fast tests use CPU Candle execution for deterministic, low-overhead validation.
- Parallel execution remains available through `TournamentConfig::with_execution_mode(ExecutionMode::Parallel)`.
- The heavier spec-aligned configuration is available as `TournamentConfig::pressure_test()`.
- All recurrent state transitions in the model go through `rule.apply(...)`.
