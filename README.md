# fractal

Rust workspace for a seven-species fractal primitive tournament.

## Commands

```bash
cargo build --release
cargo test
cargo run --example tournament
```

## Notes

- `cargo run --example tournament` uses CPU-friendly defaults so a full generation completes quickly.
- The heavier spec-aligned configuration is available as `TournamentConfig::pressure_test()`.
- All recurrent state transitions in the model go through `rule.apply(...)`.
