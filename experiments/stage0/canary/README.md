# Stage 0 Canary

The checked-in canary manifest is:

- `seed42-p1_contractive.json`

Before launching the canary, materialize the frozen bridge vocab artifact from the
manifest-owned corpus slice and tokenizer asset:

```bash
cargo run --release --example tournament -- --prepare-stage0-assets --experiment-manifest experiments/stage0/canary/seed42-p1_contractive.json
```

That preparation step is deterministic and writes:

- `experiments/stage0/assets/open_llama_3b_v2/fineweb-stage0-canary-bridge-vocab.json`

After preparation, the canary manifest can be executed with the normal manifest run path.
