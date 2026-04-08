# FlashAttention Rust Build Notes

Saved for later reference while evaluating Rust-native FlashAttention integration.

## Key idea

With Rust (`cargo` + `build.rs`), we can drive a fully parallel native CUDA/C++ build and keep it inside the repo's normal build workflow.

## Native CUDA / C++ kernel build

For the heavy native build (for example FA3 / FA4 kernels), use a `build.rs` plus the `cmake` crate.

```rust
fn main() {
    let mut config = cmake::Config::new("path/to/flash-attention/hopper");
    config
        .define("CMAKE_BUILD_PARALLEL_LEVEL", "96")
        .generator("Ninja")
        .build();

    println!("cargo::rerun-if-changed=...");
}
```

Notes:
- This is the Rust-side equivalent of tweaking `MAX_JOBS` in the upstream build.
- We can set the parallelism explicitly, or derive it from `std::thread::available_parallelism()`.
- `Ninja` is likely the right default generator for build throughput.

## Cargo behavior

- `cargo build` already parallelizes aggressively by default.
- A wrapper crate can build Rust code and native kernels in parallel across crates and inside the build script.

## Likely integration path

For a Rust wrapper around FA3 / FA4:
- build native CUDA artifacts in `build.rs`
- generate bindings with `bindgen` if needed
- expose the compiled library through Rust FFI

## Prior art

- `candle-flash-attn` already compiles FA2-era kernels as part of Cargo builds
- a similar wrapper pattern should work for FA3 / FA4

## FA4-specific note

If FA4 uses a CuTe-DSL Python -> PTX step, `build.rs` can orchestrate that too via `std::process::Command`, then hand off to `nvcc` / CMake.

## Why this matters for us

This is a plausible path if we want:
- a repo-integrated native attention backend
- deterministic build control from Rust
- full-core parallel compile behavior without relying on manual shell env setup
