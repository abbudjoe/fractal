# v3A Python CUDA Environment Recipe

This note defines the shared Linux/CUDA bootstrap path for the Python Mamba-3
research stack.

It exists to keep:

* local Linux/CUDA runs
* RunPod CUDA runs
* and the official `mamba_ssm` install contract

on one surface.

## Source Of Truth

The bootstrap primitive is:

* [scripts/bootstrap_v3a_python_mamba3_cuda_env.sh](/private/tmp/fractal-v3a-merge-G4mipf/scripts/bootstrap_v3a_python_mamba3_cuda_env.sh)

The dependency surface is:

* [scripts/requirements-v3a-python-mamba3.txt](/private/tmp/fractal-v3a-merge-G4mipf/scripts/requirements-v3a-python-mamba3.txt)

The RunPod wrapper now delegates to that script instead of carrying an
independent install flow.

## Intended Host

This recipe is for:

* Linux
* NVIDIA GPU
* CUDA-enabled toolchain (`nvcc` present)

It is not intended for macOS arm64. Local macOS development can use the repo
`.venv` for `torch` / `pytest`, but the official `mamba_ssm` + Triton stack
still belongs on Linux/CUDA.

## Standard Bootstrap

```bash
cd /path/to/fractal
bash scripts/bootstrap_v3a_python_mamba3_cuda_env.sh \
  --venv-dir .venv-cuda-mamba3
source .venv-cuda-mamba3/bin/activate
```

That will:

1. create a clean virtualenv
2. install a pinned CUDA PyTorch build into that env
3. install the Triton API level currently required by the working native
   `mamba_ssm` path
4. install the shared Python requirements
5. install `causal-conv1d` from source without letting pip rewrite the
   Torch/Triton pair
6. install official `mamba_ssm` from source with `MAMBA_FORCE_BUILD=TRUE`
7. constrain both native extension builds to the detected GPU compute
   capability when available

## Optional Torch Pin

If the host does not already provide the desired CUDA PyTorch build, install it
explicitly during bootstrap:

```bash
bash scripts/bootstrap_v3a_python_mamba3_cuda_env.sh \
  --venv-dir .venv-cuda-mamba3 \
  --torch 2.4.1 \
  --torch-index-url https://download.pytorch.org/whl/cu124
```

## RunPod Contract

RunPod Python benchmarks using:

* [scripts/runpod-tournament.sh](/private/tmp/fractal-v3a-merge-G4mipf/scripts/runpod-tournament.sh)

with:

* `--python-install-mode official-mamba3`

That path now uses the same bootstrap script as the local Linux/CUDA recipe.

For compile-focused Path 1 runs that use `torch.compile` on `A` or `A+P`
without native `mamba_ssm`, use:

* `--python-install-mode compile-safe`

That keeps Torch on its compatible bundled Triton path instead of upgrading the
environment to the newer Triton level required by the official native Mamba
stack.

For Path 1 primitive experiments that need the newer standalone Triton package
without installing native `mamba_ssm`, use:

* `--python-install-mode primitive-triton`

That env exists to host the upcoming custom primitive Triton runtime without
reusing the `compile-safe` control plane or pretending it is the same thing as
the official native Mamba stack.

## Why This Exists

Before this change, the RunPod wrapper had its own inline `mamba_ssm` bootstrap
logic. That meant local Linux/CUDA bring-up and RunPod bring-up could silently
drift.

The shared bootstrap script fixes that control-plane split.

## Current Native Mamba Contract

The current working native Mamba bring-up in this repo is pinned around:

* `torch 2.4.1` from `cu124`
* `triton 3.6.0`
* `causal-conv1d` built from source at `v1.6.1` with `--no-deps`
* official `mamba_ssm` built from source with `--no-build-isolation`

This is intentionally explicit in the bootstrap script so the env does not
silently fall back to the Torch-bundled Triton runtime or drift to a different
CUDA toolchain.

For Blackwell devices on older RunPod CUDA devel images, the device arch and
compiler arch are treated as separate contracts. The bootstrap can select the
cu128 Torch wheel for an RTX 5090 while compiling source extensions against
the highest `nvcc`-supported `+PTX` target when the local compiler cannot emit
`sm_120` directly. This avoids the broken implicit assumption that
`nvidia-smi` compute capability is always a valid `nvcc` build target.

## Compile-Safe Contract

The compile-safe Path 1 env is intentionally different:

* `torch 2.4.1` from `cu124` on pre-Blackwell CUDA hosts
* `torch 2.10.0` from `cu128` on Blackwell-or-newer hosts (`sm_120+`)
* Torch-bundled Triton matched to the selected Torch wheel
* shared Python research requirements
* no `causal-conv1d` native build
* no `mamba_ssm` install

This env exists so `torch.compile` runs for `A` and `A+P` can be evaluated
without colliding with the Triton level required by native official Mamba.
The bootstrap chooses the Blackwell-safe wheel before reusing any base-image
Torch install so RTX 5090 runs do not silently inherit an `sm_90`-only runtime.

## Primitive-Triton Contract

The primitive-triton env is a third surface:

* `torch 2.10.0` from `cu128`
* Torch's matching bundled `triton 3.6.0`
* shared Python research requirements
* no `causal-conv1d` native build
* no `mamba_ssm` install

This env exists so the primitive line can move onto the newer Triton stack on
its own typed control plane. It should be paired with:

* `runtime.env_kind = primitive-triton`
* `runtime.primitive_runtime_backend = torch|triton`

The current Triton frontier is intentionally split into two explicit lanes:

* frozen fast lane:
  * `primitive_runtime_backend=triton`
  * `primitive_state_transform_profile=block-diagonal-4`
* dense quality lane:
  * `primitive_runtime_backend=triton`
  * `primitive_state_transform_profile=dense`

The shared RunPod Triton duo runner now defaults to the frozen fast lane.
Dense remains an explicit override while the dense sequence kernel is still
being redesigned.
