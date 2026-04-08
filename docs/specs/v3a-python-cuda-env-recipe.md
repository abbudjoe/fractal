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
