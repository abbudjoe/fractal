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

1. create a virtualenv with system site packages
2. install the shared Python requirements
3. install official `mamba_ssm` from source with `MAMBA_FORCE_BUILD=TRUE`
4. constrain the build to the detected GPU compute capability when available

## Optional Torch Pin

If the host does not already provide the desired CUDA PyTorch build, install it
explicitly during bootstrap:

```bash
bash scripts/bootstrap_v3a_python_mamba3_cuda_env.sh \
  --venv-dir .venv-cuda-mamba3 \
  --torch 2.11.0 \
  --torch-index-url https://download.pytorch.org/whl/cu128
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
