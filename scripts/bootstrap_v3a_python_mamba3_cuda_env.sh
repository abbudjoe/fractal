#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: bootstrap_v3a_python_mamba3_cuda_env.sh --venv-dir PATH [options]

Bootstraps the official Python Mamba3 research environment for Linux/CUDA.

Required:
  --venv-dir PATH                  Target virtualenv directory.

Optional:
  --requirements PATH             Requirements file. Default: scripts/requirements-v3a-python-mamba3.txt
  --repo-root PATH                Repo root. Default: parent of this script
  --python BIN                    Python executable used to create the env. Default: python3
  --install-mode MODE             requirements-only|official-mamba3. Default: official-mamba3
  --torch-index-url URL           Install torch from this index before requirements (optional)
  --torch VERSION                 Explicit torch version to install before requirements (optional)
  --cuda-arch-list ARCH           Override detected CUDA arch, e.g. 8.9
  --force-recreate                Delete and recreate the target venv first
  --help                          Show this help

Environment:
  CUDA_HOME                       Defaults to /usr/local/cuda if unset
  TORCH_CUDA_ARCH_LIST            Used if --cuda-arch-list is not provided

Notes:
  - This script is intended for Linux hosts with NVIDIA CUDA support.
  - The official Mamba3 path is installed from source with MAMBA_FORCE_BUILD=TRUE.
  - When a CUDA arch is known, the Mamba build is patched to compile only that arch.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUIREMENTS_FILE="${REPO_ROOT}/scripts/requirements-v3a-python-mamba3.txt"
PYTHON_BIN="python3"
INSTALL_MODE="official-mamba3"
VENV_DIR=""
TORCH_INDEX_URL=""
TORCH_VERSION=""
CUDA_ARCH_LIST_OVERRIDE="${TORCH_CUDA_ARCH_LIST:-}"
FORCE_RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --install-mode)
      INSTALL_MODE="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --cuda-arch-list)
      CUDA_ARCH_LIST_OVERRIDE="$2"
      shift 2
      ;;
    --force-recreate)
      FORCE_RECREATE=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${VENV_DIR}" ]] || die "--venv-dir is required"
[[ "${INSTALL_MODE}" == "requirements-only" || "${INSTALL_MODE}" == "official-mamba3" ]] || die "--install-mode must be requirements-only or official-mamba3"

if [[ "$(uname -s)" != "Linux" ]]; then
  die "this bootstrap is for Linux/CUDA only"
fi
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python executable not found: ${PYTHON_BIN}"
command -v git >/dev/null 2>&1 || die "git is required"
command -v nvcc >/dev/null 2>&1 || die "nvcc is required; use a CUDA-enabled host/template"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  die "requirements file not found: ${REQUIREMENTS_FILE}"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ "${FORCE_RECREATE}" == "1" ]]; then
  rm -rf "${VENV_DIR}"
fi

"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel >/dev/null

if [[ -n "${TORCH_VERSION}" ]]; then
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    python -m pip install --no-build-isolation --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  else
    python -m pip install --no-build-isolation "torch==${TORCH_VERSION}"
  fi
fi

python -m pip install --no-build-isolation -r "${REQUIREMENTS_FILE}"

if [[ "${INSTALL_MODE}" == "requirements-only" ]]; then
  echo "bootstrapped requirements-only env at ${VENV_DIR}"
  exit 0
fi

CUDA_ARCH_LIST="${CUDA_ARCH_LIST_OVERRIDE}"
if [[ -z "${CUDA_ARCH_LIST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
fi

mamba_src_dir="$(mktemp -d)"
trap 'rm -rf "${mamba_src_dir}"' EXIT

git clone --depth 1 https://github.com/state-spaces/mamba.git "${mamba_src_dir}"

if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  patch_script_path="${mamba_src_dir}/.patch_setup_arch.py"
  cat >"${patch_script_path}" <<'PY'
import os
import pathlib
import sys

setup_path = pathlib.Path(sys.argv[1])
arch = os.environ["CUDA_ARCH_LIST"].strip()
major, minor = arch.split(".", 1)
sm = f"{major}{minor}"
lines = setup_path.read_text(encoding="utf-8").splitlines()
start = None
end = None
for index, line in enumerate(lines):
    if 'cc_flag.append("-gencode")' in line and start is None:
        start = index
    if start is not None and line.startswith("    # HACK:"):
        end = index
        break
if start is None or end is None or end <= start:
    raise SystemExit(f"failed to locate CUDA arch block in {setup_path}")
replacement = [
    '        cc_flag.append("-gencode")',
    f'        cc_flag.append("arch=compute_{sm},code=sm_{sm}")',
    "",
]
patched = lines[:start] + replacement + lines[end:]
setup_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
print(f"patched {setup_path} to build only sm_{sm}")
PY
  CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" python "${patch_script_path}" "${mamba_src_dir}/setup.py"
  export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
fi

MAMBA_FORCE_BUILD=TRUE \
python -m pip install --no-build-isolation --no-deps --no-cache-dir "${mamba_src_dir}"

python - <<'PY'
import importlib
import torch

importlib.import_module("mamba_ssm")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("mamba_ssm ok")
PY

echo "bootstrapped official Mamba3 CUDA env at ${VENV_DIR}"
