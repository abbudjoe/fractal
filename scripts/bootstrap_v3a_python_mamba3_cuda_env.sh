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
  --install-mode MODE             requirements-only|official-mamba3|compile-safe|primitive-triton. Default: official-mamba3
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
TRITON_VERSION="3.6.0"
PRIMITIVE_TRITON_TORCH_VERSION="2.4.1"
CAUSAL_CONV1D_REPO="https://github.com/Dao-AILab/causal-conv1d.git"
CAUSAL_CONV1D_GIT_REF="v1.6.1"
MAMBA_REPO="https://github.com/state-spaces/mamba.git"
CUDA_ARCH_LIST_OVERRIDE="${TORCH_CUDA_ARCH_LIST:-}"
FORCE_RECREATE=0
PRIMITIVE_TRITON_TORCH_DEPS=(
  filelock
  typing-extensions
  sympy
  networkx
  jinja2
  fsspec
  nvidia-cuda-nvrtc-cu12==12.4.99
  nvidia-cuda-runtime-cu12==12.4.99
  nvidia-cuda-cupti-cu12==12.4.99
  nvidia-cudnn-cu12==9.1.0.70
  nvidia-cublas-cu12==12.4.2.65
  nvidia-cufft-cu12==11.2.0.44
  nvidia-curand-cu12==10.3.5.119
  nvidia-cusolver-cu12==11.6.0.99
  nvidia-cusparse-cu12==12.3.0.142
  nvidia-nccl-cu12==2.20.5
  nvidia-nvtx-cu12==12.4.99
  nvidia-nvjitlink-cu12==12.4.99
)

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
[[ "${INSTALL_MODE}" == "requirements-only" || "${INSTALL_MODE}" == "official-mamba3" || "${INSTALL_MODE}" == "compile-safe" || "${INSTALL_MODE}" == "primitive-triton" ]] || die "--install-mode must be requirements-only, compile-safe, primitive-triton, or official-mamba3"

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

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1090
. "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel >/dev/null

if [[ -z "${TORCH_VERSION}" ]]; then
  detected_torch_version="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
version = torch.__version__.split("+", 1)[0]
cuda_version = getattr(torch.version, "cuda", None)
if cuda_version:
    print(f"{version}|{cuda_version}")
PY
)" || detected_torch_version=""
  if [[ -n "${detected_torch_version}" ]]; then
    TORCH_VERSION="${detected_torch_version%%|*}"
    if [[ -z "${TORCH_INDEX_URL}" ]]; then
      detected_cuda_version="${detected_torch_version#*|}"
      detected_cuda_tag="cu${detected_cuda_version//./}"
      TORCH_INDEX_URL="https://download.pytorch.org/whl/${detected_cuda_tag}"
    fi
  else
    TORCH_VERSION="2.4.1"
    if [[ -z "${TORCH_INDEX_URL}" ]]; then
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    fi
    echo "defaulting torch bootstrap to ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
  fi
fi

if [[ "${INSTALL_MODE}" == "primitive-triton" && -z "${TORCH_VERSION}" ]]; then
  TORCH_VERSION="${PRIMITIVE_TRITON_TORCH_VERSION}"
  if [[ -z "${TORCH_INDEX_URL}" ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
  fi
fi

if [[ -n "${TORCH_VERSION}" ]]; then
  echo "installing pinned torch ${TORCH_VERSION}${TORCH_INDEX_URL:+ from ${TORCH_INDEX_URL}}"
  if [[ "${INSTALL_MODE}" == "primitive-triton" ]]; then
    if [[ "${TORCH_VERSION}" != "${PRIMITIVE_TRITON_TORCH_VERSION}" ]]; then
      die "primitive-triton currently supports only --torch ${PRIMITIVE_TRITON_TORCH_VERSION}"
    fi
    if [[ -z "${TORCH_INDEX_URL}" ]]; then
      die "primitive-triton requires an explicit CUDA torch index url"
    fi
    python -m pip install --no-build-isolation --index-url "${TORCH_INDEX_URL}" \
      "${PRIMITIVE_TRITON_TORCH_DEPS[@]}"
    python -m pip install --no-build-isolation --index-url "${TORCH_INDEX_URL}" --no-deps \
      "torch==${TORCH_VERSION}"
  elif [[ -n "${TORCH_INDEX_URL}" ]]; then
    python -m pip install --no-build-isolation --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"
  else
    python -m pip install --no-build-isolation "torch==${TORCH_VERSION}"
  fi
fi

python -m pip install --no-build-isolation -r "${REQUIREMENTS_FILE}"

CUDA_ARCH_LIST="${CUDA_ARCH_LIST_OVERRIDE}"
if [[ -z "${CUDA_ARCH_LIST}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
fi

patch_cuda_setup_arch() {
  local source_dir="$1"
  if [[ -z "${CUDA_ARCH_LIST}" ]]; then
    return 0
  fi
  python "${REPO_ROOT}/python/runtime/cuda_setup_patch.py" \
    "${source_dir}/setup.py" \
    --arch "${CUDA_ARCH_LIST}"
}

if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"
fi

if [[ "${INSTALL_MODE}" == "official-mamba3" || "${INSTALL_MODE}" == "primitive-triton" ]]; then
  echo "installing triton ${TRITON_VERSION} for official mamba runtime compatibility"
  python -m pip install --no-build-isolation "triton==${TRITON_VERSION}"
fi

if [[ "${INSTALL_MODE}" == "requirements-only" ]]; then
  echo "bootstrapped requirements-only env at ${VENV_DIR}"
  exit 0
fi

if [[ "${INSTALL_MODE}" == "compile-safe" ]]; then
  echo "bootstrapped compile-safe env at ${VENV_DIR}"
  exit 0
fi

if [[ "${INSTALL_MODE}" == "primitive-triton" ]]; then
  echo "bootstrapped primitive-triton env at ${VENV_DIR}"
  exit 0
fi

causal_src_dir="$(mktemp -d)"
mamba_src_dir="$(mktemp -d)"
trap 'rm -rf "${causal_src_dir}" "${mamba_src_dir}"' EXIT

git clone --depth 1 --branch "${CAUSAL_CONV1D_GIT_REF}" "${CAUSAL_CONV1D_REPO}" "${causal_src_dir}"
if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  patch_cuda_setup_arch "${causal_src_dir}"
fi
echo "installing causal-conv1d ${CAUSAL_CONV1D_GIT_REF} from source without dependency rewrites"
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
python -m pip install --no-build-isolation --no-deps --no-cache-dir "${causal_src_dir}"

git clone --depth 1 "${MAMBA_REPO}" "${mamba_src_dir}"

if [[ -n "${CUDA_ARCH_LIST}" ]]; then
  patch_cuda_setup_arch "${mamba_src_dir}"
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
