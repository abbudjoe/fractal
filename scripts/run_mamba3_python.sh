#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
repo_modern_venv_python="${repo_root}/.venv/bin/python"
repo_legacy_venv_python="${repo_root}/.venv-mamba3/bin/python"

if [[ -n "${FRACTAL_MAMBA3_PYTHON:-}" ]]; then
  exec "${FRACTAL_MAMBA3_PYTHON}" "$@"
fi

if [[ -x "${repo_modern_venv_python}" ]]; then
  exec "${repo_modern_venv_python}" "$@"
fi

if [[ -x "${repo_legacy_venv_python}" ]]; then
  exec "${repo_legacy_venv_python}" "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  if python3 -c 'import torch' >/dev/null 2>&1; then
    exec python3 "$@"
  fi
fi

echo "error: no Python runtime with torch available for the Mamba3 parity harness." >&2
echo "hint: set FRACTAL_MAMBA3_PYTHON or create ${repo_modern_venv_python}." >&2
exit 1
