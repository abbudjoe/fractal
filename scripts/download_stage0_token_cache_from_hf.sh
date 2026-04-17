#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: download_stage0_token_cache_from_hf.sh --repo-id USER_OR_ORG/REPO --output-root PATH [options]

Downloads a Stage 0 token-cache tarball from a Hugging Face dataset repo,
verifies the SHA-256 sidecar, and extracts it under --output-root.

Required:
  --repo-id USER_OR_ORG/REPO       Hugging Face dataset repo id.
  --output-root PATH              Directory where the cache folder should be extracted.

Optional:
  --artifact NAME                 Tarball name. Default: fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst
  --repo-type TYPE                Default: dataset
  --help                          Show this help.

Environment:
  HF_TOKEN                        Required for private dataset repos unless already logged in.
EOF
}

REPO_ID=""
OUTPUT_ROOT=""
REPO_TYPE="dataset"
ARTIFACT="fineweb-cc-main-2024-10-openllama-tokens-250m-v1.tar.zst"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --repo-type)
      REPO_TYPE="$2"
      shift 2
      ;;
    --artifact)
      ARTIFACT="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${REPO_ID}" ]] || { echo "--repo-id is required" >&2; exit 2; }
[[ -n "${OUTPUT_ROOT}" ]] || { echo "--output-root is required" >&2; exit 2; }
command -v huggingface-cli >/dev/null 2>&1 || {
  echo "huggingface-cli is required; install with: python -m pip install -U huggingface_hub" >&2
  exit 1
}
command -v zstd >/dev/null 2>&1 || {
  echo "zstd is required to extract ${ARTIFACT}" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}"
WORK_DIR="${OUTPUT_ROOT}/.hf-token-cache-download"
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

huggingface-cli download "${REPO_ID}" "${ARTIFACT}" \
  --repo-type "${REPO_TYPE}" \
  --local-dir "${WORK_DIR}"
huggingface-cli download "${REPO_ID}" "${ARTIFACT}.sha256" \
  --repo-type "${REPO_TYPE}" \
  --local-dir "${WORK_DIR}"

(
  cd "${WORK_DIR}"
  shasum -a 256 -c "${ARTIFACT}.sha256"
)

tar -C "${OUTPUT_ROOT}" -I zstd -xf "${WORK_DIR}/${ARTIFACT}"
rm -rf "${WORK_DIR}"

echo "${OUTPUT_ROOT}/${ARTIFACT%.tar.zst}/manifest.json"

