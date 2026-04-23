#!/usr/bin/env bash
# Run corrected Path1 scale-leader candidates in a patched MaxText checkout.
#
# Execute from the root of a MaxText checkout after applying
# scripts/patch_maxtext_rgrp.py from this repository.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_path1_scale_leaders_${STAMP}}"
SEEDS="${SEEDS:-42}"
LANES="${LANES:-attention causal-topk-route50-layer1 mod-train-topc-route50-layer1 fixed-looped-lm input-injected-looped-lm universal-transformer-act mor-expert-choice d3-route25-accel}"
STEPS="${STEPS:-1024}"
EVAL_INTERVAL="${EVAL_INTERVAL:-256}"
EVAL_STEPS="${EVAL_STEPS:-16}"
PATH1_DIAGNOSTICS="${PATH1_DIAGNOSTICS:-true}"

BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://fractal-maxtext-runs-81f2add4}"
HF_PATH="${HF_PATH:-Salesforce/wikitext}"
HF_NAME="${HF_NAME:-wikitext-103-raw-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
HF_EVAL_SPLIT="${HF_EVAL_SPLIT:-validation}"
HF_TRAIN_FILES="${HF_TRAIN_FILES:-}"
HF_EVAL_FILES="${HF_EVAL_FILES:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-openlm-research/open_llama_3b_v2}"

SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
D_MODEL="${D_MODEL:-128}"
MLP_DIM="${MLP_DIM:-512}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-32}"
DTYPE="${DTYPE:-bfloat16}"
LR="${LR:-0.001}"

mkdir -p "${OUTDIR}"

if [[ ! -d ".venv" ]]; then
  echo "expected to run from MaxText checkout with .venv present" >&2
  exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python3}"
SUMMARY_FILTER='/number parameters:|Memstats: After params initialized:|Using \\(GB\\)|eval metrics after step|completed step:/ { print; fflush(); }'

COMMON=(
  "base_output_directory=${BASE_OUTPUT_DIRECTORY}"
  dataset_type=hf
  "hf_path=${HF_PATH}"
  "hf_name=${HF_NAME}"
  "hf_eval_split=${HF_EVAL_SPLIT}"
  "hf_train_files=${HF_TRAIN_FILES}"
  "hf_eval_files=${HF_EVAL_FILES}"
  tokenizer_type=huggingface
  "tokenizer_path=${TOKENIZER_PATH}"
  "train_split=${TRAIN_SPLIT}"
  log_period=50
  "eval_interval=${EVAL_INTERVAL}"
  "eval_steps=${EVAL_STEPS}"
  enable_checkpointing=false
  save_checkpoint_on_completion=false
  log_config=false
  decoder_block=default
  scan_layers=false
  "max_target_length=${SEQ_LEN}"
  "vocab_size=${VOCAB_SIZE}"
  "base_emb_dim=${D_MODEL}"
  "base_mlp_dim=${MLP_DIM}"
  "base_num_decoder_layers=${LAYERS}"
  "base_num_query_heads=${HEADS}"
  "base_num_kv_heads=${HEADS}"
  "head_dim=${HEAD_DIM}"
  "per_device_batch_size=${BATCH_SIZE}"
  "learning_rate=${LR}"
  "dtype=${DTYPE}"
  enable_data_shuffling=true
)

run_attention() {
  local seed="$1"
  local run_name="path1scale-attention-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${STEPS}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

run_path1_lane() {
  local lane="$1"
  local seed="$2"
  local route_fraction="0.5"
  local route_layers="1"
  local loop_count="4"
  local shared_layers="2"
  local act_threshold="0.99"
  local accel_threshold="0.6"
  local min_steps="2"

  case "${lane}" in
    causal-topk-route50-layer1)
      route_fraction="0.5"
      route_layers="1"
      ;;
    mod-train-topc-route50-layer1)
      route_fraction="0.5"
      route_layers="1"
      ;;
    fixed-looped-lm)
      loop_count="${PATH1_LOOP_COUNT:-4}"
      shared_layers="${PATH1_SHARED_LAYERS:-2}"
      ;;
    input-injected-looped-lm)
      loop_count="${PATH1_LOOP_COUNT:-4}"
      shared_layers="${PATH1_SHARED_LAYERS:-2}"
      ;;
    universal-transformer-act)
      loop_count="${PATH1_LOOP_COUNT:-4}"
      shared_layers="${PATH1_UT_SHARED_LAYERS:-1}"
      act_threshold="${PATH1_ACT_THRESHOLD:-0.99}"
      ;;
    mor-expert-choice)
      route_fraction="${PATH1_MOR_ROUTE_FRACTION:-0.25}"
      loop_count="${PATH1_MOR_LOOP_COUNT:-3}"
      shared_layers="${PATH1_MOR_SHARED_LAYERS:-1}"
      ;;
    d3-route25-accel)
      route_fraction="${PATH1_D3_ROUTE_FRACTION:-0.25}"
      loop_count="${PATH1_D3_LOOP_COUNT:-3}"
      accel_threshold="${PATH1_D3_ACCEL_THRESHOLD:-0.6}"
      min_steps="${PATH1_D3_MIN_STEPS:-2}"
      ;;
    *)
      echo "unknown Path1 lane: ${lane}" >&2
      exit 2
      ;;
  esac

  local run_name="path1scale-${lane}-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${STEPS}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    "fractal_candidate=${lane}" \
    "fractal_path1_diagnostics=${PATH1_DIAGNOSTICS}" \
    "fractal_path1_route_fraction=${route_fraction}" \
    "fractal_path1_route_layers=${route_layers}" \
    "fractal_path1_loop_count=${loop_count}" \
    "fractal_path1_shared_layers=${shared_layers}" \
    "fractal_path1_act_threshold=${act_threshold}" \
    "fractal_path1_accel_threshold=${accel_threshold}" \
    "fractal_path1_min_steps=${min_steps}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

for seed in ${SEEDS}; do
  for lane in ${LANES}; do
    case "${lane}" in
      attention)
        run_attention "${seed}"
        ;;
      *)
        run_path1_lane "${lane}" "${seed}"
        ;;
    esac
  done
done

ls -lh "${OUTDIR}"
