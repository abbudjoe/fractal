#!/usr/bin/env bash
# Run a MaxText/JAX TPU port of the 8-layer Parcae/RGRP-control proof ladder.
#
# This intentionally targets the H100 proof-ladder shape rather than the earlier
# 4-layer TPU smoke shape:
#   - 8 physical decoder layers
#   - loop count 2 over the middle layer band
#   - d_model 128, 4 heads, mlp_dim 512
#   - context 256
#   - OpenLLaMA-sized vocabulary by default
#
# Execute inside a MaxText checkout that has been patched with
# scripts/patch_maxtext_rgrp.py.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_parcae_proof_ladder_${STAMP}}"
SEEDS="${SEEDS:-42}"
LANES="${LANES:-attention parcae-looped parcae-bx parcae-rgrp-control}"
STEPS="${STEPS:-30000}"
SMOKE_STEPS="${SMOKE_STEPS:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_STEPS="${EVAL_STEPS:-64}"
LOOP_COUNT="${LOOP_COUNT:-2}"
LOOP_POLICY="${LOOP_POLICY:-fixed}"
DEPTH_DISTRIBUTION="${DEPTH_DISTRIBUTION:-poisson}"
MU_REC="${MU_REC:-${LOOP_COUNT}}"
MU_BWD="${MU_BWD:-${LOOP_COUNT}}"
MIN_LOOP_COUNT="${MIN_LOOP_COUNT:-1}"
MAX_LOOP_COUNT="${MAX_LOOP_COUNT:-0}"
PERSEQ_MAX_LOOP_COUNT="${PERSEQ_MAX_LOOP_COUNT:-4}"
PARCAE_DISCRETIZATION="${PARCAE_DISCRETIZATION:-stable-exp}"
PARCAE_CONTROL_DIAGNOSTICS="${PARCAE_CONTROL_DIAGNOSTICS:-false}"
PARCAE_CONTROL_MODE="${PARCAE_CONTROL_MODE:-gate-value}"
PARCAE_CONTROL_BOTTLENECK_DIM="${PARCAE_CONTROL_BOTTLENECK_DIM:-0}"
PARCAE_CONTROL_GATE_BLEND="${PARCAE_CONTROL_GATE_BLEND:-0.0}"
PARCAE_CONTROL_BASE_BLEND="${PARCAE_CONTROL_BASE_BLEND:-0.5}"
PARCAE_CONTROL_VALUE_SCALE="${PARCAE_CONTROL_VALUE_SCALE:-1.0}"

BASE_OUTPUT_DIRECTORY="${BASE_OUTPUT_DIRECTORY:-gs://fractal-maxtext-runs-81f2add4}"
HF_PATH="${HF_PATH:-Salesforce/wikitext}"
HF_NAME="${HF_NAME:-wikitext-103-raw-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
HF_EVAL_SPLIT="${HF_EVAL_SPLIT:-validation}"
HF_TRAIN_FILES="${HF_TRAIN_FILES:-}"
HF_EVAL_FILES="${HF_EVAL_FILES:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-openlm-research/open_llama_3b_v2}"

SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-64}"
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

steps_for_lane() {
  if [[ "${SMOKE_STEPS}" != "0" ]]; then
    echo "${SMOKE_STEPS}"
  else
    echo "${STEPS}"
  fi
}

run_attention() {
  local seed="$1"
  local steps
  steps="$(steps_for_lane)"
  local run_name="parcae8-attention-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${steps}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

run_parcae() {
  local lane="$1"
  local seed="$2"
  local steps
  steps="$(steps_for_lane)"
  local candidate
  local loop_policy="${LOOP_POLICY}"
  local max_loop_count="${MAX_LOOP_COUNT}"
  local control_mode="${PARCAE_CONTROL_MODE}"
  local control_bottleneck_dim="${PARCAE_CONTROL_BOTTLENECK_DIM}"
  local control_gate_blend="${PARCAE_CONTROL_GATE_BLEND}"
  local control_value_scale="${PARCAE_CONTROL_VALUE_SCALE}"
  case "${lane}" in
    parcae-looped)
      candidate="parcae-looped-attention"
      ;;
    parcae-bx)
      candidate="parcae-bx-looped-attention"
      ;;
    parcae-bx-perseq)
      candidate="parcae-bx-looped-attention"
      loop_policy="per-sequence"
      max_loop_count="${PERSEQ_MAX_LOOP_COUNT}"
      ;;
    parcae-rgrp-control|parcae-p20-control)
      candidate="parcae-rgrp-control-looped-attention"
      ;;
    parcae-rgrp-control-perseq|parcae-p20-control-perseq)
      candidate="parcae-rgrp-control-looped-attention"
      loop_policy="per-sequence"
      max_loop_count="${PERSEQ_MAX_LOOP_COUNT}"
      ;;
    parcae-rgrp-control-thin|parcae-p20-control-thin)
      candidate="parcae-rgrp-control-looped-attention"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      ;;
    parcae-rgrp-control-vscale0p5|parcae-p20-control-vscale0p5)
      candidate="parcae-rgrp-control-looped-attention"
      control_value_scale=0.5
      ;;
    parcae-rgrp-control-thin-vscale0p5|parcae-p20-control-thin-vscale0p5)
      candidate="parcae-rgrp-control-looped-attention"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      control_value_scale=0.5
      ;;
    parcae-rgrp-control-thin-gate|parcae-p20-control-thin-gate)
      candidate="parcae-rgrp-control-looped-attention"
      control_mode="gate-only"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      ;;
    parcae-rgrp-control-quarter|parcae-p20-control-quarter)
      candidate="parcae-rgrp-control-looped-attention"
      control_bottleneck_dim="${PARCAE_CONTROL_QUARTER_DIM:-$((D_MODEL / 4))}"
      ;;
    parcae-rgrp-control-thin-value|parcae-p20-control-thin-value)
      candidate="parcae-rgrp-control-looped-attention"
      control_mode="value-only"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      ;;
    parcae-rgrp-control-thin-gate-baseblend|parcae-p20-control-thin-gate-baseblend)
      candidate="parcae-rgrp-control-looped-attention"
      control_mode="gate-only"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      control_gate_blend="${PARCAE_CONTROL_BASE_BLEND}"
      ;;
    parcae-rgrp-control-thin-baseblend|parcae-p20-control-thin-baseblend)
      candidate="parcae-rgrp-control-looped-attention"
      control_bottleneck_dim="${PARCAE_CONTROL_THIN_DIM:-$((D_MODEL / 2))}"
      control_gate_blend="${PARCAE_CONTROL_BASE_BLEND}"
      ;;
    *)
      echo "unknown Parcae lane: ${lane}" >&2
      exit 2
      ;;
  esac
  local run_name="parcae8-${lane}-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${steps}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    "fractal_candidate=${candidate}" \
    "fractal_parcae_loop_count=${LOOP_COUNT}" \
    "fractal_parcae_loop_policy=${loop_policy}" \
    "fractal_parcae_depth_distribution=${DEPTH_DISTRIBUTION}" \
    "fractal_parcae_mu_rec=${MU_REC}" \
    "fractal_parcae_mu_bwd=${MU_BWD}" \
    "fractal_parcae_min_loop_count=${MIN_LOOP_COUNT}" \
    "fractal_parcae_max_loop_count=${max_loop_count}" \
    "fractal_parcae_discretization=${PARCAE_DISCRETIZATION}" \
    "fractal_parcae_control_diagnostics=${PARCAE_CONTROL_DIAGNOSTICS}" \
    "fractal_parcae_control_mode=${control_mode}" \
    "fractal_parcae_control_bottleneck_dim=${control_bottleneck_dim}" \
    "fractal_parcae_control_gate_blend=${control_gate_blend}" \
    "fractal_parcae_control_value_scale=${control_value_scale}" \
    fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
    fractal_rgrp_scan_unroll=3 \
    fractal_rgrp_projection_mode=sequence \
    fractal_rgrp_trig_mode=precompute \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

for seed in ${SEEDS}; do
  for lane in ${LANES}; do
    case "${lane}" in
      attention)
        run_attention "${seed}"
        ;;
      parcae-looped|parcae-bx|parcae-bx-perseq|parcae-rgrp-control|parcae-rgrp-control-perseq|parcae-p20-control|parcae-p20-control-perseq|parcae-rgrp-control-thin|parcae-p20-control-thin|parcae-rgrp-control-vscale0p5|parcae-p20-control-vscale0p5|parcae-rgrp-control-thin-vscale0p5|parcae-p20-control-thin-vscale0p5|parcae-rgrp-control-thin-gate|parcae-p20-control-thin-gate|parcae-rgrp-control-quarter|parcae-p20-control-quarter|parcae-rgrp-control-thin-value|parcae-p20-control-thin-value|parcae-rgrp-control-thin-gate-baseblend|parcae-p20-control-thin-gate-baseblend|parcae-rgrp-control-thin-baseblend|parcae-p20-control-thin-baseblend)
        run_parcae "${lane}" "${seed}"
        ;;
      *)
        echo "unknown lane: ${lane}" >&2
        exit 2
        ;;
    esac
  done
done

ls -lh "${OUTDIR}"
