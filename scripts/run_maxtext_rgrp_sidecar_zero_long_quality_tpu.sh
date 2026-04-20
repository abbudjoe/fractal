#!/usr/bin/env bash
# Run a longer matched MaxText quality comparison for attention control versus
# the layer-1 zero-init RGRP MLP sidecar.
#
# Execute this inside a patched MaxText checkout.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_sidecar_zero_long_${STAMP}}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_STEPS="${EVAL_STEPS:-10}"
mkdir -p "${OUTDIR}"

if [[ ! -d ".venv" ]]; then
  echo "expected to run from MaxText checkout with .venv present" >&2
  exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python3}"

COMMON=(
  base_output_directory=gs://fractal-maxtext-runs-81f2add4
  dataset_type=hf
  hf_path=roneneldan/TinyStories
  hf_eval_split=validation
  tokenizer_type=huggingface
  tokenizer_path=gpt2
  train_split=train
  "steps=${STEPS}"
  log_period=50
  "eval_interval=${EVAL_INTERVAL}"
  "eval_steps=${EVAL_STEPS}"
  enable_checkpointing=false
  save_checkpoint_on_completion=false
  log_config=false
  decoder_block=default
  scan_layers=false
  "data_shuffle_seed=${SEED}"
  "init_weights_seed=${SEED}"
  max_target_length=512
  vocab_size=50257
  base_emb_dim=256
  base_mlp_dim=1024
  base_num_decoder_layers=4
  base_num_query_heads=4
  base_num_kv_heads=4
  head_dim=64
  per_device_batch_size=4
  learning_rate=0.001
  dtype=bfloat16
)

run_attention() {
  local run_name="attention-unscanned-seed${SEED}-long-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

run_zero_sidecar() {
  local run_name="rgrp-sidecar-layer1-zero-seed${SEED}-long-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    fractal_candidate=rotary-gated-recurrent-state-update \
    fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
    fractal_rgrp_integration_mode=mlp-sidecar \
    fractal_rgrp_layers=1 \
    fractal_rgrp_bottleneck_dim=64 \
    fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
    fractal_rgrp_scan_unroll=3 \
    fractal_rgrp_projection_mode=sequence \
    fractal_rgrp_trig_mode=precompute \
    fractal_rgrp_side_scale=0.1 \
    fractal_rgrp_output_init=zero \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

run_attention
run_zero_sidecar

ls -lh "${OUTDIR}"
