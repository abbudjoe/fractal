#!/usr/bin/env bash
# Run a bounded MaxText RGRP MLP-sidecar ablation grid on a TPU VM.
#
# This script is intended to execute inside a patched MaxText checkout. Keeping
# the grid in a real bash script avoids losing key=value overrides through
# remote-shell quoting.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_sidecar_ablate_${STAMP}}"
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
  steps=200
  log_period=10
  eval_interval=100
  eval_steps=5
  enable_checkpointing=false
  save_checkpoint_on_completion=false
  log_config=false
  decoder_block=default
  scan_layers=false
  data_shuffle_seed=0
  init_weights_seed=0
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
  local label="$1"
  local run_name="attention-unscanned-${label}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

run_sidecar() {
  local label="$1"
  local layers="$2"
  local side_scale="$3"
  local bottleneck="$4"
  local output_init="$5"
  local run_name="rgrp-sidecar-${label}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    fractal_candidate=rotary-gated-recurrent-state-update \
    fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
    fractal_rgrp_integration_mode=mlp-sidecar \
    "fractal_rgrp_layers=${layers}" \
    "fractal_rgrp_bottleneck_dim=${bottleneck}" \
    fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
    fractal_rgrp_scan_unroll=3 \
    fractal_rgrp_projection_mode=sequence \
    fractal_rgrp_trig_mode=precompute \
    "fractal_rgrp_side_scale=${side_scale}" \
    "fractal_rgrp_output_init=${output_init}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

run_attention seed0-control

run_sidecar layer1-default 1 0.1 64 xavier

run_sidecar layer0-default 0 0.1 64 xavier
run_sidecar layer2-default 2 0.1 64 xavier
run_sidecar layer3-default 3 0.1 64 xavier

run_sidecar layer1-scale005 1 0.05 64 xavier
run_sidecar layer1-scale020 1 0.2 64 xavier

run_sidecar layer1-bn32 1 0.1 32 xavier
run_sidecar layer1-bn128 1 0.1 128 xavier

run_sidecar layer1-zero 1 0.1 64 zero

ls -lh "${OUTDIR}"
