#!/usr/bin/env bash
# Run matched MaxText seed replication for the layer-1 RGRP MLP-sidecar zero
# output-init contract on a TPU VM.
#
# Execute this inside a patched MaxText checkout. Keeping the run matrix in a
# TPU-side script avoids fragile nested SSH quoting for MaxText key=value
# overrides.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_sidecar_zero_seeds_${STAMP}}"
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
  local seed="$1"
  local run_name="attention-unscanned-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

run_sidecar() {
  local label="$1"
  local seed="$2"
  local output_init="$3"
  local run_name="rgrp-sidecar-${label}-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
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
    "fractal_rgrp_output_init=${output_init}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log"
}

for seed in 0 1 2; do
  run_attention "${seed}"
  run_sidecar layer1-default "${seed}" xavier
  run_sidecar layer1-zero "${seed}" zero
done

ls -lh "${OUTDIR}"
