#!/usr/bin/env bash
# Run matched TPU/JAX sidecar controls against the layer-1 zero-init contract.
#
# Execute this inside a patched MaxText checkout. The run matrix deliberately
# keeps model/data/optimizer/init/placement fixed and changes only
# fractal_rgrp_sidecar_type.

set -euo pipefail

STAMP="${1:-$(date -u +%Y%m%dT%H%MZ)}"
OUTDIR="${2:-$HOME/fractal_sidecar_control_rung_${STAMP}}"
SEEDS="${SEEDS:-0 1 2}"
LANES="${LANES:-rgrp tiny-mlp tiny-glu binary-tree}"
STEPS="${STEPS:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250}"
EVAL_STEPS="${EVAL_STEPS:-10}"
RUN_ATTENTION="${RUN_ATTENTION:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-0}"
mkdir -p "${OUTDIR}"

if [[ ! -d ".venv" ]]; then
  echo "expected to run from MaxText checkout with .venv present" >&2
  exit 2
fi

PYTHON="${PYTHON:-.venv/bin/python3}"
SUMMARY_FILTER='/number parameters:|Memstats: After params initialized:|Using \\(GB\\)|eval metrics after step|completed step: (0|249|499|749|999),/ { print; fflush(); }'

COMMON=(
  base_output_directory=gs://fractal-maxtext-runs-81f2add4
  dataset_type=hf
  hf_path=roneneldan/TinyStories
  hf_eval_split=validation
  tokenizer_type=huggingface
  tokenizer_path=gpt2
  train_split=train
  log_period=50
  "eval_interval=${EVAL_INTERVAL}"
  "eval_steps=${EVAL_STEPS}"
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
  local run_name="attention-unscanned-seed${seed}-control-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${steps}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

run_sidecar() {
  local sidecar_type="$1"
  local seed="$2"
  local steps
  steps="$(steps_for_lane)"
  local safe_type="${sidecar_type//-/_}"
  local run_name="sidecar-${safe_type}-layer1-zero-seed${seed}-${STAMP}"
  "${PYTHON}" -m maxtext.trainers.pre_train.train \
    "${COMMON[@]}" \
    "steps=${steps}" \
    "run_name=${run_name}" \
    "data_shuffle_seed=${seed}" \
    "init_weights_seed=${seed}" \
    fractal_candidate=rotary-gated-recurrent-state-update \
    fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
    fractal_rgrp_integration_mode=mlp-sidecar \
    fractal_rgrp_layers=1 \
    fractal_rgrp_bottleneck_dim=64 \
    "fractal_rgrp_sidecar_type=${sidecar_type}" \
    fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
    fractal_rgrp_scan_unroll=3 \
    fractal_rgrp_projection_mode=sequence \
    fractal_rgrp_trig_mode=precompute \
    fractal_rgrp_side_scale=0.1 \
    fractal_rgrp_output_init=zero \
    fractal_rgrp_tree_depth=2 \
    fractal_rgrp_slot_count=4 \
    2>&1 | tee "${OUTDIR}/${run_name}.log" | awk "${SUMMARY_FILTER}"
}

for seed in ${SEEDS}; do
  if [[ "${RUN_ATTENTION}" == "1" ]]; then
    run_attention "${seed}"
  fi
  for lane in ${LANES}; do
    run_sidecar "${lane}" "${seed}"
  done
done

ls -lh "${OUTDIR}"
