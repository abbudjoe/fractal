# JAX/TPU Kernel Architecture Lane

## Purpose

This lane exists to test whether Fractal architecture candidates can earn a
quality or efficiency signal on TPUs without spending H100 money first.

It is not a replacement for the CUDA/Triton lane. CUDA runs remain the right
surface for head-to-head speed claims against PyTorch, Mamba, FLA, and Triton.
The JAX/TPU lane is a separate proof ladder:

- reproduce an unchanged MaxText baseline cheaply
- port Fractal candidates into a MaxText-compatible seam
- compare quality, tokens/sec, compile cost, and steady-state throughput under
  equal model/data/budget contracts
- promote only candidates that survive apples-to-apples TPU runs

## Source Motivation

MaxText's current documented launch surface uses:

```sh
python3 -m maxtext.trainers.pre_train.train \
  run_name=${YOUR_JOB_NAME} \
  base_output_directory=gs://<bucket> \
  dataset_type=synthetic \
  steps=10
```

The relevant MaxText control-plane knobs include command-line overrides for
`base_output_directory`, `run_name`, `dataset_type`, `steps`,
`per_device_batch_size`, `max_target_length`, model shape settings such as
`base_emb_dim`, `base_num_query_heads`, `base_num_kv_heads`, and parallelism
settings such as `ici_fsdp_parallelism`. Those are the first-class fields
represented in `python/jax_tpu`.

Primary references:

- MaxText first run docs: https://maxtext.readthedocs.io/en/latest/tutorials/first_run.html
- MaxText monitoring/log docs: https://maxtext.readthedocs.io/en/latest/guides/monitoring_and_debugging/understand_logs_and_metrics.html
- MaxText architecture overview: https://maxtext.readthedocs.io/en/latest/reference/architecture/architecture_overview.html

## Repository Surface

- `python/jax_tpu/contracts.py` owns typed benchmark, shape, dataset,
  parallelism, candidate, and kernel contracts.
- `python/jax_tpu/maxtext.py` renders MaxText commands from validated specs.
- `python/jax_tpu/cli.py` exposes a small command emitter for scout runs.
- `python/jax_tpu/adapters/` is the future home for Fractal-specific JAX
  adapter modules.
- `python/tests/test_jax_tpu.py` holds contract tests.

## Candidate Rules

Candidates are split into two classes.

### MaxText-native candidates

These can run against unchanged MaxText. The initial candidate is:

- `attention-baseline`

Use this for setup, cost, data ingress, compile behavior, metric parsing, and
steady-state throughput.

### Patched-MaxText candidates

These require a MaxText fork or local patch that imports a Fractal adapter. The
initial registered candidate is:

- `rotary-gated-recurrent-state-update`

The first implementation should be a JAX reference block using `jax.lax.scan`
inside the FFN-side seam. Pallas or custom kernels should wait until this
reference port proves correctness and a real loss/speed signal.

## Example Commands

Emit an unchanged MaxText scout command:

```sh
python -m python.jax_tpu.cli \
  --candidate attention-baseline \
  --run-name fractal-attn-scout \
  --base-output-directory gs://fractal-maxtext-runs \
  --steps 10 \
  --seq-len 1024 \
  --d-model 512 \
  --layers 8 \
  --heads 8
```

Emit the future patched-P20 command once the MaxText adapter exists:

```sh
python -m python.jax_tpu.cli \
  --candidate rotary-gated-recurrent-state-update \
  --run-name fractal-p20-scout \
  --base-output-directory gs://fractal-maxtext-runs \
  --allow-patched-maxtext \
  --steps 10
```

Without `--allow-patched-maxtext`, patched candidates fail loudly. This is
intentional: it prevents us from accidentally claiming a custom architecture ran
when only an unchanged MaxText baseline did.

## First TPU Scout Result

Validated on 2026-04-18 with:

- project: `project-81f2add4-9e80-4335-bb6`
- bucket: `gs://fractal-maxtext-runs-81f2add4`
- TPU VM: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, editable install with
  `install_tpu_pre_train_extra_deps`

The TPU VM image had Python 3.10, while current MaxText requires Python 3.12.
The working setup used `uv python install 3.12` and a Python 3.12 venv.

The TPU default compute service account also needed bucket write access:

```sh
gcloud storage buckets add-iam-policy-binding gs://fractal-maxtext-runs-81f2add4 \
  --member=serviceAccount:727124170067-compute@developer.gserviceaccount.com \
  --role=roles/storage.objectAdmin
```

The successful synthetic scout used the live MaxText shape keys:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_emb_dim=512 \
  base_mlp_dim=2048 \
  base_num_decoder_layers=8 \
  base_num_query_heads=8 \
  base_num_kv_heads=8 \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  dataset_type=synthetic \
  dtype=bfloat16 \
  enable_checkpointing=false \
  learning_rate=0.001 \
  max_target_length=1024 \
  per_device_batch_size=1 \
  run_name=fractal-attn-scout-rerun \
  steps=10 \
  train_split=train \
  log_period=1
```

Result:

- JAX backend: TPU, one `TpuDevice`
- model size reported by MaxText: `0.075` billion parameters
- memory after parameter initialization: `0.85 / 15.75 GB`
- step 0: `785.647 tokens/s/device`, `0.314 TFLOP/s/device`
- final logged step 9: `15196.486 tokens/s/device`, `6.083 TFLOP/s/device`,
  synthetic loss `0.079`

This proves the TPU/MaxText baseline lane is operational. It does not yet prove
anything about Fractal recurrent candidates.

## First Proof Ladder

1. Run `attention-baseline` on synthetic data for 10 steps to verify install,
   TPU visibility, GCS output, logs, compile time, and command shape.
2. Run `attention-baseline` on the intended token/text input path with a tiny
   step count to verify data ingress.
3. Add the `rotary-gated-recurrent-state-update` JAX adapter under
   `python/jax_tpu/adapters/` and a matching MaxText patch.
4. Run baseline and P20 with equal shape, sequence length, batch, optimizer,
   data, and steps.
5. Report compile-inclusive throughput separately from steady-state throughput.
6. Only after correctness and signal are visible, test a Pallas/custom lowering.

## Non-Goals

- Do not use this lane to make CUDA speed claims.
- Do not port every Fractal primitive at once.
- Do not write Pallas kernels before the `jax.lax.scan` reference proves the
  architecture deserves the effort.
- Do not silently pass custom adapter flags to unchanged MaxText.
