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
- `python/jax_tpu/lm_smoke.py` owns a tiny repo-local LM-shaped
  forward/backward gate for transformer MLP vs RGRP FFN seams.
- `scripts/jax_tpu_lm_smoke.py` runs that LM gate on local JAX installs or TPU
  VMs.
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

Emit the patched RGRP command once the MaxText adapter exists:

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

Emit a tiny HF real-data ingress smoke command:

```sh
python -m python.jax_tpu.cli \
  --candidate attention-baseline \
  --run-name fractal-hf-ingress-tinystories \
  --base-output-directory gs://fractal-maxtext-runs-81f2add4 \
  --dataset-type hf \
  --hf-path roneneldan/TinyStories \
  --tokenizer-type huggingface \
  --tokenizer-path gpt2 \
  --vocab-size 50257 \
  --steps 5 \
  --seq-len 256 \
  --d-model 256 \
  --layers 4 \
  --heads 4
```

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

## First Real-Data Ingress Result

Validated on 2026-04-18 with the same `v5litepod-1` spot TPU/runtime and bucket.

Dataset choice:

- `dataset_type=hf`
- `hf_path=roneneldan/TinyStories`
- `train_split=train`
- `tokenizer_type=huggingface`
- `tokenizer_path=gpt2`
- `vocab_size=50257`

This run intentionally used public text data instead of
`joebud/fractal-fineweb-openllama-tokens`. The Fractal token cache is currently
stored as compressed archive artifacts, not Hub-native parquet/arrow rows, so
MaxText cannot stream it as an HF dataset without a conversion step.

The successful ingress smoke used:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_emb_dim=256 \
  base_mlp_dim=1024 \
  base_num_decoder_layers=4 \
  base_num_query_heads=4 \
  base_num_kv_heads=4 \
  vocab_size=50257 \
  tokenizer_type=huggingface \
  tokenizer_path=gpt2 \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  dataset_type=hf \
  hf_path=roneneldan/TinyStories \
  train_split=train \
  dtype=bfloat16 \
  enable_checkpointing=false \
  learning_rate=0.001 \
  max_target_length=256 \
  per_device_batch_size=1 \
  run_name=fractal-hf-ingress-tinystories \
  steps=5 \
  log_period=1
```

Result:

- MaxText streamed HF data and downloaded the GPT-2 tokenizer successfully.
- MaxText used packed variable-length examples; `total_weights` varied by step.
- model size reported by MaxText: `0.031` billion parameters
- memory after parameter initialization: `0.36 / 15.75 GB`
- step 0: `17.349 tokens/s/device`, loss `11.221`
- final logged step 4: `4192.803 tokens/s/device`, `0.469 TFLOP/s/device`,
  loss `9.890`

This proves the TPU lane can ingest real text through MaxText. It is not a
quality result and should not be compared against the synthetic scout.

## First Proof Ladder

1. Run `attention-baseline` on synthetic data for 10 steps to verify install,
   TPU visibility, GCS output, logs, compile time, and command shape.
2. Run `attention-baseline` on a real token/text input path with a tiny step
   count to verify data ingress.
3. Convert the Fractal HF token archives into a MaxText-native parquet or Grain
   dataset before using them as the recurring proof-ladder data source.
4. Add the `rotary-gated-recurrent-state-update` JAX adapter under
   `python/jax_tpu/adapters/` and a matching MaxText patch.
5. Run baseline and RGRP with equal shape, sequence length, batch, optimizer,
   data, and steps.
6. Report compile-inclusive throughput separately from steady-state throughput.
7. Only after correctness and signal are visible, test a Pallas/custom lowering.

## First RGRP Adapter TPU Smoke

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-smoke-20260419002026`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- JAX install: `jax[tpu]` from the official libtpu wheel index
- JAX version: `0.6.2`
- libtpu: `0.0.17`

This smoke did not use MaxText and did not train a full language model. It
tested the new JAX reference adapter directly:

```sh
python3 scripts/jax_tpu_rgrp_smoke.py \
  --batch-size 64 \
  --seq-len 256 \
  --d-model 512 \
  --state-transform block-diagonal-4 \
  --iterations 5 \
  --warmup 2
```

The adapter executed on `backend=tpu` with one `TPU_0` device.

| Surface | State transform | Compile seconds | Steady-state tok/s | Notes |
|---|---:|---:|---:|---|
| forward only | block-diagonal-4 | `1.275` | `4,308,400` | isolated scan, no LM shell |
| forward + grad | block-diagonal-4 | `4.072` | `2,888,680` | dummy loss over emitted outputs |
| forward + grad | dense | `4.113` | `2,962,100` | dense was slightly faster in this isolated TPU smoke |

Interpretation:

- The rotary gated recurrent state update primitive now has a real JAX
  `lax.scan` implementation that compiles and runs on TPU.
- The TPU path should not automatically inherit the CUDA/Triton fast-lane
  assumption that block-diagonal state transforms are best; dense state
  transform was slightly faster in this isolated smoke.
- This is a kernel/adapter smoke only. The next gate is a repo-owned LM-shaped
  integration smoke before patching the MaxText FFN-side seam.

The TPU VM was deleted after the smoke; `gcloud compute tpus tpu-vm list` was
empty afterward.

## RGRP TPU State-Transform Definition Ablation

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-ablate-20260419003810`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- JAX version: `0.6.2`
- libtpu: `0.0.17`
- shared shape: `batch_size=64`, `seq_len=256`, `d_model=512`, `bf16`
- path: forward + gradient over the isolated adapter dummy loss
- iterations: `8`
- warmup iterations: `3`

Command pattern:

```sh
python3 scripts/jax_tpu_rgrp_smoke.py \
  --batch-size 64 \
  --seq-len 256 \
  --d-model 512 \
  --state-transform <mode> \
  --iterations 8 \
  --warmup 3
```

| State transform definition | Compile seconds | Steady-state tok/s |
|---|---:|---:|
| `dense` | `4.415` | `2,963,451` |
| `block-diagonal-4` | `4.404` | `2,953,874` |
| `block-diagonal-4-masked-dense` | `4.545` | `2,968,600` |

Interpretation:

- The three definitions are effectively tied at this isolated adapter scale.
- The initial dense-over-block result should not be interpreted as an
  architectural finding.
- `block-diagonal-4-masked-dense` slightly led grouped `block-diagonal-4`,
  suggesting some of the difference is layout/lowering rather than the recurrent
  rule itself.
- TPU experiments should treat the state-transform implementation as a backend
  policy knob. CUDA/Triton may prefer explicit block-diagonal kernels, while
  TPU/XLA may prefer dense-shaped lowering even when the represented math is
  structured.

The TPU VM was deleted after the ablation; `gcloud compute tpus tpu-vm list` was
empty afterward.

## First RGRP LM-Shaped TPU Smoke

Validated on 2026-04-19 with:

- TPU VM: `fractal-lm-smoke-20260419010655`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- JAX version: `0.6.2`
- libtpu: `0.0.17`
- dtype: `bf16`
- path: tiny repo-owned transformer shell with causal attention, LM head, and
  forward + gradient over synthetic random next-token loss
- compared seams: standard MLP FFN vs rotary gated recurrent state update
  primitive FFN seam

This smoke is intentionally not MaxText and is not a language-model quality
result. It exists to test whether the RGRP adapter can participate in an
LM-shaped residual block before we spend effort patching MaxText.

Command pattern:

```sh
python3 scripts/jax_tpu_lm_smoke.py \
  --variant <mlp|rgrp> \
  --rgrp-state-transform block-diagonal-4-masked-dense \
  --vocab-size <vocab> \
  --batch-size <batch> \
  --seq-len <seq> \
  --d-model <width> \
  --layers 2 \
  --heads <heads> \
  --iterations <iters> \
  --warmup 1
```

Results:

| Shape | Seam | Params | Compile seconds | Steady-state tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|
| `vocab=4096,batch=8,seq=128,d=256,layers=2,heads=4` | MLP | `3.71M` | `3.501` | `155,619` | `8.3702` |
| `vocab=4096,batch=8,seq=128,d=256,layers=2,heads=4` | RGRP | `3.25M` | `4.006` | `132,108` | `8.3664` |
| `vocab=8192,batch=8,seq=256,d=256,layers=2,heads=4` | MLP | `5.84M` | `3.634` | `318,887` | `9.0579` |
| `vocab=8192,batch=8,seq=256,d=256,layers=2,heads=4` | RGRP | `5.38M` | `4.303` | `286,580` | `9.0701` |
| `vocab=8192,batch=4,seq=256,d=512,layers=2,heads=8` | MLP | `14.82M` | `3.980` | `156,651` | `9.1238` |
| `vocab=8192,batch=4,seq=256,d=512,layers=2,heads=8` | RGRP | `12.99M` | `4.554` | `117,765` | `9.1116` |

Interpretation:

- The RGRP seam is now integration-correct on TPU inside a complete toy
  LM-shaped forward/backward path.
- RGRP is consistently parameter-lighter than the matched MLP seam in this
  shell.
- RGRP is not yet throughput-competitive with the MLP seam in this simple XLA
  lowering. The gap was roughly `15%` at the smallest shape, `10%` at
  `seq=256,d=256`, and `25%` at `d=512`.
- The synthetic random-token losses are not quality evidence; they only prove
  numerical execution.
- Before a serious TPU quality run, the next engineering question is whether the
  MaxText integration can express this seam without adding extra layout or scan
  overhead beyond what this toy path shows.

The TPU VM was deleted after the smoke; `gcloud compute tpus tpu-vm list` was
empty afterward.

## First RGRP JAX/XLA Lowering Ladder

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-ladder-20260419014935`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- shape: `vocab=8192,batch=4,seq=256,d_model=512,layers=2,heads=8`
- path: tiny repo-owned LM-shaped forward + gradient smoke
- artifacts:
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0150Z/rgrp_lowering_ladder.jsonl`
  and
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0150Z/rgrp_lowering_ladder.md`

This ladder tested one compiler-sensitive axis at a time: state-transform
storage, trig placement, input projection placement, and `jax.lax.scan` unroll.

| Case | Seam | State | Projection | Trig | Unroll | Params | Compile seconds | Steady-state tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp-control` | MLP |  |  |  |  | `14.82M` | `3.78` | `168,831` | `9.1238` |
| `rgrp-default` | RGRP | `block-diagonal-4-masked-dense` | `sequence` | `precompute` | `1` | `12.99M` | `4.31` | `128,327` | `9.1116` |
| `state-dense` | RGRP | `dense` | `sequence` | `precompute` | `1` | `12.99M` | `4.43` | `124,389` | `9.1245` |
| `state-block4-grouped` | RGRP | `block-diagonal-4` | `sequence` | `precompute` | `1` | `12.59M` | `4.45` | `86,666` | `9.1171` |
| `trig-inside-scan` | RGRP | `block-diagonal-4-masked-dense` | `sequence` | `scan` | `1` | `12.99M` | `4.26` | `109,420` | `9.1117` |
| `projection-inside-scan` | RGRP | `block-diagonal-4-masked-dense` | `scan` | `scan` | `1` | `12.99M` | `4.24` | `108,438` | `9.1116` |
| `unroll-2` | RGRP | `block-diagonal-4-masked-dense` | `sequence` | `precompute` | `2` | `12.99M` | `4.77` | `121,109` | `9.1116` |
| `unroll-4` | RGRP | `block-diagonal-4-masked-dense` | `sequence` | `precompute` | `4` | `12.99M` | `5.37` | `142,740` | `9.1116` |
| `unroll-8` | RGRP | `block-diagonal-4-masked-dense` | `sequence` | `precompute` | `8` | `12.99M` | `7.14` | `129,227` | `9.1117` |

Interpretation:

- The best first-pass RGRP lowering was `block-diagonal-4-masked-dense` with
  sequence-wide projection, sequence-wide trig precompute, and `scan_unroll=4`.
- `scan_unroll=4` improved RGRP throughput by about `11%` over the default
  `scan_unroll=1`, at the cost of a longer compile.
- The best RGRP variant still trailed the MLP control by about `15%` steady-state
  throughput at this toy LM shape.
- Explicit grouped block-diagonal storage lowered poorly on TPU, despite having
  fewer parameters. Treat this as a backend lowering result, not as evidence
  against structured state transforms in CUDA/Triton.
- Moving trig or the packed projection inside the scan body hurt throughput.
  TPU/XLA preferred the sequence-wide packed projection/control precompute.

Next ladder:

- Keep `block-diagonal-4-masked-dense`, sequence projection, trig precompute.
- Re-test `scan_unroll in {3,4,5,6}` and add sequence length scaling
  `seq in {256,512,1024}` before MaxText patching.
- Do not use grouped block-diagonal storage for TPU unless a Pallas kernel owns
  that layout explicitly.

The TPU VM was deleted after the ladder; `gcloud compute tpus tpu-vm list` was
empty afterward.

## RGRP Scan-Unroll and Sequence Scaling Ladder

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-next-20260419020412`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- shape base: `vocab=8192,batch=4,seq=256,d_model=512,layers=2,heads=8`
- path: tiny repo-owned LM-shaped forward + gradient smoke
- stable timing: `warmup=3`, `iterations=30`
- artifacts:
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0204Z/rgrp_next_ladder_stable.jsonl`
  and
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0204Z/rgrp_next_ladder_stable.md`

| Case | Seam | Seq | Unroll | Params | Compile seconds | Steady-state tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp-control` | MLP | `256` |  | `14.82M` | `3.74` | `143,927` | `9.1238` |
| `unroll-1` | RGRP | `256` | `1` | `12.99M` | `4.34` | `109,946` | `9.1116` |
| `unroll-3` | RGRP | `256` | `3` | `12.99M` | `5.60` | `136,460` | `9.1116` |
| `unroll-4` | RGRP | `256` | `4` | `12.99M` | `5.50` | `129,154` | `9.1116` |
| `unroll-5` | RGRP | `256` | `5` | `12.99M` | `6.35` | `126,510` | `9.1116` |
| `unroll-6` | RGRP | `256` | `6` | `12.99M` | `8.70` | `135,402` | `9.1116` |
| `seq512-mlp` | MLP | `512` |  | `14.95M` | `5.61` | `297,222` | `9.1168` |
| `seq512-rgrp-unroll4` | RGRP | `512` | `4` | `13.12M` | `6.96` | `163,925` | `9.1121` |
| `seq1024-mlp` | MLP | `1024` |  | `15.22M` | `5.88` | `427,136` | `9.1092` |
| `seq1024-rgrp-unroll4` | RGRP | `1024` | `4` | `13.38M` | `8.27` | `183,856` | `9.1110` |

Interpretation:

- At `seq=256`, `scan_unroll=3` is the best stable RGRP setting in this ladder.
- `scan_unroll=3` improved throughput by about `24%` over `scan_unroll=1`.
- At `seq=256`, RGRP with `unroll=3` trailed the matched MLP control by only
  about `5%`, while using about `1.84M` fewer parameters in this toy shell.
- The initial sequence scaling rows used `unroll=4` from the previous ladder,
  so a focused long-sequence `unroll=3` vs `unroll=4` follow-up was required
  before making a MaxText recommendation.

The TPU VM was deleted after the ladder; `gcloud compute tpus tpu-vm list` was
empty afterward.

## RGRP Long-Sequence Unroll Follow-Up

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-sequnroll-20260419021521`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- shape: `vocab=8192,batch=4,d_model=512,layers=2,heads=8`
- path: tiny repo-owned LM-shaped forward + gradient smoke
- timing: `warmup=3`, `iterations=30`
- artifacts:
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0215Z/rgrp_sequence_unroll_ladder.jsonl`
  and
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0215Z/rgrp_sequence_unroll_ladder.md`

| Case | Seam | Seq | Unroll | Params | Compile seconds | Steady-state tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seq512-mlp` | MLP | `512` |  | `14.95M` | `5.62` | `338,918` | `9.1168` |
| `seq512-rgrp-unroll3` | RGRP | `512` | `3` | `13.12M` | `8.62` | `187,648` | `9.1121` |
| `seq512-rgrp-unroll4` | RGRP | `512` | `4` | `13.12M` | `6.95` | `167,513` | `9.1121` |
| `seq1024-mlp` | MLP | `1024` |  | `15.22M` | `5.98` | `426,400` | `9.1092` |
| `seq1024-rgrp-unroll3` | RGRP | `1024` | `3` | `13.38M` | `8.24` | `201,841` | `9.1110` |
| `seq1024-rgrp-unroll4` | RGRP | `1024` | `4` | `13.38M` | `8.17` | `188,018` | `9.1110` |

Interpretation:

- `scan_unroll=3` remains the best current RGRP TPU setting at longer sequence
  lengths.
- `unroll=3` beat `unroll=4` by about `12%` at `seq=512` and about `7%` at
  `seq=1024`.
- The current reference `lax.scan` RGRP lowering still scales worse than the MLP
  control as sequence length increases. At `seq=512`, RGRP reached about `55%`
  of MLP throughput; at `seq=1024`, it reached about `47%`.
- This does not resolve architecture quality. It says that, for TPU speed,
  plain `lax.scan` is not enough. A serious TPU path needs either a MaxText
  quality-only run that accepts the speed tax, or a Pallas/custom recurrent
  lowering before making efficiency claims.

Current TPU policy:

- Use `block-diagonal-4-masked-dense`, sequence projection, trig precompute, and
  `scan_unroll=3` for any near-term JAX/TPU RGRP reference run.
- Keep `scan_unroll` explicit in configs. It is a backend policy knob, not a
  mathematical part of the primitive.
- Do not claim TPU throughput competitiveness from this reference lowering.

The TPU VM was deleted after the follow-up; `gcloud compute tpus tpu-vm list`
was empty afterward.

## RGRP Pallas Forward-Kernel Smoke

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-pallas-20260419024425`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- shape: `vocab=8192,batch=4,d_model=512,layers=2,heads=8`
- path: tiny repo-owned LM-shaped forward-only smoke
- timing: `warmup=3`, `iterations=30`
- artifacts:
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0244Z/rgrp_pallas_forward_ladder.jsonl`
  and
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0244Z/rgrp_pallas_forward_ladder.md`

Implementation notes:

- The Pallas kernel is forward-only and intentionally does not claim training
  readiness. A training path would need a backward rule/custom VJP unless a
  future Pallas lowering gives us acceptable automatic differentiation.
- The first rank-1 `state @ W` lowering failed in Mosaic, so the kernel uses a
  rank-2 matmul form with fp32 accumulation.
- Mosaic also rejected pair reshape / strided gather patterns, so the kernel is
  pair-native: even/odd rotary lanes and the state-transform matrix quadrants
  are split before entering Pallas, then re-interleaved outside the kernel.
- The kernel keeps the TPU-favorable controls from prior ladders: sequence-wide
  packed projection and trig precompute.

| Case | Seam | Seq | Mode | Params | Compile seconds | Forward tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `seq256-mlp` | MLP | `256` | XLA | `14.82M` | `1.75` | `3,140,937` | `9.1238` |
| `seq256-rgrp-scan-unroll3` | RGRP | `256` | `lax.scan` | `12.99M` | `1.51` | `751,699` | `9.1115` |
| `seq256-rgrp-pallas-forward` | RGRP | `256` | Pallas | `12.99M` | `13.48` | `160,315` | `9.1116` |
| `seq512-mlp` | MLP | `512` | XLA | `14.95M` | `1.45` | `3,745,490` | `9.1168` |
| `seq512-rgrp-scan-unroll3` | RGRP | `512` | `lax.scan` | `13.12M` | `3.20` | `772,462` | `9.1120` |
| `seq512-rgrp-pallas-forward` | RGRP | `512` | Pallas | `13.12M` | `30.10` | `290,698` | `9.1120` |
| `seq1024-mlp` | MLP | `1024` | XLA | `15.22M` | `1.74` | `2,320,681` | `9.1092` |
| `seq1024-rgrp-scan-unroll3` | RGRP | `1024` | `lax.scan` | `13.38M` | `3.02` | `688,023` | `9.1109` |
| `seq1024-rgrp-pallas-forward` | RGRP | `1024` | Pallas | `13.38M` | `87.55` | `448,688` | `9.1110` |

Interpretation:

- The pair-native Pallas forward kernel compiles and runs, but it is not a win.
- Pallas forward was slower than `lax.scan` at every tested sequence length.
- The gap narrowed with longer sequences, but not enough to justify a backward
  implementation yet: Pallas reached about `21%` of scan throughput at
  `seq=256`, `38%` at `seq=512`, and `65%` at `seq=1024`.
- Compile time is also much worse for Pallas, especially at `seq=1024`.
- The likely root cause is kernel shape: one Pallas program per batch element
  hides a long recurrent loop and repeated vector-matrix work inside a single
  program. That does not expose enough tiled matrix work to TPU/Mosaic.

Current decision:

- Do not promote this Pallas forward kernel.
- Keep the code as an explicit experimental lane because it documents the
  Mosaic constraints and gives us a runnable baseline for future Pallas designs.
- A chunked/tiled follow-up was run below. It improved the Pallas lane but did
  not overtake `lax.scan`.
- Otherwise, for near-term TPU quality runs, use the `lax.scan` reference with
  `scan_unroll=3`.

The TPU VM was deleted after the Pallas smoke; `gcloud compute tpus tpu-vm list`
was empty afterward.

## RGRP Chunked Block-Tiled Pallas Smoke

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-tiled-20260419031707`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- shape: `vocab=8192,batch=4,d_model=512,layers=2,heads=8`
- path: tiny repo-owned LM-shaped forward-only smoke
- timing: `warmup=3`, `iterations=30`
- artifacts:
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0317Z/rgrp_pallas_block_tiled_ladder.jsonl`
  and
  `experiments/jax_tpu/rgrp_lowering_ladder/20260419T0317Z/rgrp_pallas_block_tiled_ladder.md`

Implementation notes:

- This adds `execution_mode=pallas-block-tiled-forward`.
- The mode is forward-only and does not claim training readiness.
- It only targets masked block-diagonal state transforms. Dense recurrent
  transforms cannot be split this way without cross-tile reductions.
- The kernel uses sequence-wide packed projection and trig precompute, then
  scans fixed-size sequence chunks through Pallas.
- TPU Pallas rejected one-logical-block tiles because the innermost block
  dimension was too small. The working implementation groups logical recurrent
  blocks into `128`-wide rotary-pair tiles, or uses the full width when the
  pair dimension is smaller than `128`.
- TPU Pallas also rejected a batch-row block shape of `1` for `batch=4`; the
  working implementation gives each program the local batch and one recurrent
  state tile.
- A vectorized `batch x 128 @ 128 x 128` carry matmul crashed Mosaic. The
  working implementation keeps the full-batch block spec but computes each
  batch row with rank-2 vector-matrix matmuls inside the Pallas program.
- The recurrent carry between chunks is fp32. Emitted sequence outputs remain
  in the configured input dtype.

| Case | Seam | Seq | Mode | Chunk | Params | Compile seconds | Forward tok/s | Synthetic loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `seq256-mlp` | MLP | `256` | XLA | - | `14.82M` | `1.78` | `3,233,229` | `9.1238` |
| `seq256-rgrp-scan-unroll3` | RGRP | `256` | `lax.scan` | `256` | `12.99M` | `1.53` | `758,862` | `9.1115` |
| `seq256-rgrp-pallas-forward` | RGRP | `256` | Pallas full | `256` | `12.99M` | `13.68` | `160,729` | `9.1116` |
| `seq256-rgrp-pallas-block-tiled-c128` | RGRP | `256` | Pallas tiled | `128` | `12.99M` | `21.14` | `164,191` | `9.1116` |
| `seq512-mlp` | MLP | `512` | XLA | - | `14.95M` | `1.53` | `3,801,110` | `9.1168` |
| `seq512-rgrp-scan-unroll3` | RGRP | `512` | `lax.scan` | `256` | `13.12M` | `3.21` | `770,767` | `9.1120` |
| `seq512-rgrp-pallas-forward` | RGRP | `512` | Pallas full | `256` | `13.12M` | `30.54` | `290,730` | `9.1120` |
| `seq512-rgrp-pallas-block-tiled-c128` | RGRP | `512` | Pallas tiled | `128` | `13.12M` | `21.30` | `306,078` | `9.1120` |
| `seq1024-mlp` | MLP | `1024` | XLA | - | `15.22M` | `1.78` | `2,317,386` | `9.1092` |
| `seq1024-rgrp-scan-unroll3` | RGRP | `1024` | `lax.scan` | `256` | `13.38M` | `3.03` | `686,487` | `9.1109` |
| `seq1024-rgrp-pallas-forward` | RGRP | `1024` | Pallas full | `256` | `13.38M` | `87.50` | `448,113` | `9.1110` |
| `seq1024-rgrp-pallas-block-tiled-c128` | RGRP | `1024` | Pallas tiled | `128` | `13.38M` | `21.28` | `483,600` | `9.1110` |

Chunk-size probe:

| Case | Chunk | Compile seconds | Forward tok/s | Interpretation |
|---|---:|---:|---:|---|
| `seq512-rgrp-pallas-block-tiled-c256` | `256` | `43.41` | `302,351` | slower than `c128` and much slower compile |
| `seq1024-rgrp-pallas-block-tiled-c256` | `256` | `43.44` | `478,042` | slower than `c128` and much slower compile |

Interpretation:

- Chunked block-tiled Pallas is a real improvement over the first Pallas lane:
  `seq1024` improves from `448k` to `484k tok/s`, and compile drops from
  `87.5s` to `21.3s`.
- It still does not beat `lax.scan`: at `seq1024`, tiled Pallas reaches about
  `70%` of scan throughput.
- The best chunk tested was `128`. Larger `256` chunks reduced inter-chunk work
  but lost throughput and doubled compile time.
- The lowering is fragile. Two plausible kernel shapes hit Mosaic failures
  before the row-wise matmul fallback compiled.

Current decision:

- Keep `pallas-block-tiled-forward` as an experimental forward-only lane.
- Do not promote it to the MaxText quality run or invest in a backward path yet.
- For the next quality-only TPU run, use the `lax.scan` reference with
  `scan_unroll=3`.
- Revisit custom TPU lowering only if we are willing to design a lower-level
  kernel contract around TPU layout constraints from the start, rather than
  adapting this Pallas prototype incrementally.

The TPU VM was deleted after the chunked/tiled smoke; `gcloud compute tpus
tpu-vm list` was empty afterward.

## First MaxText RGRP Quality-Only Run

Validated on 2026-04-19 with:

- TPU VM: `fractal-rgrp-mt-0419`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, locally patched with
  `scripts/patch_maxtext_rgrp.py`
- path: MaxText language-model training on real HF text data
- dataset: `roneneldan/TinyStories`
- tokenizer: GPT-2 Hugging Face tokenizer
- model shape: `base_emb_dim=256`, `base_mlp_dim=1024`,
  `base_num_decoder_layers=4`, `base_num_query_heads=4`,
  `base_num_kv_heads=4`, `head_dim=64`
- sequence/batch: `max_target_length=512`, `per_device_batch_size=4`
- training budget: `steps=200`, `eval_interval=100`, `eval_steps=5`
- recurrent lowering: `jax.lax.scan`, `scan_unroll=3`,
  `block-diagonal-4-masked-dense`, sequence-wide packed projection, trig
  precompute
- raw log:
  `experiments/jax_tpu/maxtext_quality/rgrp_maxtext_quality_laxscan_20260419T0418Z.log`
- summary:
  `experiments/jax_tpu/maxtext_quality/20260419T0418Z_rgrp_laxscan_quality.md`

The MaxText patch installs a Fractal-native FFN-side seam. It adds explicit
config fields for `fractal_candidate`, `fractal_rgrp_state_transform`,
`fractal_rgrp_scan_unroll`, `fractal_rgrp_projection_mode`,
`fractal_rgrp_trig_mode`, and `fractal_rgrp_residual_scale`, then routes the
default decoder layer's FFN branch through the rotary gated recurrent state
update primitive only when `fractal_candidate` is set.

Command:

```sh
python3 -m maxtext.trainers.pre_train.train \
  base_output_directory=gs://fractal-maxtext-runs-81f2add4 \
  run_name=rgrp-maxtext-quality-laxscan-20260419T0418Z \
  dataset_type=hf \
  hf_path=roneneldan/TinyStories \
  hf_eval_split=validation \
  tokenizer_type=huggingface \
  tokenizer_path=gpt2 \
  train_split=train \
  steps=200 \
  log_period=10 \
  eval_interval=100 \
  eval_steps=5 \
  enable_checkpointing=false \
  save_checkpoint_on_completion=false \
  log_config=false \
  decoder_block=default \
  fractal_candidate=rotary-gated-recurrent-state-update \
  fractal_adapter_module=python.jax_tpu.adapters.rotary_gated_recurrent_state_update \
  fractal_rgrp_state_transform=block-diagonal-4-masked-dense \
  fractal_rgrp_scan_unroll=3 \
  fractal_rgrp_projection_mode=sequence \
  fractal_rgrp_trig_mode=precompute \
  fractal_rgrp_residual_scale=1.0 \
  max_target_length=512 \
  vocab_size=50257 \
  base_emb_dim=256 \
  base_mlp_dim=1024 \
  base_num_decoder_layers=4 \
  base_num_query_heads=4 \
  base_num_kv_heads=4 \
  head_dim=64 \
  per_device_batch_size=4 \
  learning_rate=0.001 \
  dtype=bfloat16
```

Result:

| Metric | Value |
|---|---:|
| Reported parameters | `0.028B` |
| TPU memory after params init | `0.33 / 15.75 GB` |
| Per-step total TFLOPs | `0.22` |
| Step 0 train loss | `11.355` |
| Eval after step 99 | loss `4.400`, perplexity `81.436` |
| Final eval after step 199 | loss `4.066`, perplexity `58.317` |
| Final train step 199 | loss `4.102`, perplexity `60.444` |
| Regular post-compile train throughput | median `105,763 tok/s/device` |

Interpretation:

- This is the first real MaxText quality run for the rotary gated recurrent
  state update primitive.
- The patched MaxText seam is live: the run used the RGRP `lax.scan` FFN-side
  branch, trained on real HF text, evaluated on validation data, and stayed
  numerically stable.
- This run is quality-only. It does not prove a win over attention because the
  matched MaxText attention baseline has not been run under the same shape,
  data, and step budget.
- The throughput number should be read as a MaxText reference-lowering datum,
  not a final TPU efficiency claim. Prior toy ladders already showed that plain
  `lax.scan` is adequate for quality smoke testing but not yet the right path
  for speed claims.
- The next low-risk rung is an unchanged MaxText attention baseline with the
  same `TinyStories`, tokenizer, shape, sequence length, batch, optimizer, and
  `200`-step budget. Only after that comparison should longer quality runs or
  kernel work resume.

The TPU VM was deleted after copying the log; no TPU was left running.

## Matched MaxText Attention Baseline

Validated on 2026-04-19 with:

- TPU VM: `fractal-attn-mt-0419`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, unpatched
- path: unchanged MaxText `decoder_block=default`
- dataset/tokenizer/shape/budget: matched to the RGRP quality-only run above
- raw log:
  `experiments/jax_tpu/maxtext_quality/attention_maxtext_baseline_20260419T0444Z.log`
- summary:
  `experiments/jax_tpu/maxtext_quality/20260419T0444Z_attention_baseline.md`
- scorecard:
  `experiments/jax_tpu/maxtext_quality/20260419T0418Z_0444Z_maxtext_scorecard.md`

Result:

| Lane | Params | Memory After Init | Eval Loss @99 | Final Eval Loss @199 | Final Eval PPL | Final Train Loss | Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention baseline | `0.030B` | `0.35 GB` | `4.030` | `3.758` | `42.874` | `3.817` | `375,987` |
| RGRP FFN seam | `0.028B` | `0.33 GB` | `4.400` | `4.066` | `58.317` | `4.102` | `105,763` |

Interpretation:

- The unchanged MaxText transformer baseline wins this matched short-run rung
  on both quality and speed.
- RGRP is slightly smaller and slightly lower-memory, but the advantage is not
  close to enough to offset `+0.308` worse final eval loss and a `3.55x`
  baseline throughput advantage.
- The RGRP MaxText integration remains valuable because it is now real,
  documented, and benchmarked against the necessary control. But the current
  plain `lax.scan` FFN replacement should not be promoted as competitive.
- The next TPU experiment should change the hypothesis, not just repeat the
  run: try a hybrid FFN+RGRP seam, residual/ramp controls, or a larger/longer
  rung where recurrent state has a plausible role.

The TPU VM was deleted after copying the log; no TPU was left running.

## Faithful RGRP MLP-Sidecar Probe

Validated on 2026-04-19 with:

- TPU VM: `fractal-sidecar-202604190512`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched raw logs:
  `experiments/jax_tpu/maxtext_quality/attention_unscanned_maxtext_20260419T0512Z.log`
  and
  `experiments/jax_tpu/maxtext_quality/rgrp_mlp_sidecar_maxtext_20260419T0522Z.log`
- scorecard:
  `experiments/jax_tpu/maxtext_quality/20260419T0512Z_0522Z_rgrp_mlp_sidecar_scorecard.md`

This run exists because the all-layer RGRP FFN replacement was not faithful to
the earlier sparse/mid-layer positive signal. The new lane keeps MaxText
attention and MLP intact, disables layer scanning only to expose static layer
identity, and adds one bottlenecked RGRP side branch beside the MLP in layer
`1`.

Result:

| Lane | Params | Memory After Init | Eval Loss @99 | Final Eval Loss @199 | Final Eval PPL | Final Train Loss | Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention control, unscanned | `0.030B` | `0.37 GB` | `4.019` | `3.756` | `42.795` | `3.839` | `327,549` |
| RGRP one-layer MLP sidecar | `0.030B` | `0.37 GB` | `4.026` | `3.754` | `42.696` | `3.839` | `303,925` |

Interpretation:

- The faithful sidecar contract is stable and trainable.
- It is essentially tied with the matched unscanned attention control, with a
  tiny final-eval edge and about `7%` lower median throughput.
- This does not prove superiority. It does show the sparse sidecar hypothesis
  remains alive, unlike the all-layer FFN replacement lane.
- The patcher now handles two MaxText control-plane contracts that the first
  attempt exposed: numeric CLI parsing for `fractal_rgrp_layers`, and static
  sidecar enablement at module construction rather than traced layer-index
  branching inside the Linen module.

The TPU VM was deleted after copying logs; no TPU was left running.

## RGRP MLP-Sidecar Seed Replication

Validated on 2026-04-19 with:

- TPU VM: `fractal-sidecar-seeds-202604190529`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/seed_replicates_20260419T0529Z/20260419T0529Z_rgrp_mlp_sidecar_seed_replicates.md`

This replicated the one-layer MLP-sidecar contract across three matched seeds.
Both lanes used `scan_layers=false`; seeds varied `data_shuffle_seed` and
`init_weights_seed` together.

| Lane | Mean Eval Loss @99 | Mean Final Eval Loss | Mean Final Eval PPL | Mean Final Train Loss | Mean Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|
| Attention control, unscanned | `4.0147` | `3.7510` | `42.5593` | `3.9123` | `334,377` |
| RGRP one-layer MLP sidecar | `4.0167` | `3.7503` | `42.5383` | `3.9117` | `303,671` |
| Sidecar minus control | `+0.0020` | `-0.0007` | `-0.0210` | `-0.0007` | `-30,706` |

Interpretation:

- The sparse sidecar signal replicated as a near-tie, with the sidecar winning
  final eval loss on `2 / 3` seeds.
- The mean final-eval edge was only about `-0.0007`, so this is not a proof of
  superiority.
- Throughput regressed consistently by about `9.2%` on median tok/s/device.
- The sidecar remains alive as a quality hypothesis, but not as a TPU efficiency
  claim under the current `lax.scan` reference lowering.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## RGRP MLP-Sidecar Location and Knob Ablation

Validated on 2026-04-19 with:

- TPU VM: `fractal-sidecar-ablate-202604190904`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_ablation_20260419T0904Z/20260419T0904Z_rgrp_sidecar_ablation_scorecard.md`

This ablation tested whether the faithful one-layer RGRP MLP sidecar depends on
layer location or small integration knobs. It used the same 200-step TinyStories
MaxText contract as the sidecar seed replication, with seed `0`.

| Lane | Sidecar Layer | Side Scale | Bottleneck | Output Init | Final Eval Loss | Delta vs Attention | Median Tok/s/Device |
|---|---:|---:|---:|---|---:|---:|---:|
| `layer1-zero` | `1` | `0.10` | `64` | `zero` | `3.751` | `-0.005` | `304,467` |
| `layer1-default` | `1` | `0.10` | `64` | `xavier` | `3.754` | `-0.002` | `303,925` |
| `layer1-bn128` | `1` | `0.10` | `128` | `xavier` | `3.755` | `-0.001` | `278,564` |
| `attention-control` | - | - | - | - | `3.756` | `0.000` | `331,044` |
| `layer1-scale020` | `1` | `0.20` | `64` | `xavier` | `3.756` | `0.000` | `304,649` |
| `layer0-default` | `0` | `0.10` | `64` | `xavier` | `3.757` | `+0.001` | `303,385` |
| `layer1-bn32` | `1` | `0.10` | `32` | `xavier` | `3.758` | `+0.002` | `339,101` |
| `layer3-default` | `3` | `0.10` | `64` | `xavier` | `3.758` | `+0.002` | `304,174` |
| `layer1-scale005` | `1` | `0.05` | `64` | `xavier` | `3.759` | `+0.003` | `303,318` |
| `layer2-default` | `2` | `0.10` | `64` | `xavier` | `3.759` | `+0.003` | `303,970` |

Interpretation:

- Layer `1` remains the best sidecar location under the default Xavier-init
  contract.
- Zero-initializing the sidecar output projection produced the best single-seed
  result, suggesting the recurrent sidecar should start as a dormant residual
  path and earn influence during training.
- Smaller bottleneck `32` recovered speed but lost quality. Larger bottleneck
  `128` did not beat the zero-init contract.
- Do not promote a longer run yet. Replicate `layer1-zero` across seeds first.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## RGRP Layer-1 Zero-Init Sidecar Seed Replication

Validated on 2026-04-19 with:

- TPU VM: `fractal-zero-seeds-202604191448`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_zero_seed_replicates_20260419T1448Z/20260419T1448Z_rgrp_sidecar_zero_seed_replicates.md`

This replicated the best single-seed sidecar ablation: layer `1`, bottleneck
`64`, side scale `0.1`, and zero-initialized sidecar output projection. It ran
matched seeds `0`, `1`, and `2` against both the unscanned attention control and
the layer `1` Xavier-init sidecar.

| Lane | Mean Eval Loss @99 | Mean Final Eval Loss | Eval Loss Std Dev | Mean Final Train Loss | Mean Median Tok/s/Device | Wins vs Attention |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `4.0147` | `3.7510` | `0.0051` | `3.9123` | `360,203` | - |
| `layer1-default` | `4.0167` | `3.7503` | `0.0052` | `3.9117` | `302,660` | `2 / 3` |
| `layer1-zero` | `4.0143` | `3.7493` | `0.0017` | `3.9113` | `303,079` | `2 / 3` |

Paired final-eval deltas:

| Comparison | Mean Delta | Per-Seed Deltas |
|---|---:|---|
| `layer1-default - attention` | `-0.0007` | `[-0.002, +0.001, -0.001]` |
| `layer1-zero - attention` | `-0.0017` | `[-0.005, -0.003, +0.003]` |
| `layer1-zero - layer1-default` | `-0.0010` | `[-0.003, -0.004, +0.004]` |

Interpretation:

- The zero-init sidecar replicated directionally, but the effect remains small.
- Zero-init is now the preferred faithful sidecar contract because it starts as
  a dormant residual path instead of perturbing the MLP stream at step zero.
- The TPU `lax.scan` reference sidecar remains slower: about `303k` mean median
  tok/s/device versus about `360k` for attention in this run.
- Promote `layer1-zero` only as the current quality-reference sidecar contract,
  not as a speed or decisive architecture win.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## RGRP Layer-1 Zero-Init Sidecar Long Quality Run

Validated on 2026-04-19 with:

- TPU VM: `fractal-zero-long-202604191651`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_zero_long_20260419T1651Z/20260419T1651Z_rgrp_sidecar_zero_long_quality.md`

This extended the best sidecar contract from `200` to `1000` training steps on
seed `0`, with `eval_interval=250` and `eval_steps=10`.

| Eval Step | Attention Eval Loss | Layer1-Zero Eval Loss | Delta |
|---:|---:|---:|---:|
| `249` | `3.529` | `3.521` | `-0.008` |
| `499` | `3.253` | `3.225` | `-0.028` |
| `749` | `3.046` | `3.019` | `-0.027` |
| `999` | `2.952` | `2.921` | `-0.031` |

| Lane | Final Train Loss | Median Tok/s/Device | Mean Tok/s/Device | Final Eval PPL |
|---|---:|---:|---:|---:|
| `attention` | `3.255` | `343,797` | `342,806` | `19.151` |
| `layer1-zero` | `3.247` | `301,531` | `302,822` | `18.553` |

Interpretation:

- The zero-init sidecar edge widened under longer training, from the tiny
  200-step signal to a `-0.031` final eval-loss improvement at step `999`.
- The sidecar won every evaluation checkpoint in this run.
- The TPU reference-lowering speed cost remains material: about `12.3%` lower
  median tok/s/device.
- Promote `layer1-zero` to "quality-promising under longer training," but do
  not make robustness claims until the 1000-step rung is replicated across
  seeds.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## RGRP Layer-1 Zero-Init Sidecar 1000-Step Seed Replication

Validated on 2026-04-19 with:

- seed `0` source artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_zero_long_20260419T1651Z/20260419T1651Z_rgrp_sidecar_zero_long_quality.md`
- seeds `1` and `2` TPU VM: `fractal-zero-long-seeds-202604191834`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_zero_long_seed_replicates_20260419T1834Z/20260419T1834Z_rgrp_sidecar_zero_long_seed_replicates.md`

This replicated the `1000`-step layer `1` zero-init sidecar rung across seeds
`0`, `1`, and `2`.

| Seed | Attention Final Eval | Layer1-Zero Final Eval | Delta |
|---:|---:|---:|---:|
| `0` | `2.952` | `2.921` | `-0.031` |
| `1` | `2.847` | `2.832` | `-0.015` |
| `2` | `2.930` | `2.910` | `-0.020` |

| Lane | Mean Final Eval Loss | Final Eval Std Dev | Mean Final Eval PPL | Mean Final Train Loss | Mean Median Tok/s/Device |
|---|---:|---:|---:|---:|---:|
| `attention` | `2.9097` | `0.0452` | `18.3747` | `3.2643` | `339,664` |
| `layer1-zero` | `2.8877` | `0.0396` | `17.9647` | `3.2517` | `301,531` |

Interpretation:

- The layer `1` zero-init sidecar won `3 / 3` seeds at the `1000`-step rung.
- Mean final eval loss improved by `-0.0220`.
- Mean median throughput regressed by about `11.2%` under TPU `lax.scan`.
- This is now a repeatable TinyStories quality signal, not just a seed `0`
  curiosity. It is still not a broad architecture or speed claim.

The TPU VM for seeds `1` and `2` was deleted after copying logs; `gcloud compute
tpus tpu-vm list` reported no running TPU VMs.

## RGRP Sidecar Matched-Control Rung

Validated on 2026-04-19 with:

- TPU VM: `fractal-sidecar-controls-202604191928`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_sidecar_control_rung_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_control_rung_20260419T1938Z/20260419T1938Z_sidecar_control_rung.md`

This rung kept the layer `1`, bottleneck `64`, side scale `0.1`, zero-init
sidecar contract fixed and changed only the sidecar operator:

- `rgrp`: rotary gated recurrent state update with token-axis state carry
- `tiny-mlp`: matched small non-recurrent MLP sidecar
- `tiny-glu`: matched small gated non-recurrent sidecar
- `binary-tree`: matched generic differentiable binary-tree sidecar

The same patched checkout also reran the attention baseline, so this is a
same-checkout comparison rather than a stitched cross-run table.

| Lane | Mean Final Eval | Std | Mean Delta vs Attention | Wins vs Attention | Mean Final Train | Mean Median Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `2.9097` | `0.0452` | `+0.0000` | `0/3` | `3.2643` | `360,003` |
| `rgrp` | `2.8877` | `0.0396` | `-0.0220` | `3/3` | `3.2517` | `302,123` |
| `tiny-mlp` | `2.9013` | `0.0435` | `-0.0083` | `3/3` | `3.2640` | `355,576` |
| `tiny-glu` | `2.9073` | `0.0423` | `-0.0023` | `2/3` | `3.2647` | `352,543` |
| `binary-tree` | `2.9057` | `0.0451` | `-0.0040` | `3/3` | `3.2627` | `357,680` |

Pairwise against RGRP:

| Control | Per-Seed Delta vs RGRP | Mean Delta vs RGRP | Wins vs RGRP |
|---|---:|---:|---:|
| `tiny-mlp` | `[+0.021, +0.009, +0.011]` | `+0.0137` | `0/3` |
| `tiny-glu` | `[+0.027, +0.017, +0.015]` | `+0.0197` | `0/3` |
| `binary-tree` | `[+0.026, +0.011, +0.017]` | `+0.0180` | `0/3` |

Interpretation:

- The sidecar effect is not purely "any small side branch helps." The controls
  help a little, especially `tiny-mlp`, but RGRP beats every matched control on
  every seed.
- This is the strongest TPU evidence so far that the recurrent state carry is
  contributing quality, not only extra capacity.
- The speed tradeoff remains real. RGRP is roughly `16.1%` slower than
  attention on mean median tok/s/device in this same-checkout run, while the
  non-recurrent controls stay near attention speed.
- Keep `tiny-mlp` as the default matched extra-capacity control for future
  larger-data rungs.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## RGRP Sidecar WikiText-103 Rung

Validated on 2026-04-19 with:

- TPU VM: `fractal-wikitext-rung-202604192010`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_sidecar_wikitext_rung_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/sidecar_wikitext_rung_20260419T2019Z/20260419T2019Z_sidecar_wikitext_rung.md`

This rung moved the current best sidecar contract off TinyStories and onto
`Salesforce/wikitext`, config `wikitext-103-raw-v1`, with the MaxText HF input
path and `validation` split. It kept the model shape and sidecar contract fixed:
4 layers, `d_model=256`, `mlp_dim=1024`, 4 heads, sequence length 512,
1000 training steps, eval every 250 steps, 20 eval steps, and seeds `0 1 2`.

The comparison narrowed to the three decision-relevant lanes:

- `attention`: baseline transformer
- `rgrp`: layer `1`, bottleneck `64`, side scale `0.1`, zero-init rotary gated
  recurrent state update sidecar
- `tiny-mlp`: strongest matched extra-capacity non-recurrent control

| Lane | Mean Final Eval | Std | Mean Delta vs Attention | Wins vs Attention | Mean Final Train | Mean Median Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `5.9437` | `0.0052` | `+0.0000` | `0/3` | `5.8733` | `334,426` |
| `rgrp` | `5.9340` | `0.0037` | `-0.0097` | `3/3` | `5.8717` | `304,831` |
| `tiny-mlp` | `5.9380` | `0.0016` | `-0.0057` | `3/3` | `5.8787` | `336,951` |

Pairwise against RGRP:

| Control | Per-Seed Delta vs RGRP | Mean Delta vs RGRP | Wins vs RGRP |
|---|---:|---:|---:|
| `tiny-mlp` | `[+0.003, +0.002, +0.007]` | `+0.0040` | `0/3` |

Interpretation:

- RGRP beat attention on all three WikiText-103 seeds.
- RGRP also beat the strongest matched extra-capacity control on all three
  seeds.
- The margin is smaller than on TinyStories: this is a survivability signal on
  broader text, not a large-quality claim.
- The TPU `lax.scan` speed tax remains meaningful. RGRP averaged about `8.9%`
  slower than attention and about `9.5%` slower than `tiny-mlp` on mean median
  tok/s/device.
- The recurrent state carry remains alive as an architectural hypothesis because
  it survived the dataset shift and the matched-control check.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae/RGRP-Control 8-Layer MaxText Port Smoke

Validated on 2026-04-19 with:

- TPU VM: `fractal-parcae-port-202604192052`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae8_port_smoke_20260419T2057Z-parcae8-smoke/20260419T2057Z_parcae8_port_smoke.md`

This smoke retired the 4-layer cheap-shape track as the main proof ladder. It
ported the H100 Parcae/P20-control architectural shape into MaxText/JAX:

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`, matching the Torch `_middle_loop_bounds`
  rule for 8 layers
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`

The MaxText patch now has explicit candidate profiles:

- `parcae-looped-attention`
- `parcae-bx-looped-attention`
- `parcae-rgrp-control-looped-attention`

The decoder-level scaffold is:

```text
prelude layers -> normalized loop input -> looped middle layers -> coda layers
```

For `parcae-rgrp-control-looped-attention`, a full-width RGRP scan runs over the
prelude stream and controls the loop injection value/gate. This is the TPU port
target for the earlier H100 P20-control winner.

Smoke results at 5 steps:

| Lane | Logged Params | Eval Step | Eval Loss | Eval Tokens | Final Train Loss | Median Post-Step0 Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4` | `9.501` | `15,302` | `9.665` | `478,140` |
| `parcae-looped` | `0.010B` | `4` | `9.626` | `15,302` | `9.792` | `402,634` |
| `parcae-bx` | `0.010B` | `4` | `9.624` | `15,302` | `9.791` | `421,685` |
| `parcae-rgrp-control` | `0.010B` | `4` | `9.625` | `15,302` | `9.791` | `400,111` |

Interpretation:

- All four proof-ladder lanes compile, train, and evaluate under MaxText/JAX/TPU.
- The 5-step loss ordering is not quality evidence.
- This validates the port and the Flax parameter-sharing contract for the
  looped middle band.
- The next serious rung should use this 8-layer Parcae proof-ladder shape, not
  the earlier 4-layer sidecar smoke shape.
- The remaining mismatch versus the H100 proof ladder is data ingress: H100 used
  the local 750M-token OpenLLaMA-tokenized FineWeb cache, while this smoke used
  WikiText text through MaxText's HF path.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae/RGRP-Control 8-Layer MaxText Sanity Rung

Validated on 2026-04-19 with:

- TPU VM: `fractal-parcae-sanity-202604192115`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae8_sanity_20260419T2119Z-parcae8-sanity/20260419T2119Z_parcae8_sanity.md`

This run was the first non-trivial TPU/JAX rung on the 8-layer Parcae proof
ladder shape:

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `Salesforce/wikitext`, config `wikitext-103-raw-v1`
- seed: `42`
- steps: `1000`
- eval interval: `250`
- eval steps: `20`

Per lane, the configured train budget was
`1000 * 64 * 256 = 16,384,000` token positions. WikiText padding reduced this
to `15,236,828` non-padding train tokens per lane. Four eval passes consumed
`1,074,268` non-padding eval tokens per lane.

Results:

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Checkpoint Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.923` | `+0.000` | `4.979` | `219,809` | `159s` |
| `parcae-looped` | `0.010B` | `4.911` | `-0.012` | `4.974` | `260,579` | `168s` |
| `parcae-bx` | `0.010B` | `4.910` | `-0.013` | `4.974` | `279,668` | `167s` |
| `parcae-rgrp-control` | `0.010B` | `4.894` | `-0.029` | `4.959` | `325,079` | `176s` |

Eval curves:

| Lane | Step 249 | Step 499 | Step 749 | Step 999 |
|---|---:|---:|---:|---:|
| `attention` | `5.589` | `5.199` | `5.006` | `4.923` |
| `parcae-looped` | `5.599` | `5.205` | `4.998` | `4.911` |
| `parcae-bx` | `5.596` | `5.207` | `4.998` | `4.910` |
| `parcae-rgrp-control` | `5.582` | `5.184` | `4.982` | `4.894` |

Interpretation:

- All three Parcae lanes beat the attention control by the final 1,000-step
  eval.
- `parcae-rgrp-control` was the best lane in this sanity rung, at `-0.029`
  final eval loss versus attention.
- This supports climbing one more TPU/JAX rung with the faithful 8-layer shape.
- It does not prove CUDA/H100 transfer, FineWeb parity, multi-seed robustness,
  or general superiority over attention.
- The remaining major mismatch versus the H100 proof ladder is data ingress:
  this run used WikiText-103 through MaxText's HF path, not the local
  OpenLLaMA-tokenized FineWeb cache.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae/RGRP-Control 8-Layer FineWeb-EDU Rung

Validated on 2026-04-19 with:

- TPU VM: `fractal-parcae-fw-202604192148`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae8_fineweb_rung_20260419T2200Z-parcae8-fineweb-rung4096/20260419T2200Z_parcae8_fineweb_rung.md`

This rung kept the faithful 8-layer Parcae proof-ladder shape and moved from
WikiText-103 to a larger FineWeb-EDU parquet stream:

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`,
  streamed through MaxText's parquet/HF path
- train file: `train/0000.parquet`
- eval file: `train/0013.parquet`
- seed: `42`
- steps: `4096`
- eval interval: `1024`
- eval steps: `64`

Per lane, the configured train budget was
`4096 * 64 * 256 = 67,108,864` token positions. Packed variable-length examples
reduced this to `64,873,455` non-padding train tokens per lane. Four eval passes
consumed `4,046,440` non-padding eval tokens per lane.

Results:

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Fast Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.7420` | `+0.0000` | `4.7830` | `764,786` | `752s` |
| `parcae-rgrp-control` | `0.010B` | `4.7000` | `-0.0420` | `4.7400` | `573,168` | `784s` |
| `parcae-bx` | `0.010B` | `4.7110` | `-0.0310` | `4.7490` | `637,510` | `767s` |

Eval curves:

| Lane | Step 1023 | Step 2047 | Step 3071 | Step 4095 |
|---|---:|---:|---:|---:|
| `attention` | `5.3610` | `4.9770` | `4.8080` | `4.7420` |
| `parcae-rgrp-control` | `5.3520` | `4.9510` | `4.7710` | `4.7000` |
| `parcae-bx` | `5.3590` | `4.9640` | `4.7830` | `4.7110` |

Interpretation:

- Both Parcae lanes beat attention on final eval loss at the same logged `~10M`
  parameter target.
- `parcae-rgrp-control` was best, at `-0.0420` final eval loss versus attention
  and `-0.0110` versus `parcae-bx`.
- `parcae-bx` also beat attention, so the looped Parcae scaffold remains a live
  control independent of the RGRP primitive.
- Attention remains the TPU throughput winner in this lowering. On median fast
  training steps, `parcae-rgrp-control` reached about `75%` of attention and
  `parcae-bx` reached about `83%`.
- This is stronger evidence than the WikiText sanity rung because it uses a
  larger FineWeb-EDU source and a 4096-step budget, but it is still one seed and
  a small `~10M` model. Do not use it to claim general attention replacement,
  CUDA/H100 transfer, or frontier-scale competitiveness.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae/RGRP-Control 8-Layer FineWeb-EDU GCS Rung

Validated on 2026-04-19 to 2026-04-20 with:

- TPU VM: `fractal-parcae-fw2-202604192316`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae8_fineweb_gcs_rung_20260419T2327Z-parcae8-fineweb-gcs-rung8192/20260419T2327Z_parcae8_fineweb_gcs_rung.md`

This rung kept the faithful 8-layer Parcae proof-ladder shape, doubled the step
budget, and moved the FineWeb-EDU parquet files into GCS so MaxText could stream
a broader train wildcard without relying on repeated remote HTTP file access:

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- staged train shards: `0000.parquet`, `0001.parquet`, `0002.parquet`,
  `0003.parquet`
- staged eval shard: `0013.parquet`
- staged compressed bytes: `9,462,253,997` train, `2,284,902,248` eval,
  `11,747,156,245` total
- seed: `42`
- steps: `8192`
- eval interval: `2048`
- eval steps: `64`

Per lane, the configured train budget was
`8192 * 64 * 256 = 134,217,728` token positions. Packed variable-length examples
reduced this to `129,678,585` non-padding train tokens per lane. Four eval
passes consumed `4,046,440` non-padding eval tokens per lane.

Results:

| Lane | Logged Params | Final Eval Loss | Delta vs Attention | Final Train Loss | Median Fast Tok/s | Speed vs Attention | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `0.010B` | `4.4840` | `+0.0000` | `4.3470` | `765,321` | `100.0%` | `1277s` |
| `parcae-rgrp-control` | `0.010B` | `4.4010` | `-0.0830` | `4.2500` | `573,489` | `74.9%` | `1316s` |
| `parcae-bx` | `0.010B` | `4.4050` | `-0.0790` | `4.2640` | `637,510` | `83.3%` | `1279s` |

Eval curves:

| Lane | Step 2047 | Step 4095 | Step 6143 | Step 8191 |
|---|---:|---:|---:|---:|
| `attention` | `5.0260` | `4.7080` | `4.5500` | `4.4840` |
| `parcae-rgrp-control` | `5.0080` | `4.6660` | `4.4810` | `4.4010` |
| `parcae-bx` | `5.0160` | `4.6700` | `4.4810` | `4.4050` |

Interpretation:

- Both Parcae lanes beat attention at every eval checkpoint in this doubled
  FineWeb-EDU GCS rung.
- `parcae-rgrp-control` remained the loss winner, but only narrowly: final eval
  loss was `0.0040` better than `parcae-bx`.
- `parcae-bx` captured most of the quality gain while running materially faster
  than `parcae-rgrp-control`: about `83.3%` of attention speed versus `74.9%`.
- The gap between `parcae-rgrp-control` and `parcae-bx` narrowed relative to the
  4096-step rung. At 4096 steps RGRP led BX by `0.0110`; here the final gap is
  `0.0040`.
- Attention is still the TPU throughput winner, but it is no longer the quality
  winner at this `~10M` proof-ladder shape and data budget.
- This is stronger evidence than the 4096-step remote-parquet run because the
  train stream is broader and the budget is doubled. It is still one seed, a
  small model, and a TPU/JAX lowering, so do not use it to claim CUDA transfer,
  frontier scaling, or general attention replacement.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae Training Contract Retest

Validated on 2026-04-20 with:

- TPU VM: `fractal-parcae-contract-202604200132`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_contract_retest_20260420T020311Z/20260420T020311Z_parcae_contract_retest.md`

This rung retested the 8-layer Parcae proof-ladder lanes after making the
Parcae training contract explicit:

- fixed vs stochastic per-sequence loop policy
- clipped-Poisson per-sequence loop-depth sampling for the stochastic ablation
- separate recurrence and backward active-depth knobs
- explicit discretization policy
- the same outer loop policy available to both `parcae-bx` and
  `parcae-rgrp-control`

Shared shape:

- physical decoder layers: `8`
- loop count: `2`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- seed: `42`
- steps: `4096`
- eval interval: `1024`
- eval steps: `64`
- observed non-padding train tokens per lane: `64,842,323`

All Parcae lanes used `PARCAE_DISCRETIZATION=zoh` in this retest.

Results:

| Lane | Final Eval Loss | Delta vs Attention | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `4.739` | `+0.000` | `114.369` | `4.796` | `765,250` | `747s` |
| `parcae-bx` | `4.705` | `-0.034` | `110.471` | `4.766` | `638,379` | `764s` |
| `parcae-rgrp-control` | `4.681` | `-0.058` | `107.828` | `4.738` | `555,051` | `778s` |
| `parcae-bx-perseq` | `4.805` | `+0.066` | `122.068` | `4.882` | `470,291` | `790s` |
| `parcae-rgrp-control-perseq` | `4.779` | `+0.040` | `118.946` | `4.853` | `437,794` | `806s` |

Eval curves:

| Lane | Step 1023 | Step 2047 | Step 3071 | Step 4095 |
|---|---:|---:|---:|---:|
| `attention` | `5.363` | `4.971` | `4.805` | `4.739` |
| `parcae-bx` | `5.365` | `4.950` | `4.771` | `4.705` |
| `parcae-rgrp-control` | `5.335` | `4.922` | `4.746` | `4.681` |
| `parcae-bx-perseq` | `5.490` | `5.063` | `4.877` | `4.805` |
| `parcae-rgrp-control-perseq` | `5.480` | `5.042` | `4.852` | `4.779` |

Interpretation:

- Fixed `parcae-rgrp-control` was the best lane in this retest.
- Fixed `parcae-bx` also beat attention, so the Parcae loop scaffold remains a
  live control independent of the rotary gated recurrent state update primitive.
- Stochastic per-sequence loop depth hurt both quality and speed at this scale.
- Per-sequence loop depth should not become the proof-ladder default. Keep it as
  an explicit ablation only.
- Because every Parcae lane used `zoh`, this run does not isolate `zoh` against
  `stable-exp`. The next clean ablation is fixed `parcae-rgrp-control` with
  `zoh` vs fixed `parcae-rgrp-control` with `stable-exp`.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae RGRP-Control Discretization Ablation

Validated on 2026-04-20 with:

- TPU VM: `fractal-zoh-ablate-20260420033857`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_discretization_ablation_20260420T035019Z/20260420T035019Z_parcae_discretization_ablation.md`

This ablation isolated the Parcae discretization policy for the fixed
`parcae-rgrp-control` lane:

- `fractal_parcae_discretization=stable-exp`
- `fractal_parcae_discretization=zoh`

Shared shape:

- candidate: `parcae-rgrp-control-looped-attention`
- physical decoder layers: `8`
- loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- seed: `42`
- steps: `2048`
- eval interval: `512`
- eval steps: `64`
- observed non-padding train tokens per lane: `32,420,430`

Results:

| Discretization | Final Eval Loss | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|
| `stable-exp` | `5.015` | `150.584` | `4.972` | `548,988` | `467s` |
| `zoh` | `5.015` | `150.700` | `4.974` | `554,976` | `465s` |

Eval curves:

| Discretization | Step 511 | Step 1023 | Step 1535 | Step 2047 |
|---|---:|---:|---:|---:|
| `stable-exp` | `5.736` | `5.287` | `5.090` | `5.015` |
| `zoh` | `5.737` | `5.288` | `5.091` | `5.015` |

Interpretation:

- The discretization knob was effectively neutral at this `2048`-step rung.
- `stable-exp` had a tiny final perplexity edge; `zoh` had a tiny median
  throughput edge. Neither difference is large enough to promote as a finding.
- The earlier fixed-policy Parcae/RGRP-control win does not appear dependent on
  `zoh`.
- Use `stable-exp` as the proof-ladder default because it is simpler and avoids
  the extra learned `dt_raw` parameter. Keep `zoh` as an explicit ablation knob.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae 16k Scale Rung

Validated on 2026-04-20 with:

- TPU VM: `fractal-parcae-scale-20260420041649`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_scale_20260420T042711Z/20260420T042711Z_parcae_scale_16k.md`

This rung promoted the fixed-policy Parcae lanes to `16,384` training steps on
the FineWeb-EDU GCS parquet surface:

- `attention`
- `parcae-bx`
- `parcae-rgrp-control`

Shared shape:

- physical decoder layers: `8`
- Parcae loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- seed: `42`
- steps: `16,384`
- eval interval: `2,048`
- eval steps: `64`
- observed non-padding train tokens per lane: `259,352,782`
- Parcae discretization: `stable-exp`

Results:

| Lane | Final Eval Loss | Delta vs Attention | Delta vs BX | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `4.240` | `+0.000` | `+0.064` | `69.421` | `4.218` | `765,464` | `2,526s` |
| `parcae-bx` | `4.176` | `-0.064` | `+0.000` | `65.075` | `4.159` | `637,064` | `2,586s` |
| `parcae-rgrp-control` | `4.151` | `-0.089` | `-0.025` | `63.476` | `4.149` | `548,970` | `2,617s` |

Eval curves:

| Lane | 2047 | 4095 | 6143 | 8191 | 10239 | 12287 | 14335 | 16383 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `5.165` | `4.770` | `4.583` | `4.454` | `4.368` | `4.304` | `4.262` | `4.240` |
| `parcae-bx` | `5.144` | `4.710` | `4.497` | `4.381` | `4.300` | `4.237` | `4.197` | `4.176` |
| `parcae-rgrp-control` | `5.120` | `4.693` | `4.490` | `4.364` | `4.277` | `4.214` | `4.172` | `4.151` |

Interpretation:

- `parcae-rgrp-control` won every eval checkpoint against both controls.
- `parcae-bx` also beat attention, so the looped Parcae scaffold remains a live
  architectural control independent of the rotary gated recurrent state update
  primitive.
- RGRP-control still pays a fast-step speed tax: about `71.7%` of attention
  median throughput and about `86.2%` of `parcae-bx`.
- This is quality evidence for the TPU/JAX proof ladder, not a CUDA speed claim.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae 16k Seed Replication

Validated on 2026-04-20 with:

- TPU VM: `fractal-parcae-seeds-20260420T065356`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_seedrep_20260420T070416Z/20260420T070416Z_parcae_seedrep_16k.md`

This run replicated the `16,384`-step Parcae proof-ladder rung for two new
seeds after the initial seed-42 run showed `parcae-rgrp-control` beating both
attention and the Parcae B(x) scaffold control.

Shared shape matched the seed-42 rung:

- physical decoder layers: `8`
- Parcae loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=128`, `heads=4`, `head_dim=32`, `mlp_dim=512`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- new seeds: `7`, `123`
- prior comparison seed: `42`
- steps: `16,384`
- eval interval: `2,048`
- eval steps: `64`
- observed non-padding train tokens per lane:
  - seed `7`: `259,416,359`
  - seed `123`: `259,428,995`
- Parcae discretization: `stable-exp`

New seed results:

| Seed | Lane | Final Eval Loss | Delta vs Attention | Delta vs BX | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `7` | `attention` | `4.223` | `+0.000` | `+0.047` | `68.238` | `4.122` | `765,107` | `2,622s` |
| `7` | `parcae-bx` | `4.176` | `-0.047` | `+0.000` | `65.106` | `4.078` | `636,519` | `2,675s` |
| `7` | `parcae-rgrp-control` | `4.170` | `-0.053` | `-0.006` | `64.691` | `4.060` | `548,988` | `2,711s` |
| `123` | `attention` | `4.242` | `+0.000` | `+0.068` | `69.522` | `4.102` | `765,143` | `2,601s` |
| `123` | `parcae-bx` | `4.174` | `-0.068` | `+0.000` | `64.988` | `4.039` | `636,544` | `2,646s` |
| `123` | `parcae-rgrp-control` | `4.163` | `-0.079` | `-0.011` | `64.242` | `4.030` | `548,951` | `2,563s` |

Three-seed aggregate including the prior seed-42 run:

| Lane | Seed 42 | Seed 7 | Seed 123 | Mean Final Eval Loss | Mean PPL | Mean Median Fast Tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `attention` | `4.240` | `4.223` | `4.242` | `4.235` | `69.060` | `765,238` |
| `parcae-bx` | `4.176` | `4.176` | `4.174` | `4.175` | `65.056` | `636,709` |
| `parcae-rgrp-control` | `4.151` | `4.170` | `4.163` | `4.161` | `64.136` | `548,970` |

Interpretation:

- `parcae-rgrp-control` won both new seeds against attention and B(x).
- Across all three seeds, RGRP-control is now `3/3` against both controls.
- The mean RGRP-control advantage is `-0.074` eval loss vs attention and
  `-0.014` eval loss vs B(x).
- B(x) remains a required scaffold control because it explains most of the
  improvement over attention.
- RGRP-control still pays a speed tax: about `71.7%` of attention median
  throughput and about `86.2%` of B(x) at this TPU shape.

Decision: the seed-replication gate passed. Promote to a model-size scale rung
while carrying all three lanes.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae Width-Scale 16k Rung

Validated on 2026-04-20 with:

- TPU VM: `fractal-parcae-width-20260420T115012`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_width_20260420T115012Z/20260420T115012Z_parcae_width_16k.md`

This run scaled the fixed-policy Parcae proof-ladder shape from `d_model=128`
to `d_model=192` while preserving the same data, tokenizer, 8-layer schedule,
middle-band Parcae loop, step budget, eval cadence, and seed.

Shared shape:

- physical decoder layers: `8`
- Parcae loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=192`, `heads=6`, `head_dim=32`, `mlp_dim=768`
- MaxText reported parameter size: `0.017B`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- seed: `42`
- steps: `16,384`
- eval interval: `2,048`
- eval steps: `64`
- observed non-padding train tokens per lane: `259,352,782`
- Parcae discretization: `stable-exp`

Results:

| Lane | Final Eval Loss | Delta vs Attention | Delta vs BX | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `3.978` | `+0.000` | `+0.032` | `53.404` | `3.952` | `474,857` | `2,591s` |
| `parcae-bx` | `3.946` | `-0.032` | `+0.000` | `51.716` | `3.924` | `387,027` | `2,671s` |
| `parcae-rgrp-control` | `3.905` | `-0.073` | `-0.041` | `49.675` | `3.889` | `354,939` | `2,700s` |

Eval curves:

| Lane | 2047 | 4095 | 6143 | 8191 | 10239 | 12287 | 14335 | 16383 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `4.947` | `4.526` | `4.323` | `4.203` | `4.118` | `4.048` | `4.002` | `3.978` |
| `parcae-bx` | `4.911` | `4.438` | `4.266` | `4.164` | `4.084` | `4.015` | `3.968` | `3.946` |
| `parcae-rgrp-control` | `4.885` | `4.438` | `4.243` | `4.130` | `4.046` | `3.977` | `3.930` | `3.905` |

Interpretation:

- `parcae-rgrp-control` survived the first model-size scale rung.
- The final RGRP-control advantage was `0.073` eval loss over attention and
  `0.041` eval loss over B(x).
- The incremental RGRP-over-B(x) margin increased relative to the three-seed
  `d_model=128` mean (`0.014`) and relative to seed-42 at `d_model=128`
  (`0.025`).
- B(x) still beat attention, so the Parcae scaffold remains a real control.
- RGRP-control still pays a throughput tax: about `74.7%` of attention median
  throughput and about `91.7%` of B(x).

Decision: replicate `d_model=192` across at least two more seeds before
increasing width again. Carry all three lanes for stronger claims; run only
B(x) and RGRP-control if compute budget favors speed and the question is the
incremental recurrent-control edge.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae Width-Scale 16k Seed Replication

Validated on 2026-04-20 with:

- TPU VM: `fractal-parcae-wrep-20260420T143015`
- hardware: `v5litepod-1` spot in `us-west4-a`
- runtime: `v2-tpuv5-litepod`
- MaxText checkout: `AI-Hypercomputer/maxtext`, patched with
  `scripts/patch_maxtext_rgrp.py`
- runner script: `scripts/run_maxtext_parcae_proof_ladder_tpu.sh`
- matched artifact:
  `experiments/jax_tpu/maxtext_quality/parcae_width_rep_20260420T143015Z/20260420T143015Z_parcae_width_seedrep_16k.md`

This run replicated the `d_model=192` Parcae width-scale rung on two new seeds
after seed `42` showed RGRP-control beating both attention and B(x).

Shared shape:

- physical decoder layers: `8`
- Parcae loop count: `2`
- loop policy: `fixed`
- recurrence active depth: `2`
- backward active depth: `1`
- middle loop band: `layers 3..4`
- execution scaffold: `A0 A1 A2 [A3 A4] [A3 A4] A5 A6 A7`
- `d_model=192`, `heads=6`, `head_dim=32`, `mlp_dim=768`
- MaxText reported parameter size: `0.017B`
- context `256`, batch `64`
- vocab size `32000`, tokenizer `openlm-research/open_llama_3b_v2`
- dataset: `HuggingFaceFW/fineweb-edu`, config dump `CC-MAIN-2013-20`
- train files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/train/*.parquet`
- eval files:
  `gs://fractal-maxtext-runs-81f2add4/fineweb_edu_ccmain2013_20_parquet/eval/*.parquet`
- new seeds: `7`, `123`
- prior comparison seed: `42`
- steps: `16,384`
- eval interval: `2,048`
- eval steps: `64`
- Parcae discretization: `stable-exp`

New seed results:

| Seed | Lane | Final Eval Loss | Delta vs Attention | Delta vs BX | PPL | Final Train Loss | Median Fast Tok/s | Log Duration |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `7` | `attention` | `3.995` | `+0.000` | `+0.054` | `54.316` | `3.888` | `474,967` | `2,594s` |
| `7` | `parcae-bx` | `3.941` | `-0.054` | `+0.000` | `51.445` | `3.830` | `386,972` | `2,679s` |
| `7` | `parcae-rgrp-control` | `3.907` | `-0.088` | `-0.034` | `49.740` | `3.804` | `354,878` | `2,712s` |
| `123` | `attention` | `3.987` | `+0.000` | `+0.054` | `53.893` | `3.846` | `474,961` | `2,569s` |
| `123` | `parcae-bx` | `3.933` | `-0.054` | `+0.000` | `51.050` | `3.794` | `387,027` | `2,630s` |
| `123` | `parcae-rgrp-control` | `3.904` | `-0.083` | `-0.029` | `49.610` | `3.764` | `354,878` | `2,670s` |

Three-seed aggregate including the prior seed-42 run:

| Lane | Seed 7 | Seed 42 | Seed 123 | Mean Final Eval Loss | Loss Std | Mean PPL | Mean Median Fast Tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention` | `3.995` | `3.978` | `3.987` | `3.987` | `0.0069` | `53.871` | `474,928` |
| `parcae-bx` | `3.941` | `3.946` | `3.933` | `3.940` | `0.0054` | `51.404` | `387,009` |
| `parcae-rgrp-control` | `3.907` | `3.905` | `3.904` | `3.905` | `0.0012` | `49.675` | `354,898` |

Interpretation:

- `parcae-rgrp-control` beat both controls on all three `d_model=192` seeds.
- The aggregate RGRP-control margin was `-0.082` eval loss vs attention and
  `-0.035` eval loss vs B(x).
- The RGRP-over-B(x) edge strengthened with width:
  - `d_model=128` three-seed mean RGRP-vs-B(x): `-0.014`
  - `d_model=192` three-seed mean RGRP-vs-B(x): `-0.035`
- B(x) remained a strong scaffold control and beat attention on all three
  seeds.
- RGRP-control still pays a throughput tax: about `74.7%` of attention median
  throughput and about `91.7%` of B(x).

Decision: the first width-scale result is now replicated across three seeds.
The next proof-ladder rung can be a single-seed larger-width scout, carrying all
three lanes. Candidate shape: `d_model=256`, `heads=8`, `head_dim=32`,
`mlp_dim=1024`, `layers=8`, same middle-band Parcae schedule, same data,
tokenizer, context, and batch if it fits. If `batch=64` does not fit, retry at
`batch=32` and document the token-budget difference explicitly.

The TPU VM was deleted after copying logs; `gcloud compute tpus tpu-vm list`
reported no running TPU VMs.

## Parcae Width-256 Replication Failure Diagnosis

Diagnosed on 2026-04-21 after the non-spot `d_model=256` replication gate
short-ended two second-seed Parcae lanes:

- artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_width256_rep_20260421T032904Z`
- TPU VM: `fractal-parcae-w256rep-20260421t032904`
- hardware: `v5litepod-1` non-spot in `us-west4-a`
- shape: `d_model=256`, `layers=8`, `heads=8`, `head_dim=32`,
  `mlp_dim=1024`, context `256`, batch `64`
- lanes/seeds requested: `attention`, `parcae-bx`, `parcae-rgrp-control` for
  seeds `7` and `123`
- steps requested per lane: `16,384`

Completed results:

| Seed | Lane | Completed Steps | Final Eval Loss | PPL | Median Fast Tok/s | Status |
|---:|---|---:|---:|---:|---:|---|
| `7` | `attention` | `16,384` | `3.831` | `46.115` | `355,764` | complete |
| `7` | `parcae-bx` | `16,384` | `3.782` | `43.896` | `288,050` | complete |
| `7` | `parcae-rgrp-control` | `16,384` | `3.748` | `42.451` | `254,355` | complete |
| `123` | `attention` | `16,384` | `3.827` | `45.931` | `355,586` | complete |
| `123` | `parcae-bx` | `5,021` | `4.317 @ step 4,095` | `74.970` | `288,146` | invalid incomplete |
| `123` | `parcae-rgrp-control` | `3,826` | `4.755 @ step 2,047` | `116.141` | `254,386` | invalid incomplete |

Root cause:

- MaxText logged `Training stopped: load_next_batch() failed` for both invalid
  Parcae lanes.
- The concrete exception was `FileNotFoundError` for a `/psm_*` shared-memory
  segment inside the Hugging Face/Grain data path.
- MaxText treats this `StopTraining` path as graceful job completion, so the
  shell pipeline exited cleanly, the wrapper advanced to the next lane, and the
  guard collected/deleted the TPU.
- The scientific comparison was therefore partial, not failed by model quality.
  Seed `7` reproduced `parcae-rgrp-control > parcae-bx > attention`, but the
  two-seed `d_model=256` replication gate remains open.

Control-plane fix:

- `scripts/run_maxtext_parcae_proof_ladder_tpu.sh` now validates every lane log
  after MaxText exits.
- Validation fails the wrapper if MaxText logs `Training stopped`, if
  `load_next_batch()` fails, if the last completed step is not `steps - 1`, or
  if the final eval step is missing when the run contract expects it.
- The runner defaults `grain_worker_count=0` and `grain_worker_count_eval=0`.
- `scripts/patch_maxtext_rgrp.py` now patches MaxText's Hugging Face pipeline
  to actually pass `config.grain_worker_count` and
  `config.grain_worker_count_eval` through to the HF preprocessing DataLoader.

Decision: do not relaunch until this runner/patch pair is applied to the TPU
checkout. The next valid retry should run only the missing seed-`123` Parcae
lanes after confirming the patched MaxText source contains the HF worker-count
wiring.

## Parcae Width-256 Two-Seed Result

Completed on 2026-04-21 after rerunning only the invalid seed-`123` Parcae
lanes with the patched data-loader control plane:

- original artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_width256_rep_20260421T032904Z`
- retry artifact root:
  `experiments/jax_tpu/maxtext_quality/parcae_width256_rep_retry_20260421T141730Z`
- shape: `d_model=256`, `layers=8`, `heads=8`, `head_dim=32`,
  `mlp_dim=1024`, context `256`, batch `64`
- steps per lane: `16,384`
- train token positions per lane: `268,435,456`
- tokenizer: `openlm-research/open_llama_3b_v2`
- data: FineWeb-EDU `CC-MAIN-2013-20` parquet files in GCS
- discretization: `stable-exp`
- Parcae loop band: physical layers `3..4`, fixed loop count `2`

Per-seed results:

| Seed | Lane | Final Eval Loss | PPL | Final Train Loss | Median Fast Tok/s |
|---:|---|---:|---:|---:|---:|
| `7` | `attention` | `3.831` | `46.115` | `3.723` | `355,818` |
| `7` | `parcae-bx` | `3.782` | `43.896` | `3.674` | `288,080` |
| `7` | `parcae-rgrp-control` | `3.748` | `42.451` | `3.640` | `254,378` |
| `123` | `attention` | `3.827` | `45.931` | `3.683` | `355,617` |
| `123` | `parcae-bx` | `3.790` | `44.247` | `3.637` | `288,111` |
| `123` | `parcae-rgrp-control` | `3.757` | `42.838` | `3.603` | `254,359` |

Two-seed aggregate:

| Lane | Mean Final Eval Loss | Loss Std | Mean PPL | Mean Median Fast Tok/s |
|---|---:|---:|---:|---:|
| `attention` | `3.8290` | `0.0020` | `46.023` | `355,718` |
| `parcae-bx` | `3.7860` | `0.0040` | `44.072` | `288,096` |
| `parcae-rgrp-control` | `3.7525` | `0.0045` | `42.645` | `254,368` |

Interpretation:

- `parcae-rgrp-control` beat both controls on both `d_model=256` seeds.
- Mean RGRP-control margin was `-0.0765` eval loss vs attention and `-0.0335`
  eval loss vs B(x).
- The RGRP-over-B(x) edge remains close to the `d_model=192` replicated edge
  (`-0.035`) while the attention gap remains substantial.
- RGRP-control still pays a meaningful throughput tax: about `71.5%` of
  attention median throughput and about `88.3%` of B(x).

Operational note:

- The seed-`123` retry was accidentally launched as non-spot in `us-west4-a`.
  It completed and deleted cleanly, but the TPU grant email lists free v5e spot
  quota in `us-central1-a`; future `v5litepod-1` launches should use
  `us-central1-a --spot` unless a run explicitly chooses another grant-covered
  zone.

Decision: the `d_model=256` proof-ladder rung is a real positive scale signal.
The next rung is a single-seed `d_model=384` scout carrying all three lanes:
`attention`, `parcae-bx`, and `parcae-rgrp-control`. Use the same data,
context, tokenizer, layer count, Parcae loop band, and step budget. Start with
batch `64`; if it does not fit, retry batch `32` and document the token-budget
difference explicitly.

## Non-Goals

- Do not use this lane to make CUDA speed claims.
- Do not port every Fractal primitive at once.
- Do not write Pallas kernels before the `jax.lax.scan` reference proves the
  architecture deserves the effort.
- Do not silently pass custom adapter flags to unchanged MaxText.
