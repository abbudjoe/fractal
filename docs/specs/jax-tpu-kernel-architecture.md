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
5. Run baseline and P20 with equal shape, sequence length, batch, optimizer,
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
- If we continue custom TPU work, the next design should be chunked/tiled:
  expose state transform matmul tiles and sequence chunks to Pallas rather than
  placing the entire recurrent loop inside one program.
- Otherwise, for near-term TPU quality runs, use the `lax.scan` reference with
  `scan_unroll=3`.

The TPU VM was deleted after the Pallas smoke; `gcloud compute tpus tpu-vm list`
was empty afterward.

## Non-Goals

- Do not use this lane to make CUDA speed claims.
- Do not port every Fractal primitive at once.
- Do not write Pallas kernels before the `jax.lax.scan` reference proves the
  architecture deserves the effort.
- Do not silently pass custom adapter flags to unchanged MaxText.
