# v3A Rust Mamba-3 Porting Note

## Purpose

This note maps the **official PyTorch Mamba-3 block** to the Rust Path 1
baseline contract.

It is the missing layer between:

* the high-level Path 1 plan
* the baseline design note
* and the actual Rust implementation work

The source of truth for this note is:

* official README usage section for Mamba-3  
  Source: [state-spaces/mamba README](https://github.com/state-spaces/mamba?tab=readme-ov-file#mamba-3)
* official implementation file  
  Source: [mamba_ssm/modules/mamba3.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)
* official simple mixer backbone wrapper  
  Source: [mamba_ssm/models/mixer_seq_simple.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py)

This note is about **implementation faithfulness**, not speed parity.

---

## Official Surface We Are Targeting

The public usage surface in the official repo is:

```python
from mamba_ssm import Mamba3
batch, length, dim = 2, 2048, 768
x = torch.randn(batch, length, dim).to(torch.bfloat16).to("cuda")
model = Mamba3(
    d_model=dim,
    d_state=128,
    headdim=64,
    is_mimo=True,
    mimo_rank=4,
    chunk_size=16,
    is_outproj_norm=False,
    dtype=torch.bfloat16,
).to("cuda")
y = model(x)
assert y.shape == x.shape
```

The official constructor also exposes other important controls including:

* `expand`
* `ngroups`
* `rope_fraction`
* `dt_min`
* `dt_max`
* `dt_init_floor`
* `A_floor`
* `layer_idx`

Source: [README lines 365-376](https://github.com/state-spaces/mamba?tab=readme-ov-file#mamba-3), [mamba3.py constructor](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

The official repo also provides a deep sequence-model wrapper:

* `MixerModel`
* `MambaLMHeadModel`

in `mixer_seq_simple.py`, which shows how the block is actually composed into a
language model backbone.  
Source: [MixerModel](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py), [MambaLMHeadModel](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py)

---

## What The Official Block Clearly Does

From the constructor and forward path, the official block has these structural
properties:

1. **expanded inner width**
   * `d_inner = expand * d_model`
   * `nheads = d_inner / headdim`  
   Source: [mamba3.py lines 1348-1352](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

2. **one large input projection**
   The input is projected once and then split into multiple streams:
   * `z`
   * `x`
   * `B`
   * `C`
   * `dd_dt`
   * `dd_A`
   * `trap`
   * `angles`  
   Source: [mamba3.py lines 1371-1375](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1471-1493](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

3. **separate latent state and emitted output**
   The block does not define output as just “new state.”
   It computes recurrent internals, forms `y`, optionally normalizes/gates it,
   and then applies `out_proj` back to `d_model`.  
   Source: [mamba3.py lines 1427-1445](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1647-1657](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

4. **selective continuous-time style parameters**
   The forward path derives:
   * `_A = -softplus(dd_A)` with floor clamp
   * `DT = softplus(dd_dt + dt_bias)`
   * `trap = sigmoid(trap_proj)`  
   Source: [mamba3.py lines 1504-1510](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1659-1668](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

5. **rotary / angle-conditioned state interaction**
   The block carries explicit `angles`, `rope_fraction`, and rotary dimension
   logic.  
   Source: [mamba3.py lines 1356-1369](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

6. **optional MIMO mode**
   * `is_mimo`
   * `mimo_rank`
   * `mimo_*` parameters
   * different kernel paths depending on MIMO vs SISO  
   Source: [mamba3.py lines 1335-1346](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1411-1419](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

7. **chunked kernel execution**
   `chunk_size` is not cosmetic; it is part of the actual execution surface.  
   Source: [mamba3.py lines 1300-1303](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1558-1563](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py), [mamba3.py lines 1624-1633](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

8. **inference state cache**
   The block carries explicit cached state for inference:
   * `angle_dt_state`
   * `ssm_state`
   * `k_state`
   * `v_state`  
   Source: [mamba3.py lines 1458-1468](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)

9. **official backbone composition**
   The repo does not treat the block in isolation; it composes it through:
   * `create_block(...)`
   * `MixerModel`
   * `MambaLMHeadModel`

   with explicit choices for:
   * norm type (`LayerNorm` vs `RMSNorm`)
   * fused add+norm option
   * residual-in-fp32 option
   * final norm
   * tied embedding / LM head behavior  
   Source: [create_block](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py), [MixerModel init](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py), [MambaLMHeadModel](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py)

---

## What Must Match In Rust Phase 1A

To call the Rust baseline “faithful enough,” these things must match the
official implementation conceptually and structurally:

1. **expanded inner width**
   The Rust baseline must use explicit `d_model`, `d_inner`, `headdim`, and
   derived `nheads`, not a flat single-width proxy.

2. **single in-projection with split streams**
   The Rust baseline must project into distinct internal streams analogous to:
   * `z`
   * `x`
   * `B`
   * `C`
   * `dd_dt`
   * `dd_A`
   * `trap`
   * `angles`

3. **separate output path**
   The Rust baseline must preserve the distinction between:
   * recurrent latent update
   * emitted token representation
   * final output projection back to `d_model`

4. **official-style selective parameterization**
   The Rust baseline must implement the equivalent of:
   * `A = -softplus(...)` with floor clamp
   * `DT = softplus(... + dt_bias)`
   * `trap = sigmoid(...)`

5. **rotary / angle-conditioned behavior**
   The Rust baseline must preserve angle-conditioned state interaction rather
   than replacing it with an unrelated transform.

6. **MIMO control in the public config**
   Even if the first pass starts with one narrow mode, the config surface must
   explicitly acknowledge:
   * `is_mimo`
   * `mimo_rank`

7. **output shape contract**
   Input and output must remain:
   * `(batch, length, d_model)` -> `(batch, length, d_model)`

8. **backbone-level composition contract**
   The Rust baseline must preserve the existence of:
   * block factory / typed block construction
   * repeated deep mixer backbone
   * final normalization
   * LM head projection

   so that the reference lane is a real language-model backbone, not just a
   floating standalone block.

---

## What Can Be Deferred In Rust Phase 1A

These things can be deferred without breaking the architectural baseline claim:

1. **kernel-level parity**
   We do not need Triton/CUTE/TileLang parity on day one.

2. **CUDA-first execution strategy**
   CPU and Metal correctness come first for local bring-up.
   This is a deferral of backend work, not a waiver: CUDA must still be closed
   before we treat RunPod benchmarking as valid for the Path 1 baseline.

3. **full inference-cache parity**
   The training-time recurrent path is the initial priority.
   The cache API can follow after the core block is correct.

4. **full MIMO optimization**
   We may start with a more direct implementation of the MIMO math before
   optimizing it.

5. **chunked fast path**
   `chunk_size` should stay in the typed config, but the first correct Rust
   implementation may use a straightforward sequential scan rather than the
   fused chunked kernels.

6. **fused add+norm kernels**
   The first Rust pass may keep add and normalization as separate steps instead
   of matching Triton fused kernels.

These are deferrals of **performance engineering**, not of the core contract.

---

## Explicit Current Gap

The Rust baseline is now strong enough for local Path 1 comparison work, but
CUDA is still an explicit gap.

That means:

* local CPU and Metal benchmarking is allowed
* RunPod or other CUDA-only benchmark claims are **not** yet allowed
* before CUDA benchmarking, we must validate the Rust lane on a CUDA backend
  through the same parity and smoke surfaces we used locally

This keeps the benchmark story honest:

* local Rust baseline first
* CUDA backend closure second
* RunPod benchmark claims only after both are real

---

## What The Current Proxy Does Not Yet Match

The current `mamba3-proxy-v1` is useful for early validation, but it is still
missing several official-implementation properties:

* no explicit `expand -> d_inner -> nheads` contract
* no single large in-projection split into official-style streams
* no explicit `z / x / B / C / dd_dt / dd_A / trap / angles` decomposition
* no official-style `_A`, `DT`, and `trap` preprocessing path
* no typed `is_mimo` or `mimo_rank` behavior
* no official-style `out_proj` contract from `d_inner -> d_model`
* no cache-compatible recurrent interface matching the official forward/step split
* no backbone-level contract matching `MixerModel` / `MambaLMHeadModel`

So the proxy is **architecturally suggestive**, but not a faithful port target.

---

## Rust Mapping Contract

The Rust faithful baseline should introduce a block with a typed config roughly
like:

```text
RustMamba3BaselineConfig {
  d_model,
  d_state,
  expand,
  headdim,
  ngroups,
  rope_fraction,
  dt_min,
  dt_max,
  dt_init_floor,
  A_floor,
  is_mimo,
  mimo_rank,
  chunk_size,
  is_outproj_norm,
}
```

And a block contract roughly like:

```text
in_proj(x_t) -> [z, x, B, C, dd_dt, dd_A, trap, angles]
preprocess(dd_A, dd_dt, trap, B, C, angles) -> [_A, DT, trap, B, C, angles]
scan(previous_state, x, B, C, _A, DT, trap, angles) -> next_state, y_inner
optional_norm(y_inner, z)
out_proj(y_inner) -> y_t
```

This does not force exact naming in Rust, but it does force the same structural
decomposition.

At the backbone level, the Rust baseline should introduce something roughly like:

```text
RustMamba3Block
RustMamba3Backbone
RustMamba3LmHeadModel
```

with typed choices for:

* norm kind
* residual precision policy
* final norm
* LM head tie/no-tie policy

---

## Bring-Up Order

For the first faithful Rust implementation:

1. match the public config surface
2. match the in-projection split contract
3. match the preprocess contract for `_A`, `DT`, and `trap`
4. match the recurrent scan contract
5. match the separate output readout
6. match the backbone-level composition contract
7. only then worry about optimization

That order keeps the port honest.

---

## Decision Rule

We may call the Rust baseline “faithful enough for Path 1 benchmarking” only if:

* the implementation clearly matches the official block structure
* the current proxy-only shortcuts are gone from the main reference lane
* the Rust lane is reproducible on the shared Path 1 surface
* the remaining gaps are performance-oriented rather than architecture-oriented

If the remaining gaps are still architectural, then the baseline is not ready
to freeze.
