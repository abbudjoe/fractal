"""Small JAX language-model integration smoke for TPU architecture gates.

This is deliberately not a replacement training stack.  It is a cheap,
repo-owned gate that verifies whether a candidate block participates in a full
LM-shaped forward/backward path before we patch or fork MaxText.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from python.jax_tpu.adapters import rotary_gated_recurrent_state_update as rgrp

try:  # Keep local imports working when JAX is absent.
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised in local envs without JAX.
    jax = None
    jnp = None


Array = Any
Params = dict[str, Any]


SUPPORTED_VARIANTS = ("mlp", "rgrp")


@dataclass(frozen=True)
class JaxLmSmokeConfig:
    variant: str = "mlp"
    vocab_size: int = 4_096
    seq_len: int = 128
    batch_size: int = 8
    d_model: int = 256
    layers: int = 2
    heads: int = 4
    ffn_multiplier: int = 4
    rgrp_state_transform: str = "block-diagonal-4-masked-dense"
    rgrp_scan_unroll: int = 1
    rgrp_projection_mode: str = "sequence"
    rgrp_trig_mode: str = "precompute"
    rgrp_execution_mode: str = "scan"
    dtype: str = "bfloat16"

    @property
    def d_ff(self) -> int:
        return self.d_model * self.ffn_multiplier

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    def validate(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of {SUPPORTED_VARIANTS}, got {self.variant!r}")
        for name in ("vocab_size", "seq_len", "batch_size", "d_model", "layers", "heads", "ffn_multiplier"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model {self.d_model} must be divisible by heads {self.heads}")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"dtype must be bfloat16 or float32, got {self.dtype!r}")
        rgrp.RotaryGatedRecurrentStateUpdateConfig(
            d_model=self.d_model,
            state_transform=self.rgrp_state_transform,
            scan_unroll=self.rgrp_scan_unroll,
            projection_mode=self.rgrp_projection_mode,
            trig_mode=self.rgrp_trig_mode,
            execution_mode=self.rgrp_execution_mode,
            dtype=self.dtype,
        ).validate()


def jax_available() -> bool:
    return jax is not None and jnp is not None


def require_jax() -> None:
    if not jax_available():
        raise RuntimeError(
            "JAX is required for the JAX LM smoke; install JAX locally or run this on the TPU VM."
        )


def dtype_from_config(config: JaxLmSmokeConfig) -> Any:
    require_jax()
    return jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32


def _normal(key: Array, shape: tuple[int, ...], dtype: Any, scale: float = 0.02) -> Array:
    return jax.random.normal(key, shape, dtype=dtype) * scale


def _zeros(shape: tuple[int, ...], dtype: Any) -> Array:
    return jnp.zeros(shape, dtype=dtype)


def init_params(key: Array, config: JaxLmSmokeConfig) -> Params:
    require_jax()
    config.validate()
    dtype = dtype_from_config(config)
    key, tok_key, pos_key, head_key = jax.random.split(key, 4)
    block_keys = jax.random.split(key, config.layers)
    params: Params = {
        "token_embedding": _normal(tok_key, (config.vocab_size, config.d_model), dtype),
        "position_embedding": _normal(pos_key, (config.seq_len, config.d_model), dtype),
        "lm_head": _normal(head_key, (config.d_model, config.vocab_size), dtype),
        "blocks": [],
        "final_norm_weight": jnp.ones((config.d_model,), dtype=dtype),
    }
    for block_key in block_keys:
        (
            qkv_key,
            out_key,
            mlp1_key,
            mlp2_key,
            rgrp_key,
        ) = jax.random.split(block_key, 5)
        block: Params = {
            "attn_norm_weight": jnp.ones((config.d_model,), dtype=dtype),
            "ffn_norm_weight": jnp.ones((config.d_model,), dtype=dtype),
            "qkv": {
                "kernel": _normal(qkv_key, (config.d_model, 3 * config.d_model), dtype),
                "bias": _zeros((3 * config.d_model,), dtype),
            },
            "attn_out": {
                "kernel": _normal(out_key, (config.d_model, config.d_model), dtype),
                "bias": _zeros((config.d_model,), dtype),
            },
        }
        if config.variant == "mlp":
            block["mlp"] = {
                "w1": _normal(mlp1_key, (config.d_model, config.d_ff), dtype),
                "b1": _zeros((config.d_ff,), dtype),
                "w2": _normal(mlp2_key, (config.d_ff, config.d_model), dtype),
                "b2": _zeros((config.d_model,), dtype),
            }
        else:
            block["rgrp"] = rgrp.init_params(
                rgrp_key,
                rgrp.RotaryGatedRecurrentStateUpdateConfig(
                    d_model=config.d_model,
                    state_transform=config.rgrp_state_transform,
                    scan_unroll=config.rgrp_scan_unroll,
                    projection_mode=config.rgrp_projection_mode,
                    trig_mode=config.rgrp_trig_mode,
                    execution_mode=config.rgrp_execution_mode,
                    dtype=config.dtype,
                ),
            )
        params["blocks"].append(block)
    return params


def init_batch(key: Array, config: JaxLmSmokeConfig) -> tuple[Array, Array]:
    require_jax()
    tokens = jax.random.randint(
        key,
        (config.batch_size, config.seq_len + 1),
        minval=0,
        maxval=config.vocab_size,
        dtype=jnp.int32,
    )
    return tokens[:, :-1], tokens[:, 1:]


def count_params(params: Params) -> int:
    require_jax()
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(params)))


def rms_norm(hidden: Array, weight: Array, eps: float = 1.0e-5) -> Array:
    denom = jnp.sqrt(jnp.mean(hidden.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (hidden.astype(jnp.float32) / denom * weight.astype(jnp.float32)).astype(hidden.dtype)


def linear(hidden: Array, layer: Params) -> Array:
    return jnp.einsum("btd,dk->btk", hidden, layer["kernel"]) + layer["bias"]


def causal_attention(hidden: Array, block: Params, config: JaxLmSmokeConfig) -> Array:
    qkv = linear(hidden, block["qkv"])
    query, key, value = jnp.split(qkv, 3, axis=-1)
    batch_size, seq_len, _ = hidden.shape
    query = query.reshape(batch_size, seq_len, config.heads, config.head_dim).transpose(0, 2, 1, 3)
    key = key.reshape(batch_size, seq_len, config.heads, config.head_dim).transpose(0, 2, 1, 3)
    value = value.reshape(batch_size, seq_len, config.heads, config.head_dim).transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhtd,bhsd->bhts", query, key).astype(jnp.float32)
    scores = scores / np.sqrt(float(config.head_dim))
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    scores = jnp.where(mask[None, None, :, :], scores, -1.0e9)
    weights = jax.nn.softmax(scores, axis=-1).astype(hidden.dtype)
    attended = jnp.einsum("bhts,bhsd->bhtd", weights, value)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, config.d_model)
    return linear(attended, block["attn_out"])


def mlp(hidden: Array, block: Params) -> Array:
    mlp_params = block["mlp"]
    expanded = jnp.einsum("btd,df->btf", hidden, mlp_params["w1"]) + mlp_params["b1"]
    activated = jax.nn.gelu(expanded)
    return jnp.einsum("btf,fd->btd", activated, mlp_params["w2"]) + mlp_params["b2"]


def rgrp_ffn(hidden: Array, block: Params, config: JaxLmSmokeConfig) -> Array:
    rgrp_config = rgrp.RotaryGatedRecurrentStateUpdateConfig(
        d_model=config.d_model,
        state_transform=config.rgrp_state_transform,
        scan_unroll=config.rgrp_scan_unroll,
        projection_mode=config.rgrp_projection_mode,
        trig_mode=config.rgrp_trig_mode,
        execution_mode=config.rgrp_execution_mode,
        dtype=config.dtype,
    )
    outputs, _final_state = rgrp.scan(block["rgrp"], hidden, rgrp_config)
    return outputs


def forward(params: Params, input_ids: Array, config: JaxLmSmokeConfig) -> Array:
    positions = jnp.arange(config.seq_len)
    hidden = params["token_embedding"][input_ids] + params["position_embedding"][positions][None, :, :]
    for block in params["blocks"]:
        hidden = hidden + causal_attention(rms_norm(hidden, block["attn_norm_weight"]), block, config)
        ffn_input = rms_norm(hidden, block["ffn_norm_weight"])
        if config.variant == "mlp":
            hidden = hidden + mlp(ffn_input, block)
        else:
            hidden = hidden + rgrp_ffn(ffn_input, block, config)
    hidden = rms_norm(hidden, params["final_norm_weight"])
    return jnp.einsum("btd,dv->btv", hidden, params["lm_head"]).astype(jnp.float32)


def loss_fn(params: Params, input_ids: Array, target_ids: Array, config: JaxLmSmokeConfig) -> Array:
    logits = forward(params, input_ids, config)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_loss = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
    return jnp.mean(token_loss)


def benchmark_lm(
    *,
    config: JaxLmSmokeConfig,
    seed: int = 42,
    warmup: int = 1,
    iterations: int = 3,
    forward_only: bool = False,
) -> dict[str, Any]:
    require_jax()
    config.validate()
    key = jax.random.PRNGKey(seed)
    param_key, batch_key = jax.random.split(key)
    params = init_params(param_key, config)
    input_ids, target_ids = init_batch(batch_key, config)
    param_count = count_params(params)
    token_count = config.batch_size * config.seq_len
    if config.variant == "rgrp" and config.rgrp_execution_mode == "pallas-forward" and not forward_only:
        raise RuntimeError("rgrp_execution_mode='pallas-forward' is forward-only; rerun with --forward-only")

    if forward_only:
        compiled = jax.jit(lambda current_params: loss_fn(current_params, input_ids, target_ids, config))

        def run_once() -> Array:
            return compiled(params)
    else:
        compiled = jax.jit(jax.value_and_grad(lambda current_params: loss_fn(current_params, input_ids, target_ids, config)))

        def run_once() -> Array:
            loss, grads = compiled(params)
            leaves = jax.tree.leaves(grads)
            checksum = loss + sum(jnp.sum(leaf.astype(jnp.float32)) * 0.0 for leaf in leaves)
            return checksum

    compile_start = perf_counter()
    first = run_once()
    first.block_until_ready()
    compile_seconds = perf_counter() - compile_start

    for _ in range(warmup):
        value = run_once()
        value.block_until_ready()

    run_start = perf_counter()
    for _ in range(iterations):
        value = run_once()
        value.block_until_ready()
    run_seconds = perf_counter() - run_start

    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "variant": config.variant,
        "vocab_size": config.vocab_size,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "d_model": config.d_model,
        "layers": config.layers,
        "heads": config.heads,
        "ffn_multiplier": config.ffn_multiplier,
        "rgrp_state_transform": config.rgrp_state_transform if config.variant == "rgrp" else None,
        "rgrp_scan_unroll": config.rgrp_scan_unroll if config.variant == "rgrp" else None,
        "rgrp_projection_mode": config.rgrp_projection_mode if config.variant == "rgrp" else None,
        "rgrp_trig_mode": config.rgrp_trig_mode if config.variant == "rgrp" else None,
        "rgrp_execution_mode": config.rgrp_execution_mode if config.variant == "rgrp" else None,
        "dtype": config.dtype,
        "forward_only": forward_only,
        "parameter_count": param_count,
        "warmup": warmup,
        "iterations": iterations,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "tokens_per_iteration": token_count,
        "steady_state_tokens_per_second": token_count * iterations / run_seconds,
        "loss": float(first),
    }
