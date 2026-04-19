"""JAX reference adapter for the rotary gated recurrent state update primitive.

This module is intentionally small and dependency-light.  It mirrors the
current PyTorch ``P20RotaryStateOutputRuntimeSequenceMixer`` contract:

1. one packed input projection creates update gates, rotary angles, candidates,
   and output gates for the full sequence;
2. a recurrent state is transformed, rotated pairwise, and blended with the
   candidate at each token;
3. the emitted output is the updated state gated by the output gate.

The first TPU port should use this ``jax.lax.scan`` reference before we spend
time on Pallas or a MaxText-native fused kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np


ADAPTER_NAME = "rotary-gated-recurrent-state-update"
SUPPORTED_STATE_TRANSFORMS = (
    "dense",
    "block-diagonal-2",
    "block-diagonal-4",
    "block-diagonal-2-masked-dense",
    "block-diagonal-4-masked-dense",
)

try:  # Keep normal repo imports/tests working when JAX is not installed locally.
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised in local envs without JAX.
    jax = None
    jnp = None


Array = Any
Params = dict[str, Array]


@dataclass(frozen=True)
class RotaryGatedRecurrentStateUpdateConfig:
    d_model: int
    state_transform: str = "block-diagonal-4"
    dtype: str = "bfloat16"

    @property
    def angle_width(self) -> int:
        return self.d_model // 2

    @property
    def packed_width(self) -> int:
        return self.d_model + self.angle_width + self.d_model + self.d_model

    @property
    def block_count(self) -> int:
        if self.state_transform in {"block-diagonal-2", "block-diagonal-2-masked-dense"}:
            return 2
        if self.state_transform in {"block-diagonal-4", "block-diagonal-4-masked-dense"}:
            return 4
        raise ValueError(f"state_transform {self.state_transform!r} is not block diagonal")

    def validate(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model must be even for rotary pairs, got {self.d_model}")
        if self.state_transform not in SUPPORTED_STATE_TRANSFORMS:
            raise ValueError(
                f"state_transform must be one of {SUPPORTED_STATE_TRANSFORMS}, got {self.state_transform!r}"
            )
        if self.is_block_transform and self.d_model % self.block_count != 0:
            raise ValueError(
                f"d_model {self.d_model} must be divisible by block count {self.block_count}"
            )
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError(f"dtype must be bfloat16 or float32, got {self.dtype!r}")

    @property
    def is_block_transform(self) -> bool:
        return self.state_transform.startswith("block-diagonal")

    @property
    def stores_block_transform_as_dense(self) -> bool:
        return self.state_transform.endswith("-masked-dense")


def jax_available() -> bool:
    return jax is not None and jnp is not None


def require_jax() -> None:
    if not jax_available():
        raise RuntimeError(
            "JAX is required for the rotary gated recurrent state update adapter; "
            "install jax locally or run this adapter on the TPU VM environment."
        )


def dtype_from_config(config: RotaryGatedRecurrentStateUpdateConfig) -> Any:
    require_jax()
    return jnp.bfloat16 if config.dtype == "bfloat16" else jnp.float32


def _glorot_uniform(key: Array, shape: tuple[int, ...], dtype: Any) -> Array:
    fan_in = shape[-2] if len(shape) >= 2 else shape[0]
    fan_out = shape[-1]
    limit = np.sqrt(6.0 / float(fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit, dtype=dtype)


def _block_kernel_to_dense(block_kernel: Array) -> Array:
    """Materialize a block-diagonal dense kernel from grouped block weights."""

    require_jax()
    blocks, block_width, _ = block_kernel.shape
    zero = jnp.zeros((block_width, block_width), dtype=block_kernel.dtype)
    rows = []
    for row_index in range(blocks):
        cols = []
        for col_index in range(blocks):
            cols.append(block_kernel[row_index] if row_index == col_index else zero)
        rows.append(jnp.concatenate(cols, axis=1))
    return jnp.concatenate(rows, axis=0)


def init_params(
    key: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> Params:
    """Initialize adapter params using the same typed state-transform contract."""

    require_jax()
    config.validate()
    dtype = dtype_from_config(config)
    in_key, transform_key = jax.random.split(key, 2)
    params: Params = {
        "in_projection": {
            "kernel": _glorot_uniform(in_key, (config.d_model, config.packed_width), dtype),
            "bias": jnp.zeros((config.packed_width,), dtype=dtype),
        }
    }
    if config.state_transform == "dense":
        params["state_transform"] = {
            "kernel": _glorot_uniform(transform_key, (config.d_model, config.d_model), dtype),
            "bias": jnp.zeros((config.d_model,), dtype=dtype),
        }
    else:
        block_width = config.d_model // config.block_count
        block_keys = jax.random.split(transform_key, config.block_count)
        block_kernel = jnp.stack(
            [
                _glorot_uniform(block_key, (block_width, block_width), dtype)
                for block_key in block_keys
            ],
            axis=0,
        )
        kernel = _block_kernel_to_dense(block_kernel) if config.stores_block_transform_as_dense else block_kernel
        params["state_transform"] = {
            "kernel": kernel,
            "bias": jnp.zeros((config.d_model,), dtype=dtype),
        }
    return params


def params_from_torch_state_dict(
    state_dict: Mapping[str, Any],
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> Params:
    """Convert a PyTorch RGRP/P20 primitive state dict into JAX params.

    PyTorch linear layers store weights as ``[out, in]``.  The JAX adapter stores
    dense kernels as ``[in, out]`` so sequence projections can use ``x @ kernel``.
    """

    require_jax()
    config.validate()
    dtype = dtype_from_config(config)

    in_weight = np.asarray(state_dict["in_projection.projection.weight"])
    in_bias = np.asarray(state_dict["in_projection.projection.bias"])
    params: Params = {
        "in_projection": {
            "kernel": jnp.asarray(in_weight.T, dtype=dtype),
            "bias": jnp.asarray(in_bias, dtype=dtype),
        }
    }

    if config.state_transform == "dense":
        transform_weight = np.asarray(state_dict["state_transform_projection.weight"])
        transform_bias = np.asarray(state_dict["state_transform_projection.bias"])
        params["state_transform"] = {
            "kernel": jnp.asarray(transform_weight.T, dtype=dtype),
            "bias": jnp.asarray(transform_bias, dtype=dtype),
        }
    else:
        block_kernel = jnp.asarray(state_dict["state_transform_projection.weight"], dtype=dtype)
        kernel = _block_kernel_to_dense(block_kernel) if config.stores_block_transform_as_dense else block_kernel
        params["state_transform"] = {
            "kernel": kernel,
            "bias": jnp.asarray(state_dict["state_transform_projection.bias"], dtype=dtype),
        }
    return params


def split_packed_projection(config: RotaryGatedRecurrentStateUpdateConfig, packed: Array) -> tuple[Array, ...]:
    require_jax()
    d_model = config.d_model
    angle_width = config.angle_width
    return tuple(
        jnp.split(
            packed,
            (d_model, d_model + angle_width, d_model + angle_width + d_model),
            axis=-1,
        )
    )


def prepare_runtime_plan(
    params: Params,
    inputs: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> dict[str, Array]:
    """Project a full sequence into the packed recurrent control streams."""

    require_jax()
    config.validate()
    projection = params["in_projection"]
    packed = jnp.einsum("btd,dk->btk", inputs, projection["kernel"]) + projection["bias"]
    update_inputs, angle_inputs, candidate_inputs, output_inputs = split_packed_projection(config, packed)
    update_gates = jax.nn.sigmoid(update_inputs)
    return {
        "update_gates": update_gates,
        "retain_gates": 1.0 - update_gates,
        "angle_cos": jnp.cos(angle_inputs),
        "angle_sin": jnp.sin(angle_inputs),
        "candidates": jnp.tanh(candidate_inputs),
        "output_gates": jax.nn.sigmoid(output_inputs),
    }


def apply_state_transform(
    params: Params,
    state: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> Array:
    require_jax()
    transform = params["state_transform"]
    if config.state_transform == "dense" or config.stores_block_transform_as_dense:
        return jnp.matmul(state, transform["kernel"]) + transform["bias"]
    if config.state_transform.startswith("block-diagonal"):
        kernel = transform["kernel"]
        block_count = kernel.shape[0]
        block_width = kernel.shape[1]
        reshaped = state.reshape(state.shape[0], block_count, block_width)
        projected = jnp.einsum("bgi,goi->bgo", reshaped, kernel)
        return projected.reshape(state.shape) + transform["bias"]
    raise ValueError(f"unsupported state transform mode: {config.state_transform!r}")


def rotate_state_pairs_with_trig(state: Array, cos: Array, sin: Array) -> Array:
    require_jax()
    batch_size, width = state.shape
    pair_count = width // 2
    pairs = state.reshape(batch_size, pair_count, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    rotated_first = first * cos - second * sin
    rotated_second = first * sin + second * cos
    return jnp.stack((rotated_first, rotated_second), axis=-1).reshape(batch_size, width)


def scan(
    params: Params,
    inputs: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
    *,
    initial_state: Array | None = None,
) -> tuple[Array, Array]:
    """Run the causal sequence scan and return ``(emitted_outputs, final_state)``."""

    require_jax()
    config.validate()
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [batch, seq, d_model], got {inputs.shape}")
    if inputs.shape[-1] != config.d_model:
        raise ValueError(f"inputs last dim must be {config.d_model}, got {inputs.shape[-1]}")

    state = (
        jnp.zeros((inputs.shape[0], config.d_model), dtype=inputs.dtype)
        if initial_state is None
        else initial_state
    )
    plan = prepare_runtime_plan(params, inputs, config)
    time_major_plan = {
        key: jnp.swapaxes(value, 0, 1)
        for key, value in plan.items()
    }

    def step(carry: Array, slices: dict[str, Array]) -> tuple[Array, Array]:
        projected_state = apply_state_transform(params, carry, config)
        transformed_state = rotate_state_pairs_with_trig(
            projected_state,
            slices["angle_cos"],
            slices["angle_sin"],
        )
        next_state = (
            slices["update_gates"] * transformed_state
            + slices["retain_gates"] * slices["candidates"]
        )
        emitted = slices["output_gates"] * next_state
        return next_state, emitted

    final_state, outputs_time_major = jax.lax.scan(step, state, time_major_plan)
    return jnp.swapaxes(outputs_time_major, 0, 1), final_state


def smoke_loss(
    params: Params,
    inputs: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> Array:
    outputs, final_state = scan(params, inputs, config)
    return jnp.mean(outputs.astype(jnp.float32) ** 2) + 1.0e-4 * jnp.mean(final_state.astype(jnp.float32) ** 2)


def benchmark_scan(
    *,
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 512,
    state_transform: str = "block-diagonal-4",
    dtype: str = "bfloat16",
    seed: int = 42,
    warmup: int = 1,
    iterations: int = 5,
    include_grad: bool = True,
) -> dict[str, Any]:
    """Run a tiny JIT smoke benchmark for TPU/H100-vs-TPU cost probes."""

    require_jax()
    config = RotaryGatedRecurrentStateUpdateConfig(
        d_model=d_model,
        state_transform=state_transform,
        dtype=dtype,
    )
    config.validate()
    key = jax.random.PRNGKey(seed)
    param_key, input_key = jax.random.split(key)
    run_dtype = dtype_from_config(config)
    params = init_params(param_key, config)
    inputs = jax.random.normal(input_key, (batch_size, seq_len, d_model), dtype=run_dtype)

    if include_grad:
        compiled = jax.jit(jax.value_and_grad(lambda current_params: smoke_loss(current_params, inputs, config)))
        token_count = batch_size * seq_len

        def run_once() -> Array:
            loss, grads = compiled(params)
            leaves = jax.tree.leaves(grads)
            checksum = loss + sum(jnp.sum(leaf.astype(jnp.float32)) * 0.0 for leaf in leaves)
            return checksum
    else:
        compiled = jax.jit(lambda current_params: scan(current_params, inputs, config)[0])
        token_count = batch_size * seq_len

        def run_once() -> Array:
            return jnp.mean(compiled(params).astype(jnp.float32))

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
        "adapter": ADAPTER_NAME,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "state_transform": state_transform,
        "dtype": dtype,
        "include_grad": include_grad,
        "warmup": warmup,
        "iterations": iterations,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "tokens_per_iteration": token_count,
        "steady_state_tokens_per_second": token_count * iterations / run_seconds,
    }
