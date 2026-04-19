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
SUPPORTED_PROJECTION_MODES = ("sequence", "scan")
SUPPORTED_TRIG_MODES = ("precompute", "scan")
SUPPORTED_EXECUTION_MODES = ("scan", "pallas-forward")

try:  # Keep normal repo imports/tests working when JAX is not installed locally.
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised in local envs without JAX.
    jax = None
    jnp = None

try:  # Pallas is optional and only exercised on accelerator smoke runs.
    from jax.experimental import pallas as pl
except (ImportError, ModuleNotFoundError):  # pragma: no cover - local envs may not have JAX/Pallas.
    pl = None


Array = Any
Params = dict[str, Array]


@dataclass(frozen=True)
class RotaryGatedRecurrentStateUpdateConfig:
    d_model: int
    state_transform: str = "block-diagonal-4"
    dtype: str = "bfloat16"
    scan_unroll: int = 1
    projection_mode: str = "sequence"
    trig_mode: str = "precompute"
    execution_mode: str = "scan"

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
        if self.scan_unroll <= 0:
            raise ValueError(f"scan_unroll must be positive, got {self.scan_unroll}")
        if self.projection_mode not in SUPPORTED_PROJECTION_MODES:
            raise ValueError(
                f"projection_mode must be one of {SUPPORTED_PROJECTION_MODES}, got {self.projection_mode!r}"
            )
        if self.trig_mode not in SUPPORTED_TRIG_MODES:
            raise ValueError(f"trig_mode must be one of {SUPPORTED_TRIG_MODES}, got {self.trig_mode!r}")
        if self.execution_mode not in SUPPORTED_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {SUPPORTED_EXECUTION_MODES}, got {self.execution_mode!r}"
            )
        if self.projection_mode == "scan" and self.trig_mode != "scan":
            raise ValueError("projection_mode='scan' requires trig_mode='scan'")
        if self.execution_mode == "pallas-forward":
            if self.projection_mode != "sequence":
                raise ValueError("execution_mode='pallas-forward' requires projection_mode='sequence'")
            if self.trig_mode != "precompute":
                raise ValueError("execution_mode='pallas-forward' requires trig_mode='precompute'")
            if self.is_block_transform and not self.stores_block_transform_as_dense:
                raise ValueError(
                    "execution_mode='pallas-forward' requires dense-shaped state transform storage"
                )

    @property
    def is_block_transform(self) -> bool:
        return self.state_transform.startswith("block-diagonal")

    @property
    def stores_block_transform_as_dense(self) -> bool:
        return self.state_transform.endswith("-masked-dense")


def jax_available() -> bool:
    return jax is not None and jnp is not None


def pallas_available() -> bool:
    return pl is not None


def require_jax() -> None:
    if not jax_available():
        raise RuntimeError(
            "JAX is required for the rotary gated recurrent state update adapter; "
            "install jax locally or run this adapter on the TPU VM environment."
        )


def require_pallas() -> None:
    require_jax()
    if not pallas_available():
        raise RuntimeError(
            "Pallas is required for execution_mode='pallas-forward'; "
            "install a JAX build with jax.experimental.pallas or use execution_mode='scan'."
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
    return controls_from_packed(config, packed)


def controls_from_packed(
    config: RotaryGatedRecurrentStateUpdateConfig,
    packed: Array,
) -> dict[str, Array]:
    """Decode packed projection streams into recurrent update controls."""

    require_jax()
    update_inputs, angle_inputs, candidate_inputs, output_inputs = split_packed_projection(config, packed)
    update_gates = jax.nn.sigmoid(update_inputs)
    controls = {
        "update_gates": update_gates,
        "retain_gates": 1.0 - update_gates,
        "candidates": jnp.tanh(candidate_inputs),
        "output_gates": jax.nn.sigmoid(output_inputs),
    }
    if config.trig_mode == "precompute":
        controls["angle_cos"] = jnp.cos(angle_inputs)
        controls["angle_sin"] = jnp.sin(angle_inputs)
    else:
        controls["angle_inputs"] = angle_inputs
    return controls


def project_step_controls(
    params: Params,
    token_inputs: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
) -> dict[str, Array]:
    """Project one timestep inside the scan body for lowering ablations."""

    require_jax()
    projection = params["in_projection"]
    packed = jnp.matmul(token_inputs, projection["kernel"]) + projection["bias"]
    return controls_from_packed(config, packed)


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


def rotate_state_pairs_with_angles(state: Array, angle_inputs: Array) -> Array:
    require_jax()
    return rotate_state_pairs_with_trig(state, jnp.cos(angle_inputs), jnp.sin(angle_inputs))


def pallas_forward_scan(
    params: Params,
    inputs: Array,
    config: RotaryGatedRecurrentStateUpdateConfig,
    *,
    initial_state: Array | None = None,
) -> tuple[Array, Array]:
    """Run the recurrent forward pass through a single-batch Pallas program.

    This intentionally keeps the sequence-wide packed projection outside the
    kernel. The first Pallas question is narrower: can a custom recurrent loop
    lower better than ``jax.lax.scan`` when the TPU-favorable controls are
    already materialized?
    """

    require_pallas()
    config.validate()
    if initial_state is not None:
        raise ValueError("pallas-forward does not yet support non-zero initial_state")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [batch, seq, d_model], got {inputs.shape}")
    batch_size, seq_len, d_model = inputs.shape
    if d_model != config.d_model:
        raise ValueError(f"inputs last dim must be {config.d_model}, got {d_model}")
    if seq_len % 8 != 0:
        raise ValueError(f"pallas-forward requires seq_len divisible by 8 on TPU, got {seq_len}")
    if d_model % 128 != 0:
        raise ValueError(f"pallas-forward requires d_model divisible by 128 on TPU, got {d_model}")

    plan = prepare_runtime_plan(params, inputs, config)
    transform = params["state_transform"]
    transform_kernel = transform["kernel"]
    kernel_first_to_first = transform_kernel[0::2, 0::2]
    kernel_second_to_first = transform_kernel[1::2, 0::2]
    kernel_first_to_second = transform_kernel[0::2, 1::2]
    kernel_second_to_second = transform_kernel[1::2, 1::2]
    transform_bias_first = transform["bias"][0::2][None, :]
    transform_bias_second = transform["bias"][1::2][None, :]
    pair_count = config.angle_width
    update_first = plan["update_gates"][:, :, 0::2]
    update_second = plan["update_gates"][:, :, 1::2]
    retain_first = plan["retain_gates"][:, :, 0::2]
    retain_second = plan["retain_gates"][:, :, 1::2]
    candidate_first = plan["candidates"][:, :, 0::2]
    candidate_second = plan["candidates"][:, :, 1::2]
    output_gate_first = plan["output_gates"][:, :, 0::2]
    output_gate_second = plan["output_gates"][:, :, 1::2]

    def kernel(
        update_first_ref: Any,
        update_second_ref: Any,
        retain_first_ref: Any,
        retain_second_ref: Any,
        angle_cos_ref: Any,
        angle_sin_ref: Any,
        candidate_first_ref: Any,
        candidate_second_ref: Any,
        output_gate_first_ref: Any,
        output_gate_second_ref: Any,
        kernel_first_to_first_ref: Any,
        kernel_second_to_first_ref: Any,
        kernel_first_to_second_ref: Any,
        kernel_second_to_second_ref: Any,
        state_bias_first_ref: Any,
        state_bias_second_ref: Any,
        outputs_first_ref: Any,
        outputs_second_ref: Any,
        final_state_first_ref: Any,
        final_state_second_ref: Any,
    ) -> None:
        state_first = jnp.zeros((pair_count,), dtype=jnp.float32)
        state_second = jnp.zeros((pair_count,), dtype=jnp.float32)
        kernel_first_to_first = kernel_first_to_first_ref[...].astype(jnp.float32)
        kernel_second_to_first = kernel_second_to_first_ref[...].astype(jnp.float32)
        kernel_first_to_second = kernel_first_to_second_ref[...].astype(jnp.float32)
        kernel_second_to_second = kernel_second_to_second_ref[...].astype(jnp.float32)
        state_bias_first = state_bias_first_ref[0, :].astype(jnp.float32)
        state_bias_second = state_bias_second_ref[0, :].astype(jnp.float32)
        for timestep in range(seq_len):
            first = (
                (state_first[None, :] @ kernel_first_to_first)[0, :]
                + (state_second[None, :] @ kernel_second_to_first)[0, :]
                + state_bias_first
            )
            second = (
                (state_first[None, :] @ kernel_first_to_second)[0, :]
                + (state_second[None, :] @ kernel_second_to_second)[0, :]
                + state_bias_second
            )
            cos = angle_cos_ref[0, timestep, :].astype(jnp.float32)
            sin = angle_sin_ref[0, timestep, :].astype(jnp.float32)
            transformed_first = first * cos - second * sin
            transformed_second = first * sin + second * cos
            next_first = (
                update_first_ref[0, timestep, :].astype(jnp.float32) * transformed_first
                + retain_first_ref[0, timestep, :].astype(jnp.float32)
                * candidate_first_ref[0, timestep, :].astype(jnp.float32)
            )
            next_second = (
                update_second_ref[0, timestep, :].astype(jnp.float32) * transformed_second
                + retain_second_ref[0, timestep, :].astype(jnp.float32)
                * candidate_second_ref[0, timestep, :].astype(jnp.float32)
            )
            emitted_first = output_gate_first_ref[0, timestep, :].astype(jnp.float32) * next_first
            emitted_second = output_gate_second_ref[0, timestep, :].astype(jnp.float32) * next_second
            outputs_first_ref[0, timestep, :] = emitted_first.astype(outputs_first_ref.dtype)
            outputs_second_ref[0, timestep, :] = emitted_second.astype(outputs_second_ref.dtype)
            state_first = next_first
            state_second = next_second
        final_state_first_ref[0, 0, :] = state_first.astype(final_state_first_ref.dtype)
        final_state_second_ref[0, 0, :] = state_second.astype(final_state_second_ref.dtype)

    batch_time_angle = (1, seq_len, pair_count)
    outputs_first, outputs_second, final_state_first, final_state_second = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((batch_size, seq_len, pair_count), inputs.dtype),
            jax.ShapeDtypeStruct((batch_size, seq_len, pair_count), inputs.dtype),
            jax.ShapeDtypeStruct((batch_size, 1, pair_count), inputs.dtype),
            jax.ShapeDtypeStruct((batch_size, 1, pair_count), inputs.dtype),
        ),
        grid=(batch_size,),
        in_specs=[
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec((pair_count, pair_count), lambda batch: (0, 0)),
            pl.BlockSpec((pair_count, pair_count), lambda batch: (0, 0)),
            pl.BlockSpec((pair_count, pair_count), lambda batch: (0, 0)),
            pl.BlockSpec((pair_count, pair_count), lambda batch: (0, 0)),
            pl.BlockSpec((1, pair_count), lambda batch: (0, 0)),
            pl.BlockSpec((1, pair_count), lambda batch: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec(batch_time_angle, lambda batch: (batch, 0, 0)),
            pl.BlockSpec((1, 1, pair_count), lambda batch: (batch, 0, 0)),
            pl.BlockSpec((1, 1, pair_count), lambda batch: (batch, 0, 0)),
        ],
        name="rgrp_forward_scan",
    )(
        update_first,
        update_second,
        retain_first,
        retain_second,
        plan["angle_cos"],
        plan["angle_sin"],
        candidate_first,
        candidate_second,
        output_gate_first,
        output_gate_second,
        kernel_first_to_first,
        kernel_second_to_first,
        kernel_first_to_second,
        kernel_second_to_second,
        transform_bias_first,
        transform_bias_second,
    )
    outputs = jnp.stack((outputs_first, outputs_second), axis=-1).reshape((batch_size, seq_len, d_model))
    final_state = jnp.stack(
        (final_state_first[:, 0, :], final_state_second[:, 0, :]),
        axis=-1,
    ).reshape((batch_size, d_model))
    return outputs, final_state


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
    if config.execution_mode == "pallas-forward":
        return pallas_forward_scan(params, inputs, config, initial_state=initial_state)

    state = (
        jnp.zeros((inputs.shape[0], config.d_model), dtype=inputs.dtype)
        if initial_state is None
        else initial_state
    )
    if config.projection_mode == "sequence":
        plan = prepare_runtime_plan(params, inputs, config)
        time_major_plan: dict[str, Array] | Array = {
            key: jnp.swapaxes(value, 0, 1)
            for key, value in plan.items()
        }
    else:
        time_major_plan = jnp.swapaxes(inputs, 0, 1)

    def step(carry: Array, slices: dict[str, Array] | Array) -> tuple[Array, Array]:
        controls = (
            project_step_controls(params, slices, config)
            if config.projection_mode == "scan"
            else slices
        )
        projected_state = apply_state_transform(params, carry, config)
        transformed_state = (
            rotate_state_pairs_with_trig(
                projected_state,
                controls["angle_cos"],
                controls["angle_sin"],
            )
            if config.trig_mode == "precompute"
            else rotate_state_pairs_with_angles(projected_state, controls["angle_inputs"])
        )
        next_state = (
            controls["update_gates"] * transformed_state
            + controls["retain_gates"] * controls["candidates"]
        )
        emitted = controls["output_gates"] * next_state
        return next_state, emitted

    final_state, outputs_time_major = jax.lax.scan(
        step,
        state,
        time_major_plan,
        unroll=config.scan_unroll,
    )
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
    scan_unroll: int = 1,
    projection_mode: str = "sequence",
    trig_mode: str = "precompute",
    execution_mode: str = "scan",
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
        scan_unroll=scan_unroll,
        projection_mode=projection_mode,
        trig_mode=trig_mode,
        execution_mode=execution_mode,
    )
    config.validate()
    if include_grad and execution_mode == "pallas-forward":
        raise RuntimeError("execution_mode='pallas-forward' is forward-only; rerun with include_grad=false")
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
        "scan_unroll": scan_unroll,
        "projection_mode": projection_mode,
        "trig_mode": trig_mode,
        "execution_mode": execution_mode,
        "include_grad": include_grad,
        "warmup": warmup,
        "iterations": iterations,
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "tokens_per_iteration": token_count,
        "steady_state_tokens_per_second": token_count * iterations / run_seconds,
    }
