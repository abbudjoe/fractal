#!/usr/bin/env python3
"""Patch a MaxText checkout with the Fractal RGRP FFN-seam adapter.

The upstream MaxText command/config surface is intentionally left untouched in
this repository. This script applies the minimal source changes needed inside a
temporary MaxText checkout so ``fractal_candidate=rotary-gated-recurrent-state-update``
has an actual runtime implementation instead of being a paper contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


FRACTAL_RGRP_MODULE = '''# Copyright 2026
#
# Fractal local experiment patch for MaxText.

"""Rotary gated recurrent state update block for MaxText FFN seams.

This is an E2E-quality reference lane, not a fused kernel. It uses packed
sequence projections plus ``jax.lax.scan`` over the token axis.
"""

from __future__ import annotations

from typing import Any

from flax import linen as nn
import jax
import jax.numpy as jnp


SUPPORTED_STATE_TRANSFORMS = (
    "dense",
    "block-diagonal-2-masked-dense",
    "block-diagonal-4-masked-dense",
)

SUPPORTED_OUTPUT_INITS = ("zero", "xavier")
SUPPORTED_SIDECAR_TYPES = ("rgrp", "tiny-mlp", "tiny-glu", "binary-tree")
SUPPORTED_PARCAE_PROFILES = (
    "parcae-looped-attention",
    "parcae-bx-looped-attention",
    "parcae-rgrp-control-looped-attention",
    "parcae-p20-control-looped-attention",
)
SUPPORTED_PARCAE_LOOP_POLICIES = ("fixed", "per-sequence")
SUPPORTED_PARCAE_DEPTH_DISTRIBUTIONS = ("poisson",)
SUPPORTED_PARCAE_DISCRETIZATIONS = ("stable-exp", "zoh")
SUPPORTED_PARCAE_CONTROL_MODES = ("gate-value", "gate-only", "value-only")
SUPPORTED_PATH1_SCALE_PROFILES = (
    "causal-topk-route50-layer1",
    "mod-train-topc-route50-layer1",
    "fixed-looped-lm",
    "input-injected-looped-lm",
    "universal-transformer-act",
    "mor-expert-choice",
    "d3-route25-accel",
)
PARCAE_CONTROL_DIAGNOSTIC_INTERMEDIATE_PREFIX = "fractal_parcae_control_diagnostics/"
PATH1_DIAGNOSTIC_INTERMEDIATE_PREFIX = "fractal_path1_diagnostics/"
PARCAE_CONTROL_DIAGNOSTIC_BASE_METRICS = (
    "controller/control_norm_mean",
    "controller/control_norm_rms",
    "controller/gate_mean",
    "controller/gate_std",
    "controller/gate_min",
    "controller/gate_max",
    "controller/gate_saturation_low_frac",
    "controller/gate_saturation_high_frac",
    "controller/value_norm_mean",
    "controller/value_to_loop_input_norm_ratio",
    "controller/injection_delta_norm_mean",
    "controller/injection_delta_to_loop_input_ratio",
    "loop/steps_used_mean",
    "loop/early_exit_fraction",
    "stability/nan_or_inf_seen",
)


def _as_f32(value: jax.Array) -> jax.Array:
  return value.astype(jnp.float32)


def _global_l2_norm(value: jax.Array) -> jax.Array:
  return jnp.linalg.norm(_as_f32(value))


def _mean_l2_norm(value: jax.Array) -> jax.Array:
  return jnp.mean(jnp.linalg.norm(_as_f32(value), axis=-1))


def _rms(value: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.mean(jnp.square(_as_f32(value))))


def _safe_ratio(numerator: jax.Array, denominator: jax.Array) -> jax.Array:
  return numerator / jnp.maximum(denominator, jnp.asarray(1.0e-6, dtype=jnp.float32))


def _nan_or_inf_seen(*values: jax.Array) -> jax.Array:
  seen = jnp.asarray(False, dtype=jnp.bool_)
  for value in values:
    seen = seen | jnp.any(~jnp.isfinite(_as_f32(value)))
  return seen


def sow_parcae_control_diagnostic(module: nn.Module, metric_name: str, value: jax.Array) -> None:
  module.sow(
      "intermediates",
      PARCAE_CONTROL_DIAGNOSTIC_INTERMEDIATE_PREFIX + metric_name,
      jnp.asarray(value, dtype=jnp.float32),
  )


def collect_parcae_control_diagnostics(intermediate_outputs: dict[str, Any]) -> dict[str, jax.Array]:
  """Collect scalar Parcae/RGRP-control diagnostics from Flax intermediates.

  Flax stores each ``sow`` value as a tuple leaf. This collector searches by
  metric-key prefix, so it works for both scanned and unscanned module paths.
  """

  grouped: dict[str, list[jax.Array]] = {}
  for path, value in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
    metric_name = None
    for path_part in path:
      key = getattr(path_part, "key", None)
      if isinstance(key, str) and key.startswith(PARCAE_CONTROL_DIAGNOSTIC_INTERMEDIATE_PREFIX):
        metric_name = key.removeprefix(PARCAE_CONTROL_DIAGNOSTIC_INTERMEDIATE_PREFIX)
        break
    if metric_name is None:
      continue
    grouped.setdefault(metric_name, []).append(jnp.ravel(jnp.asarray(value, dtype=jnp.float32)))
  return {
      metric_name: jnp.mean(jnp.concatenate(values))
      for metric_name, values in grouped.items()
      if values
  }


def is_parcae_profile(candidate: str) -> bool:
  return candidate in SUPPORTED_PARCAE_PROFILES


def is_path1_scale_profile(candidate: str) -> bool:
  return candidate in SUPPORTED_PATH1_SCALE_PROFILES


def sow_path1_diagnostic(module: nn.Module, metric_name: str, value: jax.Array) -> None:
  module.sow(
      "intermediates",
      PATH1_DIAGNOSTIC_INTERMEDIATE_PREFIX + metric_name,
      jnp.asarray(value, dtype=jnp.float32),
  )


def collect_path1_diagnostics(intermediate_outputs: dict[str, Any]) -> dict[str, jax.Array]:
  grouped: dict[str, list[jax.Array]] = {}
  for path, value in jax.tree_util.tree_leaves_with_path(intermediate_outputs):
    metric_name = None
    for path_part in path:
      key = getattr(path_part, "key", None)
      if isinstance(key, str) and key.startswith(PATH1_DIAGNOSTIC_INTERMEDIATE_PREFIX):
        metric_name = key.removeprefix(PATH1_DIAGNOSTIC_INTERMEDIATE_PREFIX)
        break
    if metric_name is None:
      continue
    grouped.setdefault(metric_name, []).append(jnp.ravel(jnp.asarray(value, dtype=jnp.float32)))
  return {
      metric_name: jnp.mean(jnp.concatenate(values))
      for metric_name, values in grouped.items()
      if values
  }


def _validate_route_fraction(route_fraction: float) -> None:
  if route_fraction <= 0.0 or route_fraction > 1.0:
    raise ValueError(f"fractal_path1_route_fraction must be in (0, 1], got {route_fraction}")


def topc_capacity(seq_len: int, route_fraction: float) -> int:
  _validate_route_fraction(route_fraction)
  return max(1, min(seq_len, int(seq_len * route_fraction + 0.999999)))


def full_topc_indices(router_scores: jax.Array, route_fraction: float) -> jax.Array:
  if router_scores.ndim != 2:
    raise ValueError(f"router_scores must have shape [batch, seq], got {router_scores.shape}")
  capacity = topc_capacity(int(router_scores.shape[1]), route_fraction)
  _, indices = jax.lax.top_k(router_scores.astype(jnp.float32), capacity)
  return jnp.sort(indices, axis=-1)


def indices_to_mask(indices: jax.Array, seq_len: int) -> jax.Array:
  token_positions = jnp.arange(seq_len, dtype=indices.dtype)
  return jnp.any(indices[..., None] == token_positions.reshape(1, 1, seq_len), axis=1)


def full_topc_mask(router_scores: jax.Array, route_fraction: float) -> jax.Array:
  return indices_to_mask(full_topc_indices(router_scores, route_fraction), int(router_scores.shape[1]))


def causal_prefix_topk_mask(router_scores: jax.Array, route_fraction: float) -> jax.Array:
  """Prefix-top-k mask used by the decode-safe Path1 MoD approximation."""

  if router_scores.ndim != 2:
    raise ValueError(f"router_scores must have shape [batch, seq], got {router_scores.shape}")
  _validate_route_fraction(route_fraction)
  seq_len = int(router_scores.shape[1])
  positions = jnp.arange(seq_len, dtype=jnp.int32)
  current = router_scores[:, :, None].astype(jnp.float32)
  prefix = router_scores[:, None, :].astype(jnp.float32)
  query_pos = positions.reshape(1, seq_len, 1)
  key_pos = positions.reshape(1, 1, seq_len)
  in_prefix = key_pos <= query_pos
  better = prefix > current
  equal_earlier = (prefix == current) & (key_pos < query_pos)
  rank = jnp.sum((in_prefix & (better | equal_earlier)).astype(jnp.int32), axis=-1)
  capacity = jnp.maximum(
      1,
      jnp.ceil((positions + 1).astype(jnp.float32) * jnp.asarray(route_fraction, dtype=jnp.float32)).astype(jnp.int32),
  )
  return rank < capacity.reshape(1, seq_len)


def middle_loop_bounds(total_layers: int) -> tuple[int, int]:
  if total_layers <= 0:
    raise ValueError(f"total_layers must be positive, got {total_layers}")
  loop_width = max(1, total_layers // 3)
  start = max(0, (total_layers - loop_width) // 2)
  end = min(total_layers, start + loop_width)
  return start, end


def parcae_max_loop_count(loop_count: int, max_loop_count: int) -> int:
  if loop_count <= 0:
    raise ValueError(f"fractal_parcae_loop_count must be positive, got {loop_count}")
  if max_loop_count < 0:
    raise ValueError(f"fractal_parcae_max_loop_count must be non-negative, got {max_loop_count}")
  return loop_count if max_loop_count == 0 else max_loop_count


def parcae_loop_depths(
    *,
    cfg: Any,
    batch_size: int,
    deterministic: bool,
    rng: jax.Array | None,
) -> tuple[jax.Array, jax.Array, int]:
  """Return per-sequence forward and backward recurrence depths.

  The fixed path preserves the original proof-ladder behavior. The per-sequence
  path follows the Parcae training contract: sample a recurrence depth for each
  sequence, align all examples to finish on the final recurrent step, and limit
  gradients to the last ``mu_bwd`` active recurrent steps.
  """

  policy = str(cfg.fractal_parcae_loop_policy)
  if policy not in SUPPORTED_PARCAE_LOOP_POLICIES:
    raise ValueError(f"fractal_parcae_loop_policy must be one of {SUPPORTED_PARCAE_LOOP_POLICIES}, got {policy!r}")
  distribution = str(cfg.fractal_parcae_depth_distribution)
  if distribution not in SUPPORTED_PARCAE_DEPTH_DISTRIBUTIONS:
    raise ValueError(
        f"fractal_parcae_depth_distribution must be one of {SUPPORTED_PARCAE_DEPTH_DISTRIBUTIONS}, got {distribution!r}"
    )

  loop_count = int(cfg.fractal_parcae_loop_count)
  max_loop_count = parcae_max_loop_count(loop_count, int(cfg.fractal_parcae_max_loop_count))
  min_loop_count = int(cfg.fractal_parcae_min_loop_count)
  if min_loop_count <= 0:
    raise ValueError(f"fractal_parcae_min_loop_count must be positive, got {min_loop_count}")
  if min_loop_count > max_loop_count:
    raise ValueError(
        f"fractal_parcae_min_loop_count={min_loop_count} exceeds effective max loop count {max_loop_count}"
    )
  mu_rec = float(cfg.fractal_parcae_mu_rec)
  if mu_rec <= 0.0:
    raise ValueError(f"fractal_parcae_mu_rec must be positive, got {mu_rec}")
  mu_bwd = int(cfg.fractal_parcae_mu_bwd)
  if mu_bwd <= 0:
    raise ValueError(f"fractal_parcae_mu_bwd must be positive, got {mu_bwd}")

  fixed_depth = max(min(loop_count, max_loop_count), min_loop_count)
  if policy == "fixed" or deterministic:
    depths = jnp.full((batch_size,), fixed_depth, dtype=jnp.int32)
  else:
    if rng is None:
      raise ValueError("per-sequence Parcae loop policy requires a dropout RNG during training")
    if distribution == "poisson":
      # MaxText's TPU RNG backend does not support the native Poisson sampler,
      # so sample an explicitly clipped Poisson distribution from uniform random
      # numbers. Boundary buckets match the clipping semantics: values <= min
      # map to min and values >= max map to max.
      lam = jnp.asarray(mu_rec, dtype=jnp.float32)
      ks = jnp.arange(max_loop_count + 1, dtype=jnp.float32)
      pmf = jnp.exp(ks * jnp.log(lam) - lam - jax.lax.lgamma(ks + 1.0))
      buckets = []
      for depth in range(min_loop_count, max_loop_count + 1):
        if depth == min_loop_count:
          buckets.append(jnp.sum(pmf[: min_loop_count + 1]))
        elif depth == max_loop_count:
          buckets.append(jnp.maximum(1.0 - jnp.sum(pmf[:max_loop_count]), 0.0))
        else:
          buckets.append(pmf[depth])
      probs = jnp.asarray(buckets, dtype=jnp.float32)
      probs = probs / jnp.sum(probs)
      cdf = jnp.cumsum(probs)
      uniforms = jax.random.uniform(rng, shape=(batch_size,), dtype=jnp.float32)
      sampled = jnp.sum(uniforms[:, None] > cdf[None, :], axis=-1) + min_loop_count
    else:
      raise ValueError(f"unsupported Parcae depth distribution: {distribution!r}")
    depths = jnp.clip(sampled.astype(jnp.int32), min_loop_count, max_loop_count)

  bwd_depths = jnp.minimum(depths, jnp.asarray(min(mu_bwd, max_loop_count), dtype=jnp.int32))
  return depths, bwd_depths, max_loop_count


def parcae_depth_masks(
    depths: jax.Array,
    bwd_depths: jax.Array,
    loop_idx: int,
    max_loop_count: int,
) -> tuple[jax.Array, jax.Array]:
  step = jnp.asarray(loop_idx, dtype=jnp.int32)
  start = jnp.asarray(max_loop_count, dtype=jnp.int32) - depths
  grad_start = jnp.asarray(max_loop_count, dtype=jnp.int32) - bwd_depths
  active = step >= start
  with_grad = active & (step >= grad_start)
  return active.reshape((-1, 1, 1)), with_grad.reshape((-1, 1, 1))


def identity_kernel_init(key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
  del key
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError(f"identity initializer expects a square rank-2 shape, got {shape}")
  return jnp.eye(shape[0], dtype=dtype)


def _block_count(state_transform: str) -> int:
  if state_transform == "dense":
    return 1
  if state_transform.startswith("block-diagonal-2"):
    return 2
  if state_transform.startswith("block-diagonal-4"):
    return 4
  raise ValueError(f"unsupported fractal_rgrp_state_transform={state_transform!r}")


def _block_mask(width: int, block_count: int) -> jax.Array:
  if block_count == 1:
    return jnp.ones((width, width), dtype=jnp.float32)
  if width % block_count != 0:
    raise ValueError(f"RGRP width {width} must be divisible by block_count {block_count}")
  block_width = width // block_count
  block_ids = jnp.arange(width) // block_width
  return (block_ids[:, None] == block_ids[None, :]).astype(jnp.float32)


class RotaryGatedRecurrentStateUpdate(nn.Module):
  """A MaxText-local Linen adapter for the Fractal recurrent FFN seam."""

  d_model: int
  state_transform: str = "block-diagonal-4-masked-dense"
  scan_unroll: int = 3
  projection_mode: str = "sequence"
  trig_mode: str = "precompute"
  residual_scale: float = 1.0
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.d_model <= 0:
      raise ValueError(f"d_model must be positive, got {self.d_model}")
    if self.d_model % 2 != 0:
      raise ValueError(f"d_model must be even for rotary pairs, got {self.d_model}")
    if self.state_transform not in SUPPORTED_STATE_TRANSFORMS:
      raise ValueError(
          f"fractal_rgrp_state_transform must be one of {SUPPORTED_STATE_TRANSFORMS}, got {self.state_transform!r}"
      )
    if self.scan_unroll <= 0:
      raise ValueError(f"fractal_rgrp_scan_unroll must be positive, got {self.scan_unroll}")
    if self.projection_mode != "sequence":
      raise ValueError("MaxText RGRP quality lane currently supports fractal_rgrp_projection_mode='sequence' only")
    if self.trig_mode != "precompute":
      raise ValueError("MaxText RGRP quality lane currently supports fractal_rgrp_trig_mode='precompute' only")

  @property
  def angle_width(self) -> int:
    return self.d_model // 2

  @property
  def packed_width(self) -> int:
    return self.d_model + self.angle_width + self.d_model + self.d_model

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    in_kernel = self.param(
        "in_projection_kernel",
        nn.initializers.xavier_uniform(),
        (self.d_model, self.packed_width),
        self.weight_dtype,
    )
    in_bias = self.param("in_projection_bias", nn.initializers.zeros, (self.packed_width,), self.weight_dtype)
    state_kernel = self.param(
        "state_transform_kernel",
        nn.initializers.xavier_uniform(),
        (self.d_model, self.d_model),
        self.weight_dtype,
    )
    state_bias = self.param("state_transform_bias", nn.initializers.zeros, (self.d_model,), self.weight_dtype)

    inputs_dtype = inputs.dtype
    run_inputs = inputs.astype(self.dtype)
    packed = jnp.einsum(
        "btd,dk->btk",
        run_inputs,
        in_kernel.astype(self.dtype),
        precision=self.matmul_precision,
    ) + in_bias.astype(self.dtype)
    update_inputs, angle_inputs, candidate_inputs, output_inputs = jnp.split(
        packed,
        (
            self.d_model,
            self.d_model + self.angle_width,
            self.d_model + self.angle_width + self.d_model,
        ),
        axis=-1,
    )
    update_gates = jax.nn.sigmoid(update_inputs)
    retain_gates = 1.0 - update_gates
    candidates = jnp.tanh(candidate_inputs)
    output_gates = jax.nn.sigmoid(output_inputs)
    angle_cos = jnp.cos(angle_inputs)
    angle_sin = jnp.sin(angle_inputs)

    block_mask = _block_mask(self.d_model, _block_count(self.state_transform)).astype(jnp.float32)
    masked_state_kernel = state_kernel.astype(jnp.float32) * block_mask
    state_bias_f32 = state_bias.astype(jnp.float32)

    streams = {
        "update_gates": jnp.swapaxes(update_gates, 0, 1),
        "retain_gates": jnp.swapaxes(retain_gates, 0, 1),
        "candidates": jnp.swapaxes(candidates, 0, 1),
        "output_gates": jnp.swapaxes(output_gates, 0, 1),
        "angle_cos": jnp.swapaxes(angle_cos, 0, 1),
        "angle_sin": jnp.swapaxes(angle_sin, 0, 1),
    }
    initial_state = jnp.zeros((inputs.shape[0], self.d_model), dtype=jnp.float32)

    def step(state: jax.Array, controls: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
      projected = jnp.matmul(state, masked_state_kernel, precision=self.matmul_precision) + state_bias_f32
      pairs = projected.reshape(projected.shape[0], self.angle_width, 2)
      first = pairs[..., 0]
      second = pairs[..., 1]
      cos = controls["angle_cos"].astype(jnp.float32)
      sin = controls["angle_sin"].astype(jnp.float32)
      rotated = jnp.stack((first * cos - second * sin, first * sin + second * cos), axis=-1)
      transformed = rotated.reshape(projected.shape)
      next_state = (
          controls["update_gates"].astype(jnp.float32) * transformed
          + controls["retain_gates"].astype(jnp.float32) * controls["candidates"].astype(jnp.float32)
      )
      emitted = controls["output_gates"].astype(jnp.float32) * next_state
      return next_state, emitted.astype(inputs_dtype)

    _, outputs_time_major = jax.lax.scan(step, initial_state, streams, unroll=self.scan_unroll)
    outputs = jnp.swapaxes(outputs_time_major, 0, 1)
    return (outputs * jnp.asarray(self.residual_scale, dtype=outputs.dtype)).astype(inputs_dtype)


def rgrp_block(
    *,
    d_model: int,
    state_transform: str,
    scan_unroll: int,
    projection_mode: str,
    trig_mode: str,
    residual_scale: float,
    dtype: Any,
    weight_dtype: Any,
    matmul_precision: str,
    name: str | None = None,
) -> RotaryGatedRecurrentStateUpdate:
  return RotaryGatedRecurrentStateUpdate(
      d_model=d_model,
      state_transform=state_transform,
      scan_unroll=scan_unroll,
      projection_mode=projection_mode,
      trig_mode=trig_mode,
      residual_scale=residual_scale,
      dtype=dtype,
      weight_dtype=weight_dtype,
      matmul_precision=matmul_precision,
      name=name,
  )


class RGRPSidecar(nn.Module):
  """Small RGRP branch added beside, not instead of, the standard MLP."""

  d_model: int
  bottleneck_dim: int = 64
  state_transform: str = "block-diagonal-4-masked-dense"
  scan_unroll: int = 3
  projection_mode: str = "sequence"
  trig_mode: str = "precompute"
  side_scale: float = 0.1
  output_init: str = "xavier"
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.output_init not in SUPPORTED_OUTPUT_INITS:
      raise ValueError(f"fractal_rgrp_output_init must be one of {SUPPORTED_OUTPUT_INITS}, got {self.output_init!r}")
    if self.bottleneck_dim < 0:
      raise ValueError(f"fractal_rgrp_bottleneck_dim must be >= 0, got {self.bottleneck_dim}")
    width = self.bottleneck_dim or self.d_model
    if width <= 0:
      raise ValueError(f"RGRP sidecar width must be positive, got {width}")
    if width % 2 != 0:
      raise ValueError(f"RGRP sidecar width must be even for rotary pairs, got {width}")

  @property
  def width(self) -> int:
    return self.bottleneck_dim or self.d_model

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    if self.bottleneck_dim:
      side_inputs = nn.Dense(
          self.width,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.weight_dtype,
          precision=self.matmul_precision,
          kernel_init=nn.initializers.xavier_uniform(),
          name="down_projection",
      )(inputs)
    else:
      side_inputs = inputs

    side_outputs = RotaryGatedRecurrentStateUpdate(
        d_model=self.width,
        state_transform=self.state_transform,
        scan_unroll=self.scan_unroll,
        projection_mode=self.projection_mode,
        trig_mode=self.trig_mode,
        residual_scale=1.0,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        matmul_precision=self.matmul_precision,
        name="rgrp",
    )(side_inputs)

    kernel_init = nn.initializers.zeros if self.output_init == "zero" else nn.initializers.xavier_uniform()
    side_outputs = nn.Dense(
        self.d_model,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=kernel_init,
        name="up_projection",
    )(side_outputs)
    return (side_outputs * jnp.asarray(self.side_scale, dtype=side_outputs.dtype)).astype(inputs.dtype)


class TinyMLPSidecar(nn.Module):
  """Matched non-recurrent control branch for the RGRP sidecar contract."""

  d_model: int
  bottleneck_dim: int = 64
  side_scale: float = 0.1
  output_init: str = "xavier"
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.output_init not in SUPPORTED_OUTPUT_INITS:
      raise ValueError(f"fractal_rgrp_output_init must be one of {SUPPORTED_OUTPUT_INITS}, got {self.output_init!r}")
    if self.bottleneck_dim <= 0:
      raise ValueError(f"fractal_rgrp_bottleneck_dim must be positive for tiny-mlp, got {self.bottleneck_dim}")

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    hidden = nn.Dense(
        self.bottleneck_dim,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=nn.initializers.xavier_uniform(),
        name="down_projection",
    )(inputs)
    hidden = nn.gelu(hidden)
    hidden = nn.Dense(
        self.bottleneck_dim,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=nn.initializers.xavier_uniform(),
        name="inner_projection",
    )(hidden)
    hidden = nn.gelu(hidden)
    kernel_init = nn.initializers.zeros if self.output_init == "zero" else nn.initializers.xavier_uniform()
    outputs = nn.Dense(
        self.d_model,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=kernel_init,
        name="up_projection",
    )(hidden)
    return (outputs * jnp.asarray(self.side_scale, dtype=outputs.dtype)).astype(inputs.dtype)


class TinyGLUSidecar(nn.Module):
  """Matched gated control branch with no recurrent state carry."""

  d_model: int
  bottleneck_dim: int = 64
  side_scale: float = 0.1
  output_init: str = "xavier"
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.output_init not in SUPPORTED_OUTPUT_INITS:
      raise ValueError(f"fractal_rgrp_output_init must be one of {SUPPORTED_OUTPUT_INITS}, got {self.output_init!r}")
    if self.bottleneck_dim <= 0:
      raise ValueError(f"fractal_rgrp_bottleneck_dim must be positive for tiny-glu, got {self.bottleneck_dim}")

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    packed = nn.Dense(
        self.bottleneck_dim * 2,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=nn.initializers.xavier_uniform(),
        name="gate_value_projection",
    )(inputs)
    gate, value = jnp.split(packed, 2, axis=-1)
    hidden = jax.nn.sigmoid(gate) * nn.gelu(value)
    kernel_init = nn.initializers.zeros if self.output_init == "zero" else nn.initializers.xavier_uniform()
    outputs = nn.Dense(
        self.d_model,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=kernel_init,
        name="up_projection",
    )(hidden)
    return (outputs * jnp.asarray(self.side_scale, dtype=outputs.dtype)).astype(inputs.dtype)


class BinaryTreeSidecar(nn.Module):
  """Generic differentiable binary-tree expert control.

  This preserves the tree-shaped small-expert hypothesis without the recurrent
  state carry or rotary update law, so it is a control for structure rather than
  for RGRP-specific dynamics.
  """

  d_model: int
  bottleneck_dim: int = 64
  tree_depth: int = 2
  slot_count: int = 4
  side_scale: float = 0.1
  output_init: str = "xavier"
  dtype: Any = jnp.bfloat16
  weight_dtype: Any = jnp.float32
  matmul_precision: str = "default"

  def setup(self):
    if self.output_init not in SUPPORTED_OUTPUT_INITS:
      raise ValueError(f"fractal_rgrp_output_init must be one of {SUPPORTED_OUTPUT_INITS}, got {self.output_init!r}")
    if self.bottleneck_dim <= 0:
      raise ValueError(f"fractal_rgrp_bottleneck_dim must be positive for binary-tree, got {self.bottleneck_dim}")
    if self.tree_depth <= 0:
      raise ValueError(f"fractal_rgrp_tree_depth must be positive, got {self.tree_depth}")
    if self.slot_count <= 0:
      raise ValueError(f"fractal_rgrp_slot_count must be positive, got {self.slot_count}")

  @property
  def leaf_count(self) -> int:
    return 2 ** self.tree_depth

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    slots = nn.Dense(
        self.bottleneck_dim * self.slot_count,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=nn.initializers.xavier_uniform(),
        name="slot_projection",
    )(inputs)
    slots = nn.tanh(slots.reshape(*inputs.shape[:-1], self.slot_count, self.bottleneck_dim))
    constants = jnp.ones((*inputs.shape[:-1], 1, self.bottleneck_dim), dtype=slots.dtype)
    slot_bank = jnp.concatenate([slots, constants, -constants], axis=-2)
    selector_logits = self.param(
        "leaf_selector_logits",
        nn.initializers.zeros,
        (self.leaf_count, self.slot_count + 2),
        self.weight_dtype,
    )
    selector = jax.nn.softmax(selector_logits.astype(self.dtype), axis=-1)
    nodes = jnp.einsum("lc,btcw->bltw", selector, slot_bank, precision=self.matmul_precision)
    node_count = self.leaf_count
    level = 0
    while node_count > 1:
      left = nodes[:, 0::2, :, :]
      right = nodes[:, 1::2, :, :]
      gate = nn.Dense(
          self.bottleneck_dim,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.weight_dtype,
          precision=self.matmul_precision,
          kernel_init=nn.initializers.xavier_uniform(),
          name=f"node_gate_{level}",
      )(jnp.concatenate([left, right], axis=-1))
      mixed = jax.nn.sigmoid(gate) * left + (1.0 - jax.nn.sigmoid(gate)) * right
      interact = nn.Dense(
          self.bottleneck_dim,
          use_bias=True,
          dtype=self.dtype,
          param_dtype=self.weight_dtype,
          precision=self.matmul_precision,
          kernel_init=nn.initializers.xavier_uniform(),
          name=f"node_interact_{level}",
      )(left * right)
      nodes = nn.tanh(mixed + interact)
      node_count //= 2
      level += 1
    hidden = nodes[:, 0, :, :]
    kernel_init = nn.initializers.zeros if self.output_init == "zero" else nn.initializers.xavier_uniform()
    outputs = nn.Dense(
        self.d_model,
        use_bias=True,
        dtype=self.dtype,
        param_dtype=self.weight_dtype,
        precision=self.matmul_precision,
        kernel_init=kernel_init,
        name="up_projection",
    )(hidden)
    return (outputs * jnp.asarray(self.side_scale, dtype=outputs.dtype)).astype(inputs.dtype)


def sidecar_block(
    *,
    sidecar_type: str,
    d_model: int,
    bottleneck_dim: int,
    state_transform: str,
    scan_unroll: int,
    projection_mode: str,
    trig_mode: str,
    side_scale: float,
    output_init: str,
    dtype: Any,
    weight_dtype: Any,
    matmul_precision: str,
    tree_depth: int = 2,
    slot_count: int = 4,
    name: str | None = None,
) -> nn.Module:
  if sidecar_type == "rgrp":
    return RGRPSidecar(
      d_model=d_model,
      bottleneck_dim=bottleneck_dim,
      state_transform=state_transform,
      scan_unroll=scan_unroll,
      projection_mode=projection_mode,
      trig_mode=trig_mode,
      side_scale=side_scale,
      output_init=output_init,
      dtype=dtype,
      weight_dtype=weight_dtype,
      matmul_precision=matmul_precision,
      name=name,
    )
  if sidecar_type == "tiny-mlp":
    return TinyMLPSidecar(
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        side_scale=side_scale,
        output_init=output_init,
        dtype=dtype,
        weight_dtype=weight_dtype,
        matmul_precision=matmul_precision,
        name=name,
    )
  if sidecar_type == "tiny-glu":
    return TinyGLUSidecar(
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        side_scale=side_scale,
        output_init=output_init,
        dtype=dtype,
        weight_dtype=weight_dtype,
        matmul_precision=matmul_precision,
        name=name,
    )
  if sidecar_type == "binary-tree":
    return BinaryTreeSidecar(
        d_model=d_model,
        bottleneck_dim=bottleneck_dim,
        tree_depth=tree_depth,
        slot_count=slot_count,
        side_scale=side_scale,
        output_init=output_init,
        dtype=dtype,
        weight_dtype=weight_dtype,
        matmul_precision=matmul_precision,
        name=name,
    )
  raise ValueError(f"fractal_rgrp_sidecar_type must be one of {SUPPORTED_SIDECAR_TYPES}, got {sidecar_type!r}")


def layer_enabled(layer_spec: Any, layer_idx: int | None) -> bool:
  normalized = str(layer_spec).strip().lower()
  if normalized in ("", "none"):
    return False
  if normalized in ("*", "all"):
    return True
  if layer_idx is None:
    raise ValueError(
        "fractal_rgrp_layers selects specific layers, but MaxText is using scanned layers without a visible layer_idx. "
        "Set scan_layers=false for selected-layer RGRP sidecars."
    )
  enabled = {int(part.strip()) for part in normalized.split(",") if part.strip()}
  return layer_idx in enabled
'''


def replace_once(text: str, old: str, new: str, *, path: Path) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match in {path} for patch fragment; found {count}")
    return text.replace(old, new, 1)


def insert_after_once(text: str, marker: str, insert: str, *, path: Path) -> str:
    if insert in text:
        return text
    count = text.count(marker)
    if count != 1:
        raise RuntimeError(f"expected exactly one insertion marker in {path}; found {count}")
    return text.replace(marker, marker + insert, 1)


def resolve_package_dir(path: Path) -> Path:
    """Return the real MaxText import package directory for this checkout."""

    for package_dir in (path / "src" / "maxtext", path / "src" / "MaxText"):
        if (package_dir / "layers").is_dir() and (package_dir / "configs").is_dir():
            return package_dir
    raise RuntimeError(f"{path} does not look like a MaxText checkout; expected src/maxtext or src/MaxText")


def patch_types(package_dir: Path) -> None:
    types_path = package_dir / "configs" / "types.py"
    text = types_path.read_text()
    text = text.replace(
        '  fractal_rgrp_layers: str = Field("*", description="Comma-separated zero-indexed decoder layers for sidecar mode, or * for all.")\n',
        '  fractal_rgrp_layers: str | int = Field("*", description="Comma-separated zero-indexed decoder layers for sidecar mode, or * for all.")\n',
    )
    marker = '  mlp_activations: list[str] = Field(["silu", "linear"], description="Activation functions in the MLP layer.")\n'
    fields = [
        '  fractal_candidate: str = Field("", description="Fractal experimental adapter slug. Empty disables adapters.")\n',
        '  fractal_adapter_module: str = Field("", description="Fractal adapter module identifier used by external manifests.")\n',
        '  fractal_rgrp_integration_mode: str = Field("replace-mlp", description="RGRP integration mode: replace-mlp or mlp-sidecar.")\n',
        '  fractal_rgrp_layers: str | int = Field("*", description="Comma-separated zero-indexed decoder layers for sidecar mode, or * for all.")\n',
        '  fractal_rgrp_bottleneck_dim: int = Field(0, description="Optional sidecar bottleneck width. 0 keeps full hidden width.")\n',
        '  fractal_rgrp_state_transform: str = Field("block-diagonal-4-masked-dense", description="RGRP state transform contract.")\n',
        '  fractal_rgrp_scan_unroll: int = Field(3, description="RGRP lax.scan unroll factor.")\n',
        '  fractal_rgrp_projection_mode: str = Field("sequence", description="RGRP projection lowering mode.")\n',
        '  fractal_rgrp_trig_mode: str = Field("precompute", description="RGRP trigonometric lowering mode.")\n',
        '  fractal_rgrp_residual_scale: float = Field(1.0, description="Scale applied to the RGRP replacement branch.")\n',
        '  fractal_rgrp_sidecar_type: str = Field("rgrp", description="Selected MLP-sidecar operator: rgrp, tiny-mlp, tiny-glu, or binary-tree.")\n',
        '  fractal_rgrp_side_scale: float = Field(0.1, description="Scale applied to the RGRP MLP-sidecar branch.")\n',
        '  fractal_rgrp_output_init: str = Field("xavier", description="RGRP sidecar output projection init: zero or xavier.")\n',
        '  fractal_rgrp_tree_depth: int = Field(2, description="Binary-tree control sidecar depth.")\n',
        '  fractal_rgrp_slot_count: int = Field(4, description="Binary-tree control sidecar slot count.")\n',
        '  fractal_parcae_loop_count: int = Field(2, description="Number of recurrent passes through the Parcae middle layer band.")\n',
        '  fractal_parcae_loop_policy: str = Field("fixed", description="Parcae loop-depth policy: fixed or per-sequence.")\n',
        '  fractal_parcae_depth_distribution: str = Field("poisson", description="Per-sequence Parcae loop-depth distribution.")\n',
        '  fractal_parcae_mu_rec: float = Field(2.0, description="Mean forward recurrence depth for stochastic Parcae training.")\n',
        '  fractal_parcae_mu_bwd: int = Field(2, description="Number of final recurrent steps that receive gradients in stochastic Parcae training.")\n',
        '  fractal_parcae_min_loop_count: int = Field(1, description="Minimum sampled Parcae recurrence depth.")\n',
        '  fractal_parcae_max_loop_count: int = Field(0, description="Maximum sampled Parcae recurrence depth. 0 uses fractal_parcae_loop_count.")\n',
        '  fractal_parcae_discretization: str = Field("stable-exp", description="Parcae stable discretization: stable-exp or zoh.")\n',
        '  fractal_parcae_dt_raw_init: float = Field(0.54132485, description="Initial raw step size for ZOH Parcae discretization; softplus(raw)=1.0.")\n',
        '  fractal_parcae_control_diagnostics: bool = Field(False, description="Emit optional scalar diagnostics for the Parcae RGRP-control loop injection path.")\n',
        '  fractal_parcae_control_mode: str = Field("gate-value", description="Parcae RGRP-control mode: gate-value, gate-only, or value-only.")\n',
        '  fractal_parcae_control_bottleneck_dim: int = Field(0, description="Optional Parcae RGRP-control bottleneck width. 0 keeps full hidden width.")\n',
        '  fractal_parcae_control_gate_blend: float = Field(0.0, description="Blend coefficient from controller gate toward native Parcae base gate; 0 uses controller gate only.")\n',
        '  fractal_parcae_control_value_scale: float = Field(1.0, description="Scale applied to the Parcae RGRP-control value projection before loop injection.")\n',
        '  fractal_path1_route_fraction: float = Field(0.5, description="Path1 routed-token capacity fraction for TPU scale-leader ports.")\n',
        '  fractal_path1_route_layers: str | int = Field("1", description="Comma-separated zero-indexed decoder layers for Path1 routed-token ports.")\n',
        '  fractal_path1_loop_count: int = Field(4, description="Number of shared-block recurrent passes for Path1 looped ports.")\n',
        '  fractal_path1_shared_layers: int = Field(2, description="Number of shared decoder blocks inside Path1 looped ports.")\n',
        '  fractal_path1_accel_threshold: float = Field(0.6, description="Normalized acceleration threshold for Path1 recurrent-depth early exit ports.")\n',
        '  fractal_path1_min_steps: int = Field(2, description="Minimum recurrent steps before Path1 acceleration exit can halt.")\n',
        '  fractal_path1_act_threshold: float = Field(0.99, description="Universal Transformer ACT halt probability threshold for Path1 TPU ports.")\n',
        '  fractal_path1_diagnostics: bool = Field(False, description="Emit optional scalar diagnostics for Path1 TPU scale-leader ports.")\n',
    ]
    for field in reversed(fields):
        name = field.strip().split(":", 1)[0]
        field_decl = re.compile(rf"^\s*{re.escape(name)}\s*:", re.MULTILINE)
        if not field_decl.search(text):
            text = insert_after_once(text, marker, field, path=types_path)
    types_path.write_text(text)


def patch_decoders(package_dir: Path) -> None:
    decoders_path = package_dir / "layers" / "decoders.py"
    text = decoders_path.read_text()
    if "fractal_rgrp" not in text:
        text = replace_once(
            text,
            "from maxtext.layers import quantizations\n",
            "from maxtext.layers import quantizations\nfrom maxtext.layers import fractal_rgrp\n",
            path=decoders_path,
        )
    if "enable_fractal_rgrp_sidecar: bool = False" not in text:
        text = replace_once(
            text,
            "  quant: None | Quant = None\n\n  @nn.compact\n",
            "  quant: None | Quant = None\n  enable_fractal_rgrp_sidecar: bool = False\n\n  @nn.compact\n",
            path=decoders_path,
        )
    if "layer_idx: int | None = None," not in text:
        text = replace_once(
            text,
            "      attention_metadata: dict[str, Any] | None = None,\n  ):",
            "      attention_metadata: dict[str, Any] | None = None,\n      layer_idx: int | None = None,\n  ):",
            path=decoders_path,
        )
    old = '''    # MLP block.
    mlp_lnx = linears.mlp_block(
        in_features=lnx.shape[-1],
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        model_mode=model_mode,
        config=cfg,
        quant=self.quant,
        mesh=self.mesh,
    )(lnx, deterministic=deterministic)
'''
    old_replacement_lane = '''    # FFN block. The Fractal adapter intentionally plugs into the same
    # pre-normalized seam as the standard MLP so attention/output contracts stay
    # MaxText-native.
    if cfg.fractal_candidate == "rotary-gated-recurrent-state-update":
      mlp_lnx = fractal_rgrp.rgrp_block(
          d_model=lnx.shape[-1],
          state_transform=cfg.fractal_rgrp_state_transform,
          scan_unroll=cfg.fractal_rgrp_scan_unroll,
          projection_mode=cfg.fractal_rgrp_projection_mode,
          trig_mode=cfg.fractal_rgrp_trig_mode,
          residual_scale=cfg.fractal_rgrp_residual_scale,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          matmul_precision=cfg.matmul_precision,
          name="mlp",
      )(lnx)
    else:
      mlp_lnx = linears.mlp_block(
          in_features=lnx.shape[-1],
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          model_mode=model_mode,
          config=cfg,
          quant=self.quant,
          mesh=self.mesh,
      )(lnx, deterministic=deterministic)
'''
    new = '''    # FFN block. The Fractal adapter intentionally plugs into the same
    # pre-normalized seam as the standard MLP. It can replace the MLP for harsh
    # ablations, or act as a sidecar while preserving the native MLP contract.
    if (
        cfg.fractal_candidate == "rotary-gated-recurrent-state-update"
        and cfg.fractal_rgrp_integration_mode == "replace-mlp"
    ):
      mlp_lnx = fractal_rgrp.rgrp_block(
          d_model=lnx.shape[-1],
          state_transform=cfg.fractal_rgrp_state_transform,
          scan_unroll=cfg.fractal_rgrp_scan_unroll,
          projection_mode=cfg.fractal_rgrp_projection_mode,
          trig_mode=cfg.fractal_rgrp_trig_mode,
          residual_scale=cfg.fractal_rgrp_residual_scale,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          matmul_precision=cfg.matmul_precision,
          name="mlp",
      )(lnx)
    else:
      mlp_lnx = linears.mlp_block(
          in_features=lnx.shape[-1],
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          model_mode=model_mode,
          config=cfg,
          quant=self.quant,
          mesh=self.mesh,
      )(lnx, deterministic=deterministic)
      if (
          cfg.fractal_candidate == "rotary-gated-recurrent-state-update"
          and cfg.fractal_rgrp_integration_mode == "mlp-sidecar"
          and self.enable_fractal_rgrp_sidecar
      ):
        mlp_lnx = mlp_lnx + fractal_rgrp.sidecar_block(
            sidecar_type=cfg.fractal_rgrp_sidecar_type,
            d_model=lnx.shape[-1],
            bottleneck_dim=cfg.fractal_rgrp_bottleneck_dim,
            state_transform=cfg.fractal_rgrp_state_transform,
            scan_unroll=cfg.fractal_rgrp_scan_unroll,
            projection_mode=cfg.fractal_rgrp_projection_mode,
            trig_mode=cfg.fractal_rgrp_trig_mode,
            side_scale=cfg.fractal_rgrp_side_scale,
            output_init=cfg.fractal_rgrp_output_init,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            matmul_precision=cfg.matmul_precision,
            tree_depth=cfg.fractal_rgrp_tree_depth,
            slot_count=cfg.fractal_rgrp_slot_count,
            name="fractal_rgrp_sidecar",
        )(lnx)
'''
    if old_replacement_lane in text:
        text = replace_once(text, old_replacement_lane, new, path=decoders_path)
    elif "cfg.fractal_rgrp_integration_mode == \"mlp-sidecar\"" not in text:
        text = replace_once(text, old, new, path=decoders_path)
    else:
        text = text.replace(
            "          and fractal_rgrp.layer_enabled(cfg.fractal_rgrp_layers, layer_idx)\n",
            "          and self.enable_fractal_rgrp_sidecar\n",
        )
    if "enable_fractal_rgrp_sidecar=fractal_rgrp.layer_enabled(cfg.fractal_rgrp_layers, lyr)" not in text:
        text = replace_once(
            text,
            "                config=cfg, mesh=mesh, name=f\"layers_{lyr}\", quant=self.quant, model_mode=self.model_mode, **layer_kwargs\n",
            (
                "                config=cfg,\n"
                "                mesh=mesh,\n"
                "                name=f\"layers_{lyr}\",\n"
                "                quant=self.quant,\n"
                "                model_mode=self.model_mode,\n"
                "                enable_fractal_rgrp_sidecar=fractal_rgrp.layer_enabled(cfg.fractal_rgrp_layers, lyr),\n"
                "                **layer_kwargs,\n"
            ),
            path=decoders_path,
        )
    if "layer_idx=lyr," not in text:
        text = replace_once(
            text,
            "                attention_metadata=attention_metadata,\n                **layer_call_kwargs,\n",
            "                attention_metadata=attention_metadata,\n                layer_idx=lyr,\n                **layer_call_kwargs,\n",
            path=decoders_path,
        )
    path1_method = '''  def _fractal_apply_path1_scale_profile(
      self,
      y,
      RemattedBlockLayer,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk,
      slot,
      page_state,
      attention_metadata,
  ):
    """Apply bounded TPU ports of the surviving Path1 scale-leader families.

    These are explicit reference lowerings. Routed full-sequence top-C uses a
    selected-only gather/scatter block. Causal prefix top-k preserves the Path1
    selected-output semantics but is a compute proxy: the dense block is still
    evaluated before skipped-token identity scatter.
    """

    cfg = self.config
    if cfg.scan_layers:
      raise ValueError("Path1 TPU scale profiles require scan_layers=false so layer identities are explicit.")
    if cfg.decoder_block != DecoderBlockType.DEFAULT:
      raise ValueError("Path1 TPU scale profiles currently support decoder_block=default only.")
    if cfg.using_pipeline_parallelism:
      raise ValueError("Path1 TPU scale profiles currently do not support pipeline parallelism.")
    if previous_chunk is not None or page_state is not None or slot is not None:
      raise ValueError("Path1 TPU scale profiles currently support training/prefill only, not paged decode.")

    width = y.shape[-1]
    route_fraction = float(cfg.fractal_path1_route_fraction)
    diagnostics_enabled = bool(cfg.fractal_path1_diagnostics)
    layer_modules = [
        RemattedBlockLayer(
            config=cfg,
            mesh=self.mesh,
            name=f"layers_{lyr}",
            quant=self.quant,
            model_mode=self.model_mode,
        )
        for lyr in range(cfg.num_decoder_layers)
    ]

    def apply_layer(layer_input, lyr: int, segment_ids=decoder_segment_ids, positions=decoder_positions):
      layer_output, _ = layer_modules[lyr](
          layer_input,
          segment_ids,
          positions,
          deterministic,
          model_mode,
          previous_chunk=None,
          page_state=None,
          slot=None,
          kv_cache=None,
          attention_metadata=attention_metadata,
          layer_idx=lyr,
      )
      return layer_output

    def router_scores(hidden, name_suffix: str):
      router_input = rms_norm(
          num_features=width,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name=f"fractal_path1_router_norm_{name_suffix}",
          epsilon=cfg.normalization_layer_epsilon,
          kernel_axes=("norm",),
      )(hidden)
      return nn.Dense(
          1,
          use_bias=True,
          dtype=cfg.dtype,
          param_dtype=cfg.weight_dtype,
          precision=cfg.matmul_precision,
          kernel_init=nn.initializers.normal(stddev=0.02),
          bias_init=nn.initializers.zeros,
          name=f"fractal_path1_router_{name_suffix}",
      )(router_input).squeeze(-1)

    def route_dense_update(hidden, lyr: int):
      block_out = apply_layer(hidden, lyr)
      scores = router_scores(hidden, f"layer_{lyr}")
      selected_mask = fractal_rgrp.causal_prefix_topk_mask(scores, route_fraction)
      gate = jax.nn.sigmoid(scores).astype(hidden.dtype)
      mixed = hidden + gate[..., None] * (block_out - hidden)
      out = jnp.where(selected_mask[..., None], mixed, hidden)
      if diagnostics_enabled:
        fractal_rgrp.sow_path1_diagnostic(self, "routing/selected_fraction", jnp.mean(selected_mask.astype(jnp.float32)))
        fractal_rgrp.sow_path1_diagnostic(self, "routing/gate_mean", jnp.mean(gate.astype(jnp.float32)))
        fractal_rgrp.sow_path1_diagnostic(self, "routing/compute_proxy_dense_block", jnp.asarray(1.0, dtype=jnp.float32))
      return out

    def route_selected_only_update(hidden, lyr: int):
      scores = router_scores(hidden, f"layer_{lyr}")
      selected_indices = fractal_rgrp.full_topc_indices(scores, route_fraction)
      gather_hidden = jnp.take_along_axis(hidden, selected_indices[..., None], axis=1)
      selected_positions = jnp.take_along_axis(decoder_positions, selected_indices, axis=1)
      selected_segments = (
          None
          if decoder_segment_ids is None
          else jnp.take_along_axis(decoder_segment_ids, selected_indices, axis=1)
      )
      selected_out = apply_layer(gather_hidden, lyr, segment_ids=selected_segments, positions=selected_positions)
      selected_scores = jnp.take_along_axis(scores, selected_indices, axis=1)
      selected_gate = jax.nn.sigmoid(selected_scores).astype(hidden.dtype)
      gated = gather_hidden + selected_gate[..., None] * (selected_out - gather_hidden)
      batch_index = jnp.arange(hidden.shape[0], dtype=selected_indices.dtype)[:, None]
      out = hidden.at[batch_index, selected_indices].set(gated)
      if diagnostics_enabled:
        selected_mask = fractal_rgrp.indices_to_mask(selected_indices, hidden.shape[1])
        fractal_rgrp.sow_path1_diagnostic(self, "routing/selected_fraction", jnp.mean(selected_mask.astype(jnp.float32)))
        fractal_rgrp.sow_path1_diagnostic(self, "routing/gate_mean", jnp.mean(selected_gate.astype(jnp.float32)))
        fractal_rgrp.sow_path1_diagnostic(self, "routing/compute_proxy_dense_block", jnp.asarray(0.0, dtype=jnp.float32))
      return out

    if cfg.fractal_candidate in ("causal-topk-route50-layer1", "mod-train-topc-route50-layer1"):
      for lyr in range(cfg.num_decoder_layers):
        if fractal_rgrp.layer_enabled(cfg.fractal_path1_route_layers, lyr):
          if cfg.fractal_candidate == "causal-topk-route50-layer1":
            y = route_dense_update(y, lyr)
          else:
            y = route_selected_only_update(y, lyr)
        else:
          y = apply_layer(y, lyr)
      return y

    if cfg.fractal_candidate == "d3-route25-accel":
      start, end = fractal_rgrp.middle_loop_bounds(cfg.num_decoder_layers)
      for lyr in range(start):
        y = apply_layer(y, lyr)
      state = y
      previous_delta = jnp.zeros_like(state)
      active = jnp.asarray(True)
      steps_used = jnp.asarray(0.0, dtype=jnp.float32)
      threshold = jnp.asarray(float(cfg.fractal_path1_accel_threshold), dtype=jnp.float32)
      min_steps = int(cfg.fractal_path1_min_steps)
      loop_count = int(cfg.fractal_path1_loop_count)
      for loop_idx in range(loop_count):
        candidate = state
        for lyr in range(start, end):
          block_out = apply_layer(candidate, lyr)
          scores = router_scores(candidate, f"d3_{loop_idx}_{lyr}")
          selected_mask = fractal_rgrp.causal_prefix_topk_mask(scores, route_fraction)
          gate = jax.nn.sigmoid(scores).astype(candidate.dtype)
          mixed = candidate + gate[..., None] * (block_out - candidate)
          candidate = jnp.where(selected_mask[..., None], mixed, candidate)
          if diagnostics_enabled:
            fractal_rgrp.sow_path1_diagnostic(self, f"routing/d3_selected_fraction_step_{loop_idx}", jnp.mean(selected_mask.astype(jnp.float32)))
        delta = candidate - state
        accel = delta - previous_delta
        accel_norm = fractal_rgrp._global_l2_norm(accel)
        denom = fractal_rgrp._global_l2_norm(delta) + fractal_rgrp._global_l2_norm(previous_delta) + jnp.asarray(1.0e-6, dtype=jnp.float32)
        normalized_accel = accel_norm / denom
        can_halt = loop_idx + 1 >= min_steps
        halt_now = active & can_halt & (normalized_accel < threshold)
        state = jnp.where(active, candidate, state)
        steps_used = steps_used + active.astype(jnp.float32)
        active = active & ~halt_now
        previous_delta = delta
        if diagnostics_enabled:
          fractal_rgrp.sow_path1_diagnostic(self, f"loop/acceleration_norm_step_{loop_idx}", normalized_accel)
          fractal_rgrp.sow_path1_diagnostic(self, f"loop/state_norm_step_{loop_idx}", fractal_rgrp._mean_l2_norm(state))
      if diagnostics_enabled:
        fractal_rgrp.sow_path1_diagnostic(self, "loop/steps_used_mean", steps_used)
        fractal_rgrp.sow_path1_diagnostic(self, "loop/early_exit_fraction", (steps_used < loop_count).astype(jnp.float32))
      y = state
      for lyr in range(end, cfg.num_decoder_layers):
        y = apply_layer(y, lyr)
      return y

    if cfg.fractal_candidate in ("fixed-looped-lm", "input-injected-looped-lm", "universal-transformer-act", "mor-expert-choice"):
      shared_layers = int(cfg.fractal_path1_shared_layers)
      loop_count = int(cfg.fractal_path1_loop_count)
      if shared_layers <= 0:
        raise ValueError(f"fractal_path1_shared_layers must be positive, got {shared_layers}")
      if loop_count <= 0:
        raise ValueError(f"fractal_path1_loop_count must be positive, got {loop_count}")
      shared_modules = [
          RemattedBlockLayer(
              config=cfg,
              mesh=self.mesh,
              name=f"fractal_path1_shared_layers_{shared_idx}",
              quant=self.quant,
              model_mode=self.model_mode,
          )
          for shared_idx in range(shared_layers)
      ]

      def apply_shared(layer_input, shared_idx: int, segment_ids=decoder_segment_ids, positions=decoder_positions):
        layer_output, _ = shared_modules[shared_idx](
            layer_input,
            segment_ids,
            positions,
            deterministic,
            model_mode,
            previous_chunk=None,
            page_state=None,
            slot=None,
            kv_cache=None,
            attention_metadata=attention_metadata,
            layer_idx=shared_idx,
        )
        return layer_output

      def run_shared_stack(layer_input, segment_ids=decoder_segment_ids, positions=decoder_positions):
        h = layer_input
        for shared_idx in range(shared_layers):
          h = apply_shared(h, shared_idx, segment_ids=segment_ids, positions=positions)
        return h

      if cfg.fractal_candidate in ("fixed-looped-lm", "input-injected-looped-lm"):
        prompt = y
        state = jnp.zeros_like(prompt) if cfg.fractal_candidate == "input-injected-looped-lm" else prompt
        for loop_idx in range(loop_count):
          loop_input = state + prompt if cfg.fractal_candidate == "input-injected-looped-lm" else state
          state = run_shared_stack(loop_input)
          if diagnostics_enabled:
            fractal_rgrp.sow_path1_diagnostic(self, f"looped/state_norm_step_{loop_idx}", fractal_rgrp._mean_l2_norm(state))
        return state

      if cfg.fractal_candidate == "universal-transformer-act":
        threshold = jnp.asarray(float(cfg.fractal_path1_act_threshold), dtype=jnp.float32)
        state = y
        weighted = jnp.zeros_like(y)
        halting = jnp.zeros(y.shape[:2], dtype=jnp.float32)
        updates = jnp.zeros(y.shape[:2], dtype=jnp.float32)
        time_embedding = self.param(
            "fractal_path1_ut_time_embedding",
            nn.initializers.normal(stddev=0.02),
            (loop_count, width),
            cfg.weight_dtype,
        )
        halt_norm = rms_norm(
            num_features=width,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            name="fractal_path1_ut_halt_norm",
            epsilon=cfg.normalization_layer_epsilon,
            kernel_axes=("norm",),
        )
        halt_projection = nn.Dense(
            1,
            use_bias=True,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            precision=cfg.matmul_precision,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
            name="fractal_path1_ut_halt",
        )
        for loop_idx in range(loop_count):
          step_input = state + time_embedding[loop_idx].astype(cfg.dtype).reshape(1, 1, -1)
          next_state = run_shared_stack(step_input)
          gate_logits = halt_projection(halt_norm(next_state)).squeeze(-1)
          p = jax.nn.sigmoid(gate_logits).astype(jnp.float32)
          still_running = halting < threshold
          new_halted = still_running & ((halting + p) > threshold)
          continue_running = still_running & ~new_halted
          remainder = jnp.maximum(threshold - halting, 0.0)
          update_weight = jnp.where(new_halted, remainder, jnp.where(continue_running, p, 0.0))
          if loop_idx == loop_count - 1:
            update_weight = jnp.where(still_running, 1.0 - halting, update_weight)
          weighted = weighted + update_weight[..., None].astype(next_state.dtype) * next_state
          halting = halting + update_weight
          updates = updates + still_running.astype(jnp.float32)
          state = next_state
        if diagnostics_enabled:
          fractal_rgrp.sow_path1_diagnostic(self, "act/updates_mean", jnp.mean(updates))
          fractal_rgrp.sow_path1_diagnostic(self, "act/halting_mean", jnp.mean(halting))
        return weighted

      # MoR expert-choice reference: a shared stack recurs over full-sequence
      # top-C active tokens, scattering router-weighted updates back into the
      # full sequence. This preserves the active-token shrinkage semantics but
      # does not add the paper's router auxiliary loss in this MaxText path.
      state = y
      active_mask = jnp.ones(y.shape[:2], dtype=jnp.bool_)
      for loop_idx in range(loop_count):
        scores = router_scores(state, f"mor_{loop_idx}")
        masked_scores = jnp.where(active_mask, scores, jnp.full_like(scores, -1.0e30))
        selected_indices = fractal_rgrp.full_topc_indices(masked_scores, route_fraction)
        gather_state = jnp.take_along_axis(state, selected_indices[..., None], axis=1)
        selected_positions = jnp.take_along_axis(decoder_positions, selected_indices, axis=1)
        selected_segments = (
            None
            if decoder_segment_ids is None
            else jnp.take_along_axis(decoder_segment_ids, selected_indices, axis=1)
        )
        selected_out = run_shared_stack(gather_state, segment_ids=selected_segments, positions=selected_positions)
        selected_scores = jnp.take_along_axis(scores, selected_indices, axis=1)
        selected_gate = jax.nn.sigmoid(selected_scores).astype(state.dtype)
        gated = gather_state + selected_gate[..., None] * (selected_out - gather_state)
        batch_index = jnp.arange(state.shape[0], dtype=selected_indices.dtype)[:, None]
        state = state.at[batch_index, selected_indices].set(gated)
        active_mask = fractal_rgrp.indices_to_mask(selected_indices, state.shape[1])
        if diagnostics_enabled:
          fractal_rgrp.sow_path1_diagnostic(self, f"mor/active_fraction_step_{loop_idx}", jnp.mean(active_mask.astype(jnp.float32)))
          fractal_rgrp.sow_path1_diagnostic(self, f"mor/gate_mean_step_{loop_idx}", jnp.mean(selected_gate.astype(jnp.float32)))
      return state

    raise ValueError(f"unsupported Path1 TPU scale profile: {cfg.fractal_candidate!r}")

'''
    parcae_method = '''  def _fractal_apply_parcae_loop(
      self,
      y,
      RemattedBlockLayer,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      previous_chunk,
      slot,
      page_state,
      attention_metadata,
  ):
    """Apply the Parcae-inspired looped-middle scaffold.

    This intentionally mirrors the Torch Path 1 Parcae contract:

    prelude layers -> normalized loop input -> looped middle layers -> coda

    The middle layer band reuses the same layer parameters for each recurrent
    pass. This patch currently supports only the default unscanned decoder block
    used by the proof ladder; unsupported MaxText control planes fail loudly
    instead of silently changing the architecture.
    """

    cfg = self.config
    if cfg.scan_layers:
      raise ValueError("Parcae looped scaffold requires scan_layers=false so layer identities are explicit.")
    if cfg.decoder_block != DecoderBlockType.DEFAULT:
      raise ValueError("Parcae looped scaffold currently supports decoder_block=default only.")
    if cfg.using_pipeline_parallelism:
      raise ValueError("Parcae looped scaffold currently does not support pipeline parallelism.")

    start, end = fractal_rgrp.middle_loop_bounds(cfg.num_decoder_layers)
    width = y.shape[-1]
    layer_modules = [
        RemattedBlockLayer(
            config=cfg,
            mesh=self.mesh,
            name=f"layers_{lyr}",
            quant=self.quant,
            model_mode=self.model_mode,
        )
        for lyr in range(cfg.num_decoder_layers)
    ]

    def apply_layer(layer_input, lyr: int):
      layer_output, _ = layer_modules[lyr](
          layer_input,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          page_state=page_state,
          slot=slot,
          kv_cache=None,
          attention_metadata=attention_metadata,
          layer_idx=lyr,
      )
      return layer_output

    for lyr in range(start):
      y = apply_layer(y, lyr)

    loop_input = rms_norm(
        num_features=width,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="fractal_parcae_prelude_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(y)
    decay_raw = self.param(
        "fractal_parcae_decay_raw",
        nn.initializers.constant(-2.0),
        (width,),
        cfg.weight_dtype,
    )
    nonlinear_logit = self.param(
        "fractal_parcae_nonlinear_logit",
        nn.initializers.zeros,
        (width,),
        cfg.weight_dtype,
    )
    decay_rate = jax.nn.softplus(decay_raw.astype(cfg.dtype)).reshape(1, 1, -1)
    if cfg.fractal_parcae_discretization == "stable-exp":
      decay = jnp.exp(-decay_rate)
      input_scale = jnp.ones_like(decay)
    elif cfg.fractal_parcae_discretization == "zoh":
      dt_raw = self.param(
          "fractal_parcae_dt_raw",
          nn.initializers.constant(cfg.fractal_parcae_dt_raw_init),
          (width,),
          cfg.weight_dtype,
      )
      step_size = jax.nn.softplus(dt_raw.astype(cfg.dtype)).reshape(1, 1, -1)
      decay = jnp.exp(-step_size * decay_rate)
      input_scale = (1.0 - decay) / jnp.maximum(decay_rate, jnp.asarray(1.0e-6, dtype=decay_rate.dtype))
    else:
      raise ValueError(
          f"fractal_parcae_discretization must be one of {fractal_rgrp.SUPPORTED_PARCAE_DISCRETIZATIONS}, "
          f"got {cfg.fractal_parcae_discretization!r}"
      )
    nonlinear = jax.nn.sigmoid(nonlinear_logit.astype(cfg.dtype)).reshape(1, 1, -1)
    diagnostics_enabled = (
        cfg.fractal_parcae_control_diagnostics
        and cfg.fractal_candidate in ("parcae-rgrp-control-looped-attention", "parcae-p20-control-looped-attention")
    )
    diagnostics_nan_or_inf_seen = jnp.asarray(False, dtype=jnp.bool_)
    control_mode = cfg.fractal_parcae_control_mode
    control_gate_blend = cfg.fractal_parcae_control_gate_blend
    control_bottleneck_dim = cfg.fractal_parcae_control_bottleneck_dim
    control_value_scale = cfg.fractal_parcae_control_value_scale

    if cfg.fractal_candidate == "parcae-looped-attention":
      injection_logit = self.param(
          "fractal_parcae_injection_logit",
          nn.initializers.constant(-2.1972246),
          (width,),
          cfg.weight_dtype,
      )
      injection_gate = jax.nn.sigmoid(injection_logit.astype(cfg.dtype)).reshape(1, 1, -1)
      injection_value = loop_input
    elif cfg.fractal_candidate == "parcae-bx-looped-attention":
      injection_value = nn.Dense(
          width,
          use_bias=False,
          dtype=cfg.dtype,
          param_dtype=cfg.weight_dtype,
          precision=cfg.matmul_precision,
          kernel_init=fractal_rgrp.identity_kernel_init,
          name="fractal_parcae_b_value_projection",
      )(loop_input)
      injection_gate = jax.nn.sigmoid(
          nn.Dense(
              width,
              use_bias=True,
              dtype=cfg.dtype,
              param_dtype=cfg.weight_dtype,
              precision=cfg.matmul_precision,
              kernel_init=nn.initializers.zeros,
              bias_init=nn.initializers.constant(-2.1972246),
              name="fractal_parcae_b_gate_projection",
          )(loop_input)
      )
    elif cfg.fractal_candidate in ("parcae-rgrp-control-looped-attention", "parcae-p20-control-looped-attention"):
      if control_mode not in fractal_rgrp.SUPPORTED_PARCAE_CONTROL_MODES:
        raise ValueError(
            f"fractal_parcae_control_mode must be one of {fractal_rgrp.SUPPORTED_PARCAE_CONTROL_MODES}, "
            f"got {control_mode!r}"
        )
      if control_bottleneck_dim < 0:
        raise ValueError(f"fractal_parcae_control_bottleneck_dim must be non-negative, got {control_bottleneck_dim}")
      if control_gate_blend < 0.0 or control_gate_blend > 1.0:
        raise ValueError(f"fractal_parcae_control_gate_blend must be in [0, 1], got {control_gate_blend}")
      if control_value_scale <= 0.0:
        raise ValueError(f"fractal_parcae_control_value_scale must be greater than zero, got {control_value_scale}")
      control_width = control_bottleneck_dim or width
      if control_width <= 0:
        raise ValueError(f"Parcae RGRP-control width must be positive, got {control_width}")
      if control_width % 2 != 0:
        raise ValueError(f"Parcae RGRP-control width must be even for rotary pairs, got {control_width}")
      if control_width % fractal_rgrp._block_count(cfg.fractal_rgrp_state_transform) != 0:
        raise ValueError(
            "fractal_parcae_control_bottleneck_dim must be divisible by the RGRP block count "
            f"for {cfg.fractal_rgrp_state_transform!r}; got {control_width}"
        )
      if control_width == width:
        control_input = loop_input
      else:
        control_input = nn.Dense(
            control_width,
            use_bias=True,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            precision=cfg.matmul_precision,
            kernel_init=nn.initializers.xavier_uniform(),
            name="fractal_parcae_rgrp_control_down_projection",
        )(loop_input)
      control = fractal_rgrp.rgrp_block(
          d_model=control_width,
          state_transform=cfg.fractal_rgrp_state_transform,
          scan_unroll=cfg.fractal_rgrp_scan_unroll,
          projection_mode=cfg.fractal_rgrp_projection_mode,
          trig_mode=cfg.fractal_rgrp_trig_mode,
          residual_scale=1.0,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          matmul_precision=cfg.matmul_precision,
          name="fractal_parcae_rgrp_control",
      )(control_input)
      control = rms_norm(
          num_features=control_width,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="fractal_parcae_rgrp_control_norm",
          epsilon=cfg.normalization_layer_epsilon,
          kernel_axes=("norm",),
      )(control)
      base_gate = None
      if control_mode == "value-only" or control_gate_blend > 0.0:
        base_gate_logit = self.param(
            "fractal_parcae_rgrp_base_injection_logit",
            nn.initializers.constant(-2.1972246),
            (width,),
            cfg.weight_dtype,
        )
        base_gate = jax.nn.sigmoid(base_gate_logit.astype(cfg.dtype)).reshape(1, 1, -1)
      control_gate = None
      if control_mode != "value-only":
        control_gate = jax.nn.sigmoid(
            nn.Dense(
                width,
                use_bias=True,
                dtype=cfg.dtype,
                param_dtype=cfg.weight_dtype,
                precision=cfg.matmul_precision,
                kernel_init=nn.initializers.normal(stddev=1.0e-3),
                bias_init=nn.initializers.constant(-2.1972246),
                name="fractal_parcae_rgrp_gate_projection",
            )(control)
        )
        if base_gate is not None and control_gate_blend > 0.0:
          blend = jnp.asarray(control_gate_blend, dtype=cfg.dtype)
          injection_gate = (1.0 - blend) * control_gate + blend * base_gate
        else:
          injection_gate = control_gate
      else:
        injection_gate = base_gate
      control_value = None
      if control_mode != "gate-only":
        control_value = nn.Dense(
            width,
            use_bias=False,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            precision=cfg.matmul_precision,
            kernel_init=nn.initializers.normal(stddev=1.0e-3),
            name="fractal_parcae_rgrp_value_projection",
        )(control)
        control_value = control_value * jnp.asarray(control_value_scale, dtype=control_value.dtype)
        injection_value = loop_input + control_value
      else:
        injection_value = loop_input
      if diagnostics_enabled:
        loop_input_norm = fractal_rgrp._global_l2_norm(loop_input)
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, "controller/control_norm_mean", fractal_rgrp._mean_l2_norm(control)
        )
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, "controller/control_norm_rms", fractal_rgrp._rms(control)
        )
        fractal_rgrp.sow_parcae_control_diagnostic(self, "controller/gate_mean", jnp.mean(injection_gate))
        fractal_rgrp.sow_parcae_control_diagnostic(self, "controller/gate_std", jnp.std(injection_gate))
        fractal_rgrp.sow_parcae_control_diagnostic(self, "controller/gate_min", jnp.min(injection_gate))
        fractal_rgrp.sow_parcae_control_diagnostic(self, "controller/gate_max", jnp.max(injection_gate))
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, "controller/gate_saturation_low_frac", jnp.mean((injection_gate < 0.05).astype(jnp.float32))
        )
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, "controller/gate_saturation_high_frac", jnp.mean((injection_gate > 0.95).astype(jnp.float32))
        )
        if control_value is not None:
          fractal_rgrp.sow_parcae_control_diagnostic(
              self, "controller/value_norm_mean", fractal_rgrp._mean_l2_norm(control_value)
          )
          fractal_rgrp.sow_parcae_control_diagnostic(
              self,
              "controller/value_to_loop_input_norm_ratio",
              fractal_rgrp._safe_ratio(fractal_rgrp._global_l2_norm(control_value), loop_input_norm),
          )
        diagnostics_values = [loop_input, control_input, control, injection_gate, injection_value]
        if control_gate is not None:
          diagnostics_values.append(control_gate)
        if control_value is not None:
          diagnostics_values.append(control_value)
        if base_gate is not None:
          diagnostics_values.append(base_gate)
        diagnostics_nan_or_inf_seen = fractal_rgrp._nan_or_inf_seen(*diagnostics_values)
    else:
      raise ValueError(f"unsupported Parcae profile: {cfg.fractal_candidate!r}")

    rng = None if (deterministic or cfg.fractal_parcae_loop_policy == "fixed") else self.make_rng("dropout")
    loop_depths, bwd_depths, max_loop_count = fractal_rgrp.parcae_loop_depths(
        cfg=cfg,
        batch_size=loop_input.shape[0],
        deterministic=deterministic,
        rng=rng,
    )

    state = jnp.zeros_like(loop_input)
    injection = input_scale * injection_gate * injection_value
    if diagnostics_enabled:
      loop_input_norm = fractal_rgrp._global_l2_norm(loop_input)
      injection_delta = injection - loop_input
      fractal_rgrp.sow_parcae_control_diagnostic(
          self,
          "controller/injection_delta_norm_mean",
          fractal_rgrp._mean_l2_norm(injection_delta),
      )
      fractal_rgrp.sow_parcae_control_diagnostic(
          self,
          "controller/injection_delta_to_loop_input_ratio",
          fractal_rgrp._safe_ratio(fractal_rgrp._global_l2_norm(injection_delta), loop_input_norm),
      )
      fractal_rgrp.sow_parcae_control_diagnostic(
          self, "loop/steps_used_mean", jnp.mean(loop_depths.astype(jnp.float32))
      )
      fractal_rgrp.sow_parcae_control_diagnostic(
          self,
          "loop/early_exit_fraction",
          jnp.mean((loop_depths < jnp.asarray(max_loop_count, dtype=loop_depths.dtype)).astype(jnp.float32)),
      )
      diagnostics_nan_or_inf_seen = diagnostics_nan_or_inf_seen | fractal_rgrp._nan_or_inf_seen(
          injection, injection_delta
      )
    if diagnostics_enabled:
      previous_step_delta = jnp.zeros_like(state)
    for loop_idx in range(max_loop_count):
      previous_state = state
      active_mask, grad_mask = fractal_rgrp.parcae_depth_masks(
          loop_depths,
          bwd_depths,
          loop_idx,
          max_loop_count,
      )
      mixed = decay * state + injection
      for lyr in range(start, end):
        block_out = apply_layer(mixed, lyr)
        mixed = mixed + nonlinear * (block_out - mixed)
      mixed = jnp.where(grad_mask, mixed, jax.lax.stop_gradient(mixed))
      state = jnp.where(active_mask, mixed, state)
      if diagnostics_enabled:
        step_delta = state - previous_state
        # Step 0 reports acceleration against a zero previous delta.
        acceleration = step_delta - previous_step_delta
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, f"loop/state_norm_step_{loop_idx}", fractal_rgrp._mean_l2_norm(state)
        )
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, f"loop/step_delta_norm_step_{loop_idx}", fractal_rgrp._mean_l2_norm(step_delta)
        )
        fractal_rgrp.sow_parcae_control_diagnostic(
            self, f"loop/acceleration_norm_step_{loop_idx}", fractal_rgrp._mean_l2_norm(acceleration)
        )
        diagnostics_nan_or_inf_seen = diagnostics_nan_or_inf_seen | fractal_rgrp._nan_or_inf_seen(
            mixed, state, step_delta, acceleration
        )
        previous_step_delta = step_delta

    if diagnostics_enabled:
      fractal_rgrp.sow_parcae_control_diagnostic(
          self, "stability/nan_or_inf_seen", diagnostics_nan_or_inf_seen.astype(jnp.float32)
      )

    y = state
    for lyr in range(end, cfg.num_decoder_layers):
      y = apply_layer(y, lyr)
    return y

'''
    if "def _fractal_apply_path1_scale_profile(" not in text:
        text = insert_after_once(
            text,
            "  def get_pipeline_stage_module(self, decoder_blocks):\n",
            path1_method,
            path=decoders_path,
        )
        text = text.replace(
            "  def get_pipeline_stage_module(self, decoder_blocks):\n" + path1_method,
            path1_method + "  def get_pipeline_stage_module(self, decoder_blocks):\n",
            1,
        )
    if "def _fractal_apply_parcae_loop(" not in text:
        text = insert_after_once(
            text,
            "  def get_pipeline_stage_module(self, decoder_blocks):\n",
            parcae_method,
            path=decoders_path,
        )
        text = text.replace(
            "  def get_pipeline_stage_module(self, decoder_blocks):\n" + parcae_method,
            parcae_method + "  def get_pipeline_stage_module(self, decoder_blocks):\n",
            1,
        )
    if "is_path1_scale_profile(cfg.fractal_candidate)" not in text:
        loop_marker = "        else:\n          for lyr in range(cfg.num_decoder_layers):\n"
        start = text.find(loop_marker)
        if start == -1:
            raise RuntimeError(f"could not find unscanned default decoder loop in {decoders_path}")
        end_marker = "\n    assert isinstance(y, jax.Array)"
        end = text.find(end_marker, start)
        if end == -1:
            raise RuntimeError(f"could not find end of unscanned default decoder loop in {decoders_path}")
        original_block = text[start:end]
        original_body = original_block.removeprefix("        else:\n")
        indented_body = "\n".join(
            ("  " + line if line else line) for line in original_body.splitlines()
        )
        replacement = (
            "        else:\n"
            "          if fractal_rgrp.is_parcae_profile(cfg.fractal_candidate):\n"
            "            if kv_caches is not None:\n"
            "              raise ValueError(\"Parcae looped scaffold currently does not support kv_caches.\")\n"
            "            if deepstack_visual_embeds is not None:\n"
            "              raise ValueError(\"Parcae looped scaffold currently does not support deepstack visual embeddings.\")\n"
            "            y = self._fractal_apply_parcae_loop(\n"
            "                y,\n"
            "                RemattedBlockLayers[0],\n"
            "                decoder_segment_ids,\n"
            "                decoder_positions,\n"
            "                deterministic,\n"
            "                model_mode,\n"
            "                previous_chunk,\n"
            "                slot,\n"
            "                page_state,\n"
            "                attention_metadata,\n"
            "            )\n"
            "          elif fractal_rgrp.is_path1_scale_profile(cfg.fractal_candidate):\n"
            "            if kv_caches is not None:\n"
            "              raise ValueError(\"Path1 TPU scale profiles currently do not support kv_caches.\")\n"
            "            if deepstack_visual_embeds is not None:\n"
            "              raise ValueError(\"Path1 TPU scale profiles currently do not support deepstack visual embeddings.\")\n"
            "            y = self._fractal_apply_path1_scale_profile(\n"
            "                y,\n"
            "                RemattedBlockLayers[0],\n"
            "                decoder_segment_ids,\n"
            "                decoder_positions,\n"
            "                deterministic,\n"
            "                model_mode,\n"
            "                previous_chunk,\n"
            "                slot,\n"
            "                page_state,\n"
            "                attention_metadata,\n"
            "            )\n"
            "          else:\n"
            f"{indented_body}\n"
        )
        text = text[:start] + replacement + text[end:]
    decoders_path.write_text(text)


def patch_train(package_dir: Path) -> None:
    train_path = package_dir / "trainers" / "pre_train" / "train.py"
    text = train_path.read_text()
    if "from maxtext.layers import fractal_rgrp" not in text:
        text = replace_once(
            text,
            "from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss\n",
            (
                "from maxtext.layers import fractal_rgrp\n"
                "from maxtext.layers.multi_token_prediction import calculate_mtp_acceptance_rate, calculate_mtp_loss\n"
            ),
            path=train_path,
        )
    train_marker = '''  if config.use_dpo:
    scalar_metrics["learning/dpo_loss"] = aux["dpo_loss"]
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  metrics = {
'''
    train_insert = '''  if config.use_dpo:
    scalar_metrics["learning/dpo_loss"] = aux["dpo_loss"]
    scalar_metrics["learning/dpo_reward_accuracy"] = aux["reward_accuracy"]
  if config.fractal_parcae_control_diagnostics:
    scalar_metrics.update(fractal_rgrp.collect_parcae_control_diagnostics(intermediate_outputs))
  if config.fractal_path1_diagnostics:
    scalar_metrics.update(fractal_rgrp.collect_path1_diagnostics(intermediate_outputs))
  metrics = {
'''
    if "collect_parcae_control_diagnostics(intermediate_outputs)" not in text:
        text = replace_once(text, train_marker, train_insert, path=train_path)

    eval_marker = '''  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]

  return metrics
'''
    eval_insert = '''  if config.use_dpo:
    metrics["scalar"]["evaluation/dpo_reward_accuracy"] = aux["reward_accuracy"]
  if config.fractal_parcae_control_diagnostics:
    for metric_name, metric_value in fractal_rgrp.collect_parcae_control_diagnostics(
        aux["intermediate_outputs"]
    ).items():
      metrics["scalar"][f"evaluation/{metric_name}"] = metric_value
  if config.fractal_path1_diagnostics:
    for metric_name, metric_value in fractal_rgrp.collect_path1_diagnostics(
        aux["intermediate_outputs"]
    ).items():
      metrics["scalar"][f"evaluation/{metric_name}"] = metric_value

  return metrics
'''
    if 'metrics["scalar"][f"evaluation/{metric_name}"]' not in text:
        text = replace_once(text, eval_marker, eval_insert, path=train_path)
    train_path.write_text(text)


def patch_metric_logger(package_dir: Path) -> None:
    logger_path = package_dir / "common" / "metric_logger.py"
    text = logger_path.read_text()
    accumulation_marker = '''      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] += float(
            metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0)
        )

    if eval_step_count:
'''
    accumulation_insert = '''      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] += float(
            metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0)
        )
      for metric_name, metric_value in metrics["scalar"].items():
        if metric_name.startswith(("evaluation/controller/", "evaluation/loop/", "evaluation/stability/", "evaluation/routing/", "evaluation/looped/", "evaluation/act/", "evaluation/mor/")):
          self.cumulative_eval_metrics["scalar"][f"eval/{metric_name.removeprefix('evaluation/')}"] += float(metric_value)

    if eval_step_count:
'''
    if "metric_name.startswith((\"evaluation/controller/\"" not in text:
        text = replace_once(text, accumulation_marker, accumulation_insert, path=logger_path)

    averaging_marker = '''      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = (
            self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] / eval_step_count
        )

      self.write_metrics(self.cumulative_eval_metrics, step, is_training=False)
'''
    averaging_insert = '''      if self.config.use_dpo:
        self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = (
            self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] / eval_step_count
        )
      for metric_name in list(self.cumulative_eval_metrics["scalar"].keys()):
        if metric_name.startswith(("eval/controller/", "eval/loop/", "eval/stability/", "eval/routing/", "eval/looped/", "eval/act/", "eval/mor/")):
          self.cumulative_eval_metrics["scalar"][metric_name] = (
              self.cumulative_eval_metrics["scalar"][metric_name] / eval_step_count
          )

      self.write_metrics(self.cumulative_eval_metrics, step, is_training=False)
'''
    if "metric_name.startswith((\"eval/controller/\"" not in text:
        text = replace_once(text, averaging_marker, averaging_insert, path=logger_path)
    logger_path.write_text(text)


def patch_fractal_module(package_dir: Path) -> None:
    module_path = package_dir / "layers" / "fractal_rgrp.py"
    module_path.write_text(FRACTAL_RGRP_MODULE)


def patch_maxtext(path: Path) -> None:
    package_dir = resolve_package_dir(path)
    patch_fractal_module(package_dir)
    patch_types(package_dir)
    patch_decoders(package_dir)
    patch_train(package_dir)
    patch_metric_logger(package_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to a MaxText checkout")
    args = parser.parse_args()
    patch_maxtext(args.checkout.resolve())
    print(f"patched MaxText checkout: {args.checkout.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
