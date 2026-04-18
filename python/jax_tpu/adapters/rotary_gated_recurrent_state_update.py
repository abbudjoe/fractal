"""Planned JAX adapter seam for the rotary gated recurrent state update.

The first implementation should expose a MaxText-compatible FFN-side module
whose sequence recurrence is expressed with `jax.lax.scan`. Keeping this module
name stable lets manifests point at the future adapter before the MaxText fork is
patched.
"""

ADAPTER_NAME = "rotary-gated-recurrent-state-update"
