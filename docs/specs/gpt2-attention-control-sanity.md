# GPT-2 Spec vs Internal Attention Control Sanity Note

Date: 2026-04-29

Purpose: clarify what our internal attention-only control can and cannot claim when used as a GPT-2-like baseline.

## Canonical GPT-2 Reference

Source: `openai/gpt-2`, `src/model.py`.

Canonical GPT-2 small defaults:

- `n_ctx = 1024`
- `n_embd = 768`
- `n_head = 12`
- `n_layer = 12`
- learned token embeddings `wte`
- learned positional embeddings `wpe`
- pre-LN transformer blocks
- packed QKV projection `c_attn`
- full causal multi-head attention
- output projection `c_proj`
- MLP expansion `4 * n_embd`
- GELU activation
- final layer norm `ln_f`
- logits via token embedding matrix transpose

## Internal Attention Control Match

Our `Path1HybridLanguageModel` attention-only path matches the broad GPT-2 block contract when configured appropriately:

- token embedding
- optional learned positional embedding
- pre-attention `LayerNorm`
- packed QKV projection
- scaled dot-product causal attention
- attention output projection
- residual add
- pre-MLP `LayerNorm`
- GELU MLP
- residual add
- LM head over the residual stream

## Important Differences To Label

The internal control is not automatically "GPT-2 small." It becomes GPT-2-like only under a specific config.

- Attention semantics: local-window `flex-local`/`flash-local` is not canonical GPT-2. GPT-2-faithful means full causal attention, or at least `local_window >= seq_len`.
- Shape: our research rungs often use non-GPT-2 widths, heads, layers, and hourglass schedules.
- Position contract: GPT-2 uses learned positions added to the shared residual input. Our controls may use no positions or attention-only positions.
- Final norm: GPT-2 uses learned LayerNorm before logits. Some Path1 profiles may use identity or RMSNorm depending on the variant spec.
- LM head tying: GPT-2 ties logits to token embeddings. Path1 currently uses a separate `nn.Linear` output head.
- Initialization: GPT-2 uses its TensorFlow/OpenAI initialization conventions. Our PyTorch modules use default PyTorch initialization unless overridden by a scaffold.
- Runtime kernel: GPT-2 spec says nothing about SDPA/Flex/Flash kernels. Kernel choice is a runtime control, not an architecture claim.

## Recommended Labels

- `internal-attention-local`: optimized local-attention control, not GPT-2-faithful.
- `internal-gpt2-like`: internal attention model with learned shared positions, GELU MLP, full causal attention, GPT-2-style shape as closely as practical.
- `official-gpt2-spec`: architecture facts derived from `openai/gpt-2`; not a runnable CUDA baseline by itself.

## Practical Conclusion

For our scaling experiments, the internal attention-only control remains the best matched scientific control because it shares the same data loader, optimizer, dtype, loss, reports, and hardware path as RGRP. But claims should avoid calling it "GPT-2" unless the shape and attention semantics are explicitly GPT-2-like.
