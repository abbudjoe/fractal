# Standard Decoder LLMs

This note writes down the basic formal claims for standard modern decoder-style
LLMs.

The point is not novelty.
The point is to make the exact proof surface explicit and reusable when we
compare later variants.

## Setup

Assume:

- autoregressive token order
- masked causal self-attention
- deterministic inference mode
- fixed parameters `theta`

Let one decoder block act on residual stream `r_t^(ell-1)` and produce
`r_t^(ell)`.

For one attention head:

```text
q_t = W_Q r_t
k_u = W_K r_u
v_u = W_V r_u
```

and with causal mask:

```text
alpha_{t,u} = 0                    if u > t
alpha_{t,u} proportional to exp(q_t · k_u / sqrt(d_k))   if u <= t
```

Then:

```text
attn_t = sum_{u <= t} alpha_{t,u} v_u
```

The decoder stack is built from:

- causal self-attention
- residual addition
- position-wise feed-forward updates

## Definition 1.1: Decoder LM

A standard decoder LM is any stacked residual architecture in which every layer
is causal and the next-token distribution is read from the top-layer token
representation:

```text
p_theta(x_{t+1} | x_<=t) = Softmax(W r_t^(L))
```

## Theorem 1.2: Causal Decoder Correctness

A masked decoder transformer defines a causal conditional distribution over the
next token.

Proof sketch:

1. At every layer, the attention mask forbids access to positions `u > t`.
2. Position-wise feed-forward blocks are local to token position `t`.
3. Residual addition cannot introduce future-token dependence if neither branch
   contains it.
4. By repeated application of causal composition, the entire stack is causal.
5. The output head reads only the top-layer causal state at position `t`.

So the next-token distribution depends only on `x_<=t`.

## Theorem 1.3: KV-Cache Decode Equivalence

Incremental decode with cached keys and values is equivalent to full-prefix
recomputation for the same prefix, weights, and mask.

Proof sketch:

1. In full-prefix recomputation at step `t`, each query `q_t` attends to the
   set `{(k_u, v_u) : u <= t}`.
2. Under fixed weights, previously computed keys and values for `u < t` are
   unchanged by the arrival of token `x_t`.
3. Therefore the cached set of prior `(k_u, v_u)` pairs is exactly the same set
   that full-prefix recomputation would rebuild.
4. The only new objects needed at decode step `t` are the keys and values for
   the new visible token.
5. Because the attention mask and score function are unchanged, the resulting
   weighted sum is the same.

Hence incremental KV-cache decode is equivalent to full-prefix recomputation,
up to deterministic arithmetic tolerance.

## Proposition 1.4: Exact Copy Capacity

Exact self-attention can implement copy-like retrieval of a visible prior token.

Proof sketch:

1. Choose parameters so the query at position `t` scores one prior position
   `u* <= t` strictly above the others.
2. Then the attention weights concentrate on `u*`.
3. The output at `t` becomes arbitrarily close to `v_{u*}`.
4. If the output projection reads the copied content from `v_{u*}`, the model
   implements pointer-style copying from the visible context.

This does not prove that a trained model will learn the right copy map.
It proves that exact attention has the representational mechanism for it.

## Proposition 1.5: Token-To-Token Comparison Capacity

Exact self-attention can implement local or long-range token comparison over the
visible context.

Proof sketch:

1. Keys and queries can be parameterized so attention selects tokens with a
   target relation to the current token representation.
2. The attention-weighted read can gather comparison evidence from one or more
   prior tokens.
3. Subsequent feed-forward layers can map the gathered evidence to a decision
   feature in the residual stream.

This is one reason attention remains the strongest known exact token-interaction
primitive.

## Proposition 1.6: Complexity Bound

Let sequence length be `T`, width be `d`, and layer count be `L`.

Then, at a high level:

- train-time dense attention over a full visible prefix is quadratic in `T`
  per attention layer
- decode-time incremental attention is linear in visible cache length per
  attention layer

More concretely:

```text
train-time attention work ~ O(L * T^2 * d)
decode-time attention work per new token ~ O(L * T * d)
```

where constants depend on:

- head count
- projection sizes
- feed-forward width

The important structural fact is:

- dense attention keeps exact token interaction
- but pays for that exactness with context-length-sensitive cost

## Proposition 1.7: Residual Closure

If every branch feeding a residual add is causal, then the residual output is
causal.

Proof sketch:

- addition does not introduce any new information source
- it only combines already-causal branches

This proposition is trivial but useful, because it is the step that lets the
whole decoder proof stay modular.

## Boundary Of This Note

This note proves or sketches:

- causal correctness
- KV-cache equivalence
- exact copy and comparison capacity
- high-level asymptotic cost

It does not prove:

- training efficiency
- empirical quality
- emergent reasoning ability

Those belong to empirical evaluation, not the theorem surface.
