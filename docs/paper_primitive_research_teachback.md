# Paper Primitive Research Teachback

Generated: 2026-04-20
MoDA addendum added: 2026-04-20

This is a research-only artifact. It consolidates the GPT-5.4 xhigh subagent reads requested for the architecture exploration program. It is intended to be used as an implementation contract before changing model code.

Proving-ground readiness gate: see `docs/core_baseline_proving_ground_readiness.md` before implementing or promoting any core baseline.

Important fidelity note:

- Several first-pass repo variants were practical approximations, not paper-faithful implementations.
- The entries below separate paper-faithful primitives from pragmatic approximations.
- A hybrid should only be promoted after its constituent primitive has a faithful baseline or is explicitly labeled as an approximation.
- The paper at `2404.02258` is Mixture-of-Depths, not MoDA/depth-memory attention. It should not be used as evidence for cross-depth KV retrieval.
- The actual MoDA paper is `2603.15619`, Mixture-of-Depths Attention, and is included as its own primitive below.

## Cross-Paper Primitive Map

| Family | Paper(s) | Paper-faithful primitive | First repo-safe baseline |
| --- | --- | --- | --- |
| Depth-augmented attention | MoDA | Joint-softmax attention over current sequence KV plus same-token prior-depth KV | Slow PyTorch reference with explicit depth KV cache |
| Conditional token depth | MoD | Per-block top-k token routing; skipped tokens bypass whole block | MoD block with true gather/scatter and router-weighted update |
| Fixed shared recurrence | Universal Transformer, looped LMs, Ouro, recurrent depth | Reuse the same block or block group over depth | Shared recurrent block with explicit loop index and metrics |
| Input-injected loop | Looped Transformers 2311.12424, Huginn | Reinject original prompt/embedding into each loop | Additive or adapter-based input injection, not plain weight tying |
| Adaptive latent compute | Ouro, MoR, UT ACT | Learned exit distribution or token routing, trained with specific losses | Implement one paper-specific policy at a time |
| Explicit compute tokens | Pause tokens, Coconut | Add pause tokens or feed hidden states as embeddings | Treat as data/model-interface change, not a transformer block change |
| Sparse attention | NSA | Three-branch sparse attention with trainable compression and selection | Separate compressed, selected, and window branches |
| Recursive compression | RRT, MoR | Tie layers cyclically; optional depth-specific LoRA or routing | Shared middle block with absolute-depth cache and diagnostics |
| Programmed computation | Looped Transformers 2301.13196 | Hand-coded transformer simulates SUBLEQ/FLEQ over scratchpad state | A semantic interpreter or toy constructive module, not an LM primitive |

## 0. MoDA: Mixture-of-Depths Attention

Link: https://arxiv.org/abs/2603.15619

Official code: https://github.com/hustvl/MoDA

### Short Summary

MoDA means Mixture-of-Depths Attention. It is the missing depth-augmented attention primitive, and it is distinct from MoD/Mixture-of-Depths token routing. MoDA targets signal dilution in deep transformers: useful shallow-layer features can be overwritten or diluted by repeated residual updates, so deeper layers may struggle to recover them. The mechanism lets each attention head retrieve prior-depth information directly through a depth KV memory.

At layer `l`, each token query attends to two KV sets in one attention operation:

- normal sequence KV from the current layer, with causal masking for decoder LMs;
- depth KV from preceding layers at the same sequence position.

High-level findings from the paper and official repo read:

- 700M and 1.5B OLMo2-style models trained on 400B tokens improve validation perplexity and downstream tasks.
- Reported downstream averages improve from `57.11` to `58.87` at 700M and from `62.28` to `64.39` at 1.5B.
- Reported average validation PPL improves from `15.61` to `15.46` at 700M and from `13.67` to `13.47` at 1.5B.
- The paper reports a hardware-aware kernel reaching about `97.3%` of FlashAttention-2 efficiency at 64K sequence length.
- Post-norm plus MoDA worked better than pre-norm in the deeper small-model ablation.
- The public repo releases kernels and a vision integration; full LLM training recipes/configs were not released in the subagent read.

### Primitive Contract

Paper-faithful operator for token `t`, query head `h`, and KV head `kv = phi(h)`:

```text
seq_logits[j]   = dot(q[t, h], k_seq[j, kv]) / sqrt(d) + causal_mask(j <= t)
depth_logits[r] = dot(q[t, h], k_depth[t, r, kv]) / sqrt(d)

weights = softmax(concat(seq_logits, depth_logits))

out[t, h] =
    sum_j weights_seq[j]   * v_seq[j, kv]
  + sum_r weights_depth[r] * v_depth[t, r, kv]
```

Important contract details:

- Depth memory is KV memory, not hidden-state concatenation.
- Depth retrieval is same-token across layers: token `t` retrieves prior depth slots `{K_i[t], V_i[t]}`.
- Sequence logits and depth logits share one softmax.
- MoDA reuses the normal query projection; no separate depth query projection is part of the paper primitive.
- Attention-side depth KV can reuse previous layers' sequence K/V with no extra parameters.
- FFN-side depth KV is optional but beneficial in the paper: project the FFN input to extra K/V slots and append them to depth memory.
- Extra attention-side depth KV projections were tested but gave marginal gains for noticeable overhead.
- The paper does not specify routing, top-k layer selection, learned layer queries, or depth positional embeddings.

Practical approximation:

- A first repo implementation can be a slow PyTorch reference using an explicit depth cache and joint softmax.
- The official Triton kernel folds GQA groups into the query sequence dimension with `T_q = T_kv * moda_group_num`. That is a layout adapter, not a semantic change.

### Gotchas

- Do not confuse this with Mixture-of-Depths token skipping.
- Do not let depth attention see all tokens from all layers. Main MoDA sees prior-layer KV at the same token position only.
- Do not implement chunk-aware layout as a semantic change; chunk/group-aware layout is a performance detail.
- The repo's `parallel_moda_chunk_visible` variant can expose earlier token rows inside a chunk. Treat it as a separate experimental variant, not the paper-faithful default.
- Do not add a router or gate between sequence and depth branches. The single softmax allocates probability mass.
- Do not assume the official vision example includes FFN depth KV; the subagent read found it stores prior-block attention K/V.
- Do not assume production autoregressive decoding is solved by the released repo; no dedicated MoDA decode kernel was found.
- Cache implication: prefill/training needs depth KV over `T * depth_slots`; autoregressive decoding only needs a per-current-token depth buffer across earlier layers plus the normal sequence KV cache.

### Code Sketch

```python
def moda_attention_ref(q, k, v, depth_k, depth_v, causal=True):
    """
    q:       [B, T, Hkv, G, D]   conceptual GQA layout
    k, v:    [B, T, Hkv, Dv]
    depth_k: [B, T, L, Hkv, D]   token-major depth slots
    depth_v: [B, T, L, Hkv, Dv]
    returns: [B, T, Hkv, G, Dv]
    """
    B, T, Hkv, G, D = q.shape
    scale = D ** -0.5

    # Sequence logits: [B, Hkv, G, T_query, T_key]
    seq_logits = torch.einsum("bthgd,bshd->bhgts", q, k) * scale
    if causal:
        mask = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
        seq_logits = seq_logits.masked_fill(
            ~mask[None, None, None], float("-inf")
        )

    # Depth logits: each token t sees only its own depth slots.
    # [B, Hkv, G, T, L]
    depth_logits = torch.einsum("bthgd,btlhd->bhgtl", q, depth_k) * scale

    logits = torch.cat([seq_logits, depth_logits], dim=-1)
    probs = torch.softmax(logits, dim=-1)

    p_seq = probs[..., :T]
    p_dep = probs[..., T:]

    ctx_seq = torch.einsum("bhgts,bshv->bthgv", p_seq, v)
    ctx_dep = torch.einsum("bhgtl,btlhv->bthgv", p_dep, depth_v)
    return ctx_seq + ctx_dep
```

Official-kernel layout adapter, if using the released Triton API:

```python
# Conceptual q: [B, T, Hkv, G, D]
q_kernel = q.transpose(2, 3).reshape(B, T * G, Hkv, D)

# Depth slots are token-major then flattened for the kernel.
cached_k = torch.stack(prev_k_slots, dim=2).reshape(B, T * L, Hkv, D).contiguous()
cached_v = torch.stack(prev_v_slots, dim=2).reshape(B, T * L, Hkv, Dv).contiguous()

o_kernel = parallel_moda(
    q=q_kernel,
    k=k,
    v=v,
    g=None,
    cached_k=cached_k,
    cached_v=cached_v,
    moda_group_num=G,
    is_causal=True,
    head_first=False,
)
o = o_kernel.reshape(B, T, G, Hkv, Dv).transpose(2, 3)
```

Depth state update per layer:

```python
prev_k_slots.append(k_current)  # reuse attention K
prev_v_slots.append(v_current)

if use_ffn_depth_kv:
    ffn_k, ffn_v = ffn_kv_proj(ffn_input)
    prev_k_slots.append(ffn_k)
    prev_v_slots.append(ffn_v)
```

### Sources

- arXiv page: https://arxiv.org/abs/2603.15619
- arXiv PDF: https://arxiv.org/pdf/2603.15619
- arXiv source: https://arxiv.org/e-print/2603.15619
- Official repo: https://github.com/hustvl/MoDA
- Official files inspected by subagent: `libs/moda_triton/fla/ops/moda/moda_v14.py`, `libs/moda_triton/fla/ops/moda/moda_v16.py`, `libs/moda_triton/fla/ops/moda/fda_v12.py`, `libs/moda_triton/fla/ops/moda/__init__.py`, `vision_tasks/deit/models.py`, `libs/moda_triton/tests/ops/test_moda.py`, and `README.md`.

## 1. MoD: Mixture-of-Depths

Link: https://arxiv.org/pdf/2404.02258

### Short Summary

Mixture-of-Depths makes transformer depth conditional at the token level. Each block has a router that scores tokens, selects a fixed top-k capacity, runs only selected tokens through the block, and lets unselected tokens bypass the block unchanged. The main claim is that models can allocate FLOPs to harder tokens while preserving dense tensor-friendly capacity constraints.

High-level finding: routed-depth models can approach or improve dense-transformer quality at lower FLOPs when routing is trained end-to-end, but the routing primitive is nontrivial. The faithful primitive is top-k token selection through the whole block, not just masking an FFN or applying dense attention and dropping outputs.

### Primitive Contract

- For sequence length `S`, capacity `C < S`.
- Each block owns a scalar router `r_i = router(x_i)`.
- Select top `C` tokens per sequence.
- Gather selected tokens in original order.
- Run selected tokens through the full transformer block.
- Scatter updates back into the original sequence.
- Skipped tokens remain identity.
- Selected token update is router-weighted: `x_i_next = x_i + router_weight_i * block_delta_i`.
- Training top-k is non-causal over the full training sequence; autoregressive inference needs an approximation or auxiliary router.

### Gotchas

- Do not route only the MLP if claiming MoD fidelity.
- Do not run dense attention over all tokens and mask outputs after the fact.
- Do not include bypassed tokens in K/V for the routed block unless explicitly marking a deviation.
- Do not drop the router multiplier in the selected update.
- Do not pretend full-sequence top-k routing is naturally causal during incremental generation.

### Code Sketch

```python
import torch
import torch.nn as nn

class PaperFaithfulMoDBlock(nn.Module):
    def __init__(self, d_model, block, capacity_ratio=0.5):
        super().__init__()
        self.router = nn.Linear(d_model, 1)
        self.block = block
        self.capacity_ratio = capacity_ratio

    def forward(self, x, causal_mask_builder):
        # x: [B, S, D]
        B, S, D = x.shape
        C = max(1, int(round(self.capacity_ratio * S)))

        scores = self.router(x).squeeze(-1)             # [B, S]
        top_pos = scores.topk(C, dim=1).indices
        top_pos = top_pos.sort(dim=1).values             # preserve token order

        selected = x.gather(1, top_pos[..., None].expand(B, C, D))
        selected_scores = scores.gather(1, top_pos)
        selected_weight = selected_scores.sigmoid()[..., None]

        # Mask among selected original positions only.
        mask = causal_mask_builder(top_pos, top_pos)
        selected_next = self.block(selected, causal_mask=mask)
        delta = selected_next - selected

        out = x.clone()
        out.scatter_add_(1, top_pos[..., None].expand(B, C, D),
                         selected_weight * delta)
        return out, {"router_scores": scores, "selected": top_pos}
```

Sources: arXiv PDF/abstract/source for `2404.02258`; ar5iv rendering; no official implementation was found in the subagent read.

## 2. Recurrent Depth / Latent Dynamics

Link: https://arxiv.org/abs/2509.23314

### Short Summary

This paper studies recurrent-depth transformers as latent dynamical systems. Selected layers or layer groups are reused across multiple depth steps, and the model's hidden trajectory is analyzed with step norms, cosine similarity, drift, and second-order acceleration. The paper is especially relevant for diagnostics and early-exit signals, not just architecture.

High-level finding: recurrent depth creates measurable latent refinement dynamics. Fixed recurrence should be tested first; adaptive exit can then be based on hidden-state dynamics such as acceleration, not on expensive decoding at every step.

### Primitive Contract

- Use a decoder-only transformer with selected recurrent groups.
- The cited setup loops layer 4, a paired group of layers 5-6, and layer 7.
- Weights are shared inside each recurrent group.
- Training loop counts are sampled from a lognormal-Poisson schedule.
- Track `delta_k = h_{k+1} - h_k`.
- Acceleration-style exit uses `||delta_k - delta_{k-1}||`.
- Require at least two steps and often two consecutive below-threshold hits before halting.

### Gotchas

- Do not call independent extra layers "recurrent depth".
- Do not use decoding-based halting as the first primitive; it changes compute cost and objective.
- Do not omit trajectory diagnostics; they are part of the evidence surface.
- Some initialization and adapter details are inherited from Geiping-style recurrent pretraining and are not fully recoverable from the target paper alone.

### Code Sketch

```python
class RecurrentGroup(nn.Module):
    def __init__(self, blocks, max_steps=4, exit_tau=None, two_hit=True):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.max_steps = max_steps
        self.exit_tau = exit_tau
        self.two_hit = two_hit

    def one_step(self, h, **kw):
        for block in self.blocks:
            h = block(h, **kw)
        return h

    def forward(self, h, steps=None, **kw):
        steps = self.max_steps if steps is None else steps
        deltas = []
        below_hits = 0

        for k in range(steps):
            h_next = self.one_step(h, **kw)
            delta = h_next - h
            stats = {"step_norm": delta.norm(dim=-1).mean()}

            if deltas:
                accel = (delta - deltas[-1]).norm(dim=-1).mean()
                denom = delta.norm(dim=-1).mean() + deltas[-1].norm(dim=-1).mean() + 1e-8
                stats["accel"] = accel
                stats["normalized_accel"] = accel / denom

                if self.exit_tau is not None and k >= 1:
                    hit = stats["normalized_accel"] < self.exit_tau
                    below_hits = below_hits + 1 if bool(hit) else 0
                    if below_hits >= (2 if self.two_hit else 1):
                        h = h_next
                        deltas.append(delta)
                        break

            h = h_next
            deltas.append(delta)

        return h, {"steps_used": len(deltas), "deltas": deltas}
```

Sources: arXiv page/PDF for `2509.23314`; OpenReview/lab page; Geiping `2502.05171`; `seal-rg/recurrent-pretraining`.

## 3. Looped Transformers as Programmable Computers

Link: https://arxiv.org/abs/2301.13196

### Short Summary

This is a constructive expressivity paper. It hand-programs a shallow transformer encoder whose output sequence is fed back as the next input sequence. The sequence represents machine state: scratchpad, memory, instructions, and positional encodings. One loop executes one instruction-like update, allowing the transformer to emulate SUBLEQ and FLEQ-style computation.

High-level finding: looped transformers can act as programmable computers under hand-constructed weights and explicit scratchpad state. This is not a standard trained decoder LM architecture.

### Primitive Contract

- Use an external loop: `X <- TF(W, X)`.
- Input sequence is state, not just tokens.
- Positional information is appended as binary features, not added as embedding.
- Attention and MLP weights implement read/write/branch operations.
- SUBLEQ updates memory and program counter; FLEQ generalizes to function-call style blocks.

### Gotchas

- Do not claim a normal transformer block with recurrence is this paper's construction.
- Do not add standard learned positional embeddings if implementing the proof construction.
- Do not append CoT tokens; the scratchpad is explicit state.
- A semantic interpreter is a useful implementation contract, but not the same as the hand-coded transformer weights.

### Code Sketch

```python
class SubleqState:
    def __init__(self, memory, instructions, pc=0):
        self.memory = memory
        self.instructions = instructions
        self.pc = pc

def subleq_step(state):
    # instruction: (a, b, c)
    a, b, c = state.instructions[state.pc]
    state.memory[b] = state.memory[b] - state.memory[a]
    if state.memory[b] <= 0:
        state.pc = c
    else:
        state.pc += 1
    return state

class LoopedProgramComputer:
    def __init__(self, step_fn):
        self.step_fn = step_fn

    def run(self, state, max_steps):
        trace = []
        for _ in range(max_steps):
            state = self.step_fn(state)
            trace.append((list(state.memory), state.pc))
        return state, trace
```

A faithful transformer implementation would replace `subleq_step` with the paper's hard-coded attention/MLP circuits:

```python
def transformer_layer_formula(X, heads, ffn):
    # X: [D, S], column-major sequence
    Z = X
    for Q, K, V, temperature in heads:
        scores = temperature * (X.T @ K.T @ Q @ X)
        attn = scores.softmax(dim=0)
        Z = Z + V @ X @ attn
    return Z + ffn(Z)
```

Sources: arXiv page/PDF/source for `2301.13196`; official repository `jysohn1108/Looped-Transformer`.

## 4. Native Sparse Attention

Link: https://arxiv.org/pdf/2502.11089

### Short Summary

Native Sparse Attention is a trainable sparse attention mechanism for long-context LLMs. It combines three branches: compressed global blocks, selected fine-grained blocks, and local sliding-window attention. Branch outputs are gated and summed.

High-level finding: NSA aims to preserve quality while reducing long-context attention cost, but the sparse structure is not a simple mask. The compression branch and selection branch are learned and coupled.

### Primitive Contract

- Branch 1: compressed attention over learned block-compressed K/V.
- Branch 2: selected attention over fine-grained blocks chosen from compression-derived scores.
- Branch 3: local sliding-window attention.
- Outputs are gated and summed: `out = sum_i gate_i * branch_i`.
- Compression uses learned `phi`, not average pooling.
- Selection under GQA/MQA is shared across heads in the KV group.

### Gotchas

- Do not concatenate all sparse K/V into one softmax.
- Do not use average pooling as a faithful compression branch.
- Do not do dense attention first and only then select blocks.
- Do not ignore causality in block selection.
- Python gather/scatter may be primitive-faithful but not performance-faithful.

### Code Sketch

```python
def attend(q, k, v, mask=None):
    scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    p = scores.softmax(dim=-1)
    return p @ v, p

class NativeSparseAttention(nn.Module):
    def __init__(self, d_model, block_len=32, stride=16, top_blocks=16, window=512):
        super().__init__()
        self.block_len = block_len
        self.stride = stride
        self.top_blocks = top_blocks
        self.window = window
        self.phi_k = nn.Linear(block_len * d_model, d_model)
        self.phi_v = nn.Linear(block_len * d_model, d_model)
        self.gate = nn.Linear(d_model, 3)

    def compressed_kv(self, k, v):
        # Schematic: build strided causal blocks then apply learned phi.
        k_blocks = make_strided_blocks(k, self.block_len, self.stride)
        v_blocks = make_strided_blocks(v, self.block_len, self.stride)
        return self.phi_k(k_blocks.flatten(-2)), self.phi_v(v_blocks.flatten(-2))

    def forward(self, q, k, v, causal):
        kc, vc = self.compressed_kv(k, v)
        o_cmp, p_cmp = attend(q, kc, vc, mask=compressed_causal_mask(causal))

        selected_blocks = derive_top_blocks_from_compressed_probs(
            p_cmp, top_blocks=self.top_blocks
        )
        ks, vs, selected_mask = gather_fine_blocks(k, v, selected_blocks, causal)
        o_sel, _ = attend(q, ks, vs, mask=selected_mask)

        kw, vw, win_mask = gather_sliding_window(k, v, self.window, causal)
        o_win, _ = attend(q, kw, vw, mask=win_mask)

        g = self.gate(q).sigmoid()
        return g[..., 0:1] * o_cmp + g[..., 1:2] * o_sel + g[..., 2:3] * o_win
```

Sources: arXiv page/PDF/source for `2502.11089`; ACL entry; Hugging Face paper metadata.

## 5. Recurrent Networks: Easy-to-Hard Generalization

Link: https://arxiv.org/abs/2106.04537

### Short Summary

The paper shows that recurrent residual networks can learn algorithmic procedures on easy instances and generalize to harder instances by running more recurrent iterations at test time. The models are convolutional recurrent residual networks, not GRUs/LSTMs.

High-level finding: weight sharing plus extra test-time recurrence can improve extrapolation on algorithmic tasks such as prefix sums, maze solving, and chess move prediction.

### Primitive Contract

- Fully convolutional residual network.
- Encoder convolution, shared recurrent residual module, head convolutions.
- No batch norm and no convolution bias in the reported primitive.
- Train with fixed recurrences; test with more recurrences.
- Select best iteration by confidence for some tasks.

### Gotchas

- Do not replace this with an LSTM/GRU.
- Do not use fully connected heads for spatial tasks.
- Do not remove weight sharing.
- Do not claim LM relevance without adapting the task and representation.

### Code Sketch

```python
class TwoConvResidual(nn.Module):
    def __init__(self, conv, channels):
        super().__init__()
        self.net = nn.Sequential(
            conv(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            conv(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)

class RecurrentResidualPrimitive(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, conv=nn.Conv2d):
        super().__init__()
        self.encoder = conv(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.recurrent = nn.Sequential(
            TwoConvResidual(conv, hidden_ch),
            TwoConvResidual(conv, hidden_ch),
        )
        self.head = nn.Sequential(
            conv(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.ReLU(),
            conv(hidden_ch, hidden_ch // 2, 3, padding=1, bias=False),
            nn.ReLU(),
            conv(hidden_ch // 2, out_ch, 3, padding=1, bias=False),
        )

    def forward(self, x, steps):
        h = self.encoder(x)
        logits_by_step = []
        for _ in range(steps):
            h = self.recurrent(h)
            logits_by_step.append(self.head(h))
        return logits_by_step
```

Sources: arXiv page/PDF for `2106.04537`; NeurIPS/OpenReview; official repository `aks2203/easy-to-hard`.

## 6. Universal Transformer

Link: https://arxiv.org/abs/1807.03819

### Short Summary

Universal Transformer ties a transformer block across depth steps. All token positions update in parallel at each recurrent depth step, with position-plus-time coordinate embeddings injected each step. The paper also uses Adaptive Computation Time for per-position halting.

High-level finding: recurrence over depth improves algorithmic and language tasks in several settings. ACT helps structured tasks and some language tasks, but is not universally beneficial.

### Primitive Contract

- Shared self-attention plus transition block reused for `T` steps.
- Add coordinate embedding at every step: position plus time.
- Encoder step: attention with residual/norm, then transition with residual/norm.
- Decoder step: masked self-attention, cross-attention, transition.
- ACT is per-position, with accumulated halt probabilities and weighted state interpolation.

### Gotchas

- Do not use separate layers if claiming UT.
- Do not add timing embedding only once.
- Do not use global sequence-level halting for ACT.
- Do not output only the last transformed state under ACT; use weighted interpolation.

### Code Sketch

```python
class UniversalTransformerEncoder(nn.Module):
    def __init__(self, d_model, mha, transition, max_steps):
        super().__init__()
        self.mha = mha
        self.transition = transition
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.max_steps = max_steps
        self.halt = nn.Linear(d_model, 1)

    def coordinate(self, seq_len, step, device):
        return sinusoidal_position_time_embedding(seq_len, step, device)

    def step(self, h, step, mask=None):
        h = h + self.coordinate(h.shape[1], step, h.device)
        a = self.norm1(h + self.mha(h, h, h, mask=mask))
        return self.norm2(a + self.transition(a))

    def forward_fixed(self, h, mask=None):
        for t in range(self.max_steps):
            h = self.step(h, t, mask)
        return h

    def forward_act(self, h, mask=None, threshold=0.99):
        B, S, D = h.shape
        halting = h.new_zeros(B, S)
        remainders = h.new_zeros(B, S)
        n_updates = h.new_zeros(B, S)
        weighted = h.new_zeros(B, S, D)

        for t in range(self.max_steps):
            p = self.halt(h).sigmoid().squeeze(-1)
            still = halting < threshold
            new_halted = still & ((halting + p) > threshold)
            still_running = still & ~new_halted

            remainders[new_halted] = 1.0 - halting[new_halted]
            update_w = torch.where(new_halted, remainders, torch.where(still_running, p, 0.0))
            halting = halting + update_w
            n_updates = n_updates + still.float()

            h_next = self.step(h, t, mask)
            weighted = weighted + update_w[..., None] * h_next
            h = h_next

        return weighted, {"remainders": remainders, "n_updates": n_updates}
```

Sources: arXiv page/PDF/source for `1807.03819`; official Tensor2Tensor Universal Transformer files.

## 7. Looped Transformers Are Better at Learning Learning Algorithms

Link: https://arxiv.org/abs/2311.12424

### Short Summary

This paper trains a decoder transformer on synthetic in-context learning tasks by repeatedly applying the same GPT-2-style backbone. The key recurrence is not plain weight tying: each loop receives the current loop state plus the original embedded prompt.

High-level finding: a 1-layer looped transformer can match a 12-layer baseline on several synthetic learning problems with far fewer transformer-layer parameters, but it requires sufficient loop count and a truncated loss window for stability.

### Primitive Contract

- Prompt is interleaved `x_1, y_1, ..., x_i, y_i, x_{i+1}`.
- Initial loop state `Y_0 = 0`.
- Each loop: `Y_{t+1} = M_theta(Y_t + P)`.
- `M_theta` is a GPT-2-style causal decoder.
- Read predictions at `x` token positions.
- Train over a final loop window of length `T`; earlier loops may run no-grad for memory.

### Gotchas

- Do not use `Y_{t+1} = M(Y_t)` with `Y_0 = P`; the paper finds this loses the input signal.
- Do not move learned position embeddings outside the loop if reproducing the official setup.
- Do not train only the final query; the paper trains across prompt prefixes.
- Do not instantiate separate parameters for each loop.

### Code Sketch

```python
class LoopedTransformerPrimitive(nn.Module):
    def __init__(self, read_in, backbone, read_out):
        super().__init__()
        self.read_in = read_in
        self.backbone = backbone
        self.read_out = read_out

    def combine_prompt(self, xs, ys):
        y_tokens = torch.zeros_like(xs)
        y_tokens[..., 0] = ys
        return torch.stack((xs, y_tokens), dim=2).reshape(xs.shape[0], -1, xs.shape[-1])

    def forward(self, xs, ys, loops, loss_window):
        P = self.read_in(self.combine_prompt(xs, ys))
        Y = torch.zeros_like(P)
        start_grad = max(0, loops - loss_window)
        preds = []

        for t in range(loops):
            if t < start_grad:
                with torch.no_grad():
                    Y = self.backbone(inputs_embeds=Y + P)
                Y = Y.detach()
            else:
                Y = self.backbone(inputs_embeds=Y + P)
                preds.append(self.read_out(Y[:, 0::2]).squeeze(-1))
        return preds

def looped_learning_loss(model, xs, ys, loops, loss_window):
    preds = model(xs, ys, loops, loss_window)
    return torch.stack([(p - ys).square().mean() for p in preds]).mean()
```

Sources: arXiv page/PDF/source for `2311.12424`; official repository `Leiay/looped_transformer`.

## 8. Pause Tokens

Link: https://arxiv.org/abs/2310.02226

### Short Summary

Pause Tokens add one learnable special token and manually insert repeated copies to give a decoder LM extra computation positions before answering. The tokens are not generated as reasoning text; they are prefilled. Loss is delayed until after the final pause.

High-level finding: pause tokens help mainly when used during both pretraining and finetuning. Finetuning-only pauses on a standard pretrained model are mixed. Filler punctuation is not equivalent.

### Primitive Contract

- Add one special token, e.g. `<pause>`.
- Pretraining: randomly insert pauses into LM sequences; skip CE targets where the next token is pause.
- Finetuning: append `M` pauses after the prompt and supervise target tokens starting after the final pause.
- Inference: append pauses manually, then generate answer tokens after final pause.
- Do not train the model to emit pause tokens.

### Gotchas

- Do not use `"."` or another normal token as a faithful substitute.
- Do not add many distinct prompt tokens; it is one shared token repeated.
- Do not detach pause hidden states; only mask output loss.
- Do not call `generate` and expect pauses to appear.
- Pauses consume context length.

### Code Sketch

```python
IGNORE = -100

def add_pause_token(tokenizer, model, token="<pause>"):
    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(token)

def pause_pretrain_loss(model, ids, pause_id, pause_fraction=0.10):
    B, L = ids.shape
    m = round(pause_fraction * L)
    injected = torch.stack([
        random_insert_pauses_1d(ids[b], pause_id, m, final_len=L)
        for b in range(B)
    ])
    x = injected[:, :-1]
    y = injected[:, 1:].clone()
    y[y == pause_id] = IGNORE
    logits = model(input_ids=x).logits
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1), ignore_index=IGNORE)

def pause_finetune_loss(model, prefix_ids, target_ids, pause_id, m_ft):
    pauses = prefix_ids.new_full((m_ft,), pause_id)
    seq = torch.cat([prefix_ids, pauses, target_ids])
    x = seq[:-1].unsqueeze(0)
    labels = torch.full_like(seq[1:], IGNORE)
    labels[prefix_ids.numel() + m_ft - 1:] = seq[1:][prefix_ids.numel() + m_ft - 1:]
    logits = model(input_ids=x).logits[0]
    return F.cross_entropy(logits, labels, ignore_index=IGNORE)
```

Sources: arXiv page/PDF for `2310.02226`; ICLR/OpenReview; Google Research publication page.

## 9. Reasoning With Latent Thoughts: Looped Transformers

Link: https://arxiv.org/abs/2502.17416

### Short Summary

This paper studies a shallow transformer block looped repeatedly with shared weights. It argues that reasoning can benefit from effective depth even when parameter count is small. The latent thoughts are hidden-state loop iterations, not decoded natural-language tokens.

High-level finding: looped transformers can improve reasoning-style tasks more than perplexity would predict. The paper also proves several expressivity results, including simulating repeated layers and multi-step reasoning with loops.

### Primitive Contract

- Embed once.
- Apply the same `k`-layer transformer block `L` times.
- Output once from final loop state.
- Train with ordinary causal LM loss for LM experiments.
- No CoT supervision in the main looped LM primitive.

### Gotchas

- Do not instantiate `k * L` independent layers.
- Do not add latent thought tokens for the main method.
- Do not re-add position embeddings every loop for the formal primitive.
- Do not treat loops as autoregressive token-generation steps.

### Code Sketch

```python
class KLayerBlock(nn.Module):
    def __init__(self, layer_factory, k):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(k)])

    def forward(self, h, causal_mask=None):
        for layer in self.layers:
            h = layer(h, causal_mask=causal_mask)
        return h

class LoopedTransformerLM(nn.Module):
    def __init__(self, token_emb, pos_emb, block, lm_head, loops):
        super().__init__()
        self.token_emb = token_emb
        self.pos_emb = pos_emb
        self.block = block
        self.lm_head = lm_head
        self.final_norm = nn.LayerNorm(token_emb.embedding_dim)
        self.loops = loops

    def forward(self, input_ids):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.token_emb(input_ids) + self.pos_emb(pos)[None]
        mask = causal_mask(input_ids.shape[1], input_ids.device)
        states = []
        for _ in range(self.loops):
            h = self.block(h, causal_mask=mask)
            states.append(h)
        return self.lm_head(self.final_norm(h)), states
```

Sources: arXiv page/PDF/source for `2502.17416`; OpenReview ICLR 2025; ar5iv rendering.

## 10. Reasoning in Latent Space: Coconut

Link: https://arxiv.org/abs/2412.06769

### Short Summary

Coconut replaces parts of chain-of-thought text with continuous hidden states. During latent positions, the model feeds the previous hidden state directly as the next input embedding, bypassing the LM head and token embedding lookup.

High-level finding: continuous latent reasoning can reduce visible tokens and perform well on some reasoning tasks, especially where latent search is beneficial. It is not equivalent to pause tokens, because the latent state is carried forward as the next embedding.

### Primitive Contract

- Mark a latent span with `<bot>` and `<eot>`.
- For a latent position, use the previous hidden state as the next input embedding.
- Do not project, quantize, sample, decode, or re-embed latent thoughts.
- Ignore LM loss at latent positions.
- Gradients flow through latent hidden states.
- Curriculum replaces initial CoT steps with latent thoughts.

### Gotchas

- Do not detach latent hidden states.
- Do not add a reconstruction loss unless intentionally extending.
- Do not treat latent positions as pause tokens.
- Batching needs aligned latent spans or per-example handling.

### Code Sketch

```python
class CoconutPrimitive(nn.Module):
    def __init__(self, causal_lm, bot_id, eot_id, latent_id):
        super().__init__()
        self.lm = causal_lm
        self.bot_id = bot_id
        self.eot_id = eot_id
        self.latent_id = latent_id

    def forward(self, input_ids, labels):
        embeds = self.lm.get_input_embeddings()(input_ids)
        latent_positions = (input_ids == self.latent_id).nonzero(as_tuple=False)

        # Schematic single-example sequential prefix update.
        for _, pos in latent_positions:
            out = self.lm(inputs_embeds=embeds[:, :pos], output_hidden_states=True)
            prev_h = out.hidden_states[-1][:, -1]
            embeds[:, pos] = prev_h

        logits = self.lm(inputs_embeds=embeds).logits
        labels = labels.clone()
        labels[input_ids == self.latent_id] = -100
        return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
```

Sources: arXiv page/PDF/source for `2412.06769`; official `facebookresearch/coconut` repository.

## 11. Scaling Test-Time Compute With Latent Reasoning

Link: https://arxiv.org/abs/2502.05171

### Short Summary

This paper introduces a recurrent-depth decoder LM, released as Huginn, that scales test-time compute by applying a shared recurrent core many times before the coda/head predicts tokens. It uses prelude, recurrent core, and coda layers, with input embedding injected into each recurrent step through a learned adapter.

High-level finding: increasing recurrent depth improves some tasks, especially math/code/reasoning, while easy tasks saturate earlier. The model learns latent dynamics without CoT supervision.

### Primitive Contract

- `e = P(x)`.
- `s0` is sampled latent state.
- `s_i = R(e, s_{i-1})` for `i = 1..r`.
- `p = C(s_r)`.
- Prelude: 2 layers; recurrent core: 4 shared layers; coda: 2 layers in the released model.
- Each recurrence receives both current state and fixed input embedding through concat plus learned linear adapter.
- Train with random recurrent depths sampled from lognormal-Poisson.
- Large run uses truncated BPTT through last 8 recurrent iterations.
- Uses sandwich transformer block and special initialization.

### Gotchas

- Do not inject the input only once.
- Do not replace concat adapter with simple addition if claiming large-model fidelity.
- Do not use normal pre-norm instead of sandwich block for faithful reproduction.
- Do not backprop through all recurrences unless intentionally departing.
- KV cache must account for recurrent step.

### Code Sketch

```python
def sample_depth(mean_r=32, k=8, sigma=0.5, device="cpu"):
    tau = torch.normal(
        mean=torch.tensor(math.log(mean_r) - 0.5 * sigma * sigma, device=device),
        std=torch.tensor(sigma, device=device),
    )
    r = torch.poisson(torch.exp(tau)).long() + 1
    grad_steps = torch.minimum(torch.tensor(k, device=device), r)
    no_grad_steps = torch.clamp(r - k, min=0)
    return int(no_grad_steps), int(grad_steps)

class RecurrentDepthLM(nn.Module):
    def __init__(self, embed, prelude, core, coda, adapter, lm_head, mean_r=32):
        super().__init__()
        self.embed = embed
        self.prelude = nn.ModuleList(prelude)
        self.core = nn.ModuleList(core)
        self.coda = nn.ModuleList(coda)
        self.adapter = adapter
        self.lm_head = lm_head
        self.mean_r = mean_r

    def recurrent_step(self, s, e, **kw):
        h = self.adapter(torch.cat([s, e], dim=-1))
        for block in self.core:
            h = block(h, **kw)
        return h

    def forward(self, input_ids, num_steps=None):
        e = self.embed(input_ids) * (self.embed.embedding_dim ** 0.5)
        for block in self.prelude:
            e = block(e)

        s = torch.randn_like(e)
        if num_steps is None and self.training:
            n_ng, n_g = sample_depth(self.mean_r, device=e.device)
        else:
            n_ng, n_g = 0, (num_steps or self.mean_r)

        with torch.no_grad():
            for t in range(n_ng):
                s = self.recurrent_step(s, e, step_idx=t)
        for t in range(n_g):
            s = self.recurrent_step(s, e, step_idx=n_ng + t)

        h = s
        for block in self.coda:
            h = block(h)
        return self.lm_head(h)
```

Sources: arXiv page/PDF/source for `2502.05171`; official `seal-rg/recurrent-pretraining`; official Huginn model card/config/minimal code.

## 12. Scaling Latent Reasoning via Looped Language Models: Ouro

Link: https://arxiv.org/abs/2510.25741

### Short Summary

Ouro reuses a decoder transformer stack for a fixed maximum number of recurrent steps and attaches an LM head plus exit gate at each step. It trains a probability distribution over exit depths with entropy-regularized expected next-token loss, then optionally trains the gate with a focused continuation target.

High-level finding: loop depth is treated as a third scaling axis alongside parameters and data. The reported models can match larger standard baselines on several benchmarks, with gains coming from knowledge manipulation rather than raw fact storage.

### Primitive Contract

- Shared decoder stack reused for `T_max` steps.
- At each step, produce logits and an exit gate.
- Gate `lambda_t = sigmoid(linear(h_t))`.
- Exit probability:
  - `p_t = survival * lambda_t` for steps before final.
  - final step gets all remaining survival mass.
- Stage 1 loss: expected CE across step distribution minus `beta * entropy`.
- Stage 2: freeze LM, train gate with continuation targets from per-token loss improvement.
- Inference: deterministic quantile exit.

### Gotchas

- Do not instantiate separate layers per loop.
- Do not append latent tokens.
- Do not use a geometric prior; the paper uses a uniform-prior style entropy term.
- Do not train gates only with task loss; collapse is a known failure mode.
- KV cache must index recurrent depth.

### Code Sketch

```python
def exit_pdf(gate_logits):
    lambdas = [g.sigmoid() for g in gate_logits]
    survival = torch.ones_like(lambdas[0])
    probs = []
    for t, lam in enumerate(lambdas):
        if t < len(lambdas) - 1:
            p = survival * lam
            survival = survival * (1.0 - lam)
        else:
            p = survival
        probs.append(p)
    return torch.stack(probs, dim=-1)

class OuroPrimitive(nn.Module):
    def __init__(self, embed, shared_stack, norm, lm_head, d_model, steps=4):
        super().__init__()
        self.embed = embed
        self.stack = shared_stack
        self.norm = norm
        self.lm_head = lm_head
        self.exit_gate = nn.Linear(d_model, 1)
        self.steps = steps

    def forward_steps(self, input_ids):
        h = self.embed(input_ids)
        logits, gates, states = [], [], []
        for step in range(self.steps):
            h = self.stack(h, step_idx=step)
            z = self.norm(h)
            states.append(z)
            logits.append(self.lm_head(z))
            gates.append(self.exit_gate(z).squeeze(-1))
        return logits, gates, states

def ouro_stage1_loss(model, input_ids, beta=0.05):
    logits, gates, _ = model.forward_steps(input_ids)
    p = exit_pdf(gates)[:, :-1]
    ce = torch.stack([next_token_ce(z, input_ids) for z in logits], dim=-1)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)
    return ((p * ce).sum(dim=-1) - beta * entropy).mean()
```

Sources: arXiv page/source for `2510.25741`; official Ouro project page; ByteDance Ouro model cards/configs/remote modeling code.

## 13. Relaxed Recursive Transformers

Link: https://arxiv.org/abs/2410.20672

### Short Summary

Relaxed Recursive Transformers compress a pretrained decoder transformer by replacing `L` unique layers with a smaller shared block of `K = L / B` layers, applied `B` times. Strict recursion ties all block parameters. Relaxed recursion adds depth-specific LoRA deltas to shared linear matrices.

High-level finding: recursive and relaxed-recursive uptraining can recover much of a full model's quality with fewer stored parameters. The strongest relaxed recipe averages tied full-model layers into the shared block and initializes per-depth LoRA from SVD of residual weights.

### Primitive Contract

- Require `L % B == 0`.
- Use CYCLE sharing: absolute depth `ell` uses shared layer `ell % K`.
- Strict recursion shares all block parameters, including RMSNorm.
- Relaxed recursion adds per-absolute-depth LoRA for every shared linear matrix.
- Rank 0 equals strict recursion.
- Train all parameters after conversion; this is not adapter-only tuning.
- KV caches are indexed by absolute depth, not shared layer id.
- Early exits are logits after completed loops; no learned halting module is part of the paper.

### Gotchas

- Do not clone layers with identical weights; use real parameter sharing.
- Do not use Stepwise plus LoRA as the default relaxed recipe; Average plus SVD is strongest.
- Do not treat throughput claims as measured production throughput.
- Do not add ACT/router losses as if they were in the paper.

### Code Sketch

```python
def cycle_groups(num_layers, num_loops):
    assert num_layers % num_loops == 0
    k = num_layers // num_loops
    return [[j + loop * k for loop in range(num_loops)] for j in range(k)]

class DepthLoRA(nn.Module):
    def __init__(self, out_features, in_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))

    @torch.no_grad()
    def init_from_residual(self, residual):
        U, S, Vh = torch.linalg.svd(residual.float(), full_matrices=False)
        r = self.rank
        self.B.copy_((U[:, :r] * S[:r]).to(self.B.dtype))
        self.A.copy_(Vh[:r].to(self.A.dtype))

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

class RelaxedRecursiveTransformer(nn.Module):
    def __init__(self, full_model, num_loops=2, lora_rank=0):
        super().__init__()
        self.embed = full_model.embed
        self.final_norm = full_model.final_norm
        self.lm_head = full_model.lm_head
        self.L = len(full_model.layers)
        self.B = num_loops
        self.K = self.L // self.B

        self.block = average_tied_layers(full_model.layers, cycle_groups(self.L, self.B))
        self.lora = build_depth_lora_banks(self.block, self.L, lora_rank)
        if lora_rank > 0:
            svd_init_lora_from_full_model(self.lora, full_model.layers, self.block)

    def forward(self, input_ids, cache=None, return_exits=False):
        h = self.embed(input_ids)
        exits = []
        for ell in range(self.L):
            j = ell % self.K
            h = self.block[j](
                h,
                cache=None if cache is None else cache[ell],
                lora_bank=self.lora[ell],
            )
            if return_exits and j == self.K - 1:
                exits.append(self.lm_head(self.final_norm(h)))
        return {"logits": self.lm_head(self.final_norm(h)), "exit_logits": exits}
```

Sources: arXiv page/PDF/source for `2410.20672`; Hugging Face paper page. No official author code was found in the subagent read.

## 14. Mixture of Recursions

Link: https://arxiv.org/abs/2507.10524

### Short Summary

Mixture-of-Recursions combines recursive layer sharing with token-level adaptive compute. A shared middle transformer stack is reused for up to `N_r` recursions. Routers decide which tokens continue to deeper recursion. Tokens that exit early skip later shared-block computation.

High-level finding: MoR improves quality/efficiency tradeoffs versus vanilla transformers and fixed recursive transformers in reported regimes. Expert-choice routing is the strongest reported setup; token-choice is harder to balance. Larger-scale behavior is left as future work.

### Primitive Contract

- Keep explicit recursion state: hidden tensor, original token positions, active token indices, router outputs, and KV policy.
- Share transformer block parameters across recursions.
- Main sharing strategy: Middle-Cycle, with first and last layers unique and the middle block cyclically shared.
- Expert-choice routing:
  - one scalar router per recursion,
  - score currently active tokens,
  - select top-k by capacity,
  - run only selected tokens through shared stack,
  - scatter router-weighted updates back,
  - only selected tokens can be considered at the next recursion.
- Token-choice routing:
  - one router chooses depth for each token upfront,
  - token runs through all recursive prefixes up to its assigned depth.
- Training is LM loss plus router regularization; no ACT halting loss.
- KV policy must be explicit: recursion-wise cache or recursive KV sharing.

### Gotchas

- Do not run all tokens through all recursions and mask afterward.
- Sort selected indices before attention to preserve causal order.
- Do not silently mix recursion-wise caching and recursive sharing.
- Do not add ponder/halting losses unless labeling an extension.
- Full-sequence expert-choice top-k is non-causal for autoregressive decoding unless mitigated.

### Code Sketch

```python
class ExpertChoiceMoR(nn.Module):
    def __init__(self, shared_stack, routers, capacities, alpha=0.1):
        super().__init__()
        self.shared_stack = shared_stack
        self.routers = nn.ModuleList(routers)
        self.capacities = capacities
        self.alpha = alpha

    def forward(self, h, causal_mask, pos, kv_policy):
        B, T, D = h.shape
        active = torch.arange(T, device=h.device).expand(B, T)
        aux = h.new_tensor(0.0)

        for r, router in enumerate(self.routers):
            x = batched_gather(h, active)
            logits = router(x).squeeze(-1)
            gate = logits.sigmoid() * self.alpha

            k = min(active.shape[1], max(1, int(self.capacities[r] * T)))
            local = gate.topk(k, dim=1).indices.sort(dim=1).values
            idx = batched_gather(active, local)
            w = batched_gather(gate, local).unsqueeze(-1)

            y_in = batched_gather(h, idx)
            y = self.shared_stack(
                y_in,
                mask=sub_causal_mask(causal_mask, idx, idx),
                pos=batched_gather(pos, idx),
                cache=kv_policy.view_for_depth(r, idx),
            )

            h = batched_scatter_add(h, idx, w * y)
            aux = aux + bce_topk_targets(logits, local)
            active = idx

        return h, aux

class TokenChoiceMoR(nn.Module):
    def __init__(self, shared_stack, depth_router, num_recursions):
        super().__init__()
        self.shared_stack = shared_stack
        self.depth_router = depth_router
        self.num_recursions = num_recursions

    def forward(self, h, causal_mask, pos, kv_policy):
        logits = self.depth_router(h)
        probs = logits.softmax(dim=-1)
        depth = probs.argmax(dim=-1)
        work = h
        out = h.clone()

        for r in range(self.num_recursions):
            idx = positions_where(depth >= r)
            y = self.shared_stack(
                batched_gather(work, idx),
                mask=sub_causal_mask(causal_mask, idx, idx),
                pos=batched_gather(pos, idx),
                cache=kv_policy.view_for_depth(r, idx),
            )
            work = batched_scatter(work, idx, y)

            done = filter_positions(idx, depth, equals=r)
            out = batched_scatter_add(
                out, done, probs_at(done, r).unsqueeze(-1) * batched_gather(work, done)
            )

        return out, switch_balance_loss(probs, depth), router_z_loss(logits)
```

Sources: arXiv page/PDF/source for `2507.10524`; official repository `raymin0223/mixture_of_recursions`; official model, router, cache, and sharing-strategy files.

## Implementation Implications For The Fractal Harness

The next architecture work should start with primitive-faithful baselines, not hybrids. The list below is intentionally broader than the first eight architecture-kernel baselines: the omitted papers are mostly implementable, but some belong to data/interface or task-specific tracks rather than the core transformer-block track.

### Core Architecture Baselines

These are decoder/attention/recursion primitives that fit naturally into the existing model harness.

1. MoDA baseline: joint-softmax attention over current sequence KV plus same-token prior-depth KV.
2. MoD baseline: true top-k gather/scatter routed block, no dense all-token block compute.
3. NSA baseline: three separate sparse attention branches with gated sum.
4. Fixed looped LM baseline: embed once, reuse the same `k`-layer block for `L` loops, output once.
5. Input-injected loop baseline: `Y <- M(Y + P)` or adapter `R(e, s)` depending on target paper.
6. Universal Transformer baseline: tied recurrent block with step/time coordinate embeddings.
7. UT/ACT baseline: per-token halt distribution and weighted state interpolation.
8. Recurrent-depth dynamics baseline: shared recurrent groups plus step-norm, cosine, drift, and acceleration diagnostics.
9. Ouro-style learned exit baseline: looped stack with per-step LM heads, exit distribution, entropy-regularized expected CE, and Q-exit inference.
10. Recursive compression baseline: CYCLE layer sharing with absolute-depth cache, then optional RRT LoRA.
11. MoR baseline: Middle-Cycle recursive sharing plus expert-choice routing with real active-token shrinkage.

### Data And Interface Baselines

These are paper-faithful and implementable, but they change how examples are represented or how forward passes are staged rather than only changing one transformer block.

1. Pause-token baseline: add one learned `<pause>` token, insert/prepend pauses manually, mask pause targets during pretraining, and start answer generation after the final pause.
2. Coconut latent-thought baseline: replace selected CoT spans with latent positions where the previous hidden state is fed directly as the next input embedding.
3. Coconut curriculum baseline: stage training so early CoT steps are progressively replaced by fixed-length latent thought spans.

### Task-Specific Or Semantic Baselines

These are implementable, but they should not be forced into the default LM benchmark unless the harness has matching tasks.

1. Programmed loop/scratchpad baseline: implement the SUBLEQ/FLEQ-style scratchpad interpreter or a tiny function-block execution path as a semantic contract for the programmable-computer paper.
2. Algorithmic recurrent-residual baseline: implement the recurrent residual ConvNet only for prefix-sum, maze, chess, or similarly structured algorithmic tasks.
3. Confidence-vs-iteration evaluation: for algorithmic recurrence, evaluate more recurrent steps at test time and select by confidence where the task supports it.

### Defer Unless Evidence Justifies

These are not rejected, but they should follow primitive baselines and smoke tests.

1. MoDA chunk-visible variants, because they alter the same-token depth-memory semantics.
2. Recursive KV sharing variants for MoR, because they change attention semantics relative to recursion-wise caching.
3. RRT early-exit serving simulations, because the paper's throughput claims rely on oracle/post-training assumptions.
4. Full symbolic computer or large scratchpad system, because phase 1 only needs a compact semantic primitive.
5. Combined hybrids such as MoDA plus MoR plus Ouro exits, until each primitive has an isolated baseline.

Hybrids should wait until the harness has at least one regression/smoke test per primitive that proves the intended contract:

- Parameter sharing is real.
- Depth memory is same-token KV retrieval and uses a single joint attention softmax.
- Routed tokens are actually skipped computationally.
- Pause and latent-thought examples mask losses exactly where the source paper requires.
- Cache indexing includes recursion/depth where needed.
- Halting or exit loss matches the source paper.
- Metrics expose steps used, selected tokens, exit probabilities, token-depth histograms, and depth-KV mass.
