# Architecture Visualization Bundle

This directory contains eight standalone HTML pages for thinking through the
architecture pivot visually:

- `gpt.html`
- `hybrid.html`
- `p20.html`
- `fractal-v1.html`
- `fractal-v2.html`
- `native-internal-search.html`
- `v3a-hybrid-attention.html`
- `thought-channel-hybrid.html`

## What these pages are for

They are not benchmark dashboards.

Most are 3D mental-model tools for comparing:
- transformer-style token memory
- hybrid attention/state designs
- the P20 rotary gated recurrent control lane
- the single-root recursive design we tried
- the new recursive-memory architecture direction

`native-internal-search.html` is a rendered Mermaid concept page rather than a
3D scene. It is there to make a speculative architecture legible fast.

`v3a-hybrid-attention.html` is a rendered Mermaid page for the current
predictive-core Path 1 hybrid stack: local exact attention interleaved with the
reference SSM lane or the `P2` contender lane under a matched budget.

`p20.html` is a 3D page inspired by Brendan Bycroft's llm-viz style. It shows
the P20 proof-ladder model as a component-level flow: prelude residual stream,
packed recurrent controls, rotary gated state update, emitted control readout,
and looped-middle attention injection seam.

`thought-channel-hybrid.html` is a rendered Mermaid page for a more speculative
native-search architecture: a shared attention trunk plus multiple active
thought channels with compare, prune, merge, and halt control.

## How to open them

Fastest path:
- open `index.html`

If your browser blocks local script features under `file://`, serve the folder:

```bash
cd /Users/joseph/fractal
python3 -m http.server 8000 -d docs/visualizations
```

Then open:
- `http://localhost:8000/`

## Interaction

- drag to rotate
- scroll to zoom
- use the mode buttons to switch runtime views

## Reading tip

The most useful comparison is usually:
1. open two pages side by side
2. pick the same runtime phase on both
3. compare the memory story and the retrieval story first

That tends to make the architectural tradeoffs legible much faster than reading
the prose alone.
