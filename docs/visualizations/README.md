# Architecture Visualization Bundle

This directory contains four standalone HTML pages for thinking through the
architecture pivot visually:

- `gpt.html`
- `hybrid.html`
- `fractal-v1.html`
- `fractal-v2.html`

## What these pages are for

They are not benchmark dashboards.

They are 3D mental-model tools for comparing:
- transformer-style token memory
- hybrid attention/state designs
- the single-root recursive design we tried
- the new recursive-memory architecture direction

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
