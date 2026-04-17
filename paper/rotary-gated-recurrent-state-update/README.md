# Rotary Gated Recurrent State Update Report

This folder contains a buildable LaTeX technical report for the first
looped-scaffold H100 proof ladder using the rotary gated recurrent state update
primitive.

## Build

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

TinyTeX is sufficient on the local Mac environment.

## Scope

The report is intentionally framed as a technical report / compute-grant
artifact, not as a final architecture paper. The supported claim is narrow:
the rotary gated recurrent state update primitive behaves like an efficient
depth substitute in the current tiny-LM benchmark, while much deeper pure
attention can still pass it.

The primary source of truth for run artifacts is the proof-ladder scorecard
under `docs/specs/`.
