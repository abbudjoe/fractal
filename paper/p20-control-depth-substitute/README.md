# P2.0 Rotary State-Output Control Report

This folder contains a buildable LaTeX technical report for the first Path 1
Parcae/P2.0 rotary state-output control H100 proof ladder.

## Build

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

TinyTeX is sufficient on the local Mac environment.

## Scope

The report is intentionally framed as a technical report / compute-grant
artifact, not as a final architecture paper. The supported claim is narrow:
the P2.0 rotary state-output recurrent control, called `P20-control` in the
repository, behaves like an efficient depth substitute in the current tiny-LM
Path 1 benchmark, while much deeper pure attention can still pass it.

The primary source of truth for run artifacts is:

```text
docs/specs/parcae-p20-control-h100-proof-ladder.md
```
