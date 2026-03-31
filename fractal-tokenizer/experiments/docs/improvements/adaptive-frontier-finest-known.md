# Adaptive Frontier: Finest Known

## What It Is

Always recurse through known motifs when a finer known child frontier exists, and emit the finest covering non-overlapping known frontier.

## How It Works

- start at the root
- if a node has children that cleanly cover its span, recurse
- only emit the parent when no finer covering frontier exists
- fall back to bytes only when the motif tree cannot supply a known frontier

## Why It Was A Good Candidate

- directly fixes root-collapse without adding heuristic knobs
- deterministic
- preserves exact round-trip
- gives a real token frontier instead of a whole-document token

## Status

`Tried`

## Outcome

- successful
- exact round-trip preserved
- zero fallback on the benchmark corpus with a full induced vocab
- benchmark inputs now emit a useful frontier instead of one token
- current behavior resolves to a uniform `32`-token leaf frontier for the standard stress and mixed-domain inputs

## Decision

Keep as the current default model-facing policy until a better adaptive frontier policy beats it.
