# Adaptive Frontier: Greedy Known

## What It Is

Emit the first known motif encountered in the recursion tree and stop descending at that node.

## How It Works

- start at the root
- if the current motif is in vocab, emit it immediately
- only recurse when the current motif is unknown
- fall back to bytes when no known child frontier exists

## Why It Was A Good Candidate

- smallest possible stable contract
- easiest way to prove deterministic vocab IDs and exact round-trip
- minimal implementation complexity

## Status

`Tried`

## Outcome

- successful as a contract proof
- lossless
- deterministic
- too coarse for model-facing tokenization
- with a full induced vocab, benchmark inputs collapsed to a single root token

## Decision

Keep as a baseline policy only. Do not use as the default model-facing encoding path.
