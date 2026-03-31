# Chunking / Model Packaging

## What It Is

Take an already chosen frontier and package it into model-sized windows for training or inference.

## How It Works

- tokenize raw text into a frontier
- break the frontier into chunks or streaming windows
- preserve enough metadata to reconstruct, align, and evaluate the token stream

## Why It Is A Good Candidate

- necessary for real model training
- separates tokenizer behavior from sequence packaging behavior
- enables throughput and context-window experiments

## Status

`Tried`

## Trial Outcome

- Frontier used: `NoveltyAware`
- Packaging window: `8` tokens per chunk, `4096` bytes per chunk
- Stress input: `3` frontier tokens -> `1` packaged chunk
- Mixed-domain input: `32` frontier tokens -> `4` packaged chunks
- Reconstruction: exact round-trip for both inputs
- Fallback: `unknown=0`, `byte=0` for both inputs

## Decision

Packaging is a useful model-facing layer when it sits on top of a strong frontier. The tokenizer contract stayed intact, so the layer is worth keeping for model integration work.
