# Fractal Docs — Causal Memory Auditor Bundle

This folder contains the updated planning documents with **Causal Memory Auditor** integrated.

## Files

- [`recursive-memory-kernel-v1.md`](./recursive-memory-kernel-v1.md) — full architecture/spec doc
- [`recursive-memory-kernel-v1-rfc.md`](./recursive-memory-kernel-v1-rfc.md) — shorter engineering RFC
- [`recursive-memory-kernel-v1_checklist.md`](./recursive-memory-kernel-v1_checklist.md) — GitHub-ready PR checklist
- [`v2-implementation-plan.md`](./v2-implementation-plan.md) — implementation plan with phase sequencing and causal auditing
- [`fractal-hybrid-v3.md`](./fractal-hybrid-v3.md) — hybrid follow-on proposal that keeps attention as the exact interaction path while reusing recurrence and tree memory as efficiency/context subsystems
- [`hybrid-exact-attention-rescue-prevalidation.md`](./hybrid-exact-attention-rescue-prevalidation.md) — narrow falsification spec for testing whether a tiny exact-attention rescue path can validate the hybrid direction before full v3 implementation
- [`v3a-hybrid-attention-plan.md`](./v3a-hybrid-attention-plan.md) — Path 1 hybrid-attention plan for attention-only, Rust Mamba-3, and later primitive comparison work
- [`v3a-rust-mamba-baseline-checklist.md`](./v3a-rust-mamba-baseline-checklist.md) — baseline-first checklist for freezing the Rust Mamba-3 lane before contender work
- [`v3a-baseline-freeze-and-p2-checklist.md`](./v3a-baseline-freeze-and-p2-checklist.md) — gated next-step checklist for baseline freeze, `P2` definition, and the first `A + P2` run
- [`v3a-baseline-freeze-record.md`](./v3a-baseline-freeze-record.md) — freeze record template for the eventual `A` vs `A + M` baseline decision
- [`v3a-p2-primitive-contract.md`](./v3a-p2-primitive-contract.md) — working contract doc for the first improved Path 1 primitive
- [`v3a-p2-interface-ablation-plan.md`](./v3a-p2-interface-ablation-plan.md) — gated next-step plan for testing whether the main remaining `P2` gap is in the wrapper/interface rather than the primitive core
- [`v3a-p2-primitive-quality-plan.md`](./v3a-p2-primitive-quality-plan.md) — gated primitive-internal tuning ladder for `P2`, beginning with `P2.1`
- [`python-research-stack-foundation.md`](./python-research-stack-foundation.md) — Python architecture substrate for Path 1 and future mini-MoE work
- [`python-path1-phase1-wrap.md`](./python-path1-phase1-wrap.md) — wrap note for the first Python-native Path 1 migration phase and the initial CUDA trio result
- [`mamba-golf-gauntlet.md`](./mamba-golf-gauntlet.md) — public-facing proving-ground contract for `A`, native Mamba, and `P20` under Mamba-native constraints
- [`mamba-golf-scorecard-template.md`](./mamba-golf-scorecard-template.md) — scorecard template for freezing Mamba Golf runs
- [`mamba-golf-seed42-starter-scorecard.md`](./mamba-golf-seed42-starter-scorecard.md) — timestamped seed-42 starter result for the first Mamba Golf `A`/native Mamba/`P20` RunPod gauntlet
- [`p20-gdn-role-candidate.md`](./p20-gdn-role-candidate.md) — first Fractal-native `P20` redesign for the Gated-DeltaNet recurrent-block role, including update/ramp/optimizer contracts and local smoke results
- [`p20-gdn-head2head-scorecard.md`](./p20-gdn-head2head-scorecard.md) — H100 Path 1 head-to-head for `A`, native Mamba3, Fractal-native GDN, frozen `P20` Triton, and `P20-GDN-role`
- [`math-proof-program.md`](./math-proof-program.md) — shared notation and theorem-outline program for standard decoder LLMs, modern hybrid variants, `A + P2`, and the proposed Graph-of-Experts line
- [`math/README.md`](./math/README.md) — split proof notes, theorem audit, and LaTeX source for the repo’s current math-proof program
- [`native-internal-search-sketch.md`](./native-internal-search-sketch.md) — sketch note for a native internal-search runtime direction
- [`v3a-hybrid-attention-sketch.md`](./v3a-hybrid-attention-sketch.md) — sketch note for the Path 1 hybrid-attention stack
