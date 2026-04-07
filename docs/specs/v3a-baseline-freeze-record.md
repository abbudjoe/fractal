# v3A Baseline Freeze Record

Status: frozen

This document is the source of truth for the frozen Path 1 Rust baseline.

Current preparatory tracked run:

- run label: `v3a-path1-control-plane-smoke-seed42`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- purpose: prove the new Path 1 matrix ledger surface is live before real freeze runs begin

## Seed 42 Freeze Run

Status: recorded, not sufficient to freeze alone

- run label: `v3a-baseline-freeze-seed42`
- recorded at unix seconds: `1775402136`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 16 --eval-batches 4 --seed 42 --ledger-path default --run-label v3a-baseline-freeze-seed42 --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954)

Results:

- `A` / attention-only:
  - initial loss `5.8107`
  - final loss `4.1266`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - initial loss `5.6956`
  - final loss `3.7429`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/reference-ssm-hybrid/report.json)
- `A + P1` / primitive-hybrid:
  - initial loss `5.8881`
  - final loss `4.3535`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775400954/primitive-hybrid/report.json)

Interim read:

- `A + M` clearly outperformed both `A` and `A + P1` on this first freeze run
- the old primitive lane remains below both the pure-attention and Rust Mamba-3 lanes
- freeze decision remains pending until the second seeded run is recorded

## Seed 43 Freeze Run

Status: recorded, confirms the same lane ordering

- run label: `v3a-baseline-freeze-seed43`
- recorded at unix seconds: `1775403530`
- ledger: [`/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl`](/private/tmp/fractal-v3a-merge-G4mipf/docs/v3a-results-ledger.jsonl)
- command:

```bash
cargo run --quiet --bin v3a-hybrid-attention-matrix -- --steps 16 --eval-batches 4 --seed 43 --ledger-path default --run-label v3a-baseline-freeze-seed43 --output table
```

- artifact root: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450)

Results:

- `A` / attention-only:
  - initial loss `5.6271`
  - final loss `4.2000`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/attention-only/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/attention-only/report.json)
- `A + M` / reference-ssm-hybrid:
  - initial loss `5.7324`
  - final loss `3.6749`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/reference-ssm-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/reference-ssm-hybrid/report.json)
- `A + P1` / primitive-hybrid:
  - initial loss `5.8162`
  - final loss `4.2354`
  - report: [/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/primitive-hybrid/report.json](/private/tmp/fractal-v3a-merge-G4mipf/artifacts/v3a-hybrid-attention-matrix/1775402450/primitive-hybrid/report.json)

Interim read:

- `A + M` again outperformed both `A` and `A + P1`
- the ordering from seed 42 held on seed 43
- the Rust Mamba-3 hybrid is now stable enough to freeze as the Path 1 baseline comparator

## Freeze Decision

Branch snapshot used for the freeze runs:

- commit hash: `1dad47c46084a6ea31a6f2cad64ee927ea02c6d5`

Frozen Path 1 baseline:

- `A` = attention-only
- `A + M` = attention + Rust Mamba-3 baseline

Historical contender retained for comparison context:

- `A + P1` = attention + original contractive primitive

Decision summary:

- `A + M` beat `A` and `A + P1` on both recorded full-matrix seeds
- `A + P1` did not clear the Rust Mamba-3 baseline and remains the historical underpowered primitive lane
- future `P2` work should compare against this frozen `A` vs `A + M` baseline without redefining it midstream

Required contents when completed:

- freeze commit
- exact matrix commands
- run labels
- ledger references from `docs/v3a-results-ledger.jsonl`
- report/artifact paths
- final statement of the frozen `A` vs `A + M` baseline
