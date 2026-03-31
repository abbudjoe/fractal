# RunPod Audit

Use `scripts/runpod-audit.sh` to inspect preserved RunPod outputs under
`.runpod-local-logs/runpod-results`.

The wrapper stores each run in a directory with the following layout:

- `metadata/wrapper-manifest.json`
- `remote/logs/latest.log`
- `remote/manifests/run-manifest.json`
- `remote/manifests/tournament-run-manifest.json`
- `remote/artifacts/tournament-run-artifact.json`

## Commands

Artifact preservation health:

```bash
scripts/runpod-audit.sh artifacts
```

Pod hygiene plus preservation health:

```bash
scripts/runpod-audit.sh pods
```

Optional stale-window tuning:

```bash
scripts/runpod-audit.sh pods --stale-hours 4
```

## What It Flags

- missing wrapper manifest
- missing final log
- missing wrapper-side run manifest
- missing preserved tournament artifact JSON after the run is terminal
- wrapper records that still say `running` but no longer have a live pod
- live pods that have no preserved run directory

While a run is still live, the audit reports `in-progress` instead of treating
missing final artifact files as a failure. Once the wrapper reaches a terminal
state, the full artifact bundle is expected to be present.

The artifact audit is the canonical local preservation check. The pod audit is
an operational convenience for spotting stale or orphaned pod state.
