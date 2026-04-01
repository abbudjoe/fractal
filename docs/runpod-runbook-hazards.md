# RunPod Runbook Hazards

This document captures the control-plane hazards that make a RunPod attempt look like a pod failure even when the underlying GPU node is healthy.

It is a runbook checklist, not a generic infra note.

Use this alongside:
- [cloud-workflow-policy.md](/Users/joseph/fractal/docs/cloud-workflow-policy.md)
- [harness-hardening-checklist.md](/Users/joseph/fractal/docs/harness-hardening-checklist.md)
- [experiment-interface.md](/Users/joseph/fractal/docs/experiment-interface.md)
- [runpod-audit.md](/Users/joseph/fractal/docs/runpod-audit.md)

## What This Protects Against

These hazards showed up during the canonical winner-bakeoff rerun:
- manifest-mode runs failed because the wrapper injected an extra backend flag
- manifest-mode runs failed because the remote sync was not a git checkout, so branch identity was missing
- stale remote state made the dashboard look like the newest attempts were failing in the same way as older ones
- wrapper retries paid full bootstrap and compile cost before failing on a control-plane mismatch

The lesson is simple:
- do not read "pod failed" as "GPU host failed"
- first verify whether the runbook violated the experiment contract

## Dashboard Rule

The RunPod dashboard is useful but not authoritative by itself.

Treat the dashboard as:
- accurate for pod liveness
- incomplete for attempt identity
- ambiguous when the same pod is reused for multiple logical runs

When a reused pod shows a failure:
- check local preserved attempt rows first
- confirm the latest wrapper manifest for the current logical experiment
- verify whether the dashboard is showing an old failed attempt or the current one

## Hazard Checklist

### 1. Manifest Contract Drift

Hazard:
- the wrapper adds CLI flags that are illegal in manifest mode

Symptoms:
- full compile succeeds
- binary exits immediately with `InvalidConfig(...)`
- no tournament artifact is produced

Guardrail:
- manifest runs must own all run-shaping arguments
- wrapper-injected flags must be contract-safe, or explicitly allowed by the manifest parser

Check:
- compare wrapper-added args with manifest-mode parser rules before launch
- add regression tests for every wrapper-injected flag used with `--experiment-manifest`

### 2. Remote Identity Mismatch

Hazard:
- the remote tree is a synced tarball, not a git checkout, but validation expects git metadata

Symptoms:
- compile/bootstrap succeeds
- run exits before training
- log mentions branch or commit identity could not be detected

Guardrail:
- remote runs must receive explicit branch/commit identity from the wrapper
- manifest validation must not depend on remote git state when the remote is intentionally non-git

Check:
- verify exported `FRACTAL_BRANCH` and `FRACTAL_COMMIT_SHA`
- verify the wrapper manifest and remote run manifest agree on branch and commit

### 3. Stale Remote State Reuse

Hazard:
- a reused pod carries prior state, logs, cached binaries, or manifests that no longer match the current attempt contract

Symptoms:
- dashboard looks like the same failure is repeating
- old manifests remain visible after a relaunch
- preserved local rows do not match live remote state

Guardrail:
- when fixing runbook/control-plane bugs, relaunch onto a clean remote dir and clean state dir
- do not mix old state with a new logical rerun after a contract bug

Check:
- confirm `remote_dir` and `state_dir` are either cleaned or versioned for the new relaunch
- confirm the new wrapper manifest points at the new paths

### 4. Full Rebootstrap Tax

Hazard:
- every wrapper attempt re-runs apt/bootstrap and `cargo build --release` before reaching the actual experiment

Symptoms:
- failures look expensive and dramatic
- pods appear "unstable" even though the failure is deterministic and pre-run
- repeated retries burn time, disk, and credits before the same control-plane exit

Guardrail:
- treat repeated compile-before-fail patterns as a runbook bug first
- do not keep retrying until the contract mismatch is understood

Check:
- if the binary fails before `[start 1/1]`, inspect runbook semantics before retrying
- separate compile failure, contract failure, and model/runtime failure in post-run classification

### 5. Attempt Identity Ambiguity

Hazard:
- one pod hosts multiple logical experiments or retries, but the operator view does not clearly separate attempt identity from pod identity

Symptoms:
- users think the current run failed because the pod previously failed
- audit rows and dashboard rows appear contradictory

Guardrail:
- logical experiment id, attempt id, and pod id must all be preserved distinctly
- dashboards and audits should be read in that order:
  1. logical experiment
  2. latest attempt
  3. pod state

Check:
- verify `logical_name`, `attempt_id`, and `pod.id` in the wrapper manifest
- do not reason from pod id alone

### 6. Missing Preservation Before Exit

Hazard:
- a run fails or completes but local preservation has not yet copied the remote log or artifact bundle

Symptoms:
- dashboard shows failure or stop
- local archive has wrapper manifest only
- operator cannot tell whether the experiment ever started

Guardrail:
- preservation health must be checked alongside pod status
- no run should be considered accounted for until the wrapper manifest, run manifest, log, and artifact state are all visible locally

Check:
- run [runpod-audit.md](/Users/joseph/fractal/docs/runpod-audit.md) after failures and before cleanup
- classify the attempt as `in-progress`, `degraded`, or `complete` from preserved evidence, not memory

## Operator Sequence

When a pod "blows up", do this in order:

1. Check the latest local wrapper manifest for the logical experiment.
2. Confirm whether the failing row is the current attempt or an older preserved attempt.
3. Inspect the local preserved log tail.
4. Determine whether the failure happened:
   - before compile
   - during compile
   - after compile but before `[start 1/1]`
   - during train/eval
5. Only restart after the failure class is known.

## Hard Rules

- Do not treat pod reuse as attempt continuity.
- Do not trust dashboard-only failure reads for canonical reruns.
- Do not retry a manifest-mode run until wrapper-added arguments are checked.
- Do not depend on remote git introspection when the remote tree is a tar-synced snapshot.
- Do not reuse stale state dirs after a control-plane bug fix.

## Current Interpretation Rule

For canonical reruns, the source of truth is:

1. local preserved wrapper manifest
2. local preserved run manifest
3. local preserved log/artifact bundle
4. only then the RunPod dashboard

That ordering prevents stale pod history from masquerading as a live failure.
