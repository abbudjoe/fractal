# Secret Scanning

This repo now enforces one shared high-confidence secret scanner in two places:

* GitHub Actions:
  * `/Users/joseph/fractal/.github/workflows/secret-scan.yml`
* local pre-commit hooks:
  * `/Users/joseph/fractal/.githooks/pre-commit`

Both entrypoints call the same repo-owned scanner:

* `/Users/joseph/fractal/scripts/check_secrets.py`
* `/Users/joseph/fractal/python/tooling/secret_scan.py`

## Scope

The scanner is intentionally high-confidence. It checks for:

* AWS access keys
* Google API keys
* OpenAI/Anthropic-style `sk-...` keys
* GitHub token families
* Slack token families
* Hugging Face tokens
* private key blocks

It does not try to flag every generic `token=` or `secret=` identifier, because
that creates too much noise in this codebase.

## Local Install

To enable the repo-local pre-commit hook in a checkout:

```bash
./scripts/install_git_hooks.sh
```

That sets:

* `core.hooksPath = .githooks`

for the current repository clone.

## Allowlist Escape Hatch

If a line must intentionally contain a high-confidence test fixture, add:

* `secret-scan: ignore`

to that same line.
