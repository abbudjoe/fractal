from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re
import subprocess
import sys


IGNORE_PREFIXES = (
    ".git/",
    ".venv/",
    ".cmake-venv/",
    ".runpod-local-logs/",
    ".runpod-local-pids/",
    "artifacts/",
    "target/",
    "__pycache__/",
)

HIGH_CONFIDENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("openai_or_anthropic_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----|BEGIN PRIVATE KEY")),  # secret-scan: ignore
)


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line_number: int
    kind: str
    excerpt: str


def _run_git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True)


def _git_paths(cwd: Path, *args: str) -> list[str]:
    output = subprocess.check_output(["git", *args, "-z"], cwd=cwd)
    return [item.decode("utf-8") for item in output.split(b"\x00") if item]


def should_ignore_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return any(normalized.startswith(prefix) for prefix in IGNORE_PREFIXES)


def iter_tracked_paths(cwd: Path) -> list[str]:
    return [path for path in _git_paths(cwd, "ls-files") if not should_ignore_path(path)]


def iter_staged_paths(cwd: Path) -> list[str]:
    return [
        path
        for path in _git_paths(cwd, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
        if not should_ignore_path(path)
    ]


def _decode_text(blob: bytes) -> str | None:
    if b"\x00" in blob:
        return None
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return blob.decode("latin1")
        except UnicodeDecodeError:
            return None


def load_worktree_text(cwd: Path, path: str) -> str | None:
    file_path = cwd / path
    if not file_path.exists() or file_path.stat().st_size > 2_000_000:
        return None
    return _decode_text(file_path.read_bytes())


def load_staged_text(cwd: Path, path: str) -> str | None:
    try:
        blob = subprocess.check_output(["git", "show", f":{path}"], cwd=cwd)
    except subprocess.CalledProcessError:
        return None
    if len(blob) > 2_000_000:
        return None
    return _decode_text(blob)


def scan_text(path: str, text: str) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "secret-scan: ignore" in line:
            continue
        for kind, pattern in HIGH_CONFIDENCE_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            findings.append(
                SecretFinding(
                    path=path,
                    line_number=line_number,
                    kind=kind,
                    excerpt=match.group(0)[:120],
                )
            )
            break
    return findings


def scan_paths(cwd: Path, paths: list[str], *, staged: bool) -> list[SecretFinding]:
    findings: list[SecretFinding] = []
    loader = load_staged_text if staged else load_worktree_text
    for path in paths:
        text = loader(cwd, path)
        if text is None:
            continue
        findings.extend(scan_text(path, text))
    return findings


def _default_paths(cwd: Path, *, staged: bool) -> list[str]:
    return iter_staged_paths(cwd) if staged else iter_tracked_paths(cwd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="High-confidence repository secret scan.")
    parser.add_argument("--staged", action="store_true", help="Scan staged file contents instead of the tracked worktree.")
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        help="Limit scanning to a specific repo-relative path. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    paths = [path for path in args.paths if not should_ignore_path(path)] or _default_paths(cwd, staged=args.staged)
    findings = scan_paths(cwd, paths, staged=args.staged)

    if not findings:
        print(f"secret scan clean: scanned {len(paths)} file(s)")
        return 0

    print("secret scan findings:")
    for finding in findings:
        print(f"- {finding.kind}: {finding.path}:{finding.line_number} :: {finding.excerpt}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
