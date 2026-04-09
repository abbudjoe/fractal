from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from python.tooling.secret_scan import scan_text, should_ignore_path


class SecretScanTests(unittest.TestCase):
    def test_detects_high_confidence_secret(self) -> None:
        findings = scan_text("sample.txt", "api = '" + "sk-" + "abcdefghijklmnopqrstuvwxyz123456" + "'\n")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "openai_or_anthropic_key")

    def test_allows_ignored_fixture_line(self) -> None:
        findings = scan_text(
            "sample.txt",
            "token = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890' # secret-scan: ignore\n",
        )
        self.assertEqual(findings, [])

    def test_ignores_artifact_paths(self) -> None:
        self.assertTrue(should_ignore_path("artifacts/report.txt"))
        self.assertTrue(should_ignore_path(".runpod-local-logs/foo"))
        self.assertFalse(should_ignore_path("python/models/primitives.py"))
