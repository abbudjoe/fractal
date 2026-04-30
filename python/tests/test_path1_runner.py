from __future__ import annotations

import unittest

from python.runners.path1 import _filesystem_safe_name


class Path1RunnerOutputNameTests(unittest.TestCase):
    def test_filesystem_safe_name_keeps_short_names(self) -> None:
        self.assertEqual(_filesystem_safe_name("short-lane"), "short-lane")

    def test_filesystem_safe_name_hashes_long_names_stably(self) -> None:
        name = "attention-only-" + ("very-long-contract-" * 20)
        shortened = _filesystem_safe_name(name, max_length=80)

        self.assertLessEqual(len(shortened), 80)
        self.assertTrue(shortened.startswith("attention-only-"))
        self.assertRegex(shortened, r"-[0-9a-f]{12}$")
        self.assertEqual(shortened, _filesystem_safe_name(name, max_length=80))


if __name__ == "__main__":
    unittest.main()
