#!/usr/bin/env python3
"""Tests for the generic Theseus build comparison driver."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_performance import (  # noqa: E402
    markdown_report,
    parse_build,
    percent_change,
    runner_option_value,
)


class ComparePerformanceTests(unittest.TestCase):
    def test_parse_build(self):
        with tempfile.TemporaryDirectory() as tempdir:
            label, path = parse_build(f"candidate={tempdir}")
            self.assertEqual(label, "candidate")
            self.assertEqual(path, Path(tempdir).resolve())

    def test_percent_change(self):
        self.assertEqual(percent_change(10.0, 9.0), -10.0)
        self.assertIsNone(percent_change(0.0, 1.0))

    def test_runner_option_value(self):
        arguments = ["-c", "case.json", "-p", "4", "-rcuda"]
        self.assertEqual(runner_option_value(arguments, "-p", "2"), "4")
        self.assertEqual(runner_option_value(arguments, "-r", "cpu"), "cuda")
        self.assertEqual(runner_option_value(arguments, "-n", "100"), "100")

    def test_report_compares_timestep_and_timer(self):
        def result(label: str, step: float, timer: float):
            return {
                "label": label,
                "parsed": {
                    "timestep": {"critical_mean_timestep_ms": step},
                    "timers": {
                        "0": {
                            "RHSMult": {"median_ms": timer, "count": 6},
                        }
                    },
                },
            }

        report = markdown_report(
            [result("main", 10.0, 2.0), result("candidate", 9.0, 1.5)], "main"
        )
        self.assertIn("-10.00%", report)
        self.assertIn("-25.00%", report)
        self.assertIn("RHSMult", report)


if __name__ == "__main__":
    unittest.main()
