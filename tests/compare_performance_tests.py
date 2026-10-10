#!/usr/bin/env python3
"""Tests for the generic Theseus build comparison driver."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_performance import (  # noqa: E402
    markdown_report,
    parse_build,
    parse_log_input,
    percent_change,
    runner_option_value,
    validate_logs,
)


class ComparePerformanceTests(unittest.TestCase):
    def test_parse_build(self):
        with tempfile.TemporaryDirectory() as tempdir:
            label, path = parse_build(f"candidate={tempdir}")
            self.assertEqual(label, "candidate")
            self.assertEqual(path, Path(tempdir).resolve())

    def test_parse_log_input_and_validation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            first = root / "main.txt"
            second = root / "candidate.txt"
            first.write_text("main\n")
            second.write_text("candidate\n")

            self.assertEqual(parse_log_input(f"main={first}"), ("main", first.resolve()))
            records = validate_logs([("main", first), ("candidate", second)])
            self.assertEqual([record["label"] for record in records], ["main", "candidate"])

    def test_validate_logs_rejects_missing_duplicate_and_single_inputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            present = Path(tempdir) / "run.txt"
            present.write_text("timer output\n")
            with self.assertRaisesRegex(ValueError, "at least two"):
                validate_logs([("main", present)])
            with self.assertRaisesRegex(ValueError, "labels must be unique"):
                validate_logs([("same", present), ("same", present)])
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_logs([("main", present), ("missing", present.parent / "missing.txt")])

    def test_existing_logs_produce_comparison_bundle(self):
        def timer_log(scale: float) -> str:
            lines = []
            for step in range(5):
                lines.append(f"[TIMER(0)] RHSMult : {scale * (step + 1)} ms")
                lines.append(f"[TIMER(0)] Timestep : {scale * (step + 2)} ms")
            return "\n".join(lines) + "\n"

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            main_log = root / "main.log"
            candidate_log = root / "candidate.log"
            output = root / "comparison"
            main_log.write_text(timer_log(1.0))
            candidate_log.write_text(timer_log(0.5))

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_performance.py"),
                    "--log", f"main={main_log}",
                    "--log", f"candidate={candidate_log}",
                    "--reference", "main",
                    "--output", str(output),
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(results["input_mode"], "logs")
            self.assertEqual(results["skip_steps"], 3)
            self.assertEqual([entry["label"] for entry in results["logs"]],
                             ["main", "candidate"])
            self.assertTrue((output / "logs" / "main.log").is_file())
            self.assertTrue((output / "logs" / "candidate.log").is_file())
            summary = (output / "summary.md").read_text()
            self.assertIn("| Log |", summary)
            self.assertIn("-50.00%", summary)

            rejected = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_performance.py"),
                    "--log", f"main={main_log}",
                    "--log", f"candidate={candidate_log}",
                    "--output", str(root / "rejected"),
                    "--", "-n", "10",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("not valid with --log", rejected.stderr)

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
