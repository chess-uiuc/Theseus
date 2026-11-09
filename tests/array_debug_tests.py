#!/usr/bin/env python3
"""Behavioral tests for production writer and offline diagnostics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

FIXTURE = Path(sys.argv[1]).resolve()
DEVICE = sys.argv[2]
sys.argv = sys.argv[:1]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "array_debug.py"
spec = importlib.util.spec_from_file_location("array_debug", SCRIPT)
debug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(debug)


class ArrayDebugTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, name="left", mode="normal", **options):
        output = self.root / name
        config = {"output": str(output), "calls": [0], "ranks": [0],
                  "arrays": ["*"], "snapshots": True, **options}
        config_file = self.root / f"{name}.json"
        config_file.write_text(json.dumps(config))
        env = {**os.environ, "THESEUS_ARRAY_DEBUG": str(config_file)}
        result = subprocess.run([str(FIXTURE), DEVICE, mode], env=env,
                                text=True, capture_output=True)
        return output, result

    def successful(self, *args, **kwargs):
        output, result = self.capture(*args, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output

    def test_readback_preserves_values_flags_and_cached_device_writer(self):
        output = self.successful()
        records, missing = debug.load_capture(output)
        self.assertFalse(missing)
        self.assertEqual(set(records), {(0, 0, n) for n in ("initial", "updated", "empty")})
        initial = records[(0, 0, "initial")]
        updated = records[(0, 0, "updated")]
        np.testing.assert_array_equal(debug.read_array(output, initial), np.arange(12) - 2)
        np.testing.assert_array_equal(debug.read_array(output, updated), np.arange(12) + 8)
        self.assertEqual(debug.coordinate(initial, 9), {"element": 1, "equation": 1, "point": 1})
        self.assertEqual(initial["device"], "cuda" if DEVICE == "cuda" else "host")

    def test_disabled_does_not_write(self):
        env = dict(os.environ)
        env.pop("THESEUS_ARRAY_DEBUG", None)
        result = subprocess.run([str(FIXTURE), DEVICE], env=env, cwd=self.root,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(list(self.root.iterdir()))

    def test_nonfinite_summary_and_nonfinite_comparison(self):
        output = self.successful(mode="nonfinite")
        records, _ = debug.load_capture(output)
        summary = records[(0, 0, "invalid")]["summary"]
        self.assertEqual(summary["nonfinite"], 3)
        self.assertEqual(summary["nan"], 1)
        self.assertEqual(summary["positive_infinity"], 1)
        self.assertEqual(summary["negative_infinity"], 1)
        self.assertEqual(debug.compare(output, output)["status"], "different")
        self.assertEqual(debug.main(["inspect", str(output)]), 1)

    def test_comparison_reports_earliest_difference_and_tolerance(self):
        left = self.successful("left")
        right = self.successful("right")
        self.assertEqual(debug.compare(left, right)["status"], "equal")
        records, _ = debug.load_capture(right)
        record = records[(0, 0, "updated")]
        array = np.memmap(right / record["snapshot"], dtype=record["dtype"], mode="r+")
        array[9] += 0.125
        array.flush()
        del array
        result = debug.compare(left, right, chunk_values=3)
        first = result["first_divergence"]
        self.assertEqual(first["array"], "updated")
        self.assertEqual(first["mismatches"], 1)
        self.assertEqual(first["first_coordinate"], {"element": 1, "equation": 1, "point": 1})
        self.assertEqual(debug.compare(left, right, atol=0.125)["status"], "equal")
        self.assertEqual(debug.main(["compare", str(left), str(right), "--json", str(self.root / "report.json")]), 1)

    def test_missing_truncated_or_layout_mismatched_data_is_error(self):
        left = self.successful("left")
        right = self.successful("right")
        path = right / "rank000000.jsonl"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        lines[1]["shape"] = [3, 2, 2]
        path.write_text("".join(json.dumps(line) + "\n" for line in lines))
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            debug.compare(left, right)
        binary = right / lines[1]["snapshot"]
        binary.write_bytes(binary.read_bytes()[:-1])
        self.assertEqual(debug.main(["compare", str(left), str(right)]), 2)

    def test_partial_capture_can_be_inspected_but_not_compared(self):
        output = self.successful()
        path = output / "rank000000.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        records, missing = debug.load_capture(output, require_complete=False)
        self.assertTrue(records)
        self.assertEqual(missing, [(0, 0)])
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            debug.compare(output, output)

    def test_filtering_and_summary_only(self):
        output = self.successful(arrays=["init*"], snapshots=False)
        records, _ = debug.load_capture(output)
        self.assertEqual(list(records), [(0, 0, "initial")])
        self.assertFalse(list(output.glob("*.bin")))
        with self.assertRaisesRegex(ValueError, "summary-only"):
            debug.compare(output, output)

    def test_budget_layout_and_overwrite_errors(self):
        _, result = self.capture("budget", max_bytes=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("budget exceeded", result.stderr)
        _, result = self.capture("layout", mode="shape")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shape/size mismatch", result.stderr)
        self.successful("reuse")
        _, result = self.capture("reuse")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_no_selected_arrays_cannot_pass_comparison(self):
        output = self.successful(arrays=["does-not-exist"])
        with self.assertRaisesRegex(ValueError, "no selected arrays"):
            debug.compare(output, output)


if __name__ == "__main__":
    unittest.main()
