#!/usr/bin/env python3
"""Tests for the Theseus performance timer parser."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from performance_timers import parse_lines  # noqa: E402


class PerformanceTimerParserTests(unittest.TestCase):
    def test_aggregates_timer_scopes_and_samples(self):
        parsed = parse_lines(
            [
                "[TIMER(0)] RHSMult : 10.0 ms\n",
                "[TIMER(all)] RHSMult : 11.0 ms\n",
                "[TIMER(0)] RHSMult : 14.0 ms\n",
                "[TIMER(all)] RHSMult : 15.0 ms\n",
                "[TIMER(0)] EstimateStability : 1.2e-1 ms\n",
            ]
        )

        rank_zero = parsed["timers"]["0"]["RHSMult"]
        self.assertEqual(rank_zero["count"], 2)
        self.assertEqual(rank_zero["samples_ms"], [10.0, 14.0])
        self.assertEqual(rank_zero["median_ms"], 12.0)
        self.assertEqual(rank_zero["total_ms"], 24.0)
        self.assertEqual(parsed["timers"]["all"]["RHSMult"]["median_ms"], 13.0)
        self.assertAlmostEqual(
            parsed["timers"]["0"]["EstimateStability"]["mean_ms"], 0.12
        )
        self.assertIsNone(parsed["timestep"])

    def test_parses_timestep_performance_summary(self):
        parsed = parse_lines(
            """
MPI ranks         : 2
  Timed steps       :           18        18.000            18
  Total time (ms)   :       90.000        95.000       100.000
  Min step (ms)     :        4.000         4.500         5.000
  Mean step (ms)    :        5.000         5.250         5.500
  Max step (ms)     :        6.000         6.500         7.000
  Mean timestep    : 5.500 ms
  Timesteps/sec    : 181.818
  Device sync      : enabled
  MPI barrier      : disabled
""".splitlines()
        )

        summary = parsed["timestep"]
        self.assertEqual(summary["mpi_ranks"], 2)
        self.assertEqual(summary["timed_steps"]["minimum"], 18)
        self.assertEqual(summary["timed_steps"]["mean"], 18.0)
        self.assertEqual(summary["critical_mean_timestep_ms"], 5.5)
        self.assertEqual(summary["timesteps_per_second"], 181.818)
        self.assertTrue(summary["device_sync"])
        self.assertFalse(summary["mpi_barrier"])


if __name__ == "__main__":
    unittest.main()
