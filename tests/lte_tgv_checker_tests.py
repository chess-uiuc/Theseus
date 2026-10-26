#!/usr/bin/env python3
"""Prevent a CPU fallback or unequal physical times from passing device parity."""
import unittest

from lte_tgv_device_integration_test import require_device, require_matching_times


class ComparisonContractTests(unittest.TestCase):
    def test_real_cuda_configuration_is_accepted(self):
        require_device("Device configuration: cuda,cpu\n", "cuda")

    def test_cpu_fallback_is_not_cuda_coverage(self):
        with self.assertRaises(AssertionError):
            require_device("Device configuration: cpu\n", "cuda")

    def test_missing_device_report_is_rejected(self):
        with self.assertRaises(AssertionError):
            require_device("Simulation completed", "cuda")

    def test_equal_times_are_accepted(self):
        require_matching_times([0.0, 1e-4, 2e-4], [0.0, 1e-4, 2e-4])

    def test_different_cfl_times_are_rejected(self):
        with self.assertRaises(AssertionError):
            require_matching_times([0.0, 1e-4], [0.0, 1.01e-4])

    def test_missing_frame_is_rejected(self):
        with self.assertRaises(AssertionError):
            require_matching_times([0.0, 1e-4], [0.0])

    def test_nonfinite_time_is_rejected(self):
        with self.assertRaises(AssertionError):
            require_matching_times([0.0, float("nan")], [0.0, float("nan")])


if __name__ == "__main__":
    unittest.main()
