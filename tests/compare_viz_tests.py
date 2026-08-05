#!/usr/bin/env python3
"""Focused regression tests for scripts/compare_viz.py."""

from __future__ import annotations

import importlib.util
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_viz", ROOT / "scripts" / "compare_viz.py"
)
compare_viz = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_viz)


class FakeMesh:
    def __init__(self, point_data, points=None, cell_types=None):
        self.points = np.array(points) if points is not None else np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        self.point_data = point_data
        self.cell_data = {}
        self.cell_types = None if cell_types is None else np.array(cell_types)
        self.cells = []


class CompareVizTests(unittest.TestCase):
    def compare(self, mesh0, mesh1, fields=None, exclude=None, quiet=True):
        with mock.patch.object(compare_viz, "_load_mesh", side_effect=[mesh0, mesh1]):
            return compare_viz.compare_files(
                "a.vtu",
                "b.vtu",
                fields or [],
                exclude or [],
                1e-10,
                1e-12,
                False,
                quiet,
            )

    def test_missing_requested_field_is_an_error(self):
        meshes = [
            FakeMesh({"Density": np.ones((2, 1))}),
            FakeMesh({"Density": np.ones((2, 1))}),
        ]
        with self.assertRaisesRegex(ValueError, "Pressure"):
            self.compare(*meshes, fields=["Pressure"])

    def test_no_common_fields_is_an_error(self):
        meshes = [
            FakeMesh({"Density": np.ones((2, 1))}),
            FakeMesh({"Pressure": np.ones((2, 1))}),
        ]
        with self.assertRaisesRegex(ValueError, "no common point-data fields"):
            self.compare(*meshes)

    def test_excluding_every_common_field_is_an_error(self):
        meshes = [
            FakeMesh({"Density": np.ones((2, 1))}),
            FakeMesh({"Density": np.ones((2, 1))}),
        ]
        with self.assertRaisesRegex(ValueError, "no common point-data fields"):
            self.compare(*meshes, exclude=["Density"])

    def test_shape_mismatch_reports_failure_without_crashing(self):
        meshes = [
            FakeMesh({"Density": np.ones((2, 1))}),
            FakeMesh({"Density": np.ones((2, 2))}),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            ok, result = self.compare(*meshes, quiet=False)

        self.assertFalse(ok)
        self.assertIn("shape mismatch", output.getvalue())
        self.assertIn("shape_mismatch", result["point_data"]["Density"])

    def test_multicomponent_velocity_field_is_compared_as_one_array(self):
        velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
        meshes = [FakeMesh({"Velocity": velocity}), FakeMesh({"Velocity": velocity.copy()})]

        ok, result = self.compare(*meshes, fields=["Velocity"])

        self.assertTrue(ok)
        self.assertEqual(result["point_data"]["Velocity"]["ncomp"], 2)

    def test_vector_velocity_compares_against_legacy_components(self):
        velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
        current = FakeMesh({"Velocity": velocity})
        legacy = FakeMesh({
            "Horizontal V": velocity[:, 0],
            "Vertical V": velocity[:, 1],
        })

        ok, result = self.compare(current, legacy)

        self.assertTrue(ok)
        self.assertEqual(result["point_data"]["Velocity"]["ncomp"], 2)

    def test_legacy_velocity_component_difference_fails(self):
        velocity = np.array([[1.0, 2.0], [3.0, 4.0]])
        current = FakeMesh({"Velocity": velocity})
        legacy = FakeMesh({
            "Horizontal V": velocity[:, 0],
            "Vertical V": velocity[:, 1] + 1.0,
        })

        ok, result = self.compare(current, legacy)

        self.assertFalse(ok)
        self.assertFalse(result["point_data"]["Velocity"]["ok"])

    def test_different_internal_point_spacing_fails_geometry(self):
        point_data = {"Density": np.ones((3, 1))}
        regular = FakeMesh(point_data, points=[
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]
        ])
        clustered = FakeMesh(point_data, points=[
            [0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [1.0, 0.0, 0.0]
        ])

        ok, result = self.compare(regular, clustered)

        self.assertFalse(ok)
        self.assertFalse(result["geom_ok"])

    def test_different_cell_representation_fails_topology(self):
        point_data = {"Density": np.ones((2, 1))}
        high_order = FakeMesh(point_data, cell_types=[70])
        linear_subcells = FakeMesh(point_data, cell_types=[9, 9, 9, 9])

        ok, result = self.compare(high_order, linear_subcells)

        self.assertFalse(ok)
        self.assertFalse(result["topology_ok"])


if __name__ == "__main__":
    unittest.main()
