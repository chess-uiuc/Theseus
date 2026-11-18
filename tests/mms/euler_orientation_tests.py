"""Contracts for independently mapping matched solution orientations."""
from mms_paths import REPOSITORY_ROOT
import unittest
import numpy as np
from euler_orientation_study import canonical_fields, compare_fields, oriented_exact
from trigonometric_euler_study import exact_fields


class OrientationTests(unittest.TestCase):
    def test_vector_and_coordinate_mapping(self):
        rng = np.random.default_rng(72)
        keys = rng.uniform(0., 1., (24, 6))
        density = rng.uniform(1., 2., (24, 1))
        velocity = rng.uniform(-1., 1., (24, 3))
        base = dict(coordinates=keys, Density=density, Velocity=velocity,
                    Pressure=density + 1., **{'Specific Total Energy': density + 2.})
        reference = canonical_fields(base, 0)
        for orientation in (1, 2):
            mapping = (np.arange(3) + orientation) % 3
            physical = {name: values.copy() for name, values in base.items()}
            physical['coordinates'][:, np.r_[mapping, 3 + mapping]] = keys
            physical['Velocity'][:, mapping] = velocity
            order = rng.permutation(24)
            physical = {name: values[order] for name, values in physical.items()}
            candidate = canonical_fields(physical, orientation)
            compare_fields(reference, candidate)
            candidate['Velocity'][0, 0] += .001
            with self.assertRaises(RuntimeError):
                compare_fields(reference, candidate)

    def test_exact_state_transforms_scalar_and_vector_equations(self):
        coordinates = np.array([[.21, .43, .67], [.1, .3, .9]])
        original = exact_fields(coordinates[:, [1, 2, 0]], .2)
        rotated = oriented_exact(coordinates, .2, 1)
        np.testing.assert_allclose(rotated[:, [0, 4]], original[:, [0, 4]])
        np.testing.assert_allclose(rotated[:, [2, 3, 1]], original[:, [1, 2, 3]])


if __name__ == '__main__':
    unittest.main()
