"""Analytic integral and polynomial reproduction checks for physical MMS norms."""
from mms_paths import REPOSITORY_ROOT
import itertools
import unittest
import numpy as np
from mms_error_norms import p2_volume_errors, tensor_volume_errors


def fixture(dimension, elements):
    keys = []
    values = []
    for element in itertools.product(range(elements), repeat=dimension):
        center = (np.array(element) + .5)/elements
        for node in itertools.product((0., .5, 1.), repeat=dimension):
            point = (np.array(element) + node)/elements
            keys.append(np.r_[np.pad(center, (0, 3 - dimension)),
                              np.pad(point, (0, 3 - dimension))])
            x, y = point[:2]
            values.append([1., x, y, x*y] if dimension == 2 else [1., x, y, x*y, point[2]**2])
    return np.array(keys), np.array(values)


def polynomial(points, time):
    x, y = points[:, :2].T
    fields = [np.ones(len(points)), x, y, x*y]
    if points.shape[1] == 3:
        fields.append(points[:, 2]**2)
    return np.column_stack(fields)


class NormTests(unittest.TestCase):
    def test_high_order_polynomial_integrals(self):
        for degree in range(1, 22):
            reference = np.r_[-1., np.real_if_close(np.polynomial.legendre.Legendre.basis(degree).deriv().roots()).astype(float), 1.]
            points = (reference + 1.)/2.
            x, y = np.meshgrid(points, points, indexing='ij')
            coordinates = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
            centers = np.tile([.5, .5, 0.], (x.size, 1))
            keys = np.column_stack((centers, coordinates))
            values = np.column_stack((x.ravel()**degree, y.ravel()**degree,
                                      (x*y).ravel()**degree, np.ones(x.size)))
            zero = lambda positions, time: np.zeros((len(positions), 4))
            result = tensor_volume_errors(keys, values, 2, 1, 0., 1., 0., zero, degree, degree + 3)
            expected = [1./np.sqrt(2*degree + 1.), 1./np.sqrt(2*degree + 1.),
                        1./(2*degree + 1.), 1.]
            np.testing.assert_allclose(result['l2'], expected, atol=2e-13, rtol=0.)

    def test_known_physical_integrals_and_refinement(self):
        for dimension in (2, 3):
            for elements in (1, 2):
                keys, values = fixture(dimension, elements)
                zero = lambda points, time: np.zeros((len(points), dimension + 2))
                result = p2_volume_errors(keys, values, dimension, elements, 0., 1./elements, 0., zero)
                l1 = [1., .5, .5, .25]
                l2 = [1., np.sqrt(1./3.), np.sqrt(1./3.), 1./3.]
                if dimension == 3:
                    l1.append(1./3.)
                    l2.append(np.sqrt(1./5.))
                np.testing.assert_allclose(result['l1'], l1, atol=2e-14, rtol=0.)
                np.testing.assert_allclose(result['l2'], l2, atol=2e-14, rtol=0.)
                exact = p2_volume_errors(keys, values, dimension, elements, 0., 1./elements, 0., polynomial)
                np.testing.assert_allclose(exact['l2'], 0., atol=2e-14)

    def test_physical_measure_and_separate_traces(self):
        keys, values = fixture(2, 2)
        # Double every length: constant error L1 scales with area, L2 with sqrt(area).
        keys *= 2.
        values[:] = 1.
        zero = lambda points, time: np.zeros((len(points), 4))
        result = p2_volume_errors(keys, values, 2, 2, 0., 1., 0., zero)
        np.testing.assert_allclose(result['l1'], 4., atol=2e-14, rtol=0.)
        np.testing.assert_allclose(result['l2'], 2., atol=2e-14, rtol=0.)
        values[:9] = 2.
        result = p2_volume_errors(keys, values, 2, 2, 0., 1., 0., zero)
        np.testing.assert_allclose(result['l1'], 5., atol=2e-14, rtol=0.)
        np.testing.assert_allclose(result['l2'], np.sqrt(7.), atol=2e-14, rtol=0.)

    def test_vector_momentum_is_rotation_invariant(self):
        keys, values = fixture(2, 2)
        zero = lambda points, time: np.zeros((len(points), 4))
        original = p2_volume_errors(keys, values, 2, 2, 0., .5, 0., zero)
        angle = .71
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        values[:, 1:3] = values[:, 1:3] @ rotation.T
        rotated = p2_volume_errors(keys, values, 2, 2, 0., .5, 0., zero)
        for norm in ('l1', 'l2', 'sampled_max'):
            self.assertAlmostEqual(original['momentum'][norm], rotated['momentum'][norm], places=14)
        self.assertAlmostEqual(original['momentum']['l2'], np.sqrt(2./3.), places=14)
        # Integrating the vector norm differs from taking the norm of component integrals.
        self.assertGreater(original['momentum']['l1'], np.linalg.norm(original['l1'][1:3]))

    def test_missing_misaligned_and_inconsistent_nodes_rejected(self):
        keys, values = fixture(2, 1)
        zero = lambda points, time: np.zeros((len(points), 4))
        with self.assertRaises(ValueError):
            p2_volume_errors(keys[:-1], values[:-1], 2, 1, 0., 1., 0., zero)
        invalid = keys.copy()
        invalid[0, 3] += .1
        with self.assertRaises(ValueError):
            p2_volume_errors(invalid, values, 2, 1, 0., 1., 0., zero)
        inconsistent_keys = np.vstack((keys, keys[0]))
        inconsistent_values = np.vstack((values, values[0] + 1.))
        with self.assertRaises(ValueError):
            p2_volume_errors(inconsistent_keys, inconsistent_values, 2, 1, 0., 1., 0., zero)


if __name__ == '__main__':
    unittest.main()
