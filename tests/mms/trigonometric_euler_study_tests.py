#!/usr/bin/env python3
"""Independent checks of the Euler MMS study's numerical assessment contracts."""
from mms_paths import REPOSITORY_ROOT
import tempfile
import unittest
from pathlib import Path
import json
import shutil
import numpy as np
from unittest.mock import patch
from trigonometric_euler_study import exact_fields, refinement_orders, positive_finite, plot, prepare, matches_final_time, read_case


class StudyTests(unittest.TestCase):
    def test_final_time_roundoff_and_wrong_times(self):
        steps = 43404
        dt = .2/steps
        accumulated = 0.
        for _ in range(steps):
            accumulated += dt
        self.assertTrue(matches_final_time(accumulated, .2, steps, dt))
        self.assertTrue(matches_final_time(.19999999999982918, .2, steps, dt))
        for wrong in (.2 - dt, .2 + dt, .2 - .02*dt, .19, float('nan'), float('inf')):
            with self.subTest(time=wrong):
                self.assertFalse(matches_final_time(wrong, .2, steps, dt))
        # Large step counts cannot expand acceptance to a missing timestep.
        self.assertFalse(matches_final_time(.2 - .2/10**12, .2, 10**12, .2/10**12))
        self.assertFalse(matches_final_time(.2, .2, 0, dt))

    def test_reader_applies_roundoff_check_to_log_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = dict(name='control', steps=43404, dt=.2/43404, dimension=2)
            target = root/case['name']
            viz = target/'solution/ParaView'
            viz.mkdir(parents=True)
            def write(log_time, output_time, steps=43404):
                (target/'run.log').write_text(
                    f'Device configuration: cuda\ntime step: {steps}, time: {log_time}\nExact L2 Error: 1e-10\n')
                (viz/'ParaView.pvd').write_text(
                    '<VTKFile><Collection><DataSet timestep="0" file="initial.pvtu"/>'
                    f'<DataSet timestep="{output_time}" file="final.pvtu"/></Collection></VTKFile>')
            write(.19999999999982918, .19999999999982918)
            with patch('trigonometric_euler_study.cell_local_fields', side_effect=RuntimeError('field stage reached')):
                with self.assertRaisesRegex(RuntimeError, 'field stage reached'):
                    read_case(root, case, 'cuda')
            write(.2 - case['dt'], .2)
            with self.assertRaisesRegex(RuntimeError, 'incorrect final solver time'):
                read_case(root, case, 'cuda')
            write(.2, .2 - case['dt'])
            with self.assertRaisesRegex(RuntimeError, 'incorrect final output time'):
                read_case(root, case, 'cuda')
            write(.2, .2, steps=43403)
            with self.assertRaisesRegex(RuntimeError, 'incorrect final solver time'):
                read_case(root, case, 'cuda')

    def test_orders_reject_incomplete_or_nonphysical_errors(self):
        for errors in ([], [1.], [1., .1], [1., np.nan, .1], [1., 0., .1], [1., -1., .1], [1., 2., 3.]):
            with self.assertRaises(RuntimeError):
                refinement_orders(errors)
        np.testing.assert_allclose(refinement_orders([1., .125, .015625]), [3., 3.])
        with self.assertRaises(RuntimeError):
            positive_finite([np.inf], 'test')

    def test_independent_directional_variation_and_positive_states(self):
        for dimension in (2, 3):
            points = np.random.default_rng(415).uniform(-2., 2., (128, dimension))
            for time in (0., .2, .7):
                state = exact_fields(points, time)
                rho = state[:, 0]
                pressure = .4*(state[:, -1] - .5*np.sum(state[:, 1:-1]**2, axis=1)/rho)
                self.assertTrue(np.all(rho > 1.6))
                self.assertTrue(np.all(pressure > .7))
                for direction in range(dimension):
                    shifted = points.copy()
                    shifted[:, direction] += 1.
                    np.testing.assert_allclose(exact_fields(shifted, time), state, atol=1e-13, rtol=0.)
                    shifted[:, direction] -= .83
                    self.assertGreater(np.max(np.abs(exact_fields(shifted, time) - state)), .01)

    def test_preparation_reaches_requested_time_without_output_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root/'theseus'
            executable.write_text('fixture binary')
            (root/'CMakeCache.txt').write_text('fixture cache')
            source = REPOSITORY_ROOT
            results = root/'cases'
            prepare(source, results, 'cuda', executable, ranks=2)
            cases = json.loads((results/'cases.json').read_text())
            self.assertEqual(len(cases), 12)
            self.assertEqual({c['ranks'] for c in cases}, {2})
            self.assertFalse(any('mpi' in c['name'] for c in cases))
            self.assertEqual(sum(c['backend'] == 'cpu' for c in cases), 2)
            for case in cases:
                runtime = json.loads((results/case['name']/'config.json').read_text())['runTime']
                self.assertGreaterEqual(runtime['initial_save_dt'], runtime['final_time'])
                self.assertGreater(runtime['nsteps_max']*runtime['dt'], runtime['final_time'])
            metadata = json.loads((results/'source.json').read_text())
            self.assertEqual(len(metadata['executable_sha256']), 64)
            archive_source = root/'archive-source'
            inputs = ['TestCases/Euler/2D/PeriodicMMS/config.json',
                      'TestCases/NavierStokes/3D/TaylorGreenVortex/make_periodic_box.py']
            for name in inputs:
                target = archive_source/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/name, target)
            prepare(archive_source, root/'archive-results', 'cpu', executable)
            archived = json.loads((root/'archive-results/source.json').read_text())
            self.assertIsNone(archived['base_sha'])
            self.assertTrue(set(inputs).issubset(archived['source_sha256']))

    def test_plot_saved_measurements_without_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [dict(name=f'{dim}d-level-{level}', dimension=dim,
                            h=2.**(-level), l2=2.**(-3*level),
                            component_physical={'l2': [2.**(-3*level)]*(dim + 2)})
                       for dim in (2, 3) for level in (1, 2, 3)]
            (root/'measurements.json').write_text(json.dumps(records))
            plot(root)
            self.assertGreater((root/'spatial-convergence.png').stat().st_size, 1000)
            self.assertTrue((root/'spatial-convergence.pdf').read_bytes().startswith(b'%PDF'))
            self.assertGreater((root/'component-convergence.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
