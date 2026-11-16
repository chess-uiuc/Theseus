"""Contracts for extended CNS matrix plans, controls and observational reporting."""
from mms_paths import REPOSITORY_ROOT
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from cns_matrix_study import prepare, assess, select_degrees, timestep, plot
from unittest.mock import patch
import numpy as np

SOURCE = REPOSITORY_ROOT


class CNSMatrixTests(unittest.TestCase):
    def generate(self, root, transport='constant', **options):
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(SOURCE, root, transport, **options)
        return json.loads((root/'cases.json').read_text())

    def test_default_matrix_count_transport_and_exact_time(self):
        for transport in ('euler', 'constant', 'sutherland'):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cases = self.generate(root, transport)
                self.assertEqual(len(cases), 210)
                self.assertEqual({c['order'] for c in cases}, {1, 2, 3, 4, 5})
                self.assertEqual({c['cells'] for c in cases}, {12, 18, 27, 40, 60, 90})
                self.assertEqual(sum(c['time_control'] for c in cases), 30)
                for case in cases:
                    rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                    self.assertAlmostEqual(rt['dt']*case['steps'], rt['final_time'], places=14)
                    self.assertEqual(rt['mu0'], .02)
                    self.assertEqual(rt['R_gas'], 1./600. if transport == 'sutherland' else 1.)
                    self.assertEqual(rt['numerical_flux'], case['flux'])
                    self.assertEqual(rt['order'], case['order'])
                plan = json.loads((root/'matrix-plan.json').read_text())
                self.assertEqual(plan['cases'], 210)
                self.assertEqual(plan['max_unknowns'], 4*90**2*6**2)
                self.assertGreater(plan['max_steps'], 40000)

    def test_nonbinary_rates_in_assessment_and_plot(self):
        import matplotlib.axes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.generate(root, transport='euler', degrees=(2,), fluxes=('Roe',), cells_sequence=(12, 18, 27))
            records = [dict(**c, component_physical={'l2': [c['h']**3]*4}) for c in cases]
            report = assess(records)
            np.testing.assert_allclose(report['studies'][0]['observed_orders'], 3.)
            (root/'measurements.json').write_text(json.dumps(records))
            original = matplotlib.axes.Axes.semilogx
            rates = []
            def capture(ax, x, y, *args, **kwargs):
                rates.append(y)
                return original(ax, x, y, *args, **kwargs)
            with patch.object(matplotlib.axes.Axes, 'semilogx', capture):
                plot(root)
            self.assertEqual(len(rates), 3)
            np.testing.assert_allclose(rates, 3.)
            self.assertTrue((root/'density-all-orders.png').is_file())

    def test_timestep_anchor_diffusive_and_degree_scaling(self):
        dt, steps = timestep(3, 24, .001, .2)
        self.assertEqual((dt, steps), (.001, 200))
        dt_fine, fine_steps = timestep(3, 48, .001, .2)
        self.assertEqual((dt_fine, fine_steps), (.00025, 800))
        dt_high, _ = timestep(5, 48, .001, .2)
        self.assertLess(dt_high, dt_fine)
        self.assertEqual(timestep(1, 6, .001, .2)[0], .001)

    def test_custom_mesh_start_depth_and_finest_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.generate(root, degrees=(2, 4), fluxes=('Roe',), base_cells=20,
                                  start_level=0, mesh_levels=4, cells_sequence=None, ranks=3, final_time=.05)
            self.assertEqual(len(cases), 10)
            self.assertEqual({c['cells'] for c in cases}, {20, 40, 80, 160})
            self.assertEqual({c['ranks'] for c in cases}, {3})
            for degree in (2, 4):
                group = [c for c in cases if c['order'] == degree]
                main = [c for c in group if not c['time_control']]
                control = next(c for c in group if c['time_control'])
                self.assertEqual(control['cells'], 160)
                self.assertEqual(control['dt'], main[-1]['dt']/2.)
                self.assertEqual(control['steps'], main[-1]['steps']*2)

    def test_invalid_selection_and_configuration_rejected(self):
        for selection in ('', '1,1', '0', '6', 'two', '2,'):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                select_degrees(selection)
        for options in ({'mesh_levels': 2}, {'start_level': -1}, {'base_cells': 2},
                        {'reference_dt': 0.}, {'final_time': float('nan')}, {'degrees': (0,)},
                        {'fluxes': ()}, {'ranks': 0}, {'cells_sequence': (12, 12, 24)},
                        {'cells_sequence': (12, 18)}, {'cells_sequence': (24, 18, 12)}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.generate(Path(directory), **options)

    def test_observations_do_not_masquerade_as_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = self.generate(root, degrees=(2, 3), fluxes=('Roe',), mesh_levels=4)
            def records(power):
                return [dict(**c, component_physical={'l2': [c['h']**power(c)]*4}) for c in cases]
            report = assess(records(lambda c: c['order'] + 1))
            self.assertTrue(report['controls_passed'])
            self.assertTrue(report['order_expectations_met'])
            self.assertFalse(report['accuracy_qualified'])
            shortfall = assess(records(lambda c: 1))
            self.assertTrue(shortfall['passed'])
            self.assertFalse(shortfall['order_expectations_met'])
            self.assertTrue(shortfall['needs_order_review'])
            self.assertEqual(shortfall['status'], 'completed_with_order_flags')
            data = records(lambda c: c['order'] + 1)
            data[-1]['component_physical']['l2'] = [v*2. for v in data[-1]['component_physical']['l2']]
            self.assertFalse(assess(data)['controls_passed'])
            with self.assertRaisesRegex(RuntimeError, 'missing declared'):
                assess(data[:-1])
            with self.assertRaisesRegex(RuntimeError, 'missing declared'):
                assess(data[1:])


if __name__ == '__main__':
    unittest.main()
