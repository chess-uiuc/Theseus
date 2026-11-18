"""CNS manufactured-study configuration and numerical assessment contracts."""
from mms_paths import REPOSITORY_ROOT
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cns_mms_study import prepare, check
from cns_forcing_audit import audit
from cns_flux_study import prepare as prepare_fluxes, assess as assess_fluxes
from cns_sutherland_study import prepare as prepare_sutherland, assess as assess_sutherland
from cns_temporal_study import prepare as prepare_temporal, check as check_temporal

SOURCE = REPOSITORY_ROOT


class CNSStudyTests(unittest.TestCase):
    def test_spatial_controls_and_matching_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(SOURCE, root, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 7)
            self.assertEqual({c['ranks'] for c in cases}, {2})
            self.assertFalse(any('mpi' in c['name'] for c in cases))
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['mu'], case['mu'])
                self.assertEqual(rt['flow_model'], case['flow_model'])
                self.assertEqual(rt['Pr'], .72)
                self.assertEqual(rt['numerical_flux'], 'Roe')
                self.assertLessEqual(round(rt['final_time']/rt['dt']), 400)
                if case['name'] == 'euler-forcing':
                    self.assertEqual(rt['mu'], .02)
                    self.assertEqual(rt['sources'][0]['viscous_scale'], 0.)
                if case['name'] == 'zero-transport':
                    self.assertEqual(rt['flow_model'], 'navier-stokes')
                    self.assertEqual(rt['mu'], 0.)
            cases_by_name = {c['name']: c for c in cases}
            self.assertEqual(cases_by_name['half-dt']['dt'], cases_by_name['level-3']['dt']/2.)

    def test_spatial_checker_rejects_low_order_despite_clean_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(SOURCE, root)
            def data(results, case, device, **options):
                error = case['h']
                if case['name'] == 'euler-forcing':
                    error *= 10.
                values = {name: np.ones(3) for name in ('Density', 'Velocity', 'Pressure',
                                                       'Specific Total Energy')}
                return dict(**case, component_physical={'l2': [error]*4}), values
            with patch('cns_mms_study.read_case', side_effect=data):
                with self.assertRaisesRegex(RuntimeError, 'spatial order'):
                    check(root, 'cpu')

    def test_quick_study_is_bounded_and_reports_only_its_actual_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(SOURCE, root, ranks=2, quick=True)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 3)
            for case in cases:
                self.assertTrue(case['quick'])
                self.assertEqual(case['dimension'], 2)
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(round(rt['final_time']/rt['dt']), 200)
                self.assertEqual(rt['sources'], [{'type': 'trigonometric-cns-mms'}])
            def data(results, case, device, **options):
                error = case['h']**3
                return dict(**case, component_physical={'l2': [error]*4}), {}
            with patch('cns_mms_study.read_case', side_effect=data):
                check(root, 'cpu')
            report = json.loads((root/'cns-assessment.json').read_text())
            self.assertTrue(report['passed'])
            self.assertFalse(report['accuracy_qualified'])
            self.assertEqual(report['level'], 'quick')
            self.assertIsNone(report['finest_timestep_relative_change'])
            self.assertIsNone(report['zero_transport_euler_limit_passed'])
            cases[0]['quick'] = False
            (root/'cases.json').write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, 'common testing level'):
                check(root, 'cpu')

    def test_3d_configuration_and_control_failure_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare(SOURCE, root, ranks=4, dimension=3)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 5)
            self.assertEqual([c['h'] for c in cases[:3]], [1./6., 1./12., 1./24.])
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['dim'], 3)
                self.assertEqual(rt['num_equations'], 5)
                self.assertEqual(case['ranks'], 4)
                self.assertEqual(rt['exact_solution']['function'], 'TrigonometricCNSMMS')
                self.assertTrue(Path(rt['mesh_file']).is_file())

            def data(results, case, device, **options):
                self.assertEqual(options['quadrature_points'], 12)
                error = case['h']**3
                if case['name'] == 'euler-forcing':
                    error *= 10.
                return dict(**case, component_physical={'l2': [error]*5}), {}

            with patch('cns_mms_study.read_case', side_effect=data):
                check(root, 'cpu')
            report = json.loads((root/'cns-assessment.json').read_text())
            self.assertEqual(report['dimension'], 3)
            self.assertIsNone(report['zero_transport_euler_limit_passed'])
            self.assertTrue(report['accuracy_qualified'])

            for control, message in (('half-dt', 'temporally contaminated'),
                                     ('euler-forcing', 'omitted viscous forcing')):
                def contaminated(results, case, device, **options):
                    record, values = data(results, case, device, **options)
                    if case['name'] == control:
                        record['component_physical']['l2'] = [2.*case['h']**3]*5
                    return record, values
                with self.subTest(control=control), patch('cns_mms_study.read_case', side_effect=contaminated):
                    with self.assertRaisesRegex(RuntimeError, message):
                        check(root, 'cpu')

    def test_temporal_configuration_has_stability_and_spatial_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_temporal(SOURCE, root, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 8)
            for method in ('RK3SSP', 'RK4'):
                group = [c for c in cases if c['method'] == method]
                main = [c for c in group if not c['spatial_control']]
                control = next(c for c in group if c['spatial_control'])
                self.assertGreater(control['order'], main[-1]['order'])
                self.assertEqual(control['dt'], main[-1]['dt'])
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['sources'][0]['time_scale'], case['time_scale'])
                self.assertEqual(rt['conditions']['initial_conditions']['params']['x1'], case['time_scale'])
                self.assertEqual(rt['mu'], .02)
                self.assertEqual(rt['flow_model'], 'navier-stokes')
                self.assertLessEqual(round(rt['final_time']/rt['dt']), 640)

    def test_viscous_audit_is_active_and_temperature_rescaling_is_consistent(self):
        reference = audit()
        scaled = audit(gas_constant=1./600.)
        self.assertGreater(min(reference['stress_divergence']), 1e-3)
        self.assertGreater(reference['viscous_work_divergence'], 1e-3)
        self.assertGreater(reference['conduction_divergence'], 1e-3)
        np.testing.assert_allclose(reference['conduction_divergence'], scaled['conduction_divergence'],
                                   atol=1e-10, rtol=1e-8)
        zero = audit(mu=0.)
        self.assertEqual(zero['viscous_work_divergence'], 0.)
        self.assertEqual(zero['conduction_divergence'], 0.)
        np.testing.assert_array_equal(zero['stress_divergence'], [0., 0.])

    def test_sutherland_audit_detects_variable_coefficients_and_product_terms(self):
        report = audit(gas_constant=1./600., sutherland=True)
        self.assertEqual(report['transport'], 'sutherland')
        self.assertGreater(report['viscosity_range'][1]/report['viscosity_range'][0], 1.1)
        self.assertGreater(report['temperature_range'][0], 200.)
        self.assertLess(report['temperature_range'][1], 400.)
        self.assertGreater(min(report['coefficient_gradient_stress']), 1e-3)
        self.assertGreater(report['coefficient_gradient_work'], 1e-3)
        self.assertGreater(report['coefficient_gradient_conduction'], 1e-3)
        half = audit(mu=.01, gas_constant=1./600., sutherland=True)
        for key in ('stress_divergence', 'viscous_work_divergence', 'conduction_divergence',
                    'coefficient_gradient_stress', 'coefficient_gradient_work',
                    'coefficient_gradient_conduction'):
            np.testing.assert_allclose(np.asarray(half[key])*2., report[key], atol=1e-11, rtol=1e-7)
        zero = audit(mu=0., gas_constant=1./600., sutherland=True)
        self.assertEqual(zero['coefficient_gradient_conduction'], 0.)
        np.testing.assert_array_equal(zero['coefficient_gradient_stress'], [0., 0.])

    def test_sutherland_study_keeps_solver_transport_active_in_omission_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_sutherland(SOURCE, root, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 6)
            self.assertEqual({c['order'] for c in cases}, {3})
            self.assertEqual({c['ranks'] for c in cases}, {2})
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['mu0'], .02)
                self.assertEqual(rt['R_gas'], 1./600.)
                self.assertEqual(rt['Pr'], .72)
                self.assertEqual(rt['flow_model'], 'navier-stokes')
                self.assertEqual(rt['sources'][0]['type'], 'trigonometric-cns-sutherland-mms')
                self.assertEqual(rt['sources'][0]['viscous_scale'], case['viscous_scale'])
                self.assertEqual(rt['sources'][0]['coefficient_gradient_scale'],
                                 case['coefficient_gradient_scale'])
                self.assertLessEqual(round(rt['final_time']/rt['dt']), 400)
            data = []
            for case in cases:
                error = case['h']**4
                if case['name'] == 'inviscid-forcing':
                    error *= 10.
                if case['name'] == 'frozen-coefficients':
                    error *= 3.
                data.append(dict(**case, component_physical={'l2': [error]*4}))
            self.assertTrue(assess_sutherland(data)['passed'])
            fine = data[2]['component_physical']['l2']
            for index, gate in ((3, 'timestep_isolation_passed'), (4, 'viscous_control_passed'),
                                (5, 'coefficient_gradient_control_passed')):
                changed = json.loads(json.dumps(data))
                changed[index]['component_physical']['l2'] = [value*1.1 for value in fine]
                report = assess_sutherland(changed)
                self.assertFalse(report[gate])
                self.assertFalse(report['passed'])
            low_order = json.loads(json.dumps(data))
            for row in low_order[:3]:
                row['component_physical']['l2'] = [row['h']**2]*4
            self.assertFalse(assess_sutherland(low_order)['order_passed'])
            with self.assertRaisesRegex(RuntimeError, 'all six'):
                assess_sutherland(data[:-1])

    def test_selected_flux_cns_studies_require_order_and_time_controls(self):
        for transport, degree in (('constant', 2), ('sutherland', 3)):
            with self.subTest(transport=transport), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepare_fluxes(SOURCE, root, transport, fluxes=('lfr', 'Ranocha'), ranks=2)
                cases = json.loads((root/'cases.json').read_text())
                self.assertEqual(len(cases), 8)
                data = []
                for case in cases:
                    rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                    self.assertEqual(rt['numerical_flux'], case['flux'])
                    self.assertEqual(rt['order'], degree)
                    self.assertEqual(case['ranks'], 2)
                    self.assertEqual(rt['mu0'], .02)
                    self.assertEqual(rt['R_gas'], 1. if transport == 'constant' else 1./600.)
                    source_type = 'trigonometric-cns-mms' if transport == 'constant' else 'trigonometric-cns-sutherland-mms'
                    self.assertEqual(rt['sources'][0]['type'], source_type)
                    self.assertEqual(case['dt'], .0005 if case['time_control'] else .001)
                    error = case['h']**(degree + 1)
                    data.append(dict(**case, component_physical={'l2': [error]*4}))
                self.assertTrue(assess_fluxes(data)['passed'])
                with self.assertRaisesRegex(RuntimeError, 'three meshes'):
                    assess_fluxes(data[:-1])
                contaminated = json.loads(json.dumps(data))
                contaminated[-1]['component_physical']['l2'] = [2.*v for v in contaminated[-1]['component_physical']['l2']]
                self.assertFalse(assess_fluxes(contaminated)['passed'])
                low_order = json.loads(json.dumps(data))
                for row in low_order:
                    row['component_physical']['l2'] = [row['h']]*4
                self.assertFalse(assess_fluxes(low_order)['passed'])

    def test_temporal_reference_rejects_mixed_time_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_temporal(SOURCE, root)
            cases = json.loads((root/'cases.json').read_text())
            cases[0]['time_scale'] += 1.
            (root/'cases.json').write_text(json.dumps(cases))
            with self.assertRaisesRegex(ValueError, 'common time scale'):
                check_temporal(root, 'cpu')


if __name__ == '__main__':
    unittest.main()
