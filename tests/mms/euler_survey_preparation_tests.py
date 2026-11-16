"""Input contracts for extended Euler spatial and temporal verification studies."""
from mms_paths import REPOSITORY_ROOT
import collections
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from unittest.mock import patch
import unittest
import numpy as np
from euler_rhs_study import prepare as prepare_rhs, exact_time_derivative, TIMES
from trigonometric_euler_study import exact_fields
from euler_spatial_survey import prepare as prepare_spatial, write_periodic_square, completed_cases
from euler_temporal_survey import prepare as prepare_temporal, TIME_SCALE
from euler_flux_comparison import prepare as prepare_flux, FLUXES
from mms_fluxes import select_fluxes

SOURCE = REPOSITORY_ROOT


class PreparationTests(unittest.TestCase):
    def test_periodic_square_has_paired_edges_and_exact_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'mesh'
            write_periodic_square(path)
            lines = path.read_text().splitlines()
            start = lines.index('elements')
            count = int(lines[start + 1])
            edges = collections.Counter()
            for line in lines[start + 2:start + 2 + count]:
                nodes = list(map(int, line.split()[2:]))
                for a, b in zip(nodes, nodes[1:] + nodes[:1]):
                    edges[tuple(sorted((a, b)))] += 1
            self.assertEqual(count, 9)
            self.assertTrue(all(value == 2 for value in edges.values()))
            self.assertEqual(lines[lines.index('boundary') + 1], '0')

    def test_spatial_study_changes_only_resolution_and_has_time_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_spatial(SOURCE, root)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 30)
            for degree in range(1, 6):
                group = [r for r in cases if r['order'] == degree]
                fine = next(r for r in group if r['name'] == f'p{degree}-level-4')
                control = next(r for r in group if r.get('time_control'))
                self.assertEqual(control['h'], fine['h'])
                self.assertEqual(control['dt'], fine['dt']/2.)
                self.assertEqual(len({r['h'] for r in group}), 5)

    def test_partial_analysis_excludes_interrupted_and_unstarted_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [dict(name=name) for name in ('finished', 'interrupted', 'unstarted')]
            for case in cases[:2]:
                (root/case['name']).mkdir()
            (root/'finished/run.log').write_text('Total: 1 | Succeeded: 1 | Failed: 0\n')
            (root/'interrupted/run.log').write_text('time step: 375\n')
            complete, excluded = completed_cases(root, cases)
            self.assertEqual(complete, [cases[0]])
            self.assertEqual(excluded, ['interrupted', 'unstarted'])

    def test_fine_grid_study_resolves_point_one_and_has_time_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_spatial(SOURCE, root, cells=20, levels=4, timestep=.000125)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 25)
            for degree in range(1, 6):
                group = [c for c in cases if c['order'] == degree]
                main = [c for c in group if not c.get('time_control')]
                np.testing.assert_allclose([c['h'] for c in main], [.1, .05, .025, .0125])
                control = next(c for c in group if c.get('time_control'))
                self.assertEqual(control['h'], .0125)
                self.assertEqual(control['dt'], main[-1]['dt']/2.)
                for case in group:
                    rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                    self.assertFalse(rt['variable_dt'])
                    self.assertEqual(rt['dt'], case['dt'])
                    self.assertGreater(rt['nsteps_max']*rt['dt'], rt['final_time'])

    def test_flux_comparison_changes_only_flux_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_flux(SOURCE, root)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 36)
            for degree in (2, 4):
                for level in (2, 3, 4):
                    inputs = []
                    for flux in FLUXES:
                        name = f'{flux}-p{degree}-level-{level}'
                        runtime = json.loads((root/name/'config.json').read_text())['runTime']
                        self.assertEqual(runtime.pop('numerical_flux'), flux)
                        runtime.pop('output_file_path')
                        inputs.append(runtime)
                    self.assertTrue(all(runtime == inputs[0] for runtime in inputs))

    def test_flux_selection_aliases_and_invalid_requests(self):
        self.assertEqual(select_fluxes('all'), FLUXES)
        self.assertEqual(select_fluxes('LLF, roe'), ('lfr', 'Roe'))
        for request in ('', 'bogus', 'all,Roe', 'LLF,lfr', 'Roe,Roe'):
            with self.assertRaises(ValueError):
                select_fluxes(request)

    def test_quick_flux_selection_and_rank_count_reach_every_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_flux(SOURCE, root, ('Roe', 'HLL'), quick=True, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 6)
            self.assertEqual({c['flux'] for c in cases}, {'Roe', 'HLL'})
            self.assertEqual({c['ranks'] for c in cases}, {2})
            self.assertEqual({c['order'] for c in cases}, {2})
            self.assertEqual(sum(bool(c.get('time_control')) for c in cases), 0)
            for line, case in zip((root/'cases.tsv').read_text().splitlines(), cases):
                self.assertEqual(line, f"{case['name']}\t2")

    def test_full_matrix_keeps_controls_for_each_selected_flux_and_degree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_flux(SOURCE, root, ('Roe',), matrix=True, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 30)
            for degree in range(1, 6):
                group = [c for c in cases if c['order'] == degree]
                main = [c for c in group if not c.get('time_control')]
                control = next(c for c in group if c.get('time_control'))
                self.assertEqual(len(main), 5)
                self.assertEqual(control['h'], min(c['h'] for c in main))
                self.assertFalse(control['accuracy_gate'])
                self.assertEqual(control['dt'], main[-1]['dt']/2.)

    def test_rhs_subset_runs_requested_ranks_without_parity_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_rhs(SOURCE, root, ('HLL',), ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 16)
            self.assertEqual({c['flux'] for c in cases}, {'HLL'})
            self.assertEqual({c['ranks'] for c in cases}, {2})
            self.assertFalse(any(c.get('parity_control') for c in cases))

    def test_flux_qualification_retains_time_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_flux(SOURCE, root, ('Roe', 'HLL'), quick=True, qualified=True, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 8)
            self.assertEqual(sum(bool(c.get('time_control')) for c in cases), 2)

    def test_quick_flux_checker_rejects_low_order_even_with_clean_time_control(self):
        from euler_flux_comparison import check
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_flux(SOURCE, root, ('Roe',), quick=True, qualified=True)
            def low_order(results, case, device, **options):
                error = .1/(2**case['level'])
                return dict(**case, component_physical={'l2': [error]*4}), {}
            with patch('euler_flux_comparison.read_case', side_effect=low_order):
                with self.assertRaisesRegex(RuntimeError, 'order'):
                    check(root, 'cpu')
            report = json.loads((root/'flux-comparison.json').read_text())
            self.assertTrue(report[0]['timestep_control_passed'])
            self.assertFalse(report[0]['passed'])

    def test_qualification_cannot_pass_with_missing_control(self):
        from euler_flux_comparison import check as check_flux
        from euler_temporal_survey import check as check_temporal
        for kind in ('flux', 'temporal'):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if kind == 'flux':
                    prepare_flux(SOURCE, root, ('Roe',), quick=True, qualified=True)
                    control_key = 'time_control'
                    checker = check_flux
                    module = 'euler_flux_comparison'
                else:
                    prepare_temporal(SOURCE, root, quick=True, qualified=True)
                    control_key = 'spatial_control'
                    checker = check_temporal
                    module = 'euler_temporal_survey'
                cases = json.loads((root/'cases.json').read_text())
                cases = [c for c in cases if not c.get(control_key)]
                (root/'cases.json').write_text(json.dumps(cases))
                def fields(results, case, device, *values, **options):
                    error = case['h']**3 if kind == 'flux' else case['dt']**(3 if case['method'] == 'RK3SSP' else 4)
                    return dict(**case, component_physical={'l2': [error]*4}), {}
                with patch(module + '.read_case', side_effect=fields):
                    with self.assertRaisesRegex(RuntimeError, 'Missing required'):
                        checker(root, 'cpu')

    def test_flux_recipe_uses_manifest_ranks_and_requested_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            (source/'scripts').mkdir(parents=True)
            build = root/'build'
            build.mkdir()
            (build/'theseus').write_text('#!/bin/sh\nexit 0\n')
            (build/'theseus').chmod(0o755)
            output = root/'output'
            cases = output/'cases'
            cases.mkdir(parents=True)
            (output/'.theseus-recipe-prepared').touch()
            (cases/'cases.tsv').write_text('one\t3\ntwo\t4\n')
            for name in ('one', 'two'):
                (cases/name).mkdir()
            harness = source/'scripts/run_theseus.sh'
            harness.write_text('#!/bin/bash\ncat > /dev/null\nprintf "%s\\n" "$@"\n')
            harness.chmod(0o755)
            subprocess.run(['bash', str(SOURCE/'tests/recipes/integration/euler-flux-quick.sh'),
                            '--source', str(source), '--build-standard', str(build),
                            '--prefix', str(root), '--output', str(output), '--python', sys.executable,
                            '--device', 'cuda', '--stage', 'run'], check=True)
            for name, ranks in (('one', '3'), ('two', '4')):
                arguments = (cases/name/'run.log').read_text().splitlines()
                self.assertEqual(arguments[arguments.index('-p') + 1], ranks)
                self.assertEqual(arguments[arguments.index('-r') + 1], 'cuda')

    def test_rhs_study_has_constant_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_rhs(SOURCE, root)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 96)
            self.assertEqual(sum(c['constant'] for c in cases), 6)
            self.assertEqual(sum(c['ranks'] == 2 for c in cases), 0)
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['rhs_times'], list(TIMES))
                self.assertEqual(bool(rt['sources']), not case['constant'])

    def test_rhs_loop_survives_launcher_reading_standard_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'
            (source/'scripts').mkdir(parents=True)
            build = root/'build'
            build.mkdir()
            (build/'theseus').write_text('#!/bin/sh\nexit 0\n')
            (build/'theseus').chmod(0o755)
            output = root/'results'
            cases = output/'cases'
            cases.mkdir(parents=True)
            (output/'.theseus-recipe-prepared').touch()
            names = ('first', 'second', 'third')
            (cases/'cases.tsv').write_text(''.join(f'{name}\t1\n' for name in names))
            for name in names:
                (cases/name).mkdir()
            # Reproduce an MPI launcher draining stdin, without starting MPI.
            harness = source/'scripts/run_theseus.sh'
            harness.write_text('#!/bin/bash\ncat > /dev/null\nprintf "launched\\n"\n')
            harness.chmod(0o755)
            subprocess.run(['bash', str(SOURCE/'tests/recipes/integration/euler-rhs.sh'),
                            '--source', str(source), '--build-standard', str(build),
                            '--prefix', str(root), '--output', str(output),
                            '--python', sys.executable, '--stage', 'run'], check=True)
            for name in names:
                self.assertEqual((cases/name/'run.log').read_text(), 'launched\n')

    def test_independent_time_derivative(self):
        points = np.array([[.12, .37], [-.71, .2], [.31, -.43]])
        for time in TIMES:
            step = 1e-4
            finite_difference = (-exact_fields(points, time + 2*step)
                                 + 8*exact_fields(points, time + step)
                                 - 8*exact_fields(points, time - step)
                                 + exact_fields(points, time - 2*step))/(12*step)
            np.testing.assert_allclose(exact_time_derivative(points, time), finite_difference,
                                       atol=2e-11, rtol=0.)

    def test_quick_temporal_study_is_bounded_and_has_no_qualification_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_temporal(SOURCE, root, quick=True, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 6)
            self.assertFalse(any(c['spatial_control'] for c in cases))
            for method in ('RK3SSP', 'RK4'):
                group = [c for c in cases if c['method'] == method]
                self.assertEqual([c['dt'] for c in group], [.001, .0005, .00025])
            for case in cases:
                self.assertEqual(case['final_time'], .02)
                self.assertEqual(case['ranks'], 2)
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['final_time'], case['final_time'])
                self.assertLessEqual(round(rt['final_time']/rt['dt']), 80)
                self.assertFalse(rt['variable_dt'])

    def test_temporal_qualification_retains_spatial_floor_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_temporal(SOURCE, root, quick=True, qualified=True, ranks=2)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual(len(cases), 8)
            for method in ('RK3SSP', 'RK4'):
                group = [c for c in cases if c['method'] == method]
                main = [c for c in group if not c['spatial_control']]
                control = next(c for c in group if c['spatial_control'])
                self.assertEqual(control['dt'], main[-1]['dt'])
                self.assertGreater(control['order'], main[-1]['order'])

    def test_temporal_study_uses_valid_methods_and_consistent_forcing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_temporal(SOURCE, root)
            cases = json.loads((root/'cases.json').read_text())
            self.assertEqual({r['method'] for r in cases}, {'RK3SSP', 'RK4'})
            for method in ('RK3SSP', 'RK4'):
                group = [r for r in cases if r['method'] == method]
                main = [r for r in group if not r['spatial_control']]
                self.assertEqual(len(main), 4)
                self.assertEqual(len({r['order'] for r in main}), 1)
                control = next(r for r in group if r['spatial_control'])
                self.assertGreater(control['order'], main[-1]['order'])
                self.assertEqual(control['dt'], main[-1]['dt'])
            for case in cases:
                rt = json.loads((root/case['name']/'config.json').read_text())['runTime']
                self.assertEqual(rt['sources'][0]['time_scale'], TIME_SCALE)
                self.assertEqual(rt['exact_solution']['params']['x1'], TIME_SCALE)
                self.assertEqual(rt['conditions']['initial_conditions']['params']['x1'], TIME_SCALE)
                self.assertGreater(rt['nsteps_max']*rt['dt'], rt['final_time'])
                self.assertGreaterEqual(rt['initial_save_dt'], rt['final_time'])


if __name__ == '__main__':
    unittest.main()
