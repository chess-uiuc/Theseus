#!/usr/bin/env python3
"""Failure, freshness and background-execution contracts of the suite runner."""
import argparse
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('run_tests', Path(__file__).resolve().parents[1] / 'scripts/run_tests.py')
run_tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_tests)


class TestRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'repo'
        self.source.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.source)], check=True)
        (self.source / 'code.cpp').write_text('original\n')
        subprocess.run(['git', '-C', str(self.source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(self.source), '-c', 'user.name=Test',
                        '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'fixture'], check=True)
        self.bin = self.root / 'tools'
        self.bin.mkdir()
        for name in ['cmake', 'ctest', 'mpicc', 'mpicxx', 'mpiexec', 'python-stub']:
            p = self.bin / name
            p.write_text('#!/bin/sh\necho fixture tool\nexit 0\n')
            p.chmod(0o755)
        self.environment = patch.dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                                      CC='mpicc', CXX='mpicxx')
        self.environment.start()
        self.addCleanup(self.environment.stop)
        (self.source / 'scripts').mkdir()
        shutil.copy(SPEC.origin, self.source / 'scripts/run_tests.py')
        shutil.copy(Path(SPEC.origin).with_name('mpi_launcher.py'), self.source / 'scripts/mpi_launcher.py')
        (self.source / 'scripts/run_theseus.sh').write_text('exit 0\n')
        recipes = self.source / 'tests/recipes'
        recipes.mkdir(parents=True, exist_ok=True)
        (recipes / 'recipes.toml').write_text('schema = 1\n')
        self.prefix = self.root / 'prefix'
        self.prefix.mkdir()
        self.run = self.root / 'results'

    def prepare(self, kernel_stress=False):
        return run_tests.prepare(argparse.Namespace(source=self.source, results=self.run,
            prefix=self.prefix, python=str(self.bin / 'python-stub'), mpiexec='mpiexec',
            mpi_arg=[], jobs=1, timeout=5, kernel_stress=kernel_stress))

    def steps(self, run, manifest):
        return [run_tests.Command('first', [sys.executable, '-c', 'print("first")'], run,
                                 'Inviscid Isentropic Vortex — cyclic run'),
                run_tests.Command('second', [sys.executable, '-c', 'print("second")'], run,
                                 'Inviscid Isentropic Vortex — cyclic comparison', ('first',))]

    def test_single_build_geometry_and_evidence(self):
        for axis in (False, True):
            with self.subTest(axis=axis):
                build = self.root / ('axis-build' if axis else 'cartesian-build')
                (build / 'tests').mkdir(parents=True)
                (build / 'CMakeCache.txt').write_text(
                    f'AXISYMMETRIC:BOOL={"ON" if axis else "OFF"}\n'
                    'THESEUS_WITH_PLATO:BOOL=ON\nENABLE_CUDA:BOOL=ON\n')
                for name in ('theseus', 'tests/table_lookup_tests'):
                    executable = build / name
                    executable.write_text('#!/bin/sh\nexit 0\n')
                    executable.chmod(0o755)
                args = argparse.Namespace(build=build, build_standard=None,
                    build_axisymmetric=None, device='cuda')
                builds, evidence = run_tests.prepared_builds(args)
                self.assertEqual(set(builds), {'axisymmetric' if axis else 'standard'})
                self.assertIn(str(build / 'theseus'), evidence)
                run_tests.verify_prepared_builds({'build_files': evidence})
                (build / 'theseus').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Prepared build changed'):
                    run_tests.verify_prepared_builds({'build_files': evidence})

    def test_single_build_selection_covers_all_and_only_its_geometry(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        run = self.root / 'selected-suite'
        run.mkdir()
        (run / 'source').symlink_to(repository, target_is_directory=True)
        manifest = {'suite': 'integrated', 'python': sys.executable,
                    'prefix': str(self.prefix), 'device': 'cuda', 'builds': {
                        'standard': str(self.root / 'standard'),
                        'axisymmetric': str(self.root / 'axisymmetric')}}
        all_names = {c.name for c in run_tests.suite_commands(run, manifest)}
        selected = []
        for geometry in ('standard', 'axisymmetric'):
            one = dict(manifest, suite='prepared', builds={geometry: manifest['builds'][geometry]})
            commands = run_tests.suite_commands(run, one)
            names = [c.name for c in commands]
            self.assertEqual(len(names), len(set(names)))
            self.assertIn('ctest-' + geometry, names)
            ctest = next(c for c in commands if c.name == 'ctest-' + geometry)
            self.assertIn('--no-tests=error', ctest.argv)
            self.assertIn(manifest['builds'][geometry], ctest.argv)
            self.assertFalse(any(n.startswith(('configure-', 'build-')) for n in names))
            selected.append(set(names) - {'ctest-' + geometry})
        self.assertEqual(selected[0] | selected[1], all_names)
        self.assertEqual(selected[0] & selected[1], {'python-dependencies'})
        self.assertIn('recipe-golden-channel-run', selected[0])
        self.assertIn('integration-helmholtz-lteec', selected[0])
        self.assertIn('integration-axis-restart', selected[1])

    def test_single_build_rejects_ambiguous_or_incompatible_selection(self):
        args = argparse.Namespace(build=self.root, build_standard=self.root,
                                  build_axisymmetric=None, device='cpu')
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            run_tests.prepared_builds(args)
        args.build = None
        args.build_standard = None
        with self.assertRaisesRegex(ValueError, 'at least one'):
            run_tests.prepared_builds(args)

    def test_results_inside_checkout_do_not_enter_snapshot_or_hide_source_changes(self):
        self.run = self.source / 'Run' / 'test-results'
        self.prepare()
        manifest = json.loads((self.run / 'manifest.json').read_text())
        self.assertEqual(manifest['excluded_results'], 'Run/test-results')
        self.assertFalse(any(name.startswith('Run/test-results/') for name in manifest['files']))
        self.assertFalse((self.run / 'source/Run/test-results').exists())
        with patch.object(run_tests, 'suite_commands', self.steps):
            self.assertEqual(run_tests.worker(self.run), 0)
            (self.run / 'additional-output.txt').write_text('Output is not source')
            self.assertEqual(run_tests.check(self.run, self.source), 0)
            other = self.source / 'Run/another-input.json'
            other.write_text('Real input outside the exact results directory')
            with self.assertRaisesRegex(ValueError, 'stale'):
                run_tests.check(self.run, self.source)

    def test_run_directory_links_do_not_prevent_in_tree_testing(self):
        case = self.source / 'TestCases/example'
        case.mkdir(parents=True)
        (case / 'config.json').write_text('{"input": 1}')
        work = self.source / 'Run'
        work.mkdir()
        (work / 'TestCases').symlink_to('../TestCases', target_is_directory=True)
        (work / 'external').symlink_to(self.root / 'external', target_is_directory=True)
        (work / 'config-link.json').symlink_to('../TestCases/example/config.json')
        self.run = work / 'Debug/results'
        self.prepare()
        manifest = json.loads((self.run / 'manifest.json').read_text())
        self.assertIn('TestCases/example/config.json', manifest['files'])
        for name in ('Run/TestCases', 'Run/external', 'Run/config-link.json'):
            self.assertNotIn(name, manifest['files'])
        with patch.object(run_tests, 'suite_commands', self.steps):
            self.assertEqual(run_tests.worker(self.run), 0)
            self.assertEqual(run_tests.check(self.run, self.source), 0)
            (case / 'config.json').write_text('{"input": 2}')
            with self.assertRaisesRegex(ValueError, 'stale'):
                run_tests.check(self.run, self.source)

    def test_versioned_symlinks_are_not_silently_omitted(self):
        link = self.source / 'versioned-link'
        link.symlink_to('code.cpp')
        subprocess.run(['git', '-C', str(self.source), 'add', 'versioned-link'], check=True)
        with self.assertRaisesRegex(ValueError, 'Source symlinks'):
            self.prepare()

    def test_results_cannot_replace_source_or_git_metadata(self):
        for location in (self.source, self.source / '.git/new-results'):
            with self.subTest(location=location):
                self.run = location
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_prepared_cli_runs_ctest_despite_workspace_and_recipe_problems(self):
        # Exercise the real CLI and worker with fixture commands, not CFD results.
        shutil.rmtree(self.source / '.git')
        work = self.source / 'Run'
        work.mkdir()
        (work / 'TestCases').symlink_to('../TestCases', target_is_directory=True)
        (work / 'broken').symlink_to('/nonexistent/local-path')
        build = self.root / 'prepared-build'
        build.mkdir()
        (build / 'CMakeCache.txt').write_text('AXISYMMETRIC:BOOL=OFF\nTHESEUS_WITH_PLATO:BOOL=OFF\nENABLE_CUDA:BOOL=ON\n')
        (build / 'theseus').write_text('#!/bin/sh\nexit 0\n')
        (build / 'theseus').chmod(0o755)
        recipes = self.source / 'tests/recipes'
        good = recipes / 'good.sh'
        good.write_text('#!/bin/bash\necho good-recipe-ran\n')
        good.chmod(0o644)  # Recipes are invoked by Bash, so +x is unnecessary.
        (recipes / 'extra.sh').write_text('#!/bin/bash\n')
        records = ['schema = 1']
        for name, geometry in (('missing', 'standard'), ('good', 'standard'),
                               ('unavailable-axis', 'axisymmetric')):
            records.append(f"""
[[test]]
id = "{name}"
title = "{name}"
script = "tests/recipes/{name}.sh"
suites = ["integrated", "full"]
build = "{geometry}"
requires = []
""")
        (recipes / 'recipes.toml').write_text('\n'.join(records))
        results = work / 'results'
        missing_python_package = self.root / 'missing-package'
        missing_python_package.mkdir()
        (missing_python_package / 'pyvista.py').write_text('raise ImportError("fixture missing PyVista")')
        env = dict(os.environ, PYTHONPATH=str(missing_python_package))
        process = subprocess.run([sys.executable, str(self.source / 'scripts/run_tests.py'),
            'run', '--source', str(self.source), '--build', str(build),
            '--prefix', str(self.prefix), '--results', str(results),
            '--python', sys.executable, '--device', 'cuda'], text=True, capture_output=True, env=env)
        self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
        saved = json.loads((results / 'results.json').read_text())
        steps = {step['name']: step for step in saved['steps']}
        self.assertEqual(steps['python-dependencies']['state'], 'failed')
        self.assertEqual(steps['ctest-standard']['state'], 'passed')
        self.assertEqual(steps['recipe-good']['state'], 'passed')
        self.assertEqual(steps['recipe-registration-0']['state'], 'failed')
        self.assertFalse(any('unavailable-axis' in name for name in steps))
        self.assertIn('WARNING: Unregistered', process.stderr)
        self.assertFalse((results / 'source').exists())
        self.assertTrue((work / 'TestCases').is_symlink())
        # Missing table_lookup_tests and PLATO do not prevent unrelated tests.
        self.assertIn('Missing build executable', process.stderr)

    def test_invalid_recipe_manifest_reports_failure_but_schedules_other_checks(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        run = self.root / 'manifest-failure'
        (run / 'source/tests/recipes').mkdir(parents=True)
        (run / 'source/tests/recipes/recipes.toml').write_text('invalid = [')
        manifest = {'suite': 'prepared', 'python': sys.executable,
                    'prefix': str(self.prefix), 'builds': {'standard': str(self.root)}}
        commands = run_tests.suite_commands(run, manifest)
        names = {command.name for command in commands}
        self.assertIn('ctest-standard', names)
        self.assertIn('integration-helmholtz-lteec', names)
        self.assertIn('recipe-registration-0', names)

    def test_forced_research_channel_is_selected_once_only_when_enabled(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        manifest = {'suite': 'prepared', 'live_source': True, 'source': str(repository),
                    'python': sys.executable, 'prefix': str(self.prefix),
                    'builds': {'standard': str(self.root)}, 'research_source_enabled': True}
        commands = run_tests.suite_commands(self.root, manifest)
        self.assertEqual([c.name for c in commands].count('integration-research-channel'), 1)
        manifest['research_source_enabled'] = False
        self.assertNotIn('integration-research-channel', [c.name for c in run_tests.suite_commands(self.root, manifest)])
        manifest['research_source_enabled'] = True
        manifest['builds'] = {'axisymmetric': str(self.root)}
        self.assertNotIn('integration-research-channel', [c.name for c in run_tests.suite_commands(self.root, manifest)])

    def test_kernel_stress_is_disabled_by_default(self):
        self.prepare()
        manifest = json.loads((self.run / 'manifest.json').read_text())
        self.assertFalse(manifest['kernel_stress'])
        configure = [item for item in run_tests.suite_commands(self.run, manifest)
                     if item.name.startswith('configure-')]
        self.assertEqual(len(configure), 2)
        for item in configure:
            self.assertIn('-DTHESEUS_BUILD_KERNEL_STRESS=OFF', item.argv)

    def test_kernel_stress_opt_in_is_recorded_for_both_builds(self):
        self.prepare(kernel_stress=True)
        manifest = json.loads((self.run / 'manifest.json').read_text())
        self.assertTrue(manifest['kernel_stress'])
        configure = [item for item in run_tests.suite_commands(self.run, manifest)
                     if item.name.startswith('configure-')]
        self.assertEqual(len(configure), 2)
        for item in configure:
            self.assertIn('-DTHESEUS_BUILD_KERNEL_STRESS=ON', item.argv)

    def test_integrated_mode_cannot_silently_ignore_kernel_stress(self):
        args = argparse.Namespace(source=self.source, results=self.run,
                                  prefix=self.prefix, suite='integrated', kernel_stress=True)
        with self.assertRaisesRegex(ValueError, '--kernel-stress requires --suite full'):
            run_tests.prepare(args)
        self.assertFalse(self.run.exists())

    def test_pass_requires_complete_unchanged_evidence(self):
        self.prepare()
        with patch.object(run_tests, 'suite_commands', self.steps):
            self.assertEqual(run_tests.worker(self.run), 0)
            self.assertEqual(run_tests.check(self.run, self.source), 0)
            (self.source / 'README.md').write_text('Documentation only')
            self.assertEqual(run_tests.check(self.run, self.source), 0)
            (self.source / 'new_test.py').write_text('assert False')
            with self.assertRaisesRegex(ValueError, 'stale'):
                run_tests.check(self.run, self.source)
            (self.source / 'new_test.py').unlink()
            (self.source / 'code.cpp').unlink()
            with self.assertRaisesRegex(ValueError, 'stale'):
                run_tests.check(self.run, self.source)
            (self.source / 'code.cpp').write_text('original\n')
            (self.run / 'logs/first.log').write_text('replaced')
            with self.assertRaisesRegex(ValueError, 'Log changed'):
                run_tests.check(self.run, self.source)

    def test_case_readmes_are_documentation_but_case_inputs_are_material(self):
        self.prepare()
        with patch.object(run_tests, 'suite_commands', self.steps):
            self.assertEqual(run_tests.worker(self.run), 0)
            case = self.source / 'TestCases/GoldenData/Example'
            case.mkdir(parents=True)
            (case / 'README.md').write_text('Reference documentation')
            self.assertEqual(run_tests.check(self.run, self.source), 0)
            for name in ('config.json', 'reference.npz', 'mesh.msh', 'expected.md'):
                with self.subTest(name=name):
                    candidate = case / name
                    candidate.write_text('Changed test input')
                    with self.assertRaisesRegex(ValueError, 'stale'):
                        run_tests.check(self.run, self.source)
                    candidate.unlink()

    def test_failure_blocks_dependents_but_independent_checks_finish(self):
        self.prepare()
        commands = [
            run_tests.Command('build', [sys.executable, '-c', 'raise SystemExit(7)'], self.run, 'Build'),
            run_tests.Command('run', [sys.executable, '-c', 'print("must not run")'],
                             self.run, 'Simulation', ('build',)),
            run_tests.Command('compare', [sys.executable, '-c', 'print("must not run")'],
                             self.run, 'Comparison', ('run',)),
            run_tests.Command('independent', [sys.executable, '-c', 'print("independent")'],
                             self.run, 'Independent check'),
        ]
        with patch.object(run_tests, 'suite_commands', return_value=commands):
            self.assertEqual(run_tests.worker(self.run), 1)
            results = json.loads((self.run / 'results.json').read_text())
            self.assertEqual([step['state'] for step in results['steps']],
                             ['failed', 'blocked', 'blocked', 'passed'])
            self.assertEqual(results['steps'][0]['returncode'], 7)
            self.assertNotIn('elapsed_seconds', results['steps'][1])
            self.assertFalse((self.run / 'logs/run.log').exists())
            with self.assertRaisesRegex(ValueError, 'not complete'):
                run_tests.check(self.run, self.source)

    def test_summary_keeps_separate_run_and_comparison_durations(self):
        self.prepare()
        with patch.object(run_tests, 'suite_commands', self.steps):
            self.assertEqual(run_tests.worker(self.run), 0)
        results = json.loads((self.run / 'results.json').read_text())
        summary = (self.run / 'summary.txt').read_text()
        for step in results['steps']:
            self.assertGreaterEqual(step['elapsed_seconds'], 0)
            self.assertIn(f"{step['elapsed_seconds']:.2f} s  {step['label']}", summary)
        self.assertIn('cyclic run', summary)
        self.assertIn('cyclic comparison', summary)

    def test_recipe_manifest_is_discoverable_and_human_readable(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        recipes = run_tests.load_recipes(repository)
        recipe = run_tests.select_recipe(recipes, 'golden-vortex')
        self.assertEqual(recipe['build'], 'standard')
        self.assertIn('regression', recipe['suites'])
        text = recipe['script_path'].read_text()
        self.assertIn('scripts/run_theseus.sh', text)
        self.assertIn('TestCases/Euler/2D/IsentropicVortex/IsentropicVortex.mesh', text)
        self.assertIn('scripts/compare_viz.py', text)
        self.assertIn('--atol 1e-13', text)
        for discovered in recipes:
            sections = run_tests.recipe_display_commands(discovered)
            self.assertEqual([stage for stage, _ in sections], discovered['stages'])

    def test_list_and_show_recipe_commands(self):
        runner = Path(SPEC.origin)
        repository = runner.parents[1]
        listed = subprocess.run([sys.executable, str(runner), 'list',
                                 '--source', str(repository)],
                                text=True, capture_output=True, check=False)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn('golden-vortex', listed.stdout)
        self.assertIn('Suites:', listed.stdout)
        self.assertIn('regression', listed.stdout)
        shown = subprocess.run([sys.executable, str(runner), 'show', 'golden-tgv',
                                '--source', str(repository)],
                               text=True, capture_output=True, check=False)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn('scripts/run_theseus.sh', shown.stdout)
        self.assertIn('TestCases/NavierStokes/2D/TaylorGreenVortex/config.json', shown.stdout)
        self.assertIn('-t 0.0001', shown.stdout)
        self.assertIn('-n 100', shown.stdout)
        self.assertIn('prepare_golden_reference.py', shown.stdout)
        self.assertIn('compare_viz.py', shown.stdout)
        self.assertIn('--atol 1e-13 --rtol 1e-13', shown.stdout)
        self.assertNotIn('recipe_parse_arguments', shown.stdout)
        self.assertNotIn('set -euo pipefail', shown.stdout)

        suite = subprocess.run([sys.executable, str(runner), 'show', 'regression',
                                '--source', str(repository)],
                               text=True, capture_output=True, check=False)
        self.assertEqual(suite.returncode, 0, suite.stderr)
        suite_ids = ['cyclic-vortex', 'golden-vortex', 'golden-lte-vortex',
                     'golden-cavity', 'golden-tgv', 'golden-step']
        positions = [suite.stdout.index(f'# {identifier}:') for identifier in suite_ids]
        self.assertEqual(positions, sorted(positions))
        for identifier in suite_ids:
            self.assertEqual(suite.stdout.count(f'# {identifier}:'), 1)

    def test_numflux_recipe_is_registered_for_full_and_integrated_suites(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        recipe = run_tests.select_recipe(run_tests.load_recipes(repository), 'golden-numflux')
        self.assertEqual(recipe['build'], 'standard')
        self.assertTrue({'full', 'integrated', 'regression'} <= set(recipe['suites']))
        self.assertEqual(recipe['stages'], ['prepare', 'run', 'compare'])
        sections = dict(run_tests.recipe_display_commands(recipe))
        self.assertIn('cpg-hll-cavity cpg-roe-cavity lte-hll-tgv', sections['run'])
        self.assertIn('scripts/run_theseus.sh', sections['run'])
        self.assertIn('numflux_regression_test.py', sections['compare'])

    def test_recipe_suites_reject_empty_suites(self):
        recipe_script = self.source / 'tests/recipes/example.sh'
        recipe_script.write_text(
            '#!/usr/bin/env bash\n'
            '# THESEUS_SHOW run\n'
            'scripts/run_theseus.sh -c config.json\n'
            '# THESEUS_SHOW_END\n'
        )
        recipe_script.chmod(0o755)
        (self.source / 'tests/recipes/recipes.toml').write_text(
            'schema = 1\n'
            '[[suite]]\n'
            'id = "all"\n'
            'title = "All recipes"\n'
            '[[test]]\n'
            'id = "example"\n'
            'title = "Example"\n'
            'script = "tests/recipes/example.sh"\n'
            'suites = ["full"]\n'
            'build = "standard"\n'
            'requires = []\n'
            'stages = ["run"]\n'
        )
        recipes = run_tests.load_recipes(self.source)
        with self.assertRaisesRegex(ValueError, 'suite has no members: all'):
            run_tests.load_recipe_suites(self.source, recipes)

    def test_recipe_command_exposes_all_direct_execution_inputs(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        recipe = run_tests.select_recipe(run_tests.load_recipes(repository), 'golden-vortex')
        args = argparse.Namespace(
            source=repository,
            build_standard=self.root / 'build',
            prefix=self.prefix,
            results=self.root / 'recipe-output',
            device='cuda',
            python='python3',
            ranks=4,
        )
        command = run_tests.recipe_command(recipe, args)
        self.assertEqual(command[0], 'bash')
        self.assertIn(str(recipe['script_path']), command)
        for value in ('--source', '--build-standard', '--prefix', '--output',
                      '--device', 'cuda', '--python', 'python3', '--ranks', '4'):
            self.assertIn(value, command)

    def test_recipe_run_applies_one_top_level_mpi_override(self):
        script = self.root / 'recipe.sh'
        script.write_text(
            '#!/usr/bin/env bash\n'
            'set -euo pipefail\n'
            'test -x "$THESEUS_MPI_LAUNCHER"\n'
        )
        script.chmod(0o755)
        recipe = {'id': 'fixture', 'build': 'standard', 'script_path': script}
        args = argparse.Namespace(
            source=self.source,
            build_standard=self.root / 'build',
            prefix=self.prefix,
            results=self.root / 'recipe-output',
            device='cpu',
            python=sys.executable,
            ranks=2,
            mpiexec='mpiexec',
            mpi_arg=['--host', 'localhost:2'],
        )
        self.assertEqual(run_tests.run_recipe(recipe, args), 0)

    def test_recipe_environment_exposes_dependency_prefix_to_relative_configs(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        helper = repository / 'tests/recipes/recipe-environment.bash'
        build = self.root / 'build'
        build.mkdir()
        executable = build / 'theseus'
        executable.write_text('#!/bin/sh\n')
        executable.chmod(0o755)
        output = self.root / 'recipe-output'
        command = f'source "{helper}"; recipe_parse_arguments "$@"'
        subprocess.run([
            'bash', '-c', command, 'recipe-fixture',
            '--source', str(self.source),
            '--build-standard', str(build),
            '--prefix', str(self.prefix),
            '--output', str(output),
            '--python', sys.executable,
        ], check=True)
        link = output / 'tpl/install'
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.prefix.resolve())

    def test_registered_recipes_replace_legacy_simulation_commands(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        run = self.root / 'suite'
        run.mkdir()
        (run / 'source').symlink_to(repository, target_is_directory=True)
        manifest = {
            'suite': 'full',
            'python': sys.executable,
            'prefix': str(self.prefix),
            'device': 'cpu',
            'jobs': 1,
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'mpiexec': None,
            'builds': {},
        }
        commands = run_tests.suite_commands(run, manifest)
        names = [command.name for command in commands]
        expected = {
            'axis-uniform': ('prepare', 'run', 'compare'),
            'axis-convergence': ('prepare', 'run', 'compare'),
            'smoke-vortex': ('run',),
            'smoke-cavity': ('run',),
            'smoke-lte': ('run',),
            'cyclic-vortex': ('run', 'compare'),
            'golden-vortex': ('run', 'prepare', 'compare'),
            'golden-lte-vortex': ('run', 'prepare', 'compare'),
            'golden-cavity': ('run', 'prepare', 'compare'),
            'golden-tgv': ('run', 'prepare', 'compare'),
            'golden-channel': ('prepare', 'run', 'compare'),
            'golden-step': ('run', 'prepare', 'compare'),
        }
        for recipe, stages in expected.items():
            for stage in stages:
                self.assertEqual(names.count(f'recipe-{recipe}-{stage}'), 1)
        for duplicate in (
            'integration-axis-uniform',
            'integration-axis-convergence',
            'smoke-vortex', 'smoke-cavity', 'smoke-lte', 'cyclic',
            'compare-cyclic', 'golden-vortex', 'golden-lte', 'golden-cavity',
            'golden-tgv', 'golden-step', 'prepare-reference-vortex',
            'prepare-reference-lte', 'prepare-reference-cavity',
            'prepare-reference-tgv', 'prepare-reference-step', 'compare-vortex',
            'compare-lte', 'compare-cavity', 'compare-tgv', 'compare-step',
        ):
            self.assertNotIn(duplicate, names)
        by_name = {command.name: command for command in commands}
        self.assertIn('Theseus run', by_name['recipe-golden-vortex-run'].label)
        self.assertIn('recipe-golden-vortex-run',
                      by_name['recipe-golden-vortex-prepare'].requires)
        self.assertIn('recipe-golden-vortex-prepare',
                      by_name['recipe-golden-vortex-compare'].requires)
        self.assertIn('--stage', by_name['recipe-golden-vortex-run'].argv)

        manifest['suite'] = 'integrated'
        manifest['builds'] = {
            'standard': str(self.root / 'build-standard'),
            'axisymmetric': str(self.root / 'build-axisymmetric'),
        }
        names = [command.name for command in run_tests.suite_commands(run, manifest)]
        for recipe, stages in expected.items():
            for stage in stages:
                self.assertEqual(names.count(f'recipe-{recipe}-{stage}'), 1)

    def test_axisymmetric_uniform_flow_helper_prepares_and_checks_recipe_data(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        helper = repository / 'tests/axisymmetric_uniform_flow_integration_test.py'
        results = self.root / 'uniform-flow'
        prepared = subprocess.run(
            [sys.executable, str(helper), 'prepare', '--source', str(repository),
             '--results', str(results)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        output = (
            'rho(1.0,1.0), p(0.7142857142857143,0.7142857142857143)\n'
            'TotalChange: Mass: 0.0, Energy: 0.0, K.E.: 0.0\n'
        )
        for name in ('euler', 'navier-stokes'):
            for ranks in (1, 2):
                directory = results / f'{name}-{ranks}'
                config = json.loads((directory / 'config.json').read_text())
                self.assertEqual(config['runTime']['output_file_path'], str(directory))
                (directory / 'run.log').write_text(output)
        checked = subprocess.run(
            [sys.executable, str(helper), 'check', '--source', str(repository),
             '--results', str(results)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_axisymmetric_convergence_helper_prepares_and_checks_recipe_data(self):
        repository = Path(SPEC.origin).resolve().parents[1]
        helper = repository / 'tests/axisymmetric_entropy_wave_convergence_test.py'
        results = self.root / 'entropy-wave'
        prepared = subprocess.run(
            [sys.executable, str(helper), 'prepare', '--source', str(repository),
             '--results', str(results)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        for level, error in enumerate((1.0, 0.25, 0.0625)):
            directory = results / f'level-{level}'
            config = json.loads((directory / 'config.json').read_text())
            self.assertEqual(config['runTime']['ser_ref_levels'], level)
            (directory / 'run.log').write_text(f'Exact L2 Error: {error}\n')
        checked = subprocess.run(
            [sys.executable, str(helper), 'check', '--source', str(repository),
             '--results', str(results)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_recipe_discovery_warns_without_rejecting_unregistered_shell_scripts(self):
        recipes = self.source / 'tests/recipes'
        recipes.mkdir(parents=True, exist_ok=True)
        (recipes / 'recipes.toml').write_text('schema = 1\n')
        orphan = recipes / 'orphan.sh'
        orphan.write_text('#!/usr/bin/env bash\n')
        orphan.chmod(0o755)
        warnings = io.StringIO()
        with contextlib.redirect_stderr(warnings):
            self.assertEqual(run_tests.load_recipes(self.source), [])
        self.assertIn('WARNING: Unregistered', warnings.getvalue())

    def test_incomplete_missing_and_overlapping_runs(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'not complete'):
            run_tests.check(self.run, self.source)
        (self.run / 'results.json').unlink()
        with self.assertRaises(FileNotFoundError):
            run_tests.check(self.run, self.source)

    def test_snapshot_is_independent_and_detects_mutation(self):
        self.prepare()
        manifest = json.loads((self.run / 'manifest.json').read_text())
        (self.source / 'code.cpp').write_text('edited during review')
        self.assertEqual((self.run / 'source/code.cpp').read_text(), 'original\n')
        run_tests.verify_snapshot(self.run, manifest)
        (self.run / 'source/code.cpp').write_text('bad')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            run_tests.verify_snapshot(self.run, manifest)

    def test_timeout_records_failure(self):
        self.prepare()
        results = {'state': 'running', 'steps': []}
        self.assertFalse(run_tests.execute_step(self.run, results, 'hang',
            [sys.executable, '-c', 'import time; time.sleep(30)'], self.run, os.environ, 0.1))
        self.assertEqual(results['steps'][0]['state'], 'failed')

    def test_mpi_placement_defaults_yield_to_test_arguments(self):
        self.prepare()
        launcher = self.bin / 'mpiexec'
        launcher.write_text('#!' + sys.executable + '\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        helper = self.source / 'scripts/mpi_launcher.py'
        defaults = ['--host', 'localhost:4', '--map-by', 'slot:OVERSUBSCRIBE',
                    '--bind-to', 'none', '--mca', 'example', 'keep']
        for supplied, expected_defaults in [
            (['-n', '2', 'application'], defaults),
            (['--host', 'localhost:2', '--map-by', 'slot:OVERSUBSCRIBE',
              '--bind-to', 'none', '-n', '2', 'application'], ['--mca', 'example', 'keep']),
            (['--map-by=slot', '-n', '1', 'application'],
             ['--host', 'localhost:4', '--bind-to', 'none', '--mca', 'example', 'keep']),
        ]:
            output = subprocess.check_output([sys.executable, str(helper), str(launcher),
                                              json.dumps(defaults), *supplied], text=True)
            self.assertEqual(json.loads(output), expected_defaults + supplied)

    def test_background_run_survives_launcher_exit(self):
        command = [sys.executable, str(self.source / 'scripts/run_tests.py'), 'start',
                   '--source', str(self.source), '--results', str(self.run),
                   '--prefix', str(self.prefix), '--python', str(self.bin / 'python-stub')]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, timeout=10)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            results = json.loads((self.run / 'results.json').read_text())
            if results['state'] in {'passed', 'failed'}:
                break
            time.sleep(0.05)
        self.assertEqual(results['state'], 'passed', results)
        self.assertEqual(run_tests.check(self.run, self.source), 0)
        self.assertEqual(len(results['steps']), len(run_tests.suite_commands(self.run,
            json.loads((self.run / 'manifest.json').read_text()))))


if __name__ == '__main__':
    unittest.main()
