#!/usr/bin/env python3
"""Failure, freshness and background-execution contracts of the suite runner."""
import argparse
import importlib.util
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

SPEC = importlib.util.spec_from_file_location('validate', Path(__file__).resolve().parents[1] / 'scripts/validate.py')
validate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate)


class ValidationTests(unittest.TestCase):
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
        shutil.copy(SPEC.origin, self.source / 'scripts/validate.py')
        shutil.copy(Path(SPEC.origin).with_name('mpi_launcher.py'), self.source / 'scripts/mpi_launcher.py')
        (self.source / 'scripts/run_theseus.sh').write_text('exit 0\n')
        self.prefix = self.root / 'prefix'
        self.prefix.mkdir()
        self.run = self.root / 'results'

    def prepare(self):
        return validate.prepare(argparse.Namespace(source=self.source, results=self.run,
            prefix=self.prefix, python=str(self.bin / 'python-stub'), mpiexec='mpiexec',
            mpi_arg=[], jobs=1, timeout=5))

    def steps(self, run, manifest):
        return [validate.Command('first', [sys.executable, '-c', 'print("first")'], run,
                                 'Inviscid Isentropic Vortex — cyclic run'),
                validate.Command('second', [sys.executable, '-c', 'print("second")'], run,
                                 'Inviscid Isentropic Vortex — cyclic comparison', ('first',))]

    def test_pass_requires_complete_unchanged_evidence(self):
        self.prepare()
        with patch.object(validate, 'suite_commands', self.steps):
            self.assertEqual(validate.worker(self.run), 0)
            self.assertEqual(validate.check(self.run, self.source), 0)
            (self.source / 'README.md').write_text('Documentation only')
            self.assertEqual(validate.check(self.run, self.source), 0)
            (self.source / 'new_test.py').write_text('assert False')
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate.check(self.run, self.source)
            (self.source / 'new_test.py').unlink()
            (self.source / 'code.cpp').unlink()
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate.check(self.run, self.source)
            (self.source / 'code.cpp').write_text('original\n')
            (self.run / 'logs/first.log').write_text('replaced')
            with self.assertRaisesRegex(ValueError, 'Log changed'):
                validate.check(self.run, self.source)

    def test_failure_blocks_dependents_but_independent_checks_finish(self):
        self.prepare()
        commands = [
            validate.Command('build', [sys.executable, '-c', 'raise SystemExit(7)'], self.run, 'Build'),
            validate.Command('run', [sys.executable, '-c', 'print("must not run")'],
                             self.run, 'Simulation', ('build',)),
            validate.Command('compare', [sys.executable, '-c', 'print("must not run")'],
                             self.run, 'Comparison', ('run',)),
            validate.Command('independent', [sys.executable, '-c', 'print("independent")'],
                             self.run, 'Independent check'),
        ]
        with patch.object(validate, 'suite_commands', return_value=commands):
            self.assertEqual(validate.worker(self.run), 1)
            results = json.loads((self.run / 'results.json').read_text())
            self.assertEqual([step['state'] for step in results['steps']],
                             ['failed', 'blocked', 'blocked', 'passed'])
            self.assertEqual(results['steps'][0]['returncode'], 7)
            self.assertNotIn('elapsed_seconds', results['steps'][1])
            self.assertFalse((self.run / 'logs/run.log').exists())
            with self.assertRaisesRegex(ValueError, 'not complete'):
                validate.check(self.run, self.source)

    def test_summary_keeps_separate_run_and_comparison_durations(self):
        self.prepare()
        with patch.object(validate, 'suite_commands', self.steps):
            self.assertEqual(validate.worker(self.run), 0)
        results = json.loads((self.run / 'results.json').read_text())
        summary = (self.run / 'summary.txt').read_text()
        for step in results['steps']:
            self.assertGreaterEqual(step['elapsed_seconds'], 0)
            self.assertIn(f"{step['elapsed_seconds']:.2f} s  {step['label']}", summary)
        self.assertIn('cyclic run', summary)
        self.assertIn('cyclic comparison', summary)

    def test_incomplete_missing_and_overlapping_runs(self):
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()
        with self.assertRaisesRegex(ValueError, 'not complete'):
            validate.check(self.run, self.source)
        (self.run / 'results.json').unlink()
        with self.assertRaises(FileNotFoundError):
            validate.check(self.run, self.source)

    def test_snapshot_is_independent_and_detects_mutation(self):
        self.prepare()
        manifest = json.loads((self.run / 'manifest.json').read_text())
        (self.source / 'code.cpp').write_text('edited during review')
        self.assertEqual((self.run / 'source/code.cpp').read_text(), 'original\n')
        validate.verify_snapshot(self.run, manifest)
        (self.run / 'source/code.cpp').write_text('bad')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            validate.verify_snapshot(self.run, manifest)

    def test_timeout_records_failure(self):
        self.prepare()
        results = {'state': 'running', 'steps': []}
        self.assertFalse(validate.execute_step(self.run, results, 'hang',
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
        command = [sys.executable, str(self.source / 'scripts/validate.py'), 'start',
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
        self.assertEqual(validate.check(self.run, self.source), 0)
        self.assertEqual(len(results['steps']), len(validate.suite_commands(self.run,
            json.loads((self.run / 'manifest.json').read_text()))))


if __name__ == '__main__':
    unittest.main()
