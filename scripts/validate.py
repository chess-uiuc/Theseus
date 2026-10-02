#!/usr/bin/env python3
"""Run the complete local/CI validation suite and verify its saved results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]


class Command(NamedTuple):
    name: str
    argv: list[str]
    cwd: Path
    label: str
    requires: tuple[str, ...] = ()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(source):
    names = git(source, 'ls-files', '-z', '--cached', '--others', '--exclude-standard')
    files = {}
    for name in sorted(set(os.fsdecode(n) for n in names.split(b'\0') if n)):
        path = source / name
        if path.is_symlink():
            raise ValueError(f'Source symlinks are not supported in snapshots: {name}')
        if not path.exists():  # A tracked deletion is represented by its absence.
            continue
        if not path.is_file():
            raise ValueError(f'Expected a source file, not a directory/submodule: {name}')
        files[name] = {'sha256': digest(path), 'executable': bool(path.stat().st_mode & 0o111)}
    return files


def documentation(name):
    path = Path(name)
    return (path.suffix in {'.md', '.dox'} and
            (len(path.parts) == 1 or path.parts[0] == 'docs'))


def changed_files(before, after):
    return sorted(name for name in before.keys() | after.keys()
                  if before.get(name) != after.get(name))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_results(run, results):
    write_json(run / 'results.json', results)
    lines = [f"Validation: {results['state'].upper()}", f"Run: {run}", '']
    for step in results['steps']:
        elapsed = f"{step['elapsed_seconds']:.2f} s" if 'elapsed_seconds' in step else '—'
        label = step.get('label', step['name'])
        detail = step.get('reason') or step.get('log', '')
        lines.append(f"{step['state'].upper():8} {elapsed:>10}  {label} — {detail}")
    if results.get('error'):
        lines.extend(['', results['error']])
    lines.extend(['', 'Commit eligibility: run validate.py check against the current source.'])
    temporary = run / 'summary.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(run / 'summary.txt')


def suite_commands(run, manifest):
    """The full suite, shared by local validation and both CI workflows."""
    source = run / 'source'
    python = manifest['python']
    prefix = manifest['prefix']
    commands = []
    for name, command in [('cmake-version', ['cmake', '--version']),
                          ('compiler-c', [manifest['cc'], '--version']),
                          ('compiler-cxx', [manifest['cxx'], '--version']),
                          ('mpi-version', [manifest['mpiexec'], '--version']),
                          ('python-dependencies', [python, '-c',
                           'import sys, pyvista; print(sys.version); print("pyvista", pyvista.__version__)'])]:
        commands.append(Command(name, command, run, name.replace("-", " ").capitalize()))
    for axis in (False, True):
        label = 'axisymmetric' if axis else 'standard'
        build = run / ('build-' + label)
        configure = ['cmake', '-S', str(source), '-B', str(build),
                     '-DCMAKE_BUILD_TYPE=Debug', '-DBUILD_TESTING=ON',
                     f'-DCMAKE_PREFIX_PATH={prefix}', f'-DCMAKE_BUILD_RPATH={prefix}/lib',
                     f'-DCMAKE_INSTALL_RPATH={prefix}/lib', '-DTHESEUS_WITH_PLATO=ON',
                     '-DSUBCELL_FV_BLENDING=ON', '-DNO_OPT=ON',
                     f'-DAXISYMMETRIC={"ON" if axis else "OFF"}',
                     f'-DPython3_EXECUTABLE={python}',
                     f'-DMPIEXEC_EXECUTABLE={run / "bin" / "mpiexec"}']
        commands.extend([
            Command('configure-' + label, configure, run, label.capitalize() + ' configuration',
                    ('cmake-version', 'compiler-c', 'compiler-cxx', 'mpi-version')),
            Command('build-' + label, ['cmake', '--build', str(build), '-j', str(manifest['jobs'])],
                    run, label.capitalize() + ' build', ('configure-' + label,)),
            Command('ctest-' + label, ['ctest', '--test-dir', str(build), '--output-on-failure',
                                     '--no-tests=error'], run, label.capitalize() + ' CTest suite',
                    ('build-' + label, 'python-dependencies')),
        ])
    cases = {
        'vortex': ('Euler/2D/IsentropicVortex', 'IsentropicVortex', 'Inviscid Isentropic Vortex'),
        'cavity': ('NavierStokes/2D/LidDrivenCavity', 'LidDrivenCavity', 'Viscous Lid-Driven Cavity'),
        'lte': ('LTE/Euler/LTEVortex', 'LTEVortex', 'Inviscid LTE Vortex'),
        'tgv': ('NavierStokes/2D/TaylorGreenVortex', 'TaylorGreenVortex2D', 'Viscous Taylor–Green Vortex'),
        'step': ('Euler/2D/ForwardFacingStep', 'ForwardFacingStep', 'Inviscid Forward-Facing Step'),
    }

    def simulation(label, case, dt=None, steps=None):
        directory, _, title = cases[case]
        command = ['bash', str(source / 'scripts/run_theseus.sh'),
                   '-b', str(run / 'build-standard'), '-c', f'TestCases/{directory}/config.json',
                   '-o', f'../cases/{label}']
        if dt is not None:
            command += ['-t', dt, '-n', str(steps)]
        kind = 'cyclic' if label == 'cyclic' else ('smoke' if label.startswith('smoke-') else 'gold-standard')
        commands.append(Command(label, command, source, f'{title} — {kind} run',
                                ('build-standard',)))
        return run / 'cases' / label / Path(directory).name / 'ParaView'

    for case, dt in [('vortex', '0.002'), ('cavity', '0.0001'), ('lte', '1e-5')]:
        simulation('smoke-' + case, case, dt, 100)
    output = simulation('cyclic', 'vortex')
    compare = [python, str(source / 'scripts/compare_viz.py')]
    commands.append(Command('compare-cyclic', compare + [str(output / 'ParaView.pvd'),
                     '--fields', 'Density', '--atol', '5e-5', '--rtol', '2e-6'], run,
                     cases['vortex'][2] + ' — cyclic comparison', ('cyclic', 'python-dependencies')))
    for case, dt, steps, atol, rtol in [
        ('vortex', '0.001', 500, '1e-13', '1e-13'),
        ('lte', '1e-5', 200, '1e-10', '1e-10'),
        ('cavity', '0.0001', 4000, '1e-12', '1e-12'),
        ('tgv', '0.0001', 100, '1e-13', '1e-13'),
        ('step', '0.0001', 100, '3e-13', '1e-13'),
    ]:
        output = simulation('golden-' + case, case, dt, steps)
        cycle = f'Cycle{steps:06d}/data.pvtu'
        # reference = source / 'TestCases/GoldenData' / cases[case][1] / cycle
        # Automatically unpack golden data if it is zipped up 
        reference_name = cases[case][1]
        reference = run / 'references' / reference_name / cycle
        preparation = 'prepare-reference-' + case
        commands.append(Command(preparation,
                        [python, str(source / 'scripts/prepare_golden_reference.py'),
                         str(source / 'TestCases/GoldenData'), reference_name,
                         str(run / 'references' / reference_name), cycle], run,
                        cases[case][2] + ' — prepare gold-standard reference'))
        prerequisites = ('golden-' + case, 'python-dependencies', preparation)
        commands.append(Command('compare-' + case, compare + [str(output / cycle), str(reference),
                         '--atol', atol, '--rtol', rtol], run,
                         cases[case][2] + ' — gold-standard comparison',
                         prerequisites))
    return commands


def prepare(args):
    source = args.source.resolve()
    run = args.results.resolve()
    if run == source or source in run.parents:
        raise ValueError('Results must be outside the source checkout.')
    prefix = args.prefix.resolve()
    if not prefix.is_dir():
        raise ValueError(f'Missing dependency prefix: {prefix}')
    files = source_files(source)
    # Exclusive creation prevents overlapping runs from overwriting evidence.
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    results = {'state': 'preparing', 'steps': [], 'started': time.time()}
    save_results(run, results)
    try:
        snapshot = run / 'source'
        snapshot.mkdir()
        for name in files:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
            if digest(target) != files[name]['sha256']:
                raise ValueError(f'Source changed while copying: {name}')
        if source_files(source) != files:
            raise ValueError('Source changed while creating snapshot; start a fresh run.')
        def executable(value):
            found = shutil.which(value)
            if not found:
                raise ValueError(f'Executable not found: {value}')
            return str(Path(found).absolute())
        manifest = {'schema': 1, 'source': str(source), 'files': files,
                    'head': git(source, 'rev-parse', 'HEAD').decode().strip(),
                    'prefix': str(prefix), 'python': executable(args.python),
                    'cc': executable(os.environ.get('CC', 'mpicc')),
                    'cxx': executable(os.environ.get('CXX', 'mpicxx')),
                    'mpiexec': executable(args.mpiexec), 'mpi_args': args.mpi_arg,
                    'jobs': args.jobs, 'timeout': args.timeout}
        write_json(run / 'manifest.json', manifest)
        (run / 'bin').mkdir()
        launcher = run / 'bin/mpiexec'
        launcher.write_text('#!/bin/bash\nexec ' + shlex.join(
            [sys.executable, str(snapshot / 'scripts/mpi_launcher.py'),
             manifest['mpiexec'], json.dumps(manifest['mpi_args'])]) + ' "$@"\n')
        launcher.chmod(0o755)
        (snapshot / 'tpl').mkdir(exist_ok=True)
        (snapshot / 'tpl/install').symlink_to(prefix, target_is_directory=True)
        # Existing example configs resolve ../../TestCases and ../../tpl/install.
        (run / 'cases/tpl').mkdir(parents=True)
        (run / 'cases/TestCases').symlink_to(snapshot / 'TestCases', target_is_directory=True)
        (run / 'cases/tpl/install').symlink_to(prefix, target_is_directory=True)
        results['state'] = 'queued'
        save_results(run, results)
    except BaseException as error:
        results.update(state='failed', error=str(error), finished=time.time())
        save_results(run, results)
        raise
    return run


def execute_step(run, results, name, command, cwd, env, timeout, label=None):
    started = time.monotonic()
    log = run / 'logs' / (name + '.log')
    step = {'name': name, 'command': command, 'cwd': str(cwd),
            'log': str(log.relative_to(run)), 'label': label or name,
            'state': 'running', 'started': time.time()}
    results['steps'].append(step)
    save_results(run, results)
    try:
        with log.open('w') as stream:
            stream.write(f'Working directory: {cwd}\nCommand: {shlex.join(command)}\n\n')
            stream.flush()
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        step.update(returncode=code, state='passed' if code == 0 else 'failed')
    except (KeyboardInterrupt, InterruptedError) as error:
        step.update(state='interrupted', error=str(error))
        raise
    except Exception as error:
        step.update(state='failed', error=str(error))
    finally:
        step['finished'] = time.time()
        step['elapsed_seconds'] = time.monotonic() - started
        if log.exists():
            step['sha256'] = digest(log)
        save_results(run, results)
    return step['state'] == 'passed'


def verify_snapshot(run, manifest):
    for name, expected in manifest["files"].items():
        path = run / "source" / name
        if (not path.is_file() or path.is_symlink() or digest(path) != expected["sha256"]
                or bool(path.stat().st_mode & 0o111) != expected["executable"]):
            raise ValueError(f"Snapshot changed: {name}")


def worker(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    if results['state'] != 'queued':
        raise ValueError('A run may execute only once; use a new results directory.')
    results.update(state='running', pid=os.getpid())
    save_results(run, results)
    def interrupted(signum, frame):
        raise InterruptedError(f"Validation interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    env = os.environ.copy()
    env.update(PATH=str(run / 'bin') + os.pathsep + env['PATH'],
               CC=manifest['cc'], CXX=manifest['cxx'], PYTHON=manifest['python'],
               PYVISTA_OFF_SCREEN='true', QT_QPA_PLATFORM='offscreen')
    try:
        outcomes = {}
        for command in suite_commands(run, manifest):
            blockers = [name for name in command.requires if outcomes.get(name) != 'passed']
            if blockers:
                step = {'name': command.name, 'label': command.label, 'command': command.argv,
                        'cwd': str(command.cwd), 'state': 'blocked',
                        'reason': 'Requires passing: ' + ', '.join(blockers)}
                results['steps'].append(step)
                save_results(run, results)
            else:
                execute_step(run, results, command.name, command.argv, command.cwd,
                             env, manifest['timeout'], command.label)
            outcomes[command.name] = results['steps'][-1]['state']
        verify_snapshot(run, manifest)
        failed = [step for step in results['steps'] if step['state'] != 'passed']
        results['state'] = 'failed' if failed else 'passed'
        if failed:
            results['error'] = (f"{len(failed)} checks failed or blocked. "
                                f"Investigate first: {failed[0]['label']}")
    except BaseException as error:
        results.update(state='failed', error=str(error))
    results['finished'] = time.time()
    save_results(run, results)
    return 0 if results['state'] == 'passed' else 1


def check(run, source):
    manifest = json.loads((run / 'manifest.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    if manifest['schema'] != 1 or results['state'] != 'passed':
        raise ValueError(f"Validation is not complete and passing: {results['state']}")
    expected = suite_commands(run, manifest)
    steps = results['steps']
    if len(steps) != len(expected):
        raise ValueError('Incomplete suite results.')
    for step, (name, command, cwd, label, requires) in zip(steps, expected):
        if (step['name'] != name or step['command'] != command or step['cwd'] != str(cwd)
                or step['state'] != 'passed' or step.get('returncode') != 0):
            raise ValueError(f'Missing or failing result: {name}')
        if digest(run / step['log']) != step['sha256']:
            raise ValueError(f'Log changed since completion: {name}')
    verify_snapshot(run, manifest)
    differences = changed_files(manifest['files'], source_files(source))
    material = [name for name in differences if not documentation(name)]
    if material:
        raise ValueError('Results are stale; source differs: ' + ', '.join(material))
    print('PASS: full suite completed; current implementation and tests match the snapshot.')
    if differences:
        print('Documentation changes require review: ' + ', '.join(differences))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'start', 'check', '_worker'])
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--mpiexec', default='mpiexec')
    parser.add_argument('--mpi-arg', action='append', default=[])
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=3600,
                        help='Maximum seconds per command, including its child processes')
    args = parser.parse_args()
    try:
        if args.action == 'check':
            return check(args.results.resolve(), args.source.resolve())
        if args.action == '_worker':
            return worker(args.results.resolve())
        if args.prefix is None or args.jobs < 1 or args.timeout <= 0:
            parser.error('run/start require --prefix, positive --jobs and positive --timeout')
        run = prepare(args)
        if args.action == 'run':
            return worker(run)
        # Execute the frozen runner, independently of this invocation and terminal.
        with (run / 'launcher.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(run / 'source/scripts/validate.py'),
                                        '_worker', '--results', str(run)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        print(f'Started validation PID {process.pid}. Status: {run / "summary.txt"}')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Validation error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
