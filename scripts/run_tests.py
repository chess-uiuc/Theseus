#!/usr/bin/env python3
"""Run the complete local/CI test suite and verify its saved results."""
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
import tempfile
import time
from typing import NamedTuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9 and 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


class Command(NamedTuple):
    name: str
    argv: list[str]
    cwd: Path
    label: str
    requires: tuple[str, ...] = ()


RECIPE_MANIFEST = Path('tests/recipes/recipes.toml')


def load_recipes(source):
    source = Path(source).resolve()
    manifest_path = source / RECIPE_MANIFEST
    with manifest_path.open('rb') as stream:
        manifest = tomllib.load(stream)
    if manifest.get('schema') != 1:
        raise ValueError(f'Unsupported recipe manifest schema: {manifest.get("schema")}')
    recipes = []
    identifiers = set()
    registered_scripts = set()
    required = {'id', 'title', 'script', 'suites', 'build', 'requires'}
    for record in manifest.get('test', []):
        missing = required - record.keys()
        if missing:
            raise ValueError(f'Recipe is missing fields: {", ".join(sorted(missing))}')
        identifier = record['id']
        if identifier in identifiers:
            raise ValueError(f'Duplicate recipe id: {identifier}')
        identifiers.add(identifier)
        script = (source / record['script']).resolve()
        if source not in script.parents or not script.is_file():
            raise ValueError(f'Missing or invalid recipe script for {identifier}: {script}')
        if script in registered_scripts:
            raise ValueError(f'Recipe script is registered more than once: {script}')
        if not os.access(script, os.X_OK):
            raise ValueError(f'Recipe script is not executable: {script}')
        stages = record.get('stages', [])
        if len(stages) != len(set(stages)) or not set(stages) <= {'run', 'prepare', 'compare'}:
            raise ValueError(f'Invalid recipe stages for {identifier}: {stages}')
        registered_scripts.add(script)
        recipes.append({**record, 'script_path': script})
    discovered_scripts = set((source / 'tests/recipes').rglob('*.sh'))
    unregistered = sorted(discovered_scripts - registered_scripts)
    if unregistered:
        names = ', '.join(str(path.relative_to(source)) for path in unregistered)
        raise ValueError(f'Unregistered test recipe scripts: {names}')
    return sorted(recipes, key=lambda recipe: (recipe.get('order', 1000), recipe['id']))


def select_recipe(recipes, identifier):
    matches = [recipe for recipe in recipes if recipe['id'] == identifier]
    if not matches:
        raise ValueError(f'Unknown test recipe: {identifier}')
    return matches[0]


def print_recipe_list(recipes):
    print(f'{"ID":24} {"BUILD":12} {"SUITES":28} TITLE')
    for recipe in recipes:
        suites = ','.join(recipe['suites'])
        print(f'{recipe["id"]:24} {recipe["build"]:12} {suites:28} {recipe["title"]}')


def show_recipe(recipe, source):
    script = recipe['script_path']
    print(f'ID:       {recipe["id"]}')
    print(f'Title:    {recipe["title"]}')
    print(f'Build:    {recipe["build"]}')
    print(f'Suites:   {", ".join(recipe["suites"])}')
    print(f'Requires: {", ".join(recipe["requires"])}')
    print(f'Script:   {script.relative_to(Path(source).resolve())}')
    print('\n--- recipe ---')
    print(script.read_text(), end='')


def recipe_command(recipe, args, stage=None):
    build_option = '--build-' + recipe['build']
    build = getattr(args, 'build_' + recipe['build'])
    if build is None:
        raise ValueError(f'{recipe["id"]} requires {build_option}')
    command = [
        'bash', str(recipe['script_path']),
        '--source', str(args.source.resolve()),
        build_option, str(build.resolve()),
        '--prefix', str(args.prefix.resolve()),
        '--output', str(args.results.resolve()),
        '--device', args.device,
        '--python', str(Path(args.python).expanduser()),
        '--ranks', str(args.ranks),
    ]
    if stage is not None:
        command.extend(['--stage', stage])
    return command


def write_mpi_launcher(path, python, helper, mpiexec, mpi_args):
    path.write_text('#!/bin/bash\nexec ' + shlex.join(
        [str(python), str(helper), str(mpiexec), json.dumps(mpi_args)]) + ' "$@"\n')
    path.chmod(0o755)


def run_recipe(recipe, args):
    command = recipe_command(recipe, args)
    print('Recipe command: ' + shlex.join(command), flush=True)
    env = os.environ.copy()
    if not args.mpiexec:
        return subprocess.run(command, cwd=args.source.resolve(), env=env,
                              check=False).returncode
    mpiexec = shutil.which(args.mpiexec)
    if not mpiexec:
        raise ValueError(f'Executable not found: {args.mpiexec}')
    with tempfile.TemporaryDirectory(prefix='theseus-mpi-launcher-') as directory:
        launcher = Path(directory) / 'theseus-mpi-launcher'
        write_mpi_launcher(launcher, Path(args.python).expanduser(),
                           args.source.resolve() / 'scripts/mpi_launcher.py',
                           mpiexec, args.mpi_arg)
        env['THESEUS_MPI_LAUNCHER'] = str(launcher)
        return subprocess.run(command, cwd=args.source.resolve(), env=env,
                              check=False).returncode


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
    lines = [f"Test suite: {results['state'].upper()}", f"Run: {run}",
             f"Suite: {results.get('suite', 'full')}; device: {results.get('device', 'cpu')}", '']
    for step in results['steps']:
        elapsed = f"{step['elapsed_seconds']:.2f} s" if 'elapsed_seconds' in step else '—'
        label = step.get('label', step['name'])
        detail = step.get('reason') or step.get('log', '')
        lines.append(f"{step['state'].upper():8} {elapsed:>10}  {label} — {detail}")
    if results.get('error'):
        lines.extend(['', results['error']])
    if results.get('suite', 'full') == 'full':
        lines.extend(['', 'Commit eligibility: run scripts/run_tests.py check against the current source.'])
    else:
        lines.extend(['', 'Integrated-only results for the recorded binaries; not a full-suite commit gate.'])
    temporary = run / 'summary.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(run / 'summary.txt')


def integrated_commands(run, manifest, builds):
    source = run / 'source'
    device = manifest.get('device', 'cpu')
    entries = [
        ('cfl', 'Timestep and CFL', 'standard', 'timestep_cfl_integration_test.py',
         ['--database', manifest['prefix'] + '/database', '--reference', str(builds['standard'] / 'tests/table_lookup_tests')]),
        ('restart', 'Cartesian Checkpoint Restart', 'standard', 'restart_integration_test.py', []),
        ('initial-state', 'Physical Initial States', 'standard', 'physical_initial_state_integration_test.py', ['--database', manifest['prefix'] + '/database']),
        ('visualization', 'Derived Visualization', 'standard', 'derived_visualization_integration_test.py',
         ['--database', manifest['prefix'] + '/database', '--reference', str(builds['standard'] / 'tests/table_lookup_tests')]),
        ('axis-uniform', 'Axisymmetric Uniform Flow', 'axisymmetric', 'axisymmetric_uniform_flow_integration_test.py',
         ['--results', str(run / 'integrated/axis-uniform/cases')]),
        ('axis-convergence', 'Axisymmetric Entropy-Wave Convergence', 'axisymmetric', 'axisymmetric_entropy_wave_convergence_test.py', []),
        ('axis-inviscid-sphere', 'Axisymmetric Inviscid Sphere', 'axisymmetric', 'axisymmetric_flow_over_sphere_integration_test.py', ['--case', 'inviscid']),
        ('axis-viscous-sphere', 'Axisymmetric Viscous Sphere', 'axisymmetric', 'axisymmetric_flow_over_sphere_integration_test.py', ['--case', 'viscous']),
        ('axis-restart', 'Axisymmetric Checkpoint Restart', 'axisymmetric', 'restart_integration_test.py', ['--axisymmetric']),
        ('px-chamber', 'CPG PX Chamber', 'axisymmetric', 'px_chamber_integration_test.py', []),
        ('boundary-state', 'Physical Boundary States', 'axisymmetric', 'physical_boundary_integration_test.py', ['--database', manifest['prefix'] + '/database']),
        ('lte-px-chamber', 'LTE PX Chamber', 'axisymmetric', 'lte_px_chamber_integration_test.py', ['--database', manifest['prefix'] + '/database']),
    ]
    commands = []
    for name, title, geometry, script, options in entries:
        requirements = ('python-dependencies',)
        if manifest.get('suite', 'full') == 'full':
            requirements += ('build-' + geometry,)
        command = [manifest['python'], str(source / 'tests' / script),
                   '--source', str(source), '--executable', str(builds[geometry] / 'theseus'),
                   '--device', device, *options]
        commands.append(Command('integration-' + name, command, run / 'integrated' / name,
                                title + ' — integration', requirements))
    return commands


def suite_recipe_commands(run, manifest, builds):
    source = run / 'source'
    suite = manifest.get('suite', 'full')
    commands = []
    for recipe in load_recipes(source):
        if suite not in recipe['suites']:
            continue
        arguments = argparse.Namespace(
            source=source,
            build_standard=builds['standard'],
            build_axisymmetric=builds['axisymmetric'],
            prefix=Path(manifest['prefix']),
            results=run / 'recipes' / recipe['id'],
            device=manifest.get('device', 'cpu'),
            python=manifest['python'],
            ranks=2,
        )
        requirements = []
        if suite == 'full':
            requirements.append('build-' + recipe['build'])
        if {'python', 'pyvista'} & set(recipe['requires']):
            requirements.append('python-dependencies')
        previous = None
        stages = recipe.get('stages', [])
        if not stages:
            commands.append(Command(
                'recipe-' + recipe['id'],
                recipe_command(recipe, arguments),
                source,
                recipe['title'] + ' — recipe',
                tuple(requirements),
            ))
            continue
        stage_labels = {
            'run': 'Theseus run',
            'prepare': 'reference preparation',
            'compare': 'comparison',
        }
        for stage in stages:
            stage_requirements = list(requirements)
            if previous is not None:
                stage_requirements.append(previous)
            name = 'recipe-' + recipe['id'] + '-' + stage
            commands.append(Command(
                name,
                recipe_command(recipe, arguments, stage),
                source,
                recipe['title'] + ' — ' + stage_labels.get(stage, stage),
                tuple(stage_requirements),
            ))
            previous = name
    return commands


def suite_commands(run, manifest):
    """The full test suite shared by local runs and both CI workflows."""
    source = run / 'source'
    python = manifest['python']
    prefix = manifest['prefix']
    full = manifest.get('suite', 'full') == 'full'
    builds = {name: Path(manifest.get('builds', {}).get(name, run / ('build-' + name)))
              for name in ('standard', 'axisymmetric')}
    commands = [Command('python-dependencies', [python, '-c',
                'import sys, pyvista; print(sys.version); print("pyvista", pyvista.__version__)'],
                run, 'Python dependencies')]
    if full:
        for name, command in [('cmake-version', ['cmake', '--version']),
                              ('compiler-c', [manifest['cc'], '--version']),
                              ('compiler-cxx', [manifest['cxx'], '--version'])]:
            commands.append(Command(name, command, run, name.replace('-', ' ').capitalize()))
        if manifest.get('mpiexec'):
            commands.append(Command('mpi-version', [manifest['mpiexec'], '--version'], run, 'MPI version'))
        for axis in (False, True):
            label = 'axisymmetric' if axis else 'standard'
            build = builds[label]
            configure = ['cmake', '-S', str(source), '-B', str(build),
                         '-DCMAKE_BUILD_TYPE=Debug', '-DBUILD_TESTING=ON',
                         f'-DCMAKE_PREFIX_PATH={prefix}', f'-DCMAKE_BUILD_RPATH={prefix}/lib',
                         f'-DCMAKE_INSTALL_RPATH={prefix}/lib', '-DTHESEUS_WITH_PLATO=ON',
                         '-DSUBCELL_FV_BLENDING=ON', '-DNO_OPT=ON',
                         f'-DAXISYMMETRIC={"ON" if axis else "OFF"}',
                         f'-DPython3_EXECUTABLE={python}',
                         ]
            if manifest.get('mpiexec'):
                configure.append(f'-DMPIEXEC_EXECUTABLE={run / "bin" / "theseus-mpi-launcher"}')
            commands.extend([
                Command('configure-' + label, configure, run, label.capitalize() + ' configuration',
                        ('cmake-version', 'compiler-c', 'compiler-cxx')),
                Command('build-' + label, ['cmake', '--build', str(build), '-j', str(manifest['jobs'])],
                        run, label.capitalize() + ' build', ('configure-' + label,)),
                Command('ctest-' + label, ['ctest', '--test-dir', str(build), '--output-on-failure',
                                         '--no-tests=error'], run, label.capitalize() + ' CTest suite',
                        ('build-' + label, 'python-dependencies')),
            ])
    commands.extend(integrated_commands(run, manifest, builds))
    commands.extend(suite_recipe_commands(run, manifest, builds))
    return commands


def prepared_builds(args):
    builds = {}
    evidence = {}
    for geometry, directory in [('standard', args.build_standard),
                                ('axisymmetric', args.build_axisymmetric)]:
        if directory is None:
            raise ValueError('Integrated mode requires both prepared build directories.')
        directory = directory.resolve()
        cache = directory / 'CMakeCache.txt'
        values = {}
        for line in cache.read_text().splitlines():
            if '=' in line and not line.startswith(('#', '//')):
                key, value = line.split('=', 1)
                values[key.split(':', 1)[0]] = value
        enabled = {'1', 'ON', 'YES', 'TRUE'}
        axis = values.get('AXISYMMETRIC', 'OFF').upper() in enabled
        if axis != (geometry == 'axisymmetric'):
            raise ValueError(f'Wrong geometry for {geometry}: {directory}')
        if values.get('THESEUS_WITH_PLATO', '').upper() not in enabled:
            raise ValueError(f'PLATO is required: {directory}')
        if args.device == 'cuda' and values.get('ENABLE_CUDA', '').upper() not in enabled:
            raise ValueError(f'CUDA build required: {directory}')
        files = [cache, directory / 'theseus']
        if geometry == 'standard':
            files.append(directory / 'tests/table_lookup_tests')
        for path in files:
            if path != cache and not os.access(path, os.X_OK):
                raise ValueError(f'Missing executable: {path}')
            evidence[str(path)] = digest(path)
        builds[geometry] = str(directory)
    return builds, evidence


def verify_prepared_builds(manifest):
    for name, expected in manifest.get('build_files', {}).items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Prepared build changed: {name}')


def prepare(args):
    source = args.source.resolve()
    run = args.results.resolve()
    if run == source or source in run.parents:
        raise ValueError('Results must be outside the source checkout.')
    prefix = args.prefix.resolve()
    if not prefix.is_dir():
        raise ValueError(f'Missing dependency prefix: {prefix}')
    suite = getattr(args, 'suite', 'full')
    builds, build_files = prepared_builds(args) if suite == 'integrated' else ({}, {})
    files = source_files(source)
    # Exclusive creation prevents overlapping runs from overwriting evidence.
    run.mkdir(parents=True, exist_ok=False)
    (run / 'logs').mkdir()
    results = {'state': 'preparing', 'steps': [], 'started': time.time(),
               'suite': suite, 'device': getattr(args, 'device', 'cpu')}
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
                    'cc': executable(os.environ.get('CC', 'mpicc')) if suite == 'full' else None,
                    'cxx': executable(os.environ.get('CXX', 'mpicxx')) if suite == 'full' else None,
                    'mpiexec': executable(args.mpiexec) if args.mpiexec else None, 'mpi_args': args.mpi_arg,
                    'jobs': args.jobs, 'timeout': args.timeout,
                    'suite': suite, 'device': getattr(args, 'device', 'cpu'),
                    'builds': builds, 'build_files': build_files}
        write_json(run / 'manifest.json', manifest)
        (run / 'bin').mkdir()
        if manifest['mpiexec']:
            launcher = run / 'bin/theseus-mpi-launcher'
            write_mpi_launcher(launcher, manifest['python'],
                               snapshot / 'scripts/mpi_launcher.py',
                               manifest['mpiexec'], manifest['mpi_args'])
        (snapshot / 'tpl').mkdir(exist_ok=True)
        (snapshot / 'tpl/install').symlink_to(prefix, target_is_directory=True)
        for command in suite_commands(run, manifest):
            command.cwd.mkdir(parents=True, exist_ok=True)
        results['suite'] = suite
        results['device'] = manifest['device']
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
        raise InterruptedError(f"Test suite interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    env = os.environ.copy()
    env.update(PATH=str(run / 'bin') + os.pathsep + env['PATH'],
               PYTHON=manifest['python'],
               PYVISTA_OFF_SCREEN='true', QT_QPA_PLATFORM='offscreen')
    if manifest.get('mpiexec'):
        env['THESEUS_MPI_LAUNCHER'] = str(run / 'bin/theseus-mpi-launcher')
    if manifest.get('cc'):
        env.update(CC=manifest['cc'], CXX=manifest['cxx'])
    try:
        verify_prepared_builds(manifest)
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
        verify_prepared_builds(manifest)
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
        raise ValueError(f"Test suite is not complete and passing: {results['state']}")
    if manifest.get('suite', 'full') != 'full':
        raise ValueError('Integrated-only results are not a full-suite commit gate.')
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
    parser.add_argument('action', choices=['run', 'start', 'check', 'list', 'show',
                                          'run-test', '_worker'])
    parser.add_argument('test', nargs='?', help='recipe id for show or run-test')
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--prefix', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--mpiexec', help='Exceptional MPI executable override; normal users omit this so run_theseus selects the platform launcher')
    parser.add_argument('--suite', choices=['full', 'integrated'], default='full')
    parser.add_argument('--build-standard', type=Path)
    parser.add_argument('--build-axisymmetric', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--ranks', type=int, default=2,
                        help='MPI ranks for direct recipe execution (default: 2)')
    parser.add_argument('--mpi-arg', action='append', default=[])
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=3600,
                        help='Maximum seconds per command, including its child processes')
    args = parser.parse_args()
    try:
        if args.action in {'list', 'show', 'run-test'}:
            recipes = load_recipes(args.source)
            if args.action == 'list':
                if args.test is not None:
                    parser.error('list does not accept a recipe id')
                print_recipe_list(recipes)
                return 0
            if args.test is None:
                parser.error(f'{args.action} requires a recipe id')
            recipe = select_recipe(recipes, args.test)
            if args.action == 'show':
                show_recipe(recipe, args.source)
                return 0
            if args.results is None or args.prefix is None or args.ranks < 1:
                parser.error('run-test requires --results, --prefix and positive --ranks')
            if args.mpi_arg and not args.mpiexec:
                parser.error('--mpi-arg requires the exceptional --mpiexec override')
            return run_recipe(recipe, args)
        if args.results is None:
            parser.error(f'{args.action} requires --results')
        if args.action == 'check':
            return check(args.results.resolve(), args.source.resolve())
        if args.action == '_worker':
            return worker(args.results.resolve())
        if args.prefix is None or args.jobs < 1 or args.timeout <= 0:
            parser.error('run/start require --prefix, positive --jobs and positive --timeout')
        if args.suite == 'full' and (args.device != 'cpu' or args.build_standard or args.build_axisymmetric):
            parser.error('Prepared builds and GPU selection require --suite integrated.')
        if args.mpi_arg and not args.mpiexec:
            args.mpiexec = 'mpiexec'
        run = prepare(args)
        if args.action == 'run':
            return worker(run)
        # Execute the frozen runner, independently of this invocation and terminal.
        with (run / 'launcher.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(run / 'source/scripts/run_tests.py'),
                                        '_worker', '--results', str(run)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        print(f'Started test suite PID {process.pid}. Status: {run / "summary.txt"}')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'Test runner error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
