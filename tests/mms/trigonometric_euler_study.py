#!/usr/bin/env python3
"""Prepare, assess, or plot a saved multidirectional Euler MMS study."""

from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from numflux_regression_test import cell_local_fields
from mms_error_norms import tensor_volume_errors


FINAL_TIME = .2
FIELDS = ('Density', 'Velocity', 'Pressure', 'Specific Total Energy')


def exact_fields(coordinates, time):
    dimension = coordinates.shape[1]
    # Independent NumPy specification; does not import a production evaluator.
    means = np.array([2., .7, .3, .2, 1.])
    ax = np.array([.12, .07, -.05, .04, .08])
    ay = np.array([.08, -.04, .06, -.03, -.06])
    az = np.array([-.06, .03, -.04, .05, .04])
    axy = np.array([.04, .02, .03, -.02, .03])
    axz = np.array([.03, -.02, .01, .03, -.02])
    ayz = np.array([-.02, .01, -.02, .02, .02])
    index = np.arange(5)
    px = 2.*np.pi*(coordinates[:, 0, None] - .2*time) + .17*index
    py = 2.*np.pi*(coordinates[:, 1, None] + .13*time) + .29*index
    primitive = means + ax*np.sin(px) + ay*np.cos(py) + axy*np.sin(px)*np.cos(py)
    if dimension == 3:
        pz = 2.*np.pi*(coordinates[:, 2, None] - .11*time) + .31*index
        primitive += az*np.sin(pz) + axz*np.cos(px)*np.sin(pz) + ayz*np.sin(py)*np.cos(pz)
    density = primitive[:, 0]
    velocity = primitive[:, 1:dimension + 1]
    energy = primitive[:, 4]/.4 + .5*density*np.sum(velocity**2, axis=1)
    return np.column_stack((density, density[:, None]*velocity, energy))


def prepare(source, results, device, executable, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh_script = source/'TestCases/NavierStokes/3D/TaylorGreenVortex/make_periodic_box.py'
    spec = importlib.util.spec_from_file_location('periodic_box', mesh_script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    mesh3 = results/'periodic-3d.msh'
    generator.write_mesh(mesh3, 1., 1., 1., 6, 6, 6)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    records = []
    for dimension in (2, 3):
        levels = (1, 2, 3) if dimension == 2 else (0, 1, 2)
        variants = [(f'level-{level}', level, .004, True, ranks) for level in levels]
        variants.extend([('half-dt', levels[-1], .002, True, ranks),
                         ('source-off', levels[-1], .004, False, ranks)])
        if device == 'cuda':
            variants.append(('cpu-baseline', levels[-1], .004, True, ranks))
        for name, level, dt, forced, ranks in variants:
            directory = results/f'{dimension}d-{name}'
            directory.mkdir()
            config = copy.deepcopy(base)
            runtime = config['runTime']
            mesh = source/'TestCases/LTE/Euler/LTEVortex/LTEVortex.mesh' if dimension == 2 else mesh3
            runtime.update(dim=dimension, num_equations=dimension + 2, ser_ref_levels=level,
                           order=2, mesh_file=str(mesh), output_file_path=str(directory/'solution'),
                           final_time=FINAL_TIME, dt=dt, nsteps_max=round(FINAL_TIME/dt) + 2,
                           initial_save_dt=FINAL_TIME, refined_save_dt=FINAL_TIME,
                           sources=[{'type': 'trigonometric-euler-mms'}] if forced else [])
            runtime['conditions']['initial_conditions']['function'] = 'TrigonometricEulerMMS'
            runtime['exact_solution']['function'] = 'TrigonometricEulerMMS'
            (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
            cells = (3 if dimension == 2 else 6)*2**level
            length = 2. if dimension == 2 else 1.
            records.append(dict(name=directory.name, dimension=dimension, level=level, dt=dt,
                                forced=forced, ranks=ranks, h=length/cells, order=2,
                                backend='cpu' if name == 'cpu-baseline' else device))
    (results/'cases.json').write_text(json.dumps(records, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\t{c['backend']}\n" for c in records))
    if (source/'.git').exists():
        tracked = subprocess.check_output(
            ['git', '-C', str(source), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
        names = sorted(set(tracked.decode().split('\0')) - {''})
        base_sha = subprocess.check_output(
            ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    else:
        # The full runner deliberately tests a frozen source tree without .git.
        names = sorted(str(path.relative_to(source)) for path in source.rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts)
        parent_manifest = source.parent/'manifest.json'
        base_sha = None
        if source.name == 'source' and parent_manifest.is_file():
            base_sha = json.loads(parent_manifest.read_text()).get('head')
    hashes = {}
    for name in names:
        path = source/name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(executable=str(executable),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        cmake_cache_sha256=hashlib.sha256((executable.parent/'CMakeCache.txt').read_bytes()).hexdigest(),
        device=device, base_sha=base_sha,
        source=str(source), source_sha256=hashes, final_time=FINAL_TIME,
        gas_model='cpg', gamma=1.4, R_gas=1., flow_model='euler',
        numerical_flux='lfr', ode_solver='RK4', alpha_max=0.)
    (results/'CMakeCache.txt').write_bytes((executable.parent/'CMakeCache.txt').read_bytes())
    (results/'source.json').write_text(json.dumps(metadata, indent=2) + '\n')


def positive_finite(values, description):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or not np.all(values > 0.):
        raise RuntimeError(f'{description} must be finite and positive: {values}')


def refinement_orders(errors):
    if len(errors) < 3:
        raise RuntimeError('at least three refinement levels are required')
    positive_finite(errors, 'refinement errors')
    values = np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    if np.any(values <= 0.):
        raise RuntimeError(f'errors did not decrease: {errors}')
    return values


def matches_final_time(actual, expected, steps, dt):
    """Allow accumulated addition roundoff, never a significant fraction of a step."""
    if (not all(math.isfinite(value) for value in (actual, expected, dt))
            or expected < 0. or dt <= 0. or steps < 1):
        return False
    # Standard gamma_n bound for n floating-point additions of a fixed positive dt.
    scaled_epsilon = steps*np.finfo(float).eps
    if scaled_epsilon >= 1.:
        return False
    gamma = scaled_epsilon/(1. - scaled_epsilon)
    accumulation_bound = gamma*abs(steps*dt)
    tolerance = min(max(1e-13, accumulation_bound), .01*dt)
    return math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.)


def read_case(results, case, device, exact_reference=exact_fields,
              quadrature_points=10, check_solver_norm=True, final_time=FINAL_TIME):
    directory = results/case['name']
    log = (directory/'run.log').read_text()
    if f'Device configuration: {device}' not in log:
        raise RuntimeError(f'{directory}: requested backend {device} not reported')
    matches = re.findall(r'Exact L2 Error:\s*(\S+)', log)
    if not matches:
        raise RuntimeError(f'{directory}: missing final error')
    l2 = float(matches[-1])
    positive_finite(l2, 'physical aggregate L2')
    times = re.findall(r'time step:\s*(\d+), time:\s*([^,\s]+)', log)
    steps = int(times[-1][0]) if times else 0
    if (not times or ('steps' in case and steps != case['steps'])
            or not matches_final_time(float(times[-1][1]), final_time, steps, case['dt'])):
        raise RuntimeError(f'{directory}: incorrect final solver time')
    pvd = directory/'solution/ParaView/ParaView.pvd'
    frames = ET.parse(pvd).findall('.//DataSet')
    if len(frames) < 2 or float(frames[0].attrib['timestep']) != 0.:
        raise RuntimeError(f'{directory}: missing initial/final output')
    if not matches_final_time(float(frames[-1].attrib['timestep']), final_time, steps, case['dt']):
        raise RuntimeError(f'{directory}: incorrect final output time')
    data = []
    for frame in (frames[0], frames[-1]):
        fields = cell_local_fields(pvd.parent/frame.attrib['file'], FIELDS)
        for values in (fields['Density'], fields['Pressure']):
            positive_finite(values, 'density/pressure')
        rho = fields['Density'].reshape(-1)
        state = np.column_stack((rho, rho[:, None]*fields['Velocity'][:, :case['dimension']],
                                 rho*fields['Specific Total Energy'].reshape(-1)))
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f'{directory}: nonfinite conservative state')
        exact = exact_reference(fields['coordinates'][:, 3:3 + case['dimension']],
                             float(frame.attrib['timestep']))
        data.append((fields, state, state - exact))
    if np.max(np.abs(data[0][2])) > 1e-11:
        raise RuntimeError(f'{directory}: incorrect initial condition')
    rms = np.sqrt(np.mean(data[1][2]**2, axis=0))
    elements = round((2. if case['dimension'] == 2 else 1.)/case['h'])
    origin = -1. if case['dimension'] == 2 else 0.
    arguments = (data[1][0]['coordinates'], data[1][1], case['dimension'], elements,
                 origin, case['h'], final_time, exact_reference)
    physical = tensor_volume_errors(*arguments, case['order'], quadrature_points)
    lower_quadrature = tensor_volume_errors(*arguments, case['order'], quadrature_points - 2)
    if not np.allclose(physical['l2'], lower_quadrature['l2'], rtol=1e-6, atol=1e-12):
        raise RuntimeError(f'{directory}: component L2 quadrature is underresolved')
    if check_solver_norm and not math.isclose(np.linalg.norm(physical['l2']), l2, rel_tol=1e-3, abs_tol=1e-12):
        raise RuntimeError(f'{directory}: independent and solver physical norms disagree')
    return dict(**case, device=device, l2=l2, component_nodal_rms=rms.tolist(),
                component_physical=physical,
                solver_norm_relative_difference=float(np.linalg.norm(physical['l2'])/l2 - 1.)), data[1][0]


def check(results, device):
    (results/'assessment.json').unlink(missing_ok=True)
    analysis = results/'analysis'
    analysis.mkdir(exist_ok=True)
    for name in ('trigonometric_euler_study.py', 'mms_error_norms.py', 'mms_paths.py'):
        (analysis/name).write_bytes((Path(__file__).parent/name).read_bytes())
    (analysis/'numflux_regression_test.py').write_bytes(
        (REPOSITORY_ROOT/'tests/numflux_regression_test.py').read_bytes())
    cases = json.loads((results/'cases.json').read_text())
    measurements = []
    fields = {}
    for case in cases:
        record, values = read_case(results, case, case.get('backend', device))
        measurements.append(record)
        fields[case['name']] = values
        (results/'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
    for dimension in (2, 3):
        group = [r for r in measurements if r['dimension'] == dimension]
        levels = [r for r in group if '-level-' in r['name']]
        orders = refinement_orders([r['l2'] for r in levels])
        component_orders = refinement_orders([r['component_physical']['l2'] for r in levels])
        if np.any(orders < 2.5) or np.any(orders > 3.5):
            raise RuntimeError(f'{dimension}D P2 physical L2 orders: {orders}')
        if np.any(component_orders[-1] < 2.5) or np.any(component_orders[-1] > 3.5):
            raise RuntimeError(f'{dimension}D P2 final physical component L2 orders: {component_orders[-1]}')
        fine = levels[-1]
        named = {r['name'].split('d-', 1)[1]: r for r in group}
        if abs(named['half-dt']['l2']/fine['l2'] - 1.) > .01:
            raise RuntimeError(f'{dimension}D temporal contamination exceeds 1%')
        if named['source-off']['l2'] < 5.*fine['l2']:
            raise RuntimeError(f'{dimension}D source-off control did not distinguish forcing')
        if device == 'cuda':
            baseline = named['cpu-baseline']
            for field in FIELDS:
                if not np.allclose(fields[baseline['name']][field], fields[fine['name']][field],
                                   rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f'{dimension}D CPU/CUDA disagreement: {field}')
        print(f'{dimension}D physical L2 orders: {orders}; final physical component L2 orders: {component_orders[-1]}')
    with (results/'measurements.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    (results/'assessment.json').write_text(json.dumps(dict(passed=True, device=device,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + '\n')
    print('Multidirectional Euler MMS: PASS')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter
    records = json.loads((results/'measurements.json').read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for dimension, ax in zip((2, 3), axes):
        levels = [r for r in records if r['dimension'] == dimension and '-level-' in r['name']]
        h = np.array([r['h'] for r in levels])
        error = np.array([r['l2'] for r in levels])
        ax.loglog(h, error, 'o-', label='Physical aggregate L2')
        ax.loglog(h, error[-1]*(h/h[-1])**3, '--', label='Third-order reference')
        for spacing, value in zip(h, error):
            ax.annotate(f'{value:.2e}', (spacing, value), xytext=(4, 7), textcoords='offset points')
        ax.set(title=f'{dimension}D periodic Euler, P2, t={FINAL_TIME}', xlabel='Element width h', ylabel='Error')
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
    figure.savefig(results/'spatial-convergence.png', dpi=180)
    figure.savefig(results/'spatial-convergence.pdf')
    plt.close(figure)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, dimension in enumerate((2, 3)):
        levels = [r for r in records if r['dimension'] == dimension and '-level-' in r['name']]
        h = np.array([r['h'] for r in levels])
        errors = np.array([r['component_physical']['l2'] for r in levels])
        names = ['rho', 'rho*u', 'rho*v'] + (['rho*w'] if dimension == 3 else []) + ['rhoE']
        for component, name in enumerate(names):
            axes[row, 0].loglog(h, errors[:, component], 'o-', label=name)
            orders = np.log2(errors[:-1, component]/errors[1:, component])
            axes[row, 1].semilogx(np.sqrt(h[:-1]*h[1:]), orders, 'o-', label=name)
        order_axis = axes[row, 1]
        order_axis.set_xticks(np.sqrt(h[:-1]*h[1:]))
        order_axis.xaxis.set_major_formatter(ScalarFormatter())
        order_axis.xaxis.set_minor_locator(NullLocator())
        axes[row, 1].axhline(3., linestyle='--', color='black', label='Third order')
        axes[row, 0].set(title=f'{dimension}D physical component errors', xlabel='Element width h', ylabel='L2 error')
        axes[row, 1].set(title=f'{dimension}D observed orders', xlabel='Geometric mean element width', ylabel='Observed order')
        for ax in axes[row]:
            ax.grid(True, which='both', alpha=.3)
            ax.legend()
    figure.savefig(results/'component-convergence.png', dpi=180)
    figure.savefig(results/'component-convergence.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        if args.executable is None:
            parser.error('prepare requires --executable')
        prepare(args.source.resolve(), args.results.resolve(), args.device, args.executable.resolve(), args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
