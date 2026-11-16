#!/usr/bin/env python3
"""Coupled Euler temporal convergence with an independently checked spatial floor."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from mms_error_norms import headline_l2
from trigonometric_euler_study import exact_fields, read_case, FINAL_TIME
from euler_spatial_survey import write_periodic_square

TIME_SCALE = 126.
TIMESTEPS = (.001, .0005, .00025, .000125)


def prepare(source, results, quick=False, qualified=False, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    final_time = .02 if quick else FINAL_TIME
    timesteps = TIMESTEPS[:3] if quick else TIMESTEPS
    cases = []
    for method in ('RK3SSP', 'RK4'):
        variants = [(f'dt-{index}', 15, dt) for index, dt in enumerate(timesteps)]
        if not quick or qualified:
            variants.append(('spatial-control', 17, timesteps[-1]))
        for name, degree, dt in variants:
            directory = results/f'{method}-{name}'
            directory.mkdir()
            config = copy.deepcopy(base)
            runtime = config['runTime']
            runtime.update(nancheck_steps=1, order=degree, ser_ref_levels=1, mesh_file=str(mesh), ode_solver=method,
                           dt=dt, nsteps_max=round(final_time/dt) + 2, final_time=final_time,
                           initial_save_dt=final_time, refined_save_dt=final_time,
                           print_interval=round(final_time/dt), cfl_check_interval=1000, variable_dt=False,
                           output_file_path=str(directory/'solution'),
                           sources=[{'type': 'trigonometric-euler-mms', 'time_scale': TIME_SCALE}])
            condition = dict(function='TemporalTrigonometricEulerMMS', signature=1,
                             params={'x1': TIME_SCALE})
            runtime['conditions']['initial_conditions'] = condition
            runtime['exact_solution'] = condition.copy()
            (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
            cases.append(dict(name=directory.name, dimension=2, order=degree, level=1,
                              h=1./3., dt=dt, ranks=ranks, forced=True, method=method,
                              time_scale=TIME_SCALE, final_time=final_time, requires_spatial_control=not quick or qualified,
                              spatial_control=name == 'spatial-control'))
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


def check(results, device):
    records = []
    cases = json.loads((results/'cases.json').read_text())
    scales = {case.get('time_scale', TIME_SCALE) for case in cases}
    if len(scales) != 1:
        raise ValueError('Temporal study requires a common time scale')
    scale = next(iter(scales))
    reference = lambda coordinates, time: exact_fields(coordinates, time*scale)
    for case in cases:
        record, fields = read_case(results, case, device, reference, quadrature_points=24,
                                   check_solver_norm=False, final_time=case.get('final_time', FINAL_TIME))
        records.append(record)
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    assessment = []
    for method, expected in (('RK3SSP', 3), ('RK4', 4)):
        group = [r for r in records if r['method'] == method and not r['spatial_control']]
        control = next((r for r in records if r['method'] == method and r['spatial_control']), None)
        if group[0].get('requires_spatial_control') and control is None:
            raise RuntimeError('Missing required spatial-floor control')
        errors = np.array([r['component_physical']['l2'] for r in group])
        orders = np.log2(errors[:-1]/errors[1:])
        spatial_change = (np.abs(np.array(control['component_physical']['l2'])/errors[-1] - 1.)
                          if control is not None else None)
        passed = bool(np.all(np.abs(orders[-2:] - expected) < .15)
                      and (spatial_change is None or np.all(spatial_change < .01)))
        record = dict(method=method, expected_order=expected, observed_component_orders=orders.tolist(),
                      finest_spatial_control_relative_change=spatial_change.tolist() if spatial_change is not None else None,
                      spatial_control_run=control is not None, passed=passed)
        assessment.append(record)
        print(json.dumps(record), flush=True)
    (results/'temporal-survey.json').write_text(json.dumps(dict(studies=assessment,
        passed=all(r['passed'] for r in assessment)), indent=2) + '\n')
    if not all(r['passed'] for r in assessment):
        raise RuntimeError('Temporal convergence or spatial-floor isolation is unresolved; inspect retained measurements')
    print('Coupled temporal convergence: PASS')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((results/'measurements.json').read_text())
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, (method, expected) in enumerate((('RK3SSP', 3), ('RK4', 4))):
        group = [r for r in records if r['method'] == method and not r['spatial_control']]
        dt = np.array([r['dt'] for r in group])
        error = np.array([headline_l2(r) for r in group])
        for component, name in enumerate(('Density', 'Momentum', 'Energy')):
            axes[row, 0].loglog(dt, error[:, component], 'o-', label=name)
            axes[row, 1].semilogx(np.sqrt(dt[:-1]*dt[1:]), np.log2(error[:-1, component]/error[1:, component]), 'o-', label=name)
        axes[row, 0].loglog(dt, error[0, -1]*(dt/dt[0])**expected, 'k--', label=f'dt^{expected} expected')
        axes[row, 1].axhline(expected, color='black', linestyle='--', label=f'Expected {expected}')
        axes[row, 0].set(title=f'{method}: physical L2 error', xlabel='Timestep', ylabel='Error')
        axes[row, 1].set(title=f'{method}: observed temporal order', xlabel='Geometric mean timestep', ylabel='Order')
        for ax in axes[row]:
            ax.legend()
            ax.grid(True, which='both', alpha=.3)
    figure.savefig(results/'temporal-survey.png', dpi=180)
    figure.savefig(results/'temporal-survey.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'prepare-quick', 'prepare-qualified', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), ranks=args.ranks)
    elif args.action in ('prepare-quick', 'prepare-qualified'):
        prepare(args.source.resolve(), args.results.resolve(), quick=True,
                qualified=args.action == 'prepare-qualified', ranks=args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
