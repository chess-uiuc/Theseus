#!/usr/bin/env python3
"""Coupled constant-transport CNS temporal convergence on a fixed fine polynomial space."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
from euler_spatial_survey import write_periodic_square
from euler_temporal_survey import check, plot

TIME_SCALE = 500.


def prepare(source, results, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = []
    for method in ('RK3SSP', 'RK4'):
        # RK3 needs finer timesteps to isolate its cubic coupling error from the
        # higher-order quadrature of the rapidly varying manufactured forcing.
        timesteps = ((.0000625, .00003125, .000015625) if method == 'RK3SSP'
                     else (.00025, .000125, .0000625))
        variants = [(f'dt-{index}', 19, dt) for index, dt in enumerate(timesteps)]
        variants.append(('spatial-control', 21, timesteps[-1]))
        for name, degree, dt in variants:
            directory = results/f'{method}-{name}'
            directory.mkdir()
            config = copy.deepcopy(base)
            runtime = config['runTime']
            runtime.update(order=degree, ser_ref_levels=0, mesh_file=str(mesh), numerical_flux='Roe',
                           flow_model='navier-stokes', mu=.02, Pr=.72, ode_solver=method,
                           dt=dt, variable_dt=False, final_time=.01, nsteps_max=round(.01/dt) + 2,
                           nancheck_steps=1, print_interval=round(.01/dt), cfl_check_interval=1000,
                           initial_save_dt=.01, refined_save_dt=.01,
                           output_file_path=str(directory/'solution'),
                           sources=[{'type': 'trigonometric-cns-mms', 'time_scale': TIME_SCALE}])
            condition = dict(function='TemporalTrigonometricEulerMMS', signature=1,
                             params={'x1': TIME_SCALE})
            runtime['conditions']['initial_conditions'] = condition
            runtime['exact_solution'] = condition.copy()
            (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
            cases.append(dict(name=directory.name, dimension=2, order=degree, level=0, h=2./3.,
                              dt=dt, ranks=ranks, forced=True, method=method, time_scale=TIME_SCALE,
                              final_time=.01, requires_spatial_control=True,
                              spatial_control=name == 'spatial-control'))
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())
