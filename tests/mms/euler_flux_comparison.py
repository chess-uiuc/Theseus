#!/usr/bin/env python3
"""Compare supported CPG Euler flux formulations with a common manufactured solution."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from euler_spatial_survey import write_periodic_square
from mms_error_norms import headline_l2
from trigonometric_euler_study import read_case, FINAL_TIME
from mms_fluxes import FLUXES, select_fluxes


def prepare(source, results, fluxes=FLUXES, quick=False, ranks=1, matrix=False, qualified=False):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    degrees = tuple(range(1, 6)) if matrix else ((2,) if quick else (2, 4))
    levels = tuple(range(5)) if matrix else ((1, 2, 3) if quick else (2, 3, 4))
    timestep = .004 if quick else .0005
    cases = []
    for flux in fluxes:
        for degree in degrees:
            for level in levels:
                name = f'{flux}-p{degree}-level-{level}'
                directory = results/name
                directory.mkdir()
                config = copy.deepcopy(base)
                runtime = config['runTime']
                runtime.update(order=degree, ser_ref_levels=level, mesh_file=str(mesh), numerical_flux=flux,
                               dt=timestep, nsteps_max=round(FINAL_TIME/timestep) + 2, variable_dt=False,
                               print_interval=round(FINAL_TIME/timestep), cfl_check_interval=1000, final_time=FINAL_TIME, nancheck_steps=1,
                               initial_save_dt=FINAL_TIME, refined_save_dt=FINAL_TIME,
                               output_file_path=str(directory/'solution'),
                               sources=[{'type': 'trigonometric-euler-mms'}])
                runtime['conditions']['initial_conditions']['function'] = 'TrigonometricEulerMMS'
                runtime['exact_solution']['function'] = 'TrigonometricEulerMMS'
                (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
                cases.append(dict(name=name, dimension=2, flux=flux, order=degree, level=level,
                                  h=2./(3*2**level), dt=timestep, ranks=ranks, forced=True, accuracy_gate=quick,
                                  requires_timestep_control=qualified or matrix))
    if qualified or matrix:
        for flux in fluxes:
            for degree in degrees:
                fine = next(c for c in cases if c['flux'] == flux and c['order'] == degree
                            and c['level'] == levels[-1])
                case = dict(fine, name=f'{flux}-p{degree}-half-dt', dt=timestep/2.,
                            time_control=True, accuracy_gate=quick)
                directory = results/case['name']
                directory.mkdir()
                config = json.loads((results/fine['name']/'config.json').read_text())
                config['runTime'].update(dt=case['dt'], nsteps_max=round(FINAL_TIME/case['dt']) + 2,
                                         print_interval=round(FINAL_TIME/case['dt']),
                                         output_file_path=str(directory/'solution'))
                (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
                cases.append(case)
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


def check(results, device, fluxes=None):
    records = []
    for case in json.loads((results/'cases.json').read_text()):
        record, fields = read_case(results, case, device, quadrature_points=20, check_solver_norm=False)
        records.append(record)
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    fluxes = tuple(dict.fromkeys(r['flux'] for r in records))
    degrees = sorted({r['order'] for r in records})
    report = []
    for flux in fluxes:
        for degree in degrees:
            group = [r for r in records if r['flux'] == flux and r['order'] == degree and not r.get('time_control')]
            error = np.array([r['component_physical']['l2'] for r in group])
            entry = dict(flux=flux, order=degree, expected_order=degree + 1,
                         observed_component_orders=np.log2(error[:-1]/error[1:]).tolist(),
                         observed_orders=np.log2(np.array([headline_l2(r) for r in group[:-1]])/
                                                 np.array([headline_l2(r) for r in group[1:]])).tolist(),
                         quantities=['density', 'momentum', 'energy'])
            if group[0].get('accuracy_gate'):
                entry['passed'] = bool(np.all(np.array(entry['observed_component_orders'][-1]) >= degree + .5))
            control = next((r for r in records if r['flux'] == flux and r['order'] == degree and r.get('time_control')), None)
            if group[0].get('requires_timestep_control') and control is None:
                raise RuntimeError('Missing required timestep control')
            entry['timestep_control_run'] = control is not None
            if control is not None:
                entry['finest_timestep_relative_change'] = np.abs(
                    np.array(control['component_physical']['l2'])/error[-1] - 1.).tolist()
                entry['timestep_control_passed'] = bool(np.max(entry['finest_timestep_relative_change']) < .01)
                if control.get('accuracy_gate', True):
                    final_orders = np.array(entry['observed_component_orders'][-1])
                    entry['passed'] = bool(np.all(final_orders >= degree + .5)
                                           and entry['timestep_control_passed'])
            report.append(entry)
            print(json.dumps(entry), flush=True)
    (results/'flux-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if any(not entry.get('passed', True) or not entry.get('timestep_control_passed', True) for entry in report):
        raise RuntimeError('Flux spatial order or timestep-isolation check failed')


def plot(results, fluxes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter
    records = json.loads((results/'measurements.json').read_text())
    fluxes = tuple(dict.fromkeys(r['flux'] for r in records))
    degrees = sorted({r['order'] for r in records})
    figure, axes = plt.subplots(len(degrees), 3, figsize=(14, 4*len(degrees)),
                                constrained_layout=True, squeeze=False)
    for row, degree in enumerate(degrees):
        for flux in fluxes:
            group = [r for r in records if r['flux'] == flux and r['order'] == degree and not r.get('time_control')]
            h = np.array([r['h'] for r in group])
            error = np.array([headline_l2(r) for r in group])
            for component, name in enumerate(('Density', 'Momentum', 'Energy')):
                axes[row, component].loglog(h, error[:, component], 'o-', label=flux)
        for component, name in enumerate(('Density', 'Momentum', 'Energy')):
            ax = axes[row, component]
            ax.loglog(h, error[-1, component]*(h/h[-1])**(degree + 1), 'k:',
                      label=f'h^{degree + 1} expected')
            ax.set(title=f'P{degree}: {name}', xlabel='Element width', ylabel='Physical L2 error')
            ax.legend(fontsize=7)
            ax.grid(True, which='both', alpha=.3)
    figure.savefig(results/'flux-comparison.png', dpi=180)
    figure.savefig(results/'flux-comparison.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'prepare-quick', 'prepare-qualified', 'prepare-matrix', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--fluxes', default='all')
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    fluxes = select_fluxes(args.fluxes)
    if args.action in ('prepare', 'prepare-quick', 'prepare-qualified', 'prepare-matrix'):
        prepare(args.source.resolve(), args.results.resolve(), fluxes,
                quick=args.action in ('prepare-quick', 'prepare-qualified'), ranks=args.ranks,
                matrix=args.action == 'prepare-matrix', qualified=args.action == 'prepare-qualified')
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
