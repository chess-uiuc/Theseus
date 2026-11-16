#!/usr/bin/env python3
"""Selected-flux CNS spatial convergence with a timestep control per flux."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from euler_spatial_survey import write_periodic_square
from mms_error_norms import headline_l2
from mms_fluxes import FLUXES, select_fluxes
from trigonometric_euler_study import read_case


def prepare(source, results, transport, fluxes=FLUXES, ranks=1):
    if transport not in ('constant', 'sutherland'):
        raise ValueError('Choose constant or sutherland transport')
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    degree = 2 if transport == 'constant' else 3
    source_type = 'trigonometric-cns-mms' if transport == 'constant' else 'trigonometric-cns-sutherland-mms'
    cases = []
    for flux in fluxes:
        variants = [(f'level-{level}', level, .001, False) for level in (1, 2, 3)]
        variants.append(('half-dt', 3, .0005, True))
        for name, level, dt, control in variants:
            directory = results/f'{flux}-{name}'
            directory.mkdir()
            config = copy.deepcopy(base)
            runtime = config['runTime']
            runtime.update(dim=2, num_equations=4, order=degree, ser_ref_levels=level,
                           mesh_file=str(mesh), flow_model='navier-stokes', numerical_flux=flux,
                           mu=.02, mu0=.02, Pr=.72, R_gas=1. if transport == 'constant' else 1./600.,
                           dt=dt, variable_dt=False, final_time=.2, nsteps_max=round(.2/dt) + 2,
                           initial_save_dt=.2, refined_save_dt=.2, print_interval=round(.2/dt),
                           cfl_check_interval=1000, nancheck_steps=1,
                           output_file_path=str(directory/'solution'), sources=[dict(type=source_type)])
            condition = dict(function='TrigonometricCNSMMS', signature=0)
            runtime['conditions']['initial_conditions'] = condition
            runtime['exact_solution'] = condition.copy()
            (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
            cases.append(dict(name=directory.name, transport=transport, flux=flux, order=degree,
                              dimension=2, level=level, h=2./(3*2**level), dt=dt, ranks=ranks,
                              forced=True, time_control=control))
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


def assess(records):
    transports = {record['transport'] for record in records}
    if len(transports) != 1 or not transports <= {'constant', 'sutherland'}:
        raise RuntimeError('CNS flux study requires one common transport model')
    studies = []
    for flux in dict.fromkeys(record['flux'] for record in records):
        group = [r for r in records if r['flux'] == flux]
        main = sorted((r for r in group if not r['time_control']), key=lambda r: r['h'], reverse=True)
        controls = [r for r in group if r['time_control']]
        if (len(main) != 3 or len(controls) != 1 or controls[0]['h'] != main[-1]['h']
                or controls[0]['dt'] != main[-1]['dt']/2.
                or any(main[index]['h'] != 2.*main[index + 1]['h'] for index in (0, 1))):
            raise RuntimeError('Each CNS flux requires three meshes and one finest timestep control')
        degree = 2 if main[0]['transport'] == 'constant' else 3
        if any(r['order'] != degree or r['dimension'] != 2 for r in group):
            raise RuntimeError('Unexpected degree/dimension in CNS flux study')
        errors = np.array([headline_l2(record) for record in main])
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0.):
            raise RuntimeError('CNS flux study requires positive finite errors')
        orders = np.log2(errors[:-1]/errors[1:])
        component_orders = np.log2(np.array([r['component_physical']['l2'] for r in main[:-1]])/
                                   np.array([r['component_physical']['l2'] for r in main[1:]]))
        time_change = np.abs(np.array(headline_l2(controls[0]))/errors[-1] - 1.)
        order_passed = bool(np.all(np.isfinite(orders)) and np.all(orders > 0.)
                            and np.all(orders[-1] >= degree + .5))
        time_passed = bool(np.all(np.isfinite(time_change)) and np.all(time_change < .01))
        studies.append(dict(flux=flux, transport=main[0]['transport'], degree=degree,
                            expected_order=degree + 1, observed_orders=orders.tolist(),
                            observed_component_orders=component_orders.tolist(),
                            finest_timestep_relative_change=time_change.tolist(),
                            order_passed=order_passed, timestep_isolation_passed=time_passed,
                            passed=order_passed and time_passed))
    return dict(quantities=['density', 'momentum', 'energy'], studies=studies,
                passed=all(study['passed'] for study in studies))


def check(results, device):
    (results/'cns-flux-assessment.json').unlink(missing_ok=True)
    records = []
    for case in json.loads((results/'cases.json').read_text()):
        record, _ = read_case(results, case, device, quadrature_points=16, check_solver_norm=False)
        records.append(record)
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    report = assess(records)
    (results/'cns-flux-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('CNS flux order or timestep isolation unresolved; inspect retained evidence')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    records = json.loads((results/'measurements.json').read_text())
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    expected = records[0]['order'] + 1
    for flux in dict.fromkeys(r['flux'] for r in records):
        group = sorted((r for r in records if r['flux'] == flux and not r['time_control']),
                       key=lambda r: r['h'], reverse=True)
        h = np.array([r['h'] for r in group])
        errors = np.array([headline_l2(r) for r in group])
        for column in range(3):
            axes[0, column].loglog(h, errors[:, column], 'o-', label=flux)
            axes[1, column].semilogx(np.sqrt(h[:-1]*h[1:]),
                                    np.log(errors[:-1, column]/errors[1:, column])/np.log(h[:-1]/h[1:]), 'o-', label=flux)
    for column, quantity in enumerate(('Density', 'Momentum', 'Energy')):
        axes[0, column].loglog(h, errors[-1, column]*(h/h[-1])**expected,
                              'k--', label=f'h^{expected} reference')
        axes[1, column].axhline(expected, color='black', linestyle='--', label=f'Expected {expected}')
        axes[0, column].set(title=quantity, xlabel='Element width', ylabel='Physical L2 error')
        axes[1, column].set(xlabel='Geometric mean width', ylabel='Observed order')
        axes[1, column].set_xticks(np.sqrt(h[:-1]*h[1:]),
                                  [f'{value:.3g}' for value in np.sqrt(h[:-1]*h[1:])])
        axes[1, column].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=.3)
    figure.suptitle(f"{records[0]['transport']} MMS, P{records[0]['order']}")
    figure.savefig(results/'cns-fluxes.png', dpi=180)
    figure.savefig(results/'cns-fluxes.pdf')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--transport', choices=('constant', 'sutherland'), default='constant')
    parser.add_argument('--fluxes', default='all')
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    fluxes = select_fluxes(args.fluxes)
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), args.transport, fluxes, args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())
