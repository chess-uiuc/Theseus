#!/usr/bin/env python3
"""Small periodic constant-transport CNS MMS study and error-isolation controls."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
import importlib.util
from pathlib import Path
import numpy as np
from euler_spatial_survey import write_periodic_square
from mms_error_norms import headline_l2
from trigonometric_euler_study import read_case


def prepare(source, results, ranks=1, dimension=2, quick=False):
    if quick and dimension != 2:
        raise ValueError("Quick CNS study is 2D only")
    results.mkdir(parents=True, exist_ok=True)
    if dimension == 2:
        mesh = results/'periodic-square.mesh'
        write_periodic_square(mesh)
    elif dimension == 3:
        mesh = results/'periodic-cube.msh'
        script = source/'TestCases/NavierStokes/3D/TaylorGreenVortex/make_periodic_box.py'
        spec = importlib.util.spec_from_file_location('periodic_box', script)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        generator.write_mesh(mesh, 1., 1., 1., 3, 3, 3)
    else:
        raise ValueError('CNS study requires dimension 2 or 3')
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = []
    for level in (1, 2, 3):
        cases.append(dict(name=f'level-{level}', level=level, mu=.02, dt=.001,
                          source_mode='full', flow_model='navier-stokes'))
    if not quick:
        cases.extend([
            dict(name='half-dt', level=3, mu=.02, dt=.0005, source_mode='full', flow_model='navier-stokes'),
            dict(name='euler-forcing', level=3, mu=.02, dt=.001, source_mode='euler', flow_model='navier-stokes')])
        if dimension == 2:
            cases.extend([
                dict(name='zero-transport', level=3, mu=0., dt=.001,
                     source_mode='full', flow_model='navier-stokes'),
                dict(name='euler-limit', level=3, mu=0., dt=.001,
                     source_mode='euler', flow_model='euler')])
    for case in cases:
        directory = results/case['name']
        directory.mkdir()
        config = copy.deepcopy(base)
        source_type = 'trigonometric-cns-mms' if case['source_mode'] == 'full' else 'trigonometric-euler-mms'
        runtime = config['runTime']
        runtime.update(order=2, dim=dimension, num_equations=dimension + 2, ser_ref_levels=case['level'], mesh_file=str(mesh),
                       numerical_flux='Roe', flow_model=case['flow_model'], mu=case['mu'], Pr=.72,
                       dt=case['dt'], variable_dt=False, nsteps_max=round(.2/case['dt']) + 2,
                       final_time=.2, initial_save_dt=.2, refined_save_dt=.2,
                       print_interval=round(.2/case['dt']), cfl_check_interval=1000,
                       nancheck_steps=1, output_file_path=str(directory/'solution'),
                       sources=[{'type': source_type}])
        # The inviscid-forcing negative control still evolves CNS. The Euler source
        # registration intentionally rejects that model, so use the zero-viscosity
        # CNS forcing option in this dedicated negative-control configuration.
        if case['name'] == 'euler-forcing':
            runtime['sources'] = [{'type': 'trigonometric-cns-mms', 'viscous_scale': 0.}]
        condition = dict(function='TrigonometricCNSMMS', signature=0)
        runtime['conditions']['initial_conditions'] = condition
        runtime['exact_solution'] = condition.copy()
        (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
        if quick:
            case['quick'] = True
        case.update(dimension=dimension, order=2,
                    h=(2. if dimension == 2 else 1.)/(3*2**case['level']), ranks=ranks, forced=True)
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


def check(results, device):
    records = []
    fields = {}
    cases = json.loads((results/'cases.json').read_text())
    dimensions = {case['dimension'] for case in cases}
    if len(dimensions) != 1 or not dimensions <= {2, 3}:
        raise ValueError('CNS study requires one common dimension, 2 or 3')
    dimension = next(iter(dimensions))
    modes = {case.get('quick', False) for case in cases}
    if len(modes) != 1:
        raise ValueError('CNS study requires one common testing level')
    quick = next(iter(modes))
    if quick and (dimension != 2 or len(cases) != 3):
        raise ValueError('Quick CNS study requires three 2D mesh cases')
    for case in cases:
        record, values = read_case(results, case, device,
                                   quadrature_points=16 if dimension == 2 else 12,
                                   check_solver_norm=False)
        records.append(record)
        fields[case['name']] = values
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    named = {r['name']: r for r in records}
    main = [named[f'level-{level}'] for level in (1, 2, 3)]
    errors = np.array([headline_l2(r) for r in main])
    orders = np.log2(errors[:-1]/errors[1:])
    if not np.all(np.isfinite(orders)) or np.any(orders[-1] < 2.5):
        raise RuntimeError('CNS P2 spatial order is below expected minus .5')
    temporal_change = None
    viscous_omission = None
    if not quick:
        temporal_change = np.abs(np.array(headline_l2(named['half-dt']))/errors[-1] - 1.)
        viscous_omission = np.array(headline_l2(named['euler-forcing']))/errors[-1]
        if dimension == 2:
            for name in ('Density', 'Velocity', 'Pressure', 'Specific Total Energy'):
                np.testing.assert_allclose(fields['zero-transport'][name], fields['euler-limit'][name],
                                           atol=1e-10, rtol=1e-10)
        if not np.all(np.isfinite(temporal_change)) or np.max(temporal_change) > .01:
            raise RuntimeError('CNS spatial study is temporally contaminated')
        if not np.all(np.isfinite(viscous_omission)) or np.any(viscous_omission[1:] < 5.):
            raise RuntimeError('CNS energy error does not distinguish omitted viscous forcing')
    report = dict(status='completed', accuracy_qualified=not quick, passed=True, level='quick' if quick else 'qualification',
                  expected_order=3,
                  quantities=['density', 'momentum', 'energy'], observed_orders=orders.tolist(),
                  finest_timestep_relative_change=temporal_change.tolist() if temporal_change is not None else None,
                  omitted_viscous_source_error_ratio=viscous_omission.tolist() if viscous_omission is not None else None,
                  dimension=dimension, zero_transport_euler_limit_passed=True if dimension == 2 and not quick else None)
    (results/'cns-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def plot(results, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((results/'measurements.json').read_text())
    main = sorted((record for record in records if record['name'].startswith('level-')),
                  key=lambda record: record['h'], reverse=True)
    h = np.array([r['h'] for r in main])
    errors = np.array([headline_l2(r) for r in main])
    expected = main[0]['order'] + 1
    if title is None:
        title = f"Constant-transport {main[0]['dimension']}D CNS, P{main[0]['order']}/Roe"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for column, name in enumerate(('Density', 'Momentum', 'Energy')):
        axes[0].loglog(h, errors[:, column], 'o-', label=name)
        axes[1].semilogx(np.sqrt(h[:-1]*h[1:]), np.log2(errors[:-1, column]/errors[1:, column]),
                         'o-', label=name)
    axes[0].loglog(h, errors[-1, -1]*(h/h[-1])**expected, 'k--', label=f'h^{expected} reference')
    axes[1].axhline(expected, color='black', linestyle='--', label=f'Expected {expected}')
    axes[0].set(xlabel='Element width', ylabel='Physical L2 error', title=title)
    axes[1].set(xlabel='Geometric mean width', ylabel='Observed order', title='Observed spatial rates')
    from matplotlib.ticker import NullFormatter
    axes[1].set_xticks(np.sqrt(h[:-1]*h[1:]),
                      [f'{width:.3g}' for width in np.sqrt(h[:-1]*h[1:])])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.grid(True, which='both', alpha=.3)
        ax.legend()
    figure.savefig(results/'cns-spatial.png', dpi=180)
    figure.savefig(results/'cns-spatial.pdf')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'prepare-quick', 'check', 'plot'))
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--dimension', type=int, choices=(2, 3), default=2)
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action in ('prepare', 'prepare-quick'):
        prepare(args.source.resolve(), args.results.resolve(), args.ranks, args.dimension, quick=args.action == 'prepare-quick')
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
