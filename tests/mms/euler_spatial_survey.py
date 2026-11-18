#!/usr/bin/env python3
"""Extended p=1..5 Euler spatial survey; report measured versus expected orders."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from mms_error_norms import headline_l2
from trigonometric_euler_study import read_case, FINAL_TIME


def write_periodic_square(path, cells=3):
    lines = ['MFEM mesh v1.0', '', 'dimension', '2', '', 'elements', str(cells*cells)]
    vertex = lambda i, j: (i % cells) + cells*(j % cells)
    for j in range(cells):
        for i in range(cells):
            nodes = [vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)]
            lines.append('1 3 ' + ' '.join(map(str, nodes)))
    lines += ['', 'boundary', '0', '', 'vertices', str(cells*cells), '', 'nodes',
              'FiniteElementSpace', 'FiniteElementCollection: L2_T1_2D_P1', 'VDim: 2', 'Ordering: 1', '']
    for j in range(cells):
        for i in range(cells):
            for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                lines.append(f'{-1. + 2.*(i + di)/cells:.17g} {-1. + 2.*(j + dj)/cells:.17g}')
    path.write_text('\n'.join(lines) + '\n')


def prepare(source, results, cells=3, levels=5, timestep=.0005):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh, cells)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = []
    for degree in range(1, 6):
        for level in range(levels):
            name = f'p{degree}-level-{level}'
            directory = results/name
            directory.mkdir()
            config = copy.deepcopy(base)
            runtime = config['runTime']
            runtime.update(order=degree, ser_ref_levels=level, mesh_file=str(mesh),
                           dt=timestep, variable_dt=False, nsteps_max=round(FINAL_TIME/timestep) + 2, final_time=FINAL_TIME,
                           initial_save_dt=FINAL_TIME, refined_save_dt=FINAL_TIME,
                           output_file_path=str(directory/'solution'),
                           sources=[{'type': 'trigonometric-euler-mms'}])
            runtime['conditions']['initial_conditions']['function'] = 'TrigonometricEulerMMS'
            runtime['exact_solution']['function'] = 'TrigonometricEulerMMS'
            (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
            cases.append(dict(name=name, dimension=2, order=degree, level=level,
                              h=2./(cells*2**level), dt=timestep, ranks=1, forced=True))
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    prepare_controls(results)


def prepare_controls(results):
    cases = json.loads((results/'cases.json').read_text())
    for degree in range(1, 6):
        fine = min((r for r in cases if r['order'] == degree and not r.get('time_control')),
                   key=lambda r: r['h'])
        case = dict(fine, name=f'p{degree}-half-dt', dt=fine['dt']/2., time_control=True)
        directory = results/case['name']
        directory.mkdir()
        config = json.loads((results/fine['name']/'config.json').read_text())
        config['runTime'].update(dt=case['dt'], nsteps_max=round(FINAL_TIME/case['dt']) + 2,
                                 output_file_path=str(directory/'solution'))
        (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
        cases.append(case)
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')


def check(results, device):
    measurements = []
    for case in json.loads((results/'cases.json').read_text()):
        # Use the independent high-quadrature norm as primary. Retain any difference
        # from the solver's default quadrature instead of qualifying coarse grids.
        record, fields = read_case(results, case, device, quadrature_points=20,
                                   check_solver_norm=False)
        measurements.append(record)
        (results/'measurements.json').write_text(json.dumps(measurements, indent=2) + '\n')
    report = []
    for degree in range(1, 6):
        group = [r for r in measurements if r['order'] == degree and not r.get('time_control', False)]
        errors = np.array([r['component_physical']['l2'] for r in group])
        orders = np.log2(errors[:-1]/errors[1:])
        control = next(r for r in measurements if r['order'] == degree and r.get('time_control', False))
        temporal_change = np.abs(np.array(control['component_physical']['l2'])/errors[-1] - 1.)
        entry = dict(finest_timestep_relative_change=temporal_change.tolist(), polynomial_order=degree, expected_l2_order=degree + 1,
                     measured_component_orders=orders.tolist(),
                     error_decades=np.log10(errors[0]/errors[-1]).tolist())
        report.append(entry)
        print(json.dumps(entry), flush=True)
    (results/'spatial-survey.json').write_text(json.dumps(
        dict(completed=True, accuracy_qualified=False, studies=report), indent=2) + '\n')
    print('Spatial survey completed; asymptotic accuracy requires interpretation, not a broad slope-window pass.')



def completed_cases(results, cases):
    completed = []
    excluded = []
    for case in cases:
        log = results/case['name']/'run.log'
        if log.exists() and 'Total: 1 | Succeeded: 1 | Failed: 0' in log.read_text():
            completed.append(case)
        else:
            excluded.append(case['name'])
    return completed, excluded


def check_completed(results, device):
    """Analyze completed cases without qualifying an interrupted survey."""
    cases = json.loads((results/'cases.json').read_text())
    completed, excluded = completed_cases(results, cases)
    records = []
    for case in completed:
        record, _ = read_case(results, case, device, quadrature_points=20, check_solver_norm=False)
        records.append(record)
        print(f"Analyzed {case['name']}", flush=True)
        (results/'completed-measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    report = dict(status='INCOMPLETE', accuracy_qualified=False, excluded=excluded, studies=[])
    for degree in sorted({r['order'] for r in records}):
        group = sorted((r for r in records if r['order'] == degree and not r.get('time_control')),
                       key=lambda r: -r['h'])
        if len(group) < 2:
            continue
        errors = np.array([headline_l2(r) for r in group])
        h = np.array([r['h'] for r in group])
        control = next((r for r in records if r['order'] == degree and r.get('time_control')
                        and r['h'] == group[-1]['h']), None)
        entry = dict(order=degree, h=h.tolist(), quantities=['density', 'momentum', 'energy'],
                     observed_orders=(np.log(errors[:-1]/errors[1:])/np.log(h[:-1]/h[1:])[:, None]).tolist(),
                     timestep_control_complete=control is not None)
        if control:
            entry['finest_timestep_relative_change'] = (np.abs(np.array(headline_l2(control))/errors[-1] - 1.)).tolist()
        report['studies'].append(entry)
    (results/'completed-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    plot_density(results, filename='completed-measurements.json', prefix='completed-density-orders',
                 subtitle='Completed fine-grid cases only · interrupted survey')

def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter
    records = json.loads((results/'measurements.json').read_text())
    figure, axes = plt.subplots(5, 2, figsize=(11, 18), constrained_layout=True)
    for row, degree in enumerate(range(1, 6)):
        group = [r for r in records if r['order'] == degree and not r.get('time_control', False)]
        h = np.array([r['h'] for r in group])
        error = np.array([headline_l2(r) for r in group])
        expected = degree + 1
        for component, name in enumerate(('Density', 'Momentum', 'Energy')):
            axes[row, 0].loglog(h, error[:, component], 'o-', label=name)
            axes[row, 1].semilogx(np.sqrt(h[:-1]*h[1:]), np.log2(error[:-1, component]/error[1:, component]),
                                 'o-', label=name)
        axes[row, 0].loglog(h, error[-1, -1]*(h/h[-1])**expected, 'k--', label=f'h^{expected} expected')
        axes[row, 1].axhline(expected, color='black', linestyle='--', label=f'Expected {expected}')
        axes[row, 1].set_xticks(np.sqrt(h[:-1]*h[1:]))
        axes[row, 1].xaxis.set_major_formatter(ScalarFormatter())
        axes[row, 1].xaxis.set_minor_locator(NullLocator())
        axes[row, 0].set(title=f'P{degree}: physical L2 errors', xlabel='Element width h', ylabel='Error')
        axes[row, 1].set(title=f'P{degree}: observed versus expected order', xlabel='Geometric mean element width', ylabel='Order')
        for ax in axes[row]:
            ax.grid(True, which='both', alpha=.3)
            ax.legend(fontsize=8)
    figure.savefig(results/'spatial-survey.png', dpi=150)
    figure.savefig(results/'spatial-survey.pdf')
    plt.close(figure)



def plot_density(results, filename='measurements.json', prefix='density-orders', subtitle=''):
    """Compare all polynomial degrees in one density-only figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((results/filename).read_text())
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for degree in sorted({r['order'] for r in records if not r.get('time_control')}):
        group = sorted((r for r in records if r['order'] == degree
                        and not r.get('time_control', False)), key=lambda r: -r['h'])
        if len(group) < 2:
            continue
        h = np.array([r['h'] for r in group])
        error = np.array([r['component_physical']['l2'][0] for r in group])
        color = f'C{degree - 1}'
        axes[0].loglog(h, error, 'o-', color=color, label=f'P{degree}')
        axes[0].loglog(h[-3:], error[-1]*(h[-3:]/h[-1])**(degree + 1),
                       '--', color=color, alpha=.65)
        midpoint = np.sqrt(h[:-1]*h[1:])
        axes[1].semilogx(midpoint, np.log(error[:-1]/error[1:])/np.log(h[:-1]/h[1:]),
                         'o-', color=color, label=f'P{degree} (expected {degree + 1})')
        axes[1].axhline(degree + 1, color=color, linestyle='--', alpha=.65)
    axes[0].set(title='Density error by polynomial degree', xlabel='Element width h', ylabel='Physical L2 density error')
    axes[1].set(title='Density convergence order', xlabel='Geometric mean element width', ylabel='Observed order')
    for ax in axes:
        ax.grid(True, which='both', alpha=.25)
        ax.legend()
    dt = next(r['dt'] for r in records if not r.get('time_control'))
    figure.suptitle(f'Euler MMS · LLF · fixed dt={dt:g} · t=0.2\n'
                   f'Dashed lines: expected h^(p+1) slopes / orders · {subtitle}')
    figure.savefig(results/f'{prefix}.png', dpi=180)
    figure.savefig(results/f'{prefix}.pdf')
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'prepare-fine', 'prepare-controls', 'check', 'check-completed', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve())
    elif args.action == 'prepare-fine':
        prepare(args.source.resolve(), args.results.resolve(), cells=20, levels=4, timestep=.000125)
    elif args.action == 'prepare-controls':
        prepare_controls(args.results.resolve())
    elif args.action == 'check-completed':
        check_completed(args.results.resolve(), args.device)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())
        plot_density(args.results.resolve())


if __name__ == '__main__':
    main()
