#!/usr/bin/env python3
"""Measure R_h(I_h U,t) - I_h U_t using the production pure-DG Euler operator."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from mms_fluxes import FLUXES, select_fluxes
from euler_spatial_survey import write_periodic_square
from mms_error_norms import tensor_volume_errors, headline_l2
from trigonometric_euler_study import exact_fields

TIMES = (0., .2, 1.)


def exact_time_derivative(coordinates, time):
    # All operations in the independent NumPy specification are analytic.
    return np.imag(exact_fields(coordinates, time + 1e-30j))/1e-30


def prepare(source, results, fluxes=FLUXES, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = []
    for flux in fluxes:
        for degree in range(1, 6):
            for level in (1, 2, 3):
                cases.append(dict(name=f'{flux}-p{degree}-level-{level}', flux=flux,
                                  order=degree, level=level, ranks=ranks, constant=False))
        cases.append(dict(name=f'{flux}-constant', flux=flux, order=3, level=1, ranks=ranks, constant=True))
    for case in cases:
        directory = results/case['name']
        directory.mkdir()
        config = copy.deepcopy(base)
        config['runTime'].update(order=case['order'], ser_ref_levels=case['level'],
                                 numerical_flux=case['flux'], mesh_file=str(mesh),
                                 rhs_output=str(directory/'rhs'), rhs_times=TIMES,
                                 rhs_constant=case['constant'],
                                 sources=[] if case['constant'] else [{'type': 'trigonometric-euler-mms'}])
        (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{c['name']}\t{c['ranks']}\n" for c in cases))


def read_sample(results, case, sample):
    parts = [np.loadtxt(results/case['name']/f'rhs-t{sample}-rank{rank}.csv',
                        delimiter=',', skiprows=1, ndmin=2) for rank in range(case['ranks'])]
    data = np.vstack(parts)
    if data.shape[1] != 14 or not np.all(np.isfinite(data)):
        raise ValueError(f"Invalid RHS output: {case['name']}")
    return data


def check(results, device):
    cases = json.loads((results/'cases.json').read_text())
    records = []
    for case in cases:
        log = (results/case['name']/'run.log').read_text()
        if f'Device configuration: {device}' not in log:
            raise ValueError(f"Backend not confirmed: {case['name']}")
        for sample, time in enumerate(TIMES):
            data = read_sample(results, case, sample)
            coordinates = data[:, 3:5]
            if case['constant']:
                np.testing.assert_allclose(data[:, 10:], 0., atol=2e-11, rtol=0.)
                continue
            np.testing.assert_allclose(data[:, 6:10], exact_fields(coordinates, time), atol=2e-13, rtol=0.)
            defect = data[:, 10:] - exact_time_derivative(coordinates, time)
            elements = 3*2**case['level']
            zero = lambda points, time: np.zeros((len(points), 4))
            norms = tensor_volume_errors(data[:, :6], defect, 2, elements, -1., 2./elements,
                                         time, zero, case['order'], case['order'] + 2)
            records.append(dict(**case, time=time, h=2./elements, component_physical=norms))
    (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    fluxes = tuple(dict.fromkeys(c['flux'] for c in cases))
    report = []
    for time in TIMES:
        for flux in fluxes:
            for degree in range(1, 6):
                group = [r for r in records if r['time'] == time and r['flux'] == flux
                         and r['order'] == degree]
                errors = np.array([headline_l2(r) for r in group])
                report.append(dict(time=time, flux=flux, order=degree,
                                   reference_order=degree, quantities=['density', 'momentum', 'energy'],
                                   observed_orders=np.log2(errors[:-1]/errors[1:]).tolist()))
    (results/'rhs-orders.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: finite RHS, exact input, {len(fluxes)} constant-state checks.')
    print('RHS rates are observations; no universal degree-p accuracy assertion is imposed.')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = json.loads((results/'measurements.json').read_text())
    fluxes = tuple(dict.fromkeys(r['flux'] for r in records))
    for time in TIMES:
        figure, axes = plt.subplots(5, 3, figsize=(13, 17), constrained_layout=True)
        for row, degree in enumerate(range(1, 6)):
            for flux in fluxes:
                group = [r for r in records if r['time'] == time and r['flux'] == flux
                         and r['order'] == degree]
                h = np.array([r['h'] for r in group])
                error = np.array([headline_l2(r) for r in group])
                for column in range(3):
                    axes[row, column].loglog(h, error[:, column], 'o-', label=flux)
            for column, quantity in enumerate(('Density', 'Momentum', 'Energy')):
                ax = axes[row, column]
                ax.loglog(h, error[-1, column]*(h/h[-1])**degree, 'k:', label=f'h^{degree} reference')
                ax.set(title=f'P{degree}: {quantity}', xlabel='Element width', ylabel='RHS defect L2')
                ax.grid(True, which='both', alpha=.3)
                ax.legend(fontsize=7)
        figure.suptitle(f'Projected Euler RHS defect, t={time:g}')
        figure.savefig(results/f'rhs-t{time:g}.png', dpi=150)
        figure.savefig(results/f'rhs-t{time:g}.pdf')
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--fluxes', default='all')
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), select_fluxes(args.fluxes), args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
