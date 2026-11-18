#!/usr/bin/env python3
"""Coupled 2D Sutherland CNS convergence with timestep and forcing controls."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from euler_spatial_survey import write_periodic_square
from trigonometric_euler_study import read_case
from mms_error_norms import headline_l2
from cns_mms_study import plot


def prepare(source, results, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh = results/'periodic-square.mesh'
    write_periodic_square(mesh)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = [dict(name=f'level-{level}', level=level, dt=.001,
                  viscous_scale=1., coefficient_gradient_scale=1.) for level in (1, 2, 3)]
    cases.extend([
        dict(name='half-dt', level=3, dt=.0005, viscous_scale=1., coefficient_gradient_scale=1.),
        dict(name='inviscid-forcing', level=3, dt=.001, viscous_scale=0., coefficient_gradient_scale=1.),
        dict(name='frozen-coefficients', level=3, dt=.001, viscous_scale=1., coefficient_gradient_scale=0.)])
    for case in cases:
        directory = results/case['name']
        directory.mkdir()
        config = copy.deepcopy(base)
        runtime = config['runTime']
        runtime.update(dim=2, num_equations=4, order=3, ser_ref_levels=case['level'],
                       mesh_file=str(mesh), numerical_flux='Roe', flow_model='navier-stokes',
                       R_gas=1./600., mu0=.02, Pr=.72, dt=case['dt'], variable_dt=False,
                       final_time=.2, nsteps_max=round(.2/case['dt']) + 2,
                       initial_save_dt=.2, refined_save_dt=.2,
                       print_interval=round(.2/case['dt']), nancheck_steps=1, cfl_check_interval=1000,
                       output_file_path=str(directory/'solution'),
                       sources=[dict(type='trigonometric-cns-sutherland-mms',
                                     viscous_scale=case['viscous_scale'],
                                     coefficient_gradient_scale=case['coefficient_gradient_scale'])])
        condition = dict(function='TrigonometricCNSMMS', signature=0)
        runtime['conditions']['initial_conditions'] = condition
        runtime['exact_solution'] = condition.copy()
        (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
        case.update(dimension=2, order=3, h=2./(3*2**case['level']), ranks=ranks, forced=True,
                    transport='sutherland', mu0=.02, R_gas=1./600., Pr=.72)
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{case['name']}\t{case['ranks']}\n" for case in cases))


def assess(records):
    named = {record['name']: record for record in records}
    if len(named) != 6 or set(named) != {
            'level-1', 'level-2', 'level-3', 'half-dt', 'inviscid-forcing', 'frozen-coefficients'}:
        raise RuntimeError('Sutherland qualification requires all six mesh/control cases')
    main = [named[f'level-{level}'] for level in (1, 2, 3)]
    errors = np.array([headline_l2(record) for record in main])
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.):
        raise RuntimeError('Sutherland qualification requires positive finite errors')
    orders = np.log2(errors[:-1]/errors[1:])
    temporal_change = np.abs(np.array(headline_l2(named['half-dt']))/errors[-1] - 1.)
    omitted_viscous = np.array(headline_l2(named['inviscid-forcing']))/errors[-1]
    omitted_gradients = np.array(headline_l2(named['frozen-coefficients']))/errors[-1]
    order_passed = bool(np.all(np.isfinite(orders)) and np.all(orders > 0.)
                        and np.all(orders[-1] >= 3.5))
    time_passed = bool(np.all(np.isfinite(temporal_change)) and np.max(temporal_change) < .01)
    viscous_passed = bool(np.all(np.isfinite(omitted_viscous)) and np.all(omitted_viscous[1:] >= 5.))
    gradient_passed = bool(np.all(np.isfinite(omitted_gradients)) and np.all(omitted_gradients[1:] >= 2.))
    return dict(quantities=['density', 'momentum', 'energy'], expected_order=4,
                observed_orders=orders.tolist(), finest_timestep_relative_change=temporal_change.tolist(),
                omitted_viscous_source_error_ratio=omitted_viscous.tolist(),
                omitted_coefficient_gradient_error_ratio=omitted_gradients.tolist(),
                order_passed=order_passed, timestep_isolation_passed=time_passed,
                viscous_control_passed=viscous_passed, coefficient_gradient_control_passed=gradient_passed,
                passed=order_passed and time_passed and viscous_passed and gradient_passed)


def check(results, device):
    (results/'sutherland-assessment.json').unlink(missing_ok=True)
    records = []
    for case in json.loads((results/'cases.json').read_text()):
        record, _ = read_case(results, case, device, quadrature_points=16, check_solver_norm=False)
        records.append(record)
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    report = assess(records)
    (results/'sutherland-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Sutherland convergence or error-isolation controls unresolved; inspect retained evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--ranks', type=int, default=1)
    args = parser.parse_args()
    if args.ranks < 1:
        parser.error('ranks must be positive')
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve(), title='Sutherland 2D CNS, P3/Roe')
