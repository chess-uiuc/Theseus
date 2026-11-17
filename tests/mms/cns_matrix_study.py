#!/usr/bin/env python3
"""Extended CNS degree/mesh/flux surveys with explicit timestep isolation."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
from euler_spatial_survey import write_periodic_square
from mms_error_norms import headline_l2
from mms_fluxes import FLUXES, select_fluxes
from mms_regimes import REGIMES, mean_velocity_scale, regime_evidence
from trigonometric_euler_study import read_case


def select_degrees(selection):
    try:
        degrees = tuple(int(value.strip()) for value in selection.split(','))
    except ValueError as error:
        raise ValueError('Degrees must be a comma-separated subset of 1,2,3,4,5') from error
    if not degrees or len(set(degrees)) != len(degrees) or any(p not in range(1, 6) for p in degrees):
        raise ValueError('Degrees must be a unique subset of 1,2,3,4,5')
    return degrees


def timestep(degree, cells, reference_dt, final_time):
    # Anchor: demonstrated stable P3, 24 cells, dt=.001. Refine diffusively;
    # reduce higher-degree timesteps with p^-4, and never enlarge above the anchor.
    bound = reference_dt*min(1., (24./cells)**2)*min(1., (3./degree)**4)
    if not math.isfinite(bound) or bound <= 0.:
        raise ValueError('Requested refinement produces an invalid timestep')
    steps = math.ceil(final_time/bound)
    return final_time/steps, steps


def prepare(source, results, transport, fluxes=FLUXES, ranks=1, degrees=(1, 2, 3, 4, 5),
            mesh_levels=5, start_level=1, base_cells=3, reference_dt=.001, final_time=.2,
            cells_sequence=(12, 18, 27, 40, 60, 90), regime="transonic"):
    mach_evidence = regime_evidence(regime)
    effective_reference_dt = reference_dt/max(1., mean_velocity_scale(regime))
    if (transport not in ('euler', 'constant', 'sutherland') or mesh_levels < 3 or start_level < 0
            or base_cells < 3 or ranks < 1 or not degrees or len(set(degrees)) != len(degrees)
            or any(p not in range(1, 6) for p in degrees) or not fluxes or len(set(fluxes)) != len(fluxes)
            or any(flux not in FLUXES for flux in fluxes)
            or not math.isfinite(reference_dt) or reference_dt <= 0.
            or not math.isfinite(final_time) or final_time <= 0.):
        raise ValueError('Invalid CNS matrix: require transport, positive dt/time/ranks, P1-P5, '
                         'at least three meshes, nonnegative start level and base cells >=3')
    cells_sequence = tuple(cells_sequence) if cells_sequence is not None else tuple(
        base_cells*2**(start_level + i) for i in range(mesh_levels))
    if (len(cells_sequence) < 3 or any(type(n) is not int or n < 3 for n in cells_sequence)
            or any(a >= b for a, b in zip(cells_sequence, cells_sequence[1:]))):
        raise ValueError('Cells must be at least three strictly increasing integers >=3')
    mesh_levels = len(cells_sequence)
    results.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for cells in cells_sequence:
        mesh = results/f'periodic-square-{cells}.mesh'
        write_periodic_square(mesh, cells)
        meshes[cells] = mesh
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    source_type = {'euler': 'trigonometric-euler-mms', 'constant': 'trigonometric-cns-mms',
                   'sutherland': 'trigonometric-cns-sutherland-mms'}[transport]
    cases = []
    for flux in fluxes:
        for degree in degrees:
            group = []
            for index, cells in enumerate(cells_sequence):
                level = 0
                dt, steps = timestep(degree, cells, effective_reference_dt, final_time)
                group.append(dict(name=f'{flux}-p{degree}-cells-{cells}', level=level,
                                  cells=cells, dt=dt, steps=steps, time_control=False))
            control = dict(group[-1], name=f'{flux}-p{degree}-half-dt',
                           dt=group[-1]['dt']/2., steps=group[-1]['steps']*2, time_control=True)
            group.append(control)
            for case in group:
                directory = results/case['name']
                directory.mkdir()
                config = copy.deepcopy(base)
                runtime = config['runTime']
                runtime.update(dim=2, num_equations=4, order=degree, ser_ref_levels=case['level'],
                               mesh_file=str(meshes[case['cells']]),
                               flow_model='euler' if transport == 'euler' else 'navier-stokes', numerical_flux=flux,
                               mu=.02, mu0=.02, Pr=.72,
                               R_gas=1./600. if transport == 'sutherland' else 1.,
                               dt=case['dt'], variable_dt=False, final_time=final_time,
                               nsteps_max=case['steps'] + 2, print_interval=case['steps'],
                               initial_save_dt=final_time, refined_save_dt=final_time,
                               nancheck_steps=1, cfl_check_interval=1000,
                               output_file_path=str(directory/'solution'), sources=[dict(type=source_type, regime=regime)])
                condition = dict(function='TrigonometricEulerMMS' if transport == 'euler' else 'TrigonometricCNSMMS', signature=0)
                if regime != 'transonic':
                    condition = dict(function='RegimeTrigonometricMMS', signature=1,
                                     params={'x1': REGIMES.index(regime)})
                runtime['conditions']['initial_conditions'] = condition
                runtime['exact_solution'] = condition.copy()
                (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
                case.update(regime=regime, transport=transport, flux=flux, order=degree, dimension=2,
                            h=2./case['cells'], ranks=ranks, forced=True, final_time=final_time,
                            mesh_levels=mesh_levels, start_level=start_level, base_cells=base_cells,
                            unknowns=4*case['cells']**2*(degree + 1)**2)
                cases.append(case)
    plan = dict(regime=regime, mach_evidence=mach_evidence,
                effective_reference_dt=effective_reference_dt, transport=transport, degrees=list(degrees), fluxes=list(fluxes), mesh_levels=mesh_levels,
                start_level=start_level, base_cells=base_cells, cells=list(cells_sequence), reference_dt=reference_dt,
                final_time=final_time, cases=len(cases),
                max_steps=max(case['steps'] for case in cases),
                max_unknowns=max(case['unknowns'] for case in cases),
                unknown_steps=sum(case['steps']*case['unknowns'] for case in cases),
                timestep_policy='reference_dt / max(1,mean_velocity_scale) * min(1,(24/cells)^2) * min(1,(3/p)^4); '
                                'round step count upward to land at final time; finest control dt/2',
                scope='observational spatial survey; nominal-order shortfalls flagged, '
                      'timestep/nonfinite failures prevent a control pass; not new qualification')
    (results/'matrix-plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    (results/'cases.tsv').write_text(''.join(f"{case['name']}\t{case['ranks']}\n" for case in cases))
    print(json.dumps(plan, indent=2), flush=True)


def assess(records):
    transports = {record['transport'] for record in records}
    if len(transports) != 1 or not transports <= {'euler', 'constant', 'sutherland'}:
        raise RuntimeError('CNS matrix requires one common transport model')
    regimes = {record.get('regime', 'transonic') for record in records}
    if len(regimes) != 1:
        raise RuntimeError('CNS matrix requires one common Mach regime')
    regime = next(iter(regimes))
    mach_evidence = regime_evidence(regime)
    studies = []
    groups = dict.fromkeys((record['flux'], record['order']) for record in records)
    for flux, degree in groups:
        group = [r for r in records if (r['flux'], r['order']) == (flux, degree)]
        main = sorted((r for r in group if not r['time_control']), key=lambda r: r['h'], reverse=True)
        controls = [r for r in group if r['time_control']]
        count = main[0]['mesh_levels'] if main else 0
        if (count < 3 or len(main) != count or len(controls) != 1
                or any(r['mesh_levels'] != count for r in group)
                or controls[0]['h'] != main[-1]['h'] or controls[0]['dt'] != main[-1]['dt']/2.
                or any(main[i]['h'] <= main[i + 1]['h'] for i in range(len(main) - 1))):
            raise RuntimeError('CNS matrix missing declared refinement levels or finest half-dt control')
        error = np.array([headline_l2(record) for record in main])
        if not np.all(np.isfinite(error)) or np.any(error <= 0.):
            raise RuntimeError('CNS matrix requires positive finite physical errors')
        ratios = np.array([a['h']/b['h'] for a, b in zip(main, main[1:])])
        orders = np.log(error[:-1]/error[1:])/np.log(ratios)[:, None]
        component_error = np.array([record['component_physical']['l2'] for record in main])
        component_orders = np.log(component_error[:-1]/component_error[1:])/np.log(ratios)[:, None]
        change = np.abs(np.array(headline_l2(controls[0]))/error[-1] - 1.)
        isolation = bool(np.all(np.isfinite(change)) and np.all(change < .01))
        order_met = bool(np.all(np.isfinite(orders)) and np.all(orders[-1] >= degree + .5))
        studies.append(dict(flux=flux, degree=degree, expected_order=degree + 1,
                            cells=[r['cells'] for r in main], refinement_ratios=ratios.tolist(),
                            observed_orders=orders.tolist(), observed_component_orders=component_orders.tolist(),
                            monotone=bool(np.all(orders > 0.)), nominal_minus_half_met=order_met,
                            finest_timestep_relative_change=change.tolist(), timestep_isolation_passed=isolation))
    controls_passed = all(study['timestep_isolation_passed'] for study in studies)
    needs_review = any(not study['nominal_minus_half_met'] or not study['monotone'] for study in studies)
    status = ('failed_controls' if not controls_passed else
              'completed_with_order_flags' if needs_review else 'completed')
    return dict(regime=regime, mach_evidence=mach_evidence, status=status, scope='observational_survey_control_checks',
                quantities=['density', 'momentum', 'energy'], studies=studies,
                controls_passed=controls_passed, passed=controls_passed, accuracy_qualified=False,
                order_expectations_met=all(study['nominal_minus_half_met'] for study in studies),
                needs_order_review=needs_review)


def check(results, device):
    (results/'matrix-assessment.json').unlink(missing_ok=True)
    records = []
    for case in json.loads((results/'cases.json').read_text()):
        record, _ = read_case(results, case, device, quadrature_points=16,
                              check_solver_norm=False, final_time=case['final_time'])
        records.append(record)
        (results/'measurements.json').write_text(json.dumps(records, indent=2) + '\n')
    report = assess(records)
    (results/'matrix-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['controls_passed']:
        raise RuntimeError('CNS matrix timestep isolation unresolved; inspect retained observations')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cns_flux_study import plot as plot_fluxes
    records = json.loads((results/'measurements.json').read_text())
    # Keep each degree's flux comparison readable, using the established plotter.
    for degree in dict.fromkeys(record['order'] for record in records):
        directory = results/f'plots-p{degree}'
        directory.mkdir(exist_ok=True)
        subset = [record for record in records if record['order'] == degree]
        (directory/'measurements.json').write_text(json.dumps(subset, indent=2) + '\n')
        plot_fluxes(directory)
    fluxes = tuple(dict.fromkeys(record['flux'] for record in records))
    columns = min(3, len(fluxes))
    rows = math.ceil(len(fluxes)/columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5*columns, 4*rows), squeeze=False,
                                constrained_layout=True)
    for ax, flux in zip(axes.flat, fluxes):
        for degree in dict.fromkeys(record['order'] for record in records):
            group = sorted((r for r in records if r['flux'] == flux and r['order'] == degree
                            and not r['time_control']), key=lambda r: r['h'], reverse=True)
            h = np.array([r['h'] for r in group])
            density = np.array([r['component_physical']['l2'][0] for r in group])
            line, = ax.loglog(h, density, 'o-', label=f'P{degree}')
            ax.loglog(h, density[-1]*(h/h[-1])**(degree + 1), '--', color=line.get_color(), alpha=.6)
        ax.set(title=flux, xlabel='Element width', ylabel='Physical density L2 error')
        ax.legend()
        ax.grid(True, which='both', alpha=.3)
    for ax in list(axes.flat)[len(fluxes):]:
        ax.set_visible(False)
    figure.suptitle(f"{records[0]['transport']} {records[0].get('regime', 'transonic')} MMS: density by degree\nDashed h^(p+1) references", fontsize=11)
    figure.savefig(results/'density-all-orders.png', dpi=180)
    figure.savefig(results/'density-all-orders.pdf')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'plot'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--transport', choices=('euler', 'constant', 'sutherland'), required=True)
    parser.add_argument('--regime', choices=REGIMES, default='transonic')
    parser.add_argument('--fluxes', default='all')
    parser.add_argument('--degrees', default='1,2,3,4,5')
    parser.add_argument('--cells', default='12,18,27,40,60,90',
                        help='Increasing cells per direction, or geometric for legacy doubling selectors')
    parser.add_argument('--mesh-levels', type=int, default=5)
    parser.add_argument('--start-level', type=int, default=1)
    parser.add_argument('--base-cells', type=int, default=3)
    parser.add_argument('--reference-dt', type=float, default=.001)
    parser.add_argument('--final-time', type=float, default=.2)
    parser.add_argument('--ranks', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source.resolve(), args.results.resolve(), args.transport, select_fluxes(args.fluxes),
                args.ranks, select_degrees(args.degrees), args.mesh_levels, args.start_level,
                args.base_cells, args.reference_dt, args.final_time,
                None if args.cells == 'geometric' else tuple(int(n) for n in args.cells.split(',')), args.regime)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())
