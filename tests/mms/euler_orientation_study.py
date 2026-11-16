#!/usr/bin/env python3
"""Matched cyclic orientations of one manufactured Euler solution on a periodic cube."""
from mms_paths import REPOSITORY_ROOT
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from trigonometric_euler_study import exact_fields, read_case, FINAL_TIME, FIELDS


def oriented_exact(coordinates, time, orientation):
    dimension = coordinates.shape[1]
    mapping = (np.arange(dimension) + orientation) % dimension
    canonical = exact_fields(coordinates[:, mapping], time)
    physical = canonical.copy()
    physical[:, 1 + mapping] = canonical[:, 1:dimension + 1]
    return physical


def canonical_fields(fields, orientation):
    mapping = (np.arange(3) + orientation) % 3
    keys = fields['coordinates'][:, np.r_[mapping, 3 + mapping]]
    order = np.lexsort(np.round(keys, 12).T[::-1])
    canonical = {'coordinates': keys[order]}
    for name in FIELDS:
        values = fields[name]
        if name == 'Velocity':
            values = values[:, mapping]
        canonical[name] = values[order]
    return canonical


def compare_fields(reference, candidate):
    if reference['coordinates'].shape != candidate['coordinates'].shape:
        raise RuntimeError('orientation mesh/trace counts differ')
    if not np.allclose(reference['coordinates'], candidate['coordinates'], rtol=0., atol=1e-12):
        raise RuntimeError('mapped orientation coordinates disagree')
    differences = {}
    for name in FIELDS:
        if not np.allclose(reference[name], candidate[name], rtol=1e-10, atol=1e-10):
            raise RuntimeError(f'mapped orientation fields disagree: {name}')
        differences[name] = float(np.max(np.abs(reference[name] - candidate[name])))
    return differences


def prepare(source, results, device, ranks=1):
    results.mkdir(parents=True, exist_ok=True)
    mesh_script = source/'TestCases/NavierStokes/3D/TaylorGreenVortex/make_periodic_box.py'
    spec = importlib.util.spec_from_file_location('periodic_box', mesh_script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    mesh = results/'periodic.msh'
    generator.write_mesh(mesh, 1., 1., 1., 6, 6, 6)
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    cases = []
    for orientation in range(3):
        name = f'orientation-{orientation}'
        directory = results/name
        directory.mkdir()
        config = copy.deepcopy(base)
        runtime = config['runTime']
        runtime.update(dim=3, num_equations=5, order=2, ser_ref_levels=1,
                       mesh_file=str(mesh), dt=.004, nsteps_max=52, final_time=FINAL_TIME,
                       initial_save_dt=FINAL_TIME, refined_save_dt=FINAL_TIME,
                       output_file_path=str(directory/'solution'),
                       sources=[dict(type='trigonometric-euler-mms', orientation=orientation)])
        condition = dict(function='OrientedTrigonometricEulerMMS', signature=1,
                         params={'x1': orientation})
        runtime['conditions']['initial_conditions'] = condition
        runtime['exact_solution'] = condition.copy()
        (directory/'config.json').write_text(json.dumps(config, indent=2) + '\n')
        cases.append(dict(name=name, dimension=3, orientation=orientation, h=1./12.,
                          level=1, order=2, dt=.004, ranks=ranks, forced=True, backend=device))
    (results/'cases.json').write_text(json.dumps(cases, indent=2) + '\n')


def check(results, device):
    records = []
    mapped = []
    for case in json.loads((results/'cases.json').read_text()):
        orientation = case['orientation']
        reference = lambda coordinates, time: oriented_exact(coordinates, time, orientation)
        record, fields = read_case(results, case, device, reference)
        records.append(record)
        mapped.append(canonical_fields(fields, orientation))
    differences = [compare_fields(mapped[0], fields) for fields in mapped]
    canonical_norms = []
    for record in records:
        mapping = (np.arange(3) + record['orientation']) % 3
        norms = np.array(record['component_physical']['l2'])
        canonical_norms.append(norms[np.r_[0, 1 + mapping, 4]])
    if not np.allclose(canonical_norms, canonical_norms[0], rtol=1e-9, atol=1e-12):
        raise RuntimeError(f'orientation component norms disagree: {canonical_norms}')
    assessment = dict(passed=True, runs=records, mapped_field_max_abs=differences,
                      canonical_component_l2=np.asarray(canonical_norms).tolist())
    (results/'orientation.json').write_text(json.dumps(assessment, indent=2) + '\n')
    for index, values in enumerate(differences):
        print(f'orientation {index}: mapped field max absolute differences {values}')
    print('Matched cyclic Euler orientations: PASS')


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assessment = json.loads((results/'orientation.json').read_text())
    norms = np.array(assessment['canonical_component_l2'])
    figure, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for component, name in enumerate(('rho', 'rho*u', 'rho*v', 'rho*w', 'rhoE')):
        line, = ax.semilogy(range(3), norms[:, component], 'o-', label=name)
        ax.axhline(norms[0, component], color=line.get_color(), linestyle=':', alpha=.6)
    ax.set(xticks=[0, 1, 2], xlabel='Cyclic axis orientation', ylabel='Canonical component physical L2 error',
           title='Matched solution orientations: expected equal errors')
    ax.legend()
    ax.grid(True, alpha=.3)
    figure.savefig(results/'orientation.png', dpi=180)
    figure.savefig(results/'orientation.pdf')
    plt.close(figure)


def main():
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
        prepare(args.source.resolve(), args.results.resolve(), args.device, args.ranks)
    elif args.action == 'check':
        check(args.results.resolve(), args.device)
    else:
        plot(args.results.resolve())


if __name__ == '__main__':
    main()
