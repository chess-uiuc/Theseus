#!/usr/bin/env python3
"""Prepare, validate and explicitly record the short unforced channel regression."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from numflux_regression_test import cell_local_fields

STEPS = 50
CYCLES = (0, 25, 50)
DT = 2.5e-6
FIELDS = ('Density', 'Pressure', 'Temperature', 'Velocity', 'Specific Total Energy',
          'Viscosity', 'Thermal Conductivity')
RTOL = ATOL = 1e-10


def transport(build):
    cache = (Path(build)/'CMakeCache.txt').read_text()
    value = re.search(r'^SUTHERLAND:BOOL=(.*)$', cache, re.MULTILINE)
    if value is None:
        raise ValueError('Build cache must identify SUTHERLAND explicitly')
    if value[1].upper() in ('ON', 'YES', 'TRUE', '1'):
        return 'sutherland'
    if value[1].upper() in ('OFF', 'NO', 'FALSE', '0'):
        return 'tabulated'
    raise ValueError('Unrecognized SUTHERLAND build setting')


def prepare(source, build, database, results):
    case = source/'TestCases/LTE/NavierStokes/ChannelFlowRegression'
    config = json.loads((case/'config.json').read_text())
    config['runTime'].update(database_path=str(database.resolve()),
                            mesh_file=str((case/'channel_4x4x4.msh').resolve()),
                            output_file_path=str((results/'solution').resolve()))
    results.mkdir(parents=True, exist_ok=False)
    (results/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (results/'transport.json').write_text(json.dumps({'model': transport(build)}, indent=2)+'\n')


def frames(results, device):
    log = (results/'run.log').read_text()
    devices = re.findall(r'Device configuration:\s*([^\n]+)', log)
    if not devices or device not in devices[-1].strip().split(',') or (device == 'cpu' and devices[-1].strip() != 'cpu'):
        raise AssertionError(f'Requested {device}; actual backend: {devices}')
    times = {0: 0.0}
    for cycle, value in re.findall(r'time step:\s*(\d+), time:\s*([^,\s]+)', log):
        times[int(cycle)] = float(value)
    if max(times) != STEPS:
        raise AssertionError(f'Expected exactly {STEPS} steps, found {max(times)}')
    pvd = results/'solution/ParaView/ParaView.pvd'
    entries = ET.parse(pvd).findall('.//DataSet')
    if len(entries) != len(CYCLES):
        raise AssertionError('Require initial, middle and final frames')
    data = []
    for cycle, entry in zip(CYCLES, entries):
        expected_time = cycle*DT
        if not math.isclose(times.get(cycle, float('nan')), expected_time, rel_tol=1e-12, abs_tol=1e-15):
            raise AssertionError(f'Incorrect physical time for cycle {cycle}')
        if Path(entry.attrib['file']).parts[0] != f'Cycle{cycle:06d}':
            raise AssertionError('Unexpected output cycle')
        data.append(cell_local_fields(pvd.parent/entry.attrib['file'], FIELDS))
    if np.linalg.norm(data[-1]['Velocity']-data[0]['Velocity']) <= 1e-8*np.linalg.norm(data[0]['Velocity']):
        raise AssertionError('Velocity did not evolve meaningfully')
    diagnostics = re.findall(r'TotalChange: Mass:\s*([^,]+), Energy:\s*([^,]+), K.E.:\s*(\S+)', log)
    if len(diagnostics) != STEPS:
        raise AssertionError('Require integral diagnostics for every step')
    history = np.asarray(diagnostics, dtype=float)
    if not np.all(np.isfinite(history)) or np.max(np.abs(history[:, 0])) > 1e-11:
        raise AssertionError('Nonfinite diagnostics or failed mass conservation')
    seconds = float((results/'wall-seconds.txt').read_text())
    if not math.isfinite(seconds) or seconds > 600:
        raise AssertionError(f'Run exceeded ten-minute limit: {seconds}')
    return data, history, seconds


def compare(reference, actual, rtol=RTOL, atol=ATOL):
    report = {}
    if reference.keys() != actual.keys():
        raise AssertionError('Reference/actual fields differ')
    failed = []
    for name in reference:
        left, right = reference[name], actual[name]
        if left.shape != right.shape or not np.all(np.isfinite(right)):
            raise AssertionError(f'Invalid field shape/values: {name}')
        delta = np.abs(left-right)
        report[name] = {'max_absolute': float(np.max(delta)),
                        'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(left), 1e-300))}
        tolerance = 1e-12 if name == 'coordinates' else atol+rtol*np.abs(left)
        if name == 'cell_types':
            ok = np.array_equal(left, right)
        else:
            ok = np.all(delta <= tolerance)
        if not ok:
            failed.append(name)
    return report, failed


def check(source, build, results, device):
    data, history, seconds = frames(results, device)
    model = transport(build)
    golden = source/'TestCases/GoldenData/ChannelFlow'
    metadata = json.loads((golden/f'{model}.json').read_text())
    if metadata['cycles'] != list(CYCLES) or metadata['dt'] != DT:
        raise AssertionError('Golden schedule differs from test')
    report = {'transport': model, 'seconds': seconds, 'cycles': {}, 'status': 'PASS'}
    with np.load(golden/f'{model}.npz', allow_pickle=False) as reference:
        for cycle, actual in zip(CYCLES, data):
            expected = {name: reference[f'cycle{cycle}_{name}'] for name in actual}
            metrics, failed = compare(expected, actual)
            report['cycles'][str(cycle)] = {'fields': metrics, 'failed_fields': failed}
            if failed:
                report['status'] = 'FAIL'
        report['integrals'] = {'max_absolute_difference': np.max(np.abs(history-reference['integral_history']), axis=0).tolist(),
                               'final_actual': history[-1].tolist(),
                               'final_reference': reference['integral_history'][-1].tolist()}
        # Diagnostics are printed to six significant digits; allow two printed units.
        if not np.allclose(history[:, 1:], reference['integral_history'][:, 1:], rtol=2e-6, atol=1e-12):
            report['status'] = 'FAIL'
            report['integral_comparison'] = 'FAIL'
    (results/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        raise AssertionError('Channel golden comparison failed; reference was not updated')


def record(source, build, results, revision):
    # Explicit development command only. A failed regression never calls this.
    data, history, seconds = frames(results, 'cpu')
    ranks = re.findall(r'Theseus Simulation with (\d+) MPI ranks', (results/'run.log').read_text())
    if not ranks or ranks[-1] != '1':
        raise AssertionError('Golden recording requires a single-rank CPU run')
    model = transport(build)
    golden = source/'TestCases/GoldenData/ChannelFlow'
    golden.mkdir(parents=True, exist_ok=True)
    target = golden/f'{model}.npz'
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError('Refusing to replace existing golden data')
    arrays = {f'cycle{cycle}_{name}': value for cycle, frame in zip(CYCLES, data) for name, value in frame.items()}
    arrays['integral_history'] = history
    with target.open('xb') as output:
        np.savez_compressed(output, **arrays)
    metadata = {'reference_revision': revision, 'transport': model, 'cycles': CYCLES,
                'dt': DT, 'cpu_seconds': seconds, 'ranks': 1,
                'config': json.loads((results/'config.json').read_text()),
                'executable_sha256': hashlib.sha256((build/'theseus').read_bytes()).hexdigest(),
                'cache_sha256': hashlib.sha256((build/'CMakeCache.txt').read_bytes()).hexdigest()}
    case = source/'TestCases/LTE/NavierStokes/ChannelFlowRegression'
    metadata['case_sources_sha256'] = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in [case/'config.json', case/'channel_4x4x4.msh',
                                                    source/'include/ChannelFlowInitialCondition.hpp']}
    metadata['config']['runTime'].pop('database_path')
    metadata['config']['runTime'].pop('output_file_path')
    metadata['config']['runTime']['mesh_file'] = 'channel_4x4x4.msh'
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Recorded explicit CPU regression baseline: {target}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'record'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--database', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--reference-revision')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.database is None:
            parser.error('prepare requires --database')
        prepare(args.source, args.build, args.database, args.results)
    elif args.action == 'record':
        if not args.reference_revision:
            parser.error('record requires --reference-revision')
        record(args.source, args.build, args.results, args.reference_revision)
    else:
        check(args.source, args.build, args.results, args.device)


if __name__ == '__main__':
    main()
