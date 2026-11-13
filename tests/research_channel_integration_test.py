#!/usr/bin/env python3
"""Short forced/unforced research-channel checks against the same initial field."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
from numflux_regression_test import cell_local_fields

FIELDS = ('Density', 'Pressure', 'Temperature', 'Velocity')

def run_case(args, root, flux, forced, device):
    directory = root / f'{flux}-{forced}-{device}'
    directory.mkdir()
    source = args.source.resolve()
    case = source / 'TestCases/LTE/NavierStokes/ChannelFlowRegression'
    config = json.loads((case / 'config.json').read_text())
    runtime = config['runTime']
    runtime.update(nsteps_max=2, final_time=5e-6, vis_steps=1, clock_simulation=False,
                   numerical_flux=flux, channel_forcing=forced,
                   database_path=str(args.database.resolve()),
                   mesh_file=str(case / 'channel_4x4x4.msh'),
                   output_file_path=str(directory / 'solution'))
    runtime['gas_model'] = 'lte'
    runtime['solver'] = runtime.pop('plato_solver')
    runtime['conditions']['initial_conditions']['function'] = 'StreamwiseParabolicIC'
    path = directory / 'config.json'
    path.write_text(json.dumps(config, indent=2))
    log = directory / 'run.log'
    with log.open('w') as output:
        subprocess.run(['bash', str(source / 'scripts/run_theseus.sh'), '-e',
            str(args.executable.resolve()), '-c', str(path), '-o', str(directory / 'launch'),
            '-r', device, '-p', '1', '-P', '-R', '-k'], cwd=source,
            stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)
    text = log.read_text()
    devices = re.findall(r'Device configuration:\s*([^\n]+)', text)
    assert devices and device in devices[-1].strip().split(','), (device, devices)
    if device == 'cpu':
        assert devices[-1].strip() == 'cpu', devices
    assert ('Research channel forcing: enabled' in text) == forced, log
    assert 'time step: 2,' in text, log
    pvd = directory / 'solution/ParaView/ParaView.pvd'
    frames = ET.parse(pvd).findall('.//DataSet')
    assert len(frames) == 3, pvd
    times = {int(c): float(value) for c, value in re.findall(r'time step:\s*(\d+), time:\s*([^,\s]+)', text)}
    for cycle in (1, 2):
        assert np.isclose(times[cycle], cycle*2.5e-6, rtol=1e-12, atol=1e-15), times
    return [cell_local_fields(pvd.parent / frame.attrib['file'], FIELDS) for frame in frames]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='theseus-research-channel-'))
    print(f'Research channel results: {root}', flush=True)
    for flux in ('LTEECFlux', 'Pirozzoli'):
        off = run_case(args, root, flux, False, 'cpu')
        on = run_case(args, root, flux, True, 'cpu')
        for field in FIELDS:
            assert np.array_equal(off[0][field], on[0][field]), f'Initial state differs: {field}'
        delta = np.linalg.norm(on[-1]['Velocity'] - off[-1]['Velocity'])
        assert delta > 1e-8*np.linalg.norm(off[-1]['Velocity']), 'Forcing has no measurable effect'
        if args.device == 'cuda':
            gpu = run_case(args, root, flux, True, 'cuda')
            for left, right in zip(on, gpu):
                for field in FIELDS:
                    assert np.allclose(left[field], right[field], rtol=1e-10, atol=1e-10), (flux, field)
        print(f'{flux}: finite forced/unforced runs; source effect={delta:.16g}; PASS', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
