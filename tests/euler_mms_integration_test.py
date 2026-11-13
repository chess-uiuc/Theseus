#!/usr/bin/env python3
"""Periodic forced CPG Euler: spatial refinement, temporal isolation and controls."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from integration_support import run_simulation, work_directory
from numflux_regression_test import cell_local_fields


def exact_fields(coordinates, time):
    # Independent primitive-state reference; no production helper is imported.
    sine = np.sin(2.*np.pi*(coordinates[:, 0] + coordinates[:, 1] - .4*time))
    rho = 2. + .2*sine
    u = .7 + .1*sine
    v = .3 - .05*sine
    pressure = 1. + .15*sine
    energy = pressure/.4 + .5*rho*(u*u + v*v)
    return np.column_stack((rho, rho*u, rho*v, energy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    source = args.source.resolve()
    base = json.loads((source/'TestCases/Euler/2D/PeriodicMMS/config.json').read_text())
    runtime = base['runTime']
    runtime.update(gas_model='cpg', flow_model='euler', numerical_flux='lfr',
                   mesh_file=str(source/'TestCases/LTE/Euler/LTEVortex/LTEVortex.mesh'),
                   periodic=True, order=2, variable_dt=False, final_time=.2,
                   visualize=True, paraview=True, visit=False, vis_steps=10000,
                   nancheck=True, clock_simulation=False, print_interval=1,
                   R_gas=1., gamma=1.4, mu=0., checkpoint_save=False, alpha_max=0.,
                   ode_solver='RK4', precision=17,
                   conditions={'initial_conditions': {'function': 'PeriodicEulerMMS', 'signature': 0}},
                   exact_solution={'function': 'PeriodicEulerMMS', 'signature': 0},
                   visualization={'fields': ['density', 'velocity', 'pressure',
                                             'specific_total_energy'], 'mesh_mode': 'gll_subcells'},
                   sources=[{'type': 'periodic-euler-mms'}])
    with work_directory(prefix='theseus-euler-mms-') as directory:
        root = Path(directory)
        records = []

        def run(name, level, dt=.001, forced=True, device='cpu', ranks=1):
            config = copy.deepcopy(base)
            rt = config['runTime']
            rt.update(dt=dt, nsteps_max=round(.2/dt), ser_ref_levels=level,
                      output_file_path=str(root/name/'solution'))
            if not forced:
                rt['sources'] = []
            path = root/(name+'.json')
            path.write_text(json.dumps(config, indent=2))
            result = run_simulation(args.executable, path, ranks, device, timeout=240)
            assert result.returncode == 0, result.stdout
            assert f'Device configuration: {device}' in result.stdout
            match = re.search(r'Exact L2 Error:\s*(\S+)', result.stdout)
            assert match, result.stdout
            l2 = float(match[1])
            assert math.isfinite(l2) and l2 > 0.
            pvd = root/name/'solution/ParaView/ParaView.pvd'
            frames = ET.parse(pvd).findall('.//DataSet')
            assert len(frames) >= 2, frames
            assert float(frames[0].attrib['timestep']) == 0.
            assert abs(float(frames[-1].attrib['timestep']) - .2) < 1e-13
            times = re.findall(r'time step:\s*\d+, time:\s*([^,\s]+)', result.stdout)
            assert times and abs(float(times[-1]) - .2) < 1e-13, times
            fields = ('Density', 'Velocity', 'Pressure', 'Specific Total Energy')
            initial = cell_local_fields(pvd.parent/frames[0].attrib['file'], fields)
            final = cell_local_fields(pvd.parent/frames[-1].attrib['file'], fields)
            def conservative(data):
                rho = data['Density'].reshape(-1)
                velocity = data['Velocity']
                return np.column_stack((rho, rho*velocity[:, 0], rho*velocity[:, 1],
                                        rho*data['Specific Total Energy'].reshape(-1)))
            initial_error = np.max(np.abs(conservative(initial)
                - exact_fields(initial['coordinates'][:, 3:], 0.)))
            assert initial_error < 1e-11, initial_error
            errors = conservative(final) - exact_fields(final['coordinates'][:, 3:], .2)
            component_rms = np.sqrt(np.mean(errors*errors, axis=0))
            records.append(dict(name=name, level=level, dt=dt, ranks=ranks, device=device,
                                forced=forced, l2=l2, component_rms=component_rms.tolist()))
            (root/'measurements.json').write_text(json.dumps(records, indent=2))
            print(f'{name}: L2={l2:.16g}; independent conservative RMS={component_rms}', flush=True)
            return l2, component_rms, final

        refinements = [run(f'level-{level}', level) for level in (1, 2, 3)]
        orders = [math.log(refinements[i][0]/refinements[i+1][0], 2) for i in (0, 1)]
        # P2 DG on smooth Cartesian data should approach order p+1=3.
        assert all(2.5 < order < 3.5 for order in orders), orders
        for equation in range(4):
            component_order = math.log(refinements[1][1][equation]/refinements[2][1][equation], 2)
            assert 2.5 < component_order < 3.5, (equation, component_order)
        half_dt = run('fine-half-dt', 3, dt=.0005)
        assert abs(half_dt[0]/refinements[-1][0] - 1.) < .01, 'Temporal error contaminates spatial study'
        unforced = run('source-off', 3, forced=False)
        assert unforced[0] > 5.*refinements[-1][0], 'Source-off control did not distinguish forcing'
        parallel = run('mpi-parity', 3, ranks=2)
        for field in ('Density', 'Velocity', 'Pressure', 'Specific Total Energy'):
            assert np.allclose(parallel[2][field], refinements[-1][2][field],
                               rtol=1e-10, atol=1e-10), field
        if args.device == 'cuda':
            gpu = run('cuda-parity', 3, device='cuda')
            for field in ('Density', 'Velocity', 'Pressure', 'Specific Total Energy'):
                assert np.allclose(gpu[2][field], refinements[-1][2][field],
                                   rtol=1e-10, atol=1e-10), field
        (root/'convergence.json').write_text(json.dumps({'orders': orders, 'runs': records}, indent=2))
        print(f'Periodic CPG Euler MMS: spatial orders={orders}; PASS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
