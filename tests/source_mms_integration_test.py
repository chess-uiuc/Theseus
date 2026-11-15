#!/usr/bin/env python3
"""Uniform manufactured energy solution with independent RK3 convergence checks."""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from integration_support import run_simulation, work_directory
from numflux_regression_test import cell_local_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    source = args.source.resolve()
    base = json.loads((source/'TestCases/LTE/Euler/LTEVortex/config.json').read_text())
    runtime = base['runTime']
    runtime.update(gas_model='cpg', flow_model='euler', numerical_flux='lfr',
                   mesh_file=str(source/'TestCases/LTE/Euler/LTEVortex/LTEVortex.mesh'),
                   order=2, ser_ref_levels=0, variable_dt=False, final_time=.001,
                   visualize=True, paraview=True, visit=False, vis_steps=10000,
                   nancheck=True, clock_simulation=False, print_interval=1,
                   R_gas=1., gamma=1.4, mu=0., checkpoint_save=False,
                   conditions={'initial_conditions': {'state': {
                       'density': 1., 'pressure': 1., 'velocity': [0., 0.]}}},
                   visualization={'fields': ['density', 'velocity', 'pressure',
                                             'specific_total_energy']},
                   sources=[{'type': 'affine', 'state': [0., 0., 0., 1000.],
                             'time': [0., 0., 0., 1e6]}])
    # E'=1000E+10^6 t, E(0)=1/(gamma-1); spatial flux divergence is zero.
    exact = math.exp(1.)/.4 + math.expm1(1.) - 1.
    fields = ('Density', 'Velocity', 'Pressure', 'Specific Total Energy')
    with work_directory(prefix='theseus-source-mms-') as directory:
        root = Path(directory)

        def run(name, steps, model, device, ranks=1, spatial=False):
            config = copy.deepcopy(base)
            rt = config['runTime']
            rt.update(dt=.001/steps, nsteps_max=steps, flow_model=model,
                      output_file_path=str(root/name/'solution'))
            if spatial:
                rt.update(dt=1e-4, final_time=1e-4, nsteps_max=1,
                          ode_solver='ForwardEuler',
                          sources=[{'type': 'affine', 'gradient': [
                              [0., 0., 0., 2.], [0., 0., 0., -3.]]}])
            path = root/(name+'.json')
            path.write_text(json.dumps(config, indent=2))
            result = run_simulation(args.executable, path, ranks, device, timeout=120)
            assert result.returncode == 0, result.stdout
            assert f'Device configuration: {device}' in result.stdout, result.stdout
            pvd = root/name/'solution/ParaView/ParaView.pvd'
            frames = ET.parse(pvd).findall('.//DataSet')
            data = cell_local_fields(pvd.parent/frames[-1].attrib['file'], fields)
            assert np.allclose(data['Density'], 1., rtol=0., atol=1e-11)
            assert np.max(np.abs(data['Velocity'])) < 1e-10
            assert np.allclose(data['Pressure'], .4*data['Specific Total Energy'],
                               rtol=1e-11, atol=1e-11)
            if spatial:
                coordinates = data['coordinates'][:, 3:]
                expected = 2.5 + 1e-4*(2.*coordinates[:, 0] - 3.*coordinates[:, 1])
                assert np.allclose(data['Specific Total Energy'], expected,
                                   rtol=1e-11, atol=1e-11), name
                print(f'{name}: physical-coordinate source PASS', flush=True)
                return data, 0.
            error = float(np.max(np.abs(data['Specific Total Energy'] - exact)))
            print(f'{name}: error={error:.16g}', flush=True)
            return data, error

        for model in ('euler', 'cns'):
            errors = [run(f'{model}-{steps}', steps, model, 'cpu')[1]
                      for steps in (8, 16, 32)]
            orders = [math.log(errors[i]/errors[i+1], 2) for i in (0, 1)]
            assert all(2.8 < order < 3.2 for order in orders), (model, errors, orders)
            print(f'{model}: RK3 convergence orders={orders}; PASS', flush=True)
        for model in ('euler', 'cns'):
            run(model+'-spatial', 1, model, args.device, ranks=2, spatial=True)
        serial, _ = run('serial-parity', 16, 'cns', 'cpu')
        parallel, _ = run('mpi-parity', 16, 'cns', 'cpu', ranks=2)
        for field in fields:
            assert np.allclose(serial[field], parallel[field], rtol=1e-10, atol=1e-10), field
        # A second physical consumer checks momentum drag and its energy work
        # through the same RHS interface, against an independent uniform solution.
        def drag_case(device):
            config = copy.deepcopy(base)
            rt = config['runTime']
            rt.update(dt=.001/64, nsteps_max=64, flow_model='cns',
                      output_file_path=str(root/('drag-'+device)/'solution'),
                      sources=[{'type': 'linear-drag', 'coefficient': 1000.}])
            rt['conditions']['initial_conditions']['state']['velocity'] = [.4, -.2]
            path = root/('drag-'+device+'.json')
            path.write_text(json.dumps(config, indent=2))
            result = run_simulation(args.executable, path, 2, device, timeout=120)
            assert result.returncode == 0, result.stdout
            assert f'Device configuration: {device}' in result.stdout
            pvd = root/('drag-'+device)/'solution/ParaView/ParaView.pvd'
            frames = ET.parse(pvd).findall('.//DataSet')
            data = cell_local_fields(pvd.parent/frames[-1].attrib['file'], fields)
            assert np.allclose(data['Density'], 1., rtol=0., atol=1e-11)
            expected_velocity = np.array([.4, -.2])*math.exp(-1.)
            velocity_error = float(np.max(np.abs(data['Velocity'][:, :2] - expected_velocity)))
            expected_energy = 2.5 + .1*math.exp(-2.)
            energy_error = float(np.max(np.abs(data['Specific Total Energy'] - expected_energy)))
            # At 64 RK3 steps, O(dt^3) temporal error is well below 1e-6.
            assert velocity_error < 1e-6, velocity_error
            assert energy_error < 1e-6, energy_error
            print(f'Drag {device}: velocity error={velocity_error:.16g}; energy error={energy_error:.16g}; PASS', flush=True)
            return data

        drag_cpu = drag_case('cpu')
        if args.device == 'cuda':
            drag_gpu = drag_case('cuda')
            for field in fields:
                assert np.allclose(drag_cpu[field], drag_gpu[field], rtol=1e-10, atol=1e-10), field
            for model in ('euler', 'cns'):
                cpu, _ = run(model+'-cpu', 16, model, 'cpu')
                gpu, _ = run(model+'-cuda', 16, model, 'cuda')
                for field in fields:
                    assert np.allclose(cpu[field], gpu[field], rtol=1e-10, atol=1e-10), field
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
