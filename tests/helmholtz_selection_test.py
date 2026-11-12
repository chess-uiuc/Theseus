#!/usr/bin/env python3
"""Check typed factory selection and the entropy-API rejection on two ranks."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

from integration_support import run_simulation, work_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--available', type=int, choices=(0, 1), required=True)
    parser.add_argument('--plato', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    case = args.source / 'TestCases/NavierStokes/3D/TaylorGreenVortex'
    spec = importlib.util.spec_from_file_location('tgv_mesh', case / 'make_periodic_box.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    with work_directory(prefix='helmholtz-selection-') as temporary:
        root = Path(temporary)
        mesh = root / 'periodic.msh'
        generator.write_mesh(mesh, 2*math.pi, 2*math.pi, 2*math.pi, 4, 4, 4)
        base = json.loads((case / 'config.json').read_text())
        base['runTime'].update(
            order=1, mesh_file=str(mesh), ser_ref_levels=0, gas_mixture='air5',
            plato_solver='LTE_table_rhoT_(air5)', N_rho=11, N_T=21,
            rho_min=0.5, rho_max=2.0, T_min=200, T_max=2000,
            rho_dist='log', T_dist='log', flow_model='cns',
            nsteps_max=1, final_time=1, dt=1e-7, variable_dt=False,
            visualize=False, paraview=False, visit=False, check_cfl=False,
        )
        base['runTime']['conditions'] = {'initial_conditions': {
            'signature': 1, 'function': 'CompressibleTaylorGreenVortexIC',
            'params': {'x1': 0.2}}}
        # The default generated configuration carries its prefix's database.
        choices = [('lte', 'LTEECFlux'), ('lte_helmholtz', 'Pirozzoli')]
        if args.plato:
            choices.append(('lte', 'LLF'))
        for index, (model, flux) in enumerate(choices):
            runtime = base['runTime']
            runtime.update(gas_model=model, numerical_flux=flux,
                           output_file_path=str(root / f'output-{index}'))
            path = root / f'case-{index}.json'
            path.write_text(json.dumps(base, indent=2) + '\n')
            result = run_simulation(args.executable, path, 2, 'cpu')
            legacy = model == 'lte' and flux == 'LLF'
            if legacy or args.available:
                expected = 'GasModel:  LTE:' if legacy else 'GasModel:  LTE-Helmholtz:'
                if result.returncode or expected not in result.stdout or f"NumFlux:   {'LFR' if legacy else flux}" not in result.stdout:
                    raise AssertionError(result.stdout)
                if legacy and 'Constructing Helmholtz' in result.stdout:
                    raise AssertionError('Existing LTE selection silently changed EOS')
            elif result.returncode == 0 or 'requires PLATO with plato_get_species_entropy' not in result.stdout:
                raise AssertionError('Missing API did not cause the required explicit rejection:\n' + result.stdout)
            print(f'{model}/{flux}: expected selection/rejection PASS', flush=True)


if __name__ == '__main__':
    main()
