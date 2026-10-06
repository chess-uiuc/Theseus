#!/usr/bin/env python3
"""Physical IC -> selected EOS -> conservative checkpoint, and restart bypass."""
import argparse
import array
import copy
import json
import re
import shutil
import struct
from pathlib import Path

from integration_support import run_simulation, work_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--database', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    base = json.loads((args.source/'TestCases/LTE/Euler/LTEVortex/config.json').read_text())
    rt = base['runTime']
    rt.update(mesh_file=str(args.source/'TestCases/LTE/Euler/LTEVortex/LTEVortex.mesh'),
              database_path=args.database, order=2, ser_ref_levels=0, N_rho=25, N_T=25,
              rho_min=0.05, rho_max=1.1, T_min=250., T_max=3000.,
              visualize=False, variable_dt=False, dt=1e-8, final_time=1.,
              nsteps_max=2, print_interval=1, checkpoint_save=True,
              checkpoint_dt=1e-8, clock_simulation=False)
    rt['conditions']={'initial_conditions':{'state':{'pressure':60000.,'temperature':1200.,'velocity':[10.,-2.]}}}
    with work_directory(prefix='theseus-physical-ic-') as tmp:
        root=Path(tmp)
        def run(name, config, ranks=2, error=None):
            output=root/name
            output.mkdir(exist_ok=True)
            config=copy.deepcopy(config)
            config['runTime']['output_file_path']=str(output)
            path=root/(name+'.json'); path.write_text(json.dumps(config))
            result = run_simulation(args.executable, path, ranks, args.device, timeout=90)
            log=result.stdout+result.stderr
            if error:
                if result.returncode==0 or error not in log:
                    raise AssertionError(f'{name}: expected {error!r}\n{log}')
            elif result.returncode:
                raise RuntimeError(f'{name}: {result.returncode}\n{log}')
            return output,log

        def checkpoint(output, cycle, rank):
            return output/f'Checkpoints/Cycle{cycle}/checkpoint_cycle_{cycle}.{rank:08d}.chk'

        def read_state(path):
            with path.open('rb') as stream:
                assert stream.readline()==b'THESEUS_CHECKPOINT_RAW_V1\n'
                size=struct.unpack('=Q',stream.read(8))[0]
                data=array.array('d'); data.frombytes(stream.read())
                assert len(data)==size and size%4==0
                return data,size//4

        energies={}
        for model in ('cpg','lte'):
            config=copy.deepcopy(base); config['runTime']['gas_model']=model
            output,log=run(model,config)
            matches=re.findall(r'rho\(([^,]+),([^\)]+)\), p\(([^,]+),([^\)]+)\), T\(([^,]+),([^\)]+)\)',log)
            assert matches,log
            for values in matches:
                values=list(map(float,values))
                assert all(abs(p-60000)<1 for p in values[2:4]),values
                assert all(abs(t-1200)<0.02 for t in values[4:6]),values
            state,n=read_state(checkpoint(output,2,0))
            rho=state[0]; energies[model]=state[3*n]
            for i in range(n):
                assert abs(state[i]-rho)<1e-10
                assert abs(state[n+i]/state[i]-10)<1e-9
                assert abs(state[2*n+i]/state[i]+2)<1e-9
            if model=='cpg':
                assert abs(rho-60000/(287.05*1200))<1e-12
                assert abs(energies[model]-(60000/0.4+0.5*rho*104))<1e-7
            for pair_name,pair in (
                ('rhoT',dict(density=rho,temperature=1200.)),
                ('rhoP',dict(density=rho,pressure=60000.))):
                alternative=copy.deepcopy(config)
                alternative['runTime']['conditions']['initial_conditions']={'state':dict(pair,velocity=[10.,-2.])}
                other,_=run(model+'-'+pair_name,alternative)
                other_state,other_n=read_state(checkpoint(other,2,0))
                assert abs(other_state[3*other_n]/energies[model]-1)<1e-9
            # Invalid unused state proves restart neither parses nor converts IC.
            restart=copy.deepcopy(config)
            restart['runTime'].update(checkpoint_load=True,checkpoint_cycle=1)
            restart['runTime']['conditions']['initial_conditions']={'state':{'temperature':-10}}
            dest=root/(model+'-restart'); dest.mkdir()
            shutil.copytree(output/'Checkpoints/Cycle1',dest/'Checkpoints/Cycle1')
            resumed,_=run(model+'-restart',restart)
            for rank in range(2):
                assert checkpoint(output,2,rank).read_bytes()==checkpoint(resumed,2,rank).read_bytes()
            # Also exercise 1-rank execution on a device-selected run.
            run(model+'-serial',config,1)
        assert abs(energies['cpg']-energies['lte'])>0.001*abs(energies['cpg'])

        for name,profile in (
          ('blob',dict(type='thermal-blob',radius=.5,pressure=60000,ambient_temperature=1200,peak_temperature=1500)),
          ('vortex',dict(type='vortex',radius=.5,speed=10,strength=.2,density=.2,temperature=1200,shape_gamma=1.4,shape_gas_constant=287.05))):
            for model in ('cpg','lte'):
                config=copy.deepcopy(base); config['runTime']['gas_model']=model
                config['runTime']['conditions']['initial_conditions']={'physical_profile':profile}
                output,_=run(model+'-'+name,config)
                data,n=read_state(checkpoint(output,2,0))
                assert max(data[:n])-min(data[:n])>1e-9

        for name,state,message in (
          ('missing',dict(pressure=60000,velocity=[0,0]),'exactly two'),
          ('extra',dict(pressure=60000,temperature=1200,density=.2,velocity=[0,0]),'exactly two'),
          ('invalid-species',dict(pressure=60000,temperature=1200,velocity=[0,0],species=[1]),'unsupported field'),
          ('bounds',dict(density=.001,temperature=1200,velocity=[0,0]),'outside')):
            config=copy.deepcopy(base)
            config['runTime']['conditions']['initial_conditions']={'state':state}
            run(name,config,error=message)
        config=copy.deepcopy(base)
        config['runTime']['conditions']['initial_conditions']={'physical_profile':dict(type='thermal-blob',radius=.5,pressure=60000,ambient_temperature=1200,peak_temperature=5000)}
        run('bad-profile',config,error='physical_profile at (')
    print('Physical IC conversions, profiles, rejection and conservative restart: PASS')

if __name__=='__main__':
    main()
