#!/usr/bin/env python3
"""Selected-EOS boundary packing exercised by axisymmetric CNS on one/two ranks."""
import argparse
import array
import copy
import json
import math
from pathlib import Path

from integration_support import run_simulation, work_directory
import re
import struct


def rectangle(path):
    nx, ny = 4, 4
    node = lambda i, j: 1 + j*(nx+1) + i
    elements = []
    def element(kind, tag, nodes):
        elements.append(f'{len(elements)+1} {kind} 2 {tag} {tag} ' + ' '.join(map(str,nodes)))
    for j in range(ny):
        for i in range(nx):
            element(3,1,[node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)])
    for i in range(nx):
        element(1,2,[node(i,0),node(i+1,0)])
        element(1,5,[node(i+1,ny),node(i,ny)])
    for j in range(ny):
        element(1,3,[node(0,j+1),node(0,j)])
        element(1,4,[node(nx,j),node(nx,j+1)])
    lines=['$MeshFormat','2.2 0 8','$EndMeshFormat','$PhysicalNames','5',
           '2 1 "Domain"','1 2 "Axis"','1 3 "Inflow"','1 4 "Right"','1 5 "Top"',
           '$EndPhysicalNames','$Nodes',str((nx+1)*(ny+1))]
    lines += [f'{node(i,j)} {1+i/nx} {j/ny} 0' for j in range(ny+1) for i in range(nx+1)]
    lines += ['$EndNodes','$Elements',str(len(elements)),*elements,'$EndElements']
    path.write_text('\n'.join(lines)+'\n')


def state(output, cycle=3):
    path=output/f'Checkpoints/Cycle{cycle}/checkpoint_cycle_{cycle}.00000000.chk'
    with path.open('rb') as stream:
        assert stream.readline()==b'THESEUS_CHECKPOINT_RAW_V1\n'
        size=struct.unpack('=Q',stream.read(8))[0]
        values=array.array('d'); values.frombytes(stream.read())
        assert len(values)==size and size%4==0
        return values,size//4


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--database',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    base=json.loads((args.source/'TestCases/Axisymmetric/NavierStokes/PXChamber/config.json').read_text())
    with work_directory(prefix='theseus-boundaries-') as temporary:
        root=Path(temporary)
        mesh=root/'rectangle.msh'; rectangle(mesh)
        uniform=root/'uniform.dat'; uniform.write_text('2 1\n0 0 1200 0 0 0\n1 0 1200 0 0 0\n')
        # No axis row: also exercise the existing even/odd axis extension.
        heated=root/'heated.dat'; heated.write_text('2 1\n0.2 0 1600 10 2 0\n1 0 1200 -1 -2 0\n')
        physical={'pressure':60000.,'temperature':1200.,'velocity':[0.,0.]}
        runtime=base['runTime']
        runtime.update(mesh_file=str(mesh),order=2,ser_ref_levels=0,
                       visualize=False,variable_dt=False,dt=1e-7,final_time=1.,nsteps_max=3,
                       print_interval=1,checkpoint_save=True,checkpoint_dt=1e-7,
                       database_path=args.database,gas_mixture='air5',solver='LTE_table_rhoT_(air5)',
                       N_rho=25,N_T=25,rho_min=.05,rho_max=1.1,T_min=250.,T_max=3000.,
                       rho_dist='log',T_dist='log')
        runtime['conditions']={'initial_conditions':{'state':physical},'boundary_conditions':{
            'Axis':{'type':'axis'},
            'Inflow':{'type':'radial-profile','file':str(uniform),'pressure':60000.},
            'Right':{'type':'exterior-state','state':physical},
            'Top':{'type':'exterior-state','state':physical}}}

        def run(name,config,ranks=2,error=None):
            output=root/name; output.mkdir()
            config=copy.deepcopy(config); config['runTime']['output_file_path']=str(output)
            file=output/'config.json'; file.write_text(json.dumps(config))
            result = run_simulation(args.executable, file, ranks, args.device, timeout=90)
            log=result.stdout+result.stderr
            if error is not None:
                assert result.returncode != 0 and error in log, (name,log)
                assert 'Boundary' in log,log
                return
            if result.returncode: raise RuntimeError(f'{name}\n{log}')
            matches=re.findall(r'rho\(([^,]+),([^\)]+)\), p\(([^,]+),([^\)]+)\), T\(([^,]+),([^\)]+)\)',log)
            assert matches,log
            ranges=[list(map(float,values)) for values in matches]
            assert all(math.isfinite(v) and v>0 for row in ranges for v in row),ranges
            return output,ranges

        energies={}; heated_states={}
        for model in ('cpg','lte'):
            config=copy.deepcopy(base); config['runTime']['gas_model']=model
            for ranks in (1,2):
                output,ranges=run(f'{model}-uniform-{ranks}',config,ranks)
                for values in ranges:
                    assert all(abs(v-60000)<1 for v in values[2:4]),values
                    assert all(abs(v-1200)<.02 for v in values[4:6]),values
                values,n=state(output); energies[model]=values[3*n]
                for i in range(n):
                    assert abs(values[n+i])<1e-8 and abs(values[2*n+i])<1e-8
            hot=copy.deepcopy(config)
            hot['runTime']['conditions']['boundary_conditions']['Inflow']['file']=str(heated)
            hot_ranges=[]
            for ranks in (1,2):
                output,ranges=run(f'{model}-heated-{ranks}',hot,ranks)
                assert ranges[-1][5]>1200.001,ranges
                hot_ranges.append(ranges[-1])
                if ranks==1: heated_states[model]=state(output)[0]
            for a,b in zip(*hot_ranges):
                assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),hot_ranges
        assert abs(energies['cpg']-energies['lte'])>.001*abs(energies['cpg'])

        legacy=copy.deepcopy(base); legacy['runTime']['gas_model']='cpg'
        boundaries=legacy['runTime']['conditions']['boundary_conditions']
        boundaries['Inflow']={'type':'cpg-radial-profile','file':str(heated),'pressure':60000.}
        for name in ('Right','Top'):
            boundaries[name]={'type':'cpg-exterior-state','pressure':60000.,'temperature':1200.}
        output,_=run('legacy-cpg',legacy,1)
        values,_=state(output)
        for a,b in zip(values,heated_states['cpg']):
            assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)

        invalid=root/'invalid.dat'; invalid.write_text('2 1\n0 0 5000 0 0 0\n1 0 5000 0 0 0\n')
        short=root/'short.dat'; short.write_text('2 1\n0 0 1200 0 0 0\n0.1 0 1200 0 0 0\n')
        for name,profile,message in [('bounds',invalid,"'Inflow' at radius"),('coverage',short,'outside profile coverage')]:
            config=copy.deepcopy(base); config['runTime']['gas_model']='lte'
            config['runTime']['conditions']['boundary_conditions']['Inflow']['file']=str(profile)
            run(name,config,error=message)
        config=copy.deepcopy(base); config['runTime']['gas_model']='lte'
        config['runTime']['conditions']['boundary_conditions']['Top']['state']={'pressure':60000.,'velocity':[0,0]}
        run('missing-temperature',config,error='exactly two')
        config=copy.deepcopy(base); config['runTime']['gas_model']='lte'
        config['runTime']['conditions']['boundary_conditions']['Inflow']['file']=str(root/'missing.dat')
        run('missing-file',config,error='Cannot read profile')
    print('Physical constant/profile boundaries, LTE/CPG energy, legacy parity, MPI and rejection: PASS')


if __name__=='__main__':
    main()
