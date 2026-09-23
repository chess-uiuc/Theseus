#!/usr/bin/env python3
"""CPG profile BC: quiescent-state preservation and heated serial/MPI startup."""
import argparse
import base64
import importlib.util
import json
import math
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def values(node):
    s=''.join(node.text.split())
    # MFEM writes the UInt32 byte count and payload as separate base64 blocks.
    size=struct.unpack('<I',base64.b64decode(s[:8]))[0]
    raw=base64.b64decode(s[8:])
    if len(raw)!=size:
        raise RuntimeError('Unexpected VTK binary array length')
    return struct.unpack('<'+'d'*(size//8),raw)


def stats(output):
    pvd=output/'ParaView/ParaView.pvd'
    last=ET.parse(pvd).findall('.//DataSet')[-1]
    pvtu=pvd.parent/last.attrib['file']
    fields={k:[] for k in ['Density','Pressure','Velocity']}
    axis=[]
    for piece in ET.parse(pvtu).findall('.//Piece'):
        root=ET.parse(pvtu.parent/piece.attrib['Source'])
        xyz=values(root.find('.//Points/DataArray'))
        arrays={a.attrib['Name']:values(a) for a in root.findall('.//PointData/DataArray')}
        for k in fields: fields[k].extend(arrays[k])
        v=arrays['Velocity'];width=len(v)//(len(xyz)//3)
        axis += [abs(v[width*i+1]) for i in range(len(xyz)//3) if abs(xyz[3*i+1])<1e-13]
    rho,p=fields['Density'],fields['Pressure']
    T=[b/(287.05*a) for a,b in zip(rho,p)]
    result=[min(rho),max(rho),min(p),max(p),min(T),max(T),max(axis)]
    if not all(math.isfinite(v) for v in result) or min(result[:6])<=0:
        raise RuntimeError(f'Invalid thermodynamic state: {result}')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True);p.add_argument('--mpiexec',required=True)
    a=p.parse_args();case=a.source/'TestCases/Axisymmetric/NavierStokes/PXChamber'
    spec=importlib.util.spec_from_file_location('px_case',case/'run_case.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[]
    with tempfile.TemporaryDirectory(prefix='px-chamber-test-') as tmp:
        for uniform,ranks in [(True,1),(False,1),(False,2)]:
            out=Path(tmp)/f'{uniform}-{ranks}'
            cfg=module.prepare(out,True,2e-7,1e-8)
            c=json.loads(cfg.read_text());r=c['runTime']
            if uniform:
                profile=out/'uniform.dat';profile.write_text('2 1\n0 0 300 0 0 0\n1 0 300 0 0 0\n')
                r['conditions']['boundary_conditions']['Inflow']['file']=str(profile)
            cfg.write_text(json.dumps(c))
            run=subprocess.run([a.mpiexec,'-n',str(ranks),str(a.executable),'-d','cpu','-c',str(cfg)],
                               cwd=out,capture_output=True,text=True,timeout=120)
            if run.returncode: raise RuntimeError(run.stdout+run.stderr)
            result=stats(out)
            if uniform:
                expected=[10000/(287.05*300)]*2+[10000]*2+[300]*2+[0]
                for x,y in zip(result,expected):
                    if not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-9):
                        raise RuntimeError(f'Uniform state changed: {result}')
            else:
                if result[5]<301: raise RuntimeError('Inlet did not heat the chamber')
                # Axis reflection is weak: interior nodal ur need not be identically zero.
                # Exact parity of the numerical exterior state is checked by kernel tests.
                results.append(result)
        for x,y in zip(*results):
            if not math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-8):
                raise RuntimeError(f'Serial/MPI mismatch: {results}')
    print('Quiescent preservation, heated profile, positive states, and MPI agreement passed')

if __name__=='__main__': main()
