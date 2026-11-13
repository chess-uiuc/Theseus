#!/usr/bin/env python3
"""Small Gmsh2.2 channel: periodic x/z, walls at y=-1,+1; no external mesher."""
import argparse
import math
from pathlib import Path


def write_mesh(path, nx=4, ny=4, nz=4):
    if min(nx, ny, nz) < 2:
        raise ValueError('Use at least two cells in each direction')
    lx, lz = 4*math.pi, 4*math.pi/3
    node = lambda i, j, k: 1+i+(nx+1)*(j+(ny+1)*k)
    elements = []
    def quad(tag, indices):
        elements.append((3, tag, indices))
    for k in range(nz):
        for j in range(ny):
            for i, tag in ((0, 1), (nx, 2)):
                quad(tag, [node(i,j,k), node(i,j+1,k), node(i,j+1,k+1), node(i,j,k+1)])
    for k in range(nz):
        for i in range(nx):
            for j, tag in ((0, 3), (ny, 4)):
                quad(tag, [node(i,j,k), node(i,j,k+1), node(i+1,j,k+1), node(i+1,j,k)])
    for j in range(ny):
        for i in range(nx):
            for k, tag in ((0, 5), (nz, 6)):
                quad(tag, [node(i,j,k), node(i+1,j,k), node(i+1,j+1,k), node(i,j+1,k)])
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append((5, 7, [node(i,j,k), node(i+1,j,k), node(i+1,j+1,k), node(i,j+1,k),
                                        node(i,j,k+1), node(i+1,j,k+1), node(i+1,j+1,k+1), node(i,j+1,k+1)]))
    lines = ['$MeshFormat', '2.2 0 8', '$EndMeshFormat', '$PhysicalNames', '7']
    for tag, name in enumerate(('XMin', 'XMax', 'BottomWall', 'TopWall', 'ZMin', 'ZMax'), 1):
        lines.append(f'2 {tag} "{name}"')
    lines.extend(['3 7 "Fluid"', '$EndPhysicalNames', '$Nodes', str((nx+1)*(ny+1)*(nz+1))])
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                lines.append(f'{node(i,j,k)} {lx*i/nx:.17g} {-1+2*j/ny:.17g} {lz*k/nz:.17g}')
    lines.extend(['$EndNodes', '$Elements', str(len(elements))])
    for eid, (kind, tag, indices) in enumerate(elements, 1):
        lines.append(f'{eid} {kind} 2 {tag} {tag} ' + ' '.join(map(str, indices)))
    lines.extend(['$EndElements', '$Periodic', '2', '2 1 2',
                  f'Affine 1 0 0 {lx:.17g} 0 1 0 0 0 0 1 0 0 0 0 1', str((ny+1)*(nz+1))])
    for k in range(nz+1):
        for j in range(ny+1):
            lines.append(f'{node(0,j,k)} {node(nx,j,k)}')
    lines.extend(['2 5 6', f'Affine 1 0 0 0 0 1 0 0 0 0 1 {lz:.17g} 0 0 0 1', str((nx+1)*(ny+1))])
    for j in range(ny+1):
        for i in range(nx+1):
            lines.append(f'{node(i,j,0)} {node(i,j,nz)}')
    lines.append('$EndPeriodic')
    Path(path).write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('channel_4x4x4.msh'))
    args = parser.parse_args()
    write_mesh(args.output)
