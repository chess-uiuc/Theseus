#!/usr/bin/env python3
import argparse
import math

def factor_scale(scale):
    """
    Factor an integer total element-count scale into directional multipliers.

    The product sx*sy*sz equals scale.

    Rule:
      - Factor scale into primes.
      - Assign each factor to the currently least-scaled direction.
      - Tie-break direction order is x, then y, then z.

    This gives:
      2  -> (2,1,1)
      4  -> (2,2,1)
      5  -> (5,1,1)
      6  -> (3,2,1)
      8  -> (2,2,2)
      16 -> (4,2,2)
    """

    if scale < 1:
        raise ValueError("scale must be a positive integer")

    factors = []
    n = scale

    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1 if p == 2 else 2

    if n > 1:
        factors.append(n)

    # Assign larger factors first. This gives better aspect-ratio balance
    # for scales like 12 -> (3,2,2), 18 -> (3,3,2), etc.
    factors.sort(reverse=True)

    mult = [1, 1, 1]

    for f in factors:
        # choose currently smallest multiplier;
        # tie-breaker is x, then y, then z
        d = min(range(3), key=lambda q: (mult[q], q))
        mult[d] *= f

    return tuple(mult)

SURF = {
    "xmin": 1,
    "xmax": 2,
    "ymin": 3,
    "ymax": 4,
    "zmin": 5,
    "zmax": 6,
}

VOL_TAG = 7


def fmt(x):
    # Avoid ugly -0.000000000000000e+00
    if abs(x) < 1.0e-15:
        x = 0.0
    return f"{x:.17g}"


def node_id(i, j, k, nx, ny, nz):
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)


def hex_nodes(i, j, k, nx, ny, nz):
    n000 = node_id(i,     j,     k,     nx, ny, nz)
    n100 = node_id(i + 1, j,     k,     nx, ny, nz)
    n110 = node_id(i + 1, j + 1, k,     nx, ny, nz)
    n010 = node_id(i,     j + 1, k,     nx, ny, nz)

    n001 = node_id(i,     j,     k + 1, nx, ny, nz)
    n101 = node_id(i + 1, j,     k + 1, nx, ny, nz)
    n111 = node_id(i + 1, j + 1, k + 1, nx, ny, nz)
    n011 = node_id(i,     j + 1, k + 1, nx, ny, nz)

    return [n000, n100, n110, n010, n001, n101, n111, n011]


def quad(eid, tag, nodes):
    # Gmsh type 3 = 4-node quadrangle
    # Two tags: physical tag, elementary/geometrical tag
    return f"{eid} 3 2 {tag} {tag} " + " ".join(map(str, nodes))


def hex_el(eid, nodes):
    # Gmsh type 5 = 8-node hexahedron
    return f"{eid} 5 2 {VOL_TAG} {VOL_TAG} " + " ".join(map(str, nodes))


def write_mesh(path, dx, dy, dz, nx, ny, nz):
    xs = [dx * i / nx for i in range(nx + 1)]
    ys = [dy * j / ny for j in range(ny + 1)]
    zs = [dz * k / nz for k in range(nz + 1)]

    num_nodes = (nx + 1) * (ny + 1) * (nz + 1)

    elements = []
    eid = 1

    # Boundary quads.
    #
    # The exact orientation usually does not matter for MFEM periodic
    # node-pair identification, but these are chosen consistently.
    for k in range(nz):
        for j in range(ny):
            # xmin
            nodes = [
                node_id(0, j,     k,     nx, ny, nz),
                node_id(0, j + 1, k,     nx, ny, nz),
                node_id(0, j + 1, k + 1, nx, ny, nz),
                node_id(0, j,     k + 1, nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["xmin"], nodes))
            eid += 1

            # xmax
            nodes = [
                node_id(nx, j,     k,     nx, ny, nz),
                node_id(nx, j,     k + 1, nx, ny, nz),
                node_id(nx, j + 1, k + 1, nx, ny, nz),
                node_id(nx, j + 1, k,     nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["xmax"], nodes))
            eid += 1

    for k in range(nz):
        for i in range(nx):
            # ymin
            nodes = [
                node_id(i,     0, k,     nx, ny, nz),
                node_id(i,     0, k + 1, nx, ny, nz),
                node_id(i + 1, 0, k + 1, nx, ny, nz),
                node_id(i + 1, 0, k,     nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["ymin"], nodes))
            eid += 1

            # ymax
            nodes = [
                node_id(i,     ny, k,     nx, ny, nz),
                node_id(i + 1, ny, k,     nx, ny, nz),
                node_id(i + 1, ny, k + 1, nx, ny, nz),
                node_id(i,     ny, k + 1, nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["ymax"], nodes))
            eid += 1

    for j in range(ny):
        for i in range(nx):
            # zmin
            nodes = [
                node_id(i,     j,     0, nx, ny, nz),
                node_id(i + 1, j,     0, nx, ny, nz),
                node_id(i + 1, j + 1, 0, nx, ny, nz),
                node_id(i,     j + 1, 0, nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["zmin"], nodes))
            eid += 1

            # zmax
            nodes = [
                node_id(i,     j,     nz, nx, ny, nz),
                node_id(i,     j + 1, nz, nx, ny, nz),
                node_id(i + 1, j + 1, nz, nx, ny, nz),
                node_id(i + 1, j,     nz, nx, ny, nz),
            ]
            elements.append(quad(eid, SURF["zmax"], nodes))
            eid += 1

    # Volume hexes.
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                elements.append(hex_el(eid, hex_nodes(i, j, k, nx, ny, nz)))
                eid += 1

    with open(path, "w") as f:
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")

        f.write("$Nodes\n")
        f.write(f"{num_nodes}\n")
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    nid = node_id(i, j, k, nx, ny, nz)
                    f.write(f"{nid} {fmt(x)} {fmt(y)} {fmt(z)}\n")
        f.write("$EndNodes\n")

        f.write("$Elements\n")
        f.write(f"{len(elements)}\n")
        for line in elements:
            f.write(line + "\n")
        f.write("$EndElements\n")

        # Periodic surface maps.
        #
        # Format:
        #   dim slaveTag masterTag
        #   Affine ...
        #   numNodePairs
        #   slaveNode masterNode
        #
        # The affine map sends the low-side surface to the high-side surface.
        f.write("$Periodic\n")
        f.write("3\n")

        # x-periodic: xmin -> xmax
        f.write(f"2 {SURF['xmin']} {SURF['xmax']}\n")
        f.write(
            "Affine "
            f"1 0 0 {fmt(dx)} "
            "0 1 0 0 "
            "0 0 1 0 "
            "0 0 0 1\n"
        )
        f.write(f"{(ny + 1) * (nz + 1)}\n")
        for k in range(nz + 1):
            for j in range(ny + 1):
                f.write(
                    f"{node_id(0, j, k, nx, ny, nz)} "
                    f"{node_id(nx, j, k, nx, ny, nz)}\n"
                )

        # y-periodic: ymin -> ymax
        f.write(f"2 {SURF['ymin']} {SURF['ymax']}\n")
        f.write(
            "Affine "
            "1 0 0 0 "
            f"0 1 0 {fmt(dy)} "
            "0 0 1 0 "
            "0 0 0 1\n"
        )
        f.write(f"{(nx + 1) * (nz + 1)}\n")
        for k in range(nz + 1):
            for i in range(nx + 1):
                f.write(
                    f"{node_id(i, 0, k, nx, ny, nz)} "
                    f"{node_id(i, ny, k, nx, ny, nz)}\n"
                )

        # z-periodic: zmin -> zmax
        f.write(f"2 {SURF['zmin']} {SURF['zmax']}\n")
        f.write(
            "Affine "
            "1 0 0 0 "
            "0 1 0 0 "
            f"0 0 1 {fmt(dz)} "
            "0 0 0 1\n"
        )
        f.write(f"{(nx + 1) * (ny + 1)}\n")
        for j in range(ny + 1):
            for i in range(nx + 1):
                f.write(
                    f"{node_id(i, j, 0, nx, ny, nz)} "
                    f"{node_id(i, j, nz, nx, ny, nz)}\n"
                )

        f.write("$EndPeriodic\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structured triply-periodic Gmsh 2.2 hex mesh."
    )
    parser.add_argument("-o", "--output", default="periodic_box.msh")
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dy", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("-s", "--scale", type=int, default=1)

    args = parser.parse_args()

    if args.nx <= 0 or args.ny <= 0 or args.nz <= 0:
        raise ValueError("nx, ny, nz must all be positive")

    sx, sy, sz = factor_scale(args.scale)

    dx = args.dx * sx
    dy = args.dy * sy
    dz = args.dz * sz

    nx = args.nx * sx
    ny = args.ny * sy
    nz = args.nz * sz

    print("Base mesh:")
    print(f"  DX,DY,DZ = {args.dx}, {args.dy}, {args.dz}")
    print(f"  NX,NY,NZ = {args.nx}, {args.ny}, {args.nz}")
    print(f"  elements = {args.nx * args.ny * args.nz}")
    print("")
    print("Scale:")
    print(f"  scale    = {args.scale}")
    print(f"  sx,sy,sz = {sx}, {sy}, {sz}")
    print("")
    print("Generated mesh:")
    print(f"  DX,DY,DZ = {dx}, {dy}, {dz}")
    print(f"  NX,NY,NZ = {nx}, {ny}, {nz}")
    print(f"  elements = {nx * ny * nz}")

    write_mesh(args.output, dx, dy, dz, nx, ny, nz)

if __name__ == "__main__":
    main()
