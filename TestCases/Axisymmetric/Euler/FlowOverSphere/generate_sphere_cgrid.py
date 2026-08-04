#!/usr/bin/env python3
"""Generate an all-quadrilateral axisymmetric C-grid around a sphere.

The mesh lies in the meridional (z,r) plane and is written to Gmsh as

    x = z   (axial coordinate)
    y = r   (radial coordinate)
    z = 0

The mesh has two structured zones sharing a conforming interface:

1. A sphere-normal boundary-layer collar.
2. A transition zone from the outer collar semicircle to a rectangular
   inlet/farfield/outlet boundary.

The symmetry axis is r=0 from the inlet to the upstream sphere pole and from
its downstream pole to the outlet.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path

PHYS_INLET = 1
PHYS_OUTLET = 2
PHYS_AXIS = 3
PHYS_SPHERE = 4
PHYS_FARFIELD = 5
PHYS_FLUID = 6


@dataclass(frozen=True)
class Point:
    z: float
    r: float


def geometric_ratio(n: int, first: float, total: float) -> float:
    """Solve first * (1 + q + ... + q**(n-1)) = total for q >= 1."""
    if n < 1:
        raise ValueError("n must be positive")
    if first <= 0.0 or total <= 0.0:
        raise ValueError("first spacing and total thickness must be positive")
    if first * n > total * (1.0 + 1.0e-13):
        raise ValueError(
            "first-layer spacing is too large for the requested collar thickness"
        )
    if abs(first * n - total) <= 1.0e-13 * total:
        return 1.0

    def series(q: float) -> float:
        if abs(q - 1.0) < 1.0e-12:
            return first * n
        return first * (q**n - 1.0) / (q - 1.0)

    lo, hi = 1.0, 1.05
    while series(hi) < total:
        hi *= 1.2
        if hi > 20.0:
            raise RuntimeError("could not bracket geometric growth ratio")
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if series(mid) < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def collar_radii(radius: float, thickness: float, n: int, first: float) -> tuple[list[float], float]:
    q = geometric_ratio(n, first, thickness)
    radii = [radius]
    h = first
    for _ in range(n):
        radii.append(radii[-1] + h)
        h *= q
    radii[-1] = radius + thickness
    return radii, q


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def outer_segment_counts(nt: int, zmin: float, zmax: float, rmax: float) -> tuple[int, int, int]:
    """Allocate boundary edges proportionally while pinning both rectangle corners."""
    axial = zmax - zmin
    perimeter = axial + 2.0 * rmax
    nin = max(1, round(nt * rmax / perimeter))
    nout = max(1, round(nt * rmax / perimeter))
    ntop = nt - nin - nout
    if ntop < 1:
        raise ValueError("nt is too small to represent all three rectangular sides")
    return nin, ntop, nout


def outer_path_index(
    i: int, nt: int, zmin: float, zmax: float, rmax: float
) -> tuple[Point, int]:
    """Return outer-boundary node i; rectangle corners are exact mesh nodes."""
    nin, ntop, nout = outer_segment_counts(nt, zmin, zmax, rmax)
    if i <= nin:
        return Point(zmin, rmax * i / nin), PHYS_INLET
    if i <= nin + ntop:
        k = i - nin
        return Point(zmin + (zmax - zmin) * k / ntop, rmax), PHYS_FARFIELD
    k = i - nin - ntop
    return Point(zmax, rmax * (1.0 - k / nout)), PHYS_OUTLET


def outer_edge_physical(i: int, nt: int, zmin: float, zmax: float, rmax: float) -> int:
    nin, ntop, _ = outer_segment_counts(nt, zmin, zmax, rmax)
    if i < nin:
        return PHYS_INLET
    if i < nin + ntop:
        return PHYS_FARFIELD
    return PHYS_OUTLET


def signed_quad_area(a: Point, b: Point, c: Point, d: Point) -> float:
    pts = (a, b, c, d)
    area2 = 0.0
    for k, p0 in enumerate(pts):
        p1 = pts[(k + 1) % 4]
        area2 += p0.z * p1.r - p1.z * p0.r
    return 0.5 * area2


def node(i: int, j: int, nt: int) -> int:
    return j * (nt + 1) + i + 1


def generate(
    output: Path,
    radius: float,
    center_z: float,
    zmin: float,
    zmax: float,
    rmax: float,
    nt: int,
    nbl: int,
    nouter: int,
    first_layer: float,
    collar_thickness: float,
    outer_clustering: float,
) -> None:
    if radius <= 0.0:
        raise ValueError("sphere radius must be positive")
    if not (zmin < center_z - radius and center_z + radius < zmax):
        raise ValueError("sphere must lie strictly inside the axial domain")
    if rmax <= radius + collar_thickness:
        raise ValueError("rmax must exceed radius + collar_thickness")
    if nt < 8 or nbl < 1 or nouter < 1:
        raise ValueError("require nt >= 8, nbl >= 1, and nouter >= 1")
    if collar_thickness <= 0.0:
        raise ValueError("collar thickness must be positive")
    if outer_clustering <= 0.0:
        raise ValueError("outer clustering exponent must be positive")

    radii, growth = collar_radii(radius, collar_thickness, nbl, first_layer)
    nrad = nbl + nouter
    coords: list[list[Point]] = [[Point(0.0, 0.0) for _ in range(nt + 1)] for _ in range(nrad + 1)]

    # Uniform angular points give uniform tangential spacing on the sphere.
    theta = [pi * (1.0 - i / nt) for i in range(nt + 1)]

    # Exact sphere-normal boundary-layer collar.
    for j, rho in enumerate(radii):
        for i, th in enumerate(theta):
            coords[j][i] = Point(center_z + rho * cos(th), rho * sin(th))

    # Ruled/transfinite transition from the outer collar to the rectangular C boundary.
    collar_outer = radii[-1]
    for jj in range(1, nouter + 1):
        raw = jj / nouter
        # Power controls clustering at the collar; smoothstep removes endpoint slope jumps.
        eta = smoothstep(raw**outer_clustering)
        j = nbl + jj
        for i, th in enumerate(theta):
            inner = Point(center_z + collar_outer * cos(th), collar_outer * sin(th))
            outer, _ = outer_path_index(i, nt, zmin, zmax, rmax)
            coords[j][i] = Point(
                inner.z + eta * (outer.z - inner.z),
                inner.r + eta * (outer.r - inner.r),
            )

    min_area = float("inf")
    max_area = 0.0
    for j in range(nrad):
        for i in range(nt):
            area = signed_quad_area(
                coords[j][i], coords[j][i + 1], coords[j + 1][i + 1], coords[j + 1][i]
            )
            if area <= 1.0e-16:
                raise RuntimeError(
                    f"folded or degenerate cell ({i}, {j}), signed area={area:.8e}; "
                    "increase collar thickness, increase nouter, or reduce outer clustering"
                )
            min_area = min(min_area, area)
            max_area = max(max_area, area)

    lines = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$PhysicalNames", "6",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "axis"',
        '1 4 "sphere"', '1 5 "farfield"', '2 6 "fluid"',
        "$EndPhysicalNames",
        "$Nodes", str((nt + 1) * (nrad + 1)),
    ]

    # Gmsh x is physical z; Gmsh y is physical r.
    for j in range(nrad + 1):
        for i in range(nt + 1):
            p = coords[j][i]
            lines.append(f"{node(i, j, nt)} {p.z:.16g} {p.r:.16g} 0")
    lines.append("$EndNodes")

    boundary_count = 2 * nrad + 2 * nt
    volume_count = nt * nrad
    lines.extend(["$Elements", str(boundary_count + volume_count)])
    element = 1

    # Upstream axis: rectangular inlet corner to upstream sphere pole.
    for j in range(nrad, 0, -1):
        lines.append(
            f"{element} 1 2 {PHYS_AXIS} {PHYS_AXIS} "
            f"{node(0, j, nt)} {node(0, j - 1, nt)}"
        )
        element += 1

    # Downstream axis: downstream sphere pole to rectangular outlet corner.
    for j in range(nrad):
        lines.append(
            f"{element} 1 2 {PHYS_AXIS} {PHYS_AXIS} "
            f"{node(nt, j, nt)} {node(nt, j + 1, nt)}"
        )
        element += 1

    # Sphere wall.
    for i in range(nt):
        lines.append(
            f"{element} 1 2 {PHYS_SPHERE} {PHYS_SPHERE} "
            f"{node(i, 0, nt)} {node(i + 1, 0, nt)}"
        )
        element += 1

    # Rectangular outer C boundary, with physical tags determined segment-by-segment.
    for i in range(nt):
        physical = outer_edge_physical(i, nt, zmin, zmax, rmax)
        lines.append(
            f"{element} 1 2 {physical} {physical} "
            f"{node(i + 1, nrad, nt)} {node(i, nrad, nt)}"
        )
        element += 1

    # Fluid quadrilaterals.
    for j in range(nrad):
        for i in range(nt):
            lines.append(
                f"{element} 3 2 {PHYS_FLUID} {PHYS_FLUID} "
                f"{node(i, j, nt)} {node(i + 1, j, nt)} "
                f"{node(i + 1, j + 1, nt)} {node(i, j + 1, nt)}"
            )
            element += 1

    lines.append("$EndElements")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    wall_ds = pi * radius / nt
    print(f"wrote                 : {output}")
    print(f"coordinate convention : x=z, y=r, z=0")
    print(f"cells                 : {nt * nrad}")
    print(f"sphere cells          : {nt}")
    print(f"boundary-layer cells  : {nbl}")
    print(f"outer radial cells    : {nouter}")
    print(f"first wall spacing    : {first_layer:.8g}")
    print(f"collar growth ratio   : {growth:.8g}")
    print(f"wall tangential ds    : {wall_ds:.8g}")
    print(f"minimum cell area     : {min_area:.8g}")
    print(f"maximum cell area     : {max_area:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=Path("FlowOverSphere_CGrid.msh"))
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--center-z", type=float, default=0.0)
    parser.add_argument("--zmin", type=float, default=-1.0)
    parser.add_argument("--zmax", type=float, default=4.0)
    parser.add_argument("--rmax", type=float, default=1.0)
    parser.add_argument("--nt", type=int, default=500, help="cells along the sphere")
    parser.add_argument("--nbl", type=int, default=40, help="sphere-normal collar cells")
    parser.add_argument("--nouter", type=int, default=210, help="collar-to-farfield cells")
    parser.add_argument("--first-layer", type=float, default=1.0e-4)
    parser.add_argument("--collar-thickness", type=float, default=0.08)
    parser.add_argument(
        "--outer-clustering", type=float, default=0.7,
        help="less than 1 spreads cells away from collar; greater than 1 clusters there",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(
        output=args.output,
        radius=args.radius,
        center_z=args.center_z,
        zmin=args.zmin,
        zmax=args.zmax,
        rmax=args.rmax,
        nt=args.nt,
        nbl=args.nbl,
        nouter=args.nouter,
        first_layer=args.first_layer,
        collar_thickness=args.collar_thickness,
        outer_clustering=args.outer_clustering,
    )


if __name__ == "__main__":
    main()
