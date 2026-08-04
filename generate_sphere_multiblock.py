#!/usr/bin/env python3
"""Generate a conforming all-quadrilateral axisymmetric sphere mesh.

Coordinate convention in the Gmsh file:

    x = z   axial coordinate
    y = r   radial coordinate
    z = 0

Topology
--------
The upper semicircle of the sphere is divided into four equal angular blocks.
Those blocks terminate on a rectangular near-body interface

    [center_z - 3R, center_z + 3R] x [0, 3R].

The first and fourth sphere quadrants connect to the left and right sides of
that rectangle.  The two middle quadrants connect to its top-left and top-right
halves.  Three additional blocks connect the inner rectangle to the rectangular
farfield boundary.

Each near-body block contains an exact sphere-normal boundary-layer collar,
followed by a smooth transition to the inner rectangle.
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


def lerp(a: Point, b: Point, t: float) -> Point:
    return Point(a.z + t * (b.z - a.z), a.r + t * (b.r - a.r))


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def stretched_fraction(k: int, n: int, exponent: float) -> float:
    if n <= 0:
        raise ValueError("number of intervals must be positive")
    if exponent <= 0.0:
        raise ValueError("clustering exponent must be positive")
    raw = k / n
    return smoothstep(raw**exponent)


def geometric_ratio(n: int, first: float, total: float) -> float:
    """Solve first*(1+q+...+q**(n-1)) = total for q >= 1."""
    if n < 1 or first <= 0.0 or total <= 0.0:
        raise ValueError("invalid geometric-series parameters")
    if first * n > total * (1.0 + 1.0e-13):
        raise ValueError("first-layer spacing is too large for collar thickness")
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
            raise RuntimeError("could not bracket collar growth ratio")
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


def signed_quad_area(a: Point, b: Point, c: Point, d: Point) -> float:
    pts = (a, b, c, d)
    area2 = 0.0
    for i, p0 in enumerate(pts):
        p1 = pts[(i + 1) % 4]
        area2 += p0.z * p1.r - p1.z * p0.r
    return 0.5 * area2


class Mesh:
    def __init__(self) -> None:
        self.points: list[Point] = []
        self.point_ids: dict[tuple[float, float], int] = {}
        self.quads: list[tuple[int, int, int, int]] = []
        self.edges: list[tuple[int, int, int]] = []
        self.min_area = float("inf")
        self.max_area = 0.0

    @staticmethod
    def key(p: Point) -> tuple[float, float]:
        return (round(p.z, 14), round(p.r, 14))

    def add_point(self, p: Point) -> int:
        key = self.key(p)
        if key in self.point_ids:
            return self.point_ids[key]
        node_id = len(self.points) + 1
        self.points.append(p)
        self.point_ids[key] = node_id
        return node_id

    def add_block(self, coords: list[list[Point]], name: str) -> list[list[int]]:
        nv = len(coords) - 1
        nu = len(coords[0]) - 1
        ids = [[self.add_point(coords[j][i]) for i in range(nu + 1)] for j in range(nv + 1)]
        for j in range(nv):
            for i in range(nu):
                a, b = coords[j][i], coords[j][i + 1]
                c, d = coords[j + 1][i + 1], coords[j + 1][i]
                area = signed_quad_area(a, b, c, d)
                if area <= 1.0e-16:
                    raise RuntimeError(
                        f"folded/degenerate cell in {name} at (i={i}, j={j}), area={area:.8e}"
                    )
                self.min_area = min(self.min_area, area)
                self.max_area = max(self.max_area, area)
                self.quads.append((ids[j][i], ids[j][i + 1], ids[j + 1][i + 1], ids[j + 1][i]))
        return ids

    def add_edge_chain(self, chain: list[int], physical: int, reverse: bool = False) -> None:
        nodes = list(reversed(chain)) if reverse else chain
        for a, b in zip(nodes[:-1], nodes[1:]):
            self.edges.append((a, b, physical))


def sphere_point(center_z: float, rho: float, theta: float) -> Point:
    return Point(center_z + rho * cos(theta), rho * sin(theta))


def inner_rectangle_segments(center_z: float, radius: float) -> list[tuple[Point, Point]]:
    zl = center_z - 3.0 * radius
    zr = center_z + 3.0 * radius
    rt = 3.0 * radius
    zc = center_z
    return [
        (Point(zl, 0.0), Point(zl, rt)),
        (Point(zl, rt), Point(zc, rt)),
        (Point(zc, rt), Point(zr, rt)),
        (Point(zr, rt), Point(zr, 0.0)),
    ]


def near_body_block(
    quadrant: int,
    center_z: float,
    radius: float,
    radii: list[float],
    nq: int,
    ntransition: int,
    transition_exponent: float,
    target: tuple[Point, Point],
) -> list[list[Point]]:
    # theta decreases from pi to zero as the surface path moves left-to-right.
    theta0 = pi - quadrant * (pi / 4.0)
    theta1 = theta0 - pi / 4.0
    thetas = [theta0 + (theta1 - theta0) * i / nq for i in range(nq + 1)]

    coords: list[list[Point]] = []
    for rho in radii:
        coords.append([sphere_point(center_z, rho, th) for th in thetas])

    collar_rho = radii[-1]
    for k in range(1, ntransition + 1):
        eta = stretched_fraction(k, ntransition, transition_exponent)
        row: list[Point] = []
        for i, th in enumerate(thetas):
            inner = sphere_point(center_z, collar_rho, th)
            outer = lerp(target[0], target[1], i / nq)
            row.append(lerp(inner, outer, eta))
        coords.append(row)
    return coords


def axis_distribution(a: float, b: float, n: int, exponent: float = 1.0) -> list[float]:
    """Monotone coordinates from a to b, optionally clustered toward a."""
    if n < 1:
        raise ValueError("number of intervals must be positive")
    values = []
    for i in range(n + 1):
        t = i / n
        if exponent != 1.0:
            t = smoothstep(t**exponent)
        values.append(a + t * (b - a))
    values[0], values[-1] = a, b
    return values


def rectangular_block(
    z0: float,
    z1: float,
    r0: float,
    r1: float,
    nz: int,
    nr: int,
    z_exponent: float = 1.0,
    r_exponent: float = 1.0,
) -> list[list[Point]]:
    zs = axis_distribution(z0, z1, nz, z_exponent)
    rs = axis_distribution(r0, r1, nr, r_exponent)
    return [[Point(z, r) for z in zs] for r in rs]


def generate(
    output: Path,
    radius: float,
    center_z: float,
    zmin: float,
    zmax: float,
    rmax: float,
    nq: int,
    nbl: int,
    ntransition: int,
    ntop: int,
    nleft: int,
    nright: int,
    first_layer: float,
    collar_thickness: float,
    transition_exponent: float,
    outer_exponent: float,
) -> None:
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if nq < 2 or nbl < 1 or ntransition < 1 or ntop < 1 or nleft < 1 or nright < 1:
        raise ValueError("require nq>=2 and positive cell counts")

    zl = center_z - 3.0 * radius
    zr = center_z + 3.0 * radius
    rt = 3.0 * radius
    if not (zmin < zl < zr < zmax):
        raise ValueError("farfield must extend beyond the inner rectangle in z")
    if rmax <= rt:
        raise ValueError("rmax must exceed 3*radius")
    if collar_thickness <= 0.0 or radius + collar_thickness >= 3.0 * radius:
        raise ValueError("collar thickness must lie between zero and 2*radius")

    radii, growth = collar_radii(radius, collar_thickness, nbl, first_layer)
    segments = inner_rectangle_segments(center_z, radius)
    mesh = Mesh()

    near_ids: list[list[list[int]]] = []
    for q in range(4):
        coords = near_body_block(
            q, center_z, radius, radii, nq, ntransition,
            transition_exponent, segments[q]
        )
        near_ids.append(mesh.add_block(coords, f"near-body quadrant {q + 1}"))

    # Five axis-aligned rectangular farfield blocks surround the inner rectangle.
    # Their shared edges use exactly matching node counts.
    left_lower_ids = mesh.add_block(
        rectangular_block(zmin, zl, 0.0, rt, nleft, nq, 1.0, 1.0),
        "left-lower outer block",
    )
    left_upper_ids = mesh.add_block(
        rectangular_block(zmin, zl, rt, rmax, nleft, ntop, 1.0, outer_exponent),
        "left-upper outer block",
    )
    top_ids = mesh.add_block(
        rectangular_block(zl, zr, rt, rmax, 2 * nq, ntop, 1.0, outer_exponent),
        "top-center outer block",
    )
    right_lower_ids = mesh.add_block(
        rectangular_block(zr, zmax, 0.0, rt, nright, nq, 1.0, 1.0),
        "right-lower outer block",
    )
    right_upper_ids = mesh.add_block(
        rectangular_block(zr, zmax, rt, rmax, nright, ntop, 1.0, outer_exponent),
        "right-upper outer block",
    )

    # Sphere wall: each near-body block's j=0 row.
    for ids in near_ids:
        mesh.add_edge_chain(ids[0], PHYS_SPHERE)

    # Axis from inlet to upstream pole and downstream pole to outlet.
    mesh.add_edge_chain(left_lower_ids[0], PHYS_AXIS)
    mesh.add_edge_chain([row[0] for row in near_ids[0]], PHYS_AXIS, reverse=True)
    mesh.add_edge_chain([row[-1] for row in near_ids[3]], PHYS_AXIS)
    mesh.add_edge_chain(right_lower_ids[0], PHYS_AXIS)

    # Outer rectangular boundaries.
    mesh.add_edge_chain([row[0] for row in left_lower_ids], PHYS_INLET)
    mesh.add_edge_chain([row[0] for row in left_upper_ids], PHYS_INLET)
    mesh.add_edge_chain(left_upper_ids[-1], PHYS_FARFIELD)
    mesh.add_edge_chain(top_ids[-1], PHYS_FARFIELD)
    mesh.add_edge_chain(right_upper_ids[-1], PHYS_FARFIELD)
    mesh.add_edge_chain([row[-1] for row in right_lower_ids], PHYS_OUTLET)
    mesh.add_edge_chain([row[-1] for row in right_upper_ids], PHYS_OUTLET)

    lines = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat",
        "$PhysicalNames", "6",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "axis"',
        '1 4 "sphere"', '1 5 "farfield"', '2 6 "fluid"',
        "$EndPhysicalNames",
        "$Nodes", str(len(mesh.points)),
    ]
    for node_id, p in enumerate(mesh.points, start=1):
        lines.append(f"{node_id} {p.z:.16g} {p.r:.16g} 0")
    lines.append("$EndNodes")

    lines.extend(["$Elements", str(len(mesh.edges) + len(mesh.quads))])
    elem = 1
    for a, b, physical in mesh.edges:
        lines.append(f"{elem} 1 2 {physical} {physical} {a} {b}")
        elem += 1
    for a, b, c, d in mesh.quads:
        lines.append(f"{elem} 3 2 {PHYS_FLUID} {PHYS_FLUID} {a} {b} {c} {d}")
        elem += 1
    lines.append("$EndElements")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total_near_radial = nbl + ntransition
    print(f"wrote                    : {output}")
    print("coordinate convention    : x=z, y=r, z=0")
    print(f"inner rectangle          : [{zl:g}, {zr:g}] x [0, {rt:g}]")
    print(f"quadrant cells           : {nq} per sphere quadrant")
    print(f"sphere-surface cells     : {4 * nq}")
    print(f"boundary-layer cells     : {nbl}")
    print(f"inner transition cells   : {ntransition}")
    print(f"top farfield layers      : {ntop}")
    print(f"upstream outer cells     : {nleft}")
    print(f"downstream outer cells   : {nright}")
    print(f"total quadrilateral cells: {len(mesh.quads)}")
    print(f"first wall spacing       : {first_layer:.8g}")
    print(f"collar growth ratio      : {growth:.8g}")
    print(f"minimum cell area        : {mesh.min_area:.8g}")
    print(f"maximum cell area        : {mesh.max_area:.8g}")
    print(f"near-body radial layers  : {total_near_radial}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", type=Path, default=Path("FlowOverSphere_Multiblock.msh"))
    p.add_argument("--radius", type=float, default=0.1)
    p.add_argument("--center-z", type=float, default=0.0)
    p.add_argument("--zmin", type=float, default=-1.0)
    p.add_argument("--zmax", type=float, default=4.0)
    p.add_argument("--rmax", type=float, default=1.0)
    p.add_argument("--nq", type=int, default=80, help="cells on each sphere quadrant")
    p.add_argument("--nbl", type=int, default=30, help="exact sphere-normal collar layers")
    p.add_argument("--ntransition", type=int, default=50, help="collar-to-inner-rectangle layers")
    p.add_argument("--ntop", type=int, default=100, help="layers above the inner rectangle")
    p.add_argument("--nleft", type=int, default=100, help="cells from inlet to inner rectangle")
    p.add_argument("--nright", type=int, default=300, help="cells from inner rectangle to outlet")
    p.add_argument("--first-layer", type=float, default=1.0e-4)
    p.add_argument("--collar-thickness", type=float, default=0.05)
    p.add_argument(
        "--transition-exponent", type=float, default=0.8,
        help="near-body transition clustering exponent",
    )
    p.add_argument(
        "--outer-exponent", type=float, default=0.8,
        help="outer-block clustering exponent",
    )
    return p.parse_args()


def main() -> None:
    a = parse_args()
    generate(
        a.output, a.radius, a.center_z, a.zmin, a.zmax, a.rmax,
        a.nq, a.nbl, a.ntransition, a.ntop, a.nleft, a.nright, a.first_layer,
        a.collar_thickness, a.transition_exponent, a.outer_exponent,
    )


if __name__ == "__main__":
    main()
