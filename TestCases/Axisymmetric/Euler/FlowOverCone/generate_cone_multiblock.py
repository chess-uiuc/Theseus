#!/usr/bin/env python3
"""Generate a conforming quadrilateral mesh for axisymmetric cone flow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import tan, radians
from pathlib import Path


PHYS_INLET = 1
PHYS_OUTLET = 2
PHYS_AXIS = 3
PHYS_CONE = 4
PHYS_FARFIELD = 5
PHYS_FLUID = 6


@dataclass(frozen=True)
class Point:
    z: float
    r: float


def stretched_fraction(index: int, count: int, exponent: float) -> float:
    raw = index / count
    return raw**exponent


def signed_quad_area(a: Point, b: Point, c: Point, d: Point) -> float:
    points = (a, b, c, d)
    twice_area = 0.0
    for index, first in enumerate(points):
        second = points[(index + 1) % 4]
        twice_area += first.z * second.r - second.z * first.r
    return 0.5 * twice_area


class Mesh:
    def __init__(self) -> None:
        self.points: list[Point] = []
        self.point_ids: dict[tuple[float, float], int] = {}
        self.quads: list[tuple[int, int, int, int]] = []
        self.edges: list[tuple[int, int, int]] = []
        self.min_area = float("inf")
        self.max_area = 0.0

    @staticmethod
    def key(point: Point) -> tuple[float, float]:
        return round(point.z, 14), round(point.r, 14)

    def add_point(self, point: Point) -> int:
        key = self.key(point)
        if key not in self.point_ids:
            self.point_ids[key] = len(self.points) + 1
            self.points.append(point)
        return self.point_ids[key]

    def add_block(self, coordinates: list[list[Point]], name: str) -> list[list[int]]:
        ids = [[self.add_point(point) for point in row] for row in coordinates]
        for j in range(len(coordinates) - 1):
            for i in range(len(coordinates[0]) - 1):
                a = coordinates[j][i]
                b = coordinates[j][i + 1]
                c = coordinates[j + 1][i + 1]
                d = coordinates[j + 1][i]
                area = signed_quad_area(a, b, c, d)
                if area <= 1.0e-16:
                    raise RuntimeError(
                        f"folded/degenerate cell in {name} at ({i}, {j}): {area:.8e}"
                    )
                self.min_area = min(self.min_area, area)
                self.max_area = max(self.max_area, area)
                self.quads.append((ids[j][i], ids[j][i + 1],
                                   ids[j + 1][i + 1], ids[j + 1][i]))
        return ids

    def add_edge_chain(self, nodes: list[int], physical: int) -> None:
        self.edges.extend((a, b, physical) for a, b in zip(nodes[:-1], nodes[1:]))


def axial_coordinates(start: float, end: float, count: int) -> list[float]:
    return [start + (end - start) * i / count for i in range(count + 1)]


def block_coordinates(z_values: list[float], radial_count: int,
                      radial_max: float, cone_slope: float,
                      radial_exponent: float) -> list[list[Point]]:
    rows: list[list[Point]] = []
    for j in range(radial_count + 1):
        eta = stretched_fraction(j, radial_count, radial_exponent)
        row = []
        for z in z_values:
            wall = max(0.0, z * cone_slope)
            row.append(Point(z, wall + eta * (radial_max - wall)))
        rows.append(row)
    return rows


def write_mesh(output: Path, cone_angle: float, zmin: float, zmax: float,
               radial_max: float, upstream_cells: int, cone_cells: int,
               radial_cells: int, radial_exponent: float) -> None:
    if not (zmin < 0.0 < zmax):
        raise ValueError("the domain must place the cone tip at z=0")
    if cone_angle <= 0.0 or cone_angle >= 45.0:
        raise ValueError("cone angle must lie between zero and 45 degrees")
    if min(upstream_cells, cone_cells, radial_cells) < 1:
        raise ValueError("all cell counts must be positive")
    slope = tan(radians(cone_angle))
    if radial_max <= zmax * slope:
        raise ValueError("radial farfield must lie above the cone surface")

    mesh = Mesh()
    upstream = mesh.add_block(
        block_coordinates(axial_coordinates(zmin, 0.0, upstream_cells),
                          radial_cells, radial_max, slope, radial_exponent),
        "upstream block",
    )
    downstream = mesh.add_block(
        block_coordinates(axial_coordinates(0.0, zmax, cone_cells),
                          radial_cells, radial_max, slope, radial_exponent),
        "cone block",
    )

    mesh.add_edge_chain([row[0] for row in upstream], PHYS_INLET)
    mesh.add_edge_chain(upstream[0], PHYS_AXIS)
    mesh.add_edge_chain(downstream[0], PHYS_CONE)
    mesh.add_edge_chain([row[-1] for row in downstream], PHYS_OUTLET)
    mesh.add_edge_chain(upstream[-1], PHYS_FARFIELD)
    mesh.add_edge_chain(downstream[-1], PHYS_FARFIELD)

    lines = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "6",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "axis"',
        '1 4 "cone"', '1 5 "farfield"', '2 6 "fluid"',
        "$EndPhysicalNames", "$Nodes", str(len(mesh.points)),
    ]
    for node_id, point in enumerate(mesh.points, start=1):
        lines.append(f"{node_id} {point.z:.16g} {point.r:.16g} 0")
    lines.extend(["$EndNodes", "$Elements", str(len(mesh.edges) + len(mesh.quads))])
    element = 1
    for a, b, physical in mesh.edges:
        lines.append(f"{element} 1 2 {physical} {physical} {a} {b}")
        element += 1
    for a, b, c, d in mesh.quads:
        lines.append(f"{element} 3 2 {PHYS_FLUID} {PHYS_FLUID} {a} {b} {c} {d}")
        element += 1
    lines.append("$EndElements")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote                    : {output}")
    print("coordinate convention    : x=z, y=r, z=0")
    print(f"cone half-angle          : {cone_angle:g} degrees")
    print(f"upstream cells           : {upstream_cells}")
    print(f"cone-surface cells       : {cone_cells}")
    print(f"radial cells             : {radial_cells}")
    print(f"total quadrilateral cells: {len(mesh.quads)}")
    print(f"minimum cell area        : {mesh.min_area:.8g}")
    print(f"maximum cell area        : {mesh.max_area:.8g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("FlowOverCone.msh"))
    parser.add_argument("--cone-angle", type=float, default=10.0)
    parser.add_argument("--zmin", type=float, default=-0.5)
    parser.add_argument("--zmax", type=float, default=2.0)
    parser.add_argument("--rmax", type=float, default=1.2)
    parser.add_argument("--upstream-cells", type=int, default=24)
    parser.add_argument("--cone-cells", type=int, default=96)
    parser.add_argument("--radial-cells", type=int, default=48)
    parser.add_argument("--radial-exponent", type=float, default=1.15,
                        help="power-law clustering toward the cone wall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_mesh(args.output, args.cone_angle, args.zmin, args.zmax, args.rmax,
               args.upstream_cells, args.cone_cells, args.radial_cells,
               args.radial_exponent)


if __name__ == "__main__":
    main()
