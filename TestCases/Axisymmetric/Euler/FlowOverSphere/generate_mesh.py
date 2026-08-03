#!/usr/bin/env python3
"""Generate a body-fitted quadrilateral meridional mesh around a unit sphere."""

from math import sqrt
from pathlib import Path


NZ = 40
NR = 12
Z_MIN = -4.0
Z_MAX = 6.0
R_MAX = 5.0


def node(i: int, j: int) -> int:
    return j * (NZ + 1) + i + 1


def bottom_radius(z: float) -> float:
    return sqrt(max(0.0, 1.0 - z*z)) if -1.0 <= z <= 1.0 else 0.0


def main() -> None:
    lines = [
        "$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "6",
        '1 1 "inlet"', '1 2 "outlet"', '1 3 "axis"',
        '1 4 "sphere"', '1 5 "farfield"', '2 6 "fluid"',
        "$EndPhysicalNames", "$Nodes", str((NZ + 1) * (NR + 1)),
    ]
    for j in range(NR + 1):
        eta = (j / NR) ** 1.5
        for i in range(NZ + 1):
            z = Z_MIN + (Z_MAX - Z_MIN) * i / NZ
            r0 = bottom_radius(z)
            r = r0 + eta * (R_MAX - r0)
            lines.append(f"{node(i, j)} {z:.16g} {r:.16g} 0")
    lines.extend(["$EndNodes", "$Elements"])

    boundary_count = 2 * NR + 2 * NZ
    volume_count = NZ * NR
    lines.append(str(boundary_count + volume_count))
    element = 1
    for j in range(NR):
        lines.append(f"{element} 1 2 1 1 {node(0, j)} {node(0, j + 1)}")
        element += 1
    for j in range(NR):
        lines.append(f"{element} 1 2 2 2 {node(NZ, j)} {node(NZ, j + 1)}")
        element += 1
    for i in range(NZ):
        z_mid = Z_MIN + (Z_MAX - Z_MIN) * (i + 0.5) / NZ
        physical = 4 if -1.0 < z_mid < 1.0 else 3
        lines.append(
            f"{element} 1 2 {physical} {physical} {node(i, 0)} {node(i + 1, 0)}"
        )
        element += 1
    for i in range(NZ):
        lines.append(
            f"{element} 1 2 5 5 {node(i, NR)} {node(i + 1, NR)}"
        )
        element += 1
    for j in range(NR):
        for i in range(NZ):
            lines.append(
                f"{element} 3 2 6 6 {node(i, j)} {node(i + 1, j)} "
                f"{node(i + 1, j + 1)} {node(i, j + 1)}"
            )
            element += 1
    lines.append("$EndElements")
    Path(__file__).with_name("FlowOverSphere.msh").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
