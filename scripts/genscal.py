#!/usr/bin/env python3
"""
Generate structured tensor-product Gmsh meshes for Theseus scalability cases.

This version is intended for weak-scaling studies where the requested scale
factor S is a power of two.  Instead of isotropically scaling by S^(1/d), it
consumes one factor of two at a time by doubling one coordinate direction.

For each doubled direction, BOTH the physical domain length and the number of
mesh elements in that direction are doubled.  Therefore

  - total volume/area element count scales exactly by S,
  - element sizes remain exactly nominal,
  - periodic domains remain periodic because extents are integer multiples of
    the nominal periodic length.

Examples, 3D TGV:
  S=1:  factors (1,1,1)
  S=2:  factors (2,1,1)
  S=4:  factors (2,2,1)
  S=8:  factors (2,2,2)
  S=16: factors (4,2,2)

Cases:
  - isentropic-vortex / iv: 2D quad mesh, centered square domain [-1,1]^2.
  - taylor-green-vortex / tgv: 3D hex mesh on
      [-pi,pi] x [-pi,pi] x [0,2*pi]
    expanded periodically by power-of-two direction factors.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CaseSpec:
    name: str
    dim: int
    nominal_n: tuple[int, ...]
    nominal_bounds: tuple[tuple[float, float], ...]
    aliases: tuple[str, ...] = ()


_CASE_LIST: tuple[CaseSpec, ...] = (
    # Nominal uploaded MFEM mesh is effectively 3 x 3 quads on [-1, 1]^2.
    CaseSpec(
        name="isentropic-vortex",
        aliases=("iv", "isentropic_vortex"),
        dim=2,
        nominal_n=(3, 3),
        nominal_bounds=((-1.0, 1.0), (-1.0, 1.0)),
    ),
    # Nominal TGV mesh: 32 x 32 x 32 hexes on [-pi, pi] x [-pi, pi] x [0, 2*pi].
    CaseSpec(
        name="taylor-green-vortex",
        aliases=("tgv", "taylor_green_vortex"),
        dim=3,
        nominal_n=(16, 16, 16),
        nominal_bounds=((-math.pi, math.pi), (-math.pi, math.pi), (0.0, 2.0 * math.pi)),
    ),
)

CASES: dict[str, CaseSpec] = {}
for _case in _CASE_LIST:
    CASES[_case.name] = _case
    for _alias in _case.aliases:
        CASES[_alias] = _case


def parse_positive_power_of_two(value: str) -> int:
    """argparse type: positive integer power of two."""
    try:
        scale = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"S must be an integer power of two, got {value!r}") from exc

    if scale < 1 or scale & (scale - 1):
        raise argparse.ArgumentTypeError(f"S must be a positive power of two, got {scale}")
    return scale


def direction_factors(scale: int, dim: int, order: Sequence[int] | None = None) -> tuple[int, ...]:
    """
    Convert S=2^k into per-direction factors by cycling through directions.

    Default order is x,y,z,... .  For S=2 in 3D this gives (2,1,1);
    for S=4 it gives (2,2,1); for S=8 it gives (2,2,2).
    """
    if order is None:
        order = tuple(range(dim))
    else:
        order = tuple(order)

    if sorted(order) != list(range(dim)):
        raise ValueError(f"direction order must be a permutation of 0..{dim-1}; got {order}")

    k = scale.bit_length() - 1
    factors = [1] * dim
    for i in range(k):
        factors[order[i % dim]] *= 2
    return tuple(factors)


def parse_direction_order(text: str, dim: int) -> tuple[int, ...]:
    """Parse strings like xyz, xzy, xy, yx."""
    text = text.lower().strip()
    axes = "xyz"[:dim]
    mapping = {axis: i for i, axis in enumerate(axes)}
    if len(text) != dim or sorted(text) != sorted(axes):
        raise argparse.ArgumentTypeError(
            f"direction order for {dim}D case must be a permutation of {axes!r}, got {text!r}"
        )
    return tuple(mapping[ch] for ch in text)


def compute_counts_and_bounds(
    case: CaseSpec,
    scale: int,
    factors: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...]]:
    """
    Scale counts and bounds by identical per-direction integer factors.

    Centered dimensions remain centered.  Non-centered dimensions, such as the
    TGV z direction [0, 2*pi], keep their lower bound fixed and extend the upper
    bound.
    """
    counts = tuple(n0 * f for n0, f in zip(case.nominal_n, factors))

    bounds: list[tuple[float, float]] = []
    for (lo, hi), f in zip(case.nominal_bounds, factors):
        length = hi - lo
        center = 0.5 * (lo + hi)

        # Keep [0, L] style dimensions anchored at zero; otherwise expand about center.
        if abs(lo) < 1e-15:
            bounds.append((lo, lo + f * length))
        else:
            half = 0.5 * f * length
            bounds.append((center - half, center + half))

    return counts, tuple(bounds)


def write_geo(case: CaseSpec, counts: tuple[int, ...], bounds: tuple[tuple[float, float], ...], out_geo: pathlib.Path) -> None:
    if case.dim == 2:
        (x0, x1), (y0, y1) = bounds
        nx, ny = counts
        geo = f"""
SetFactory("Built-in");
Mesh.MshFileVersion = 2.2;

Point(1) = {{{x0:.17g}, {y0:.17g}, 0, 1.0}};
Point(2) = {{{x1:.17g}, {y0:.17g}, 0, 1.0}};
Point(3) = {{{x1:.17g}, {y1:.17g}, 0, 1.0}};
Point(4) = {{{x0:.17g}, {y1:.17g}, 0, 1.0}};

Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 1}};
Curve Loop(1) = {{1, 2, 3, 4}};
Plane Surface(1) = {{1}};

Transfinite Curve {{1, 3}} = {nx + 1};
Transfinite Curve {{2, 4}} = {ny + 1};
Transfinite Surface {{1}} = {{1, 2, 3, 4}};
Recombine Surface {{1}};

Physical Surface("fluid") = {{1}};
Physical Curve("xmin") = {{4}};
Physical Curve("xmax") = {{2}};
Physical Curve("ymin") = {{1}};
Physical Curve("ymax") = {{3}};

Periodic Curve {{2}} = {{4}} Translate {{{x1 - x0:.17g}, 0, 0}};
Periodic Curve {{3}} = {{1}} Translate {{0, {y1 - y0:.17g}, 0}};
"""
    else:
        (x0, x1), (y0, y1), (z0, z1) = bounds
        nx, ny, nz = counts
        geo = f"""
SetFactory("Built-in");
Mesh.MshFileVersion = 2.2;

Point(1) = {{{x0:.17g}, {y0:.17g}, {z0:.17g}, 1.0}};
Point(2) = {{{x1:.17g}, {y0:.17g}, {z0:.17g}, 1.0}};
Point(3) = {{{x1:.17g}, {y1:.17g}, {z0:.17g}, 1.0}};
Point(4) = {{{x0:.17g}, {y1:.17g}, {z0:.17g}, 1.0}};
Point(5) = {{{x0:.17g}, {y0:.17g}, {z1:.17g}, 1.0}};
Point(6) = {{{x1:.17g}, {y0:.17g}, {z1:.17g}, 1.0}};
Point(7) = {{{x1:.17g}, {y1:.17g}, {z1:.17g}, 1.0}};
Point(8) = {{{x0:.17g}, {y1:.17g}, {z1:.17g}, 1.0}};

Line(1) = {{1, 2}}; Line(2) = {{2, 3}}; Line(3) = {{3, 4}}; Line(4) = {{4, 1}};
Line(5) = {{5, 6}}; Line(6) = {{6, 7}}; Line(7) = {{7, 8}}; Line(8) = {{8, 5}};
Line(9) = {{1, 5}}; Line(10) = {{2, 6}}; Line(11) = {{3, 7}}; Line(12) = {{4, 8}};

Curve Loop(1) = {{1, 2, 3, 4}};       Plane Surface(1) = {{1}};
Curve Loop(2) = {{5, 6, 7, 8}};       Plane Surface(2) = {{2}};
Curve Loop(3) = {{1, 10, -5, -9}};    Plane Surface(3) = {{3}};
Curve Loop(4) = {{2, 11, -6, -10}};   Plane Surface(4) = {{4}};
Curve Loop(5) = {{3, 12, -7, -11}};   Plane Surface(5) = {{5}};
Curve Loop(6) = {{4, 9, -8, -12}};    Plane Surface(6) = {{6}};

Surface Loop(1) = {{1, 2, 3, 4, 5, 6}};
Volume(1) = {{1}};

Transfinite Curve {{1, 3, 5, 7}} = {nx + 1};
Transfinite Curve {{2, 4, 6, 8}} = {ny + 1};
Transfinite Curve {{9, 10, 11, 12}} = {nz + 1};
Transfinite Surface "*";
Transfinite Volume {{1}} = {{1, 2, 3, 4, 5, 6, 7, 8}};
Recombine Surface "*";

Physical Volume("fluid") = {{1}};
Physical Surface("zmin") = {{1}};
Physical Surface("zmax") = {{2}};
Physical Surface("ymin") = {{3}};
Physical Surface("xmax") = {{4}};
Physical Surface("ymax") = {{5}};
Physical Surface("xmin") = {{6}};

Periodic Surface {{4}} = {{6}} Translate {{{x1 - x0:.17g}, 0, 0}};
Periodic Surface {{5}} = {{3}} Translate {{0, {y1 - y0:.17g}, 0}};
Periodic Surface {{2}} = {{1}} Translate {{0, 0, {z1 - z0:.17g}}};
"""

    out_geo.parent.mkdir(parents=True, exist_ok=True)
    out_geo.write_text(geo.lstrip())


def default_stem(case: CaseSpec, scale: int, counts: tuple[int, ...], factors: tuple[int, ...]) -> str:
    count_text = "x".join(str(n) for n in counts)
    factor_text = "x".join(str(f) for f in factors)
    return f"{case.name}_S{scale}_f{factor_text}_n{count_text}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate power-of-two weak-scaling Gmsh meshes for Theseus cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("case", choices=sorted(CASES), help="case name or alias")
    parser.add_argument("-S", "--scale", type=parse_positive_power_of_two, required=True,
                        help="volume/area element scale factor; must be a positive power of two")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None, help="output .msh path")
    parser.add_argument("--geo-only", action="store_true", help="write .geo but do not run gmsh")
    parser.add_argument("--gmsh", default="gmsh", help="gmsh executable")
    parser.add_argument("--direction-order", default=None,
                        help="axis order for consuming powers of two; e.g. xy/yx for IV, xyz/xzy for TGV")
    args = parser.parse_args()

    case = CASES[args.case]
    order = parse_direction_order(args.direction_order, case.dim) if args.direction_order else tuple(range(case.dim))
    factors = direction_factors(args.scale, case.dim, order)
    counts, bounds = compute_counts_and_bounds(case, args.scale, factors)

    stem = default_stem(case, args.scale, counts, factors)
    out_msh = args.output or pathlib.Path(stem + ".msh")
    out_geo = out_msh.with_suffix(".geo")

    write_geo(case, counts, bounds, out_geo)

    lengths = tuple(hi - lo for lo, hi in bounds)
    h = tuple(L / n for L, n in zip(lengths, counts))
    nominal_lengths = tuple(hi - lo for lo, hi in case.nominal_bounds)
    nominal_h = tuple(L / n for L, n in zip(nominal_lengths, case.nominal_n))
    nelem = math.prod(counts)
    nominal_nelem = math.prod(case.nominal_n)

    print(f"case          : {case.name}")
    print(f"scale S       : {args.scale}")
    print(f"axis order    : {''.join('xyz'[i] for i in order)}")
    print(f"factors       : {factors}")
    print(f"counts        : {counts}  ({nelem} volume/area elements)")
    print(f"actual scale  : {nelem // nominal_nelem:g}")
    print(f"bounds        : {bounds}")
    print(f"h             : {h}")
    print(f"nominal h     : {nominal_h}")
    print(f"geo           : {out_geo}")

    if not args.geo_only:
        cmd = [args.gmsh, f"-{case.dim}", "-format", "msh2", str(out_geo), "-o", str(out_msh)]
        print("running       :", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"mesh          : {out_msh}")


if __name__ == "__main__":
    main()
