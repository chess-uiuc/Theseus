#!/usr/bin/env python3
"""Measure wake, separation, and drag in the viscous sphere demonstration."""

from __future__ import annotations

import argparse
import base64
import json
from math import degrees, pi
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def pvd_datasets(path: Path) -> list[Path]:
    if path.suffix != ".pvd":
        return [path]
    datasets = ET.parse(path).getroot().findall(".//DataSet")
    if not datasets:
        raise RuntimeError(f"no datasets found in {path}")
    return [path.parent / item.attrib["file"] for item in datasets]


def decode_vtk_array(element: ET.Element) -> np.ndarray:
    vtk_types = {
        "Float64": "<f8", "Float32": "<f4", "Int32": "<i4",
        "UInt32": "<u4", "Int64": "<i8", "UInt64": "<u8", "UInt8": "u1",
    }
    if element.attrib.get("format") != "binary":
        raise RuntimeError("built-in VTK reader supports inline binary arrays only")
    encoded = "".join((element.text or "").split())
    if len(encoded) < 8:
        raise RuntimeError("invalid VTK binary array")
    header = base64.b64decode(encoded[:8])
    byte_count = int.from_bytes(header[:4], "little")
    payload = base64.b64decode(encoded[8:])[:byte_count]
    if len(payload) != byte_count:
        raise RuntimeError("truncated VTK binary array")
    values = np.frombuffer(payload, dtype=vtk_types[element.attrib["type"]])
    components = int(element.attrib.get("NumberOfComponents", "1"))
    return values.reshape((-1, components)) if components > 1 else values


def load_vtu_piece(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    root = ET.parse(path).getroot()
    points_element = root.find(".//Points/DataArray")
    if points_element is None:
        raise RuntimeError(f"{path} contains no VTK points")
    fields: dict[str, np.ndarray] = {}
    for element in root.findall(".//PointData/DataArray"):
        name = element.attrib.get("Name")
        if name:
            fields[name] = decode_vtk_array(element)
    return decode_vtk_array(points_element), fields


def load_vtk_xml(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if path.suffix == ".vtu":
        return load_vtu_piece(path)
    if path.suffix != ".pvtu":
        raise RuntimeError(f"unsupported visualization file: {path}")
    root = ET.parse(path).getroot()
    pieces = [path.parent / item.attrib["Source"]
              for item in root.findall(".//Piece")]
    if not pieces:
        raise RuntimeError(f"{path} contains no VTK pieces")
    loaded = [load_vtu_piece(piece) for piece in pieces]
    names = set(loaded[0][1])
    if any(set(fields) != names for _, fields in loaded[1:]):
        raise RuntimeError("parallel VTK pieces contain different point fields")
    return (np.concatenate([points for points, _ in loaded]),
            {name: np.concatenate([fields[name] for _, fields in loaded])
             for name in names})


def load_solution(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        points, fields = load_vtk_xml(path)
    else:
        mesh = pv.read(path)
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        points = np.asarray(mesh.points)
        fields = {name: np.asarray(values)
                  for name, values in mesh.point_data.items()}
    required = ("Density", "Pressure", "Velocity")
    missing = [name for name in required if name not in fields]
    if missing:
        raise RuntimeError(f"visualization output is missing fields: {missing}")
    return points, fields["Density"], fields["Pressure"], fields["Velocity"]


def grouped_profile(coordinate: np.ndarray, values: np.ndarray,
                    digits: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(coordinate, digits)
    unique, inverse = np.unique(rounded, return_inverse=True)
    counts = np.bincount(inverse)
    if values.ndim == 1:
        means = np.bincount(inverse, weights=values)/counts
    else:
        means = np.column_stack([
            np.bincount(inverse, weights=values[:, component])/counts
            for component in range(values.shape[1])
        ])
    return unique, means


def zero_crossing(x: np.ndarray, y: np.ndarray, start_negative: bool) -> float:
    for index in range(len(x)-1):
        if start_negative and y[index] <= 0.0 < y[index+1]:
            fraction = -y[index]/(y[index+1]-y[index])
            return float(x[index] + fraction*(x[index+1]-x[index]))
        if not start_negative and y[index] >= 0.0 > y[index+1]:
            fraction = y[index]/(y[index]-y[index+1])
            return float(x[index] + fraction*(x[index+1]-x[index]))
    raise RuntimeError("requested zero crossing was not found")


def numerical_metrics(points: np.ndarray, density: np.ndarray,
                      pressure: np.ndarray, velocity: np.ndarray,
                      radius: float, rho_inf: float, u_inf: float,
                      mu: float) -> dict[str, float]:
    z = points[:, 0]
    radial = points[:, 1]
    spherical_radius = np.hypot(z, radial)

    axis_mask = (np.abs(radial) < 1.0e-10) & (z > radius+1.0e-8)
    axis_z, axis_velocity = grouped_profile(z[axis_mask], velocity[axis_mask])
    order = np.argsort(axis_z)
    axis_z = axis_z[order]
    axis_velocity = axis_velocity[order]
    wake = axis_z <= 8.0*radius
    if not np.any(axis_velocity[wake, 0] < -1.0e-4*u_inf):
        raise RuntimeError("no resolved reverse flow was found behind the sphere")
    reattachment_z = zero_crossing(axis_z[wake], axis_velocity[wake, 0], True)

    whole_axis = ((np.abs(radial) < 1.0e-10) &
                  (np.abs(z) > 1.08*radius))
    axis_radial_max = float(np.max(np.abs(velocity[whole_axis, 1])))

    collar = ((spherical_radius >= radius-1.0e-9) &
              (spherical_radius <= radius+0.08*radius) &
              (radial >= -1.0e-12))
    theta_all = np.arctan2(radial[collar], z[collar])
    theta_keys = np.round(theta_all, 10)
    theta_samples = np.unique(theta_keys)
    theta_values: list[float] = []
    pressure_values: list[float] = []
    shear_values: list[float] = []
    for theta_key in theta_samples:
        if theta_key <= 1.0e-7 or theta_key >= pi-1.0e-7:
            continue
        selected = np.flatnonzero(collar)[theta_keys == theta_key]
        local_radius, local_pressure = grouped_profile(
            spherical_radius[selected], pressure[selected], digits=11)
        _, local_velocity = grouped_profile(
            spherical_radius[selected], velocity[selected], digits=11)
        local_order = np.argsort(local_radius)
        local_radius = local_radius[local_order]
        local_pressure = local_pressure[local_order]
        local_velocity = local_velocity[local_order]
        exterior = np.flatnonzero(local_radius > radius+1.0e-8)
        wall = np.flatnonzero(np.abs(local_radius-radius) <= 1.0e-8)
        if len(exterior) == 0 or len(wall) == 0:
            continue
        wall_index = wall[0]
        outer_index = exterior[0]
        theta = float(theta_key)
        tangent = np.array((np.sin(theta), -np.cos(theta)))
        wall_ut = float(local_velocity[wall_index, :2] @ tangent)
        outer_ut = float(local_velocity[outer_index, :2] @ tangent)
        spacing = float(local_radius[outer_index]-local_radius[wall_index])
        theta_values.append(theta)
        pressure_values.append(float(local_pressure[wall_index]))
        shear_values.append(mu*(outer_ut-wall_ut)/spacing)

    theta = np.asarray(theta_values)
    wall_pressure = np.asarray(pressure_values)
    wall_shear = np.asarray(shear_values)
    order = np.argsort(theta)
    theta = theta[order]
    wall_pressure = wall_pressure[order]
    wall_shear = wall_shear[order]
    if len(theta) < 30:
        raise RuntimeError("too few sphere-normal collar samples were found")

    # theta=pi is the upstream pole; phi is measured downstream from it.
    phi = pi-theta
    phi_order = np.argsort(phi)
    phi_increasing = phi[phi_order]
    shear_front_to_rear = wall_shear[phi_order]
    separation_candidates = np.flatnonzero(
        (phi_increasing[:-1] > np.radians(70.0)) &
        (shear_front_to_rear[:-1] >= 0.0) &
        (shear_front_to_rear[1:] < 0.0)
    )
    if len(separation_candidates) == 0:
        raise RuntimeError("surface-shear separation point was not found")
    separation_index = int(separation_candidates[0])
    lower = separation_candidates[0]
    x0, x1 = phi_increasing[lower:lower+2]
    y0, y1 = shear_front_to_rear[lower:lower+2]
    separation_phi = float(x0 + y0*(x1-x0)/(y0-y1))

    area_weight = 2.0*pi*radius*radius*np.sin(theta)
    pressure_force = np.trapezoid(-wall_pressure*np.cos(theta)*area_weight, theta)
    viscous_force = np.trapezoid(wall_shear*np.sin(theta)*area_weight, theta)
    reference_force = 0.5*rho_inf*u_inf*u_inf*pi*radius*radius

    return {
        "wake_reattachment_z_over_R": reattachment_z/radius,
        "recirculation_length_over_D": (reattachment_z-radius)/(2.0*radius),
        "minimum_axis_uz_over_Uinf":
            float(np.min(axis_velocity[wake, 0])/u_inf),
        "separation_angle_from_upstream_deg": degrees(separation_phi),
        "pressure_drag_coefficient": pressure_force/reference_force,
        "viscous_drag_coefficient": viscous_force/reference_force,
        "total_drag_coefficient": (pressure_force+viscous_force)/reference_force,
        "maximum_axis_abs_ur_over_Uinf": axis_radial_max/u_inf,
        "surface_angle_samples": int(len(theta)),
        "separation_bracket_index": separation_index,
    }


def relative_change(current: float, previous: float) -> float:
    scale = max(abs(current), abs(previous), 1.0e-14)
    return abs(current-previous)/scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="ParaView.pvd or final .pvtu/.vtu dataset")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--rho-inf", type=float, default=1.0)
    parser.add_argument("--u-inf", type=float, default=0.3)
    parser.add_argument("--mu", type=float, default=0.006)
    parser.add_argument("--json", type=Path, help="optional JSON output")
    parser.add_argument("--check", action="store_true",
                        help="fail unless demonstration tolerances are satisfied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = pvd_datasets(args.input)
    final = numerical_metrics(*load_solution(datasets[-1]), args.radius,
                              args.rho_inf, args.u_inf, args.mu)
    report: dict[str, object] = {"final": final}
    monitored = (
        "recirculation_length_over_D",
        "separation_angle_from_upstream_deg",
        "total_drag_coefficient",
    )
    changes: dict[str, float] = {}
    if len(datasets) >= 2:
        previous = numerical_metrics(*load_solution(datasets[-2]), args.radius,
                                    args.rho_inf, args.u_inf, args.mu)
        changes = {key: relative_change(final[key], previous[key])
                   for key in monitored}
        report["previous"] = previous
        report["relative_change"] = changes

    checks = {
        "steady_separated_wake":
            0.2 <= final["recirculation_length_over_D"] <= 1.5,
        "separation_angle_110_to_140_deg":
            110.0 <= final["separation_angle_from_upstream_deg"] <= 140.0,
        "drag_coefficient_0.9_to_1.3":
            0.9 <= final["total_drag_coefficient"] <= 1.3,
        "axis_radial_velocity_below_1e-6_Uinf":
            final["maximum_axis_abs_ur_over_Uinf"] <= 1.0e-6,
        "last_output_changes_below_2_percent":
            bool(changes) and max(changes.values()) <= 0.02,
    }
    report["checks"] = checks
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.json:
        args.json.write_text(output+"\n", encoding="utf-8")
    if args.check and not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise SystemExit(f"sphere demonstration checks failed: {failed}")


if __name__ == "__main__":
    main()
