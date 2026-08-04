#!/usr/bin/env python3
"""Measure cone shock angle and surface state from Theseus ParaView output."""

from __future__ import annotations

import argparse
import base64
import json
from math import atan2, cos, degrees, pi, sin, sqrt, tan
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def taylor_maccoll_rhs(theta: float, state: np.ndarray, gamma: float) -> np.ndarray:
    radial, polar = state
    thermal = 0.5 * (gamma - 1.0) * (1.0 - radial*radial - polar*polar)
    denominator = thermal - polar*polar
    numerator = polar*polar*radial - thermal * (
        2.0*radial + polar/tan(theta)
    )
    return np.array((polar, numerator/denominator))


def integrate_to_cone(beta: float, cone: float, mach: float,
                      gamma: float, steps: int = 4000) -> tuple[np.ndarray, float, float]:
    freestream = 1.0 / sqrt(1.0 + 2.0/((gamma - 1.0)*mach*mach))
    normal_mach = mach*sin(beta)
    density_ratio = ((gamma + 1.0)*normal_mach*normal_mach) / (
        (gamma - 1.0)*normal_mach*normal_mach + 2.0
    )
    state = np.array((freestream*cos(beta), -freestream*sin(beta)/density_ratio))
    immediate_state = state.copy()
    step = (cone - beta)/steps
    theta = beta
    for _ in range(steps):
        k1 = taylor_maccoll_rhs(theta, state, gamma)
        k2 = taylor_maccoll_rhs(theta + 0.5*step, state + 0.5*step*k1, gamma)
        k3 = taylor_maccoll_rhs(theta + 0.5*step, state + 0.5*step*k2, gamma)
        k4 = taylor_maccoll_rhs(theta + step, state + step*k3, gamma)
        state += step*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        theta += step
    postshock_speed = float(np.linalg.norm(immediate_state))
    return state, normal_mach, postshock_speed


def mach_from_normalized_speed(speed: float, gamma: float) -> float:
    return sqrt(2.0*speed*speed / ((gamma - 1.0)*(1.0-speed*speed)))


def taylor_maccoll_reference(mach: float, cone_angle: float,
                             gamma: float) -> dict[str, float]:
    cone = np.radians(cone_angle)
    mach_angle = np.arcsin(1.0/mach)
    samples = np.linspace(mach_angle + 1.0e-4, 0.5*pi - 1.0e-4, 240)
    previous_beta = float(samples[0])
    previous_residual = integrate_to_cone(previous_beta, cone, mach, gamma)[0][1]
    bracket = None
    for beta in samples[1:]:
        residual = integrate_to_cone(float(beta), cone, mach, gamma, steps=1000)[0][1]
        if residual*previous_residual <= 0.0:
            bracket = previous_beta, float(beta)
            break
        previous_beta, previous_residual = float(beta), residual
    if bracket is None:
        raise RuntimeError("could not bracket the weak Taylor-Maccoll shock solution")
    lower, upper = bracket
    for _ in range(55):
        midpoint = 0.5*(lower + upper)
        lower_value = integrate_to_cone(lower, cone, mach, gamma, steps=1600)[0][1]
        midpoint_value = integrate_to_cone(midpoint, cone, mach, gamma, steps=1600)[0][1]
        if lower_value*midpoint_value <= 0.0:
            upper = midpoint
        else:
            lower = midpoint
    beta = 0.5*(lower + upper)
    surface_state, normal_mach, postshock_speed = integrate_to_cone(
        beta, cone, mach, gamma
    )
    surface_mach = mach_from_normalized_speed(float(np.linalg.norm(surface_state)), gamma)
    postshock_mach = mach_from_normalized_speed(postshock_speed, gamma)
    pressure_jump = 1.0 + 2.0*gamma/(gamma + 1.0)*(normal_mach**2 - 1.0)
    surface_pressure_ratio = pressure_jump * (
        (1.0 + 0.5*(gamma - 1.0)*postshock_mach**2) /
        (1.0 + 0.5*(gamma - 1.0)*surface_mach**2)
    )**(gamma/(gamma - 1.0))
    cp = (surface_pressure_ratio - 1.0)/(0.5*gamma*mach*mach)
    return {
        "shock_angle_deg": degrees(beta),
        "surface_mach": surface_mach,
        "surface_pressure_ratio": surface_pressure_ratio,
        "surface_cp": cp,
    }


def final_dataset(path: Path) -> Path:
    if path.suffix != ".pvd":
        return path
    datasets = ET.parse(path).getroot().findall(".//DataSet")
    if not datasets:
        raise RuntimeError(f"no datasets found in {path}")
    return path.parent / datasets[-1].attrib["file"]


def decode_vtk_array(element: ET.Element) -> np.ndarray:
    vtk_types = {
        "Float64": "<f8", "Float32": "<f4", "Int32": "<i4",
        "UInt32": "<u4", "UInt8": "u1",
    }
    if element.attrib.get("format") != "binary":
        raise RuntimeError("built-in VTK reader supports inline binary arrays only")
    encoded = "".join((element.text or "").split())
    if len(encoded) < 8:
        raise RuntimeError("invalid VTK binary array")
    byte_count = int.from_bytes(base64.b64decode(encoded[:8])[:4], "little")
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
    fields = {}
    for element in root.findall(".//PointData/DataArray"):
        name = element.attrib.get("Name")
        if name:
            fields[name] = decode_vtk_array(element)
    return decode_vtk_array(points_element), fields


def load_vtk_xml(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    path = final_dataset(path)
    if path.suffix == ".vtu":
        return load_vtu_piece(path)
    if path.suffix != ".pvtu":
        raise RuntimeError(f"unsupported visualization file: {path}")
    root = ET.parse(path).getroot()
    pieces = [path.parent / element.attrib["Source"]
              for element in root.findall(".//Piece")]
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
        mesh = pv.read(final_dataset(path))
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


def bin_centers(start: float, stop: float, count: int) -> np.ndarray:
    width = (stop - start)/count
    return np.linspace(start + 0.5*width, stop - 0.5*width, count)


def numerical_metrics(points: np.ndarray, density: np.ndarray,
                      pressure: np.ndarray, velocity: np.ndarray,
                      cone_angle: float, gamma: float, mach: float,
                      z_start: float, z_stop: float,
                      bins: int) -> dict[str, float]:
    z = points[:, 0]
    radius = points[:, 1]
    slope = tan(np.radians(cone_angle))
    available_z = np.unique(np.round(z[(z >= z_start) & (z <= z_stop)], 12))
    centers = bin_centers(z_start, z_stop, bins)
    shock_points: list[tuple[float, float]] = []
    wall_pressure: list[float] = []
    wall_mach: list[float] = []
    for center in centers:
        station = available_z[np.argmin(np.abs(available_z-center))]
        indices = np.flatnonzero(np.abs(z-station) <= 1.0e-11)
        if len(indices) < 5:
            continue
        order = indices[np.argsort(radius[indices])]
        rounded_r = np.round(radius[order], 12)
        unique_r, inverse = np.unique(rounded_r, return_inverse=True)
        counts = np.bincount(inverse)
        local_r = unique_r
        local_p = np.bincount(inverse, weights=pressure[order])/counts
        local_rho = np.bincount(inverse, weights=density[order])/counts
        local_velocity = np.column_stack([
            np.bincount(inverse, weights=velocity[order, component])/counts
            for component in range(2)
        ])
        wall = station*slope
        fluid = np.flatnonzero(local_r >= wall-1.0e-10)
        if len(fluid) < 5:
            continue
        wall_index = fluid[np.argmin(local_r[fluid]-wall)]
        wall_pressure.append(float(local_p[wall_index]))
        speed = float(np.linalg.norm(local_velocity[wall_index]))
        sound_speed = sqrt(gamma*local_p[wall_index]/local_rho[wall_index])
        wall_mach.append(speed/sound_speed)
        gradient = np.abs(np.diff(local_p)/np.diff(local_r))
        midpoint_r = 0.5*(local_r[:-1]+local_r[1:])
        candidates = np.flatnonzero(midpoint_r > wall + 0.03)
        if len(candidates):
            shock_index = candidates[np.argmax(gradient[candidates])]
            shock_points.append((float(station), float(midpoint_r[shock_index])))
    if len(shock_points) < max(4, bins//3):
        raise RuntimeError("too few axial samples were available to fit the shock")
    fit = np.polyfit(np.asarray(shock_points)[:, 0],
                     np.asarray(shock_points)[:, 1], 1)
    shock_angle = degrees(atan2(float(fit[0]), 1.0))
    p_inf = 1.0/gamma
    q_inf = 0.5*mach*mach
    mean_pressure = float(np.mean(wall_pressure))
    mean_mach = float(np.mean(wall_mach))
    return {
        "shock_angle_deg": shock_angle,
        "surface_mach": mean_mach,
        "surface_mach_std": float(np.std(wall_mach)),
        "surface_pressure_ratio": mean_pressure/p_inf,
        "surface_pressure_ratio_std": float(np.std(wall_pressure))/p_inf,
        "surface_cp": (mean_pressure-p_inf)/q_inf,
        "shock_fit_samples": len(shock_points),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="ParaView.pvd or final .pvtu/.vtu dataset")
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--cone-angle", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--z-start", type=float, default=0.35)
    parser.add_argument("--z-stop", type=float, default=1.55)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--json", type=Path, help="optional JSON output")
    parser.add_argument("--check", action="store_true",
                        help="fail unless demonstration tolerances are satisfied")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = taylor_maccoll_reference(args.mach, args.cone_angle, args.gamma)
    numerical = numerical_metrics(*load_solution(args.input), args.cone_angle,
                                  args.gamma, args.mach, args.z_start,
                                  args.z_stop, args.bins)
    report = {"numerical": numerical, "taylor_maccoll": reference}
    report["difference"] = {
        key: numerical[key]-reference[key]
        for key in ("shock_angle_deg", "surface_mach",
                    "surface_pressure_ratio", "surface_cp")
    }
    checks = {
        "shock_angle_within_1_deg":
            abs(report["difference"]["shock_angle_deg"]) <= 1.0,
        "surface_mach_within_0.05":
            abs(report["difference"]["surface_mach"]) <= 0.05,
        "surface_pressure_ratio_within_0.02":
            abs(report["difference"]["surface_pressure_ratio"]) <= 0.02,
        "surface_mach_std_below_0.02": numerical["surface_mach_std"] <= 0.02,
        "surface_pressure_ratio_std_below_0.01":
            numerical["surface_pressure_ratio_std"] <= 0.01,
    }
    report["demonstration_checks"] = checks
    report["demonstration_passed"] = all(checks.values())
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.check and not report["demonstration_passed"]:
        raise SystemExit("cone-flow demonstration checks failed")


if __name__ == "__main__":
    main()
