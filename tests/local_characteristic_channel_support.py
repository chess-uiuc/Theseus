#!/usr/bin/env python3
"""Prepare generic axisymmetric channel cases and retain independent uniform-state errors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def mesh(path):
    nx, ny = 8, 4
    nodes = [(i / nx, j / ny, 0.0) for j in range(ny + 1) for i in range(nx + 1)]

    def node(i, j):
        return j * (nx + 1) + i + 1

    elements = []

    def add(kind, tag, ids):
        elements.append(f"{len(elements) + 1} {kind} 2 {tag} {tag} " + " ".join(map(str, ids)))

    for j in range(ny):
        for i in range(nx):
            add(3, 1, [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)])
    for j in range(ny):
        add(1, 2, [node(0, j + 1), node(0, j)])
        add(1, 3, [node(nx, j), node(nx, j + 1)])
    for i in range(nx):
        add(1, 4, [node(i, 0), node(i + 1, 0)])
        add(1, 5, [node(i + 1, ny), node(i, ny)])
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "5",
             '2 1 "volume"', '1 2 "left"', '1 3 "right"',
             '1 4 "bottom"', '1 5 "top"', "$EndPhysicalNames", "$Nodes", str(len(nodes))]
    lines.extend(f"{i + 1} {x} {y} {z}" for i, (x, y, z) in enumerate(nodes))
    lines.extend(["$EndNodes", "$Elements", str(len(elements)), *elements, "$EndElements"])
    path.write_text("\n".join(lines) + "\n")


def prepare(source, results):
    results.mkdir(parents=True, exist_ok=True)
    mesh(results / "channel.msh")
    initial = {"pressure": 172230.0, "temperature": 400.0, "velocity": [80.0, 0.0]}
    for model in ("euler", "cns"):
        for ranks in (1, 2):
            directory = results / f"{model}-{ranks}"
            directory.mkdir()
            runtime = {
                "order": 2, "dim": 2, "num_equations": 4, "precision": 15,
                "visualize": True, "paraview": True, "visit": False, "vis_steps": 1,
                "output_file_path": str(directory), "nancheck": True, "nancheck_steps": 1,
                "clock_simulation": False, "variable_dt": False, "dt": 1e-7, "final_time": 2e-6,
                "print_interval": 1, "flow_model": model, "gas_model": "cpg",
                "gamma": 1.4, "R_gas": 287.05, "mu": 1.8e-5, "Pr": 0.72,
                "ser_ref_levels": 0, "mesh_file": str(results / "channel.msh"),
                "numerical_flux": "LLF", "ode_solver": "RK3SSP",
                "conditions": {
                    "initial_conditions": {"state": initial},
                    "boundary_conditions": {
                        "left": {"type": "characteristic-inlet", "state": initial},
                        "right": {"type": "characteristic-pressure-outlet", "state": initial},
                        "bottom": {"type": "axis"}, "top": {"type": "slip"},
                    },
                },
            }
            config = {"compileTime": {"AXISYMMETRIC": True}, "runTime": runtime}
            (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (results / "expectations.json").write_text(json.dumps({
        "source": str(source), "density": 1.5, "pressure": initial["pressure"],
        "velocity": [80.0, 0.0], "temperature": 400.0,
        "final_time": 2e-6, "relative_tolerance": 1e-9,
    }, indent=2) + "\n")


def check(results):
    expected = json.loads((results / "expectations.json").read_text())
    measurements = []
    failures = []
    for model in ("euler", "cns"):
        for ranks in (1, 2):
            directory = results / f"{model}-{ranks}"
            collections = list(directory.rglob("*.pvd"))
            if len(collections) != 1:
                raise RuntimeError(f"Expected one PVD collection in {directory}: {collections}")
            reader = pv.get_reader(collections[0])
            if len(reader.time_values) < 2 or not np.isclose(
                    reader.time_values[-1], expected["final_time"], rtol=1e-12, atol=1e-18):
                raise RuntimeError(f"Missing final-time output: {collections[0]}")
            for index, time in enumerate(reader.time_values):
                reader.set_active_time_point(index)
                output = reader.read()
                grid = output.combine() if isinstance(output, pv.MultiBlock) else output
                for field, target, scale in (
                        ("Density", expected["density"], expected["density"]),
                        ("Pressure", expected["pressure"], expected["pressure"]),
                        ("Velocity", expected["velocity"], 80.0)):
                    values = np.asarray(grid.point_data[field])
                    if field == "Velocity":
                        values = values[:, :2]
                    error = float(np.max(np.abs(values - np.asarray(target))) / scale)
                    passed = bool(np.all(np.isfinite(values)) and error <= expected["relative_tolerance"])
                    measurements.append({"model": model, "ranks": ranks, "time": time,
                                         "field": field, "scaled_max_error": error, "passed": passed})
                    if not passed:
                        failures.append(measurements[-1])
    (results / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"Uniform local-characteristic failures: {failures}")
    print(f"PASS: {len(measurements)} full-field/time uniform-state checks; {results / 'measurements.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source.resolve(), args.results.resolve())
    else:
        check(args.results.resolve())


if __name__ == "__main__":
    main()
