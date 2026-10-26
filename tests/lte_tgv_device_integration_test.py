#!/usr/bin/env python3
"""p=2 3D viscous LTE evolution: fixed DT/CFL and CPU/device field parity."""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv

from integration_support import run_simulation, work_directory


def require_device(log, device):
    configurations = re.findall(r"Device configuration:\s*([^\n]+)", log)
    if not configurations or device not in configurations[-1].split(","):
        raise AssertionError(f"Requested {device}, reported device configurations: {configurations}")


def collection(output):
    path = output / "ParaView/ParaView.pvd"
    entries = ET.parse(path).findall(".//DataSet")
    if len(entries) < 3:
        raise AssertionError(f"Missing initial/intermediate/final output: {path}")
    return [(float(entry.attrib["timestep"]), path.parent / entry.attrib["file"])
            for entry in entries]


def require_matching_times(a, b):
    if len(a) != len(b) or any(
        not math.isfinite(x) or not math.isfinite(y)
        or not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15)
        for x, y in zip(a, b)
    ):
        raise AssertionError(f"Different output times: {a} versus {b}")


def validate_evolution(output):
    frames = collection(output)
    fields = {"Density", "Pressure", "Temperature", "Velocity"}
    initial = None
    final = None
    for time, path in frames:
        mesh = pv.read(path)
        if not fields.issubset(mesh.point_data.keys()):
            raise AssertionError(f"Missing required fields: {path}")
        current = {}
        for field in fields:
            values = np.asarray(mesh.point_data[field])
            if not np.all(np.isfinite(values)):
                raise AssertionError(f"Nonfinite {field} at {time}")
            if field != "Velocity" and np.min(values) <= 0:
                raise AssertionError(f"Nonpositive {field} at {time}")
            current[field] = values.copy()
        if initial is None:
            initial = current
        final = current
    if frames[0][0] != 0.0 or frames[-1][0] <= 0:
        raise AssertionError("Simulation did not advance from time zero")
    if np.ptp(initial["Velocity"]) < 1.0 or np.ptp(initial["Density"]) < 1e-4:
        raise AssertionError("Fixture is not a nonuniform compressible vortex")
    if np.max(np.abs(final["Velocity"] - initial["Velocity"])) < 1e-8:
        raise AssertionError("Vortex did not evolve")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    case = args.source / "TestCases/NavierStokes/3D/TaylorGreenVortex"
    spec = importlib.util.spec_from_file_location("tgv_mesh", case / "make_periodic_box.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    with work_directory(prefix="theseus-lte-tgv-") as temporary:
        root = Path(temporary)
        mesh = root / "periodic.msh"
        generator.write_mesh(mesh, 2 * math.pi, 2 * math.pi, 2 * math.pi, 4, 4, 4)
        base = json.loads((case / "config.json").read_text())
        runtime = base["runTime"]
        runtime.update(
            order=2, dim=3, num_equations=5, ser_ref_levels=0,
            mesh_file=str(mesh), gas_model="lte", gas_mixture="air11",
            plato_solver="LTE_table_rhoT_(air11)", database_path=args.database,
            N_rho=31, N_T=41, rho_min=0.5, rho_max=2.0,
            T_min=200.0, T_max=2000.0, rho_dist="log", T_dist="log",
            flow_model="cns", numerical_flux="LLF", nsteps_max=20, final_time=1.0,
            visualize=True, paraview=True, visit=False, vis_steps=10,
            print_interval=1, check_cfl=True, nancheck=True, clock_simulation=False,
        )
        runtime["conditions"] = {"initial_conditions": {
            "signature": 1, "function": "CompressibleTaylorGreenVortexIC",
            "params": {"x1": 0.2},
        }}
        runtime["visualization"] = {"fields": ["density", "velocity", "pressure", "temperature"],
                                    "mesh_mode": "gll_subcells"}
        for mode in ("fixed-dt", "fixed-cfl"):
            config = copy.deepcopy(base)
            config["runTime"].update(variable_dt=mode == "fixed-cfl", dt=1e-5, cfl=0.3)
            frames = []
            outputs = []
            # CPU-only environments verify evolution and repeatability. CUDA
            # environments must execute a real same-binary CPU/CUDA comparison.
            for label, device in (("cpu", "cpu"), ("candidate", args.device)):
                output = root / f"{mode}-{label}"
                output.mkdir()
                config["runTime"]["output_file_path"] = str(output)
                path = root / f"{mode}-{label}.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                result = run_simulation(args.executable, path, 1, device, timeout=300)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                require_device(result.stdout, device)
                steps = re.findall(r"time step:\s*(\d+),", result.stdout)
                if not steps or int(steps[-1]) != 20:
                    raise AssertionError(f"Expected 20 steps, got {steps}")
                frames.append(validate_evolution(output))
                outputs.append(output)
            require_matching_times([time for time, _ in frames[0]], [time for time, _ in frames[1]])
            command = [sys.executable, str(args.source / "scripts/compare_viz.py"),
                       str(outputs[0] / "ParaView/ParaView.pvd"),
                       str(outputs[1] / "ParaView/ParaView.pvd"),
                       "--rtol", "1e-10", "--atol", "1e-10"]
            compared = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            (root / f"{mode}-comparison.log").write_text(compared.stdout + compared.stderr)
            print(compared.stdout, flush=True)
            if compared.returncode:
                raise AssertionError(compared.stdout + compared.stderr)
            print(f"{mode}: CPU/{args.device} field comparison PASS", flush=True)
    if args.device == "cpu":
        print("CPU repeatability/evolution PASS; CUDA parity NOT RUN (CPU-only mode)")


if __name__ == "__main__":
    main()
