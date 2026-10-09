#!/usr/bin/env python3
"""Smoke-test axisymmetric Euler flow over a sphere in serial and MPI."""

import argparse
import json
import math
import re
from pathlib import Path

from integration_support import run_simulation, work_directory


RANGE_PATTERN = re.compile(
    r"rho\(([^,]+),([^\)]+)\), p\(([^,]+),([^\)]+)\)"
)


def run_case(executable: Path, source: Path, case: Path, ranks: int,
             device: str) -> tuple[float, ...]:
    case = source / case
    config = json.loads((case / "config.json").read_text())
    runtime = config["runTime"]
    runtime.update({
        "mesh_file": str((source / "TestCases/Axisymmetric/Euler/"
                          "FlowOverSphere/FlowOverSphere.msh").resolve()),
        "output_file_path": "",
        "visualize": False,
        "paraview": False,
        "clock_simulation": False,
        "final_time": min(runtime["final_time"], 0.02),
        "print_interval": 1,
    })

    with work_directory(prefix="theseus-axis-sphere-") as tmp:
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        result = run_simulation(executable, config_path, ranks, device, timeout=360)
    if result.returncode != 0:
        raise RuntimeError(
            f"{case.name} ({ranks} ranks) exited with "
            f"{result.returncode}:\n{result.stdout}"
        )
    matches = RANGE_PATTERN.findall(result.stdout)
    if not matches:
        raise RuntimeError(
            f"{case.name} ({ranks} ranks) reported no state ranges:\n"
            f"{result.stdout}"
        )
    values = tuple(float(value) for value in matches[-1])
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise RuntimeError(
            f"{case.name} ({ranks} ranks) has invalid state range {values}"
        )
    if values[1] - values[0] < 0.1 or values[3] - values[2] < 0.1:
        raise RuntimeError(
            f"{case.name} ({ranks} ranks) did not develop a body-flow "
            f"response: {values}"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", choices=("inviscid", "viscous"),
                        required=True)
    args = parser.parse_args()

    cases = {
        "inviscid": Path("TestCases/Axisymmetric/Euler/FlowOverSphere"),
        "viscous": Path(
            "TestCases/Axisymmetric/NavierStokes/ViscousFlowOverSphere"),
    }
    case = cases[args.case]
    serial = run_case(
        args.executable.resolve(), args.source.resolve(),
        case, 1, args.device)
    parallel = run_case(
        args.executable.resolve(), args.source.resolve(),
        case, 2, args.device)
    for index, (one, two) in enumerate(zip(serial, parallel)):
        if not math.isclose(one, two, rel_tol=0.0, abs_tol=1.0e-5):
            raise RuntimeError(
                f"{case.name} serial/MPI range component {index} "
                f"differs: {one} vs {two}"
            )


if __name__ == "__main__":
    main()
