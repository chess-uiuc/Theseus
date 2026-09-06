#!/usr/bin/env python3
"""Exercise variable-DT selection and fixed-DT CFL reporting in Theseus."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


INITIAL_DT_PATTERN = re.compile(r"Initial Timestep DT: ([^\s]+)")
NOMINAL_CFL_PATTERN = re.compile(
    r"Estimated CFL: ([^\s]+) \(actual shortened-step CFL: ([^\s\)]+)\)"
)


def mpi_launcher(mpiexec: Path) -> list[str]:
    version = subprocess.run(
        [str(mpiexec), "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    ).stdout
    launcher = [str(mpiexec)]
    if "Open MPI" in version or "OpenRTE" in version:
        launcher.extend(
            ["--host", "localhost:2", "--map-by", "slot:OVERSUBSCRIBE",
             "--bind-to", "none"]
        )
    return launcher


def run_case(executable: Path, mpiexec: Path, numproc_flag: str,
             config: dict, ranks: int) -> str:
    with tempfile.TemporaryDirectory(prefix="theseus-cfl-") as tempdir:
        root = Path(tempdir)
        config_path = root / "config.json"
        config["runTime"]["output_file_path"] = str(root)
        config_path.write_text(json.dumps(config, indent=2) + "\n",
                               encoding="utf-8")
        result = subprocess.run(
            mpi_launcher(mpiexec)
            + [numproc_flag, str(ranks), str(executable),
               "-d", "cpu", "-c", str(config_path)],
            cwd=root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=120, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Theseus CFL case ({ranks} ranks) exited with "
            f"{result.returncode}:\n{result.stdout}"
        )
    return result.stdout


def base_config(source: Path) -> dict:
    case = source / "TestCases/NavierStokes/2D/LidDrivenCavity"
    config = json.loads((case / "config.json").read_text(encoding="utf-8"))
    runtime = config["runTime"]
    runtime.update({
        "mesh_file": str((case / "LidDrivenCavity.msh").resolve()),
        "visualize": False,
        "paraview": False,
        "visit": False,
        "clock_simulation": False,
        # The variable-DT assertion concerns the estimate made before the
        # first ODE step.  Existing solution-regression cases own evolution
        # and NaN validation.
        "nancheck": False,
        "print_interval": 1,
    })
    return config


def expected_initial_rate(config: dict) -> float:
    runtime = config["runTime"]
    order = runtime["order"]
    if order != 3:
        raise RuntimeError("CFL integration reference is calibrated for order 3")

    # The cavity mesh is a uniform 16-by-16 grid over [0,2]^2, hence h=1/8.
    cell_width = 0.125
    gamma = runtime["gamma"]
    mach = runtime["conditions"]["initial_conditions"]["params"]["x1"]
    sound_speed = 1.0/mach
    advection_scale = 1.05*9.64849524786
    advection_rate = advection_scale*2.0*sound_speed/cell_width

    viscosity = runtime["mu"]
    bulk_viscosity = 2.0/3.0
    momentum_diffusivity = 4.0*viscosity/3.0 + bulk_viscosity
    thermal_diffusivity = viscosity*gamma/runtime["Pr"]
    effective_diffusivity = max(momentum_diffusivity, thermal_diffusivity)
    diffusion_scale = 1.25*82.9000427145
    diffusion_rate = (
        diffusion_scale*effective_diffusivity*2.0/(cell_width*cell_width)
    )
    return advection_rate + diffusion_rate


def check_variable_dt(executable: Path, source: Path, mpiexec: Path,
                      numproc_flag: str) -> None:
    config = base_config(source)
    runtime = config["runTime"]
    target_cfl = 0.2
    runtime.update({
        "variable_dt": True,
        "cfl": target_cfl,
        "final_time": 1.0,
        "nsteps_max": 1,
    })
    expected_dt = target_cfl/expected_initial_rate(config)
    measured = []
    for ranks in (1, 2):
        output = run_case(executable, mpiexec, numproc_flag, config, ranks)
        match = INITIAL_DT_PATTERN.search(output)
        if not match:
            raise RuntimeError(f"No initial variable timestep reported:\n{output}")
        actual_dt = float(match.group(1))
        if not math.isclose(actual_dt, expected_dt, rel_tol=2.0e-12,
                            abs_tol=1.0e-15):
            raise RuntimeError(
                f"{ranks}-rank initial DT: expected {expected_dt:.16e}, "
                f"got {actual_dt:.16e}"
            )
        measured.append(actual_dt)
    if measured[0] != measured[1]:
        raise RuntimeError(
            f"Serial and two-rank timesteps differ: {measured}"
        )


def check_fixed_dt_reporting(executable: Path, source: Path, mpiexec: Path,
                             numproc_flag: str) -> None:
    config = base_config(source)
    runtime = config["runTime"]
    fixed_dt = 2.0e-5
    shortened_dt = 5.0e-6
    runtime.update({
        "variable_dt": False,
        "dt": fixed_dt,
        "cfl_check_interval": 2,
        "print_interval": 2,
        "final_time": fixed_dt + shortened_dt,
        "nsteps_max": 2,
    })
    output = run_case(executable, mpiexec, numproc_flag, config, 1)
    matches = NOMINAL_CFL_PATTERN.findall(output)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one fixed-DT CFL report, found {len(matches)}:\n"
            f"{output}"
        )
    nominal, actual = (float(value) for value in matches[0])
    expected_ratio = shortened_dt/fixed_dt
    if not actual < nominal:
        raise RuntimeError(
            f"Shortened-step CFL {actual} is not below nominal CFL {nominal}"
        )
    if not math.isclose(actual/nominal, expected_ratio, rel_tol=2.0e-12,
                        abs_tol=1.0e-15):
        raise RuntimeError(
            f"Shortened/nominal CFL ratio: expected {expected_ratio}, "
            f"got {actual/nominal}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument("--numproc-flag", required=True)
    args = parser.parse_args()

    executable = args.executable.resolve()
    source = args.source.resolve()
    mpiexec = args.mpiexec.resolve()
    check_variable_dt(executable, source, mpiexec, args.numproc_flag)
    check_fixed_dt_reporting(executable, source, mpiexec, args.numproc_flag)
    print("variable-DT selection and fixed-DT CFL reporting passed")


if __name__ == "__main__":
    main()
