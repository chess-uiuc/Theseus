#!/usr/bin/env python3
"""Exercise variable-DT selection and fixed-DT CFL reporting in Theseus."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from integration_support import run_simulation, work_directory


INITIAL_DT_PATTERN = re.compile(r"Initial Timestep DT: ([^\s]+)")
NOMINAL_CFL_PATTERN = re.compile(
    r"Estimated CFL: ([^\s]+) \(actual shortened-step CFL: ([^\s\)]+)\)"
)


def run_case(executable: Path, device: str,
             config: dict, ranks: int) -> str:
    with work_directory(prefix="theseus-cfl-") as tempdir:
        root = Path(tempdir)
        config_path = root / "config.json"
        config["runTime"]["output_file_path"] = str(root)
        config_path.write_text(json.dumps(config, indent=2) + "\n",
                               encoding="utf-8")
        result = run_simulation(executable, config_path, ranks, device)
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
    stokes_coeff = 2.0/3.0
    long_visc = (2.0 - stokes_coeff)*viscosity
    momentum_diffusivity = max(viscosity, long_visc)  # 4.0*viscosity/3.0 + bulk_viscosity
    thermal_diffusivity = viscosity*gamma/runtime["Pr"]
    effective_diffusivity = max(momentum_diffusivity, thermal_diffusivity)
    diffusion_scale = 1.25*82.9000427145
    diffusion_rate = (
        diffusion_scale*effective_diffusivity*2.0/(cell_width*cell_width)
    )
    return advection_rate + diffusion_rate


def check_variable_dt(executable: Path, source: Path, device: str) -> None:
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
        output = run_case(executable, device, config, ranks)
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


def check_fixed_dt_reporting(executable: Path, source: Path, device: str) -> None:
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
    output = run_case(executable, device, config, 1)
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


def check_boundary_initial_dt(executable: Path, device: str, config: dict,
                              expected_dt: float, label: str) -> None:
    measured = []
    for ranks in (1, 2):
        output = run_case(executable, device, config, ranks)
        match = INITIAL_DT_PATTERN.search(output)
        if match is None:
            raise RuntimeError(f"No initial timestep reported:\n{output}")
        actual_dt = float(match.group(1))
        if not math.isclose(actual_dt, expected_dt, rel_tol=2e-12, abs_tol=1e-15):
            raise RuntimeError(
                f"{label}, ranks={ranks}: expected DT {expected_dt:.16e}, "
                f"got {actual_dt:.16e}"
            )
        measured.append(actual_dt)
    if measured[0] != measured[1]:
        raise RuntimeError(f"{label}: serial and two-rank timesteps differ: {measured}")
    print(f"{label}: initial DT {measured[0]:.16e}, one/two ranks PASS", flush=True)


def boundary_config(source: Path) -> dict:
    config = base_config(source)
    config["runTime"].update(
        flow_model="euler", gas_model="cpg", gamma=1.4, R_gas=287.05,
        numerical_flux="LLF", variable_dt=True, cfl=0.1,
        final_time=1e-10, nsteps_max=1,
    )
    return config


def check_prescribed_boundary_dt(executable: Path, source: Path, device: str,
                                 reference: Path, database: Path) -> None:
    oracle = subprocess.run(
        [str(reference), "plato_BoundaryAcoustic_reference"],
        capture_output=True, text=True, timeout=120,
    )
    if oracle.returncode != 0:
        raise RuntimeError(f"AIR11 acoustic reference failed:\n{oracle.stdout}{oracle.stderr}")
    prefix = "BOUNDARY_ACOUSTIC_REFERENCE "
    references = [json.loads(line[len(prefix):]) for line in oracle.stdout.splitlines()
                  if line.startswith(prefix)]
    if len(references) != 1:
        raise RuntimeError(f"Missing AIR11 acoustic reference:\n{oracle.stdout}")
    lte = references[0]
    cpg_hot_speed = math.sqrt(1.4 * 287.05 * 10000)
    lte_hot_speed = lte["boundary_speeds"]["10000"]["exterior-state"]
    if math.isclose(lte_hot_speed, cpg_hot_speed, rel_tol=0.01):
        raise RuntimeError("AIR11 hot reference does not distinguish LTE from CPG")

    # Order-three mapped acoustic scale and square cell width 1/8. The
    # volume contributes 2*c; a vertical boundary contributes |ux|+c.
    cell_width = 0.125
    advection_scale = 1.05 * 9.64849524786
    inlet_velocity = 100.0
    with work_directory(prefix="theseus-boundary-cfl-") as directory:
        for model in ("cpg", "lte"):
            for temperature in (300, 10000):
                profile = Path(directory) / f"{model}-{temperature}.dat"
                profile.write_text(
                    f"2 1\n0 0 300 {inlet_velocity} 0 0\n"
                    f"2 0 {temperature} {inlet_velocity} 0 0\n"
                )
                for boundary_kind in ("exterior-state", "radial-profile"):
                    config = boundary_config(source)
                    runtime = config["runTime"]
                    if model == "lte":
                        runtime.update(lte["runtime"])
                        runtime["database_path"] = str(database)
                        interior_speed = lte["interior_speed"]
                        exterior_speed = lte["boundary_speeds"][str(temperature)][boundary_kind]
                    else:
                        interior_speed = math.sqrt(1.4 * 287.05 * 300)
                        exterior_speed = math.sqrt(1.4 * 287.05 * temperature)
                    conditions = runtime["conditions"]
                    conditions["initial_conditions"] = {"state": {
                        "pressure": 10000.0, "temperature": 300.0,
                        "velocity": [0.0, 0.0],
                    }}
                    boundaries = conditions["boundary_conditions"]
                    # All sides prescribed; vertical faces sample the full
                    # radial profile and include the axial velocity in |u.n|.
                    for name in boundaries:
                        if boundary_kind == "exterior-state":
                            boundaries[name] = {"type": boundary_kind, "state": {
                                "pressure": 10000.0, "temperature": temperature,
                                "velocity": [inlet_velocity, 0.0],
                            }}
                        else:
                            boundaries[name] = {
                                "type": boundary_kind, "pressure": 10000.0,
                                "file": str(profile),
                            }
                    boundary_speed = inlet_velocity + exterior_speed
                    if temperature == 10000 and boundary_speed <= 2 * interior_speed:
                        raise RuntimeError("Hot boundary must control the initial timestep")
                    expected_rate = advection_scale * max(2 * interior_speed, boundary_speed) / cell_width
                    expected_dt = runtime["cfl"] / expected_rate
                    label = f"{model} {boundary_kind}, T={temperature}"
                    check_boundary_initial_dt(executable, device, config, expected_dt, label)
                    # Keep the committed Euler checks above; add CNS checks with
                    # the same cold initial state and prescribed boundary data.
                    runtime["flow_model"] = "cns"
                    if model == "lte":
                        interior_diffusivity = lte["interior_diffusivity"]
                        exterior_diffusivity = lte["boundary_diffusivities"][str(temperature)][boundary_kind]
                    else:
                        density = 10000 / (287.05 * 300)
                        interior_diffusivity = max(4 / 3, 1.4 / runtime["Pr"]) * runtime["mu"] / density
                        exterior_diffusivity = interior_diffusivity * temperature / 300
                    diffusion_rate = (1.25 * 82.9000427145
                                      * max(interior_diffusivity, exterior_diffusivity)
                                      * 2 / cell_width**2)
                    expected_viscous_dt = runtime["cfl"] / (expected_rate + diffusion_rate)
                    check_boundary_initial_dt(executable, device, config, expected_viscous_dt,
                                              label + " CNS boundary diffusion")


def check_supersonic_boundary_dt(executable: Path, source: Path, device: str) -> None:
    config = boundary_config(source)
    runtime = config["runTime"]
    conditions = runtime["conditions"]
    conditions["initial_conditions"] = {"state": {
        "density": 1.4, "pressure": 1.0, "velocity": [0.0, 0.0],
    }}
    # The registered forward-step vector has rho=1.4, p=1, ux=3, uy=0.
    # Thus c=1, the volume rate uses 2, and vertical inflow faces use 4.
    for name in conditions["boundary_conditions"]:
        conditions["boundary_conditions"][name] = {
            "type": "supersonic-inflow", "vector": "ForwardFacingStepLeftBCVector",
        }
    expected_rate = 1.05 * 9.64849524786 * 4 / 0.125
    check_boundary_initial_dt(executable, device, config, runtime["cfl"] / expected_rate,
                              "legacy supersonic-inflow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args()

    executable = args.executable.resolve()
    source = args.source.resolve()
    check_variable_dt(executable, source, args.device)
    check_fixed_dt_reporting(executable, source, args.device)
    check_prescribed_boundary_dt(executable, source, args.device,
                                 args.reference.resolve(), args.database.resolve())
    check_supersonic_boundary_dt(executable, source, args.device)
    print("variable-DT, fixed-DT CFL and CPG/AIR11 boundary acoustic checks passed")


if __name__ == "__main__":
    main()
