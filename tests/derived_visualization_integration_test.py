#!/usr/bin/env python3
"""Check emitted CPG/LTE fields, selection, both writers, and restart output."""
import argparse
import copy
import json
from pathlib import Path

from integration_support import run_simulation, work_directory
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


FIELDS = {
    "density": "Density",
    "velocity": "Velocity",
    "pressure": "Pressure",
    "temperature": "Temperature",
    "sound_speed": "Sound Speed",
    "mach_number": "Mach Number",
    "specific_internal_energy": "Specific Internal Energy",
    "internal_energy_density": "Internal Energy Density",
    "specific_total_energy": "Specific Total Energy",
    "viscosity": "Viscosity",
    "thermal_conductivity": "Thermal Conductivity",
}


def check_values(actual, expected, label):
    actual = np.asarray(actual)
    expected = np.broadcast_to(expected, actual.shape)
    if not np.all(np.isfinite(actual)):
        raise AssertionError(f"{label}: nonfinite values")
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=label)


def vtk_fields(output):
    collection = next(output.glob("ParaView/*.pvd"))
    datasets = ET.parse(collection).findall(".//DataSet")
    assert datasets, collection
    return [(float(entry.attrib["timestep"]), pv.read(collection.parent / entry.attrib["file"]))
            for entry in datasets]


def check_vtk(output, reference, selected):
    datasets = vtk_fields(output)
    names = {FIELDS[field] for field in selected}
    for time, mesh in datasets:
        assert set(mesh.point_data.keys()) == names, (time, mesh.point_data.keys(), names)
        if "Velocity" in names:
            assert mesh.point_data["Velocity"].shape == (mesh.n_points, 2)
        for name in names:
            check_values(mesh.point_data[name], reference[name], f"{output.name}: {name} at {time}")
    return datasets


def check_visit(output, reference):
    roots = list(output.glob("VisIt*.mfem_root"))
    assert roots, list(output.iterdir())
    for root_file in roots:
        metadata = json.loads(root_file.read_text())
        fields = metadata["dsets"]["main"]["fields"]
        assert set(fields) == set(reference), fields.keys()
        for name, description in fields.items():
            pattern = description["path"]
            field_file = root_file.parent / (pattern % 0)
            lines = field_file.read_text().splitlines()
            ordering = next(index for index, line in enumerate(lines) if line.startswith("Ordering:"))
            values = np.fromstring(" ".join(lines[ordering + 1:]), sep=" ")
            assert values.size, field_file
            if name == "Velocity":
                values = values.reshape(2, -1).T
                check_values(values, reference[name], name)
            else:
                check_values(values, reference[name], name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    oracle = subprocess.run([str(args.reference), "plato_Visualization_reference"],
                            capture_output=True, text=True, check=True, timeout=90)
    references = {}
    for line in oracle.stdout.splitlines():
        if line.startswith("VISUALIZATION_REFERENCE "):
            _, model, values = line.split(" ", 2)
            references[model] = json.loads(values)
    assert set(references) == {"cpg", "lte"}, oracle.stdout
    assert abs(references["cpg"]["Specific Internal Energy"] -
               references["lte"]["Specific Internal Energy"]) > 1

    case = args.source / "TestCases/LTE/Euler/LTEVortex"
    base = json.loads((case / "config.json").read_text())
    runtime = base["runTime"]
    runtime.update(mesh_file=str(case / "LTEVortex.mesh"), database_path=args.database,
                   order=2, ser_ref_levels=0, N_rho=25, N_T=25,
                   rho_min=0.05, rho_max=1.1, T_min=250.0, T_max=3000.0,
                   visualize=True, paraview=True, visit=False, variable_dt=False,
                   dt=1e-8, final_time=1.0, nsteps_max=2, vis_steps=1,
                   checkpoint_save=True, checkpoint_dt=1e-8, clock_simulation=False)
    runtime["conditions"] = {"initial_conditions": {"state": {
        "pressure": 60000, "temperature": 1200, "velocity": [10, -2]}}}
    runtime["visualization"] = {"fields": list(FIELDS), "mesh_mode": "gll_subcells"}

    with work_directory(prefix="theseus-derived-output-") as temporary:
        root = Path(temporary)

        def run(name, config, ranks=1, restart_from=None):
            output = root / name
            output.mkdir()
            config = copy.deepcopy(config)
            config["runTime"]["output_file_path"] = str(output)
            if restart_from is not None:
                shutil.copytree(restart_from / "Checkpoints/Cycle1", output / "Checkpoints/Cycle1")
            config_path = output / "config.json"
            config_path.write_text(json.dumps(config))
            result = run_simulation(args.executable, config_path, ranks, args.device, timeout=90)
            if result.returncode:
                raise RuntimeError(f"{name}\n{result.stdout}\n{result.stderr}")
            return output

        for model in ("cpg", "lte"):
            config = copy.deepcopy(base)
            config["runTime"]["gas_model"] = model
            reference = references[model]
            for mode in ("gll_subcells", "vtk_high_order"):
                config["runTime"]["visualization"]["mesh_mode"] = mode
                output = run(f"{model}-{mode}", config, ranks=2)
                datasets = check_vtk(output, reference, FIELDS)
                assert len(datasets) == 3, len(datasets)
                if mode == "gll_subcells":
                    restart = copy.deepcopy(config)
                    restart["runTime"].update(checkpoint_load=True, checkpoint_cycle=1)
                    restart["runTime"]["conditions"]["initial_conditions"] = {"state": {"temperature": -1}}
                    resumed = run(f"{model}-restart", restart, ranks=2, restart_from=output)
                    resumed_data = check_vtk(resumed, reference, FIELDS)
                    assert resumed_data[-1][0] == datasets[-1][0]
                    for name in reference:
                        np.testing.assert_array_equal(resumed_data[-1][1].point_data[name],
                                                      datasets[-1][1].point_data[name])

            selected = copy.deepcopy(config)
            selected["runTime"]["visualization"]["fields"] = ["temperature", "mach_number"]
            output = run(f"{model}-selection", selected)
            check_vtk(output, reference, ["temperature", "mach_number"])

            visit = copy.deepcopy(config)
            visit["runTime"].update(paraview=False, visit=True)
            output = run(f"{model}-visit", visit)
            check_visit(output, reference)

            disabled = copy.deepcopy(config)
            disabled["runTime"]["visualize"] = False
            output = run(f"{model}-disabled", disabled)
            assert not list(output.glob("**/*.vtu"))
            assert not list(output.glob("*.mfem_root"))

    print("Derived CPG/LTE fields, selection, ParaView/VisIt, MPI and restart: PASS")


if __name__ == "__main__":
    main()
