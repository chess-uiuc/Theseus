#!/usr/bin/env python3
"""LTE chamber startup through physical inputs, walls, profiles and derived output."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


def final_fields(output):
    collection = output / "ParaView/ParaView.pvd"
    last = ET.parse(collection).findall(".//DataSet")[-1]
    mesh = pv.read(collection.parent / last.attrib["file"])
    expected = {"Density", "Velocity", "Pressure", "Temperature", "Sound Speed", "Mach Number",
                "Specific Internal Energy", "Viscosity", "Thermal Conductivity"}
    assert set(mesh.point_data.keys()) == expected, mesh.point_data.keys()
    fields = {name: np.asarray(mesh.point_data[name]) for name in expected}
    for name, values in fields.items():
        assert np.all(np.isfinite(values)), name
    for name in ("Density", "Pressure", "Temperature", "Sound Speed", "Viscosity",
                 "Thermal Conductivity"):
        assert fields[name].min() > 0, (name, fields[name].min())
    assert fields["Velocity"].shape == (mesh.n_points, 2)
    assert np.all(fields["Mach Number"] >= 0)
    speed = np.linalg.norm(fields["Velocity"], axis=1)
    np.testing.assert_allclose(fields["Mach Number"], speed / fields["Sound Speed"],
                               rtol=1e-10, atol=1e-12)
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    case = args.source / "TestCases/Axisymmetric/NavierStokes/PXChamber"
    spec = importlib.util.spec_from_file_location("px_case", case / "run_case.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    launcher = [args.mpiexec]
    version = subprocess.run(launcher + ["--version"], capture_output=True, text=True).stdout
    if "Open MPI" in version or "OpenRTE" in version:
        launcher += ["--host", "localhost:2", "--map-by", "slot:OVERSUBSCRIBE", "--bind-to", "none"]

    with tempfile.TemporaryDirectory(prefix="theseus-lte-chamber-") as temporary:
        root = Path(temporary)
        uniform = root / "uniform"
        config_path = helper.prepare(uniform, True, 2e-8, 1e-9, "lte", args.database)
        profile = uniform / "uniform.dat"
        profile.write_text("2 1\n0 0 300 0 0 0\n1 0 300 0 0 0\n")
        config = json.loads(config_path.read_text())
        config["runTime"]["conditions"]["boundary_conditions"]["Inflow"]["file"] = str(profile)
        config_path.write_text(json.dumps(config))
        command = launcher + ["-n", "1", str(args.executable),
                   "-d", args.device, "-c", str(config_path)]
        result = subprocess.run(command, cwd=uniform, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        fields = final_fields(uniform)
        np.testing.assert_allclose(fields["Temperature"], 300, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(fields["Pressure"], 10000, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(fields["Velocity"], 0, rtol=0, atol=1e-9)

        statistics = []
        for ranks in (1, 2):
            output = root / f"heated-{ranks}"
            command = [sys.executable, str(case / "run_case.py"),
                       "--executable", str(args.executable), "--output", str(output),
                       "--ranks", str(ranks), "--coarse", "--gas-model", "lte",
                       "--database", args.database, "--mpiexec", args.mpiexec,
                       "--device", args.device, "--dt", "1e-9", "--final-time", "2e-8"]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
            if result.returncode:
                log = (output / "run.log").read_text() if (output / "run.log").exists() else ""
                raise RuntimeError(result.stdout + result.stderr + log)
            fields = final_fields(output)
            assert fields["Temperature"].max() > 301, fields["Temperature"].max()
            statistics.append({name: [values.min(), values.max(), values.mean()]
                               for name, values in fields.items()})
        for name in statistics[0]:
            np.testing.assert_allclose(statistics[0][name], statistics[1][name],
                                       rtol=1e-9, atol=1e-8, err_msg=name)
    print("LTE chamber: uniform walls, heated profile, derived fields and MPI agreement PASS")


if __name__ == "__main__":
    main()
