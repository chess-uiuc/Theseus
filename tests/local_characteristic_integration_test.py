#!/usr/bin/env python3
"""Visible recipe input/assertion stages for generic channels and LTE chamber."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pyvista as pv
import local_characteristic_channel_support as channel


def prepare(source, results, database):
    for provider in ("reference", "radial", "total", "reverse"):
        directory = results / provider
        channel.prepare(source, directory)
        profile = directory / "profile.dat"
        profile.write_text("2 1\n0 0 400 80 0 0\n1 0 400 80 0 0\n")
        for model in ("euler", "cns"):
            for ranks in (1, 2):
                path = directory / f"{model}-{ranks}" / "config.json"
                config = json.loads(path.read_text())
                rt = config["runTime"]
                bc = rt["conditions"]["boundary_conditions"]
                if provider == "radial":
                    bc["left"] = {"type": "characteristic-radial-inlet", "file": str(profile), "pressure": 172230}
                elif provider == "total":
                    # Independent CPG total/static relation for a matching state.
                    T0 = 400 + 80**2 / (2 * 3.5 * 287.05)
                    p0 = 172230 * (T0 / 400)**3.5
                    bc["left"] = {"type": "characteristic-total-inlet", "total_pressure": p0,
                                   "total_temperature": T0, "direction": [1, 0]}
                elif provider == "reverse":
                    rt["conditions"]["initial_conditions"]["state"]["velocity"] = [-80, 0]
                    bc["left"]["state"] = {"pressure": 172230, "temperature": 700, "velocity": [80, 0]}
                    bc["right"]["state"]["velocity"] = [0, 0]
                if provider == "reference" and ((model == "euler" and ranks == 1) or
                                                  (model == "cns" and ranks == 2)):
                    rt["variable_dt"] = True
                    rt["cfl"] = 0.01
                path.write_text(json.dumps(config, indent=2) + "\n")
        if provider == "reverse":
            path = directory / "expectations.json"
            expected = json.loads(path.read_text())
            expected["velocity"] = [-80, 0]
            path.write_text(json.dumps(expected, indent=2) + "\n")
    case = source / "TestCases/Axisymmetric/NavierStokes/PXChamber"
    for scenario in ("heated", "reverse"):
        for ranks in (1, 2):
            directory = results / f"chamber-{scenario}-{ranks}"
            directory.mkdir(parents=True)
            shutil.copyfile(case / "px_chamber_coarse.msh", directory / "coarse.msh")
            config = json.loads((case / "config-lte-local-characteristic.json").read_text())
            rt = config["runTime"]
            rt.update(mesh_file=str(directory / "coarse.msh"), database_path=str(database),
                      output_file_path=str(directory), dt=1e-8, final_time=1e-6,
                      initial_save_dt=1e-7, vis_steps=10, print_interval=10)
            rt["conditions"]["boundary_conditions"]["Inflow"]["file"] = str(case / "Tuvw_jet_inlet_profile.dat")
            if scenario == "reverse":
                rt["conditions"]["initial_conditions"]["state"]["velocity"] = [-20, 0]
            (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def chamber_fields(directory):
    reader = pv.get_reader(directory / "ParaView/ParaView.pvd")
    assert np.isclose(reader.time_values[-1], 1e-6, atol=1e-18, rtol=1e-12), "Missing final time"
    reader.set_active_time_point(len(reader.time_values) - 1)
    grid = reader.read().combine()
    for name, values in grid.point_data.items():
        assert np.all(np.isfinite(values)), name
    for name in ("Density", "Temperature", "Pressure", "Sound Speed"):
        assert np.min(grid[name]) > 0, name
    coordinates, inverse = np.unique(np.round(grid.points[:, :2], 12), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    fields = {}
    for name, raw in grid.point_data.items():
        values = np.asarray(raw).reshape(grid.n_points, -1)
        fields[name] = np.column_stack([np.bincount(inverse, weights=values[:, d]) / counts
                                        for d in range(values.shape[1])])
    return coordinates, fields


def check(results):
    for provider in ("reference", "radial", "total", "reverse"):
        channel.check(results / provider)
    measurements = {}
    for scenario in ("heated", "reverse"):
        reference = None
        for ranks in (1, 2):
            name = f"chamber-{scenario}-{ranks}"
            xyz, fields = chamber_fields(results / name)
            inlet = (np.abs(xyz[:, 0] - 0.4) < 1e-11) & (xyz[:, 1] < 0.0868 - 1e-11)
            core = inlet & (xyz[:, 1] < 0.04)
            ux = fields["Velocity"][inlet, 0]
            temp = fields["Temperature"][inlet, 0]
            if scenario == "heated":
                assert fields["Temperature"].max() > 301, "No heated-inlet response"
                assert ux.max() > 0, "No incoming velocity response"
                assert ux.min() < -1e-8, "No resolved outgoing trace on the nominal inlet"
            else:
                assert np.max(fields["Velocity"][core, 0]) < -19, "Nominal inlet suppressed outgoing flow"
                assert np.max(fields["Temperature"][core, 0]) < 300.1, "Hot profile overwrote outgoing gas"
            measurements[name] = {"inlet_ux_min": float(ux.min()), "inlet_ux_max": float(ux.max()),
                                  "inlet_T_max": float(temp.max()), "T_max": float(fields["Temperature"].max()),
                                  "incoming_trace_points": int(np.count_nonzero(ux > 1e-8)),
                                  "outgoing_trace_points": int(np.count_nonzero(ux < -1e-8))}
            np.savetxt(results / f"{name}-inlet.csv", np.column_stack((xyz[inlet, 1], ux, temp)),
                       delimiter=",", header="radius,ux,temperature", comments="")
            if reference is None:
                reference = xyz, fields
            else:
                np.testing.assert_array_equal(xyz, reference[0])
                for field in fields:
                    np.testing.assert_allclose(fields[field], reference[1][field], rtol=1e-9, atol=1e-8,
                                               err_msg=f"{scenario}: {field}")
    (results / "chamber-measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
    print("PASS: four uniform channel families plus LTE heated/reverse chamber through 1 microsecond")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source.resolve(), args.results.resolve(), args.database.resolve())
    else:
        check(args.results.resolve())


if __name__ == "__main__":
    main()
