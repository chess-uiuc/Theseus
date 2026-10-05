#!/usr/bin/env python3
"""Run the CPG or LTE chamber startup from any directory; optionally use a coarse mesh."""
import argparse
import json
from pathlib import Path
import subprocess

CASE = Path(__file__).resolve().parent


def coarse_mesh(path):
    # Keep the inlet/wall junction exact, and cluster points near the hot jet.
    xs = [.4 + .65*i/24 for i in range(25)]
    ys = [.0868*(i/20)**1.5 for i in range(21)]
    ys += [.0868 + (1-.0868)*(i/16)**1.5 for i in range(1,17)]
    nx, ny = len(xs), len(ys)
    node = lambda i,j: 1+j*nx+i
    elements=[]
    def add(kind,tag,ids):
        elements.append(f'{len(elements)+1} {kind} 2 {tag} {tag} '+ ' '.join(map(str,ids)))
    for j in range(ny-1):
        for i in range(nx-1):
            add(3,1,[node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)])
    for i in range(nx-1):
        add(1,2,[node(i,0),node(i+1,0)])
        add(1,6,[node(i+1,ny-1),node(i,ny-1)])
    for j in range(ny-1):
        add(1,3 if j<20 else 4,[node(0,j+1),node(0,j)])
        add(1,5,[node(nx-1,j),node(nx-1,j+1)])
    lines=['$MeshFormat','2.2 0 8','$EndMeshFormat','$PhysicalNames','6',
           '2 1 "Unspecified"','1 2 "Axis"','1 3 "Inflow"','1 4 "Left"',
           '1 5 "Right"','1 6 "Top"','$EndPhysicalNames','$Nodes',str(nx*ny)]
    lines += [f'{node(i,j)} {x:.17g} {y:.17g} 0' for j,y in enumerate(ys) for i,x in enumerate(xs)]
    lines += ['$EndNodes','$Elements',str(len(elements)),*elements,'$EndElements']
    path.write_text('\n'.join(lines)+'\n')


def prepare(output, coarse=False, final_time=1e-7, dt=1e-9,
            gas_model="cpg", database=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if gas_model not in ("cpg", "lte"):
        raise ValueError("Expected gas_model cpg or lte")
    if gas_model == "lte" and database is None:
        raise ValueError("LTE requires the PLATO database path")

    config_name = "config-lte.json" if gas_model == "lte" else "config.json"
    config = json.loads((CASE / config_name).read_text())
    runtime = config["runTime"]
    runtime["mesh_file"] = str(CASE / "px_chamber_axi.msh")
    if coarse:
        coarse_mesh(output / "coarse.msh")
        runtime["mesh_file"] = str(output / "coarse.msh")
    runtime["conditions"]["boundary_conditions"]["Inflow"]["file"] = str(
        CASE / "Tuvw_jet_inlet_profile.dat")
    if database is not None:
        runtime["database_path"] = str(Path(database).resolve())
    runtime.update(output_file_path=str(output), final_time=final_time, dt=dt,
                   initial_save_dt=final_time / 10, print_interval=100, vis_steps=1000)
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--coarse", action="store_true")
    parser.add_argument("--final-time", type=float, default=1e-7)
    parser.add_argument("--dt", type=float, default=1e-9)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--gas-model", choices=("cpg", "lte"), default="cpg")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.gas_model == "lte" and args.database is None:
        parser.error("--gas-model lte requires --database")

    config = prepare(args.output, args.coarse, args.final_time, args.dt,
                     args.gas_model, args.database)
    command = [args.mpiexec]
    version = subprocess.run(command + ["--version"], capture_output=True, text=True).stdout
    if "Open MPI" in version or "OpenRTE" in version:
        command += ["--host", f"localhost:{args.ranks}", "--map-by", "slot:OVERSUBSCRIBE",
                    "--bind-to", "none"]
    command += ["-n", str(args.ranks), str(args.executable.resolve()),
                "-d", args.device, "-c", str(config)]
    with (config.parent / "run.log").open("w") as log:
        subprocess.run(command, cwd=config.parent, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Finished:", config.parent / "ParaView/ParaView.pvd")


if __name__ == "__main__":
    main()
