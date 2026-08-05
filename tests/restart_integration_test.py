#!/usr/bin/env python3
"""End-to-end checkpoint save/restart equivalence test."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(executable: Path, config: Path, cwd: Path, mpiexec: Path, numproc_flag: str) -> None:
    version = subprocess.run(
        [str(mpiexec), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    launcher = [str(mpiexec)]
    if "Open MPI" in version or "OpenRTE" in version:
        launcher.extend(
            ["--host", "localhost:2", "--map-by", "slot:OVERSUBSCRIBE", "--bind-to", "none"]
        )
    result = subprocess.run(
        launcher
        + [numproc_flag, "2", str(executable), "-d", "cpu", "-c", str(config)],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Theseus exited with {result.returncode}:\n{result.stdout}")


def write_config(base: dict, path: Path, output: Path, *, load: bool) -> None:
    config = json.loads(json.dumps(base))
    runtime = config["runTime"]
    runtime.update(
        {
            "mesh_file": str(Path(runtime["mesh_file"]).resolve()),
            "output_file_path": str(output),
            "visualize": False,
            "checkpoint_save": True,
            "checkpoint_load": load,
            "checkpoint_cycle": 1,
            "checkpoint_dt": 0.001,
            "variable_dt": False,
            "dt": 0.001,
            "final_time": 1.0,
            "nsteps_max": 2,
        }
    )
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument("--numproc-flag", required=True)
    args = parser.parse_args()

    case = args.source / "TestCases/Euler/2D/IsentropicVortex"
    base = json.loads((case / "config.json").read_text(encoding="utf-8"))
    # Make the case mesh independent of this test's temporary working directory.
    original_mesh = base["runTime"]["mesh_file"]
    base["runTime"]["mesh_file"] = str((case / Path(original_mesh).name).resolve())

    with tempfile.TemporaryDirectory(prefix="theseus-restart-") as tempdir:
        root = Path(tempdir)
        continuous = root / "continuous"
        restarted = root / "restarted"
        continuous.mkdir()
        restarted.mkdir()

        continuous_config = root / "continuous.json"
        restart_config = root / "restart.json"
        write_config(base, continuous_config, continuous, load=False)
        run(args.executable.resolve(), continuous_config, root,
            args.mpiexec.resolve(), args.numproc_flag)

        source_cycle = continuous / "Checkpoints/Cycle1"
        if not source_cycle.is_dir():
            raise RuntimeError(f"Initial run did not create {source_cycle}")
        shutil.copytree(source_cycle, restarted / "Checkpoints/Cycle1")

        write_config(base, restart_config, restarted, load=True)
        run(args.executable.resolve(), restart_config, root,
            args.mpiexec.resolve(), args.numproc_flag)

        for rank in range(2):
            filename = f"checkpoint_cycle_2.{rank:08d}.chk"
            continuous_state = continuous / "Checkpoints/Cycle2" / filename
            restarted_state = restarted / "Checkpoints/Cycle2" / filename
            if continuous_state.read_bytes() != restarted_state.read_bytes():
                raise RuntimeError(
                    f"Restarted rank {rank} state differs from the uninterrupted state at cycle 2"
                )

        metadata = json.loads(
            (restarted / "Checkpoints/Cycle2/checkpoint_cycle_2.json").read_text()
        )
        required = {
            "format_version",
            "state_format",
            "real_bytes",
            "mpi_ranks",
            "order",
            "dimension",
            "num_equations",
            "global_elements",
            "global_dofs",
        }
        missing = required.difference(metadata)
        if missing:
            raise RuntimeError(f"Restart metadata is missing: {sorted(missing)}")


if __name__ == "__main__":
    main()
