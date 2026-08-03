#!/usr/bin/env python3
"""Run a cylindrical L2 convergence study for an exact entropy wave."""

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


ERROR_PATTERN = re.compile(r"Exact L2 Error: ([^\s]+)")


def run_level(executable: Path, source: Path, mpiexec: Path,
              numproc_flag: str, level: int) -> float:
    case = source / "TestCases/Axisymmetric/Euler/EntropyWave/config.json"
    config = json.loads(case.read_text())
    runtime = config["runTime"]
    runtime["ser_ref_levels"] = level
    runtime["mesh_file"] = str(
        (source / "TestCases/NavierStokes/2D/LidDrivenCavity/"
         "LidDrivenCavity.msh").resolve()
    )
    with tempfile.TemporaryDirectory(prefix="theseus-axis-convergence-") as tmp:
        runtime["output_file_path"] = tmp
        config_path = Path(tmp) / "config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        command = [str(mpiexec), numproc_flag, "1", str(executable),
                   "-d", "cpu", "-c", str(config_path)]
        result = subprocess.run(
            command, cwd=tmp, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=180, check=False
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"refinement level {level} exited with {result.returncode}:\n"
            f"{result.stdout}"
        )
    matches = ERROR_PATTERN.findall(result.stdout)
    if not matches:
        raise RuntimeError(
            f"refinement level {level} did not report an L2 error:\n"
            f"{result.stdout}"
        )
    return float(matches[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument("--numproc-flag", required=True)
    args = parser.parse_args()

    errors = [
        run_level(args.executable.resolve(), args.source.resolve(),
                  args.mpiexec.resolve(), args.numproc_flag, level)
        for level in range(3)
    ]
    rates = [math.log(errors[index]/errors[index + 1], 2.0)
             for index in range(2)]
    print(f"axisymmetric entropy-wave L2 errors: {errors}")
    print(f"axisymmetric entropy-wave rates: {rates}")
    if not all(errors[index + 1] < errors[index] for index in range(2)):
        raise RuntimeError(f"L2 errors are not decreasing: {errors}")
    if min(rates) < 1.7:
        raise RuntimeError(
            f"expected approximately second-order convergence: {rates}"
        )


if __name__ == "__main__":
    main()
