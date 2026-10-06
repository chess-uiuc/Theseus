#!/usr/bin/env python3
"""Check exact axisymmetric Euler/CNS uniform flow in serial and MPI."""

import argparse
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


RANGE_PATTERN = re.compile(
    r"rho\(([^,]+),([^\)]+)\), p\(([^,]+),([^\)]+)\)"
)
CHANGE_PATTERN = re.compile(
    r"TotalChange: Mass: ([^,]+), Energy: ([^,]+), K\.E\.: ([^\s]+)"
)


def run_case(executable: Path, source: Path, results: Path, case: Path,
             ranks: int, device: str) -> tuple[tuple[float, ...],
                                                tuple[float, ...]]:
    config = json.loads((source / case).read_text())
    runtime = config["runTime"]
    runtime["mesh_file"] = str(
        (source / "TestCases/NavierStokes/2D/LidDrivenCavity/"
         "LidDrivenCavity.msh").resolve()
    )
    runtime["print_interval"] = 1

    directory = Path(tempfile.mkdtemp(prefix=f"{case.parts[2]}-{ranks}-", dir=results))
    runtime["output_file_path"] = str(directory)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = ["bash", str(source / "scripts/run_theseus.sh"),
               "-e", str(executable), "-c", str(config_path),
               "-o", str(directory / "launch"), "-p", str(ranks),
               "-r", device, "-P", "-k"]
    log_path = directory / "run.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=source, text=True, stdout=log,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
    output = log_path.read_text()
    print(f"{case} ({ranks} ranks): {log_path}")
    if result.returncode != 0:
        raise RuntimeError(
            f"{case} ({ranks} ranks) exited with {result.returncode}:\n"
            f"{output}"
        )
    matches = RANGE_PATTERN.findall(output)
    changes = CHANGE_PATTERN.findall(output)
    if not matches or not changes:
        raise RuntimeError(
            f"{case} ({ranks} ranks) did not report ranges and changes:\n"
            f"{output}"
        )
    return (tuple(float(value) for value in matches[-1]),
            tuple(float(value) for value in changes[-1]))


def assert_close(actual: float, expected: float, tolerance: float,
                 description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(
            f"{description}: expected {expected:.16e}, got {actual:.16e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=Path("uniform-flow-results"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)

    expected = (1.0, 1.0, 1.0/1.4, 1.0/1.4)
    cases = (
        Path("TestCases/Axisymmetric/Euler/UniformFlow/config.json"),
        Path("TestCases/Axisymmetric/NavierStokes/UniformFlow/config.json"),
    )
    for case in cases:
        serial_range, serial_change = run_case(
            args.executable.resolve(), args.source.resolve(),
            results, case, 1, args.device)
        parallel_range, parallel_change = run_case(
            args.executable.resolve(), args.source.resolve(),
            results, case, 2, args.device)
        for label, values in (("serial", serial_range),
                              ("two-rank", parallel_range)):
            for index, (actual, target) in enumerate(zip(values, expected)):
                assert_close(actual, target, 5.0e-7,
                             f"{case} {label} range component {index}")
        for label, changes in (("serial", serial_change),
                               ("two-rank", parallel_change)):
            for index, change in enumerate(changes):
                assert_close(change, 0.0, 1.0e-9,
                             f"{case} {label} change component {index}")
        for index, (one, two) in enumerate(zip(serial_range, parallel_range)):
            assert_close(two, one, 2.0e-13,
                         f"{case} serial/MPI range component {index}")


if __name__ == "__main__":
    main()
