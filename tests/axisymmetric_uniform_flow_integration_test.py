#!/usr/bin/env python3
"""Prepare and check exact axisymmetric Euler/CNS uniform-flow recipe data."""

import argparse
import json
import math
import re
from pathlib import Path


RANGE_PATTERN = re.compile(
    r"rho\(([^,]+),([^\)]+)\), p\(([^,]+),([^\)]+)\)"
)
CHANGE_PATTERN = re.compile(
    r"TotalChange: Mass: ([^,]+), Energy: ([^,]+), K\.E\.: ([^\s]+)"
)


CASES = {
    "euler": Path("TestCases/Axisymmetric/Euler/UniformFlow/config.json"),
    "navier-stokes": Path(
        "TestCases/Axisymmetric/NavierStokes/UniformFlow/config.json"
    ),
}


def prepare_case(source: Path, results: Path, name: str, case: Path,
                 ranks: int) -> None:
    config = json.loads((source / case).read_text())
    runtime = config["runTime"]
    runtime["mesh_file"] = str(
        (source / "TestCases/NavierStokes/2D/LidDrivenCavity/"
         "LidDrivenCavity.msh").resolve()
    )
    runtime["print_interval"] = 1
    directory = results / f"{name}-{ranks}"
    directory.mkdir(parents=True)
    runtime["output_file_path"] = str(directory)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def read_case(results: Path, name: str, ranks: int) -> tuple[tuple[float, ...],
                                                              tuple[float, ...]]:
    log_path = results / f"{name}-{ranks}" / "run.log"
    if not log_path.is_file():
        raise RuntimeError(f"missing simulation log: {log_path}")
    output = log_path.read_text()
    print(f"{name} ({ranks} ranks): {log_path}")
    matches = RANGE_PATTERN.findall(output)
    changes = CHANGE_PATTERN.findall(output)
    if not matches or not changes:
        raise RuntimeError(
            f"{name} ({ranks} ranks) did not report ranges and changes:\n"
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
    parser.add_argument("action", choices=("prepare", "check"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    results = args.results.resolve()
    if args.action == "prepare":
        results.mkdir(parents=True, exist_ok=True)
        for name, case in CASES.items():
            for ranks in (1, 2):
                prepare_case(source, results, name, case, ranks)
        return

    expected = (1.0, 1.0, 1.0/1.4, 1.0/1.4)
    for name, case in CASES.items():
        serial_range, serial_change = read_case(results, name, 1)
        parallel_range, parallel_change = read_case(results, name, 2)
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
