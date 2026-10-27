#!/usr/bin/env python3
"""Prepare and check a cylindrical L2 entropy-wave convergence recipe."""

import argparse
import json
import math
import re
from pathlib import Path

ERROR_PATTERN = re.compile(r"Exact L2 Error: ([^\s]+)")


def prepare_level(source: Path, results: Path, level: int) -> None:
    case = source / "TestCases/Axisymmetric/Euler/EntropyWave/config.json"
    config = json.loads(case.read_text())
    runtime = config["runTime"]
    runtime["ser_ref_levels"] = level
    runtime["mesh_file"] = str(
        (source / "TestCases/NavierStokes/2D/LidDrivenCavity/"
         "LidDrivenCavity.msh").resolve()
    )
    directory = results / f"level-{level}"
    directory.mkdir(parents=True)
    runtime["output_file_path"] = str(directory)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def read_level(results: Path, level: int) -> float:
    log_path = results / f"level-{level}" / "run.log"
    if not log_path.is_file():
        raise RuntimeError(f"missing simulation log: {log_path}")
    output = log_path.read_text()
    matches = ERROR_PATTERN.findall(output)
    if not matches:
        raise RuntimeError(
            f"refinement level {level} did not report an L2 error:\n{output}"
        )
    return float(matches[-1])


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
        for level in range(3):
            prepare_level(source, results, level)
        return

    errors = [read_level(results, level) for level in range(3)]
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
