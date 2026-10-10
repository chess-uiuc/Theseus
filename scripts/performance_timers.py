#!/usr/bin/env python3
"""Parse and summarize Theseus runtime timer output."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMER_RE = re.compile(
    rf"^\[TIMER\((?P<scope>[^)]+)\)\]\s+(?P<name>.+?)\s*:\s*"
    rf"(?P<value>{NUMBER})\s+ms\s*$"
)
TRIPLE_RE = re.compile(
    rf"^\s*(?P<label>Timed steps|Total time \(ms\)|Min step \(ms\)|"
    rf"Mean step \(ms\)|Max step \(ms\))\s*:\s*"
    rf"(?P<minimum>{NUMBER})\s+(?P<mean>{NUMBER})\s+(?P<maximum>{NUMBER})\s*$"
)


def _aggregate(samples: list[float]) -> dict[str, object]:
    return {
        "count": len(samples),
        "total_ms": sum(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def parse_lines(lines: Iterable[str], skip_steps: int = 3) -> dict[str, object]:
    if skip_steps < 0:
        raise ValueError("skip_steps must be nonnegative")
    input_lines = [line.rstrip("\n") for line in lines]
    timestep_indices = [
        index for index, line in enumerate(input_lines)
        if (match := TIMER_RE.match(line))
        and match.group("scope") == "0"
        and match.group("name") == "Timestep"
    ]
    if skip_steps == 0:
        cutoff = -1
    elif skip_steps <= len(timestep_indices):
        cutoff = timestep_indices[skip_steps - 1]
    else:
        cutoff = len(input_lines)

    samples: dict[str, dict[str, list[float]]] = {}
    timestep: dict[str, object] = {}

    triple_names = {
        "Timed steps": "timed_steps",
        "Total time (ms)": "total_time_ms",
        "Min step (ms)": "minimum_step_ms",
        "Mean step (ms)": "mean_step_ms",
        "Max step (ms)": "maximum_step_ms",
    }

    for index, line in enumerate(input_lines):
        match = TIMER_RE.match(line)
        if match:
            scope = match.group("scope")
            name = match.group("name")
            if index <= cutoff:
                continue
            samples.setdefault(scope, {}).setdefault(name, []).append(
                float(match.group("value"))
            )
            continue

        match = TRIPLE_RE.match(line)
        if match:
            values: dict[str, float | int] = {
                "minimum": float(match.group("minimum")),
                "mean": float(match.group("mean")),
                "maximum": float(match.group("maximum")),
            }
            if match.group("label") == "Timed steps":
                values["minimum"] = int(values["minimum"])
                values["maximum"] = int(values["maximum"])
            timestep[triple_names[match.group("label")]] = values
            continue

        stripped = line.strip()
        scalar_patterns = (
            (r"^MPI ranks\s*:\s*(\d+)$", "mpi_ranks", int),
            (rf"^Mean timestep\s*:\s*({NUMBER})\s+ms$", "critical_mean_timestep_ms", float),
            (rf"^Timesteps/sec\s*:\s*({NUMBER})$", "timesteps_per_second", float),
        )
        for pattern, key, conversion in scalar_patterns:
            scalar = re.match(pattern, stripped)
            if scalar:
                timestep[key] = conversion(scalar.group(1))
                break
        else:
            setting = re.match(r"^(Device sync|MPI barrier)\s*:\s*(enabled|disabled)$", stripped)
            if setting:
                key = setting.group(1).lower().replace(" ", "_")
                timestep[key] = setting.group(2) == "enabled"

    timers = {
        scope: {name: _aggregate(values) for name, values in sorted(named.items())}
        for scope, named in sorted(samples.items())
    }
    return {
        "analysis": {
            "skip_steps": skip_steps,
            "step_markers_present": bool(timestep_indices),
            "observed_steps": len(timestep_indices),
        },
        "timers": timers,
        "timestep": timestep or None,
    }


def parse_log(path: Path, skip_steps: int = 3) -> dict[str, object]:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        result = parse_lines(stream, skip_steps=skip_steps)
    result["source"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Theseus stdout/stderr log")
    parser.add_argument("-o", "--output", type=Path, help="write JSON to this path")
    parser.add_argument("--skip-steps", type=int, default=3,
                        help="warm-up timesteps excluded from analysis (default: 3)")
    args = parser.parse_args()

    if args.skip_steps < 0:
        parser.error("--skip-steps must be nonnegative")
    encoded = json.dumps(
        parse_log(args.log, skip_steps=args.skip_steps), indent=2, sort_keys=True
    ) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
