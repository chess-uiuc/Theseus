#!/usr/bin/env python3
"""Compare one identical Theseus run across two or more build directories."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from performance_timers import parse_log


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_theseus.sh"
RESERVED_RUNNER_OPTIONS = {"-b", "-e", "-o"}
COMPARISON_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CXX_COMPILER",
    "ENABLE_TIMERS",
    "ENABLE_TIMER_SYNC_DEVICE",
    "ENABLE_TIMER_BARRIER",
    "ENABLE_CUDA",
    "SUBCELL_FV_BLENDING",
    "AXISYMMETRIC",
    "SUTHERLAND",
    "THESEUS_WITH_PLATO",
)


def parse_build(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("build must be LABEL=BUILD_DIRECTORY")
    label, directory = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise argparse.ArgumentTypeError(f"invalid build label: {label!r}")
    if not directory:
        raise argparse.ArgumentTypeError("build directory must not be empty")
    return label, Path(directory).expanduser().resolve()


def read_cmake_cache(build_dir: Path) -> dict[str, str]:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        raise ValueError(f"missing CMake cache: {cache_path}")
    cache: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        cache[key] = value
    return cache


def runner_option_value(arguments: list[str], option: str, default: str) -> str:
    for index, token in enumerate(arguments):
        if token == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"runner option {option} requires a value")
            return arguments[index + 1]
        if token.startswith(option) and len(token) > len(option):
            return token[len(option):]
    return default


def validate_builds(builds: list[tuple[str, Path]], runner_args: list[str]) -> list[dict[str, object]]:
    if len(builds) < 2:
        raise ValueError("at least two --build arguments are required")
    labels = [label for label, _ in builds]
    if len(set(labels)) != len(labels):
        raise ValueError("build labels must be unique")

    for token in runner_args:
        if token in RESERVED_RUNNER_OPTIONS or any(
            token.startswith(option) and token != option for option in RESERVED_RUNNER_OPTIONS
        ):
            raise ValueError(f"{token!r} is controlled by the comparison tool")

    records: list[dict[str, object]] = []
    for label, build_dir in builds:
        executable = build_dir / "theseus"
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"Theseus executable is missing or not executable: {executable}")
        cache = read_cmake_cache(build_dir)
        if cache.get("ENABLE_TIMERS") != "ON":
            raise ValueError(f"build {label!r} does not have ENABLE_TIMERS=ON")
        records.append(
            {
                "label": label,
                "build_directory": str(build_dir),
                "executable": str(executable),
                "cmake": {key: cache.get(key) for key in COMPARISON_CACHE_KEYS},
            }
        )

    reference = records[0]["cmake"]
    for record in records[1:]:
        differences = [
            key for key in COMPARISON_CACHE_KEYS
            if record["cmake"].get(key) != reference.get(key)
        ]
        if differences:
            details = ", ".join(
                f"{key}: {reference.get(key)!r} != {record['cmake'].get(key)!r}"
                for key in differences
            )
            raise ValueError(f"build {record['label']!r} is not comparable: {details}")

    device = runner_option_value(runner_args, "-r", "cpu")
    try:
        ranks = int(runner_option_value(runner_args, "-p", "2"))
    except ValueError as error:
        raise ValueError("runner option -p must be an integer") from error
    for record in records:
        cache = record["cmake"]
        if device in {"cuda", "hip"} and cache.get("ENABLE_TIMER_SYNC_DEVICE") != "ON":
            raise ValueError(
                f"build {record['label']!r} requires ENABLE_TIMER_SYNC_DEVICE=ON for {device}"
            )
        if ranks > 1 and cache.get("ENABLE_TIMER_BARRIER") != "ON":
            raise ValueError(
                f"build {record['label']!r} requires ENABLE_TIMER_BARRIER=ON for {ranks} ranks"
            )
    return records


def percent_change(reference: float, candidate: float) -> float | None:
    return None if reference == 0.0 else 100.0 * (candidate - reference) / reference


def selected_timers(parsed: dict[str, object]) -> tuple[str, dict[str, object]]:
    timers = parsed["timers"]
    scope = "all" if "all" in timers else "0"
    return scope, timers.get(scope, {})


def markdown_report(results: list[dict[str, object]], reference_label: str) -> str:
    by_label = {result["label"]: result for result in results}
    reference = by_label[reference_label]
    lines = ["# Theseus performance comparison", "", f"Reference: `{reference_label}`", ""]
    lines += ["## Timestep performance", "", "| Build | Mean timestep (ms) | Change |", "|---|---:|---:|"]
    ref_step = reference["parsed"].get("timestep") or {}
    ref_value = ref_step.get("critical_mean_timestep_ms")
    for result in results:
        summary = result["parsed"].get("timestep") or {}
        value = summary.get("critical_mean_timestep_ms")
        if value is None or ref_value is None:
            value_text, change_text = "n/a", "n/a"
        else:
            change = percent_change(float(ref_value), float(value))
            value_text = f"{value:.6f}"
            change_text = "—" if result["label"] == reference_label else f"{change:+.2f}%"
        lines.append(f"| `{result['label']}` | {value_text} | {change_text} |")

    ref_scope, ref_timers = selected_timers(reference["parsed"])
    timer_names = sorted({name for result in results for name in selected_timers(result["parsed"])[1]})
    lines += ["", f"## Construct timers (`TIMER({ref_scope})` reference scope)", ""]
    header = "| Timer | " + " | ".join(str(result["label"]) for result in results) + " |"
    lines += [header, "|---|" + "---:|" * len(results)]
    for name in timer_names:
        ref_timer = ref_timers.get(name)
        ref_median = ref_timer.get("median_ms") if ref_timer else None
        cells = []
        for result in results:
            _, timers = selected_timers(result["parsed"])
            timer = timers.get(name)
            if timer is None:
                cells.append("missing")
            elif result["label"] == reference_label or ref_median is None:
                cells.append(f"{timer['median_ms']:.6f} ms ({timer['count']} calls)")
            else:
                change = percent_change(float(ref_median), float(timer["median_ms"]))
                change_text = "n/a" if change is None else f"{change:+.2f}%"
                cells.append(f"{timer['median_ms']:.6f} ms, {change_text} ({timer['count']} calls)")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="append", required=True, type=parse_build,
                        metavar="LABEL=DIRECTORY")
    parser.add_argument("--reference", help="reference build label (default: first build)")
    parser.add_argument("--output", type=Path, help="new result directory")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER,
                        help="arguments passed to run_theseus.sh after --")
    args = parser.parse_args()

    if Path.cwd().resolve() != ROOT:
        parser.error(f"run this command from the repository root: {ROOT}")
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    try:
        builds = validate_builds(args.build, runner_args)
    except ValueError as error:
        parser.error(str(error))
    reference = args.reference or str(builds[0]["label"])
    if reference not in {record["label"] for record in builds}:
        parser.error(f"unknown reference build: {reference!r}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or Path(f"performance-comparison-{timestamp}")).resolve()
    if output.exists():
        parser.error(f"output path already exists: {output}")
    (output / "logs").mkdir(parents=True)
    (output / "runs").mkdir()

    results: list[dict[str, object]] = []
    for record in builds:
        label = str(record["label"])
        run_root = ROOT / f".performance-comparison-{os.getpid()}-{label}"
        log_path = output / "logs" / f"{label}.log"
        command = [
            "/bin/bash", str(RUNNER), "-e", str(record["executable"]),
            "-o", run_root.name, *runner_args,
        ]
        print(f"Running {label}...")
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if run_root.exists():
            shutil.move(str(run_root), output / "runs" / label)
        if completed.returncode != 0:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            raise SystemExit(f"build {label!r} failed with exit code {completed.returncode}:\n{tail}")
        parsed = parse_log(log_path)
        if not parsed["timers"]:
            raise SystemExit(f"build {label!r} produced no detailed timer output")
        if not parsed["timestep"]:
            raise SystemExit(f"build {label!r} produced no timestep summary")
        results.append({**record, "command": command, "parsed": parsed})

    payload = {"reference": reference, "runner_arguments": runner_args, "builds": results}
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown_report(results, reference)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
