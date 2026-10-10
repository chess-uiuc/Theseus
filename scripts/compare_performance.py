#!/usr/bin/env python3
"""Compare Theseus performance from builds or existing timer logs."""

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


def parse_log_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("log must be LABEL=LOG_FILE")
    label, filename = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise argparse.ArgumentTypeError(f"invalid label: {label!r}")
    if not filename:
        raise argparse.ArgumentTypeError("log file must not be empty")
    return label, Path(filename).expanduser().resolve()


def validate_logs(logs: list[tuple[str, Path]]) -> list[dict[str, object]]:
    if len(logs) < 2:
        raise ValueError("at least two --log arguments are required")
    labels = [label for label, _ in logs]
    if len(set(labels)) != len(labels):
        raise ValueError("log labels must be unique")

    records = []
    for label, path in logs:
        if not path.is_file():
            raise ValueError(f"log file does not exist: {path}")
        records.append({"label": label, "source_log": str(path)})
    return records


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


def analyzed_timestep_ms(parsed: dict[str, object]) -> float | None:
    _, timers = selected_timers(parsed)
    timer = timers.get("Timestep")
    if timer:
        return float(timer["mean_ms"])
    summary = parsed.get("timestep") or {}
    value = summary.get("critical_mean_timestep_ms")
    return None if value is None else float(value)


def markdown_report(
    results: list[dict[str, object]], reference_label: str, input_mode: str = "builds"
) -> str:
    by_label = {result["label"]: result for result in results}
    reference = by_label[reference_label]
    skip_steps = reference["parsed"].get("analysis", {}).get("skip_steps", 0)
    lines = [
        "# Theseus performance comparison", "", f"Reference: `{reference_label}`",
        f"Warm-up timesteps excluded: {skip_steps}", "",
    ]
    input_heading = "Log" if input_mode == "logs" else "Build"
    lines += [
        "## Timestep performance", "",
        f"| {input_heading} | Analyzed mean timestep (ms) | Change |",
        "|---|---:|---:|",
    ]
    ref_value = analyzed_timestep_ms(reference["parsed"])
    for result in results:
        value = analyzed_timestep_ms(result["parsed"])
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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--build", action="append", type=parse_build,
                        metavar="LABEL=DIRECTORY",
                        help="run and compare a labeled build directory (repeatable)")
    inputs.add_argument("--log", action="append", type=parse_log_input,
                        metavar="LABEL=LOG_FILE",
                        help="compare an existing labeled timer log (repeatable)")
    parser.add_argument("--reference", help="reference build label (default: first build)")
    parser.add_argument("--output", type=Path, help="new result directory")
    parser.add_argument("--skip-steps", type=int, default=3,
                        help="warm-up timesteps excluded from analysis (default: 3)")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER,
                        help="arguments passed to run_theseus.sh after --")
    args = parser.parse_args()

    if args.skip_steps < 0:
        parser.error("--skip-steps must be nonnegative")
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    input_mode = "logs" if args.log is not None else "builds"
    try:
        if input_mode == "logs":
            if runner_args:
                raise ValueError("run_theseus arguments after -- are not valid with --log")
            records = validate_logs(args.log)
        else:
            if Path.cwd().resolve() != ROOT:
                raise ValueError(f"run build comparisons from the repository root: {ROOT}")
            records = validate_builds(args.build, runner_args)
    except ValueError as error:
        parser.error(str(error))
    reference = args.reference or str(records[0]["label"])
    if reference not in {record["label"] for record in records}:
        parser.error(f"unknown reference label: {reference!r}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or Path(f"performance-comparison-{timestamp}")).resolve()
    if output.exists():
        parser.error(f"output path already exists: {output}")
    (output / "logs").mkdir(parents=True)
    if input_mode == "builds":
        (output / "runs").mkdir()

    results: list[dict[str, object]] = []
    for record in records:
        label = str(record["label"])
        log_path = output / "logs" / f"{label}.log"
        result_record = dict(record)
        if input_mode == "logs":
            shutil.copy2(record["source_log"], log_path)
        else:
            run_root = ROOT / f".performance-comparison-{os.getpid()}-{label}"
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
                raise SystemExit(
                    f"build {label!r} failed with exit code {completed.returncode}:\n{tail}"
                )
            result_record["command"] = command
        parsed = parse_log(log_path, skip_steps=args.skip_steps)
        if not parsed["timers"]:
            raise SystemExit(f"input {label!r} contains no analyzed detailed timer output")
        if analyzed_timestep_ms(parsed) is None:
            raise SystemExit(f"input {label!r} contains no analyzed timestep timing")
        results.append({**result_record, "parsed": parsed})

    payload: dict[str, object] = {
        "input_mode": input_mode,
        "reference": reference,
        "runner_arguments": runner_args,
        "skip_steps": args.skip_steps,
    }
    payload[input_mode] = results
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown_report(results, reference, input_mode=input_mode)
    (output / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
