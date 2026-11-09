#!/usr/bin/env python3
"""Inspect or compare host-boundary array captures without loading whole runs into RAM."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


CONTEXT = ("time", "dimension", "equations", "order", "gas", "flux", "ranks")


def load_capture(directory, require_complete=True):
    directory = Path(directory)
    paths = sorted(directory.glob("rank*.jsonl"))
    if not paths:
        raise ValueError(f"No manifests in {directory}")
    arrays = {}
    incomplete = []
    requested = set()
    completed = set()
    for path in paths:
        active = None
        count = 0
        seen_calls = set()
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid manifest {path}:{line_number}: {error}") from error
                if record.get("schema") != 1:
                    raise ValueError(f"Unsupported schema in {path}:{line_number}")
                rank, call = record["rank"], record["call"]
                if (type(rank) is not int or rank < 0 or type(call) is not int or call < 0
                        or path.name != f"rank{rank:06d}.jsonl"):
                    raise ValueError(f"Invalid rank/call in {path}")
                key = (rank, call)
                kind = record.get("kind")
                if kind == "begin":
                    if active is not None or call in seen_calls:
                        raise ValueError(f"Nested/duplicate call in {path}")
                    active = record
                    seen_calls.add(call)
                    count = 0
                    config = record["configuration"]
                    requested.update((r, c) for r in config["ranks"] for c in config["calls"])
                elif kind == "array":
                    if active is None or key != (active["rank"], active["call"]):
                        raise ValueError(f"Array outside its call in {path}")
                    if any(record.get(field) != active.get(field) for field in CONTEXT):
                        raise ValueError(f"Changed call context in {path}")
                    shape, axes = record["shape"], record["axes"]
                    if (not shape or len(shape) != len(axes)
                            or any(type(n) is not int or n < 0 for n in shape)
                            or math.prod(shape) != record["size"]):
                        raise ValueError(f"Invalid shape in {path}")
                    if record["dtype"] not in ("<f4", ">f4", "<f8", ">f8"):
                        raise ValueError(f"Unsupported dtype in {path}")
                    if record["sequence"] != count:
                        raise ValueError(f"Invalid array order in {path}")
                    array_key = (*key, record["name"])
                    if array_key in arrays:
                        raise ValueError(f"Duplicate array {array_key}")
                    snapshot = record["snapshot"]
                    if snapshot is not None:
                        if not isinstance(snapshot, str) or Path(snapshot).name != snapshot:
                            raise ValueError(f"Invalid snapshot filename in {path}")
                        binary = directory / snapshot
                        expected = record["size"] * np.dtype(record["dtype"]).itemsize
                        if not binary.is_file() or binary.stat().st_size != expected:
                            raise ValueError(f"Missing/truncated/oversized snapshot: {binary}")
                    arrays[array_key] = record
                    count += 1
                elif kind == "end":
                    if active is None or key != (active["rank"], active["call"]) or count != record["arrays"]:
                        raise ValueError(f"Invalid end marker in {path}")
                    completed.add(key)
                    active = None
                else:
                    raise ValueError(f"Unknown record kind in {path}")
        if active is not None:
            incomplete.append((active["rank"], active["call"]))
    missing = requested - completed
    if require_complete and (missing or incomplete):
        raise ValueError(f"Incomplete capture; missing completed rank/call pairs: {sorted(missing)}")
    if require_complete and not arrays:
        raise ValueError("Capture contains no selected arrays")
    return arrays, sorted(missing)


def coordinate(record, flat_index):
    if flat_index is None:
        return None
    result = {}
    remainder = int(flat_index)
    for axis, extent in reversed(list(zip(record["axes"], record["shape"]))):
        result[axis] = remainder % extent
        remainder //= extent
    return {axis: result[axis] for axis in record["axes"]}


def read_array(directory, record):
    if record["snapshot"] is None:
        raise ValueError(f"No snapshot for {record['name']}; summary-only captures cannot be compared")
    if record["size"] == 0:
        return np.empty(0, dtype=record["dtype"])
    return np.memmap(Path(directory) / record["snapshot"], dtype=record["dtype"], mode="r")


def compare_array(left_dir, right_dir, left, right, atol, rtol, chunk_values):
    for field in (*CONTEXT, "shape", "axes", "dtype", "sequence"):
        if left[field] != right[field]:
            raise ValueError(f"Incompatible {left['name']} {field}: {left[field]} vs {right[field]}")
    a, b = read_array(left_dir, left), read_array(right_dir, right)
    mismatches = nonfinite = bit_differences = 0
    first = worst = None
    maximum = 0.0
    for start in range(0, a.size, chunk_values):
        aa = np.asarray(a[start:start + chunk_values])
        bb = np.asarray(b[start:start + chunk_values])
        finite = np.isfinite(aa) & np.isfinite(bb)
        nonfinite += int(np.count_nonzero(~finite))
        unsigned = np.dtype(left["dtype"].replace("f", "u"))
        bit_differences += int(np.count_nonzero(aa.view(unsigned) != bb.view(unsigned)))
        # Use float64 arithmetic even for float32 data. Overflow is a reported infinite error.
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
            scale = np.maximum(np.abs(aa.astype(np.float64)), np.abs(bb.astype(np.float64)))
            tolerance = atol + rtol * scale
            bad = ~finite | (difference > tolerance)
        indices = np.flatnonzero(bad)
        mismatches += int(indices.size)
        if first is None and indices.size:
            first = start + int(indices[0])
        finite_diff = np.where(finite, difference, -1.0)
        if finite_diff.size:
            index = int(np.argmax(finite_diff))
            value = float(finite_diff[index])
            if value > maximum:
                maximum, worst = value, start + index
    return {"mismatches": mismatches, "nonfinite_pairs": nonfinite,
            "bit_differences": bit_differences,
            "max_absolute_error": maximum if math.isfinite(maximum) else "infinity",
            "first_flat_index": first, "first_coordinate": coordinate(left, first),
            "left_value": repr(float(a[first])) if first is not None else None,
            "right_value": repr(float(b[first])) if first is not None else None,
            "worst_finite_coordinate": coordinate(left, worst)}


def compare(left, right, atol=0.0, rtol=0.0, names=None, rank=None, chunk_values=1048576):
    a, _ = load_capture(left)
    b, _ = load_capture(right)
    if set(a) != set(b):
        raise ValueError(f"Missing/different arrays: left-only={sorted(set(a)-set(b))}, "
                         f"right-only={sorted(set(b)-set(a))}")
    results = []
    for key in sorted(a, key=lambda key: (key[1], key[0], a[key]["sequence"])):
        r, call, name = key
        if (names and name not in names) or (rank is not None and r != rank):
            continue
        result = compare_array(left, right, a[key], b[key], atol, rtol, chunk_values)
        results.append({"rank": r, "call": call, "array": name, **result})
    if not results:
        raise ValueError("No matching selected arrays")
    first = next((record for record in results if record["mismatches"]), None)
    return {"status": "different" if first else "equal", "atol": atol, "rtol": rtol,
            "first_divergence": first, "arrays": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="report summaries, including partial failed runs")
    inspect.add_argument("directory", type=Path)
    compare_parser = commands.add_parser("compare", help="compare matching complete snapshots")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--atol", type=float, default=0.0)
    compare_parser.add_argument("--rtol", type=float, default=0.0)
    compare_parser.add_argument("--array", action="append")
    compare_parser.add_argument("--rank", type=int)
    compare_parser.add_argument("--chunk-values", type=int, default=1048576)
    for sub in (inspect, compare_parser):
        sub.add_argument("--json", type=Path, dest="json_output")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            arrays, missing = load_capture(args.directory, require_complete=False)
            if not arrays:
                raise ValueError("Capture contains no selected arrays")
            records = []
            for key, record in arrays.items():
                r, call, name = key
                summary = record["summary"]
                print(f"rank={r} call={call} {name}: size={record['size']} "
                      f"nonfinite={summary['nonfinite']} finite-range="
                      f"[{summary['minimum']}, {summary['maximum']}]")
                for sample in summary["invalid_samples"]:
                    print(f"  {coordinate(record, sample['index'])}: {sample['value']}")
                records.append(record)
            report = {"status": "partial" if missing else "complete", "missing_calls": missing,
                      "arrays": records}
            code = 1 if missing or any(r["summary"]["nonfinite"] for r in records) else 0
            if missing:
                print(f"INCOMPLETE rank/calls: {missing}")
        else:
            if (not math.isfinite(args.atol) or not math.isfinite(args.rtol)
                    or args.atol < 0 or args.rtol < 0 or args.chunk_values < 1):
                raise ValueError("Tolerances must be finite/nonnegative; chunk-values must be positive")
            report = compare(args.left, args.right, args.atol, args.rtol,
                             args.array, args.rank, args.chunk_values)
            for record in report["arrays"]:
                print(f"rank={record['rank']} call={record['call']} {record['array']}: "
                      f"mismatches={record['mismatches']} bit-differences={record['bit_differences']} "
                      f"max-abs={record['max_absolute_error']}")
            if report["first_divergence"]:
                first = report["first_divergence"]
                print(f"FIRST DIVERGENCE: rank={first['rank']} call={first['call']} "
                      f"{first['array']} {first['first_coordinate']}: "
                      f"{first['left_value']} vs {first['right_value']}")
            code = int(report["status"] != "equal")
        if args.json_output:
            args.json_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return code
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
