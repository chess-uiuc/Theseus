#!/usr/bin/env python3
"""Prepare a golden reference without modifying the source snapshot."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def prepare_reference(golden_root, case, destination, expected_file):
    directory = golden_root / case
    archive = golden_root / (case + ".tgz")
    if directory.is_dir():
        shutil.copytree(directory, destination)
    elif archive.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            subprocess.run(["tar", "-xzf", str(archive), "-C", temporary], check=True)
            shutil.move(str(Path(temporary) / case), destination)
    else:
        raise FileNotFoundError(f"Golden reference missing: {directory} or {archive}")

    reference = destination / expected_file
    if not reference.is_file():
        raise FileNotFoundError(f"Expected golden reference file missing: {reference}")
    print(f"Golden reference ready: {reference}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden_root", type=Path)
    parser.add_argument("case")
    parser.add_argument("destination", type=Path)
    parser.add_argument("expected_file")
    args = parser.parse_args()
    prepare_reference(args.golden_root, args.case, args.destination, args.expected_file)


if __name__ == "__main__":
    main()
