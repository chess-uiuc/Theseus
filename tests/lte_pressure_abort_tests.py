#!/usr/bin/env python3
"""Exercise fatal LTE pressure inversion reporting in an isolated process."""
import argparse
from pathlib import Path
import resource
import subprocess


def disable_core_dump():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    result = subprocess.run([str(args.executable.resolve()), args.backend],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=30, preexec_fn=disable_core_dump)
    print(result.stdout, end='')
    assert f'INVOKING_LTE_PRESSURE_ABORT backend={args.backend}' in result.stdout, result.stdout
    assert result.returncode != 0, 'Pressure inversion failure returned normally'
    if args.backend == 'cpu':
        assert 'Secant method did not converge in internal_energy_from_pressure' in result.stdout
    else:
        assert 'DEVICE_ABORT_OBSERVED:' in result.stdout, result.stdout
    print(f'LTE pressure inversion {args.backend} failure reporting: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
