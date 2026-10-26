#!/usr/bin/env python3
"""Compare diagnostic snapshots from identical one-rank PX Chamber runs."""
import argparse
from pathlib import Path

import numpy as np


def read(prefix, call, phase, dtype, raw=False):
    path = Path(f"{prefix}.rhs{call}.{phase}.bin")
    data = np.fromfile(path, dtype=dtype)
    if not data.size or (not raw and data.size % 4):
        raise ValueError(f"Invalid four-equation snapshot: {path}")
    return data if raw else data.reshape(4, -1)


def report(label, cpu, cuda):
    if cpu.shape != cuda.shape:
        raise ValueError(f"Shape mismatch for {label}: {cpu.shape} vs {cuda.shape}")
    for equation, name in enumerate(("rho", "mx", "mr", "rhoE")):
        a, b = cpu[equation], cuda[equation]
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            print(f"  {label} {name}: NONFINITE VALUES")
            continue
        difference = np.abs(a - b)
        index = int(np.argmax(difference))
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        relative = float(difference[index]) / scale if scale else 0.0
        print(f"  {label} {name}: max_abs={difference[index]:.12e} "
              f"scaled={relative:.6e} dof={index} "
              f"cpu={a[index]:.12e} cuda={b[index]:.12e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_prefix")
    parser.add_argument("cuda_prefix")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--calls", type=int, default=6)
    parser.add_argument("--raw-phases", nargs="+")
    args = parser.parse_args()
    for call in range(args.calls):
        print(f"RHS {call}: RK3 step={call // 3}, stage={call % 3 + 1}")
        if args.raw_phases:
            for phase in args.raw_phases:
                a = read(args.cpu_prefix, call, phase, args.dtype, raw=True)
                b = read(args.cuda_prefix, call, phase, args.dtype, raw=True)
                if a.shape != b.shape:
                    raise ValueError(f"Shape mismatch: {phase}")
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    print(f"  {phase}: NONFINITE VALUES")
                    continue
                difference = np.abs(a - b)
                index = int(np.argmax(difference))
                scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
                relative = float(difference[index]) / scale if scale else 0.0
                print(f"  {phase}: max_abs={difference[index]:.12e} "
                      f"scaled={relative:.6e} flat_index={index} "
                      f"cpu={a[index]:.12e} cuda={b[index]:.12e}")
            continue
        cpu, cuda = {}, {}
        for phase in ("input", "volume", "interior", "boundary"):
            try:
                cpu[phase] = read(args.cpu_prefix, call, phase, args.dtype)
                cuda[phase] = read(args.cuda_prefix, call, phase, args.dtype)
            except FileNotFoundError as error:
                print(f"  Missing snapshot: {error.filename}; stopping")
                raise SystemExit(2)
            report(phase, cpu[phase], cuda[phase])
        report("interior_only", cpu["interior"] - cpu["volume"],
               cuda["interior"] - cuda["volume"])
        report("boundary_only", cpu["boundary"] - cpu["interior"],
               cuda["boundary"] - cuda["interior"])


if __name__ == "__main__":
    main()
