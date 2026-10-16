#!/usr/bin/env python3
"""Print existing first-RHS snapshots at one identified face point."""
import argparse
import numpy as np


def load(prefix, phase, dtype):
    return np.fromfile(f"{prefix}.rhs0.{phase}.bin", dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_rhs_prefix")
    parser.add_argument("cuda_rhs_prefix")
    parser.add_argument("cpu_flux_prefix")
    parser.add_argument("cuda_flux_prefix")
    parser.add_argument("--face", type=int, default=30982)
    parser.add_argument("--point", type=int, default=0)
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    args = parser.parse_args()
    if args.face < 0 or not 0 <= args.point < args.nfp:
        parser.error("Invalid face/point")
    for name, rhs, flux in (("CPU", args.cpu_rhs_prefix, args.cpu_flux_prefix),
                            ("CUDA", args.cuda_rhs_prefix, args.cuda_flux_prefix)):
        print(f"{name}: face={args.face} point={args.point}")
        for phase in ("face_input", "face_grad_x", "face_grad_y"):
            values = load(rhs, phase, args.dtype).reshape(-1, 2, 4, args.nfp)
            for side in (0, 1):
                row = values[args.face, side, :, args.point]
                print(f"  {phase} side={side}: " + " ".join(f"{v:.17e}" for v in row))
        normals = load(rhs, "face_normals", args.dtype).reshape(-1, args.nfp, 2)
        print("  normal: " + " ".join(f"{v:.17e}" for v in normals[args.face, args.point]))
        for phase in ("face_inviscid", "face_viscous", "face_combined"):
            values = load(flux, phase, args.dtype).reshape(-1, 4, args.nfp)
            row = values[args.face, :, args.point]
            print(f"  {phase}: " + " ".join(f"{v:.17e}" for v in row))
        values = load(flux, "face_properties", args.dtype).reshape(-1, 2, 5, args.nfp)
        for side in (0, 1):
            row = values[args.face, side, :, args.point]
            print(f"  properties side={side} [T p c mu kappa]: "
                  + " ".join(f"{v:.17e}" for v in row))


if __name__ == "__main__":
    main()
