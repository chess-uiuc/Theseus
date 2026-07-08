# Theseus

[![CI Quick](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-quick.yml)
[![CI Nightly](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/ci-nightly.yml)
[![Doxygen Docs](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml/badge.svg)](https://github.com/MTCam/Theseus/actions/workflows/doxygen.yml)

Theseus is a compressible flow solver developed at the Center for Hypersonics and Entry Systems Studies (CHESS) at the University of Illinois [1]. It uses the MFEM finite element library [3], and targets modern and emerging HPC architectures.

Theseus formulation and numerics are based on the work of Hasanli [2], and its original implementation, [Prandtl](https://github.com/chess-uiuc/Prandtl) [4]. Theseus started out as a Prandtl modernization effort, but diverged substantially into a new codebase that is lighter and faster, while currently less feature-complete than Prandtl.

## Installation

Theseus requires an MFEM installation, and can use whatever compute device MFEM was configured for (for example CPU, CUDA, or HIP).

You can use an existing MFEM installation, or install dependencies with:

```bash
bash scripts/install-deps.sh
```

Useful `install-deps.sh` options:

- `JOBS` (default `4`): parallel build jobs
- `DEVICE` (default `cpu`): set to `cpu`, `cuda`, or `hip`
- `PREFIX` (default `./tpl/install`): install location used as `CMAKE_PREFIX_PATH`
- `CUDA_ARCH` (default `75`): CUDA arch used when `DEVICE=cuda`
- `HIP_ARCH` (optional): HIP arch used when `DEVICE=hip`

Examples:

```bash
# CPU build dependencies
JOBS=8 DEVICE=cpu bash scripts/install-deps.sh

# CUDA dependencies
JOBS=8 DEVICE=cuda CUDA_ARCH=80 bash scripts/install-deps.sh
```

## Building Theseus

```bash
export CC=mpicc
export CXX=mpicxx
mkdir -p build
cd build
cmake ../ -DCMAKE_PREFIX_PATH=/path/to/installation/of/mfem [-DENABLE_CUDA=YES]
make
make test
```

## Quick smoke tests

`ci-quick.yml` runs:

- direct smoke runs using `scripts/run_theseus.sh`
- regression checks using `scripts/compare_viz.py`
- unit tests via `ctest`

Two quick local smoke examples:

```bash
scripts/run_theseus.sh -b "$(pwd)/build" -c TestCases/Euler/2D/IsentropicVortex/config.json -t "0.002" -n 100
scripts/run_theseus.sh -b "$(pwd)/build" -c TestCases/NavierStokes/2D/LidDrivenCavity/config.json -t "0.0001" -n 100
```

## Documentation

We're working on putting together more detailed documentation.  [Start here](docs/README.md)

## References

1. Center for Hypersonics & Entry Systems Studies (CHESS), University of Illinois Urbana-Champaign. https://chess.grainger.illinois.edu/
2. Hasanli, F. (2025). *Implementation of a High Order Discontinuous Galerkin Spectral Element Method for the Euler and Navier--Stokes Equations on Unstructured Grids*. M.S. thesis, UIUC. https://hdl.handle.net/2142/129734
3. Anderson, R. et al. (2021). *MFEM: A modular finite element methods library*. Computers & Mathematics with Applications, 81, 42--74. https://doi.org/10.1016/j.camwa.2020.06.009
4. Prandtl repository: https://github.com/chess-uiuc/Prandtl

## AI Assistance Disclaimer

AI-enhanced tools (including GitHub Copilot and ChatGPT) are used extensively in this project to support code review, testing, and documentation efforts. Theseus numerics, mathematical formulation, and core intellectual content are original work by the human authors of the code and cited references. AI tools are used to improve project implementation and infrastructure, and all contributions are extensively reviewed by human authors.
