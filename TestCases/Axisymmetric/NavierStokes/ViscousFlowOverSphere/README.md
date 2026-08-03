# Axisymmetric Viscous Flow Over a Sphere

This case is a first Theseus counterpart to the viscous sphere case in Tulio's
axisymmetry notes. It uses a nondimensional Mach 0.3 freestream and an adiabatic
no-slip unit sphere at diameter-based Reynolds number 100:

```text
rho_inf = 1, a_inf = 1, U_inf = 0.3, D = 2, mu = 0.006
Re_D = rho_inf U_inf D / mu = 100
```

Polynomial order 3 matches the reported calculation. The body-fitted mesh is
shared with the Euler sphere case.

Theseus does not yet provide subsonic characteristic inflow and outflow
conditions. The outer boundary therefore uses fixed freestream state at the
inlet and farfield and extrapolation at the outlet. This case exercises the CNS
axisymmetric body-flow path, but wake separation should not be treated as a
validated reproduction until the boundary treatment, mesh convergence, and
steady-state convergence have been qualified.

Run from the repository root with an axisymmetric build:

```sh
scripts/run_theseus.sh -b /path/to/axisymmetric-build \
  -c TestCases/Axisymmetric/NavierStokes/ViscousFlowOverSphere/config.json -p 2
```
