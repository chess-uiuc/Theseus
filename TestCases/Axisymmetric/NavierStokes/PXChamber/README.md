# PX chamber: first CPG startup case

See the [living case record](../../../../docs/px-chamber.md) for setup details,
added tests, recorded validation, limitations, and planned refinements.

This is a swirl-free, axisymmetric PX CNS startup case, not a converged chamber solution
or an LTE validation. Coordinates are x = 0.4–1.05 m, r = y = 0–1 m. The supplied mesh
has 75,375 quadrilaterals. Its Pointwise comment sections were removed and PhysicalNames
moved ahead of Nodes for MFEM compatibility; coordinates, connectivity, and tags
are unchanged. The original user mesh is untouched.

## Model and boundaries

- CPG air: gamma=1.4, R=287.05 J/(kg K), constant mu=1.8e-5 Pa s, Pr=0.72.
- Initially quiescent gas at 10 kPa and 300 K.
- `Axis`: existing axis BC on r=0.
- `Inflow`: stationary radial T, ux, ur exterior state at 10 kPa, up to r=0.0868 m.
- `Left`: no-slip isothermal wall, stationary at 300 K.
- `Top`: quiescent ambient exterior state, 10 kPa and 300 K.
- `Right`: extrapolation-based (`supersonic-outflow`). The name does not
  imply that this subsonic flow becomes supersonic.

The inlet and top states enter the numerical flux weakly. They are not
characteristic subsonic inlet/opening BCs, and the right does not impose a
back pressure. The top is not HEGEL's total-condition inlet.
The new exterior-state path also supplies an entropy-variable boundary correction
in the auxiliary gradient equation and an interior-gradient viscous boundary flux. This first implementation is
explicitly CPG-only and rejects LTE use.

The data file starts with `points blocks`, then `r flag T ux ur swirl` rows.
Flag 0 is selected (flag 1 is identical in this supplied file). Swirl is
intentionally ignored (for now). Interpolation is linear in the primitive quantities,
followed by CPG conversion at each boundary quadrature point. Within the first
sample radius, T and ux remain constant and ur is linear through zero at the
axis. This supplies even T/ux and odd ur extensions. Signed velocities,
including weak local reverse axial flow, are retained. Radii above table
coverage are errors; the profile is applied only to Inflow. The input T is
assumed static temperature.

## Build and run

From the repository root on this Mac:

```sh
export PATH=/usr/bin:/opt/homebrew/bin:$PATH
export CC=/opt/homebrew/bin/mpicc CXX=/opt/homebrew/bin/mpicxx
cmake -S . -B build-px -DAXISYMMETRIC=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/Users/mtcampbe/CHESS/Development/Experimental-Codes/Repos/Theseus/tpl/install
cmake --build build-px -j 4
```

Run the supplied full mesh for a short startup (100 steps):

```sh
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-full --ranks 2
```

Run the 864-cell coarse mesh for a longer startup (20,000 steps):

```sh
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-coarse --ranks 2 \
  --coarse --dt 1e-8 --final-time 2e-4
```

The helper creates a resolved config, run.log, and ParaView/ParaView.pvd in the
output directory. It handles OpenMPI localhost placement. This is a local CPU
runner; accelerator correctness has not been tested. Config paths in the raw
config.json are repository-root relative; use the helper from other directories.
Output directories should be distinct for separate experiments.

## Verification and limitations

RadialProfileTests checks interpolation, axis extension, CPG closure, selected
flag handling and malformed input. PXChamberIntegration checks quiescent-state
preservation and heated startup on one and two ranks, using the written VTK
fields to check positive density/pressure/temperature and rank agreement.
AxisymmetricGeometryTests checks the new entropy-state boundary correction and existing parity.

The default full-mesh run is only 0.1 microsecond. The longer coarse example
is 0.2 ms; neither is a steady-state calculation. Simulation needs further
checks with longer integration.
Existing axis reflection likewise does not make every interior
nodal radial velocity exactly zero. CPG and constant transport at these
plasma temperatures aren't correct - but this is just a test. We need to
add/test pressure/characteristic boundaries, assess axis and wall errors under
refinement, verify numerical boundary fluxes, and eventually switch to an appropriate
LTE mixture/transport model Prathamesh provided thermo tables for this case, but we
don't currently have Theseus path for using pre-existing tables.
