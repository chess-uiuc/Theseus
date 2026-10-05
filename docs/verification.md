# Verification and CI matrix

Theseus uses unit tests, integration tests, direct example smoke runs, and
golden-data regressions. Passing one class does not replace another: integration
tests exercise cross-component behavior, while golden tests protect established
numerical output.

## CI configurations

The quick and nightly workflows build two executables:

- a Cartesian build used by the existing CTest suite, example smokes, and
  golden-data regressions;
- an axisymmetric build configured with `-DAXISYMMETRIC=ON` for the complete
  axisymmetric integration suite below.

Both workflows invoke the [central validation runner](validation-runner.md). It
runs every registered CTest in each configuration, followed by the smoke and
regression cases below. Cartesian-only checks register in the Cartesian build;
axisymmetric-only checks register in the axisymmetric build. Results, CTest logs,
CMake caches and simulation output are retained as CI artifacts.

## CI integration tests

| CTest name | Configuration and execution | Required result |
| --- | --- | --- |
| `TimestepCFLIntegration` | Cartesian order-3 CNS cavity, one and two MPI ranks | The initial variable timestep matches the independently calculated mapped advective-plus-viscous stability rate; serial and MPI timesteps agree. Fixed-DT reporting occurs at the configured check interval, and a final-time-shortened step reports a proportionally smaller actual CFL. |
| `DerivedVisualizationIntegration` | Cartesian CPG/LTE, one and two MPI ranks | Emitted thermodynamic, energy and transport fields match individual gas queries; both ParaView mesh modes, VisIt, selected fields, disabled output and restart are checked. |
| `PhysicalInitialStateIntegration` | Cartesian CPG/LTE, one and two MPI ranks | Physical constant/profile initialization, thermodynamic pairs, rejection and conservative restart. |
| `PhysicalBoundaryIntegration` | Axisymmetric CPG/LTE CNS, one and two MPI ranks | Uniform-state preservation, heated radial inflow response, selected-EOS energy, primitive interpolation, legacy CPG parity and coordinated rejection. |
| `CheckpointRestartIntegration` | Axisymmetric Euler uniform flow, two MPI ranks, two cycles | A restarted cycle produces byte-identical per-rank checkpoint state and ParaView output to an uninterrupted run. Metadata must contain the required format, state, geometry, MPI, and discretization fields and identify axisymmetric geometry. |
| `AxisymmetricUniformFlowIntegration` | Exact Euler and CNS uniform axial flow, one and two MPI ranks | Density and pressure remain at the exact values, conserved-integral changes remain negligible, and serial/MPI results agree. |
| `AxisymmetricEntropyWaveConvergence` | Exact Euler entropy wave over three mesh levels | Cylindrical L2 errors decrease and both observed convergence rates are at least `1.7`. |
| `AxisymmetricInviscidSphereIntegration` | Mach 2 Euler sphere, one and two MPI ranks | Both runs complete with finite positive density and pressure, develop nonuniform density and pressure ranges, and agree in their final reported ranges within `1e-5` absolute tolerance. |
| `AxisymmetricViscousSphereIntegration` | Mach 0.3, `Re_D=100`, order-3 CNS sphere, one and two MPI ranks | The same positivity, body-flow-response, and serial/MPI agreement requirements as the inviscid sphere case. |

The sphere tests accept the MFEM device selected by the CMake cache variable
`AXISYMMETRIC_TEST_DEVICE`. CI currently tests `cpu`; CUDA and HIP are untested.

The viscous sphere test exercises the axisymmetric CNS body-flow path. It is not
yet a quantitative validation of wake separation distance; that requires
suitable subsonic characteristic boundaries, steady-state and mesh-convergence
studies, and automated separation-point extraction.

Run the axisymmetric CI integration suite locally with:

```sh
ctest --test-dir build-axisymmetric \
  -R '^(Axisymmetric.*(Integration|Convergence)|CheckpointRestartIntegration)$' \
  --output-on-failure
```

## Additional permanent axisymmetric tests

The complete axisymmetric CTest suite also includes:

- `AxisymmetryConfigTests`: build/configuration, mesh, coordinate, and state
  contract checks;
- `AxisymmetricGeometryTests`: cylindrical measure and inviscid/viscous source
  checks, including analytic axis limits;
- `AxisymmetricUniformFlowIntegration`: exact Euler and CNS uniform axial flow
  in serial and two-rank MPI, with a configurable MFEM device;
- `AxisymmetricEntropyWaveConvergence`: three mesh levels, decreasing
  cylindrical L2 error, and a minimum observed convergence rate of `1.7`;
- checkpoint/restart and sphere tests listed above.

Run all registered tests with:

```sh
ctest --test-dir build --output-on-failure
ctest --test-dir build-axisymmetric --output-on-failure
```

## Direct CI smoke runs

These runs require successful completion and the configured NaN checks. They do
not compare against reference fields.

| Case | Fixed timestep | Maximum steps |
| --- | ---: | ---: |
| Euler Isentropic Vortex | `0.002` | 100 |
| CNS Lid-Driven Cavity | `0.0001` | 100 |
| LTE Euler Vortex | `1e-5` | 100 |

## Regression comparisons

The cyclic Isentropic Vortex regression compares the first and last datasets in
its ParaView collection for density, with `atol=5e-5` and `rtol=2e-6`.

The golden-data regressions compare all available point fields and mesh
topology at the specified output cycle:

| Case | Timestep | Output cycle | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: | ---: | ---: |
| Isentropic Vortex | `0.001` | 500 | `1e-13` | `1e-13` |
| LTE Vortex | `1e-5` | 200 | `1e-10` | `1e-10` |
| Lid-Driven Cavity | `0.0001` | 4000 | `1e-12` | `1e-12` |
| Taylor-Green Vortex 2D | `0.0001` | 100 | `1e-13` | `1e-13` |
| Forward-Facing Step | `0.0001` | 100 | `3e-13` | `1e-13` |

`scripts/validate.py` is the executable source of truth for the full suite. Update
this matrix whenever a case, step count, tolerance, build configuration, or
required assertion changes.

## PX chamber startup

The [PX chamber case record](px-chamber.md) describes the stationary CPG/LTE radial
profile, boundary conditions, meshes, and current modeling limitations.

| Test | What is run and checked |
| --- | --- |
| `RadialProfileTests` | Primitive interpolation, axis extension, signed velocity and CPG state conversion; selection of the supplied profile blocks; rejection of a missing flag, duplicate radii, negative temperature, and evaluation above radial coverage. |
| `prescribed_state_boundary_correction_uses_exterior_entropy` in `AxisymmetricGeometryTests` | The boundary correction is zero for matching entropy states and equals the expected difference for a known perturbation. |
| `LTEPXChamberIntegration` | Coarse LTE chamber: quiescent 300 K wall preservation, then heated startup on one/two ranks for 20 steps of `1e-9 s`. Checks nine derived fields, Mach consistency and serial/MPI field statistics. |
| `PXChamberIntegration` | Generates the 864-quadrilateral coarse chamber mesh and runs 20 steps of size `1e-8 s` to `2e-7 s`: quiescent 300 K gas on one rank, then the supplied heated profile on one and two MPI ranks. |

The integration test reads the final written VTK fields. It checks positive,
finite density, pressure and reconstructed CPG temperature; quiescent-state
preservation; and a heated-case maximum temperature above 301 K. It compares
serial/MPI density, pressure and temperature extrema and the maximum absolute
interior nodal radial velocity on the axis. Uniform-state tolerances are
`rtol=1e-10, atol=1e-9`; serial/MPI statistics use `rtol=1e-9, atol=1e-8`.
It does not compare the complete fields or require exactly zero nodal radial
velocity during heated startup.

The profile and boundary-correction tests run in the regular CTest suite.
`PXChamberIntegration` and `LTEPXChamberIntegration` run in the axisymmetric CTest suite in Quick and Nightly CI. These are startup/regression checks, not demonstrations
of steady state, conservation-budget closure, mesh convergence, or quantitative
agreement with the reference plasma calculation.

The supplied 75,375-cell mesh is included in the case and is the runner's
default. It completed a separate two-rank, 100-step startup to `1e-7 s`, but
that full-mesh run is **not** part of automated CI. A separate coarse run reached
`2e-4 s` in 20,000 steps; it is also outside the short CI test.


## Physical state and boundary conversion

`PhysicalStateConversionTests` includes analytic LTE boundary packing checks at
axis, interpolated and endpoint radii, conservative descriptor offsets, legacy
CPG parity and invalid inputs. It also checks matching and perturbed isothermal-wall
entropy corrections for both CPG and LTE normalization. `PhysicalStatePLATOTests` checks real-air conversion.
The physical integration tests above honor `PHYSICAL_STATE_TEST_DEVICE` (default
`cpu`). See [physical state configuration](physical-state-conversion.md) for input
syntax and device build options. All are selected automatically by the full runner.


## Shared gas-property API

`GasPropertyTests_standard` and `GasPropertyTests_sutherland` check the host API
against CPG formulas, analytic LTE tables and existing individual queries. The tests
exercise table endpoints, interior interpolation, requested-only results, unavailable
properties and the number of LTE temperature recoveries. Both variants run in the
standard and axisymmetric CTest suites, without requiring PLATO. These tests cover
the shared API used by selected visualization fields.


## Visualization fields

`VisualizationFieldsTests` checks scalar/vector values and stable field ownership in
1D/2D/3D, identical ParaView/VisIt registrations, selected-only fields, and one combined
property request per point. Algebraic selections make no EOS property requests.
`DerivedVisualizationIntegration` uses a small real-air table and emitted files to
check CPG/LTE output against individual EOS/transport queries. It checks dimension-sized
velocity arrays, both ParaView mesh modes, two-rank MPI, VisIt files, selection and
bitwise-equal final derived fields after conservative restart. Numerical comparisons
use `rtol=1e-9, atol=1e-10`. The test uses `PHYSICAL_STATE_TEST_DEVICE`.
