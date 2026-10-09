# Test inventory

The full suite combines unit/helper tests, integrated simulations, smoke runs and
numerical regression comparisons. [Running tests](validation-runner.md) explains
how to execute it locally or on an HPC system. [Adding tests](adding-tests.md)
explains registration and checker structure.

Quick and Nightly CI use Cartesian and axisymmetric CPU builds. CTest runs the
unit/helper tests. The central runner executes the integrated checks below and
records their results individually.

## Integrated tests

| Summary name | Configuration and execution | Required result |
| --- | --- | --- |
| **Timestep and CFL** | Cartesian order-3 CNS cavity, one and two MPI ranks | The initial variable timestep matches the independently calculated mapped advective-plus-viscous stability rate; serial and MPI timesteps agree. Fixed-DT reporting occurs at the configured check interval, and a final-time-shortened step reports a proportionally smaller actual CFL. |
| **Derived Visualization** | Cartesian CPG/LTE, one and two MPI ranks | Emitted thermodynamic, energy and transport fields match individual gas queries; both ParaView mesh modes, VisIt, selected fields, disabled output and restart are checked. |
| **Physical Initial States** | Cartesian CPG/LTE, one and two MPI ranks | Physical constant/profile initialization, thermodynamic pairs, rejection and conservative restart. |
| **Physical Boundary States** | Axisymmetric CPG/LTE CNS, one and two MPI ranks | Uniform-state preservation, heated radial inflow response, selected-EOS energy, primitive interpolation, legacy CPG parity and coordinated rejection. |
| **Cartesian Checkpoint Restart** | Cartesian vortex, two MPI ranks, two cycles | Restarted conservative state and final visualization match the uninterrupted run byte for byte; checkpoint metadata identifies Cartesian geometry. |
| **Axisymmetric Checkpoint Restart** | Axisymmetric Euler uniform flow, two MPI ranks, two cycles | A restarted cycle produces byte-identical per-rank checkpoint state and ParaView output to an uninterrupted run. Metadata must contain the required format, state, geometry, MPI, and discretization fields and identify axisymmetric geometry. |
| **Axisymmetric Uniform Flow** | Exact Euler and CNS uniform axial flow, one and two MPI ranks | Density and pressure remain at the exact values, conserved-integral changes remain negligible, and serial/MPI results agree. |
| **Axisymmetric Entropy-Wave Convergence** | Exact Euler entropy wave over three mesh levels | Cylindrical L2 errors decrease and both observed convergence rates are at least `1.7`. |
| **Axisymmetric Inviscid Sphere** | Mach 2 Euler sphere, one and two MPI ranks | Both runs complete with finite positive density and pressure, develop nonuniform density and pressure ranges, and agree in their final reported ranges within `1e-5` absolute tolerance. |
| **Axisymmetric Viscous Sphere** | Mach 0.3, `Re_D=100`, order-3 CNS sphere, one and two MPI ranks | The same positivity, body-flow-response, and serial/MPI agreement requirements as the inviscid sphere case. |

The integrated runner passes the selected MFEM device to all integrated simulations.
CI uses CPU; prepared GPU builds can be exercised with `--suite integrated --device cuda`.

The viscous sphere test exercises the axisymmetric CNS body-flow path. It is not
yet a quantitative validation of wake separation distance; that requires
suitable subsonic characteristic boundaries, steady-state and mesh-convergence
studies, and automated separation-point extraction.

Run the complete integrated suite against prepared Cartesian and axisymmetric builds
using the command in [Validation runner](validation-runner.md#integrated-tests-with-prepared-builds-hpc-or-workstation).

## Unit and helper tests

CTest includes configuration, state layout, gas physics, flux, mesh, geometry,
checkpoint, table lookup, field evaluation and test-infrastructure checks.
`AxisymmetryConfigTests` checks the geometry/state contract;
`AxisymmetricGeometryTests` checks cylindrical measures and source terms, including
axis limits. Specialized physics checks are described below.

Run the registered unit/helper tests with:

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

The [PX chamber guide](px-chamber.md) describes the stationary CPG/LTE radial
profile, boundary conditions, meshes, and current modeling limitations.

| Test | What is run and checked |
| --- | --- |
| `RadialProfileTests` | Primitive interpolation, axis extension, signed velocity and CPG state conversion; selection of the supplied profile blocks; rejection of a missing flag, duplicate radii, negative temperature, and evaluation above radial coverage. |
| `prescribed_state_boundary_correction_uses_exterior_entropy` in `AxisymmetricGeometryTests` | The boundary correction is zero for matching entropy states and equals the expected difference for a known perturbation. |
| **LTE PX Chamber** | Coarse LTE chamber: quiescent 300 K wall preservation, then heated startup on one/two ranks for 20 steps of `1e-9 s`. Checks nine derived fields, Mach consistency and serial/MPI field statistics. |
| **CPG PX Chamber** | Generates the 864-quadrilateral coarse chamber mesh and runs 20 steps of size `1e-8 s` to `2e-7 s`: quiescent 300 K gas on one rank, then the supplied heated profile on one and two MPI ranks. |

The CPG chamber check reads the final written VTK fields. It checks positive,
finite density, pressure and reconstructed CPG temperature; quiescent-state
preservation; and a heated-case maximum temperature above 301 K. It compares
serial/MPI density, pressure and temperature extrema and the maximum absolute
interior nodal radial velocity on the axis. Uniform-state tolerances are
`rtol=1e-10, atol=1e-9`; serial/MPI statistics use `rtol=1e-9, atol=1e-8`.
It does not compare the complete fields or require exactly zero nodal radial
velocity during heated startup.

The profile and boundary-correction tests run in the regular CTest suite.
**CPG PX Chamber** and **LTE PX Chamber** run through the central runner with the axisymmetric executable in Quick and Nightly CI. These are startup/regression checks, not demonstrations
of steady state, conservation-budget closure, mesh convergence, or quantitative
agreement with the reference plasma calculation.

The chamber's default mesh has 75,375 cells; the integrated checks use the
864-cell coarse mesh. Their acceptance criteria describe short startup behavior.
Steady-state predictions require separate convergence and conservation studies.

## Physical state and boundary conversion

`PhysicalStateConversionTests` includes analytic LTE boundary packing checks at
axis, interpolated and endpoint radii, conservative descriptor offsets, legacy
CPG parity and invalid inputs. It also checks matching and perturbed isothermal-wall
entropy corrections for both CPG and LTE normalization. `PhysicalStatePLATOTests` checks real-air conversion.
The physical integration tests above honor the runner’s `--device` selection (default
`cpu`). See [physical state configuration](physical-state-conversion.md) for input
syntax. All are selected automatically by the full runner.


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
**Derived Visualization** uses a small real-air table and emitted files to
check CPG/LTE output against individual EOS/transport queries. It checks dimension-sized
velocity arrays, both ParaView mesh modes, two-rank MPI, VisIt files, selection and
bitwise-equal final derived fields after conservative restart. Numerical comparisons
use `rtol=1e-9, atol=1e-10`. The test uses the integrated runner’s `--device` selection.
