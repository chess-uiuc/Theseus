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

The axisymmetric integration tests are separate CI steps so a failure identifies
the affected capability directly. Their CTest XML, CMake cache, and build logs
are included in failure artifacts.

## CI integration tests

| CTest name | Configuration and execution | Required result |
| --- | --- | --- |
| `TimestepCFLIntegration` | Cartesian order-3 CNS cavity, one and two MPI ranks | The initial variable timestep matches the independently calculated mapped advective-plus-viscous stability rate; serial and MPI timesteps agree. Fixed-DT reporting occurs at the configured check interval, and a final-time-shortened step reports a proportionally smaller actual CFL. |
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

The workflow files are the executable source of truth for CI commands. Update
this matrix whenever a case, step count, tolerance, build configuration, or
required assertion changes.

## PX chamber startup

The [PX chamber case record](px-chamber.md) describes the stationary CPG radial
profile, boundary conditions, meshes, and current modeling limitations.

| Test | What is run and checked |
| --- | --- |
| `RadialProfileTests` | Primitive interpolation, axis extension, signed velocity and CPG state conversion; selection of the supplied profile blocks; rejection of a missing flag, duplicate radii, negative temperature, and evaluation above radial coverage. |
| `prescribed_state_boundary_correction_uses_exterior_entropy` in `AxisymmetricGeometryTests` | The boundary correction is zero for matching entropy states and equals the expected difference for a known perturbation. |
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
`PXChamberIntegration` is explicitly selected by the Quick and Nightly
axisymmetric CI steps. These are startup/regression checks, not demonstrations
of steady state, conservation-budget closure, mesh convergence, or quantitative
agreement with the reference plasma calculation.

The supplied 75,375-cell mesh is included in the case and is the runner's
default. It completed a separate two-rank, 100-step startup to `1e-7 s`, but
that full-mesh run is **not** part of automated CI. A separate coarse run reached
`2e-4 s` in 20,000 steps; it is also outside the short CI test.
