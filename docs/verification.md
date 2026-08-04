# Verification and CI matrix

Theseus uses unit tests, integration tests, direct example smoke runs, and
golden-data regressions. Passing one class does not replace another: integration
tests exercise cross-component behavior, while golden tests protect established
numerical output.

## CI configurations

The quick and nightly workflows build two executables:

- a Cartesian build used by the existing CTest suite, example smokes, and
  golden-data regressions;
- an axisymmetric build configured with `-DAXISYMMETRIC=ON` for the restart and
  sphere integration tests below.

The axisymmetric integration tests are separate CI steps so a failure identifies
the affected capability directly. Their CTest XML, CMake cache, and build logs
are included in failure artifacts.

## CI integration tests

| CTest name | Configuration and execution | Required result |
| --- | --- | --- |
| `CheckpointRestartIntegration` | Axisymmetric Euler uniform flow, two MPI ranks, two cycles | A restarted cycle produces byte-identical per-rank checkpoint state and ParaView output to an uninterrupted run. Metadata must contain the required format, state, geometry, MPI, and discretization fields and identify axisymmetric geometry. |
| `AxisymmetricInviscidSphereIntegration` | Mach 2 Euler sphere, one and two MPI ranks | Both runs complete with finite positive density and pressure, develop nonuniform density and pressure ranges, and agree in their final reported ranges within `1e-5` absolute tolerance. |
| `AxisymmetricViscousSphereIntegration` | Mach 0.3, `Re_D=100`, order-3 CNS sphere, one and two MPI ranks | The same positivity, body-flow-response, and serial/MPI agreement requirements as the inviscid sphere case. |

The sphere tests accept the MFEM device selected by the CMake cache variable
`AXISYMMETRIC_TEST_DEVICE`. CI currently tests `cpu`; CUDA and HIP are untested.

The viscous sphere test exercises the axisymmetric CNS body-flow path. It is not
yet a quantitative validation of wake separation distance; that requires
suitable subsonic characteristic boundaries, steady-state and mesh-convergence
studies, and automated separation-point extraction.

Run the three CI integration tests locally with an axisymmetric build:

```sh
ctest --test-dir build-axisymmetric \
  -R '^CheckpointRestartIntegration$' --output-on-failure
ctest --test-dir build-axisymmetric \
  -R '^AxisymmetricInviscidSphereIntegration$' --output-on-failure
ctest --test-dir build-axisymmetric \
  -R '^AxisymmetricViscousSphereIntegration$' --output-on-failure
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
