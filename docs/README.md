# Theseus Documentation

## Using Theseus

The top-level [README](../README.md) covers dependency installation, building,
and quick smoke runs. Simulation behavior is controlled by each test case's
`config.json` input file.

### Runtime and input configuration

- [Visualization output](visualization.md): enable visualization, select output
  fields, and understand the available field names.
- [Physical states and profiles](physical-state-conversion.md): selected-EOS initial
  conditions and exterior/profile boundary inputs
- [Migrating physical inputs](physical-state-migration.md): CPG/LTE compatibility,
  table coverage and a runnable LTE chamber example
- [Helmholtz LTE and LTEEC](helmholtz-eos.md): build requirements, configuration,
  table bounds and transport choices
- General input-file reference (coming soon)
- [Checkpoints and restarts](checkpoints.md): save solution state, validate
  restart compatibility, and resume a run through the standard helper
- [Axisymmetric formulation](axisymmetry.md): configure swirl-free cylindrical
  Euler/CNS runs and understand their verification status
- [Performance comparison](performance-comparison.md): compare timestep and
  construct-level timings across two or more compatible builds

## Theory

- [Governing equations](theory.md)
- [DGSEM discretization](discretization.md)
- [Numerical fluxes](numflux.md)
- [Boundary conditions](boundaryconditions.md)

## Developer guide

- [Helmholtz LTE interfaces](helmholtz-development.md): table ownership, thermodynamic
  queries, temperature recovery, discrete means and flux formulas
- [Bicubic interpolation](helmholtz-interpolation.md): cell views, derivatives and
  stable divided differences

- [Adding tests](adding-tests.md): choose a test type, register it, and write an integrated checker
- [Shared gas-property evaluation](gas-properties.md): request several properties
  while reusing LTE temperature recovery

## Verification

[Running tests](test-runner.md): framework diagram, local and HPC commands,
single-build testing, results and failure diagnosis.

- [Helmholtz LTEEC CPU/CUDA checks](lteec-hpc-validation.md): one build, kernel
  oracles and automatic same-binary evolution comparisons
- [Optional kernel stress tests](kernel-stress-testing.md): independent references,
  large batches, guards and explicit host/backend agreement
- [Verification and CI matrix](verification.md): integration assertions, smoke
  cases, golden-data tolerances, and local reproduction commands


See the [verification and CI matrix](verification.md) for the maintained test
inventory and exact regression tolerances.
