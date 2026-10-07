# Theseus Documentation

Documentation is a work in progress.

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
- General input-file reference (coming soon)
- [Checkpoints and restarts](checkpoints.md): save solution state, validate
  restart compatibility, and resume a run through the standard helper
- [Axisymmetric formulation](axisymmetry.md): configure swirl-free cylindrical
  Euler/CNS runs and understand their verification status
- [Verification and CI matrix](verification.md): integration assertions, smoke
  cases, golden-data tolerances, and local reproduction commands

## Theory

- [Governing equations](theory.md)
- [DGSEM discretization](discretization.md)
- [Numerical fluxes](numflux.md)
- [Boundary conditions](boundaryconditions.md)

## Developer guide

- [Adding tests](adding-tests.md): choose a test type, register it, and write an integrated checker
- Runtime configuration
- [Shared gas-property evaluation](gas-properties.md): request several properties
  while reusing LTE temperature recovery
- Physics models
- Adding new components

## Verification

[Running tests](validation-runner.md): framework diagram, local and HPC commands,
results and failure diagnosis.

See the [verification and CI matrix](verification.md) for the maintained test
inventory and exact regression tolerances.
