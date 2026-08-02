# Theseus Documentation

Documentation is a work in progress.

## Using Theseus

The top-level [README](../README.md) covers dependency installation, building,
and quick smoke runs. Simulation behavior is controlled by each test case's
`config.json` input file.

### Runtime and input configuration

- [Visualization output](visualization.md): enable visualization, select output
  fields, and understand the available field names.
- General input-file reference (coming soon)
- [Checkpoints and restarts](checkpoints.md): save solution state, validate
  restart compatibility, and resume a run through the standard helper

## Theory

- [Governing equations](theory.md)
- [DGSEM discretization](discretization.md)
- [Numerical fluxes](numflux.md)
- [Boundary conditions](boundaryconditions.md)

## Developer Guide (Coming soon)

- Architecture
- Runtime configuration
- Physics models
- Adding new components

## Verification (Coming soon)

- Isentropic vortex
- Taylor-Green vortex
- Forward-facing step
