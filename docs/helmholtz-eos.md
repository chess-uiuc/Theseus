# Using Helmholtz LTE and LTEEC

Helmholtz LTE obtains equilibrium thermodynamic properties from a table of
mass-specific Helmholtz free energy as a function of density and temperature.
Use `gas_model: "lte_helmholtz"` to select it. `LTEECFlux` is a two-state flux
for this EOS with Lax–Friedrichs face dissipation.

## Build requirements

Build with `THESEUS_WITH_PLATO=ON`. PLATO must supply
`plato_get_species_entropy` in both its header and library, and its database must
contain the selected mixture. CMake reports support in
`THESEUS_PLATO_HAS_ENTROPY` in the build cache. Selecting Helmholtz LTE without
this API fails during setup.

Debug and Release builds are supported. Subcell blending may be compiled on or
off. Neither the EOS nor LTEEC requires the optional kernel stress tests or array
debugging instrumentation. For CUDA execution, Theseus and MFEM must be built
with compatible CUDA support.

## Configure a simulation

Set these entries inside the configuration's `runTime` object. This example is
a thermodynamic/flux configuration fragment, not a complete simulation input:

```json
{
  "gas_model": "lte_helmholtz",
  "numerical_flux": "LTEECFlux",
  "gas_mixture": "air5",
  "plato_solver": "LTE_table_rhoT_(air5)",
  "N_rho": 31,
  "N_T": 41,
  "rho_min": 0.01,
  "rho_max": 3.0,
  "T_min": 100,
  "T_max": 2000,
  "rho_dist": "log",
  "T_dist": "linear"
}
```

Choose the mixture, solver, table bounds and resolution for the physical problem.
The example bounds are not suitable for every flow. Density and temperature
bounds must be positive and increasing; each grid needs at least two nodes.
States encountered during evolution and temperature recovery must be supported
by the table. Table resolution controls interpolation accuracy as well as cost.

| Selection | EOS / flux behavior |
| --- | --- |
| `lte_helmholtz` + `LTEECFlux` (alias `LTEEC`) | Helmholtz EOS and LTEEC flux with Lax face dissipation |
| `lte_helmholtz` + `Pirozzoli`, `HLL` or `LLF` | Helmholtz EOS with the selected flux |
| `lte` + `LTEECFlux` or `LTEEC` | Helmholtz EOS |
| `lte` + other supported LTE fluxes | Nine-plane LTE EOS |

## Transport and initialization

`SUTHERLAND=OFF` uses tabulated viscosity. `SUTHERLAND=ON` uses Sutherland
viscosity. Thermal conductivity is tabulated in both builds; it is not computed
from viscosity and a fixed Prandtl number.

Use [physical-state inputs](physical-state-conversion.md) to prescribe temperature,
pressure or density through the selected EOS. Conservative input must use the
same internal-energy convention as the selected thermodynamic model.

## Run an example and check a build

The [short ChannelFlow case](../TestCases/LTE/NavierStokes/ChannelFlow/README.md)
provides a complete unforced three-dimensional configuration and mesh. It is a
50-step regression rather than a sustained turbulent-channel calculation.

[CPU/CUDA validation](lteec-hpc-validation.md) gives kernel and flow-comparison
commands. [Kernel stress tests](kernel-stress-testing.md) describe optional large
batch checks. The [test inventory](verification.md) lists assertions and tolerances.

For table ownership, interpolation, temperature recovery and flux equations, see
the [developer reference](helmholtz-development.md).
