# PlasmatronX chamber-only axisymmetric case

This swirl-free compressible Navier–Stokes example supports legacy CPG startup and
an LTE physical-input configuration. It is a short startup exercise, not a steady
or quantitatively validated plasma-chamber prediction.

## Geometry and input files

The [case directory](../TestCases/Axisymmetric/NavierStokes/PXChamber/) contains:

| File | Purpose |
| --- | --- |
| `config.json` | Legacy CPG parameters and boundaries |
| `config-lte.json` | LTE physical states, table settings and derived output |
| `px_chamber_axi.msh` | Supplied Pointwise/Gmsh mesh normalized for MFEM |
| `Tuvw_jet_inlet_profile.dat` | Supplied stationary radial temperature/velocity data |
| `run_case.py` | Resolves paths, selects gas/device, launches MPI and optionally generates a coarse mesh |

Coordinates span x = 0.4–1.05 m and r = y = 0–1 m. The inlet radius is 0.0868 m.
The supplied mesh has 75,936 nodes and 75,375 quadrilaterals. The coarse option has
864 quadrilaterals and retains the inlet/wall junction. Mesh comment sections were
removed and PhysicalNames moved before Nodes for MFEM input; geometry/tags are retained.
The mesh and profile came from Prathamesh's reference case. Supplied external thermo
tables are not read by this example; LTE uses PLATO-generated air5 tables.

The executable requires `AXISYMMETRIC=ON`. States are ordinary physical
`[rho, rho*ux, rho*ur, rho*E]`, without radius weighting. There is no swirl equation
or volumetric plasma-heating term. Initial gas is quiescent at 10 kPa and 300 K.
Both configurations use order 1, LLF, BR1 viscous terms, RK3SSP, fixed timestep and
subcell blending with alpha_max = 0.5.

## Boundaries and profile

| Group | Location | Treatment |
| --- | --- | --- |
| Axis | r = 0 | Axis reflection/regularity |
| Inflow | x = 0.4, r ≤ 0.0868 | Stationary radial T, ux, ur; 10 kPa exterior pressure |
| Left | x = 0.4, r > 0.0868 | Stationary no-slip wall at 300 K |
| Top | r = 1 | Quiescent exterior state at 10 kPa, 300 K |
| Right | x = 1.05 | `supersonic-outflow` extrapolation |

Legacy inputs use `cpg-radial-profile`, `cpg-exterior-state` and `cpg_state`.
LTE uses `radial-profile`, `exterior-state` and a physical initial `state` instead.
The same selected EOS constructs initial and exterior conservative energies.
Wall entropy scaling also comes from that EOS.

The profile header is `points blocks`; data columns are `r flag T ux ur swirl`.
Flag 0 is selected. The supplied flag-1 block is identical. Swirl is ignored.
Temperature and signed velocities are linearly interpolated before conversion.
Below the first radius, T and ux remain constant while ur scales to zero at the axis.
Evaluation above the final radius is rejected. The profile spans approximately
0.00025–0.991239 m; its first sample is approximately 10,684 K and 316.17 m/s axial
velocity. Temperature is interpreted as static temperature.

Exterior targets enter numerical fluxes weakly, including the entropy-gradient
boundary correction. They do not instantly set the interior trace to the target.
The inlet/top are not characteristic subsonic boundaries. Right extrapolation does
not impose back pressure, and its type name does not make the flow supersonic.
The reference HEGEL top total-condition inlet is not reproduced by the ambient Top.

## Build and CPG startup

From the repository root, with the platform's MPI compiler environment:

```sh
cmake -S . -B build-px -DAXISYMMETRIC=ON -DTHESEUS_WITH_PLATO=YES \
  -DSUBCELL_FV_BLENDING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TPL_PREFIX"
cmake --build build-px -j 4
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-cpg --ranks 2 --coarse
```

CPG uses gamma = 1.4, R = 287.05 J/(kg K), mu = 1.8e-5 Pa·s and Pr = 0.72.
The default helper settings are dt = 1e-9 s and final time = 1e-7 s (100 steps).
Omit `--coarse` to use the supplied mesh. Distinct output directories keep experiments
separate. The helper writes resolved `config.json`, `run.log` and ParaView output.
Raw configuration paths are repository-root relative; the helper resolves them.

## LTE startup

Use a PLATO-enabled build and an installed air5 database:

```sh
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-lte --ranks 2 --coarse \
  --gas-model lte --database "$PLATO_DATABASE" --device cpu \
  --dt 1e-9 --final-time 2e-8
```

The LTE configuration uses a 101×201 logarithmic density/temperature grid:
1e-4–1.1 kg/m³ and 250–15,000 K. This covers the cold state and hot, low-density
inlet targets; it is not a table-resolution convergence claim. Conductivity and
viscosity come from the LTE model's transport tables in the default build.
Temperature, sound speed, Mach, internal energy and transport fields are written
alongside density, velocity and pressure. Do not reconstruct LTE temperature with
a fixed gas constant.

`--device cuda` selects CUDA when the executable and dependencies support it.
The helper defaults to CPU and accepts `--mpiexec` for a platform launcher. Its
OpenMPI defaults use localhost slots; a launcher wrapper can supply other site
placement behavior. See [physical-state migration](physical-state-migration.md)
for configuration changes and [visualization](visualization.md) for output fields.

## Verification and limits

`PXChamberIntegration` retains the CPG quiescent and heated startup checks.
`LTEPXChamberIntegration` uses the coarse mesh for 20 steps to 2e-8 s, checking
quiescent wall preservation, heated startup on one/two ranks, finite positive
thermodynamic/transport fields, Mach consistency and rank agreement. It exercises
the helper's LTE/device options and uses the written EOS Temperature field.
The focused isothermal-wall kernel test checks matching and perturbed entropy states
for both EOS normalizations. See the [verification matrix](verification.md).

These short tests do not establish steady state, mesh/order/table convergence,
physical wall-heat-flux accuracy, exact inlet trace recovery or closed conservation
budgets. Axis conditions are weak; interior nodal radial velocity is not forced to
zero pointwise. Longer chamber predictions require pressure/characteristic boundary
modeling, refinement studies, flux-budget diagnostics and a justified mixture model.
CPG at the supplied hot-profile temperatures is a compatibility exercise. The LTE
example uses air5, not the reference air-11 model, and does not include swirl.
