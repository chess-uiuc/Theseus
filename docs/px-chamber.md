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
| `config-lte-local-characteristic.json` | Local characteristic LTE inputs |
| `config-local-characteristic.json` | Local characteristic CPG inputs |
| `px_chamber_coarse.msh` | Explicit short-test mesh |
| `run_case.py` | Retained legacy automated-test utility; not needed for user runs |

Coordinates span x = 0.4–1.05 m and r = y = 0–1 m. The inlet radius is 0.0868 m.
The supplied mesh has 75,936 nodes and 75,375 quadrilaterals. The coarse option has
864 quadrilaterals and retains the inlet/wall junction. Mesh comment sections were
removed and PhysicalNames moved before Nodes for MFEM input; geometry/tags are retained.
The mesh and profile came from Prathamesh's reference case. Supplied external thermo
tables are not read by this example; LTE uses PLATO-generated air11 tables.

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

## Build and run

Build an axisymmetric, PLATO-enabled executable for LTE:

```bash
cmake -S . -B build-px -DAXISYMMETRIC=ON -DTHESEUS_WITH_PLATO=YES \
  -DSUBCELL_FV_BLENDING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$TPL_PREFIX"
cmake --build build-px -j 4
```

Use `config-lte-local-characteristic.json` for the local characteristic LTE
boundaries, or `config-local-characteristic.json` for CPG. Original `config.json`
and `config-lte.json` keep the legacy boundaries described above.

Copy the selected JSON into your input directory. Set absolute `runTime.mesh_file`
and `runTime.conditions.boundary_conditions.Inflow.file` paths. For LTE add
`runTime.database_path` with the installed PLATO database directory. Keep
`output_file_path` as `"."` and set timestep, final time and output cadence in JSON.

```bash
scripts/run_theseus.sh -e /absolute/path/to/build-px/theseus \
  -c /absolute/path/to/PXChamber/config-lte-local-characteristic.json \
  -o /absolute/path/to/new-run -p 4 -r cuda -P
```

Use `-r cpu` for CPU execution; CUDA requires compatible dependencies and an
ENABLE_CUDA build. The harness selects the platform MPI/scheduler launcher and
preserves JSON settings with `-P`. No case-specific Python launcher is required.
For the shown input directory, outputs are under `new-run/PXChamber/`.
Use a fresh output directory for each independent run.

Choose `px_chamber_axi.msh` for the original mesh or explicitly select
`px_chamber_coarse.msh` for short verification. LTE uses PLATO-generated air11
with the configured101×201 density/temperature table; do not reconstruct LTE
T with a constant gas constant. CPG uses its configured gamma/R and is a
compatibility exercise at the supplied hot-profile temperatures.

See [local characteristic boundaries](local-characteristic-boundaries.md) for
new boundary equations, input contracts, gaps from HEGEL and physical limitations.

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
example uses air11 and does not include swirl. Matching a mixture name does not
establish identical tables or a matched reference simulation.
