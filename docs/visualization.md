# Visualization output

Theseus writes visualization data when `runTime.visualize` is `true`. The
`runTime.visualization.fields` array controls which solution fields are included:

```json
"visualize": true,
"visualization": {
  "fields": ["density", "velocity", "pressure"]
}
```

Supported selections and output array names are:

| Selection | Output name | Meaning (usual SI units) |
| --- | --- | --- |
| `density` | Density | Mass density, kg/m³ |
| `velocity` | Velocity | Velocity vector, m/s |
| `pressure` | Pressure | Selected-EOS pressure, Pa |
| `temperature` | Temperature | Selected-EOS temperature, K |
| `sound_speed` | Sound Speed | Selected-EOS sound speed, m/s |
| `mach_number` | Mach Number | Velocity magnitude divided by sound speed |
| `specific_internal_energy` | Specific Internal Energy | Internal energy per mass, J/kg |
| `internal_energy_density` | Internal Energy Density | Internal energy per volume, J/m³ |
| `specific_total_energy` | Specific Total Energy | Total energy per mass, including kinetic energy, J/kg |
| `viscosity` | Viscosity | Dynamic viscosity, Pa·s |
| `thermal_conductivity` | Thermal Conductivity | Thermal conductivity, W/(m·K) |
| `blending_coefficient` | Blending Coeff | Subcell blending coefficient; requires `SUBCELL_FV_BLENDING` |

If `visualization` or `visualization.fields` is omitted, the defaults remain
`density`, `velocity`, `pressure`, and `blending_coefficient` when available.
Additional quantities require explicit selection. Duplicate field names are ignored.
An empty list, unknown name, or unavailable blending field is a configuration error.

For example, temperature and Mach output can be selected without writing pressure
or sound speed:

```json
"visualization": {"fields": ["temperature", "mach_number"]}
```

Pressure, temperature, sound speed and transport properties use the simulation's
selected gas model. LTE conductivity uses the conductivity table, even when
Sutherland viscosity is enabled. CPG conductivity uses `mu * cp / Pr`. Energy fields
come from the conservative state after subtracting kinetic energy where appropriate;
the EOS energy reference is retained, so internal energy need not be positive.
Axisymmetric values are physical quantities without radius weighting.

`velocity` is one logical selection in every spatial dimension. It is written
as a single vector-valued `Velocity` array with the simulation dimension as its
component count.

## Mesh representation

ParaView output supports two mesh representations through
`runTime.visualization.mesh_mode`:

```json
"visualization": {
  "fields": ["density", "velocity", "pressure"],
  "mesh_mode": "gll_subcells"
}
```

- `vtk_high_order` writes VTK Lagrange cells using regularly spaced reference
  points. Select it explicitly when compatibility with older Theseus output is
  required.
- `gll_subcells` writes the solver's Gauss–Lobatto nodes as points and connects
  adjacent nodes with ordinary linear VTK cells. Field values are therefore
  stored at the original DGSEM nodal locations without interpolation to a
  regularly spaced high-order representation. This is the default.

`gll_subcells` is intended for the tensor-product segment, quadrilateral, and
hexahedral elements used by the DGSEM discretization. ParaView and other VTK
tools can process its linear cells without special high-order-element support.


## Evaluation and field ownership

`VisualizationConfig` holds each field's configuration/output names, component
count and property dependency. `VisualizationFields` allocates only selected derived
fields, references the existing density/blending fields, and registers the same
selection with either ParaView or VisIt. Field addresses remain stable for the data
collection's lifetime.

Each output event uses one host traversal and one combined gas-property request per
point. LTE shares temperature recovery and table-cell weights among requested
properties. Mach requests sound speed internally without requiring a Sound Speed
output array. Purely algebraic selections perform no EOS property evaluation;
density/blending-only selections require no derived-field traversal. Disabling
visualization creates no derived fields and performs no visualization evaluation.

Derived quantities are evaluated at solution nodes. High-order output interpolates
those derived fields to its output points; it does not perform another EOS evaluation
at each interpolated coordinate.

Device solutions transfer to the host at output. Parsing, registry lookup and field
allocation occur outside timestep kernels. The optional `ENABLE_TIMERS` build flag
reports the `VisualizationFields` update separately from file writing. Checkpoint
values remain conservative; derived fields are recomputed after loading a restart.

`VisualizationFieldsTests` covers registration, values, lifetime and dependencies
in 1D/2D/3D. `DerivedVisualizationIntegration` checks emitted CPG/LTE fields against
individual gas-model queries, both ParaView mesh modes, VisIt output, selected fields,
two-rank MPI and conservative restart. The integration test honors
`PHYSICAL_STATE_TEST_DEVICE` for device runs.
