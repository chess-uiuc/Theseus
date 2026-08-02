# Visualization output

Theseus writes visualization data when `runTime.visualize` is `true`. The
`runTime.visualization.fields` array controls which solution fields are included:

```json
"visualize": true,
"visualization": {
  "fields": ["density", "velocity", "pressure"]
}
```

Supported field selections are:

- `density`
- `velocity`
- `pressure`
- `blending_coefficient`, when Theseus is built with `SUBCELL_FV_BLENDING`

If `visualization` or `visualization.fields` is omitted, Theseus writes every
available field, preserving the behavior of existing configuration files.
Duplicate field names are ignored. An empty list, an unknown field, or a field
that is unavailable in the current build is a configuration error.

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
