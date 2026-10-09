# Using physical states with CPG and LTE

Use physical input forms when the selected gas model should determine conservative
energy. Keep mathematical conservative benchmarks and checkpoints in conservative
form. Merely changing `gas_model` does not reinterpret legacy state functions.

## Constant initial and exterior states

Replace an initial `cpg_state` with an explicit physical state:

```json
"initial_conditions": {
  "state": {"pressure": 10000, "temperature": 300, "velocity": [0, 0]}
}
```

Replace a CPG exterior boundary with:

```json
"Top": {
  "type": "exterior-state",
  "state": {"pressure": 10000, "temperature": 300, "velocity": [0, 0]}
}
```

The physical forms require exactly two of `pressure`, `density`, and `temperature`,
and an explicit velocity vector of the spatial dimension. They reject unknown fields,
composition and passive scalars. Units must match the simulation's consistent unit
system; the chamber example uses SI. Axisymmetric velocity is `[axial, radial]`.
The evolved state remains `[rho, rho*u..., rho*E]` with no radius weighting.

## Stationary profiles

Replace `cpg-radial-profile` with `radial-profile`:

```json
"Inflow": {
  "type": "radial-profile",
  "file": "TestCases/Axisymmetric/NavierStokes/PXChamber/Tuvw_jet_inlet_profile.dat",
  "flag": 0,
  "pressure": 10000
}
```

The existing six-column file format, primitive interpolation, axis extension,
coverage checks, signed velocities and swirl-free interpretation remain the same.
Remove profile-local `gamma` or `R_gas` overrides: the selected gas supplies the EOS.
Conversion happens after interpolation at actual boundary restriction coordinates.
See [physical state configuration](physical-state-conversion.md) for the complete
input contract and error behavior.

Walls still take wall temperature/velocity or heat-flux data; they are not full
exterior-state specifications. The isothermal wall uses the entropy normalization
of its EOS: `1/(R*Twall)` for CPG and `1/Twall` for LTE. Axis, symmetry and extrapolation
boundaries operate on the interior state. Physical exterior states impose numerical
flux targets, not characteristic subsonic inlet/outlet conditions.

## LTE table coverage

Set `gas_model` to `lte` and supply the mixture, database and table ranges. Cover the
physical initial state, every sampled boundary state, and anticipated transient
states. High temperature at fixed pressure often requires densities far below the
cold-gas density. Both forward density/temperature coverage and inverse energy
coverage matter. A CPG energy estimate cannot establish LTE coverage.

Startup conversion rejects unsupported/out-of-table states before device-cache
creation. Errors identify the initial-condition path or boundary/radius. Do not
bypass these checks or broaden ranges blindly after a failure; inspect the requested
physical state and EOS range. Table resolution and transient coverage remain modeling
choices that require convergence/sensitivity checks.

The runnable [LTE chamber example](px-chamber.md#lte-startup) combines physical
initial data, a radial inlet, an exterior state, an isothermal wall and derived output.
Its air5 table is an example; it does not reproduce an air-11 reference calculation.

## Visualization

Request selected model-consistent quantities explicitly:

```json
"visualization": {
  "fields": ["density", "velocity", "pressure", "temperature", "mach_number",
             "specific_internal_energy", "viscosity", "thermal_conductivity"]
}
```

Defaults retain Density, Velocity, Pressure and available Blending Coeff output.
Use the Temperature field for LTE; `p/(rho*constant_R)` is not an LTE temperature
reconstruction. See [visualization](visualization.md) for field meanings and costs.

## Compatibility

- `cpg_state`, `cpg-exterior-state` and `cpg-radial-profile` retain CPG-only restrictions.
- Registered conservative functions, including legacy `LTEVortexIC` and `LTEBlobIC`,
  retain their formulas despite their names. Use `physical_profile` when the selected
  EOS should construct their energy. These prescribed profiles are not exact LTE
  solutions merely because conversion is EOS-consistent.
- Constant conservative inflow vectors and exact-solution references are not converted.
- Restart loads conservative values unchanged and bypasses unused initial inputs;
  boundary configuration still applies. Derived fields are recomputed from the loaded
  state. Reinterpreting a checkpoint with another gas model is not a physical-state
  migration procedure.
- Arbitrary runtime boundary callbacks and composition/scalar conversion are unsupported.
  Future device-dependent boundaries require concrete typed gas dispatch, not host
  callbacks captured in kernels.
