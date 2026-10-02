# PlasmatronX chamber-only axisymmetric case

This is an ongoing setup and verification record for the first CPG chamber
startup case. We'll keep it updated as the model, boundary conditions, inputs,
tests, or acceptance criteria change.

Implementation Status: executable startup case with passing CI Quick-equivalent checks;
not a steady-state or quantitatively validated plasma-chamber prediction.

## Case files and provenance

The case lives in
[`TestCases/Axisymmetric/NavierStokes/PXChamber`](../TestCases/Axisymmetric/NavierStokes/PXChamber/).

| File | Purpose |
| --- | --- |
| `config.json` | CPG parameters, named BCs, discretization, short startup settings |
| `px_chamber_axi.msh` | Supplied Pointwise/Gmsh mesh, normalized for MFEM input |
| `Tuvw_jet_inlet_profile.dat` | Supplied stationary radial primitive profile |
| `run_case.py` | Resolves paths, launches MPI, optionally generates the coarse mesh |
| `README.md` | Build and run quick start |

Inputs from Prathamesh: the mesh (reworked in PW) is in 
`px_chamber_axi.msh`, and (T,u,v,w) profile/reference configuration under
`PX/hegel-files`. Original profile input is used and extended to the axis,
or interpolated for intermediate values. Thermo tables unused so far.

MFEM rejected the original mesh's leading `$Comments` section. In the case
copy, comment sections were removed and `$PhysicalNames` was moved before
`$Nodes`. Pointwise must have output an older `gmsh` format.

## Geometry and equations

The Prathamesh/PW mesh spans **x = 0.4–1.05 m** and **r = y = 0–1 m**, with 75,936
nodes and 75,375 quadrilateral cells. Its axial length is 0.65 m; the existing
axial offset is retained. The inlet radius is **0.0868 m**.

To run, Theseus executable must be built with `AXISYMMETRIC=ON`. The case uses the existing
swirl-free axisymmetric compressible Navier–Stokes equations, including viscous
geometric terms. The four evolved variables are ordinary physical
`[rho, rho*ux, rho*ur, rho*E]`, not radius-weighted variables.

| Setting | Initial choice |
| --- | --- |
| Gas | Calorically perfect gas (`cpg`) |
| gamma | 1.4 |
| R | 287.05 J/(kg K) |
| Dynamic viscosity | Constant 1.8e-5 Pa s |
| Prandtl number | 0.72 |
| Initial gas | Quiescent, 10,000 Pa, 300 K |
| Polynomial order | 1 |
| Numerical flux | LLF |
| Viscous discretization | BR1 |
| Time integrator | RK3SSP |
| Subcell FV blending | Enabled; alpha_max = 0.5 |

There is no swirl equation and no added volumetric plasma heating. Energy
enters through the heated boundary data. The current simulation uses CPG
EOS; LTE is a later phase.

## Boundary conditions

Names are case-sensitive and match the mesh physical groups.

| Group | Location | Current treatment |
| --- | --- | --- |
| `Axis` | r = 0 | Existing `axis` reflection and regularity treatment |
| `Inflow` | x = 0.4, 0 <= r <= 0.0868 | `cpg-radial-profile`: measured T, ux, ur and fixed 10,000 Pa exterior state |
| `Left` | x = 0.4, r > 0.0868 | `no-slip-isothermal`, zero wall velocity, 300 K |
| `Top` | r = 1 | `cpg-exterior-state`: quiescent gas at 10,000 Pa and 300 K |
| `Right` | x = 1.05 | Existing `supersonic-outflow` extrapolation |

The inlet/top states enter the numerical flux **weakly**. This first pass does
not implement characteristic subsonic conditions needed for the top boundary.
The right boundary does not impose outlet pressure, despite the specified ambient
pressure elsewhere. The type name `supersonic-outflow` does not mean this case is
supersonic, it is just the BC we used for now.

The reference HEGEL input uses a top total-condition inlet with inward radial
direction. The present quiescent ambient exterior state is a deliberate first
approximation, not a reproduction of that reference BC.

A new prescribed exterior-state BC type supplies an entropy-variable difference
as the boundary correction in the auxiliary gradient equation, and includes
the interior-gradient viscous flux.
It is explicitly restricted to 2D CPG; switching `gas_model` to LTE is not yet
supported by these new input paths.

## Radial profile contract

The existing input format is retained:

```text
number_of_points number_of_blocks
R flag T ux ur swirl
...
```

Units are meters, kelvin, and meters/second. Temperature is currently interpreted
as **static temperature** (need to check).
The supplied header is `335 2`. Select `flag: 0`; the flag-1 block is identical
apart from the flag. These blocks are not concatenated into duplicate radii.
Swirl is read but intentionally ignored.

Within the selected block, radius must increase strictly and temperature must
be positive. Invalid rows, nonfinite values, count mismatches, and missing flags
are rejected. The table spans r = 0.00025 to approximately 0.991239 m, covering
the complete inlet.

- Interpolate T, ux, and ur linearly in radius, then construct the CPG state.
- For 0 <= r <= the first sample radius, keep T and ux constant and scale ur
  linearly to zero at the axis. This defines even T/ux and odd ur extensions.
- Preserve signed axial/radial velocities, including weak local reverse flow.
- Reject evaluation above the table's radial coverage; do not silently extrapolate.
- Apply the table only to `Inflow`; it is not a wall-temperature profile.

The first sample is approximately 10,684 K and 316.17 m/s axial velocity.
The sampled inlet is subsonic for the selected CPG constants (maximum Mach
approximately 0.153). The exterior state is sampled once in boundary restriction
point order and stored in device-readable payloads.

Relevant implementation files: `include/RadialProfile.hpp`, BC descriptors/cache
and kernels, `include/RHSOperator_impl.hpp`, the Euler/CNS boundary callers, and
`src/Simulation.cpp`. Numeric wall values and a uniform `cpg_state` initializer
avoid adding case-specific registrations to the condition factory.

## Reproducing runs

Build with the appropriate local MFEM dependency prefix and `-DAXISYMMETRIC=ON`;
see the [case quick start](../TestCases/Axisymmetric/NavierStokes/PXChamber/README.md)
for the tested Mac build command. From the repository root:

```sh
# Supplied mesh: 100 steps, 1e-7 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-full --ranks 2

# Coarse 864-cell mesh: 20,000 steps, 2e-4 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-coarse --ranks 2 \
  --coarse --dt 1e-8 --final-time 2e-4
```

Use distinct output directories for separate experiments. Each contains the
resolved `config.json`, `run.log`, and `ParaView/ParaView.pvd`. The runner handles
OpenMPI localhost placement and uses the CPU device. Raw config paths are
repository-root relative; the helper makes them absolute.

## Tests added and what they establish

| Test | Checks | Limits |
| --- | --- | --- |
| `RadialProfileTests` | Linear interpolation; axis extension; signed velocity; CPG pressure/velocity recovery; equality of supplied blocks; missing flag, duplicate radii and negative-temperature rejection; upper-coverage rejection | Unit checks, not a boundary-flow solution |
| `prescribed_state_boundary_correction_uses_exterior_entropy` inside `AxisymmetricGeometryTests` | Zero boundary correction for matching entropy states; expected correction for a known perturbation | Kernel-level check |
| `PXChamberIntegration` | Coarse 20-step startup to 2e-7 s; quiescent preservation on one rank; heated runs on one/two ranks; positive density, pressure and temperature; Tmax > 301 K; rank agreement | Compares extrema and an axis-velocity statistic, not every field value or integrated conservation |

The integration test reads the written VTK data. Uniform-state tolerances are
relative 1e-10 / absolute 1e-9. Heated serial/MPI statistics use relative 1e-9 /
absolute 1e-8. Temperature is reconstructed from p/(rho R).

The existing axis-reflection tests check the parity operation. The chamber test
does not assert identically zero interior nodal radial velocity during heated
startup: axis conditions are imposed weakly. Need to evaluate that error vs.
refinement (it is not absolutely enforced).

CI Quick runs the profile and kernel tests in its regular CTest step to make sure
the case keeps running.

## Recorded validation

| Check | Recorded outcome |
| --- | --- |
| Cartesian/Plato CTest | 17/17 passed |
| Axisymmetric CTest | 21/21 passed |
| Quick smoke cases | IsentropicVortex, LidDrivenCavity, LTEVortex passed |
| Quick golden comparisons | IsentropicVortex, LTEVortex, LidDrivenCavity, TaylorGreenVortex2D, ForwardFacingStep passed at workflow tolerances |
| Cyclic vortex comparison | Passed, Density atol 5e-5 / rtol 2e-6 |
| Expanded Quick/Nightly selector | Lists the existing five integration/convergence checks plus PXChamberIntegration |

These are local CPU/Mac/OpenMPI equivalents, not a claim that GitHub-hosted
Ubuntu CI or accelerator execution was run.

| Chamber run | Time | Temperature range | Pressure range |
| --- | --- | --- | --- |
| Supplied mesh, two ranks | 1e-7 s | 300–566 K | 9999–10849 Pa |
| Coarse mesh, two ranks | 2e-4 s | approximately 300–4109 K | approximately 10000–11022 Pa |

Both completed with positive reported thermodynamic states. The coarse run's
maximum interior nodal axis radial velocity was approximately 0.307 m/s. Neither
run establishes steady state, mesh convergence, exact profile recovery, or a
closed mass/energy budget. A prescribed exterior target of 10,684 K does not
make the transient interior inlet trace instantly attain that value.

## Plan for what to do next

1. Inspect simulation fields: transients, inlet trace; quantify profile recovery,
   wall-temperature/no-slip error and axis error. Check timestep sensitivity.
2. Add supporting numerical boundary mass/energy flux diagnostics, including accumulation
   and wall heat transfer, using axisymmetric 2*pi*r weighting.
3. Add supporting BCs: subsonic inlet/open/outlet treatment, including pressure control and
   backflow; test pressure response and acoustic reflection before longer runs.
4. Convergence: mesh/order sensitivity and run long enough to see meaningful results.
5. Switch to LTE: EOS-compatible BCs as needed. Note:The reference input from Prathamesh mentions
   air-11; the available Theseus default is air5.
6. Add swirl option to axi treatment
iff --git a/docs/px-chamber.md b/docs/px-chamber.md
ew file mode 100644
ndex 0000000..c42dafa
-- /dev/null
++ b/docs/px-chamber.md
@ -0,0 +1,212 @@
# PlasmatronX chamber-only axisymmetric case

This is an ongoing setup and verification record for the first CPG chamber
startup case. We'll keep it updated as the model, boundary conditions, inputs,
tests, or acceptance criteria change.

Implementation Status: executable startup case with passing CI Quick-equivalent checks;
not a steady-state or quantitatively validated plasma-chamber prediction.

## Case files and provenance

The case lives in
[`TestCases/Axisymmetric/NavierStokes/PXChamber`](../TestCases/Axisymmetric/NavierStokes/PXChamber/).

| File | Purpose |
| --- | --- |
| `config.json` | CPG parameters, named BCs, discretization, short startup settings |
| `px_chamber_axi.msh` | Supplied Pointwise/Gmsh mesh, normalized for MFEM input |
| `Tuvw_jet_inlet_profile.dat` | Supplied stationary radial primitive profile |
| `run_case.py` | Resolves paths, launches MPI, optionally generates the coarse mesh |
| `README.md` | Build and run quick start |

Inputs from Prathamesh: the mesh (reworked in PW) is in 
`px_chamber_axi.msh`, and (T,u,v,w) profile/reference configuration under
`PX/hegel-files`. Original profile input is used and extended to the axis,
or interpolated for intermediate values. Thermo tables unused so far.

MFEM rejected the original mesh's leading `$Comments` section. In the case
copy, comment sections were removed and `$PhysicalNames` was moved before
`$Nodes`. Pointwise must have output an older `gmsh` format.

## Geometry and equations

The Prathamesh/PW mesh spans **x = 0.4–1.05 m** and **r = y = 0–1 m**, with 75,936
nodes and 75,375 quadrilateral cells. Its axial length is 0.65 m; the existing
axial offset is retained. The inlet radius is **0.0868 m**.

To run, Theseus executable must be built with `AXISYMMETRIC=ON`. The case uses the existing
swirl-free axisymmetric compressible Navier–Stokes equations, including viscous
geometric terms. The four evolved variables are ordinary physical
`[rho, rho*ux, rho*ur, rho*E]`, not radius-weighted variables.

| Setting | Initial choice |
| --- | --- |
| Gas | Calorically perfect gas (`cpg`) |
| gamma | 1.4 |
| R | 287.05 J/(kg K) |
| Dynamic viscosity | Constant 1.8e-5 Pa s |
| Prandtl number | 0.72 |
| Initial gas | Quiescent, 10,000 Pa, 300 K |
| Polynomial order | 1 |
| Numerical flux | LLF |
| Viscous discretization | BR1 |
| Time integrator | RK3SSP |
| Subcell FV blending | Enabled; alpha_max = 0.5 |

There is no swirl equation and no added volumetric plasma heating. Energy
enters through the heated boundary data. The current simulation uses CPG
EOS; LTE is a later phase.

## Boundary conditions

Names are case-sensitive and match the mesh physical groups.

| Group | Location | Current treatment |
| --- | --- | --- |
| `Axis` | r = 0 | Existing `axis` reflection and regularity treatment |
| `Inflow` | x = 0.4, 0 <= r <= 0.0868 | `cpg-radial-profile`: measured T, ux, ur and fixed 10,000 Pa exterior state |
| `Left` | x = 0.4, r > 0.0868 | `no-slip-isothermal`, zero wall velocity, 300 K |
| `Top` | r = 1 | `cpg-exterior-state`: quiescent gas at 10,000 Pa and 300 K |
| `Right` | x = 1.05 | Existing `supersonic-outflow` extrapolation |

The inlet/top states enter the numerical flux **weakly**. This first pass does
not implement characteristic subsonic conditions needed for the top boundary.
The right boundary does not impose outlet pressure, despite the specified ambient
pressure elsewhere. The type name `supersonic-outflow` does not mean this case is
supersonic, it is just the BC we used for now.

The reference HEGEL input uses a top total-condition inlet with inward radial
direction. The present quiescent ambient exterior state is a deliberate first
approximation, not a reproduction of that reference BC.

A new prescribed exterior-state BC type supplies an entropy-variable difference
as the boundary correction in the auxiliary gradient equation, and includes
the interior-gradient viscous flux.
It is explicitly restricted to 2D CPG; switching `gas_model` to LTE is not yet
supported by these new input paths.

## Radial profile contract

The existing input format is retained:

```text
number_of_points number_of_blocks
R flag T ux ur swirl
...
```

Units are meters, kelvin, and meters/second. Temperature is currently interpreted
as **static temperature** (need to check).
The supplied header is `335 2`. Select `flag: 0`; the flag-1 block is identical
apart from the flag. These blocks are not concatenated into duplicate radii.
Swirl is read but intentionally ignored.

Within the selected block, radius must increase strictly and temperature must
be positive. Invalid rows, nonfinite values, count mismatches, and missing flags
are rejected. The table spans r = 0.00025 to approximately 0.991239 m, covering
the complete inlet.

- Interpolate T, ux, and ur linearly in radius, then construct the CPG state.
- For 0 <= r <= the first sample radius, keep T and ux constant and scale ur
  linearly to zero at the axis. This defines even T/ux and odd ur extensions.
- Preserve signed axial/radial velocities, including weak local reverse flow.
- Reject evaluation above the table's radial coverage; do not silently extrapolate.
- Apply the table only to `Inflow`; it is not a wall-temperature profile.

The first sample is approximately 10,684 K and 316.17 m/s axial velocity.
The sampled inlet is subsonic for the selected CPG constants (maximum Mach
approximately 0.153). The exterior state is sampled once in boundary restriction
point order and stored in device-readable payloads.

Relevant implementation files: `include/RadialProfile.hpp`, BC descriptors/cache
and kernels, `include/RHSOperator_impl.hpp`, the Euler/CNS boundary callers, and
`src/Simulation.cpp`. Numeric wall values and a uniform `cpg_state` initializer
avoid adding case-specific registrations to the condition factory.

## Reproducing runs

Build with the appropriate local MFEM dependency prefix and `-DAXISYMMETRIC=ON`;
see the [case quick start](../TestCases/Axisymmetric/NavierStokes/PXChamber/README.md)
for the tested Mac build command. From the repository root:

```sh
# Supplied mesh: 100 steps, 1e-7 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-full --ranks 2

# Coarse 864-cell mesh: 20,000 steps, 2e-4 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-coarse --ranks 2 \
  --coarse --dt 1e-8 --final-time 2e-4
```

Use distinct output directories for separate experiments. Each contains the
resolved `config.json`, `run.log`, and `ParaView/ParaView.pvd`. The runner handles
OpenMPI localhost placement and uses the CPU device. Raw config paths are
repository-root relative; the helper makes them absolute.

## Tests added and what they establish

| Test | Checks | Limits |
| --- | --- | --- |
| `RadialProfileTests` | Linear interpolation; axis extension; signed velocity; CPG pressure/velocity recovery; equality of supplied blocks; missing flag, duplicate radii and negative-temperature rejection; upper-coverage rejection | Unit checks, not a boundary-flow solution |
| `prescribed_state_boundary_correction_uses_exterior_entropy` inside `AxisymmetricGeometryTests` | Zero boundary correction for matching entropy states; expected correction for a known perturbation | Kernel-level check |
| `PXChamberIntegration` | Coarse 20-step startup to 2e-7 s; quiescent preservation on one rank; heated runs on one/two ranks; positive density, pressure and temperature; Tmax > 301 K; rank agreement | Compares extrema and an axis-velocity statistic, not every field value or integrated conservation |

The integration test reads the written VTK data. Uniform-state tolerances are
relative 1e-10 / absolute 1e-9. Heated serial/MPI statistics use relative 1e-9 /
absolute 1e-8. Temperature is reconstructed from p/(rho R).

The existing axis-reflection tests check the parity operation. The chamber test
does not assert identically zero interior nodal radial velocity during heated
startup: axis conditions are imposed weakly. Need to evaluate that error vs.
refinement (it is not absolutely enforced).

CI Quick runs the profile and kernel tests in its regular CTest step to make sure
the case keeps running.

## Recorded validation

| Check | Recorded outcome |
| --- | --- |
| Cartesian/Plato CTest | 17/17 passed |
| Axisymmetric CTest | 21/21 passed |
| Quick smoke cases | IsentropicVortex, LidDrivenCavity, LTEVortex passed |
| Quick golden comparisons | IsentropicVortex, LTEVortex, LidDrivenCavity, TaylorGreenVortex2D, ForwardFacingStep passed at workflow tolerances |
| Cyclic vortex comparison | Passed, Density atol 5e-5 / rtol 2e-6 |
| Expanded Quick/Nightly selector | Lists the existing five integration/convergence checks plus PXChamberIntegration |

These are local CPU/Mac/OpenMPI equivalents, not a claim that GitHub-hosted
Ubuntu CI or accelerator execution was run.

| Chamber run | Time | Temperature range | Pressure range |
| --- | --- | --- | --- |
| Supplied mesh, two ranks | 1e-7 s | 300–566 K | 9999–10849 Pa |
| Coarse mesh, two ranks | 2e-4 s | approximately 300–4109 K | approximately 10000–11022 Pa |

Both completed with positive reported thermodynamic states. The coarse run's
maximum interior nodal axis radial velocity was approximately 0.307 m/s. Neither
run establishes steady state, mesh convergence, exact profile recovery, or a
closed mass/energy budget. A prescribed exterior target of 10,684 K does not
make the transient interior inlet trace instantly attain that value.

## Plan for what to do next

1. Inspect simulation fields: transients, inlet trace; quantify profile recovery,
   wall-temperature/no-slip error and axis error. Check timestep sensitivity.
2. Add supporting numerical boundary mass/energy flux diagnostics, including accumulation
   and wall heat transfer, using axisymmetric 2*pi*r weighting.
3. Add supporting BCs: subsonic inlet/open/outlet treatment, including pressure control and
   backflow; test pressure response and acoustic reflection before longer runs.
4. Convergence: mesh/order sensitivity and run long enough to see meaningful results.
5. Switch to LTE: EOS-compatible BCs as needed. Note:The reference input from Prathamesh mentions
   air-11; the available Theseus default is air5.
6. Add swirl option to axi treatment
iff --git a/docs/px-chamber.md b/docs/px-chamber.md
ew file mode 100644
ndex 0000000..c42dafa
-- /dev/null
++ b/docs/px-chamber.md
@ -0,0 +1,212 @@
# PlasmatronX chamber-only axisymmetric case

This is an ongoing setup and verification record for the first CPG chamber
startup case. We'll keep it updated as the model, boundary conditions, inputs,
tests, or acceptance criteria change.

Implementation Status: executable startup case with passing CI Quick-equivalent checks;
not a steady-state or quantitatively validated plasma-chamber prediction.

## Case files and provenance

The case lives in
[`TestCases/Axisymmetric/NavierStokes/PXChamber`](../TestCases/Axisymmetric/NavierStokes/PXChamber/).

| File | Purpose |
| --- | --- |
| `config.json` | CPG parameters, named BCs, discretization, short startup settings |
| `px_chamber_axi.msh` | Supplied Pointwise/Gmsh mesh, normalized for MFEM input |
| `Tuvw_jet_inlet_profile.dat` | Supplied stationary radial primitive profile |
| `run_case.py` | Resolves paths, launches MPI, optionally generates the coarse mesh |
| `README.md` | Build and run quick start |

Inputs from Prathamesh: the mesh (reworked in PW) is in 
`px_chamber_axi.msh`, and (T,u,v,w) profile/reference configuration under
`PX/hegel-files`. Original profile input is used and extended to the axis,
or interpolated for intermediate values. Thermo tables unused so far.

MFEM rejected the original mesh's leading `$Comments` section. In the case
copy, comment sections were removed and `$PhysicalNames` was moved before
`$Nodes`. Pointwise must have output an older `gmsh` format.

## Geometry and equations

The Prathamesh/PW mesh spans **x = 0.4–1.05 m** and **r = y = 0–1 m**, with 75,936
nodes and 75,375 quadrilateral cells. Its axial length is 0.65 m; the existing
axial offset is retained. The inlet radius is **0.0868 m**.

To run, Theseus executable must be built with `AXISYMMETRIC=ON`. The case uses the existing
swirl-free axisymmetric compressible Navier–Stokes equations, including viscous
geometric terms. The four evolved variables are ordinary physical
`[rho, rho*ux, rho*ur, rho*E]`, not radius-weighted variables.

| Setting | Initial choice |
| --- | --- |
| Gas | Calorically perfect gas (`cpg`) |
| gamma | 1.4 |
| R | 287.05 J/(kg K) |
| Dynamic viscosity | Constant 1.8e-5 Pa s |
| Prandtl number | 0.72 |
| Initial gas | Quiescent, 10,000 Pa, 300 K |
| Polynomial order | 1 |
| Numerical flux | LLF |
| Viscous discretization | BR1 |
| Time integrator | RK3SSP |
| Subcell FV blending | Enabled; alpha_max = 0.5 |

There is no swirl equation and no added volumetric plasma heating. Energy
enters through the heated boundary data. The current simulation uses CPG
EOS; LTE is a later phase.

## Boundary conditions

Names are case-sensitive and match the mesh physical groups.

| Group | Location | Current treatment |
| --- | --- | --- |
| `Axis` | r = 0 | Existing `axis` reflection and regularity treatment |
| `Inflow` | x = 0.4, 0 <= r <= 0.0868 | `cpg-radial-profile`: measured T, ux, ur and fixed 10,000 Pa exterior state |
| `Left` | x = 0.4, r > 0.0868 | `no-slip-isothermal`, zero wall velocity, 300 K |
| `Top` | r = 1 | `cpg-exterior-state`: quiescent gas at 10,000 Pa and 300 K |
| `Right` | x = 1.05 | Existing `supersonic-outflow` extrapolation |

The inlet/top states enter the numerical flux **weakly**. This first pass does
not implement characteristic subsonic conditions needed for the top boundary.
The right boundary does not impose outlet pressure, despite the specified ambient
pressure elsewhere. The type name `supersonic-outflow` does not mean this case is
supersonic, it is just the BC we used for now.

The reference HEGEL input uses a top total-condition inlet with inward radial
direction. The present quiescent ambient exterior state is a deliberate first
approximation, not a reproduction of that reference BC.

A new prescribed exterior-state BC type supplies an entropy-variable difference
as the boundary correction in the auxiliary gradient equation, and includes
the interior-gradient viscous flux.
It is explicitly restricted to 2D CPG; switching `gas_model` to LTE is not yet
supported by these new input paths.

## Radial profile contract

The existing input format is retained:

```text
number_of_points number_of_blocks
R flag T ux ur swirl
...
```

Units are meters, kelvin, and meters/second. Temperature is currently interpreted
as **static temperature** (need to check).
The supplied header is `335 2`. Select `flag: 0`; the flag-1 block is identical
apart from the flag. These blocks are not concatenated into duplicate radii.
Swirl is read but intentionally ignored.

Within the selected block, radius must increase strictly and temperature must
be positive. Invalid rows, nonfinite values, count mismatches, and missing flags
are rejected. The table spans r = 0.00025 to approximately 0.991239 m, covering
the complete inlet.

- Interpolate T, ux, and ur linearly in radius, then construct the CPG state.
- For 0 <= r <= the first sample radius, keep T and ux constant and scale ur
  linearly to zero at the axis. This defines even T/ux and odd ur extensions.
- Preserve signed axial/radial velocities, including weak local reverse flow.
- Reject evaluation above the table's radial coverage; do not silently extrapolate.
- Apply the table only to `Inflow`; it is not a wall-temperature profile.

The first sample is approximately 10,684 K and 316.17 m/s axial velocity.
The sampled inlet is subsonic for the selected CPG constants (maximum Mach
approximately 0.153). The exterior state is sampled once in boundary restriction
point order and stored in device-readable payloads.

Relevant implementation files: `include/RadialProfile.hpp`, BC descriptors/cache
and kernels, `include/RHSOperator_impl.hpp`, the Euler/CNS boundary callers, and
`src/Simulation.cpp`. Numeric wall values and a uniform `cpg_state` initializer
avoid adding case-specific registrations to the condition factory.

## Reproducing runs

Build with the appropriate local MFEM dependency prefix and `-DAXISYMMETRIC=ON`;
see the [case quick start](../TestCases/Axisymmetric/NavierStokes/PXChamber/README.md)
for the tested Mac build command. From the repository root:

```sh
# Supplied mesh: 100 steps, 1e-7 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-full --ranks 2

# Coarse 864-cell mesh: 20,000 steps, 2e-4 s total.
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-coarse --ranks 2 \
  --coarse --dt 1e-8 --final-time 2e-4
```

Use distinct output directories for separate experiments. Each contains the
resolved `config.json`, `run.log`, and `ParaView/ParaView.pvd`. The runner handles
OpenMPI localhost placement and uses the CPU device. Raw config paths are
repository-root relative; the helper makes them absolute.

## Tests added and what they establish

| Test | Checks | Limits |
| --- | --- | --- |
| `RadialProfileTests` | Linear interpolation; axis extension; signed velocity; CPG pressure/velocity recovery; equality of supplied blocks; missing flag, duplicate radii and negative-temperature rejection; upper-coverage rejection | Unit checks, not a boundary-flow solution |
| `prescribed_state_boundary_correction_uses_exterior_entropy` inside `AxisymmetricGeometryTests` | Zero boundary correction for matching entropy states; expected correction for a known perturbation | Kernel-level check |
| `PXChamberIntegration` | Coarse 20-step startup to 2e-7 s; quiescent preservation on one rank; heated runs on one/two ranks; positive density, pressure and temperature; Tmax > 301 K; rank agreement | Compares extrema and an axis-velocity statistic, not every field value or integrated conservation |

The integration test reads the written VTK data. Uniform-state tolerances are
relative 1e-10 / absolute 1e-9. Heated serial/MPI statistics use relative 1e-9 /
absolute 1e-8. Temperature is reconstructed from p/(rho R).

The existing axis-reflection tests check the parity operation. The chamber test
does not assert identically zero interior nodal radial velocity during heated
startup: axis conditions are imposed weakly. Need to evaluate that error vs.
refinement (it is not absolutely enforced).

CI Quick runs the profile and kernel tests in its regular CTest step to make sure
the case keeps running.

## Recorded validation

| Check | Recorded outcome |
| --- | --- |
| Cartesian/Plato CTest | 17/17 passed |
| Axisymmetric CTest | 21/21 passed |
| Quick smoke cases | IsentropicVortex, LidDrivenCavity, LTEVortex passed |
| Quick golden comparisons | IsentropicVortex, LTEVortex, LidDrivenCavity, TaylorGreenVortex2D, ForwardFacingStep passed at workflow tolerances |
| Cyclic vortex comparison | Passed, Density atol 5e-5 / rtol 2e-6 |
| Expanded Quick/Nightly selector | Lists the existing five integration/convergence checks plus PXChamberIntegration |

These are local CPU/Mac/OpenMPI equivalents, not a claim that GitHub-hosted
Ubuntu CI or accelerator execution was run.

| Chamber run | Time | Temperature range | Pressure range |
| --- | --- | --- | --- |
| Supplied mesh, two ranks | 1e-7 s | 300–566 K | 9999–10849 Pa |
| Coarse mesh, two ranks | 2e-4 s | approximately 300–4109 K | approximately 10000–11022 Pa |

Both completed with positive reported thermodynamic states. The coarse run's
maximum interior nodal axis radial velocity was approximately 0.307 m/s. Neither
run establishes steady state, mesh convergence, exact profile recovery, or a
closed mass/energy budget. A prescribed exterior target of 10,684 K does not
make the transient interior inlet trace instantly attain that value.

## Plan for what to do next

1. Inspect simulation fields: transients, inlet trace; quantify profile recovery,
   wall-temperature/no-slip error and axis error. Check timestep sensitivity.
2. Add supporting numerical boundary mass/energy flux diagnostics, including accumulation
   and wall heat transfer, using axisymmetric 2*pi*r weighting.
3. Add supporting BCs: subsonic inlet/open/outlet treatment, including pressure control and
   backflow; test pressure response and acoustic reflection before longer runs.
4. Convergence: mesh/order sensitivity and run long enough to see meaningful results.
5. Switch to LTE: EOS-compatible BCs as needed. Note:The reference input from Prathamesh mentions
   air-11; the available Theseus default is air5.
6. Add swirl option to axi treatment
