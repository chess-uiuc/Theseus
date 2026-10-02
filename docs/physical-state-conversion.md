# Host physical-state conversion

`GasModelInterface::ConservativeFromPhysical(input, output)` constructs a contiguous
canonical state `[rho, rho*u..., rho*E]` using the selected gas model. This is a
host-only startup API. Initial conditions use it before projection; physical
boundary inputs use it before device-cache construction.
Legacy registered initial conditions supply conservative states directly.

```cpp
PhysicalStateInput input{PressureTemperature{101325.0, 300.0}, {20.0, 0.0}};
mfem::Vector conservative;
gas.ConservativeFromPhysical(input, conservative);
```

The thermodynamic variant accepts exactly one pair: `PressureTemperature`,
`DensityTemperature`, or `DensityPressure`. Velocity must contain exactly the
model dimension's number of components. Values use the simulation's consistent
unit system (normally Pa, K, kg/m^3 and m/s). Axisymmetric velocity is axial/radial;
no radius weighting is applied. Scalars and composition are not supported by this
API's current CPG/LTE implementations. Conservative state views must not be used
as primitive input views; the existing rho/u/p conversion uses `PointPrimitiveView`.

Invalid inputs throw `std::invalid_argument`; the output is unchanged on validation
failure. Configuration errors include the initial-condition path; spatial conversion errors
also include sample coordinates. Typed input rules out missing/excess thermodynamic
quantities and the JSON reader rejects them before conversion.

CPG uses the model's gas constant and gamma. LTE uses its existing forward table
interpolation. With one coordinate fixed, pressure is piecewise linear: the host
solver examines all grid intervals and solves the unique bracket directly. It
rejects missing roots, multiple roots and constant-pressure intervals. It neither
extrapolates nor constructs another table.

LTE checks forward density/temperature bounds, inverse specific-energy bounds,
and the packed conservative state's inverse-temperature recovery. The latter
uses the runtime Newton update and stopping rules, with bounds guards before every
lookup. Both paths share `LTETemperatureRecovery`: tolerance 1e-12 for the relative
temperature update and failure after more than 100 updates. The round-trip check
requires `abs(T_recovered - T_input) <= 1e-9 * max(1, abs(T_input))`.
Energy itself need not be positive because EOS reference offsets can differ.
Malformed inverse guesses, nonpositive heat capacity, nonconvergence, and excessive
kinetic-energy cancellation are rejected on the host. Physical-state conversion
uses the forward energy table rather than the runtime pressure-to-energy routine.

Validation targets:

```sh
ctest --test-dir BUILD -R '^(PhysicalState.*Tests|GasPhysicsTestingSuite|StateTestingSuite)$' --output-on-failure
```

`PhysicalStateConversionTests` uses an analytic LTE table without PLATO. It checks
all three pairs in 1D/2D/3D, endpoints, invalid inputs, ambiguity, bounds, inverse
guesses, unchanged output on rejection, and the CPG primitive dispatch regression.
`PhysicalStatePLATOTests` uses a small real-air table and checks round trips and
high-temperature CPG/LTE energy differences. `PhysicalInitialStateIntegration` exercises constant states for both gas models,
all thermodynamic pairs, profiles, rejection and conservative restart in actual
simulations with one/two ranks. `PhysicalBoundaryIntegration` runs axisymmetric
CPG/LTE CNS with constant and radial-profile boundaries, checks uniform-state
preservation and heated inflow response, compares serial/MPI results and legacy
CPG payloads, and exercises coordinated rejection of invalid boundary inputs.

For a device build, use the platform's CUDA-enabled MFEM/PLATO prefix and actual
CUDA architecture (these environment variables must be supplied by the operator):

```sh
cmake -S . -B build-state-cuda -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH="$DEVICE_TPL_PREFIX" \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DTHESEUS_WITH_PLATO=YES -DSUBCELL_FV_BLENDING=ON -DNO_OPT=ON \
  -DPHYSICAL_STATE_TEST_DEVICE=cuda
cmake --build build-state-cuda -j 4
ctest --test-dir build-state-cuda \
  -R '^(PhysicalState.*Tests|PhysicalInitialStateIntegration|GasPhysicsTestingSuite|StateTestingSuite)$' \
  --output-on-failure
```

The conversion unit tests execute on the host even in a CUDA build.
`PhysicalInitialStateIntegration` uses the configured device for the simulation,
including RHS execution after host initialization. Its coverage includes initial
conditions and conservative restart; it excludes boundary-profile conversion.

## Initial-condition configuration

Choose one source under `runTime.conditions.initial_conditions`. A model-neutral
constant state specifies exactly two of pressure, density and temperature, plus
an explicit velocity vector. Composition/scalars and unknown fields are rejected.

```json
{"state": {"pressure": 60000, "temperature": 1200, "velocity": [10, -2]}}
```

For LTE, the resulting state must fit the configured density, temperature and
inverse-energy table ranges. The gas model is constructed once before projection.
On restart, the entire initial-condition selection is unused: it is neither parsed
nor converted, and loaded conservative values are retained.

Two explicitly physical 2D profiles are available. Both are centered at the origin:

```json
{"physical_profile": {
  "type": "thermal-blob", "radius": 0.5, "pressure": 60000,
  "ambient_temperature": 1200, "peak_temperature": 1500
}}
```

The blob has zero velocity and constant pressure. Temperature is
`ambient + (peak - ambient) * exp(-r²/radius²)`; the selected EOS determines density
and conservative energy.

```json
{"physical_profile": {
  "type": "vortex", "radius": 0.5, "speed": 10, "strength": 0.2,
  "density": 0.2, "temperature": 1200,
  "shape_gamma": 1.4, "shape_gas_constant": 287.05
}}
```

The vortex uses the same prescribed spatial shape as legacy `LTEVortexIC`, including
its temperature floor of 20% of the background value. Its independent thermodynamic
pair is density/temperature. `shape_gamma` and `shape_gas_constant` only define the
spatial perturbation; simulated pressure/energy come from the selected gas EOS.
This is a prescribed vortex profile, not a claim of an exact LTE isentropic solution.
The Cartesian vortex is not an axis-regular meridional flow; choose physically
appropriate profiles for axisymmetric runs.

`LTEVortexIC`, `LTEBlobIC`, and other registered conservative functions return
conservative values directly. The `cpg_state` form requires 2D CPG. The explicit
physical forms cannot be combined with legacy `function`, `signature`, `params` or `cpg_state` selection keys.
Constant conversion runs once; spatial conversion/checks run only during projection.
The coefficient is released after projection. Conversion checks execute during
host state construction, outside RHS and timestep kernels. Initialization errors are coordinated across MPI ranks
before boundary/device finalization.

## Initialization responsibilities

`Simulation::LoadConfig` builds spaces and the RHS/gas once, calls
`InitializeSolution`, then prepares boundaries and finalizes the device cache.
`InitializeSolution` loads a conservative checkpoint or obtains one local
coefficient from `MakeInitialCondition` and projects it. All initialization errors
pass through the startup MPI error collective.

- `PhysicalProfiles.hpp`: spatial formulas returning physical state descriptions.
- `PhysicalStateConfig.hpp`: JSON validation and construction of those descriptions.
- `InitialCondition.hpp`: the single host coefficient factory, including the legacy
  conservative-function adapter. It returns a coefficient or throws; no fallback
  dispatch is required in `Simulation`.
- `GasModelInterface::ConservativeFromPhysical`: selected-EOS conversion; table
  ownership belongs to the RHS gas/cache.
- `LTETemperatureRecovery.hpp`: small host/device numerical iteration state shared
  by the checked startup path and runtime LTE EOS. Lookup guards remain in the
  startup adapter; runtime uses table views and reports failure through abort/trap.

Physical profile callbacks and the host gas reference live only for projection.
Device kernels receive the concrete device-ready gas, never these
callbacks or the virtual interface. The shared iteration helper has no allocations,
virtual dispatch, strings, synchronization, or table ownership.


## Physical boundary configuration

Boundary names under `runTime.conditions.boundary_conditions` select mesh attribute
sets. `exterior-state` uses the selected gas model and the same physical-state
syntax as initial conditions:

```json
"Outlet": {
  "type": "exterior-state",
  "state": {"pressure": 60000, "temperature": 1200, "velocity": [0, 0]}
}
```

Specify exactly two of pressure, density and temperature, and one velocity per
spatial dimension. These are prescribed exterior states for the numerical flux,
not characteristic pressure-outlet conditions. Scalars/composition are unsupported.

A stationary 2D, swirl-free radial profile uses pressure plus a file of temperature
and velocity samples:

```json
"Inflow": {
  "type": "radial-profile",
  "file": "inlet.dat",
  "flag": 0,
  "pressure": 60000
}
```

The file begins with `points_per_block number_of_blocks`; each following row is
`radius flag temperature axial_velocity radial_velocity swirl_velocity`. The flag
selects one block and defaults to zero. Swirl is parsed but ignored. Selected radii
must increase strictly, temperature must be positive, and a supplied axis row must
have zero radial velocity. File paths are relative to the process working directory.

At each actual boundary restriction point, the reader linearly interpolates
**temperature and signed velocities**, then converts pressure, temperature and
velocity through the selected EOS. It does not interpolate conservative energy.
Below the first sample, temperature and axial velocity remain constant and radial
velocity decreases linearly to zero at the axis. Radii above the last sample are
rejected, apart from the existing `1e-12` endpoint roundoff allowance. The radius
is the second mesh coordinate; states are not radius-weighted.

For LTE, every evaluated state must fit the configured forward and inverse tables.
Errors identify the boundary and, for sampled profiles, the radius and restriction
point. Conversion failures are coordinated across ranks before device-cache creation.
Only `pressure`, `file` and optional `flag` are accepted with `radial-profile`;
EOS constants come from the selected gas model.

`BoundaryStateConfig.hpp` parses constants and retains profile primitives.
`PackBoundaryPointStates` in `BoundaryStatePacking.hpp` resolves profiles during
RHS finalization after boundary geometry is available. Each profile is reconstructed
once, then sampled at restriction points. The device receives ordinary constant
conservative payloads; conversion, parsing and validation do not run in RHS kernels.
No second gas model or LTE table is constructed.

The legacy `cpg-exterior-state` and `cpg-radial-profile` forms retain their existing
2D CPG restrictions and conversion formulas. Existing conservative inflow vectors,
walls, symmetry, axis and outflow boundaries retain their meanings. To use LTE,
select the physical forms above and physical initial conditions; do not bypass the
legacy CPG checks.

`PhysicalBoundaryIntegration` is registered in the PLATO-enabled axisymmetric
build. Run it with `ctest --test-dir BUILD -R '^PhysicalBoundaryIntegration$'
--output-on-failure`. It honors `PHYSICAL_STATE_TEST_DEVICE`; for CUDA use an
axisymmetric CUDA build with that option set to `cuda`.
