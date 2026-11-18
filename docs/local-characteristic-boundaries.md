# Local characteristic open boundaries

These reusable single-gas boundaries support CPG and LTE, CPU and CUDA assembly.
Their local acoustic algebra and frozen-property total-inlet model follow the
inspected HEGEL implementation. They retain Theseus's selected numerical Riemann
flux. They are not exact nonlinear characteristic or nonreflecting conditions.
No boundary ODE integration or runtime thermodynamic-consistency audit is used.

## Configuration

```json
{
  "feed": {
    "type": "characteristic-inlet",
    "state": {"pressure": 10000, "temperature": 600, "velocity": [80, 0]}
  },
  "exit": {
    "type": "characteristic-pressure-outlet",
    "state": {"pressure": 10000, "temperature": 300, "velocity": [0, 0]}
  },
  "reservoir": {
    "type": "characteristic-total-inlet",
    "total_pressure": 10000,
    "total_temperature": 300,
    "direction": [0, -1]
  }
}
```

Use the existing physical-state contract for `state`: exactly two of pressure,
temperature and density, plus a dimension-sized velocity vector. A pressure
outlet's reference pressure is imposed; reference temperature and tangential
velocity provide incoming material on backflow. Its reference normal velocity
does not prescribe backflow speed. `characteristic-radial-inlet` replaces `state`
with `file`, `flag` and `pressure`, using the existing stationary radial T/ux/ur
format. The radial adapter is two-dimensional and swirl-free. Constant inputs
support dimensions1–3; extra transported scalars/composition are unsupported.
Validation of prescribed data occurs on the host during parsing/packing.

## Local acoustic construction

Let n be the unit outward normal, q=u·n, i the interior, a the prescribed
reference and Z_i=rho_i*c_i. Normal vectors are normalized for state construction;
the mapped normal is retained for numerical-flux scaling.

For inward/resting interior flow at a reference inlet:

p_b=(p_a+p_i+Z_i(q_i-q_a))/2,
q_b=q_i−(p_b−p_i)/Z_i.

Incoming material uses rho_b=rho_a+(p_b−p_a)/c_i² and reference tangential
velocity. If the resulting flow is outgoing, use interior material with
rho_b=rho_i+(p_b−p_i)/c_i² and interior tangential velocity.

When the interior at a nominal inlet is outward, use the pressure-outlet branch
at p_a instead of forcing the inward profile. For an outward branch or a
pressure-outlet boundary:

p_b=p_a, q_b=q_i−(p_a−p_i)/Z_i.

Outgoing density/tangential velocity follow the interior correction above.
Incoming pressure-boundary material uses the reference static temperature and
tangential velocity; density comes from EOS(p_b,T_a). Actual normal velocity is
always q_b. Zero target normal speed selects interior material. This selection
is local at every boundary point; different parts can flow in opposite directions.

For rho,p targets, T is obtained from the selected EOS. The inlet profile's T/u
are reference data, not exactly prescribed final boundary T/u. All conserved
energies use actual e(rho,T)+|u|²/2, including the LTE energy reference.

## Frozen-property total inlet

For inward/resting interior flow, use local EOS R_i,gamma_i,c_i,T_i and
H_i=gamma_i R_i T_i/(gamma_i−1)+|u_i|²/2. Set

J=−q_i−2c_i/(gamma_i−1),
a=1+2/(gamma_i−1), b=2J,
k=(gamma_i−1)(J²/2−H_i).

Choose the larger quadratic root c_b of a*c_b²+b*c_b+k=0, then
V_b=2c_b/(gamma_i−1)+J and f=1+(gamma_i−1)V_b²/(2c_b²).
Use p_b=p0*f^(−gamma_i/(gamma_i−1)), T_b=T0/f, u_b=V_b*d.
Density and energy use the actual selected EOS at p_b,T_b.
Outward interior flow uses the pressure-outlet branch at p0, retaining outgoing
material. An outgoing target produced by the total formula also retains interior
material/tangential velocity, a deliberate reversal adaptation beyond plain HEGEL. The direction vector is normalized on the host.

**This is a frozen-property perfect-gas approximation even with an LTE EOS.**
It does not enforce a general equilibrium-EOS isentropic reservoir. Local gamma
and R do not remove that approximation. The quadratic follows the inspected
HEGEL model, including its use of interior H and its lack of direction projection
in the quadratic; general oblique-flow acoustic compatibility is not guaranteed.

## EOS conversions and computational work

CPG conversions are algebraic. LTE conversions restrict the existing forward
bilinear pressure table to fixed density or fixed temperature, bracket using
binary search, and interpolate its scalar piecewise-linear root directly. This
assumes stable monotone pressure along each table coordinate. No secant/Newton
or acoustic-path iteration is introduced. An out-of-table query/root propagates
NaN to normal solver diagnostics; it is not clamped into the table. The existing
table owner supplies concrete device pointers; no host callback enters a kernel.

The shared state constructor has a CUDA noinline boundary to avoid duplicating
its body in flux, gradient and CFL callers; CPU compilation uses ordinary inline
semantics. This is a structural measure, not evidence of CUDA compile/runtime
performance. Measure the target compiler/device before accepting its cost.

## DG flux and viscous treatment

The physical target is supplied as the exterior state to the selected numerical
flux. HEGEL's reflected FV ghost is not used. This is weak enforcement: neither
interior traces nor numerical mass flux are overwritten. Across thermal/density
jumps, the numerical mass-flux sign can differ from target q_b; validate the
composed flux, not merely the target velocity.

Incoming reference/total branches impose target static T and velocity weakly in
entropy-gradient lifting, retaining the mass entropy component. Pressure-branch backflow (including a nominal inlet whose outward interior
pressure branch reverses) imposes T and tangential velocity, retaining interior normal velocity for the
diffusive trace. Outgoing lifting is zero. Constitutive stress/heat flux uses the
assembled primitive gradients, with inviscid-minus-viscous face flux. The same
state constructor supplies inviscid, gradient and stability boundary states.

## Differences from HEGEL and limitations of correctness

| Difference from inspected HEGEL | Reason / consequence |
| --- | --- |
| Local outgoing pressure/material branch at a nominal inlet | Required recirculation support; differs from plain SUB_IN, resembles explicit mixed-boundary selection |
| Incoming material on pressure backflow is explicit | Required data-count closure; not plain HEGEL SUB_OUT extrapolated material |
| Physical target enters Theseus Riemann flux, not reflected FV ghost | DG versus FV reconstruction/stencil; discrete parity is not claimed |
| BR1 entropy lifting, not ghost-cell viscous gradients | Same intended primitive constraints, different discretization |
| Supersonic outward interior extrapolates | No incoming inviscid characteristics; plain HEGEL subsonic formulas have no automatic switch |
| Independent profile adapter, table provider, no swirl | Profile reader and static-temperature use confirmed; complete fork/corner treatment and whole-case parity not established |

The separate physical-accuracy limitations are: frozen-property total/static
relations; locally linear rather than finite-amplitude acoustic corrections;
possible inadmissible states under large pressure changes; discontinuous material
selection at reversal; no shock-exact/choking construction. Supersonic incoming
flow and a subsonic-to-sonic target are not supplied with an additional transition
model: the local formulas are evaluated without a special abort or clipping
layer. Successful evaluation is not evidence of physically correct all-Mach
transitions. This implementation is intended for subsonic open boundaries and
outward supersonic extrapolation; test operation within that scope.

## Run PXChamber directly

Use `config-lte-local-characteristic.json` for LTE or
`config-local-characteristic.json` for CPG in
`TestCases/Axisymmetric/NavierStokes/PXChamber`. Existing configs remain legacy.
Copy the selected JSON and set absolute `runTime.mesh_file`,
`runTime.conditions.boundary_conditions.Inflow.file`, and (for LTE)
`runTime.database_path`. `px_chamber_coarse.msh` is an explicit supplied test mesh;
the original `px_chamber_axi.msh` remains the full mesh. Keep output_file_path
as `"."` for harness-managed output. Set timestep/final time/output cadence in JSON.

```bash
scripts/run_theseus.sh -e /absolute/path/to/build-axis/theseus \
  -c /absolute/path/to/PXChamber/config-lte-local-characteristic.json \
  -o /absolute/path/to/new-run -p 4 -r cuda -P
```

Use `-r cpu` for CPU execution. All case settings are in JSON; no case-specific
Python launcher is required. The Top uses the approximate total inlet; Inflow
uses the stationary reference profile at10kPa; Right uses10kPa reference pressure
and300K incoming material. These are case inputs, not solver constants.

## Verification

`LocalCharacteristic_CPU` and CUDA-build `_CUDA` exercise analytic local equations
and an independently reduced CPG limit of the total quadratic; CPG dimensions1–3,
manufactured LTE, rotated/scaled normals, incoming/outgoing states, LTE energy
offset/conversions, signed LLF mass flux and entropy/viscous assembly.

The central `local-characteristic` recipe includes uniform reference/radial/total
and reverse channels, Euler/CNS, one/two ranks, plus real-air11 heated and reverse
chamber evolution through1microsecond at dt=1e-8, with finite positive fields,
thermal/flow response and complete coordinate-grouped rank comparisons. Logs,
fields, measurements JSON and inlet CSVs are retained. These tests do not establish
whole-chamber validation, developed recirculation, all-Mach behavior, measured wave
reflection or CUDA cost/parity until those measurements are separately collected.

## Implementation map and exercising the tests

| Source | Responsibility |
| --- | --- |
| `include/LocalCharacteristicBoundary.hpp` | Shared local face-state algebra and material-direction choice |
| `include/EOS.hpp`, `GasModel.hpp`, `LTEGasModel.hpp` | Concrete CPG/LTE thermodynamic conversions; direct LTE table roots |
| `include/BoundaryStateConfig.hpp`, `BoundaryStatePacking.hpp` | Host validation and existing physical/profile payload packing |
| `include/bc_kernels.hpp` | Riemann exterior, incoming/outgoing lifting, constitutive viscous face flux |
| `include/NSOperator_impl.hpp`, `DGSEMIntegrator.hpp`, `dgsem_cache.hpp` | Same-RHS conservative trace before entropy-buffer reuse |
| `include/RHSOperator_impl.hpp` | Constructed acoustic/diffusive boundary stability states |

The unit fixtures include gamma1.4 and5/3, a calorically imperfect table with
nonconstant cv/gamma, independent nonzero shear/conduction, and pressure-driven
reversal requiring a natural normal-velocity diffusive trace. The solver recipe
also exercises variable-CFL Euler/CNS paths. Its heated startup requires resolved
incoming and outgoing signed interior traces on the nominal inlet; that is not
proof of developed physical recirculation.

```bash
cmake --build "$BUILD_AXIS" --target theseus local_characteristic_tests -j 2
ctest --test-dir "$BUILD_AXIS" -R '^LocalCharacteristic_CUDA$' --output-on-failure
bash tests/recipes/integration/local-characteristic.sh \
  --source "$PWD" --build-axisymmetric "$BUILD_AXIS" --prefix "$TPL_PREFIX" \
  --output "$RESULTS" --python "$PYTHON" --device cuda
```

Use the CPU CTest name and `--device cpu` for CPU builds. Run CUDA checks in the
appropriate allocation. The recipe exposes its actual `run_theseus.sh` commands;
its Python stages generate/check test inputs and fields. Test registration alone
is not a passing result. The full central runner discovers this recipe and CTest
entries; no separate CI-only path is used.

`ThermoTableTestingSuite` additionally includes `plato_LocalCharacteristic_reference`: actual
air11 cold/profile thermal jumps at three radii, independent host pressure inversion,
matching-state preservation and an independent LLF mass-flux calculation. It records
constructed normal velocity and signed numerical mass flux, alongside the legacy
exterior-state flux. The cold/hot transient can produce outward LLF mass diffusion
even with an inward target; this test measures that discrepancy rather than treating
a target-velocity check as proof of incoming numerical mass flux.

## EOS backend support

The local characteristic families support CPG and the original `LTE` property-table
backend. `LTE_Helmholtz` configurations may use their existing boundary treatments;
combining that backend with a local characteristic family is rejected during host
configuration. Its pressure inversion and entropy trace require separate support.
The selected backend is never silently changed. The total inlet retains the
frozen-property approximation described above for original LTE.
