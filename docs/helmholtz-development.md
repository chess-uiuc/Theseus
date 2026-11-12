# Helmholtz LTE developer reference

`LTEHelmholtzTable.hpp` defines a six-plane table with six property planes. `LTEHelmholtzEOS.hpp` evaluates its thermodynamics and
the symmetric discrete gradients used by LTEEC. For runtime configuration and transport choices, see [Using Helmholtz LTE](helmholtz-eos.md).

## Storage and ownership

`Theseus::HelmholtzTable::Data` reuses the LTE owner vectors. Its forward table
has exactly six property-major planes: mass-specific Helmholtz free energy H,
physical derivatives H_rho, H_T, H_rhoT, dynamic viscosity and thermal conductivity.
The flat index is `property*nx*ny + temperature_index*nx + density_index`.
Density is contiguous. Density and temperature grids must be finite, positive
and strictly increasing; the inverse energy grid must be finite and strictly
increasing. Derivatives refer to physical coordinates even on logarithmic grids.

The inverse table has `nx*ny` entries: initial temperature guesses on the
rho/energy grid. The energy grid has `ny` entries, matching the owner dimensions.

Use `HostView(data, layout)` for host queries and `DeviceView(data, layout)`
during host device-cache construction. Both validate dimensions and grids;
`DeviceView` rebinds every pointer using MFEM's `Read()`. Retain the owner until
all kernels finish and recreate views after resizing any owner vector. View
construction and validation belong outside point kernels.

## Querying a state

```cpp
namespace HT = Theseus::HelmholtzTable;
namespace HE = Theseus::HelmholtzEOS;

const auto tables = HT::HostView(data, HT::Layout{nx, ny});
const auto state = HE::At(tables, density, temperature);
if (!state.valid)
  {
    throw std::invalid_argument("State outside the Helmholtz table");
  }
const auto pressure = state.pressure();
const auto energy = state.energy();
```

`At` selects one cell for all subsequent queries. Both grid endpoints are valid;
nonfinite or out-of-domain coordinates return an invalid evaluation before any
cell dereference. Check `valid` before querying properties. This lookup does not
clamp or extrapolate. Table contents must represent the intended physical model;
a valid coordinate alone does not establish thermodynamic stability.

The common Hermite interpolant supplies

- p=rho^2 H_rho, e=H-T H_T, s=-H_T, g=H+rho H_rho;
- cv=-T H_TT and cp=cv+T rho^2 H_rhoT^2/p_rho;
- c^2=p_rho-rho^2 H_rhoT^2/H_TT, gamma=cp/cv and R=p/(rho T).

Pressure and Gibbs coordinate derivatives are available directly. The queries
read the four corner planes without materializing a coefficient tensor or
calling pressure recursively from Gibbs. `transport<Layout::mu_idx>(tables)`
and `transport<Layout::lambda_idx>(tables)` use bilinear interpolation of the
two transport planes. They do not apply a Sutherland or Prandtl-number model.

## Temperature recovery

`TemperatureFromEnergy(tables, density, specific_internal_energy)` returns a
`Recovery` containing temperature, iteration count and status. Check for
`RecoveryStatus::converged` before using the temperature. Other statuses identify
an out-of-domain coordinate/guess/step, invalid energy or nonpositive/nonfinite
cv, or the 100-iteration limit.

A bilinear inverse-table guess starts Newton iteration on e=H-T H_T, with the
same interpolant's cv as derivative. The step tolerance is
`1e-9 + 1e-12*temperature`. Every forward evaluation is preceded by a domain
check. Only a correction that already satisfies this tolerance may be projected
to the nearest temperature endpoint if it leaves the domain. A larger departing
step fails explicitly. This handles converged boundary states without evaluating
an invalid cell. A rectangular inverse energy grid does not guarantee that every
rho/energy combination has a forward root.

## Discrete thermodynamic means

`DiscreteGradient<Quantity::pressure>` and `DiscreteGradient<Quantity::gibbs>`
return the symmetric Itoh–Abe gradient of f/T in (rho,beta=1/T).
`ThermodynamicMeans(tables, left, right)` combines them as
`rho_bar=D_rho(p/T)/D_rho(g/T)` and
`e_bar=D_beta(g/T)-D_beta(p/T)/rho_bar`.

Both endpoint evaluations must be valid and come from the supplied table.
Endpoint-averaged physical derivatives replace quotients when the density or
beta difference is smaller than `2.22e-15 + 1.49e-8*max(coordinate1,coordinate2)`.
Unneeded mixed-state evaluations are omitted when both coordinates use this
limit. No callback, derivative output array, or temperature recovery is nested
inside these routines: callers recover each endpoint temperature once.

Endpoints are ordered by density, then temperature, before evaluating the
symmetric gradient. This keeps subtraction order identical for reversed face
orientations. It preserves the mathematical gradient. Same-cell finite differences use the
stable evaluation described below. Across-cell quotients subtract endpoint values
and can be sensitive to cancellation.

## Gas interface and PLATO generation

`LTEHelmholtzGas` in `LTEGasModel.hpp` binds the Helmholtz EOS, table view and
LTE transport to the standard conservative-state gas interface. For host construction, initialize PLATO for the desired mixture and solver,
provide density and temperature grids, and build the table:

```cpp
#include "LTEGasModel.hpp"
#include "LTEHelmholtzTableBuilder.hpp"

namespace HT = Theseus::HelmholtzTable;
HT::Data data;
Theseus::LTETable::uniform_grid(17, 0.8, 1.2, data.rho_grid);
Theseus::LTETable::uniform_grid(41, 250, 350, data.T_grid);

plato_initialize("LTE_table_rhoT_(air5)", "air5", "empty", "empty", database_path);
const auto tables = HT::Build(data);
plato_finalize();

Theseus::LTEHelmholtzGas gas(physics, layout, tables);
```

CMake supplies `USE_PLATO` and `THESEUS_PLATO_HAS_ENTROPY`, the generated Theseus
configuration header and PLATO linkage when its compile/link capability probe
passes. PLATO must provide `plato_get_species_entropy` in both its
header and library. CMake tests that capability before enabling the real-air test;
analytic tests remain available without PLATO. The builder computes H=e-Ts, H_rho=p/rho^2, H_T=-s and
H_rhoT=alpha/(rho^2*beta_T), and tabulates viscosity and conductivity.

`Build` allocates exactly six forward planes. Its inverse grid covers PLATO and
represented forward nodal energies, with an additional allowance of
`64*epsilon*max(|H|+|T H_T|)` for floating-point cancellation. Each inverse guess
is solved against the forward interpolant and bounded to its temperature domain.
Energy-grid entries outside an individual density column's forward energy range
store the closest temperature endpoint as a guess. Runtime recovery checks
the actual forward domain and convergence; this construction does not create a
forward state outside the supplied rho/T grid.

Use `gas.ConservativeFromPhysical(input, output)` for all three physical input
pairs. Density/temperature input directly queries the interpolant. Pressure input
reconstructs each cell's pressure polynomial (quartic in density, cubic in
temperature), isolates roots using derivative critical points, and requires one
unique root over the table. It detects interior root pairs, tangencies and
constant root intervals rather than relying on endpoint signs alone. Conservative
packing is followed by a runtime energy-to-temperature round-trip check.

Runtime queries and entropy/primitive-gradient conversion use concrete device
pointers from `gas.to_device(host_data)`. The owner is
`host_data.lteTableData`; preserve it throughout kernel execution. The inverse
bilinear interpolation uses difference form, preserving constant guesses exactly.
`EvaluateProperties` recovers temperature once for the requested properties.
Sutherland mode changes viscosity; conductivity remains tabulated, using the conductivity table plane.

Bicubic Hermite interpolation is C1 across cells. Its second derivatives, and
therefore cv/cp/acoustics, can have different one-sided values at a node. Gas-query
consistency checks use the actual recovered coordinates; independent PLATO
comparisons separately measure table approximation accuracy.

## LTEEC volume and face flux

`LTEECFlux.hpp` recovers temperature once at each conservative endpoint and shares
its Hermite cell across pressure, sound speed and discrete thermodynamic means.
It computes normal flux components directly, without a full inviscid flux tensor,
Hermite coefficient block, dissipation array, derivative output array or nested
thermodynamic callback. All thermodynamic work finishes before writing output.

For mean velocity u_bar, the central mass flux is rho_bar*(u_bar dot n).
Momentum adds mean(p)*n, and energy uses e_bar+0.5*(u1 dot u2), with cross pressure
work 0.5*(p1*u2+p2*u1) dot n. The volume direction is the arithmetic mean of the
two metric rows. The face flux additionally subtracts 0.5*lambda*(U2-U1), with
lambda=max(|u1 dot n|+c1*|n|, |u2 dot n|+c2*|n|). Nonunit normal magnitude enters
once, and a zero normal produces zero flux. The configured LTE state has no
passive scalar equations.

Helmholtz construction requires PLATO's entropy API. Construction failures on
MPI rank zero are broadcast before other ranks enter table-data broadcasts.

## Stable same-cell discrete gradients

When both endpoint evaluations share one Hermite cell, finite quotients use
analytic polynomial divided differences instead of subtracting completed
pressure/Gibbs values. The identities are

- D_rho(p)=mean(rho^2)*D_rho(H_rho)+(rho1+rho2)*mean(H_rho).
- D_rho(g)=D_rho(H)+mean(rho)*D_rho(H_rho)+mean(H_rho).
- D_beta(f/T)=mean(f)-mean(T)*D_T(f), for beta=1/T.

These evaluate the same discrete gradient of the same interpolant. Near-equal coordinates use the derivative limits described above. Across-cell
queries evaluate finite quotients. Canonical endpoint ordering gives reversed
faces identical arithmetic ordering.

The [test inventory](verification.md#helmholtz-polynomial-helper) describes
analytic and high-precision reference coverage. The
[interpolation interface](helmholtz-interpolation.md) documents basis evaluation
and physical divided differences.

### Shared Lax dissipation

`DissipationOperator.hpp` provides `Lax_dissipation_coefficient(un_left,
un_right, c_left, c_right, normal_magnitude)`, the half maximum normal wave
speed. `un` is velocity dotted with the unnormalized normal, so the acoustic
term includes its magnitude once. `Lax_dissipation(coefficient, left, right)`
returns one conservative component of dissipation. LTEEC uses these scalar
interfaces with its already recovered endpoint thermodynamics.

The gas/state-view overload `Lax_dissipation(gas, left, right, normal, out)`
writes the mass, active momentum and energy components into caller storage. It
borrows the state views without constructing additional conservative arrays.
The caller supplies output storage distinct from the input states. This interface
allows other fluxes to select Lax dissipation without duplicating its formula;
flux selection determines which dissipation operator is used.
