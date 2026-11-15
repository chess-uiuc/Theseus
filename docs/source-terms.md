# Explicit physical source terms

Configure additive conservative rates in `runTime.sources`. An absent field or
empty array installs no source system. During timestepping that path only checks
a null pointer: no source kernel, source buffer allocation, transfer, reduction,
MPI call or source preparation runs. No source data is added to the physics device
cache.

Configured sources each compute and add their contribution in a device kernel.
They run after fluid volume assembly and DG/FV blending, before the volume RHS is
scattered back. The same interface serves Euler and Navier–Stokes, point and
element volume assembly. Existing axisymmetric geometric sources remain separate.
Rates are per physical volume per unit time, with no extra quadrature weight or
Jacobian factor. All sources read the same current RK-stage state.

## Configured sources

An `affine` source adds, per conservative equation,

`constant + time * stage_time + state * U + gradient dot position`.

Each coefficient array contains one value per conservative equation: density,
momentum components, total energy density, then supported scalar components.
Omitted arrays are zero. `gradient` contains one coefficient array per dimension.
For example, in two dimensions:

```json
"sources": [
  {
    "type": "affine",
    "state": [0, 0, 0, 1000],
    "time": [0, 0, 0, 1000000]
  }
]
```

This adds `1000*rhoE + 1e6*t` to the total-energy derivative. Coefficients must be
finite and arrays must match the equation count. Coordinates are packed once at
initialization, only if a source requests them. Multiple entries add independently.

A `linear-drag` source removes momentum at rate `-coefficient * momentum_density`
and applies the local work `velocity dot momentum_source` to total energy:

```json
"sources": [{"type": "linear-drag", "coefficient": 1000}]
```

The nonnegative coefficient has units of inverse time. Density and passive scalars
receive no drag contribution. The fluid CFL estimate does not include source
stiffness; select a stable explicit timestep for the source.

`periodic-euler-mms` supplies the analytic source for the
[periodic CPG Euler manufactured case](../TestCases/Euler/2D/PeriodicMMS/README.md).
Its source implementation validates Cartesian 2D CPG Euler, gamma=1.4 and four
equations. The manufactured solution varies density, both velocities and pressure;
the case is a bounded inviscid verification, not viscous, boundary or LTE accuracy
verification.

## Source ownership and execution

`SourceTerms.hpp` defines the generic source lifecycle and `PointSource<Model>`
adapter. Neither that adapter, the RHS nor its cache names any source model.
`SourceRegistry.hpp` maps configuration names to host factories. The application
composition point is `src/SourceConfiguration.cpp`; individual implementations
and validation live in `include/sources/`.

A host `Sources::System` owns its source objects. The RHS receives it through
`SetSources` before `Finalize`. Its generic `AddStage(stage, rhs)` call applies
sources. Each source has:

- `Needs()`: declare whether physical coordinates are required.
- `Initialize(MeshView)`: validate layout and allocate any persistent storage.
- `Prepare(Stage)`: optional preparation before that source is evaluated.
- `Add(Stage, rhs)`: launch the concrete source evaluator.

A normal point source uses `PointSource<Model>`. Its model supplies
`MakeEvaluator(stage)`, returning a concrete device-ready evaluator whose call is:

```cpp
MFEM_HOST_DEVICE
void operator()(const Point &point, const SourceAccumulator &output) const;
```

The point provides conservative state, layout, requested coordinates, physical
stage time and the distinct outer step size. The accumulator adds physical rates
to the proper conservative equation/point. Domain attribute markers are honored.
The adapter owns the host model and captures only its concrete evaluator and
`DeviceMesh` data. Host virtual interfaces, factories, owners and the MPI
communicator never enter a kernel. The adapter uses one point kernel per source;
there is no intermediate per-point source-rate array.

To add a source, implement its model and registration function, register it in
`BuiltinRegistry()` in `src/SourceConfiguration.cpp`, and add behavior tests.
`Affine.hpp`, `LinearDrag.hpp` and `EulerMMS.hpp` are complete examples. No RHS or
cache modification is required. Use the existing build targets and include paths.
The registry translation unit is compiled with CUDA in CUDA builds, so factories
instantiate the same device adapter for every source.

## Optional preparation

Analytic models inherit `AnalyticModel`, whose initialization/preparation methods
are no-ops. A source that needs preparation implements those methods in its own
model. `Initialize` receives element count, DOFs, dimension, equation layout,
device quadrature weights, domain markers and the mesh communicator. `Prepare`
receives device stage state, stage time and actual outer step size after event
limiting. The step size is held fixed across its RK stages.

Allocate scratch during initialization and keep it alive through source application.
Preparation can perform required device reductions or MPI communication; that cost
belongs to that source. Ordinary analytic sources require no such operations.
`MakeEvaluator` binds scalar values or device pointers into the prepared data;
never capture host tables, virtual gas interfaces or the owner itself. EOS queries
must use the concrete device-ready gas view with its table ownership preserved.
Attach sources before finalization and reinitialize after a discretization change.

## Verification

CTest registers `SourceTerms_CPU` and `SourceTerms_MPI_CPU`; CUDA builds add
`SourceTerms_CUDA` and `SourceTerms_MPI_CUDA`. Coverage includes independent source
composition, dimensions/scalars, domain masks, guards/input preservation, factory
validation, lifecycle, persistent preparation and empty systems called with null
arrays. Missing/empty configuration returns no source system. Analytic Euler
forcing is checked against independent finite differences of state/physical fluxes.
Host verification explicitly synchronizes both compared input vectors and rejects
nonfinite expected values.

The central suite registers **Manufactured source evolution** and **Periodic CPG
Euler MMS spatial convergence**. The first checks Euler/NS RK3 temporal convergence,
physical coordinate packing, uniform drag and its energy work against an analytic
solution, MPI and selected CPU/CUDA fields. The second checks P2 spatial convergence,
independent conservative output fields, timestep isolation and a source-off control.
Source field parity uses `rtol=atol=1e-10`; existing golden expectations are unchanged.

Inside an existing CUDA allocation, build all targets and run:

```bash
cmake --build "$THESEUS_BUILD" -j 4
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run \
  --source "$THESEUS_SOURCE" --build "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cuda \
  --results "$TEST_RESULTS"
```

Allocate at least two MPI ranks/devices. Keep the existing compiler, TPL and launcher
settings; the shared harness owns placement. `TEST_RESULTS` must be a new directory.
See [test runner](test-runner.md) for full CPU and prepared-build procedures.
