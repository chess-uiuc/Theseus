# Forced ChannelFlow research build

This research variant combines the repaired Helmholtz/LTEEC/wall numerical code
with the original channel initializer and target-state forcing. The channel feedback is a consumer of the general source-term framework.
Its controller, initializer and transport/dissipation choices remain research-specific.

## Build and run your existing case

Keep the compiler, CUDA architecture, dependency prefix and run environment from
your working build. Retain these settings from the original channel build:

```text
-DTHESEUS_WITH_PLATO=ON -DSUTHERLAND=ON
-DSUBCELL_FV_BLENDING=OFF
```

`CHANNEL_RESEARCH_COMPATIBILITY=ON` is the default in this research variant.
It preserves Sutherland conductivity `mu*phys.cp/Pr` and Pirozzoli/Lax dissipation
in the solver executable. Core analytic test targets retain their normal models;
separate research tests check the compatibility choices. Set this option OFF
only if you intentionally want tabulated conductivity and Pirozzoli/Roe instead.

PLATO must provide `plato_get_species_entropy` in its header and library. Build
against your installed TPLs; do not copy another machine's paths.

The supplied AIR5 channel JSON can keep:

```json
"gas_model": "lte",
"solver": "LTE_table_rhoT_(air5)",
"numerical_flux": "LTEECFlux"
```

The patch does not create or modify your production `TestCases/LTE/NavierStokes/ChannelFlow`
directory; the bundled small fixture is in `ChannelFlowRegression`.

Keep `StreamwiseParabolicIC` and its parameters, mesh, table, order, timestep,
walls and output controls. The research compatibility path maps this channel's
`lte` selection to Helmholtz for LTEEC or Pirozzoli and accepts the old `solver`
key. Explicit `lte_helmholtz` and `plato_solver` also work. No initializer rename
or replacement with the small unforced test case is required.

Configure the controller explicitly in `runTime`:

```json
"sources": [{
  "type": "channel-feedback",
  "target_density": 1.0,
  "target_streamwise_momentum": 515.68342221905391,
  "relaxation_factor": 0.7
}]
```

The general framework always compiles; no `SOURCE_TERMS` build option is needed.
An explicit `sources` list takes precedence over the legacy channel settings;
`"sources": []` disables all sources. When `sources` is absent, the existing
`StreamwiseParabolicIC` automatic enabling and `channel_forcing` override remain
supported. The enabled log message is `Research channel forcing: enabled`.

## Preserved controller

- Global volume averages are recomputed at every RHS evaluation.
- `tau=0.7*dt_real`, with the actual step size passed before each RK step.
- Mass/momentum relaxation targets remain rho=1 and rhou=515.68342221905391.
- Energy source remains local u_x times the momentum source; the commented
  energy-target relaxation has not been activated.
- Source is added by the general source kernel after fluid RHS assembly and
  before restriction scatter, without changing the conserved state or fluid CFL.
- Persistent buffers reduce element integrals on device. Each RK stage reads
  back three rank-local scalars, followed by one three-value host MPI_Allreduce.
  This synchronization belongs to the global feedback controller.
- Initialization computes global volume from the same physical quadrature weights
  used for the averages. This agrees with geometric volume on the affine channel
  mesh; on curved meshes it uses the solver quadrature rather than a separate
  geometric volume integration.

The initializer function is retained. Its `std::rand()` perturbations can still
vary with decomposition/initialization order; identical initial conservative
fields are needed for meaningful CPU/GPU comparison.

## Focused GPU checks

Build the enabled targets, then inside the existing GPU allocation run:

```bash
ctest --test-dir "$THESEUS_BUILD" \
  -R '^ResearchChannel.*_(CPU|CUDA)$' --output-on-failure

"$PYTHON" "$THESEUS_SOURCE/tests/research_channel_integration_test.py" \
  --source "$THESEUS_SOURCE" --executable "$THESEUS_BUILD/theseus" \
  --database "$TPL_PATH/database" --device cuda
```

The short flow check uses an affordable mesh/table/order, the original initializer,
and both fluxes. It checks forced/unforced behavior and CPU/CUDA agreement.
It does not substitute those test settings into your production case.

The compatibility transport law intentionally differs from core tabulated-
conductivity references. Do not loosen those core expectations to accept a
research model change. Full core regression validation uses its stated model;
research compatibility has its own checks.
