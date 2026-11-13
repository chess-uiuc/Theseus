# Forced ChannelFlow research build

This research variant combines the repaired Helmholtz/LTEEC/wall numerical code
with the original channel initializer and target-state forcing. It does not
require a main merge or a general source-term framework.

## Build and run your existing case

Keep the compiler, CUDA architecture, dependency prefix and run environment from
your working build. Retain these settings from the original channel build:

```text
-DTHESEUS_WITH_PLATO=ON -DSOURCE_TERMS=ON -DSUTHERLAND=ON
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

Forcing is enabled automatically for `StreamwiseParabolicIC` when compiled with
`SOURCE_TERMS=ON`. The log prints `Research channel forcing: enabled`.
`"channel_forcing": false` explicitly disables it for a control run. Other
initializers are unforced unless `"channel_forcing": true` is supplied.
Requesting forcing without the compile option fails rather than silently running
unforced.

## Preserved controller

- Global volume averages are recomputed at every RHS evaluation.
- `tau=0.7*dt_real`, with the actual step size passed before each RK step.
- Mass/momentum relaxation targets remain rho=1 and rhou=515.68342221905391.
- Energy source remains local u_x times the momentum source; the commented
  energy-target relaxation has not been activated.
- Source is added once per element after DG/FV blending. Component reads and
  updates use scalars, without gathered state/source arrays.

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
