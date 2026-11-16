# Repeating MMS evaluations

The [MMS verification guide](mms-verification.md) defines the exact solutions,
physical fluxes, forcing and archived results. This page groups the commands,
testing matrices and acceptance criteria needed to reproduce them.

For the complete GPU campaign, start with the [HPC batch recipe](mms-hpc.md):
two builds, a 90-task Slurm array, pilot jobs, submission commands and retry rules.

## Prepare a build and choose a workflow

**Prerequisites and artifacts**

- Absolute paths to a source checkout, matching dependency prefix, existing build and Python interpreter.
- Python packages from [requirements-ci.txt](../ci-resources/requirements-ci.txt).
- A constant-transport CPG build for current CNS studies; configure FV blending off for smooth DG accuracy studies.
- A fresh result directory per run, containing inputs, solver logs, fields, measurements, assessments and plots.

Set paths for your installation; the following placeholders must be replaced:

```bash
export THESEUS_SOURCE=/absolute/path/to/Theseus
export THESEUS_BUILD=/absolute/path/to/optimized-build
export TPL_PATH=/absolute/path/to/tpl/install
export PYTHON=/absolute/path/to/python-environment/bin/python
export RESULTS_ROOT=/absolute/path/to/ValidationRuns/mms
```

The recipe commands below use supplied executables without configuring or
compiling. If needed, create an optimized CPU accuracy build first, using your
MPI compiler environment and the installed dependency prefix:

```bash
CC=mpicc CXX=mpicxx cmake -S "$THESEUS_SOURCE" -B "$THESEUS_BUILD" \
  -DCMAKE_PREFIX_PATH="$TPL_PATH" -DPython3_EXECUTABLE="$PYTHON" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DTHESEUS_WITH_PLATO=ON -DSUBCELL_FV_BLENDING=OFF
cmake --build "$THESEUS_BUILD" --target theseus -j 4
```

For a GPU run, use an existing CUDA build and replace `--device cpu` with
`--device cuda` inside a GPU allocation. The shared
[run_theseus.sh](../scripts/run_theseus.sh) harness owns scheduler selection,
MPI launch and device placement. Use the site's normal launcher/device visibility;
workstation localhost overrides do not belong in portable recipes. Exceptional
sites may set `THESEUS_MPI_LAUNCHER` at the top level. Arrange GPU visibility per
local rank and execute on the target system; CPU launch/mock-device checks do
not establish GPU numerical accuracy.

| Level | Existing-build entry points | Purpose |
| --- | --- | --- |
| Quick | `euler-build-verification`, `cns-build-verification` | Bounded order checks using already-qualified settings |
| Qualification | `euler-mms-qualification`, `cns-mms-qualification` | Accuracy plus source/timestep/spatial-floor controls |
| Extended studies | Individual `mms-verification` recipes | Degree/flux surveys, large GPU matrix and diagnostics |
| Broad code suite | `start --suite full` or `run --suite full` | Fresh Cartesian/axisymmetric builds and registered code tests; MMS level selected separately |

Multiple ranks are an execution choice for speed. This MMS infrastructure adds
no rank-count comparisons; existing MPI regressions elsewhere are unchanged.
`show <recipe-or-suite-id>` displays actual Bash commands, and `list` lists
registered recipes. Preparation, simulation and comparison/plotting are separate
recipe stages; see [test-runner.md](test-runner.md) for staged execution.

## Quick verification of an existing build

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test euler-build-verification \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --ranks 2 --fluxes all --results "$RESULTS_ROOT/euler-quick-001"

"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test cns-build-verification \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --ranks 2 --results "$RESULTS_ROOT/cns-quick-001"
```

| Recipe | Matrix and steps per case | Required result |
| --- | --- | --- |
| `mms-euler-flux-quick` | P2, 2D, 6/12/24 cells per direction, six production fluxes by default; RK4, $t_f=0.2$, $\Delta t=0.004$, 50 steps | Finest component physical $L_2$ orders $\ge2.5$ |
| `mms-euler-temporal-quick` | RK3SSP/RK4, P15 on $6\times6$ cells; $\sigma=126$, $t_f=0.02$, $\Delta t=0.001/0.0005/0.00025$, 20/40/80 steps | Both component refinement orders within 0.15 of nominal 3/4 |
| `mms-cns-quick` | P2/Roe, 2D, 6/12/24 cells; $\mu=0.02$, $\mathrm{Pr}=0.72$, $R=1$; RK4, $t_f=0.2$, $\Delta t=0.001$, 200 steps | Finest density/vector-momentum/energy physical $L_2$ orders $\ge2.5$ |

The Euler suite has 18 spatial plus six temporal cases at the default six fluxes;
the CNS suite has three cases. CNS took about 6.6 seconds including analysis and
plotting on an optimized two-rank CPU build. Initial values, finite positive
density/pressure, actual final times and independent physical errors are checked.
Component data remain available even when plots emphasize vector momentum.

Quick runs omit qualification controls, relying on the stated settings having
qualified temporal/spatial error separation. They do not claim fresh qualification.
Requalify after changes that invalidate that separation; short runs do not
establish untested dimensions, degrees, fluxes, transport or boundary behavior.

**Major points**

- Small developer workflows against existing builds; no rebuild and no additional MPI-parity cases.
- The same recipes appear in broad routine/PR testing, alongside unrelated code tests.
- Keep `measurements.json`, assessments and expected-slope plots with the run.

## Qualification and its controls

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test euler-mms-qualification \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --ranks 2 --fluxes all --results "$RESULTS_ROOT/euler-qualified-001"

"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test cns-mms-qualification \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --ranks 2 --results "$RESULTS_ROOT/cns-qualified-001"
```

Run one row independently by replacing the suite ID with its recipe ID. The
broad runner selects these recipes once each at `--mms-level qualification`,
without also running their quick subsets. Euler qualification uses
$[-1,1]^2$ and $[0,1]^3$; CNS uses the same respective domains.

| Recipe | Matrix | Controls and gates |
| --- | --- | --- |
| `mms-euler` | P2/LLF, 2D/3D, 6/12/24 cells; $t_f=0.2$, RK4, $\Delta t=0.004$ | Finest half-$\Delta t$ and source-off in each dimension; aggregate spatial rates and final component rates in 2.5–3.5; half-step aggregate change $<1\%$; source-off error $>5$ times forced error |
| `mms-euler-orientation` | One 3D P2 solution under three matched cyclic orientations | Mapped fields `rtol=atol=1e-10`; component $L_2$ norms `rtol=1e-9`, `atol=1e-12` |
| `mms-euler-temporal-qualified` | Six quick Euler time cases plus two P17 finest-step controls | Same order gate as quick; P15/P17 finest component errors agree within $1\%$ |
| `mms-euler-flux-qualified` | Quick selected-flux spatial cases plus one finest half-step case per flux (24 cases for six fluxes) | Same component order lower bound; every component's half-step change $<1\%$ |
| `mms-cns-qualification` | Seven 2D P2/Roe cases, 6/12/24 cells; $\mu=0.02$, $\mathrm{Pr}=0.72$, $R=1$, $t_f=0.2$, $\Delta t=0.001$ | Finest half-step, omitted-viscous-forcing, zero-transport CNS and Euler; finest headline orders $\ge2.5$; half-step change $<1\%$; omission increases momentum/energy errors $\ge5$ times; zero-transport fields agree at `rtol=atol=1e-10` |
| `mms-cns-temporal` | Eight cases, P19 on $3\times3$ cells, two P21 spatial controls; $\mu=0.02$, $\mathrm{Pr}=0.72$, $R=1$, $\sigma=500$, $t_f=0.01$ | RK3SSP: $\Delta t=6.25\times10^{-5}/3.125\times10^{-5}/1.5625\times10^{-5}$, 160/320/640 steps. RK4: $2.5\times10^{-4}/1.25\times10^{-4}/6.25\times10^{-5}$, 40/80/160 steps. Both component refinement orders within 0.15 of nominal; spatial-control changes $<1\%$ |
| `mms-cns-3d-qualification` | Five 3D P2/Roe cases, 6/12/24 cells, same CNS transport/time settings as 2D | Finest half-step and omitted-viscous-forcing controls; same order, timestep and omission gates as 2D. Does not repeat the zero-transport comparison |

The Euler `mms-euler` CUDA recipe additionally compares finest CPU/CUDA fields
from the same binary at `rtol=atol=1e-10`. This adds device coverage, not a
rank-count comparison. Other recipe/device coverage is stated by its own checker;
using `--device cuda` does not imply that every recipe runs a CPU comparison.

CNS half-step cases use $\Delta t=0.0005$ (400 steps). The three main 3D CNS
meshes took about 4.3 minutes on two optimized CPU ranks; finest half-step and
omission controls add substantial work. These and high-degree temporal controls
are qualification work, not per-edit smoke tests.

**Artifact inventory**

- Euler: convergence, temporal, flux and orientation assessments with per-case measurements.
- CNS spatial: `cns-assessment.json`, `measurements.json`, `cns-spatial.png`/PDF; temporal: `temporal-survey.json` and PNG/PDF.
- Exact state/time-scale and orientation settings: [solution definitions](mms-verification.md#exact-fields-fluxes-rhs-and-parameters) and [orientation mapping](mms-verification.md#orientation-check).

## Extended studies and flux selection

Flux-capable recipes accept `--fluxes all` or a comma-separated subset such as
`--fluxes LLF,Roe,Ranocha`. Supported fluxes are LLF (alias LFR), Roe, HLL,
Chandrashekar, Ranocha and Pirozzoli. Invalid/duplicate selections fail before
launch. Fixed baseline/orientation/temporal recipes keep their documented flux.
Selection is applied during preparation; execution, checking and plots follow
the saved manifest. `--ranks N` selects execution ranks, not a verification mode.
No recipe edits source or rebuilds to switch numerical flux or dissipation.

| Recipe | Matrix and purpose |
| --- | --- |
| `mms-euler-matrix` | P1–P5, six meshes (12/18/27/40/60/90 cells) plus finest half-step per degree, per selected flux; 35 cases per flux, 210 for all six. Resolution-dependent fixed timesteps, $t_f=0.2$; reports rates against $h^{p+1}$ and checks time isolation without an unqualified nominal-order gate |
| `mms-euler-spatial-survey` | Single-flux P1–P5 spatial survey with five meshes and finest half-step controls; 30 cases. This legacy survey launches one rank per case |
| `mms-euler-spatial-fine` | P1–P5 fine-grid extension, $h=0.1/0.05/0.025/0.0125$, $\Delta t=0.000125$, plus finest half-step controls; 25 cases. Expensive; completed-case recovery is available |
| `mms-euler-temporal-survey` | Euler RK3SSP/RK4, four timesteps $0.001/0.0005/0.00025/0.000125$, $t_f=0.2$, $\sigma=126$, P15 on $6\times6$ cells with P17 finest-step controls; 10 cases |
| `mms-euler-flux-comparison` | P2/P4 on 12/24/48 cells for selected production fluxes, $\Delta t=0.0005$, $t_f=0.2$; 36 cases for six fluxes. Observational comparison |
| `mms-euler-rhs` | Projected production RHS defects, P1–P5, 6/12/24 meshes, six fluxes plus constant states: 96 cases sampled at $t=0,0.2,1$ |

For the extended spatial matrix inside a GPU allocation:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-euler-matrix \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cuda \
  --fluxes all --degrees 1,2,3,4,5 --cells 12,18,27,40,60,90 \
  --reference-dt .001 --final-time .2 --ranks 1 --timeout 86400 \
  --results "$RESULTS_ROOT/euler-matrix-001"
```

Mesh widths in the main matrix are $h=2/N$ on the width-two domain; the coarsest
mesh has 144 elements; choose ranks appropriate to that mesh and the allocation. Use the
same command with another row's recipe ID to run just that study. Inspect `show`
and select individual studies to control cost: invoking the entire
`mms-verification` suite also runs the expensive fine-grid extension.

The RHS recipe needs an additional target, built in the same configured build:

```bash
cmake --build "$THESEUS_BUILD" --target euler_rhs_probe -j 4
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-euler-rhs \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --fluxes all --ranks 2 --results "$RESULTS_ROOT/euler-rhs-001"
```

**Major points**

- Broad degree/flux matrices are explicit studies, outside routine quick selection.
- Reported expected slopes are reference behavior; survey observations are not automatically nominal-order qualification.
- The [RHS diagnostic](mms-verification.md#projected-rhs-diagnostic) evaluates an instantaneous defect separately from evolved-solution convergence.

## Reuse results and recover interrupted studies

Plots can be regenerated from saved measurements without rerunning the solver.
For example, with a completed CNS spatial recipe directory:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/mms/cns_mms_study.py" plot \
  --source "$THESEUS_SOURCE" --results "$RESULTS_ROOT/cns-qualified-001/mms-cns-qualification/cases"
```

For a direct single-recipe invocation, its `cases` directory is immediately
under the chosen result directory; suites place each recipe under its ID. Use
the actual saved layout. To analyze completed cases from an interrupted Euler
spatial survey:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/mms/euler_spatial_survey.py" check-completed \
  --source "$THESEUS_SOURCE" --results "$SURVEY_RESULTS/cases" --device cpu
```

This does not resume simulations. It excludes unstarted/interrupted cases,
reports missing controls explicitly and writes `completed-measurements.json`,
`completed-analysis.json` and `completed-density-orders.png`/PDF. The survey
remains incomplete; a plot cannot convert it into a passing qualification.

**Artifact inventory**

- Preserve raw inputs/logs/fields and compact measurements/assessments together.
- Archived documentation plots have separate [measurement provenance](assets/mms/provenance.json) and are not overwritten by later runs.
- Recovery output states which cases and controls remain missing.

## Broad suite, CI and source correspondence

`ci-quick.yml` invokes the broad shared runner's `run` and `check` actions with
default `--mms-level quick`. It selects the two quick Euler recipes and the quick
CNS recipe, alongside registered CTests and unrelated integration/smoke/golden
checks. `ci-nightly.yml` uses the same runner with `--mms-level qualification`,
selecting the longer Euler/CNS recipes instead of their quick subsets. The
recorded level is saved in the manifest/summary and reconstructed by `check`.
Developer existing-build suites are MMS-only shortcuts, not the entire PR suite.

For an agreed full-code validation, this example selects qualification coverage:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" start --suite full \
  --source "$THESEUS_SOURCE" --prefix "$TPL_PATH" --python "$PYTHON" \
  --device cpu --fluxes all --mms-level qualification \
  --results "$RESULTS_ROOT/full-qualified-001"
```

Omit `--mms-level qualification` to retain the broad suite with quick MMS, as in
PR CI. `start` launches independently; `run` executes in the foreground, suitable
for an HPC batch script that owns the wait. Full-suite builds are the normal
Cartesian/axisymmetric CPU configurations, not the supplied optimized accuracy
build. After completion:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" check \
  --source "$THESEUS_SOURCE" --results "$RESULTS_ROOT/full-qualified-001"
```

**Major points**

- Full-suite pass plus current-source/result correspondence is distinct from existing-build recipe success.
- All-flux full coverage at either recorded MMS level can satisfy the full-suite check; reduced-flux diagnostics cannot.
- Broad regression, configuration-specific MMS evidence and predictive-use validation serve different purposes; see the [evidence limits](mms-verification.md#evidence-limits-and-prediction-use).

## Sutherland source consistency checks

The variable-transport source has independent consistency checks alongside the
explicit coupled study below; the routine quick/qualification recipes above
remain constant-transport studies.
The registered source tests compile both transport variants explicitly, so they
can run from either configured build without rebuilding solver source between cases:

```bash
cmake --build "$THESEUS_BUILD" --target trigonometric_sutherland_sutherland_tests \
  trigonometric_sutherland_constant_tests -j 4
ctest --test-dir "$THESEUS_BUILD" \
  -R 'TrigonometricSutherland_CPU|SutherlandTransportMismatch_CPU' --output-on-failure

"$PYTHON" "$THESEUS_SOURCE/tests/mms/cns_forcing_audit.py" --sutherland \
  --output "$RESULTS_ROOT/sutherland-forcing-audit.json"
```

The audit independently differences physical fluxes with varying coefficients and
with each sample's coefficients frozen. Their difference quantifies the product-rule
terms, alongside stress, viscous work and conduction magnitudes. It uses
$R=1/600$, $\mu_0=0.02$, $\mathrm{Pr}=0.72$. `TrigonometricSutherland_CUDA` is
also registered in CUDA builds; run it on the target device for device evidence.

**Artifacts and coverage**

- Source oracle in 2D/3D, several time/transport scales, compiled transport agreement, zero-viscosity Euler limit, and source diagnostic controls.
- Guards in both directions prevent constant/variable source–build mismatches.
- Sampled transport ranges and coefficient-gradient contributions in the audit JSON; these checks do not establish evolved Sutherland convergence.

## Coupled Sutherland spatial qualification

`mms-cns-sutherland-qualification` is an explicit recipe for an existing
`SUTHERLAND=ON` build. It is separate from the constant-transport qualification
suite; the routine broad suite does not reconfigure its solver to run this study.
Use the optimized build command above with `-DSUTHERLAND=ON` and a distinct build
directory, then run:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-cns-sutherland-qualification \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_SUTHERLAND_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --ranks 2 --results "$RESULTS_ROOT/sutherland-qualified-001"
```

Use `--device cuda` with an existing CUDA/Sutherland build inside your GPU
allocation. The shared harness handles launch and device placement as above.

| Setting | Value |
| --- | --- |
| Space and flux | P3/Roe on periodic $[-1,1]^2$, 6/12/24 cells per direction |
| Transport | $R=1/600$, $\mu_0=0.02$, $\mathrm{Pr}=0.72$; actual solver coefficients vary with temperature |
| Time integration | RK4, fixed $\Delta t=0.001$, $t_f=0.2$, 200 steps per ordinary case |
| Cases | Three meshes, finest half-step (400 steps), finest inviscid-only forcing, finest forcing without coefficient gradients |
| Spatial gate | Positive refinement rates; finest density/vector-momentum/energy physical $L_2$ orders $\ge3.5$ against nominal 4 |
| Isolation gates | Half-step error changes $<1\%$; omitted viscous forcing increases momentum/energy errors at least fivefold; omitted coefficient-gradient forcing increases them at least twofold |

P3 reduces truncation error so the weaker coefficient-gradient terms can be
resolved without a larger mesh. Both omitted-term controls retain the same
variable transport in the solver. They alter the analytic forcing only; they do
not test a different physical transport model. Source/transport consistency and
contribution audits accompany the evolved-solution evidence.

**Artifact inventory**

- The six-case study passed on a two-rank optimized CPU build. A separate 48-cell diagnostic used $\Delta t=0.00025$; see the [results and scope](mms-verification.md#sutherland-cns-spatial-convergence).
- Six saved configs and per-case logs/fields, `measurements.json`, `sutherland-assessment.json`, and `cns-spatial.png`/PDF with $h^4$ references.
- Explicit CPU/GPU recipe; no source edits, rank-parity comparisons or dissipation rebuilds between cases.
- This study covers 2D P3/Roe spatial accuracy; Sutherland temporal and coupled 3D accuracy require separate evidence.

## Selected-flux CNS studies

These explicit recipes evaluate the production numerical fluxes with the qualified
CNS spatial setups. Each selected flux gets three 2D meshes (6/12/24 cells per
direction) and one finest half-timestep control: four cases per flux, or 24 for
all six. Constant transport uses P2, $\mu=0.02$, $R=1$, $\mathrm{Pr}=0.72$;
Sutherland uses P3, $\mu_0=0.02$, $R=1/600$, $\mathrm{Pr}=0.72$. Both use RK4,
$t_f=0.2$, $\Delta t=0.001$ with control $\Delta t=0.0005$.

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-cns-fluxes \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --fluxes all --ranks 2 --results "$RESULTS_ROOT/cns-fluxes-001"

"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-cns-sutherland-fluxes \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_SUTHERLAND_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cpu \
  --fluxes LLF,Roe,Ranocha --ranks 2 --results "$RESULTS_ROOT/sutherland-flux-subset-001"
```

`--fluxes all` or any valid production subset works in either recipe. Use the
matching constant/Sutherland build; no recipe edits source or rebuilds to change
flux or dissipation. Use `--device cuda` with the corresponding CUDA build.
Results follow the saved manifest rather than later command-line selections.

The finest density, vector-momentum and energy orders must be at least $p+0.5$
(2.5 for P2, 3.5 for P3), and each headline error's half-timestep change must be
below 1%. Component orders remain in the assessment for diagnosis. These studies
repeat timestep isolation for each flux but do not repeat the source-omission,
coefficient-gradient or zero-transport controls already covered by the baseline
qualification workflows. They are outside the default routine CI selection.
`sutherland-mms-verification` contains the six-case Roe baseline qualification,
the selected-flux study and the extended degree/refinement matrix below (210 cases
with all defaults). Use individual recipes to limit scope/cost.

**Artifacts and scope**

- Per-flux configs, logs and fields, `measurements.json`, `cns-flux-assessment.json` and `cns-fluxes.png`/PDF.
- Separate supplied builds for the two transport laws; runtime production-flux selection within each build.
- A selectable convergence workflow does not establish that every flux passes; inspect each completed assessment before claiming coverage.

## Configurable CNS degree and refinement matrices

`mms-euler-matrix`, `mms-cns-matrix` and `mms-cns-sutherland-matrix` are extended **observational
surveys**, distinct from the committed quick/qualification tests. They vary
polynomial degree, flux and refinement depth using the same periodic 2D exact
family. The constant/Sutherland build must match the chosen recipe.

| Option | Default | Meaning |
| --- | --- | --- |
| `--cells` | `12,18,27,40,60,90` | Explicit increasing cell counts; `geometric` enables the doubling selectors below |
| `--degrees` | `1,2,3,4,5` | Unique subset of P1–P5 |
| `--mesh-levels` | `5` | Number of successive meshes, at least three |
| `--start-level` | `1` | First serial refinement level, nonnegative |
| `--base-cells` | `3` | Periodic base cells per direction, at least three |
| `--reference-dt` | `0.001` | Timestep anchor at P3 and 24 cells per direction |
| `--final-time` | `0.2` | Common physical final time |
| `--fluxes` | `all` | All six production fluxes or a named subset |

The default sequence uses 12/18/27/40/60/90 cells per direction. On the width-two
box the widths are 0.166667/0.111111/0.0740741/0.05/0.0333333/0.0222222,
with 6/9/13.5/20/30/45 elements per primitive wavelength. The ratios are
1.5/1.5/1.48148/1.5/1.5. Each case has its own periodic mesh, without serial
refinement. Assessment and plots use the actual adjacent spacing ratio:

$$
p_{\mathrm{obs},i}=\frac{\log(E_i/E_{i+1})}{\log(h_i/h_{i+1})}.
$$

Use `--cells geometric` to retain $N_k=N_0 2^{\ell_0+k}$ with the legacy
`--base-cells`, `--start-level` and `--mesh-levels` selectors. These selectors
only select geometry when `--cells geometric` is supplied.
Each degree/flux group adds one finest-grid half-timestep control. The count is
$n_p n_F(L+1)$: **210 per physics configuration, 630 for Euler, constant CNS and
Sutherland CNS together**. All six runtime fluxes are included by default;
compile-time dissipation is not varied.

For a bounded local comparison, use `--cells 12,18,27`, `--degrees 1,2,3,4,5`
and `--fluxes Roe`: 20 cases per configuration, 60 total. Keep final time 0.2.
This shorter sequence investigates changes relative to previous coarse studies;
it does not replace the six-mesh HPC evidence. Existing committed CI quick and
qualification cases and thresholds remain unchanged. Historical plots retain their
original meshes; new results must be labelled with the new sequence, not mixed
into a single refinement curve.

### Timestep policy and preflight cost

The starting policy is conservative relative to the tested P3/24-cell anchor:

$$
\Delta t_{\mathrm{bound}}=\Delta t_{\mathrm{ref}}
\min\left(1,\left(\frac{24}{N}\right)^2\right)
\min\left(1,\left(\frac{3}{p}\right)^4\right).
$$

The step count is $n=\lceil t_f/\Delta t_{\mathrm{bound}}\rceil$, and the actual
fixed step is $\Delta t=t_f/n$. The control halves that step and doubles $n$.
Coarser meshes and lower degrees never enlarge the step above the anchor.
This scaling is an initial explicit-viscous stability policy, not a universal
stability proof or a guarantee that temporal error is negligible. Every group's
finest control must demonstrate headline error changes below 1%.

`matrix-plan.json` records selectors, timestep policy, case count, maximum step
count and conservative-state unknown count. `cases.json` includes every case's
cells, degree, actual timestep, steps, final time and unknown count. The default
CNS plan's largest case has 1,166,400 conservative unknowns, and its longest
control has 43,404 steps. GPU acceleration does not remove the timestep cost;
prepare and inspect the plan before committing an allocation to the full matrix.

### Run on an existing GPU build

Inside an allocation with the normal site launcher/device visibility:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-cns-matrix \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cuda --ranks 1 \
  --fluxes all --degrees 1,2,3,4,5 --cells 12,18,27,40,60,90 \
  --reference-dt .001 --final-time .2 --timeout 86400 \
  --results "$RESULTS_ROOT/cns-matrix-gpu-001"

"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test mms-cns-sutherland-matrix \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_SUTHERLAND_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" --device cuda --ranks 1 \
  --fluxes all --degrees 1,2,3,4,5 --cells 12,18,27,40,60,90 \
  --reference-dt .001 --final-time .2 --timeout 86400 \
  --results "$RESULTS_ROOT/sutherland-matrix-gpu-001"
```

Set `--timeout` to suit the campaign/allocation; the runner default is one hour
per recipe. Run these foreground commands from your batch script. No source
patching, transport/dissipation changes or rebuilding occurs between cases.
The shared harness controls MPI/GPU placement; choose ranks appropriate to both
the coarsest mesh and the allocation. Initial/final fields only are requested.
Python analysis and field output still consume CPU time and storage.

For a smaller first GPU check, select `--degrees 2,3 --fluxes Roe` and `--cells 12,18,27`. For deeper evaluation, supply a longer explicit
`--cells` list. The physical parameters/final time should remain fixed within a
refinement study; a short-time pilot tests workflow execution, not the same
asymptotic behavior as the full-time campaign.

### Prepare, inspect, execute and analyze separately

The Bash recipe exposes stages for preflight and batch execution:

```bash
RECIPE="$THESEUS_SOURCE/tests/recipes/integration/cns-sutherland-matrix.sh"
RUN_DIR="$RESULTS_ROOT/sutherland-staged-001"
args=(--source "$THESEUS_SOURCE" --build-standard "$THESEUS_SUTHERLAND_BUILD"
      --prefix "$TPL_PATH" --python "$PYTHON" --device cuda --ranks 1
      --output "$RUN_DIR" --fluxes all --degrees 1,2,3,4,5
      --cells 12,18,27,40,60,90
      --reference-dt .001 --final-time .2)
bash "$RECIPE" "${args[@]}" --stage prepare
"$PYTHON" -m json.tool "$RUN_DIR/cases/matrix-plan.json"

# Execute inside the intended allocation, using the same build and prepared plan.
bash "$RECIPE" "${args[@]}" --stage run
bash "$RECIPE" "${args[@]}" --stage compare
```

After preparation, execution/checking use the saved manifest. Later changes to
command-line selectors do not change the prepared cases. Use a fresh output
directory for a different plan; do not silently mix campaigns.

**Artifacts and interpretation**

- `matrix-plan.json`, cases/configs/logs/fields and `measurements.json` preserve the full study specification and data.
- `matrix-assessment.json` records all refinement rates, component diagnostics, monotonicity, finest half-step changes and per-group `nominal_minus_half_met` flags.
- `controls_passed`/`passed` describe survey control checks, **not nominal-order qualification**. `accuracy_qualified` is always false. `order_expectations_met`, `needs_order_review` and status `completed_with_order_flags` identify shortfalls even when controls pass. Nonfinite/invalid data or failed time isolation prevent a control pass.
- `density-all-orders.png`/PDF compares all degrees per flux with $h^{p+1}$ references. `plots-pN/cns-fluxes.png`/PDF shows each degree's density/momentum/energy errors and rates across fluxes.
- Existing quick and qualification case counts, assertions and selection remain unchanged. These larger surveys stay opt-in; target-device numerical evidence must still be collected on the actual GPU build.
