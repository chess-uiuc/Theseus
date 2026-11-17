# MMS capability and documentation inventory

This inventory describes the Euler/CNS verification capabilities and their public
entry points. The tests verify selected smooth calorically perfect gas equations
and convergence; they do not establish all application/model configurations.

## Directory layout

Runtime state/source evaluators are grouped under `include/verification/mms/`,
with source-registry adapters in its `sources/` subdirectory and exact-state
registration in `Registration.hpp`. The general source factory delegates to those
MMS registration functions. General source
infrastructure remains in its normal location. Tests, independent references,
convergence tools and plotting are grouped under `tests/mms/`; general test
support remains under `tests/`. Production code does not depend on test files.

Runtime source and initial/exact-state names, recipe IDs, CTest names and binary
target names are unchanged. Direct Python tool paths use `tests/mms/`. Reconfigure
and rebuild after moving headers/C++ sources; eligible CUDA builds retain their
source language properties. No separate verification executable/build switch is
introduced. [Runtime layout](../include/verification/mms/README.md) and
[test-tool layout](../tests/mms/README.md) describe the dependencies.

## Implementation

| Capability | Files / interface |
| --- | --- |
| Periodic multidirectional Euler state, derivatives and forcing | `include/verification/mms/TrigonometricEulerMMS.hpp`, `include/verification/mms/sources/TrigonometricEuler.hpp` |
| Subsonic/transonic/supersonic mean-speed variants | Source `regime` selector; `RegimeTrigonometricMMS` initial/exact states; conservative Mach bounds in matrix plans |
| Constant-transport CNS Hessians and forcing | `include/verification/mms/TrigonometricCNSMMS.hpp`, `include/verification/mms/sources/TrigonometricCNS.hpp` |
| Sutherland transport and coefficient-gradient forcing | `include/verification/mms/TrigonometricSutherlandMMS.hpp`, `include/verification/mms/sources/TrigonometricSutherland.hpp` |
| Runtime source selection and validation | `src/SourceConfiguration.cpp`; `trigonometric-euler-mms`, `trigonometric-cns-mms`, `trigonometric-cns-sutherland-mms` |
| Existing numerical fluxes | LLF, Roe, HLL, Chandrashekar, Ranocha, Pirozzoli; source patches/rebuilds are not used to change dissipation within a matrix |

## Workflows and analysis

| Capability | Entry points |
| --- | --- |
| Complete existing-build/CI suite | `scripts/run_tests.py`; quick MMS by default, qualification with `--mms-level qualification` |
| Recipe discovery and executable commands | `run_tests.py list`, `show`; `tests/recipes/recipes.toml`, shared `recipe-environment.bash` |
| Euler quick/qualification and reference fields | `tests/mms/trigonometric_euler_study.py`, Euler recipes |
| Constant CNS quick/2D/3D qualification | `tests/mms/cns_mms_study.py`, `cns-quick.sh`, `cns-mms.sh`, `cns-3d.sh` |
| Sutherland qualification | `tests/mms/cns_sutherland_study.py`, `cns-sutherland.sh` |
| Temporal verification | `tests/mms/euler_temporal_survey.py`, `tests/mms/cns_temporal_study.py`; Euler/CNS temporal recipes |
| All/subset flux comparisons | `tests/mms/euler_flux_comparison.py`, `tests/mms/cns_flux_study.py`, `tests/mms/mms_fluxes.py`; flux recipes |
| Configurable Euler/constant/Sutherland spatial matrix | `tests/mms/cns_matrix_study.py`; `mms-euler-matrix`, `mms-cns-matrix`, `mms-cns-sutherland-matrix` |
| Orientation investigation | `tests/mms/euler_orientation_study.py`, `euler-orientation.sh` |
| Projected RHS defect investigation | `tests/mms/euler_rhs_probe.cpp`, `tests/mms/euler_rhs_study.py`, `euler-rhs.sh` |
| Legacy spatial/fine-grid surveys and partial analysis | `tests/mms/euler_spatial_survey.py`; spatial-survey/fine recipes |
| Physical conservative error integration | `tests/mms/mms_error_norms.py`; vector momentum headline, component records retained |
| Independent forcing contribution audit | `tests/mms/cns_forcing_audit.py` |
| Roundoff-aware final time validation | `tests/mms/trigonometric_euler_study.py`; bounded accumulated-time tolerance and declared step checks |

Matrix output includes `matrix-plan.json`, `cases.json`, solver inputs/logs/fields,
`measurements.json`, `matrix-assessment.json` and plots. A successful observational
matrix checks controls; numerical order shortfalls remain explicit flags. It does
not silently become a nominal-order qualification. Automatic case-level resume is
not implemented; saved-field analysis and plot regeneration are independent stages.

## Automated checks

`tests/CMakeLists.txt` registers the helper and source checks; recipe registration
selects integrated runs through the central runner. Coverage includes:

- Euler/CNS/Sutherland source values and derivatives, physical-flux differences,
  uniform-gamma assumptions, transport/build mismatch guards and zero transport.
- Production BR1 gradients, constant states and selected gradient refinement.
- Physical error quadrature, orientation interpretation, preparation/selection,
  order assessment, time/spatial controls and runner behavior.
- Unequal refinement ratios in both numerical assessment and plots; final-time
  roundoff acceptance with wrong-time, missing-step and wrong-step-count rejection.
- Independent LLF/scalar-dissipation formulas, normal scaling, conservation,
  entropy and kinetic-energy identities in `tests/test_riemann_fluxes.cpp`.
- Documentation links, Bash example syntax, compact-evidence hashes and portable
  public metadata in `tests/mms/mms_documentation_tests.py`.

The new scalar identity tests execute on CPU. Source/gradient CUDA targets are
registered in eligible builds; the completed GPU matrix supplies the documented
coupled 2D runtime evidence. Do not infer a device-specific check from a CPU pass.

## Documentation and retained evidence

| Document / assets | Contents |
| --- | --- |
| [Verification guide](mms-verification.md) | Governing-equation links; exact fields/coefficients; fluxes/RHS; baseline and GPU spatial/temporal/3D plots; observed exceptions; references and future scope |
| [Evaluation guide](mms-evaluations.md) | Quick, qualification, extended and existing-build recipes; selectors; gates; timestep policy; staged execution and recovery |
| [HPC guide](mms-hpc.md) | Two CUDA builds; 90 seven-case jobs; pilot/dependency submission; remaining jobs; analysis without GPU/allocation; compact data collection; retry/resume limits |
| [Source-term interface](source-terms.md) | Source registration, units and runtime input interfaces |
| [Test runner](test-runner.md), [adding tests](adding-tests.md), [verification overview](verification.md) | Central suite, CTest/recipe registration and broader verification context |
| `docs/assets/mms/` | Six baseline figures with measurement JSON and provenance |
| `docs/assets/mms/hpc-matrix/` | Sanitized full-matrix measurements, 450 rates, coverage/flag summary, figures and SHA-256 provenance |

The public evidence contains no private build/output paths, usernames, account
names or scheduler IDs. Run-generated identities and raw logs intentionally
record the execution environment; inspect/redact those before public sharing.

## Scope still outstanding

Explicit subsonic and Mach-above-three supersonic mean-speed variants are available
for the extended matrices alongside the original mildly transonic baseline.
Full new-regime survey evidence remains to be collected. Hypersonic families and
literal Roy coefficient sets remain planned. Sutherland
temporal and coupled 3D accuracy, LTE MMS and boundary extensions remain separate
work. Two Euler P4 scalar-flux configurations retain documented order shortfalls;
no production scalar-dissipation defect was demonstrated by the pointwise audit.
