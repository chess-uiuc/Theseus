# Optional numerical kernel stress tests

These tests exercise Helmholtz thermodynamic means, LTEEC volume/face fluxes,
Lax dissipation and viscous-face assembly with repeated allocations and large
batches. They check independent mathematical references, output guards and
agreement between ordinary host evaluations and the selected execution backend.
They require no PLATO installation, mesh or simulation configuration.

The tests are **disabled by default**. Enabling them adds test executables and
CTest entries; it does not change the solver's numerical methods, transport or
flux selection. `BUILD_TESTING=ON` is required.

## Enable and run in an existing build

Use your existing source/build paths and compiler environment. Reconfiguring an
existing build preserves its CUDA, dependency and architecture settings.

```bash
cmake -S "$THESEUS_SOURCE" -B "$THESEUS_BUILD" \
  -DBUILD_TESTING=ON -DTHESEUS_BUILD_KERNEL_STRESS=ON
cmake --build "$THESEUS_BUILD" --target \
  helmholtz_mean_stress_tests_reference \
  helmholtz_mean_stress_tests_channel \
  lteec_flux_stress_tests_standard \
  lteec_flux_stress_tests_sutherland -j 2
ctest --test-dir "$THESEUS_BUILD" -L kernel-stress --output-on-failure
```

A CPU build registers five entries. A CUDA build also registers four CUDA entries;
run them inside a GPU allocation. CUDA entries fail if the CUDA backend is not
selected. To select just those device entries:

```bash
ctest --test-dir "$THESEUS_BUILD" -L kernel-stress -R '_CUDA$' --output-on-failure
```

To return to the default configuration:

```bash
cmake -S "$THESEUS_SOURCE" -B "$THESEUS_BUILD" -DTHESEUS_BUILD_KERNEL_STRESS=OFF
```

This removes the optional targets/tests from the generated build. Existing
executable files may remain in a reused build directory but are no longer selected.

## What each test establishes

| Test | Reference and coverage |
| --- | --- |
| `KernelStressReference` | Regenerate the stored channel expectations with 80-digit Decimal arithmetic; verify the generator against an independent analytic polynomial in all four difference/limit combinations. |
| `HelmholtzMeanStress_reference_CPU/CUDA` | Seven independent near-state reference cases, both endpoint orientations, permuted assignments and batches of 1, 127, 128, 129, 4,097, 65,537 and 1,048,579 evaluations, repeated eight times. |
| `HelmholtzMeanStress_channel_CPU/CUDA` | Fifteen actual close-state channel pairs with literal corner jets and independent 80-digit reference means; same batch, reversal, host/backend and guard coverage. |
| `LTEECFlux_stress_standard_CPU/CUDA` | Analytic volume/face flux and entropy checks, Lax helper, viscous-face heat-conduction assembly, orientation and guards. Dimensions 1–3, orders 2/7/15/31/63/127 and face/point execution patterns, repeated four times. |
| `LTEECFlux_stress_sutherland_CPU/CUDA` | The same fixture compiled with Sutherland viscosity, exercising that transport path while retaining tabulated conductivity. |

The largest 3D face invocation has 16,384 points. The mean fixtures test valid,
selected close states; they are not an exhaustive EOS-domain or extreme-state
sweep. Flux checks share the regular fixture's independent analytic oracle and
assertions. Sutherland's physical viscosity law is separately checked by gas tests.

Mean accuracy and host/backend agreement use the mean roundoff bound:
4096 machine epsilons times `max(1, abs(reference))`. Analytic flux/assembly checks
use a `2e-10 * max(1, abs(reference))` bound. Stress host/backend
flux/helper/assembly comparisons use `1e-10 * max(1, abs(host_value))`. Output
guards must remain exactly equal to their sentinel value. Any failed check or
nonfinite output returns a failing exit code; no failures are converted to skips.

Each executable accepts `[cpu|cuda] [quick|full]`. CTest selects `full`. `quick`
uses smaller batches/repetition counts for development; it is not the full stress
coverage. Successful output reports check counts, reference error where applicable,
and the maximum host/backend scaled difference. Detailed successful CTest output
is available in `Testing/Temporary/LastTest.log` inside the build directory.

## Reference provenance

`tests/helmholtz_channel_difference_inputs.json` contains binary64 cell bounds,
corner values/derivatives and endpoint states. It includes close pairs found in
the second RK stage of an unforced AIR5 channel transient, with selected close-state configurations and seeded samples.
The reference polynomial represents the bicubic Helmholtz interpolant.

`helmholtz_channel_difference_reference.py` computes a physical power-basis
polynomial using Decimal at 80-digit precision, evaluates the intended discrete
gradients and derivative limits, and generates `helmholtz_channel_difference_cases.hpp`.
The expectations are independent mathematical calculations, **not current CPU
outputs**. The generator and inputs use only the Python standard library. Verify
reproducibility and the analytic generator checks with:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/helmholtz_stress_reference_tests.py"
```

`helmholtz_difference_cases.hpp` contains a second independent reference set.
Reference-accuracy assertions and host/backend agreement assertions are reported
separately. Neither is a whole-flow physical validation.

## Full central-suite opt-in

The central runner keeps stress testing off unless `--kernel-stress` is given.
Add it to your usual full-suite command; all other environment and MPI options
stay as appropriate for your machine:

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" start \
  --source "$THESEUS_SOURCE" --results "$TEST_RESULTS" \
  --prefix "$TPL_PATH" --python "$PYTHON" --jobs 2 --kernel-stress
```

The runner records this choice and enables the CMake option in both its Cartesian
and axisymmetric builds. Their normal CTest stages discover and execute the
registered tests. `--kernel-stress` requires the full suite; integrated-only mode
does not run CTest and rejects that flag. On HPC use the normal foreground `run`
action when the batch script should own the wait.
