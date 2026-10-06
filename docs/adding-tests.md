# Adding tests

Every new capability, construct or behavioral change must include automated tests.
Bug fixes must add or extend regression coverage. Choose tests that establish the
intended behavior, including relevant invalid inputs and boundary cases.

Register tests in the shared validation suite. **Do not add individual test commands
to CI workflow files.** Workflows prepare the environment, invoke `validate.py`, and
publish results. Test selection belongs to the registration paths below, so the same
checks are available locally, on HPC systems and in CI.

Choose the kind of test by what it needs to execute.

| Test kind | Purpose | Register in |
| --- | --- | --- |
| Unit/helper | Check a function, data structure or infrastructure script | `tests/CMakeLists.txt` |
| Integrated | Run Theseus and check simulation behavior | `integrated_commands` in `scripts/validate.py` |
| Smoke or golden case | Run an example, optionally compare its visualization output | `suite_commands` in `scripts/validate.py` |

Register each check once. The full suite runs CTest and the integrated registry;
registering the same simulation in both would execute it twice.

## Add a unit or helper test

Follow a nearby test in `tests/CMakeLists.txt`: define any executable target and
register it with `add_test`. Use a descriptive name. Run it directly with CTest:

```bash
ctest --test-dir /path/to/build -R '^YourTestName$' --output-on-failure
```

For a new C++ test file `tests/my_feature_tests.cpp`, add a target following this
pattern (replace the illustrative names with your feature's names):

```cmake
add_executable(my_feature_tests unit_test_main.cpp my_feature_tests.cpp)
target_include_directories(my_feature_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(my_feature_tests PRIVATE ${THESEUS_MFEM_TARGET} MPI::MPI_CXX)
add_test(NAME MyFeatureTests COMMAND my_feature_tests)
```

Use `TEST(...)` and the assertion macros from `unit_test.hpp`, as demonstrated in
[physical_state_tests.cpp](../tests/physical_state_tests.cpp). Call the production
function and compare against an independently derived result. Return zero after the
assertions. Link only the dependencies required by the test.

A Python infrastructure check can be registered directly:

```cmake
add_test(NAME MyHelperTests
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/my_helper_tests.py)
```

Configure the build after changing registration, build the target, then run the
CTest command above. No entry in the workflow or Python integrated registry is needed.

The full runner discovers registered tests in both build configurations. Use CMake
conditions when a check requires a particular geometry or dependency.

## Add an integrated test

An integrated checker has three jobs: prepare a physical case, launch Theseus,
and assert the expected result. Keep those steps visible in the code.

Use [the entropy-wave checker](../tests/axisymmetric_entropy_wave_convergence_test.py)
as a compact example of multiple simulations followed by a numerical assertion.
[The restart checker](../tests/restart_integration_test.py) demonstrates checkpoint
setup and output comparisons.

### Prepare inputs and run the simulation

Accept `--source`, `--executable` and `--device`. Add explicit arguments for any
additional data your checker needs. Use absolute paths for meshes, databases and
output directories in generated configurations.

The helpers in `tests/integration_support.py` retain a working directory and invoke
the platform harness. A checker can use them like this:

```python
with work_directory(prefix="my-check-") as directory:
    output = Path(directory)
    config["runTime"]["output_file_path"] = str(output)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    result = run_simulation(args.executable, config_path,
                            ranks=2, device=args.device)
    if result.returncode != 0:
        raise RuntimeError(result.stdout)

    # Read output and assert the physical result here.
```

`work_directory` creates a unique directory under the checker's working directory
and retains it on success or failure. `run_simulation` writes a simulation log beside
the configuration and returns captured output and an exit status.

All simulation launches go through `run_theseus.sh`. Do not add `mpiexec`, `srun`,
`ibrun`, hostname detection or device-ID assignment to a checker. Python handles
inputs and assertions; the harness handles platform execution.

For rejection tests, require both a nonzero exit status and the expected diagnostic.
For numerical tests, derive the expected result independently of the implementation.
Choose tolerances for the numerical method and the quantity being checked.

### Register the checker

Add an entry to `integrated_commands` in `scripts/validate.py`. Its fields are:

| Field | Meaning |
| --- | --- |
| Identifier | Unique short name used in log and result paths |
| Title | Human-readable name shown in the summary |
| Geometry | `standard` or `axisymmetric` build |
| Script | Checker filename under `tests/` |
| Options | Extra checker arguments, such as a database or reference executable |

For example, the registry entry for the existing entropy-wave checker is:

```python
('axis-convergence', 'Axisymmetric Entropy-Wave Convergence', 'axisymmetric',
 'axisymmetric_entropy_wave_convergence_test.py', []),
```

To add another checker, place its script under `tests/`, choose a unique identifier
and descriptive title, select its build geometry, and add its tuple to `entries`.
Use the options list for extra inputs. A new required executable must also be
accounted for in `prepared_builds`, so HPC users receive a clear prerequisite check.

The runner supplies source, executable and device arguments. It creates a dedicated
working directory under `integrated/` and records one timed result for the checker.
Do not also register this simulation in CTest.

### Exercise the checker

Run the Python script directly from a scratch directory, using absolute source and
executable paths. For example:

```bash
python /path/to/Theseus/tests/axisymmetric_entropy_wave_convergence_test.py \
  --source /path/to/Theseus \
  --executable /path/to/build-axis/theseus --device cpu
```

Use the same command inside an allocation with `--device cuda` and a CUDA executable.
The fallback `mpiexec` launcher is resolved from PATH; scheduler selection belongs
to the harness.

Then run the [central suite](validation-runner.md) and check that the new entry
appears once, executes, and reports the intended result. Update the
[test inventory](verification.md) with its purpose, geometry and acceptance criteria.
Changes to established expectations or tolerances require an explanation of why the
previous numerical contract was incorrect. Discuss proposed changes with developers
or maintainers before modifying accepted data or test criteria.

## Add a smoke case

A smoke test checks that a configuration starts, completes and produces the required
output. Use an integrated numerical check when success also depends on physical values,
convergence, conservation, restart behavior or another property of the result.

In `suite_commands`, add the case to `cases`, then add its identifier and timestep
to the smoke loop. For example, the existing vortex case is defined by:

```python
'vortex': ('Euler/2D/IsentropicVortex', 'IsentropicVortex',
           'Inviscid Isentropic Vortex'),
```

The three fields are the configuration directory relative to `TestCases`, the golden
reference directory name, and the summary title. The smoke loop supplies 100 steps;
use an explicit `simulation` call if a new case needs another duration. Check that
the case's final-time setting allows the requested steps to complete.

## Add golden results for regression

A golden regression compares a simulation with an approved reference dataset.
It protects numerical behavior; comparison with a result generated by the same code
is not, on its own, evidence of physical correctness.

1. Choose the configuration, mesh, order, timestep, output cycle and MPI rank count.
   Establish the reference's credibility with analytic results, independent validation,
   or an already verified implementation.
2. Generate candidate output with `run_theseus.sh` in a separate directory. Record the
   source revision, exact command, configuration and compiler/dependency environment.
3. Inspect the candidate fields and check the intended physical behavior. Document
   why the reference and comparison tolerances are appropriate before adopting it.
4. Store the reference cycle and every piece file named by its `data.pvtu`, then
   register the simulation and comparison in `suite_commands`.

For example, this generates a *candidate* for the existing vortex recipe, using the
harness's default two ranks. It does not install or approve a new reference:

```bash
scripts/run_theseus.sh -b /path/to/build \
  -c TestCases/Euler/2D/IsentropicVortex/config.json \
  -o ../candidate-vortex -p 2 -r cpu -t 0.001 -n 500
```

The candidate cycle is `../candidate-vortex/IsentropicVortex/ParaView/Cycle000500`.
Keep reference-generation notes with the case documentation: recipe, provenance,
validation rationale and tolerances. Put discussion and review history in project records.

### Store the reference

The directory layout is:

```text
TestCases/GoldenData/<case>/Cycle000100/data.pvtu
TestCases/GoldenData/<case>/Cycle000100/proc000000.vtu
TestCases/GoldenData/<case>/Cycle000100/proc000001.vtu
```

The cycle and number of pieces must match the registered recipe. Large references
may instead be stored in `TestCases/GoldenData/<case>.tgz`. From the `GoldenData`
directory, create and inspect an archive with:

```bash
tar -czf MyCase.tgz MyCase/
tar -tzf MyCase.tgz
```

The archive must contain `MyCase/Cycle...`, not just the cycle's contents. Ensure
the chosen directory or archive is tracked by Git and included in transfers.
Preparation prefers an existing directory; only when that directory is absent does
it extract the archive. Avoid keeping a stale unpacked directory beside an updated
archive. Missing data or failed extraction makes reference preparation fail.

### Register the comparison

Add the case mapping described above and an entry in the golden loop of
`suite_commands`. The existing vortex entry is:

```python
('vortex', '0.001', 500, '1e-13', '1e-13'),
# identifier, timestep, output cycle/step count, absolute tolerance, relative tolerance
```

The runner launches the simulation, prepares its reference and invokes
`compare_viz.py`. Simulation and comparison have separate results and timings.
The current golden helper uses the Cartesian build; a case requiring another geometry
needs explicit runner support rather than being assigned the wrong executable.

Inspect a comparison directly when developing it:

```bash
python scripts/compare_viz.py /path/to/candidate/data.pvtu \
  /path/to/reference/data.pvtu --atol 1e-13 --rtol 1e-13
```

Those tolerances are the vortex example, not universal defaults. Derive tolerances
for the case's quantities and numerical method. Do not regenerate an established
reference or relax its tolerances merely to eliminate a failure; diagnose and discuss
changes to that numerical contract first.

## Complete the contribution

Before requesting review:

- Run the focused check and inspect its assertions and failure diagnostics.
- Run the shared suite and verify that the test is discovered, executes once in
  each intended configuration, and reports its result. Check that no second CI or
  CTest entry executes the same integrated simulation.
- For integrated tests, exercise the selected device path on the target platform
  and retain its logs; record any outstanding platform validation in project records.
- Update the [test inventory](verification.md) with the recipe and acceptance criteria.

Direct execution alone does not register coverage. Changes to a test's registration,
inputs or reference files must travel with the checker and runner changes.

## Harness options used by checkers

| Option | Purpose |
| --- | --- |
| `-e` | Select the Theseus executable |
| `-p`, `-r` | Set rank count and device |
| `-P` | Preserve input settings; explicit command-line overrides still apply |
| `-R` | Retain the per-case working directory for restart |
| `-k` | Let the checker validate outputs instead of requiring standard ParaView files |
| `-o` | Set the harness working directory; accepts absolute paths |

With `-P`, the configuration's `output_file_path` controls simulation output.
`-P` alone does not prevent working-directory cleanup. A failing simulation still
makes the harness fail when `-k` is used. The invocation directory must contain
`scripts/` so the harness can locate platform wrappers; `run_simulation` sets it
to the source root.

## Reading the runner implementation

The runner's main responsibilities are grouped into these functions:

| Function | Responsibility |
| --- | --- |
| `integrated_commands` | List integrated checkers and their build/input requirements |
| `suite_commands` | Assemble build, unit, integrated, smoke and comparison commands |
| `prepare` | Snapshot source, create result directories and record run settings |
| `prepared_builds` | Validate supplied builds and record their binaries/cache hashes |
| `worker` | Execute the commands in order and block checks whose prerequisites failed |
| `execute_step` | Capture one command's output, duration and exit status |
| `save_results` | Write the summary and machine-readable results |
| `check` | Verify a completed full run against the checkout |

Adding a checker normally changes the integrated registry, the checker script and
the inventory. It does not require changes to the execution or logging machinery.
