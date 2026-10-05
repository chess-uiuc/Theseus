# Adding tests

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
previous numerical contract was incorrect; see the repository's contribution rules
in [AGENTS.md](../AGENTS.md).

## Add a smoke or golden case

Use the case table and simulation calls in `suite_commands`. These invoke
`run_theseus.sh` and record simulation time separately from comparison time.
A golden comparison uses `compare_viz.py` with explicit tolerances.

Store references at `TestCases/GoldenData/<case>/...`, or in `<case>.tgz` containing
the same top-level case directory. Reference preparation prefers the directory,
falls back to the archive, and checks for the expected cycle's `data.pvtu`.
Include the piece files referenced by that file. Add the case and tolerances to
the inventory.

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
