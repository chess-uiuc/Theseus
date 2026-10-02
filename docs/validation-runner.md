# Validation runner

`scripts/validate.py` runs the same full suite locally and in Quick/Nightly CI.
It creates a source snapshot, clean Cartesian and axisymmetric builds, per-command
logs, a machine-readable result and a short text summary. Python 3.9 or later,
Git, CMake/CTest, MPI compilers, Bash, the installed MFEM/PLATO dependencies and
`ci-resources/requirements-ci.txt` Python packages are required.

## Run the suite

Set the compiler environment and choose a Python interpreter with the test
requirements installed. Supply the installed dependency prefix explicitly.
Results must be in a **new directory outside the source checkout**.

```bash
export CC=mpicc CXX=mpicxx
python3 scripts/validate.py run \
  --prefix /path/to/tpl/install \
  --python /path/to/test-environment/bin/python \
  --results ../validation/run-001 --jobs 4
```

`run` waits for completion and exits nonzero on failure. CI uses this command.
`start` accepts the same options, launches an independent local process, and
returns its PID and summary path immediately:

```bash
python3 scripts/validate.py start \
  --prefix /path/to/tpl/install \
  --python /path/to/test-environment/bin/python \
  --results ../validation/run-002 --jobs 4
cat ../validation/run-002/summary.txt
```

The background process needs no agent, network service or repeated polling. It
continues after the launcher exits; the machine must remain running. Each command
has a timeout, default 3600 seconds; set `--timeout SECONDS` if needed. A failure
does not stop independent checks. Failed prerequisites block dependent checks:
a failed build blocks its tests and simulations, and a failed simulation blocks
its comparison. Failed or blocked required checks make the overall run fail. Terminating the worker with SIGTERM also terminates its active
command process group. An abrupt kill or machine shutdown can leave a run marked
running: that is incomplete evidence and cannot pass `check`.

MPI arguments are passed to both CTest integration tests and the simulation
harness. Placement options (`--host`, `--map-by`, `--bind-to`) act as defaults:
explicit options supplied by an integration test take precedence. Other launcher
arguments are retained. For a local OpenMPI installation requiring explicit
placement, append:

```bash
--mpi-arg=--host --mpi-arg=localhost:4 \
--mpi-arg=--map-by --mpi-arg=slot:OVERSUBSCRIBE \
--mpi-arg=--bind-to --mpi-arg=none
```

Use `--mpiexec /path/to/mpiexec` to select the launcher. These placement arguments
are host-specific; CI using MPICH does not require them.

## Read and verify results

Each results directory contains:

| Path | Contents |
| --- | --- |
| `summary.txt` | Overall state, descriptive check names, separate elapsed times, failure/blocking reasons and log paths |
| `results.json` | Commands, working directories, exit codes, timestamps, elapsed seconds and log hashes |
| `manifest.json` | Source file hashes/modes, base commit, dependency prefix, compiler/interpreter paths and run options |
| `logs/` | One combined stdout/stderr log per command, including tool versions |
| `source/` | Snapshot of the source files used by the build and tests |
| `build-standard/`, `build-axisymmetric/` | Clean builds, CMake caches and CTest details |
| `cases/` | Simulation output and comparison inputs |
| `launcher.log` | Background worker startup errors (`start` only) |

To check whether a completed result applies to the current checkout:

```bash
python3 scripts/validate.py check --results ../validation/run-002
```

`check` exits zero only when every required command passed, its saved command and
log match the expected suite, the snapshot is intact, and the current source
matches. It detects modified, added and deleted files, including uncommitted and
untracked non-ignored files. Pure Markdown/Doxygen documentation edits under
`docs/`, and root-level Markdown/Doxygen files, are reported separately and do
not invalidate numerical results. They still require review.

The snapshot includes Git-tracked files and non-ignored untracked files, retaining
executable permissions. Ignored build/output/dependency directories are excluded.
Source symlinks and submodules are rejected rather than silently tested through
mutable external paths. Required source inputs must not be hidden by ignore rules.
Edits during snapshot creation cause an error; edits afterward leave the snapshot
intact and are checked for freshness against the working checkout.

Dependencies are shared from the supplied installed prefix, not copied. Do not
change the compiler, interpreter environment or dependency installation during a
run. If those change afterward, run the suite again even if the source is unchanged.
The records are local validation evidence, not a cryptographic attestation of the
external toolchain. Existing result directories are never overwritten or resumed.

## Suite and failure diagnosis

The suite configures Debug builds with PLATO, subcell blending and `NO_OPT`, runs
all registered CTests in both applicable configurations, then runs the smoke,
cyclic and five golden comparisons in the [verification matrix](verification.md).
`suite_commands` defines the commands and numerical comparison tolerances. Register
new tests in CTest where possible so both CI and local validation discover them.

The summary updates as each check starts or finishes. Simulations and comparisons
have separate entries and elapsed times; short internal identifiers remain in log
filenames. Elapsed times use a monotonic clock. Running/blocked entries show a dash
until a command has an actual duration. Both CI workflows print the summary in an
always-run step, including when validation fails, and upload the detailed records.

After collecting results, read the first failing command's log and reproduce that command
from its recorded working directory. Diagnose the cause before editing code or
test expectations. Changes to assertions, tolerances or reference data need a
numerical or specification justification; a passing result alone is not evidence
that the change is correct. After a fix, run focused checks, then start the full
suite in a fresh results directory. Results from different snapshots are not
combined. A missing, incomplete, failed or stale run cannot authorize a commit.


Committed tests are regression contracts: suspected errors in their assertions,
tolerances or reference data require evidence and discussion before modification.
New tests may be corrected during development before their first commit without
separate approval. Their expectations still need independent justification; a test
must not be tuned to hide a defect. After its first commit, the regression rule applies.
