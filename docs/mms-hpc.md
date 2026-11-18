# Running the three-regime MMS matrix on HPC

Build two CUDA executables, then run **90 independent jobs per regime**, seven
cases per job. Subsonic, transonic and supersonic campaigns together produce
270 groups / 1,890 simulations. Each group covers one physics model, degree and
flux on 12/18/27/40/60/90 cells, followed by a finest half-timestep control.
Spatial measurements and secondary control outcomes are retained separately.
No full regression suite or dissipation-source changes are part of this campaign.

This is a **Slurm template**. Supply your site's compiler/MPI/CUDA modules,
CUDA-enabled dependency prefix, account, partitions and GPU resource syntax.
Repeat module setup in the scripts if the site does not preserve the environment.
Keep source, dependencies and executables fixed throughout the campaign.

## Set the paths once

Run from a submission directory on storage visible to compute nodes:

```bash
export THESEUS_SOURCE=/absolute/path/to/Theseus
export TPL_PATH=/absolute/path/to/cuda-dependencies/install
export PYTHON=/absolute/path/to/python-environment/bin/python
export BUILD_ROOT=/absolute/path/to/ValidationRuns/mms/gpu-builds
export RESULTS_ROOT=/absolute/path/to/ValidationRuns/mms/hpc-regimes
export CUDA_ARCH=80  # Replace with your GPU architecture.
export ACCOUNT=your_account
export BUILD_PARTITION=your_build_partition
export GPU_PARTITION=your_gpu_partition
mkdir -p "$BUILD_ROOT" "$RESULTS_ROOT/logs"
for regime in subsonic transonic supersonic; do
  mkdir -p "$RESULTS_ROOT/$regime/logs"
done
"$PYTHON" -c 'import numpy, meshio, pyvista, vtk, matplotlib'
```

[Python requirements](../ci-resources/requirements-ci.txt) apply to preparation and
analysis. Use fresh result directories. Existing builds may be reused only when
source, transport switches and dependency/toolchain settings match the campaign.
Changing Python analysis alone does not require rebuilding the solver.

**Logs belong inside the results tree.** The submission commands below set Slurm
output paths there; solver logs stay beside each case. `#SBATCH` directives do
not expand shell environment variables, so pass these paths with `sbatch --output`.
No external log-directory symlink is needed.

## Build job: two executables

Save as `build-mms.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mms-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
set -euo pipefail
# Load the site's matching compiler/MPI/CUDA modules here if needed.
: "${THESEUS_SOURCE:?}" "${TPL_PATH:?}" "${PYTHON:?}"
: "${BUILD_ROOT:?}" "${CUDA_ARCH:?}"
for transport in constant sutherland; do
  sutherland=OFF
  if [[ "$transport" == sutherland ]]; then sutherland=ON; fi
  CC=mpicc CXX=mpicxx cmake -S "$THESEUS_SOURCE" -B "$BUILD_ROOT/$transport" \
    -DCMAKE_PREFIX_PATH="$TPL_PATH" -DPython3_EXECUTABLE="$PYTHON" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DTHESEUS_WITH_PLATO=ON \
    -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DSUBCELL_FV_BLENDING=OFF -DSUTHERLAND="$sutherland"
  cmake --build "$BUILD_ROOT/$transport" --target theseus -j
done
```

The **constant build serves Euler and constant CNS**; the Sutherland build serves
Sutherland CNS. The build partition needs the CUDA compiler, but no GPU allocation
unless the site requires one. Use fresh build directories after toolchain changes.

## Matrix array: one GPU and seven cases per task

Save as `run-mms.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mms-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
# Load the site's matching compiler/MPI/CUDA modules here if needed.
: "${THESEUS_SOURCE:?}" "${TPL_PATH:?}" "${PYTHON:?}"
: "${BUILD_ROOT:?}" "${RESULTS_ROOT:?}" "${SLURM_ARRAY_TASK_ID:?}" "${MMS_REGIME:?}"
case "$MMS_REGIME" in
  subsonic|transonic|supersonic) ;;
  *) echo "Invalid MMS_REGIME: $MMS_REGIME" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1
# The harness owns srun/MPI and GPU selection; omit workstation launcher overrides.
physics=(euler constant sutherland)
recipes=(mms-euler-matrix mms-cns-matrix mms-cns-sutherland-matrix)
fluxes=(lfr Roe HLL Chandrashekar Ranocha Pirozzoli)
id=$SLURM_ARRAY_TASK_ID
if (( id < 0 || id >= 90 )); then exit 2; fi
model=$((id / 30))
degree=$(((id % 30) / 6 + 1))
flux=${fluxes[$((id % 6))]}
build="$BUILD_ROOT/constant"
if (( model == 2 )); then build="$BUILD_ROOT/sutherland"; fi
campaign="$RESULTS_ROOT/$MMS_REGIME"
output="$campaign/${physics[$model]}/p$degree/$flux"
export MPLCONFIGDIR="$campaign/cache/${SLURM_ARRAY_JOB_ID}_${id}/matplotlib"
export XDG_CACHE_HOME="$campaign/cache/${SLURM_ARRAY_JOB_ID}_${id}"
printf 'Physics=%s degree=%s flux=%s regime=%s output=%s\n' \
  "${physics[$model]}" "$degree" "$flux" "$MMS_REGIME" "$output"
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test "${recipes[$model]}" \
  --source "$THESEUS_SOURCE" --build-standard "$build" --prefix "$TPL_PATH" \
  --python "$PYTHON" --device cuda --ranks 1 \
  --degrees "$degree" --fluxes "$flux" --cells "${MMS_CELLS:-12,18,27,40,60,90}" \
  --regime "$MMS_REGIME" --reference-dt "${MMS_REFERENCE_DT:-.001}" --final-time .2 \
  --timeout "${MMS_TIMEOUT:-1700}" --results "$output"
```

One rank uses the allocated GPU. Do not wrap the Python runner in another `srun`.
`lfr` is the runtime name for LLF. IDs 0–29 are Euler, 30–59 constant CNS and
60–89 Sutherland CNS; each block contains P1–P5 in the listed six-flux order.
IDs 13/43/73 are P3/Roe pilots. Supersonic P5 IDs are 24–29, 54–59 and 84–89.
The same two executables serve all regimes; runtime inputs select the state/source.

## Submit, inspect, then expand

The examples below are for a **fresh campaign**, not resubmission over completed
results. Submit the build and three pilots per regime:

```bash
bash -n build-mms.sbatch run-mms.sbatch
build_job=$(sbatch --parsable --account="$ACCOUNT" --partition="$BUILD_PARTITION" \
  --output="$RESULTS_ROOT/logs/build-%j.log" build-mms.sbatch)
build_job=${build_job%%;*}
printf 'Build job ID: %s\n' "$build_job"
for regime in subsonic transonic supersonic; do
  sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
    --dependency="afterok:${build_job:?Build job ID is required}" \
    --export="ALL,MMS_REGIME=$regime,MMS_TIMEOUT=550" \
    --time=00:10:00 --job-name="mms-$regime" \
    --output="$RESULTS_ROOT/$regime/logs/%x-%A_%a.log" \
    --array=13,43,73%3 run-mms.sbatch
done
```

For matching existing builds, skip build submission and omit the dependency.
Inspect solver/backend completion, measurements, assessment and control outcomes
before expanding. Order/control flags must stay visible; they are distinct from
solver failure. Pilots are part of the campaign, so exclude their IDs afterward.

```bash
# Remaining subsonic/transonic jobs: 30-minute allocations.
for regime in subsonic transonic; do
  sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
    --export="ALL,MMS_REGIME=$regime,MMS_TIMEOUT=1700" \
    --time=00:30:00 --job-name="mms-$regime" \
    --output="$RESULTS_ROOT/$regime/logs/%x-%A_%a.log" \
    --array=0-12,14-42,44-72,74-89%4 run-mms.sbatch
done
# Remaining supersonic P1-P4 jobs: exclude pilots and all P5 tasks.
sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --export="ALL,MMS_REGIME=supersonic,MMS_TIMEOUT=1700" \
  --time=00:30:00 --job-name=mms-supersonic \
  --output="$RESULTS_ROOT/supersonic/logs/%x-%A_%a.log" \
  --array=0-12,14-23,30-42,44-53,60-72,74-83%4 run-mms.sbatch
# Supersonic P5 jobs: longer allocation and matching runner timeout.
sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --export="ALL,MMS_REGIME=supersonic,MMS_TIMEOUT=3500" \
  --time=01:00:00 --job-name=mms-supersonic-p5 \
  --output="$RESULTS_ROOT/supersonic/logs/%x-%A_%a.log" \
  --array=24-29,54-59,84-89%4 run-mms.sbatch
```

`%4` applies to **each array independently**; these four submissions can use up
to 16 GPUs together. Lower concurrency as needed. A direct `--array=0-89%4`
campaign is an alternative to the pilot/remaining split, with a fresh root and a
wallclock covering its most expensive task; splitting by cost is usually preferable.
Command-line `--time` overrides a script's `#SBATCH --time`. Editing a script does
not change already-submitted jobs. Keep the runner timeout below the allocation.

Measured GPU stepping for the original P3 pilots was under three minutes per
seven-case group; observed job completion was within about five minutes. A
supersonic P5 Sutherland/HLL group requires about 51 minutes of stepping. Thus
30 minutes is insufficient for that group, while three hours is unnecessarily
conservative. These timings are hardware-specific, exclude some overhead and are
not guarantees: calibrate with a high-degree pilot (e.g. ID 86) before requesting
long arrays. Do not assume P3 pilots establish P5 runtime. Memory is also a site
starting point, not a measured universal requirement.

**There is no automatic case-level resume.** Preserve an interrupted task's output
under another name before resubmitting only its ID; all seven cases rerun. Do not
rerun completed groups. Solver failure, order flags and control failures are
separate findings. A failed comparison can still leave complete, useful spatial
measurements; preserve those and the assessment rather than deleting them.

## Reanalyze saved fields without an allocation

Analysis does not launch Theseus or need a GPU. On a host permitting field
postprocessing, use the same Python environment and keep fields/run metadata intact.
The `--device cuda` argument checks the recorded backend, not analysis hardware.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$RESULTS_ROOT/cache/analysis/matplotlib"
export XDG_CACHE_HOME="$RESULTS_ROOT/cache/analysis"
for regime in subsonic transonic supersonic; do
  for physics in euler constant sutherland; do
    for degree in 1 2 3 4 5; do
      for flux in lfr Roe HLL Chandrashekar Ranocha Pirozzoli; do
        cases="$RESULTS_ROOT/$regime/$physics/p$degree/$flux/cases"
        if [[ -f "$cases/measurements.json" &&
              ! -f "$cases/measurements.before-reanalysis.json" ]]; then
          cp "$cases/measurements.json" "$cases/measurements.before-reanalysis.json"
        fi
        "$PYTHON" "$THESEUS_SOURCE/tests/mms/cns_matrix_study.py" check \
          --source "$THESEUS_SOURCE" --results "$cases" --transport "$physics" --device cuda
        "$PYTHON" "$THESEUS_SOURCE/tests/mms/cns_matrix_study.py" plot \
          --source "$THESEUS_SOURCE" --results "$cases" --transport "$physics"
      done
    done
  done
done
```

Put the loop in a Bash script with `set -euo pipefail`, capturing its output inside
`$RESULTS_ROOT/logs`. It stops at a failed check; inspect that assessment instead
of assuming later groups were analyzed. Select only the desired regime/degree to
limit work. To plot existing measurements without reading fields, invoke only
`plot`; this does not change control status or imply a pass. Reanalysis rewrites
measurements/assessments/plots, not solver fields or logs. The final-time guard
permits bounded accumulation roundoff, capped at 1% of a timestep, while retaining
declared-step, wrong-time and missing-output checks.

## Collect a compact evidence archive

All logs now live inside the campaign tree. After execution/reanalysis:

```bash
tar -czf "$PWD/mms-regimes-results.tgz" \
  --exclude='*/solution' --exclude='*/launch' --exclude='*/cache' --exclude='*/tpl' \
  -C "$RESULTS_ROOT" .
```

The archive retains plans, inputs, run identities, solver/Slurm logs, measurements,
assessments and plots. Include partial/failed groups so coverage gaps remain visible.
Keep original fields until analysis is accepted. For a diagnostic requiring field
inspection, **omit the solution exclusion** and package only that study directory.
Symlinks archive as links by default: linking external logs into results does not
package their contents. Keep actual log files inside the campaign tree.

Raw identities/configs/logs contain real paths and possibly account information;
inspect/redact them before public sharing. Publish sanitized numerical evidence
as in the [verification guide](mms-verification.md). Keep disposable raw runs in
ValidationRuns; the user controls deletion. See the [evaluation guide](mms-evaluations.md)
for stage selection, explicit mesh lists, optional diagnostics and assessment details.
