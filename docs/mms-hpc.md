# Running the full MMS matrix on HPC

Build two optimized CUDA executables, then submit **90 independent jobs, seven
cases per job**. Together these produce 630 simulations. Each job covers one
physics configuration, one degree and one flux on 12/18/27/40/60/90 cells per
direction, followed by the finest half-timestep control. No full regression suite
or dissipation-source changes are part of this campaign.

This is a **Slurm template**. Replace paths, account, partitions, GPU resource
syntax and wall limits for your site. Load the site's matching compiler, MPI,
CUDA and dependency modules before submission; repeat those module commands in
both batch scripts if your site does not preserve the environment. The dependency
prefix must contain CUDA-enabled MFEM and the dependencies required by Theseus.
Use a fixed source checkout and dependency installation throughout the campaign.

## Set the paths once

Run from a campaign directory on shared storage, visible on compute nodes:

```bash
export THESEUS_SOURCE=/absolute/path/to/Theseus
export TPL_PATH=/absolute/path/to/cuda-dependencies/install
export PYTHON=/absolute/path/to/python-environment/bin/python
export BUILD_ROOT=/absolute/path/to/ValidationRuns/mms/gpu-builds
export RESULTS_ROOT=/absolute/path/to/ValidationRuns/mms/hpc-matrix-001
export MMS_REGIME=transonic  # Or subsonic / supersonic; use a separate RESULTS_ROOT.
export CUDA_ARCH=80  # Replace with your target GPU architecture.
export ACCOUNT=your_account
export BUILD_PARTITION=your_build_partition
export GPU_PARTITION=your_gpu_partition
mkdir -p logs "$BUILD_ROOT" "$RESULTS_ROOT"
"$PYTHON" -c 'import numpy, meshio, pyvista, vtk, matplotlib'
```

Python packages are listed in [requirements-ci.txt](../ci-resources/requirements-ci.txt).
Existing CUDA builds may be reused if their source, transport switches and
dependency/toolchain settings match the intended campaign. Analysis-only script
changes do not require a solver rebuild.

## Build job: two executables

Save as `build-mms.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mms-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
# Load your site's compiler/MPI/CUDA modules here if needed.
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

The **constant build serves Euler and constant CNS**. The Sutherland build serves
Sutherland CNS. Use fresh build directories if their dependency/toolchain settings
have changed. The build partition needs access to the CUDA compiler; it need not
allocate a GPU unless required by the site.

## Matrix array: one GPU and seven cases per task

Save as `run-mms.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mms-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.log
set -euo pipefail
# Load your site's compiler/MPI/CUDA modules here if needed.
: "${THESEUS_SOURCE:?}" "${TPL_PATH:?}" "${PYTHON:?}"
: "${BUILD_ROOT:?}" "${RESULTS_ROOT:?}" "${SLURM_ARRAY_TASK_ID:?}"
export OMP_NUM_THREADS=1
# Never carry the workstation localhost launcher into an HPC allocation.
# Use the site's normal Theseus harness setup and GPU visibility.
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
output="$RESULTS_ROOT/${physics[$model]}/p$degree/$flux"
export MPLCONFIGDIR="$RESULTS_ROOT/cache/${SLURM_ARRAY_JOB_ID}_${id}/matplotlib"
export XDG_CACHE_HOME="$RESULTS_ROOT/cache/${SLURM_ARRAY_JOB_ID}_${id}"
printf 'Physics=%s degree=%s flux=%s output=%s\n' \
  "${physics[$model]}" "$degree" "$flux" "$output"
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test "${recipes[$model]}" \
  --source "$THESEUS_SOURCE" --build-standard "$build" --prefix "$TPL_PATH" \
  --python "$PYTHON" --device cuda --ranks 1 \
  --degrees "$degree" --fluxes "$flux" --cells 12,18,27,40,60,90 \
  --regime "${MMS_REGIME:-transonic}" --reference-dt .001 --final-time .2 --timeout 3300 --results "$output"
```

The one-hour wall limit and 55-minute recipe timeout are initial requests,
**not guaranteed GPU runtime estimates**. Adjust them using pilot and high-degree
timings on the target machine. Memory is also a starting request to adjust for your site.
The harness owns `srun`/MPI and GPU placement; do not wrap the Python runner in an
extra `srun`. One rank uses the one GPU assigned to this job.

| Array IDs | Physics | Build | Tasks / simulations |
| --- | --- | --- | --- |
| 0–29 | Euler | constant | 30 / 210 |
| 30–59 | Constant CNS | constant | 30 / 210 |
| 60–89 | Sutherland CNS | sutherland | 30 / 210 |

Within each block: P1 through P5, each with fluxes in the order listed in the
script. `lfr` is the runtime name for LLF.

## Submit, inspect, then expand

```bash
bash -n build-mms.sbatch run-mms.sbatch
build_job=$(sbatch --parsable --account="$ACCOUNT" \
  --partition="$BUILD_PARTITION" build-mms.sbatch)
build_job=${build_job%%;*}
# Three pilot jobs: P3/Roe for each physics configuration.
printf 'Build job ID: %s\n' "$build_job"
sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:${build_job:?Build job ID is required}" --array=13,43,73%3 run-mms.sbatch
```

Inspect pilot logs, GPU execution and `cases/matrix-assessment.json` before
expanding. The pilot jobs use the full six-mesh sequence. They are part of the
campaign, so do not submit them again into their existing result directories.
Once the builds and pilots succeed:

```bash
# Remaining 87 tasks, at most four GPUs/jobs active simultaneously.
sbatch --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --array=0-12,14-42,44-72,74-89%4 run-mms.sbatch
```

Change `%4` to your desired concurrency. If reusing existing verified builds, skip
the build submission and omit the pilot dependency. A direct full campaign uses
`--array=0-89%4` **instead of** the pilot/remaining split, with a fresh result root.

Each task retains configs, logs, fields, measured errors, assessments and plots.
Inspect both timestep controls and order flags: a successful job does not by itself
establish nominal order. Matrix surveys retain order shortfalls for review.

**There is no automatic case-level resume.** A completed job's data remain intact.
To retry a failed or interrupted task, retain its old result directory under a
separate name before resubmitting just its array ID; all seven cases will rerun.
No solver rebuild is needed between matrix tasks. Keep raw results under your
central ValidationRuns tree and retain compact plots/tables elsewhere before
manually deleting any runs.

See [evaluation details](mms-evaluations.md#configurable-cns-degree-and-refinement-matrices)
for preparation/analysis stages, timestep policy and alternative selectors.

## Reanalyze saved fields without an allocation

Analysis does not launch Theseus or require a GPU. On a host where field
postprocessing is permitted, process completed groups sequentially with the same
Python environment. Keep the original fields and build/run metadata in place.
The `--device cuda` argument checks the recorded solver backend; it does not
request analysis hardware. Use `cpu` for fields from CPU simulations.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR="$RESULTS_ROOT/cache/analysis/matplotlib"
export XDG_CACHE_HOME="$RESULTS_ROOT/cache/analysis"
for physics in euler constant sutherland; do
  for degree in 1 2 3 4 5; do
    for flux in lfr Roe HLL Chandrashekar Ranocha Pirozzoli; do
      cases="$RESULTS_ROOT/$physics/p$degree/$flux/cases"
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
```

Put the loop in a Bash script with `set -euo pipefail` and capture its output to a
log. To process only P5, change the degree loop to `for degree in 5`. To regenerate
plots from measurements without reading fields, invoke only `plot`. A failed check
must be resolved; plots alone do not establish an assessment pass. Analysis rewrites
measurements, assessments and plots, not solver fields or run logs.

The checker permits bounded accumulation roundoff in final timestamps, capped at
1% of a timestep, and checks declared step counts. This accommodates long
fixed-step integrations while retaining wrong-time and missing-step rejection.

## Collect a compact evidence archive

Run after the campaign and any reanalysis complete:

```bash
tar -czf mms-full-matrix.tar.gz \
  --exclude='*/solution' --exclude='*/launch' --exclude='./cache' \
  -C "$RESULTS_ROOT" .
# Run from the submission directory; this packages the campaign's Slurm logs.
tar -czf mms-scheduler-logs.tar.gz logs/
```

Measurements, assessments, plans, configs, solver logs and plots remain in the
compact archive. Include partial/failed groups so missing coverage is visible.
Raw fields are excluded and must remain available until the analysis is accepted.
The archived run identities, configs and logs contain actual paths, build details
and possibly scheduler/account information; inspect or redact them before public
sharing. Keep portable public evidence to numerical tables, plots and sanitized
configuration metadata, as in the [verification guide](mms-verification.md).
