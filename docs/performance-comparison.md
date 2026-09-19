# Comparing performance between builds

`scripts/compare_performance.py` runs one identical Theseus case with two or
more existing build directories and compares both overall timestep performance
and the detailed runtime timers. It is intended for quick development-time
checks, not automated performance qualification.

## Prepare comparable builds

Configure every build with the same compiler, dependencies, optimization level,
and Theseus options. Detailed timers are required:

```bash
cmake -S /path/to/source -B /path/to/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/dependencies \
  -DENABLE_TIMERS=ON
cmake --build /path/to/build -j
```

For CUDA or HIP measurements, also use:

```text
-DENABLE_TIMER_SYNC_DEVICE=ON
```

For a run with more than one MPI rank, also use:

```text
-DENABLE_TIMER_BARRIER=ON
```

The comparison command reads each `CMakeCache.txt` and rejects builds whose
relevant options differ. The tool does not check out source revisions or create
builds; preparing the reference and candidate builds remains explicit.

## Run a comparison

Run the command from the Theseus repository root. Each `--build` has a unique
label and a build-directory path. The first build is the reference unless
`--reference` selects another label. Everything after `--` is passed to
`scripts/run_theseus.sh`.

```bash
python3 scripts/compare_performance.py \
  --build main=/path/to/main/build \
  --build candidate=/path/to/candidate/build \
  --reference main \
  --output tgv-p8-cfl-comparison \
  -- \
  -c TestCases/NavierStokes/3D/TaylorGreenVortex/config.json \
  -m /path/to/tgv.msh \
  -y 8 \
  -s 0.1 \
  -n 100 \
  -p 1 \
  -r cuda \
  -z
```

There is one execution per build, in command-line order. Select enough
timesteps with `-n` to make initialization and short-lived clock variation
small relative to the measured run.

The comparison tool controls `run_theseus.sh` options `-b`, `-e`, and `-o` and
rejects them in the arguments after `--`.

## Fixed-DT and fixed-CFL measurements

Use `-t` for a fixed timestep:

```text
-t 1e-12
```

Use `-s` for a fixed target CFL with a dynamically estimated timestep:

```text
-s 0.1
```

The options are mutually exclusive. `-s` enables variable timestepping even if
the original case configuration used a fixed timestep. Fixed-CFL logs include:

- `EstimateStabilityInitial`, the one-time initial estimate;
- `EstimateStability`, the recurring per-step estimate.

The dedicated timestep summary measures the ODE step itself. The enclosing
`Timestepping` timer also contains stability estimation and other loop overhead.

## User-supplied meshes

The `-m` option supports three forms:

- `-m replacement.msh` keeps the mesh directory from the case configuration;
- `-m meshes/replacement.msh` resolves the path from the launch directory;
- `-m /absolute/path/replacement.msh` uses the absolute path unchanged.

Absolute paths are recommended when comparing builds from different source
trees or when meshes live on platform scratch storage.

## Results

The output directory contains:

```text
comparison/
  logs/
    main.log
    candidate.log
  runs/
    main/
    candidate/
  results.json
  summary.md
```

`summary.md` reports the critical-rank mean timestep and the median duration and
invocation count for every detailed timer. `results.json` retains the raw timer
samples, aggregates, build metadata, and executed commands.

For multi-rank runs, construct comparisons use `[TIMER(all)]`, the maximum time
across ranks. Single-rank runs use `[TIMER(0)]`.

Detailed timers are nested. For example, `RHSMult`, `NSRHS`, and
`MultCNS_Volume` overlap. Their totals must not be added or interpreted as
independent percentages of the timestep. Compare the same timer name across
builds to identify where a change occurred.

Large percentage changes in extremely short timers may be operationally
irrelevant. Consider both the relative change and the absolute duration, and
consult the raw samples when a conclusion matters.

## Parse an existing log

The timer parser can also be used independently:

```bash
python3 scripts/performance_timers.py run.log -o run-timers.json
```
