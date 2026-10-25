# Checkpoints and restarts

Checkpoints save the conservative solution so a simulation can resume from a saved
cycle. Configure checkpoint output in `runTime`:

```json
"checkpoint_save": true,
"checkpoint_dt": 0.02,
"checkpoints_folder": "Checkpoints",
"checkpoint_load": false,
"checkpoint_cycle": 200
```

| Setting | Meaning | Default |
| --- | --- | --- |
| `checkpoint_save` | Write checkpoints during the simulation and at its completion. | `false` |
| `checkpoint_dt` | Physical-time interval between checkpoints; must be positive when saving or loading is enabled. | `0.01` |
| `checkpoints_folder` | Checkpoint directory, resolved beneath `output_file_path` for a relative path. | `Checkpoints` |
| `checkpoint_load` | Initialize the solution from a checkpoint. | `false` |
| `checkpoint_cycle` | Positive cycle number to load when restarting through the JSON settings. | `0` |

Checkpoint times participate in timestep event scheduling. A step can be shortened
to reach a checkpoint time; the nominal fixed timestep or CFL policy still
controls subsequent steps.

Each `CycleN` directory contains one solution file per MPI rank and a JSON metadata
file. The metadata file is written after every rank finishes writing its solution.

## Select a checkpoint

From the source directory, use `-L` to select the highest-numbered complete
checkpoint in the case output directory:

```sh
scripts/run_theseus.sh -L -b build -c path/to/Case/config.json -o ExistingRun -p 2
```

Use `-C 200` to select a specific cycle:

```sh
scripts/run_theseus.sh -C 200 -b build -c path/to/Case/config.json -o ExistingRun -p 2
```

Both options enable loading and set `checkpoint_cycle` in the patched input file.
The checkpoint location is normally `ExistingRun/Case/Checkpoints`, where `Case`
is the directory containing the input configuration. With `-P`, path resolution
uses the preserved `output_file_path`. Relative output paths are resolved from
the case work directory.

Selection requires readable cycle metadata and all expected nonempty rank files.
Recorded rank count, polynomial order, dimension and equation count must match
when available. `-L` checks candidates in descending cycle order and skips those
that do not qualify. It uses cycle numbers, not file modification times. `-C`
fails if the requested cycle does not qualify. If selection fails, Theseus is not
launched. `-L` and `-C` cannot be combined with each other or with `-w`.

To select a cycle in JSON, set `checkpoint_load=true` and `checkpoint_cycle=N`,
then pass `-R` to the launcher. `-R` preserves the configured loading flag.

## Continue the simulation

Use the same mesh, polynomial order, dimension, equation count and MPI rank count
as the saved run. Keep gas physics and other model settings consistent with the
saved state. Solver validation checks the recorded state and discretization
metadata, but does not establish gas-model or mesh-file identity.

`nsteps_max` is the total stopping cycle, not a count of additional steps. For
example, restarting cycle 200 with `nsteps_max=300` advances 100 more steps. Set
`final_time` beyond the saved simulation time as well.

Checkpoint metadata identifies physical conservative state `U` for Cartesian and
axisymmetric geometry. Axisymmetric loading requires explicit conservative-state
and geometry metadata. Cartesian metadata containing only time and cycle is
accepted with a warning that compatibility cannot be verified.

## ParaView output across restarts

Restarting in the same output directory retains collection entries earlier than
the restart time in `ParaView/ParaView.pvd`. The restarted cycle is written once,
and subsequent visualization outputs extend that collection.

Restarting from an earlier checkpoint replaces collection entries at and after
that checkpoint time with the resumed trajectory. Existing later cycle directories
can remain on disk without being referenced by the collection.

To retain visualization history in a separate output directory, copy the complete
`ParaView` directory, including its `.pvd` manifest and referenced cycle files,
along with the required checkpoints. Checkpoint files alone do not contain the
visualization collection history.

## Development checks

The launcher tests cover cycle ordering, incomplete checkpoints, metadata/layout
checks, JSON overrides, path resolution and protection against combining selection
with directory wiping. Cartesian and axisymmetric restart integration tests check
per-rank solution files, final visualization data and collection history against
an uninterrupted simulation. They exercise JSON, latest-complete and prescribed
cycle selection through the launcher.
