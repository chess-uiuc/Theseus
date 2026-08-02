# Checkpoints and restarts

Theseus can periodically save its solution and resume from a saved cycle. The
checkpoint controls are entries in `runTime`:

```json
"checkpoint_save": true,
"checkpoint_dt": 0.02,
"checkpoints_folder": "Checkpoints",
"checkpoint_load": false,
"checkpoint_cycle": 200
```

- `checkpoint_save` enables checkpoint writing. It defaults to `false`.
- `checkpoint_dt` is the simulation-time interval between checkpoints and must
  be greater than zero when saving or loading is enabled. It defaults to
  `0.01`.
- `checkpoints_folder` is resolved beneath `output_file_path` and defaults to
  `Checkpoints`.
- `checkpoint_load` enables restart loading. It defaults to `false`.
- `checkpoint_cycle` selects the cycle to load and must be greater than zero
  when `checkpoint_load` is true.

Each cycle directory contains one state file per MPI rank and a JSON metadata
file. A checkpoint is complete when its metadata file is present; Theseus
writes that file only after every rank has finished writing its state.

## Restarting a run

Use the same mesh, polynomial order, dimension, equation count, and MPI rank
count used to create the checkpoint. Set `checkpoint_load` to `true`, set
`checkpoint_cycle` to the desired saved cycle, and leave
`checkpoints_folder` pointing at the checkpoint collection. `nsteps_max` is the
total cycle at which the run stops, not the number of additional steps. For
example, restarting cycle 200 with `nsteps_max` set to 300 advances 100 more
steps.

New checkpoints record the MPI and discretization layout. Theseus rejects a
restart when that metadata does not match the current run. Older checkpoints
containing only `time` and `cycle` remain readable, but Theseus warns that
their compatibility cannot be verified.

The run helper disables restart loading by default so smoke and regression
runs do not accidentally consume old files. Pass `-R` to preserve
`checkpoint_load` from the input configuration:

```sh
scripts/run_theseus.sh \
  -R \
  -b "$(pwd)/build" \
  -c TestCases/Euler/2D/DoubleMachReflection/config.json \
  -o ./RestartRun
```

The helper changes `output_file_path` to the selected run directory. Therefore
the checkpoint collection must exist beneath that run directory before a
restart. Copy or move the desired `Checkpoints` directory there when starting
from a separate output location.
