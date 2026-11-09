# LTEEC channel collection points

This consumer instrumentation requires the reusable toolkit from main; see
[array debugging](array-debugging.md) for the library API and capture format.
The following collection points and channel recipe belong to this branch.

Build with `-DTHESEUS_DEBUG_ARRAYS=ON` to enable host-side observation points.
Keep the optimization and CUDA architecture settings of the failing build. These
hooks do not add statements, captures, parameters or scratch to device lambdas.
The observer is compiled separately as host C++. With this option OFF, the solver
has no inspector member or hook calls. With it ON but `THESEUS_ARRAY_DEBUG` unset,
the observer does not read arrays, synchronize devices or create output files.

An active observer copies selected arrays into private host storage through
MFEM `Memory::CopyToHost`. It does not use `HostRead`, validate the source's host
copy, or invalidate its device copy. This matters when later kernels reuse an
already acquired device write pointer: inspecting the array must not cause a
later inspection to reuse stale host data. Observation still adds transfers,
host allocation, I/O and synchronization. It can change timing and hide a race;
unchanged kernel source is not a guarantee of identical generated instructions.
Preserve the original failing executable and verify the symptom with capture
both disabled and enabled. CUDA/HIP execution uses MFEM's normal memory/backend
ordering; independently managed streams require explicit ordering before capture.

## Build and test

Add the observer option to the configuration used for the failure, for example:

```sh
cmake -S /path/to/Theseus -B /path/to/cuda-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$TPL_PATH" \
  -DCMAKE_BUILD_RPATH="$TPL_PATH/lib" -DCMAKE_INSTALL_RPATH="$TPL_PATH/lib" \
  -DENABLE_CUDA=YES -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DSUBCELL_FV_BLENDING=OFF -DSUTHERLAND=ON \
  -DENABLE_POINT_PARALLEL_INTERIOR_FACES=YES \
  -DENABLE_POINT_PARALLEL_VOLUME=YES -DTHESEUS_WITH_PLATO=YES \
  -DSOURCE_TERMS=ON -DTHESEUS_DEBUG_ARRAYS=ON \
  -DPython3_EXECUTABLE="$PYTHON"
cmake --build /path/to/cuda-build --target theseus array_debug_fixture -j 2
ctest --test-dir /path/to/cuda-build -R '^ArrayDebug' --output-on-failure
```

CTest registers `ArrayDebugCPU` and, for CUDA builds, `ArrayDebugCUDA`. The latter
requires a CUDA-capable compute node. The fixture tests actual kernel-produced
values, a cached device pointer written again after observation, source validity
flags, nonfinite detection, filters, empty arrays, byte limits and snapshot
comparison. Run the shared suite through `scripts/run_tests.py` for broader
regression coverage; do not infer solver correctness from these toolkit tests.

## Capture configuration

Create a JSON file outside the source/build trees. Use a new absolute output
directory for each run, shared across that run's ranks. Set all ranks' environment
to the same configuration file. Example for a first-RHS comparison:

```json
{
  "output": "/absolute/results/channel-cpu",
  "tag": "channel Release first RHS",
  "calls": [0],
  "ranks": [0],
  "arrays": ["input", "restricted.volume", "rhs.*"],
  "snapshots": true,
  "max_bytes": 1073741824
}
```

Set `THESEUS_ARRAY_DEBUG=/absolute/capture.json` before launching the solver.
`calls` counts RHS evaluations starting at zero, not timesteps or RK stages.
For RK3SSP, calls 0, 1, 2 normally comprise the first step; time is also recorded.
Use call 0 initially so later trajectory differences do not obscure the origin.
`ranks` selects communicator-local MPI ranks; it does not gather them. Array
patterns are exact names or prefixes ending in `*`. Rank/call/array selection is
checked before readback. Unknown keys, invalid shapes and reuse of an existing
rank manifest are errors. More than one inspector/operator must use distinct
output directories.

`snapshots` defaults to false: summaries only, with no binary array files.
`max_bytes` limits total binary output per selected rank, across selected calls,
and defaults to 1 GiB. Exceeding it is an error, never a silently truncated
snapshot. Summary-only mode still copies/scans each complete selected array.
Selecting a prefix can capture several large arrays. Start with coarse RHS
arrays and narrow the next run to the first divergent phase. There are no
per-point device logging threads or host-to-device writes.

Each rank writes `rankNNNNNN.jsonl` and optional binary arrays. Manifest entries
contain dimension, equation count, polynomial order, gas, flux, time, MFEM backend
family, datatype/endianness, array axes/shape, finite min/max, negative/zero
counts, NaN/+Inf/-Inf counts and up to eight invalid indices. Begin/end markers
identify completed calls. Binary files contain native `mfem::real_t` values in
the recorded C-order layout. Save the exact input configuration, mesh,
CMakeCache, source diff and compiler/build log alongside the capture; `tag` is a
user annotation, not automatic provenance verification.

## Available observation points

| Array | Layout / meaning |
| --- | --- |
| `input` | equation, true DOF; conservative RHS input |
| `restricted.volume` | element, equation, point; restricted conservative input |
| `entropy` | equation, local DOF; entropy conversion output |
| `gradient.entropy.0` through `.2` | equation, local DOF; entropy gradients by physical direction |
| `gradient.primitive.0` through `.2` | equation, local DOF; converted gradients |
| `volume.inviscid` | element, equation, point; after point-parallel inviscid kernel, before sources/viscosity |
| `volume.viscous_flux` | element, reference direction, equation, point; cached flux before differentiation |
| `volume.jacobian`, `volume.metrics` | element/node geometric inputs; metrics also have reference/physical direction axes |
| `source.volume_average`, `source.target` | source-control cache slots, when source terms are compiled |
| `volume.residual` | element, equation, point; completed volume residual before scatter |
| `rhs.volume` | equation, local DOF; cumulative volume result |
| `interior.normals`, `interior.weight_minus`, `interior.weight_plus` | face/point geometry and scaling inputs |
| `interior.input` | face, side, equation, point; conservative face states |
| `interior.gradient.0` through `.2` | same face layout; primitive gradients |
| `interior.residual` | same face layout; complete face-kernel output before scatter |
| `interior.scattered` | equation, local DOF; scattered face contribution before accumulation |
| `rhs.interior` | equation, local DOF; cumulative volume + interior result |
| `boundary.normals`, `boundary.weights` | face/point geometry and scaling inputs |
| `boundary.input`, `boundary.gradient.0` through `.2`, `boundary.residual` | face, equation, point; boundary kernel inputs/output before scatter |
| `rhs.boundary` | equation, local DOF; cumulative volume + interior + boundary result |
| `rhs.final` | equation, true DOF; final RHS after transpose/essential-DOF handling |

Directions exist only for active dimensions. Boundary arrays are absent on ranks
without boundary faces. `volume.inviscid` and `volume.viscous_flux` require
point-parallel volume execution. The CNS hooks are the initial integration;
`input`, `restricted.volume` and `rhs.final` also surround Euler RHS calls.
A face residual combines inviscid and viscous work within one existing kernel;
these hooks cannot expose its internal temporary arrays without changing that
kernel. They can identify which existing kernel first produces a bad result.

## Run CPU and CUDA from one executable

Run these inside the site's allocation, from the source root containing
`scripts/`. Prepare CPU and CUDA JSON capture configurations with different output
directories and identical selections. Use an absolute case config whose database
path is valid on that node. The harness resolves the relative mesh path against
the case directory and owns MPI/scheduler/device launch selection.

```sh
cd "$THESEUS_SOURCE"
export THESEUS_ARRAY_DEBUG=/absolute/capture-cpu.json
bash scripts/run_theseus.sh -e "$THESEUS_EXE" -c "$CHANNEL_CONFIG" \
  -P -k -p 1 -r cpu -n 1 -o /absolute/results/cpu-run \
  > /absolute/results/cpu.log 2>&1
export THESEUS_ARRAY_DEBUG=/absolute/capture-cuda.json
bash scripts/run_theseus.sh -e "$THESEUS_EXE" -c "$CHANNEL_CONFIG" \
  -P -k -p 1 -r cuda -n 1 -o /absolute/results/cuda-run \
  > /absolute/results/cuda.log 2>&1
unset THESEUS_ARRAY_DEBUG
"$PYTHON" scripts/array_debug.py inspect /absolute/results/channel-cuda
"$PYTHON" scripts/array_debug.py compare \
  /absolute/results/channel-cpu /absolute/results/channel-cuda \
  --atol 1e-10 --rtol 1e-10 --json /absolute/results/comparison.json
```

The example tolerances are exploratory; inspect exact differences too by omitting
them. Comparisons use `abs(a-b) <= atol + rtol*max(abs(a),abs(b))` pointwise.
Matching NaNs or infinities are always failures. Bit differences are reported
separately, including signed zero. The tool processes snapshots in chunks through
read-only memory maps; it does not load both complete runs into RAM.

Exit status: 0 means finite agreement within requested tolerances; 1 means
differences/nonfinite values (or an incomplete capture for `inspect`); 2 means
invalid/missing/incompatible data. `compare` refuses partial calls, missing
selected calls/ranks/arrays, wrong file sizes, different layouts/context and
summary-only captures. `inspect` can show completed records from a failed partial
run, although a torn JSON line remains a parse error. `--array NAME` (repeatable)
and `--rank N` restrict comparison output after capture validation.

Use identical mesh ordering, rank count, decomposition and conservative inputs.
The tool compares storage indices, not physical coordinates, and does not match
DOFs across decompositions or align different meshes. Rank counts/context and
array shapes are checked; mesh/config equality must be established separately.
Check `input` first, including the channel's random initial perturbations. A
mismatched initial state invalidates a later CPU/GPU diagnosis. One rank and a
saved conservative checkpoint make an unambiguous starting point.

If input agrees and `rhs.volume` diverges, capture entropy, gradients,
`volume.inviscid`, `volume.viscous_flux` and `volume.residual`. If volume agrees
but `rhs.interior` diverges, capture interior inputs/gradients/residual/scattered
arrays. Matching kernel inputs with differing pre-scatter outputs localize the
problem inside that kernel. Matching pre-scatter output but differing scattered
output localizes it to scatter. These observations guide a focused generated-code
inspection without putting diagnostics inside the suspect point kernel.

## Add an observation point

Pass an owning MFEM vector and its actual storage layout on the host after the
producer's launch and before the consumer. For example:

```cpp
#ifdef THESEUS_DEBUG_ARRAYS
this->array_inspector.Record("my_phase.output", output,
  {"element", "equation", "point"}, {elements, equations, points});
#endif
```

For a standalone tool, construct `Theseus::Debug::ArrayInspector`, call
`BeginCall(comm, time, dim, neq, order, gas_name, flux_name)`, record any number of
vectors, then `EndCall()`. Link `theseus_array_debug`. Layout products must exactly
match vector size. Do not wrap an arbitrary CUDA pointer as a host vector; pass
the existing MFEM owner. Do not capture the inspector in `mfem::forall`, call it
from a device helper, or add it to the device cache.
