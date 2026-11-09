# Host-side array debugging toolkit

The toolkit inspects MFEM arrays on the host between kernel launches. It provides
nonfinite/range summaries, optional binary snapshots and an offline comparator.
It has no dependency on a particular gas model, flux or case, and does not require
PLATO. Main supplies the library, tools and tests; it does not add collection
calls to solver algorithms. A consumer adds those calls at the boundaries it
needs. Case-specific collection belongs with that case's investigation.

## Build and use the library

`theseus_array_debug` is a host C++ library, also in CUDA builds. Standalone tools
can link it directly. `-DTHESEUS_DEBUG_ARRAYS=ON` links it to `theseus` and defines
`THESEUS_DEBUG_ARRAYS` for optional consumer hooks. This switch alone captures
nothing: the caller must create an inspector and call its methods. Keep the
optimization settings of the failing build; this toolkit does not change them.

Own one inspector per observed operator/run, outside device caches and device
lambdas. After MPI initialization, surround a logical evaluation with:

```cpp
#include "ArrayDebug.hpp"

// Persistent host-side object, for example an operator member.
Theseus::Debug::ArrayInspector inspector;

// Host statements between existing kernel launches:
inspector.BeginCall(comm, time, dim, neq, order, gas_name, flux_name);
inspector.Record("input", input, {"equation", "dof"}, {neq, input.Size() / neq});
// Launch the existing producer, without changing its body.
inspector.Record("residual", residual,
                 {"element", "equation", "point"}, {elements, neq, points});
inspector.EndCall();
```

Link the caller against `theseus_array_debug`. Optional hooks may be guarded by
`#ifdef THESEUS_DEBUG_ARRAYS`. Pass owning MFEM vectors and their actual C-order
storage layouts. Shape products must equal vector sizes. Do not wrap arbitrary
CUDA pointers as host vectors, put the inspector in a device cache, or capture it
in `mfem::forall`. Keep observation outside producer/consumer kernel bodies.
No kernel instrumentation or solver captures are installed by the toolkit itself.

## Runtime configuration

Set `THESEUS_ARRAY_DEBUG` to an absolute JSON configuration path. When unset, the
inspector does not read arrays, synchronize devices or write output. Example:

```json
{
  "output": "/absolute/results/cpu",
  "tag": "optimized first evaluation",
  "calls": [0],
  "ranks": [0],
  "arrays": ["input", "residual"],
  "snapshots": true,
  "max_bytes": 1073741824
}
```

Calls are zero-based `BeginCall` invocations, not implicitly timesteps or RK
stages. Ranks are communicator-local. Patterns are exact array names or prefixes
ending in `*`. Unselected ranks/calls/arrays do not perform readback. Unknown
keys, invalid selections/layouts and existing rank manifests cause errors. Use a
fresh output directory per run, and distinct directories for multiple inspectors.

`snapshots` defaults to false: summaries only. `max_bytes` defaults to 1 GiB of
binary output per selected rank across all calls. Exceeding it causes an error,
never a silently truncated snapshot. Summary mode still copies/scans each full
selected vector. Start with a few coarse boundaries and narrow the next capture
around the first divergence. Large runs may need a deliberately increased budget.

Each rank writes `rankNNNNNN.jsonl`, plus optional native `mfem::real_t` binary
files. Records include time, dimension, equation count, order, gas/flux labels,
backend family, datatype/endianness, axes/shape, finite min/max, negative/zero
counts, NaN/+Inf/-Inf counts and up to eight invalid indices. Begin/end markers
identify completed calls. Labels need not describe a gas model; they identify
consumer context for compatibility checks. Save mesh/input hashes, source diff,
CMakeCache and compiler/dependency provenance alongside the capture. The `tag`
annotation does not establish that provenance automatically.

## Inspect and compare

Use an interpreter with NumPy, for example the project's prescribed Python:

```sh
"$PYTHON" scripts/array_debug.py inspect /absolute/results/cuda
"$PYTHON" scripts/array_debug.py compare \
  /absolute/results/cpu /absolute/results/cuda \
  --atol 1e-10 --rtol 1e-10 --json /absolute/results/comparison.json
```

Default tolerances are zero. Choose application-appropriate tolerances;
1e-10 above is an exploratory example, not a solver acceptance requirement.
Comparisons use `abs(a-b) <= atol + rtol*max(abs(a),abs(b))` pointwise. Matching
NaNs or infinities always fail. Bit differences are reported separately,
including signed zero. Read-only memory maps and chunking bound comparison
working memory. The result identifies the first differing selected array in
capture order, logical coordinates and both values. `--array NAME` (repeatable)
and `--rank N` filter comparison after validating capture completeness.

Exit codes: 0 finite agreement within tolerances; 1 differences/nonfinite data
(or incomplete `inspect` capture); 2 invalid/missing/incompatible data. Compare
rejects partial calls, missing selected calls/ranks/arrays, different layouts or
context, wrong file sizes and summary-only captures. Inspect can show completed
records from a failed partial run; a torn JSON line is a parse error.

Use identical stored inputs and indexing. This compares storage indices, not
physical coordinates; it does not align different meshes or decompositions.
Rank counts/context/shapes are checked, but mesh/config equality must be
established separately. Device families are recorded and intentionally allowed
to differ; check actual backends when diagnosing CPU/device agreement. Comparing
two CPU captures is also useful and is supported.

## Readback behavior and limits

The observer uses const `Memory::CopyToHost` into private host storage. It does
not validate the source's own host copy or invalidate its device copy. This
preserves source validity flags when later kernels reuse a previously acquired
device write pointer. Using `Vector::HostRead()` at those boundaries could leave
a host-valid copy that later writes through cached pointers cannot invalidate.

Active observation still allocates host storage, transfers data, synchronizes
and writes files. It can alter timing and mask races; unchanged kernel bodies
do not guarantee identical generated machine code. Preserve the failing binary
and verify that the failure persists with observation enabled. Independently
managed streams require explicit ordering before observation. The observer sees
completed, exposed arrays; it cannot inspect internal local arrays in a fused
kernel or catch an invalid access before the producer finishes.

## Tests

CTest registers `ArrayDebugCPU`, and `ArrayDebugCUDA` when CUDA is enabled.
The fixture exercises actual kernel-produced values, repeated writes through a
cached device pointer after capture, source memory-validity flags, nonfinite
values, filters, empty arrays, byte limits and comparator failure handling.
It has no PLATO dependency. Run on the appropriate compute node:

```sh
cmake --build "$BUILD" --target array_debug_fixture -j 2
ctest --test-dir "$BUILD" -R '^ArrayDebug' --output-on-failure
```

The shared suite discovers these registered tests. They validate the toolkit,
not the consumer's physics, numerical flux or collection-point choices.
