# Validate Helmholtz LTE on CPU and CUDA

Use a CUDA-enabled Cartesian build to run both CPU and CUDA checks. The kernel
tests compare results with mathematical references. The Taylor–Green checks run
CPU and CUDA with the same executable and compare solution fields. The channel
regression compares either backend with stored CPU reference fields.

## Choose paths and prepare the build

Set these paths for your installation. Use a new output directory for each run:

```bash
export THESEUS_SOURCE=/absolute/path/to/Theseus
export THESEUS_BUILD=/absolute/path/to/cuda-build
export TPL_PATH=/absolute/path/to/tpl/install
export PYTHON=/absolute/path/to/python
export TEST_RESULTS=/absolute/path/to/new/test-results
```

The Python environment needs `ci-resources/requirements-ci.txt`. The dependency
prefix must contain CUDA-enabled MFEM, compatible MPI libraries, and PLATO with
the entropy API and database described in [Helmholtz LTE](helmholtz-eos.md).
Check Python before submitting a job:

```bash
"$PYTHON" -c 'import numpy, meshio, pyvista, vtk'
```

An existing Debug build with blending ON and Sutherland OFF is suitable. CUDA
architecture and compiler settings are machine-specific. For a new build, an
example configuration is:

```bash
cmake -S "$THESEUS_SOURCE" -B "$THESEUS_BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_PREFIX_PATH="$TPL_PATH" \
  -DPython3_EXECUTABLE="$PYTHON" \
  -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DAXISYMMETRIC=OFF -DBUILD_TESTING=ON \
  -DSUBCELL_FV_BLENDING=ON -DSUTHERLAND=OFF \
  -DTHESEUS_WITH_PLATO=ON
cmake --build "$THESEUS_BUILD" --parallel 4
```

Set `CUDA_ARCH` to the architecture required by your GPU before configuring.
Use the compiler/module environment associated with your dependency installation.
The build cache should contain `THESEUS_PLATO_HAS_ENTROPY:INTERNAL=1`.

## Run inside a GPU allocation

Keep your site's allocation directives and module setup. Simulation tests launch
through `scripts/run_theseus.sh`; see [platform launch selection](test-runner.md#platform-launch-selection).
The following checks require no array-debug build option or capture configuration.

### Kernel correctness

```bash
ctest --test-dir "$THESEUS_BUILD" \
  -R '^(Helmholtz.*|LTEECFlux_.*)_(CPU|CUDA)$' \
  --output-on-failure
```

These cover interpolation, EOS queries/recovery, gas conversion, flux/dissipation
and viscous-face assembly. Tests use analytic or high-precision references.
CUDA entries require an actual CUDA backend. A CPU pass alone provides no CUDA
execution coverage. Optional larger workloads are documented in
[kernel stress testing](kernel-stress-testing.md).

### CPU/CUDA flow agreement

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/lte_tgv_device_integration_test.py" \
  --source "$THESEUS_SOURCE" --executable "$THESEUS_BUILD/theseus" \
  --database "$TPL_PATH/database" --device cuda \
  --numerical-flux LTEECFlux --mixture air5

"$PYTHON" "$THESEUS_SOURCE/tests/lte_tgv_device_integration_test.py" \
  --source "$THESEUS_SOURCE" --executable "$THESEUS_BUILD/theseus" \
  --database "$TPL_PATH/database" --device cuda \
  --gas-model lte_helmholtz --numerical-flux Pirozzoli --mixture air5
```

Each command runs CPU and CUDA for 20 RK3 steps at fixed timestep, then repeats
at fixed CFL. It compares density, velocity, pressure and temperature at steps
0, 10 and 20 with `rtol=atol=1e-10`, checks matching physical times and backend
selection, and requires finite, positive thermodynamic fields and evolution.
It prints the simulation directories and comparison results. The two fluxes
are each checked across backends, not against one another.

### Short channel regression

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test golden-channel \
  --source "$THESEUS_SOURCE" --build-standard "$THESEUS_BUILD" \
  --prefix "$TPL_PATH" --python "$PYTHON" \
  --results "$TEST_RESULTS" --device cuda --ranks 1
```

This runs 50 fixed steps and compares three output frames with the CPU golden
for the build's viscosity setting. Runtime blending is zero for this case even
when blending support is compiled in. See the
[case guide](../TestCases/LTE/NavierStokes/ChannelFlow/README.md) for fields,
tolerances and runtime bounds.

## Interpret and retain results

A nonzero exit status indicates failure. In a batch script that must execute
all commands, use an `if command; then ...; else ...; fi` block for each and
accumulate a final failure status. See [running tests](test-runner.md) for a
central suite that collects failures automatically.

Retain batch output, CTest logs and the simulation/comparison directories printed
by the checkers. The channel comparison writes `comparison.json` under the recipe
output. Use the first failed assertion and its command/log to reproduce a failure.
Reference accuracy, CPU/CUDA agreement and flow-regression stability are separate
properties; passing these tests establishes their stated assertions for the
covered cases.
