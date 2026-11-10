# CPU/CUDA validation for Helmholtz LTEEC

Use one CUDA-enabled Cartesian build for both CPU and GPU execution. The kernel
tests check each backend against independent numerical references. The evolution
checker runs CPU and CUDA from the same executable and compares their fields.

Keep your site's existing compiler modules, Slurm allocation and launcher setup.
Run the test commands inside a GPU allocation. The commands below do not add a
launcher wrapper or change scheduler selection: simulation checks use the existing
`run_theseus.sh` harness.

## Source and dependencies

Use the supplied candidate source in a fresh directory. Do not unpack it over an
existing Theseus checkout. The complete candidate source is already patched; the
accompanying main-based patch is for review/integration, not a second application.
The channel investigation's research IC/source/debug additions remain separate.

Choose the source and CUDA-enabled TPL installation explicitly. From the extracted
`Theseus` directory:

```bash
export THESEUS_SOURCE="$PWD"
export THESEUS_BUILD="$THESEUS_SOURCE/build-helm"
export THESEUS_EXE="$THESEUS_BUILD/theseus"
export PYTHON=python3
export TPL_PATH=/absolute/path/to/your/cuda/tpl/install
export DBG_ROOT=/absolute/path/for/your/test-output
```

Retain your existing `TPL_PATH` and `DBG_ROOT` values when they already identify
the installation and output location you want. These are explicit choices, not
paths inferred by the test checker.

`PYTHON` must identify your HPC Python environment with the project test packages.
The Mac shared Python path is not an HPC requirement. Check the environment:

```bash
"$PYTHON" -c 'import sys, numpy, meshio, pyvista, vtk; print(sys.executable)'
```

The TPL prefix must contain CUDA-enabled MFEM, compatible MPI libraries, PLATO
with `plato_get_species_entropy` in its header and library, and `database/mixture`
and `database/thermo`. The full Python requirement list is
`ci-resources/requirements-ci.txt`. Use the compilers associated with that prefix.

## Configure and build once

This uses the reported CUDA architecture90 and point-parallel configuration.
Use a fresh build directory when changing TPL installations.

```bash
cmake -S "$THESEUS_SOURCE" -B "$THESEUS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_PREFIX_PATH="$TPL_PATH" \
  -DCMAKE_BUILD_RPATH="$TPL_PATH/lib" \
  -DCMAKE_INSTALL_RPATH="$TPL_PATH/lib" \
  -DPython3_EXECUTABLE="$(command -v "$PYTHON")" \
  -DENABLE_CUDA=YES \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DAXISYMMETRIC=OFF \
  -DNO_OPT=OFF \
  -DSUBCELL_FV_BLENDING=OFF \
  -DSUTHERLAND=ON \
  -DENABLE_POINT_PARALLEL_INTERIOR_FACES=YES \
  -DENABLE_POINT_PARALLEL_VOLUME=YES \
  -DTHESEUS_WITH_PLATO=YES

cmake --build "$THESEUS_BUILD" --parallel 4 --target \
  theseus \
  helmholtz_interpolation_tests helmholtz_eos_tests \
  helmholtz_gas_tests_standard helmholtz_gas_tests_sutherland \
  lteec_flux_tests_standard lteec_flux_tests_sutherland

grep '^THESEUS_PLATO_HAS_ENTROPY:' "$THESEUS_BUILD/CMakeCache.txt"
```

The final line should report `THESEUS_PLATO_HAS_ENTROPY:INTERNAL=1`. A missing API
causes Helmholtz selection to fail explicitly; legacy LTE is not substituted.
These canned tests do not need the channel forcing, array-capture build flag or
capture configuration files.

## Run the canned checks

Keep the working resource directives and module setup in your batch script. Set
`DBG_ROOT` explicitly to the output location you want to use; you may retain its
value from your existing channel batch script. The checkers create unique
subdirectories there and print the exact simulation/log locations.

```bash
mkdir -p "$DBG_ROOT"
cd "$DBG_ROOT"

ctest --test-dir "$THESEUS_BUILD" \
  -R '^(Helmholtz.*|LTEECFlux_.*)_(CPU|CUDA)$' \
  --output-on-failure
```

Expected: all selected CPU and CUDA tests pass. These include interpolation, EOS,
gas queries and the production viscous-face call chain. The flux tests cover
orders2/4/7 using small kernel fixtures, so they do not require an order7 channel
CPU simulation. CUDA tests reject failure to select the real CUDA backend.

Then run the paired LTEEC evolution test:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/lte_tgv_device_integration_test.py" \
  --source "$THESEUS_SOURCE" \
  --executable "$THESEUS_EXE" \
  --database "$TPL_PATH/database" \
  --device cuda \
  --numerical-flux LTEECFlux \
  --mixture air5
```

The checker automatically runs the CPU reference and CUDA candidate for20 RK3
steps with fixed DT, then repeats with fixed CFL. It checks positive finite
nonuniform evolving fields and compares density, velocity, pressure and temperature
at steps0/10/20. Tolerances are `rtol=atol=1e-10`; bitwise identity is not required.
It also verifies matching physical times and actual backend selection.

Expected final messages include:

```text
fixed-dt: CPU/cuda field comparison PASS
fixed-cfl: CPU/cuda field comparison PASS
```

Finally run the same-EOS Pirozzoli control:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/lte_tgv_device_integration_test.py" \
  --source "$THESEUS_SOURCE" \
  --executable "$THESEUS_EXE" \
  --database "$TPL_PATH/database" \
  --device cuda \
  --gas-model lte_helmholtz \
  --numerical-flux Pirozzoli \
  --mixture air5
```

This compares CPU and CUDA for Pirozzoli on the same Helmholtz framework. It does
not require the LTEEC and Pirozzoli solutions to match one another.

Use your existing batch stdout/stderr capture; no extra capture configs or log
naming scheme is required. Retain the printed test directories. If a check fails,
send the batch output and the simulation/comparison log identified by the checker;
keep its data files for inspection. A comparison failure is evidence to investigate,
not a reason to relax the tolerance. Passing these cases establishes agreement for
the tested fixtures/evolution; channel-specific reproduction follows separately.
