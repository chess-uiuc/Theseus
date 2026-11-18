# MMS verification tools and independent references

This directory groups manufactured-solution source/gradient tests, independent
reference calculations, error integration, convergence preparation/assessment and
plots. General test helpers remain in `tests/`; `mms_paths.py` supplies the source
root and shared-test import path when these tools are executed directly.

Run tools by their full path, for example:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/mms/cns_matrix_study.py" prepare \
  --source "$THESEUS_SOURCE" --results "$RESULTS_ROOT/plan/cases" \
  --transport constant --degrees 2,3 --fluxes Roe --cells 12,18,27
```

Preparation writes inputs and a plan; it does not run the solver. Use the normal
recipes to execute cases through `run_theseus.sh`. Recipe IDs, executable target
names and output schemas are stable across this directory reorganization.

C++ source/gradient fixtures and Python helper tests are registered in the parent
`tests/CMakeLists.txt`, preserving CTest discovery. CUDA language properties remain
on the moved source files. The RHS probe executable remains at the build path
`tests/euler_rhs_probe`, even though its source is grouped here.

Independent references must not import the runtime manufactured formulas they
check. Runtime state/source headers are under `include/verification/mms/` and have
no dependency on this directory. See [the inventory](../../docs/mms-inventory.md),
[evaluation guide](../../docs/mms-evaluations.md) and [HPC guide](../../docs/mms-hpc.md).
