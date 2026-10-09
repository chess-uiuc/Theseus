# PX chamber startup

This axisymmetric, swirl-free CNS example has two configurations:

- `config.json`: legacy CPG startup and regression coverage.
- `config-lte.json`: physical initial/exterior/profile states converted by the LTE
  air5 EOS, with temperature and transport visualization.

See the [case guide](../../../../docs/px-chamber.md) for geometry, boundaries, build
options, table ranges and limitations, and the [migration guide](../../../../docs/physical-state-migration.md)
for the physical input contract. These are short startup examples, not converged
plasma-chamber predictions.

Build with `AXISYMMETRIC=ON`, `SUBCELL_FV_BLENDING=ON`, and `THESEUS_WITH_PLATO=YES`
for LTE. From the repository root:

```sh
python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-cpg --ranks 2 --coarse

python3 TestCases/Axisymmetric/NavierStokes/PXChamber/run_case.py \
  --executable build-px/theseus --output run-px-lte --ranks 2 --coarse \
  --gas-model lte --database "$PLATO_DATABASE" --device cpu \
  --dt 1e-9 --final-time 2e-8
```

Use `--device cuda` with a compatible CUDA build. Omit `--coarse` for the supplied
75,375-cell mesh. The helper resolves paths and writes configuration, log and ParaView
files under the requested output directory. Choose a new directory for each run.
