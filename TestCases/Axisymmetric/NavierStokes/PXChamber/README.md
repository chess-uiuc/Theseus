# PXChamber boundary configurations

Use `config-lte-local-characteristic.json` for the local HEGEL-style characteristic
LTE treatment, or `config-local-characteristic.json` for CPG. The original
`config.json` and `config-lte.json` retain the working legacy exterior/outflow case.
The new total inlet deliberately uses the documented frozen-property approximation;
LTE density and conserved energy still come from the actual EOS.

Set absolute mesh/profile/database paths and run with the normal harness:

```bash
scripts/run_theseus.sh -e /absolute/path/to/build-axis/theseus \
  -c /absolute/path/to/PXChamber/config-lte-local-characteristic.json \
  -o /absolute/path/to/new-run -p 4 -r cuda -P
```

Use `-r cpu` for CPU execution. Set dt, final_time and visualization cadence in
JSON; `-P` preserves those settings. No case-specific Python launcher is required.
Use `px_chamber_axi.msh` for the original full mesh, or explicitly select
`px_chamber_coarse.msh` for short verification runs. Build with AXISYMMETRIC=ON,
THESEUS_WITH_PLATO=YES and SUBCELL_FV_BLENDING=ON; device use needs compatible CUDA.

The [local boundary guide](../../../../docs/local-characteristic-boundaries.md)
contains equations, input examples, implementation/EOS details, distinct gaps from
HEGEL and physical correctness, and registered tests. The
[geometry/case guide](../../../../docs/px-chamber.md) describes the legacy case.
