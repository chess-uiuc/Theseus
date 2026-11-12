# Short unforced LTEEC channel transient

This case exercises periodic interior faces and no-slip, isothermal walls with the
Helmholtz LTE EOS, LTEEC volume/face flux and Lax face dissipation. It is a short
regression, not a sustained forced-channel calculation or a resolved turbulence
benchmark. No body force is applied.

The checked-in Gmsh mesh has 4×4×4 hexahedra, periodic in x and z. The box is
[0,4*pi] x [-1,1] x [0,4*pi/3]. BottomWall and TopWall are at 293.15 K. Order 2
and 50 fixed steps of 2.5e-6 produce outputs at cycles 0, 25, 50. Runtime alpha_max=0
keeps this a DG comparison even in builds with FV blending available.

ChannelFlowIC takes reference temperature and Mach parameters. It constructs a
parabolic streamwise velocity and deterministic, periodic perturbations. The
initial conservative energy uses gamma=1.4 and R=287.05, while the evolution uses LTE. Consequently its
reference temperature/Mach parameters do not prescribe exact LTE temperature/Mach.
The perturbations are reproducible across ranks; no random-number sequence is used.

## Run the case

Use your existing source/build/dependency paths and allocation. These commands
show the actual launcher and mesh selection; the recipe handles regression input
copies and uses the same launcher.

```bash
cd "$THESEUS_SOURCE"
scripts/run_theseus.sh -e "$THESEUS_BUILD/theseus" \
  -c TestCases/LTE/NavierStokes/ChannelFlowRegression/config.json \
  -m "$THESEUS_SOURCE/TestCases/LTE/NavierStokes/ChannelFlowRegression/channel_4x4x4.msh" \
  -r cpu -p 1 -n 50 -o "$CHANNEL_OUTPUT"
```

The Helmholtz path requires a PLATO installation with the entropy API. The config
uses its compiled database default; the regression recipe supplies the selected
prefix/database explicitly. Sutherland viscosity may be on or off. Conductivity
remains tabulated in either case, matching the numerical framework's convention.

## Golden regression

`golden-channel` is registered once in the Cartesian/full/integrated regression
suites. It uses 50 fixed steps, requires three outputs at matching times, finite
positive thermodynamic/transport fields, meaningful velocity evolution and mass
conservation. It compares cell-local fields and printed integral histories with
an explicit single-rank CPU reference. Two references cover tabulated viscosity
and Sutherland viscosity; the build cache selects the appropriate model.

Field comparisons use rtol=atol=1e-10; geometry uses absolute 1e-12. Printed energy
and kinetic-energy change histories use rtol=2e-6/atol=1e-12 to accommodate their
six-significant-digit formatting. Mass change must stay below 1e-11. A failed
comparison fails the test and never updates a reference. Goldens are behavioral
baselines, not independent mathematical accuracy references.

The launcher process group has a 540-second failure timeout, leaving margin for
small input preparation and comparison. The checker rejects run times over 600 s.
The step count is fixed, not chosen based on elapsed time during a test. Timing
measurements and baseline provenance are recorded with the golden data.

```bash
"$PYTHON" "$THESEUS_SOURCE/scripts/run_tests.py" run-test golden-channel \
  --source "$THESEUS_SOURCE" --results "$CHANNEL_TEST_OUTPUT" \
  --prefix "$TPL_PATH" --build-standard "$THESEUS_BUILD" \
  --python "$PYTHON" --device cpu --ranks 1
```

Use `--device cuda` in a GPU allocation to check the same CPU baseline. Multi-rank
outputs are compared using cell-local geometric identifiers, not partition order.


## Mesh regeneration

```bash
"$PYTHON" "$THESEUS_SOURCE/TestCases/LTE/NavierStokes/ChannelFlowRegression/make_mesh.py" \
  --output "$THESEUS_SOURCE/TestCases/LTE/NavierStokes/ChannelFlowRegression/channel_4x4x4.msh"
```

The generator needs only the Python standard library.
