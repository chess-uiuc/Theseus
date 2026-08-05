# Axisymmetric Supersonic Flow Over a Cone

This case models Mach 2 inviscid flow over a cone with a $10^\circ$ half-angle.
The mesh is the meridional $(z,r)$ half-plane. Revolving the slip wall around
$r=0$ produces the three-dimensional cone.

The case is intended as a visual and quantitative demonstration of the
axisymmetric Euler implementation. It produces an attached conical shock and
exercises nonzero cylindrical source terms, the axis boundary, and an oblique
slip wall.

## Generate the mesh

The checked-in Gmsh 2.2 mesh is generated without external dependencies:

```sh
cd TestCases/Axisymmetric/Euler/FlowOverCone
python3 generate_cone_multiblock.py
```

The generator creates conforming upstream and body-fitted cone blocks, checks
every quadrilateral for positive area, and reports mesh-quality statistics.

## Run

From the repository root, run on CUDA with an axisymmetric build:

```sh
scripts/run_theseus.sh -b /path/to/axisymmetric-build \
  -c TestCases/Axisymmetric/Euler/FlowOverCone/config.json \
  -r cuda -p 1
```

The default run uses order 2, a fixed timestep of $5\times10^{-4}$, and final
time $t=1.5$. The run harness writes output to `RunTheseus/FlowOverCone`. A
shorter run can show shock formation, but the verification metrics should be
taken from a solution whose reported values have stopped changing appreciably.

## Verification metrics

The postprocessor reads the final ParaView dataset, fits the shock location,
samples the cone-surface state, and independently solves the Taylor--Maccoll
equations for the configured freestream and cone angle:

```sh
python3 TestCases/Axisymmetric/Euler/FlowOverCone/cone_metrics.py \
  RunTheseus/FlowOverCone/ParaView/ParaView.pvd \
  --check --json cone_metrics.json
```

The script requires NumPy. It can use PyVista when available and otherwise
reads Theseus's inline-binary VTK XML output directly. It reports:

- shock angle $\beta$ from a least-squares fit to the strongest pressure
  gradient at a sequence of axial stations;
- mean surface Mach number and its axial standard deviation;
- mean surface pressure ratio $p_s/p_\infty$ and its standard deviation;
- surface pressure coefficient
  $$
  C_p=\frac{p_s-p_\infty}{\tfrac12\rho_\infty U_\infty^2};
  $$
- differences between the numerical and Taylor--Maccoll values.

With `--check`, the script also applies demonstration-level acceptance limits:

- shock angle within $1^\circ$;
- surface Mach number within 0.05;
- surface pressure ratio within 0.02;
- surface Mach standard deviation below 0.02;
- surface pressure-ratio standard deviation below 0.01.

These limits detect a missing, detached, or badly resolved conical-flow
response. They are regression criteria for this mesh and run duration, not a
replacement for a mesh/order convergence study.

For a useful demonstration, the pressure and Mach contours should show a
straight attached shock and a nearly uniform conical region between the shock
and wall. The fitted shock angle should be close to the Taylor--Maccoll value,
and surface-state variation should decrease as the solution approaches steady
conical flow and the mesh is refined.

This case is not registered as a short CI test. Quantitative tolerances should
be established through a mesh/order convergence study before it is promoted to
an automated validation test.
