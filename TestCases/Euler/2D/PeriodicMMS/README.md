# Periodic CPG Euler manufactured solution

This smooth, forced Euler case verifies spatial convergence through the general
source framework. It uses CPG gamma=1.4 on the existing periodic square mesh
[-1,1]². There are no physical boundary conditions. With
`theta=2*pi*(x+y-0.4*t)`, prescribe:

| Primitive | Exact value |
| --- | --- |
| Density rho | `2 + 0.2*sin(theta)` |
| Velocity u | `0.7 + 0.1*sin(theta)` |
| Velocity v | `0.3 - 0.05*sin(theta)` |
| Pressure p | `1 + 0.15*sin(theta)` |

The conservative state is
`U=[rho, rho*u, rho*v, p/(gamma-1)+rho*(u*u+v*v)/2]`.
Set `S=U_t+F_x,x+F_y,y` using the ideal-gas Euler fluxes. The
`periodic-euler-mms` source evaluates these analytic derivatives in the device
kernel at each stage. `PeriodicEulerMMS` initializes the conservative state at
t=0 and supplies the exact solution for the solver's physical L1/L2 error norms.
The gas-model, dimension, gamma and geometry restrictions are validated at startup.

For the derivative calculation, let primes denote differentiation with respect
to theta and `w=u+v`. Then:

- `S_rho = 2*pi*(-0.4*rho' + (rho*w)')`;
- `S_mx = 2*pi*(-0.4*(rho*u)' + (rho*u*w)' + p')`;
- `S_my = 2*pi*(-0.4*(rho*v)' + (rho*v*w)' + p')`;
- `S_E = 2*pi*(-0.4*E' + ((E+p)*w)')`.

These are physical rates without extra mesh weights. The forcing is nonzero in
all four equations. Unit fixtures check it against independent fourth-order
finite differences of conservative state and ideal-gas physical fluxes.

## Run and assess

Run the checked-in medium mesh case with existing build paths:

```bash
bash "$THESEUS_SOURCE/scripts/run_theseus.sh" \
  -e "$THESEUS_BUILD/theseus" \
  -c "$THESEUS_SOURCE/TestCases/Euler/2D/PeriodicMMS/config.json" \
  -r cpu -p 1 -o "$MMS_OUTPUT" -P -R -k
```

The mesh path is relative to this config; the shared harness resolves it. The
case uses order 2, RK4, fixed dt=0.001, t_final=0.2, and disables FV blending to
measure the smooth DG discretization. The periodic base mesh has 3×3 elements;
refinement levels 1/2/3 produce 6×6, 12×12 and 24×24 grids.

The complete registered convergence check is:

```bash
"$PYTHON" "$THESEUS_SOURCE/tests/euler_mms_integration_test.py" \
  --source "$THESEUS_SOURCE" --executable "$THESEUS_BUILD/theseus" --device cpu
```

Run from an output directory where retained results should be created. The checker
uses the standard harness for every solver launch. Use `--device cuda` inside a GPU
allocation to add CPU/CUDA field comparison.

The checker measures the solver's physical conservative L2 error and independently
reconstructs all four conservative fields from output using separate Python
primitive formulas. It checks initial nodal values, decreasing error and approach
to P2's expected third-order accuracy: refinement orders must lie between 2.5 and
3.5 for total L2 and for the final nodal component refinement. It also requires:

- Less than 1% change in finest-grid L2 error when dt is halved.
- A source-off control error at least five times larger than the forced error.
- Serial/two-rank fields matching at `rtol=atol=1e-10`.
- CPU/CUDA fields matching at the same tolerance when CUDA is selected.

`measurements.json` and `convergence.json` retain norms, orders, timestep and rank
settings alongside all configs, logs and fields. This is one bounded inviscid,
periodic verification case, not verification of viscous terms, physical boundaries
or LTE models. Separate temporal source-infrastructure tests cover RK-stage time
and coordinate packing.
