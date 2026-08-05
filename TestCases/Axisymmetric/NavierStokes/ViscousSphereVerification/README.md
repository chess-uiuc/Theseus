# Axisymmetric Viscous Flow Over a Sphere

This case is a quantitative, body-fitted demonstration of steady laminar flow
over a sphere. The meridional $(z,r)$ mesh represents a three-dimensional
sphere when revolved about $r=0$. The nondimensional freestream and transport
properties are

$$
\rho_\infty=1,\qquad a_\infty=1,\qquad U_\infty=0.3,
\qquad R=1,\qquad \mu=0.006,
$$

which give

$$
M_\infty=0.3,\qquad
Re_D=\frac{\rho_\infty U_\infty(2R)}{\mu}=100.
$$

At this Reynolds number the expected axisymmetric flow contains a steady
separated recirculation region behind the sphere. The case is separate from
the short `ViscousFlowOverSphere` CI smoke test: it uses a substantially larger
domain, a resolved sphere-normal boundary-layer collar, a longer integration,
and quantitative wake and force measurements.

## Mesh

The checked-in Gmsh 2.2 mesh was made with the repository-level multiblock
generator:

```sh
python3 generate_sphere_multiblock.py \
  --output TestCases/Axisymmetric/NavierStokes/ViscousSphereVerification/ViscousSphereVerification.msh \
  --radius 1 --center-z 0 --zmin -10 --zmax 40 --rmax 10 \
  --nq 24 --nbl 8 --ntransition 16 --ntop 24 --nleft 24 --nright 72 \
  --first-layer 0.03 --collar-thickness 0.5 \
  --transition-exponent 0.8 --outer-exponent 0.8
```

All dimensional mesh arguments are scaled consistently with $R=1$. The mesh
has 96 cells around the sphere, 8 exact sphere-normal collar layers followed
by 16 transition layers, and outer boundaries at $z=-10R$, $z=40R$, and
$r=10R$, measured from the sphere center.

## Build and run

Because viscosity and axisymmetry are compile-time features in this version,
configure the executable from the repository root with this case:

```sh
cmake -S . -B build-viscous-sphere \
  -DCONFIG_FILE=TestCases/Axisymmetric/NavierStokes/ViscousSphereVerification/config.json \
  -DAXISYMMETRIC=ON \
  -DCMAKE_CXX_FLAGS=-DPARABOLIC \
  -DCMAKE_CUDA_FLAGS=-DPARABOLIC \
  -DMFEM_DIR=/path/to/mfem/lib/cmake/mfem
cmake --build build-viscous-sphere -j
```

For a CUDA build, also supply the project and MFEM CUDA options appropriate to
the target system, including `-DENABLE_CUDA=ON` and its CUDA architecture.

Then run, for example, on one CUDA device:

```sh
scripts/run_theseus.sh -b build-viscous-sphere \
  -c TestCases/Axisymmetric/NavierStokes/ViscousSphereVerification/config.json \
  -r cuda -p 1
```

The default order is 2. A variable timestep at CFL 0.3 accounts for both the
advective and viscous stability limits. The default final time is $t=40$, or
$tU_\infty/D=6$ convective times. Before interpreting the steady metrics,
confirm that successive saved states no longer show a material trend; extend
`final_time` when necessary.

Theseus does not yet provide subsonic characteristic farfield conditions. The
fixed freestream inlet and radial farfield and extrapolating outlet are placed
far from the body to reduce their influence. A mesh/order/domain convergence
study remains necessary for formal validation.

## Quantitative checks

The postprocessor reads the final ParaView dataset and reports:

- recirculation length from the downstream-axis zero crossing of $u_z$;
- separation angle from the sign change of near-wall tangential velocity;
- pressure, viscous, and total drag coefficients using the axisymmetric surface
  measure $dS=2\pi R^2\sin\theta\,d\theta$;
- maximum radial velocity on the symmetry axis;
- changes in the principal metrics between the last two saved states.

Run it from the repository root:

```sh
python3 TestCases/Axisymmetric/NavierStokes/ViscousSphereVerification/sphere_metrics.py \
  RunTheseus/ViscousSphereVerification/ParaView/ParaView.pvd \
  --check --json sphere_metrics.json
```

The demonstration checks use deliberately broad published-flow ranges at
$Re_D=100$: a separated steady wake, separation between $110^\circ$ and
$140^\circ$ from the upstream stagnation point, and total drag coefficient
between 0.9 and 1.3. They also require small radial velocity on the axis and
limited change between the last two outputs. These criteria are intended to
catch a missing or grossly incorrect axisymmetric viscous response; they are
not a substitute for a refinement study.

Useful primary comparisons are [Pruppacher, Le Clair, and Hamielec,
“Some relations between drag and flow pattern of viscous flow past a sphere
and a cylinder at low and intermediate Reynolds numbers,” *Journal of Fluid
Mechanics* 44 (1970)](https://doi.org/10.1017/S0022112070002148), and
[Johnson and Patel, “Flow past a sphere up to a Reynolds number of 300,”
*Journal of Fluid Mechanics* 378
(1999)](https://doi.org/10.1017/S0022112098003206).
