# Axisymmetric formulation

Theseus implements a two-dimensional axisymmetric formulation for
Euler and compressible Navier-Stokes (CNS) operators. This document defines the
formulation, configuration contract, and current verification status.

## Scope and coordinates

Mesh coordinate 0 is the axial coordinate, `z`, and mesh coordinate 1 is the radial
coordinate, `r`:

```text
X => x[0] = z
Y => x[1] = r >= 0
```

An axisymmetric mesh must have topological and spatial dimension 2. Negative
radial coordinates are invalid. A domain is not required to include the axis;
when it does, the axis is the boundary $r = 0$ and uses the dedicated `axis`
boundary condition.

The state is ordered using the normal `StateLayout` convention:

$$
U =
\begin{bmatrix}
\rho \\
\rho u_z \\
\rho u_r \\
\rho E
\end{bmatrix}
$$

## State invariant

The evolved state is ordinary physical conservative state $\mathbf{U}$, not the
radius-weighted state $r\mathbf{U}$. Initial conditions, gas models, numerical fluxes,
CFL calculations, visualization, and checkpoints must all consume or store
$\mathbf{U}$ with the same meaning as in a Cartesian simulation.

Axisymmetry is represented by explicit geometric terms in the spatial
operator. Radius weighting is used only where the mathematical measure requires
it, such as physical integral diagnostics. It must not be encoded implicitly in
the solution vector.

## Governing inviscid equations

Away from the axis, the Euler equations are written as the existing
Cartesian-like divergence in $\left(z,r\right)$ plus a geometric source:

$$
\partial_t U + \partial_z F_z(U) + \partial_r F_r(U)
= -\frac{1}{r}
\begin{bmatrix}
\rho u_r\\
\rho u_z u_r\\
\rho u_r^2\\
(\rho E+p)u_r
\end{bmatrix}.
$$

The discrete operator adds this contribution exactly once. It does not
simultaneously radius-weight Cartesian fluxes or the evolved state.

## Viscous geometric terms

For CNS, let $\mathbf{u}=(u_z,u_r)$ and use the swirl-free cylindrical velocity
divergence

$$
\nabla\!\cdot\mathbf{u}
= \frac{\partial u_z}{\partial z}
+ \frac{\partial u_r}{\partial r}
+ \frac{u_r}{r}.
$$

With the Stokes hypothesis used by Theseus, the stress components needed by the
meridional operator are

$$
\begin{aligned}
\tau_{zz} &= \mu\left(2\frac{\partial u_z}{\partial z}
- \frac{2}{3}\nabla\!\cdot\mathbf{u}\right), \\
\tau_{rr} &= \mu\left(2\frac{\partial u_r}{\partial r}
- \frac{2}{3}\nabla\!\cdot\mathbf{u}\right), \\
\tau_{\theta\theta} &= \mu\left(2\frac{u_r}{r}
- \frac{2}{3}\nabla\!\cdot\mathbf{u}\right), \\
\tau_{zr}=\tau_{rz} &= \mu\left(
\frac{\partial u_z}{\partial r}+\frac{\partial u_r}{\partial z}\right).
\end{aligned}
$$

The radial heat flux is $q_r=-\kappa\,\partial_r T$. The Cartesian-like
viscous divergence uses the meridional viscous fluxes in the $z$ and $r$
directions. The remaining cylindrical contribution added to the right-hand
side is

$$
S_{\mathrm{axi}}^{V}
= \frac{1}{r}
\begin{bmatrix}
0 \\
\tau_{zr} \\
\tau_{rr}-\tau_{\theta\theta} \\
u_z\tau_{zr}+u_r\tau_{rr}-q_r
\end{bmatrix}.
$$

The full CNS geometric source is the inviscid source above plus
$S_{\mathrm{axi}}^{V}$. The operator adds these terms once; it does not also
radius-weight the evolved state or meridional flux divergence.

These source terms are volume terms and do not depend on a boundary normal. On
an off-axis curved or oblique boundary, the numerical flux uses the actual
meridional normal $\mathbf{n}=(n_z,n_r)$ through
$F_n=F_z n_z+F_r n_r$; no additional source correction involving
$\hat{\mathbf r}\!\cdot\mathbf{n}$ is needed.

## Axis regularity

At $r = 0$, radial velocity is odd and vanishes. Density, pressure, energy, and
axial velocity are even in radius. Singular-looking $f/r$ terms must use their
analytic parity/L'Hopital limits at axis nodes; clipping radius to a small
positive value is not an acceptable regularization.

## Configuration contract

Axisymmetry remains selected at build time with `-DAXISYMMETRIC=ON`. If a case
contains `compileTime.AXISYMMETRIC`, that value must match the executable.
Axisymmetric configurations require:

- `runTime.dim = 2`
- `runTime.num_equations = 4`
- a two-dimensional mesh with `x[1] >= 0`

An `axis` boundary must lie on $r = 0$; Theseus validates its boundary
quadrature points before starting the simulation. Axis boundaries are invalid
in Cartesian builds.

## Testing coverage

The axisymmetric regression suite currently exercises:

- ideal/CPG gas with the Chandrashekar numerical flux;
- Euler and CNS uniform axial flow, both serial and two-rank MPI;
- an Euler entropy-wave convergence study using the cylindrical norm;
- axisymmetric checkpoint/restart equivalence and ParaView output equivalence;
- Mach 2 inviscid flow over a sphere in serial and two-rank MPI;
- Mach 0.3, `Re_D=100` viscous flow over a sphere in serial and two-rank MPI;
- the CPU MFEM backend.

The uniform-flow and sphere integration tests accept a configurable MFEM
backend through the CMake cache variable `AXISYMMETRIC_TEST_DEVICE`.
Accelerator builds can qualify the same Euler and CNS paths, for example:

```sh
cmake -S . -B build-axis-gpu \
  -DAXISYMMETRIC=ON \
  -DAXISYMMETRIC_TEST_DEVICE=cuda
cmake --build build-axis-gpu
ctest --test-dir build-axis-gpu \
  -R 'Axisymmetric(UniformFlow|InviscidSphere|ViscousSphere)Integration' \
  --output-on-failure
```

Use `hip` instead of `cuda` for an MFEM HIP build. CUDA and HIP execution paths
are device-oriented and are not disabled by axisymmetry, but they remain "officially"
untested until this regression is run on corresponding accelerator hardware.

See the [verification and CI matrix](verification.md) for the exact integration
assertions, direct smoke cases, and golden-data tolerances.

## Output and restart semantics

Local output fields are physical quantities recovered directly from $\mathbf{U}$.
Checkpoints store $\mathbf{U}$ and record the axisymmetric geometry and state convention
in compatibility metadata. Restarts reject ambiguous legacy axisymmetric
checkpoints that cannot establish whether they contain $\mathbf{U}$ or $r\mathbf{U}$.

Mass, total energy, and kinetic-energy diagnostics use the revolved-domain
cylindrical measure ${2}{\pi}{r}{dA}$. Visualization fields remain local physical
density, velocity, and pressure; they are not radius-weighted.
