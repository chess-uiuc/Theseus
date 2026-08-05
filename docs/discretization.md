# DGSEM discretization

Theseus uses a nodal discontinuous Galerkin spectral element method (DGSEM) on
tensor-product elements [4, 5]. Within each element, the solution is represented
at tensor-product Gauss-Lobatto nodes using nodal Lagrange polynomials. The same
nodes define the quadrature rule, giving a diagonal mass matrix. Tensor-product
differentiation matrices and the summation-by-parts property provide the volume
derivatives and their connection to numerical surface fluxes.

For the conservation law

$$
\frac{\partial U}{\partial t}+\nabla\!\cdot F(U)=0,
$$

the semidiscrete residual combines element-volume derivatives with numerical
fluxes on interior and boundary faces. The inviscid operator supports
entropy-conservative or dissipative two-state numerical fluxes selected by the
case configuration. Optional subcell finite-volume blending adds a shock-robust
residual without changing the meaning of the evolved conservative state $U$.

Explicit $s$-stage Runge-Kutta methods advance the resulting method-of-lines
system in time.

## Entropy-stable BR1 viscous discretization

The compressible Navier-Stokes operator uses an entropy-stable form of the BR1
method [6]. Directly differentiating conservative variables and then converting
their gradients can lose the entropy structure needed by the viscous
discretization. Theseus instead constructs the entropy state first and applies
the DG gradient operator to that state.

For a calorically perfect gas, define

$$
s = \log p-\gamma\log\rho,
\qquad
\beta=\frac{\rho}{p}.
$$

The entropy state used by the implementation is

$$
S(U)=
\begin{bmatrix}
\displaystyle \frac{\gamma-s}{\gamma-1}
-\frac{\beta}{2}|\mathbf{u}|^2 \\
\beta u_1 \\
\vdots \\
\beta u_d \\
-\beta
\end{bmatrix}.
$$

The viscous-gradient path is:

1. Evaluate $S(U)$ pointwise at every volume degree of freedom.
2. Apply the BR1 DG gradient operator to obtain a discrete approximation
   $G_S\approx\nabla S$. Interior faces use the centered BR1 trace, while
   boundary traces are supplied by the physical boundary condition.
3. At each volume point and in each spatial direction, use the local
   thermodynamic Jacobian to convert entropy-state gradients to primitive
   gradients:

$$
G_V
= \frac{\partial V}{\partial S}\bigg|_{U}G_S,
\qquad
V=(\rho,u_1,\ldots,u_d,p)^T.
$$

4. Compute $\nabla T$ from $\nabla\rho$ and $\nabla p$ through the gas model,
   then form the viscous stress tensor, heat flux, and viscous energy flux.
5. Apply the DG divergence operator to the viscous flux. BR1 uses the centered
   viscous flux on interior faces, with boundary fluxes determined by the wall,
   symmetry, inflow, or outflow condition.

The arrays that initially contain $\nabla S$ are overwritten pointwise by
$\nabla V$ before viscous flux evaluation. This keeps the BR1 lifting in entropy
variables while allowing the transport kernels to consume the primitive
gradients they require.

## Axisymmetric CNS

For an axisymmetric CNS simulation, the same entropy-state BR1 procedure
produces the meridional primitive gradients. The viscous flux kernel then uses
the cylindrical velocity divergence, including $u_r/r$, and forms the
azimuthal stress $\tau_{\theta\theta}$. The spatial operator adds the remaining
cylindrical viscous source terms described in the
[axisymmetric formulation](axisymmetry.md). At $r=0$, parity-based analytic
limits replace direct division by $r$.
