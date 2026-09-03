# Timestep and CFL estimation

For fixed timestep, the configured `runTime.dt` is constant.  The actual DT
sent to the ODE solver may be shortened in order to fall exactly on an
upcoming configured event (e.g. I/O, or final time).  An estimated CFL is
computed every `runTime.cfl_check_interval` steps; this interval defaults to
`print_interval`.  The reported nominal value uses the configured fixed
timestep, and if the most recent step was shortened for I/O, its smaller actual
CFL is also reported just for good measure.

For fixed CFL mode (variable DT),

```
dt = target_cfl / stability_rate
```

and growth in DT is limited by `runTime.max_dt_growth`, which defaults to 1.2.
Growth limiting is done to prevent gigantic increases in DT (gigantic decreases
are not limited).  This is an ad-hoc measure to prevent things like transients,
shocks, or other flow-features (or the numerics that deal with them) from introducing
sudden significant (overly optimistic) increases in DT.

More precisely, after step $n$ the next nominal timestep is

$$
\Delta t_{n+1} = \min\left(
  \frac{C_{\mathrm{target}}}{\sigma(U_{n+1})},
  g_{\max}\Delta t_n
\right),
$$

where $g_{\max}$ is `runTime.max_dt_growth`.  The initial variable timestep
uses only the first term.  The timestep actually sent to the ODE solver is

$$
\Delta t_{\mathrm{actual}} = \min\left(
  \Delta t_{\mathrm{nominal}},
  t_{\mathrm{final}}-t,
  t_{\mathrm{next\ visualization}}-t,
  t_{\mathrm{next\ checkpoint}}-t
\right),
$$

with events that have already been reached omitted from the minimum.

## Stability estimate

The reported CFL and the timestep calculation use the same global stability
rate,

$$
\sigma(U) = \max\left(\sigma_{\mathrm{vol}},
                       \sigma_{\mathrm{face}}\right)
            + \sigma_{\mathrm{diff}},
\qquad
C_{\mathrm{est}}(\Delta t) = \Delta t\,\sigma(U).
$$

Each component is first maximized over the process-local nodes or face points.
An MPI maximum reduction is then performed independently on all three
components before they are combined as above.  The volume and face advective
bounds are alternatives for the same advective operator, hence their maximum;
the diffusive contribution is additive.

### Volume advection

At each volume node and mapped coordinate direction, the advective scan uses

$$
\sigma_{\mathrm{vol}} = \max_q\left[
  s_a(p)\sum_{i=1}^{d}
  \frac{|\boldsymbol{u}_q\mathbin{\cdot}\boldsymbol{a}_{i,q}|
        +c_q\|\boldsymbol{a}_{i,q}\|}{J_q}
\right],
$$

where $q$ is a volume node, $\boldsymbol{a}_i$ is row $i$ of the
adjugate-Jacobian matrix, $J$ is the mapping Jacobian, $\boldsymbol{u}$ is the
physical velocity, and $c$ is the local sound speed.  The polynomial scale
$s_a(p)$ is the computed spectral radius of the periodic one-dimensional,
unit-speed, unit-cell upwind DGSEM advection operator, including a five-percent
margin.  Orders 1 through 12 are tabulated; higher orders use
$s_a(p)=0.65(p+1)^2$.

### Face advection

Interior-face scans use the maximum left/right normal-aligned acoustic speed.
Boundary-face scans use the interior trace and also include prescribed
supersonic-inflow states.  At an interior face point,

$$
\lambda_n = \max_{s\in\{-,+\}}
 \left(|\boldsymbol{u}_s\mathbin{\cdot}\boldsymbol{n}|
       +c_s\|\boldsymbol{n}\|\right),
\qquad
\sigma_f = \lambda_n
 \max\left(|w_f^-|,|w_f^+|\right)s_f(p).
$$

For a boundary point the same expression uses the interior state and its
single face weight; for prescribed supersonic inflow, the maximum also includes
the prescribed exterior state.  Here the stored $\boldsymbol{n}$ and $w_f$
collectively contain the mapped face geometry and endpoint lifting/Jacobian
factors.  If $\ell_{\mathrm{end}}=1/w_{\mathrm{end}}$ denotes the endpoint
lifting factor, then $s_f(p)=s_a(p)/\ell_{\mathrm{end}}$ avoids applying it
twice.  Finally, $\sigma_{\mathrm{face}}$ is the maximum of $\sigma_f$ over all
interior and boundary face points.

Dissipative numerical fluxes evaluate any characteristic speeds they require
internally; those values are not returned, stored, or reduced by the residual
assembly.  In particular, the scalar dissipation added to the Chandrashekar
face flux uses the mapped normal coefficient
$\max_s(|\boldsymbol{u}_s\cdot\boldsymbol{n}|+c_s\|\boldsymbol{n}\|)$; tangential
velocity does not inflate this dissipation.  The timestep estimator evaluates
its normal-aligned bound independently of the selected numerical flux, so its
result is invariant under a change of flux implementation.

### Viscous contribution

For viscous flows, the estimate uses reference spectral radii of the coupled
periodic scalar BR1 auxiliary-gradient and divergence operators for orders 1
through 12, with a 25-percent margin.  Higher orders use the continuation
$s_d(p)=0.5(p+1)^4$.  At volume node $q$, define

$$
\nu_{\mathrm{mom}} =
 \frac{\max\left(\mu,\frac{4}{3}\mu+\mu_b\right)}{\rho},
\qquad
\alpha = \frac{\kappa\gamma}{\rho c_p},
\qquad
\chi = \max(\nu_{\mathrm{mom}},\alpha),
$$

where $\mu$ and $\mu_b$ are shear and bulk viscosity, $\kappa$ is thermal
conductivity, and $c_p$ is the constant-pressure heat capacity.  The viscous
rate is

$$
\sigma_{\mathrm{diff}} = \max_q\left[
 s_d(p)\,\chi_q
 \sum_{i=1}^{d}\frac{\|\boldsymbol{a}_{i,q}\|^2}{J_q^2}
\right].
$$

All transport and thermodynamic quantities are evaluated from the local state
through the same gas model used by the residual, including Sutherland and LTE
models.  For inviscid flows, $\sigma_{\mathrm{diff}}=0$.

### Fixed-timestep reporting

For fixed timestepping, the periodically reported nominal value is

$$
C_{\mathrm{nominal}} = \Delta t_{\mathrm{fixed}}\,\sigma(U).
$$

If an output, checkpoint, or final-time event shortened the completed step,
the diagnostic additionally reports

$$
C_{\mathrm{actual}} = \Delta t_{\mathrm{actual}}\,\sigma(U),
$$

which is necessarily no larger than the nominal value for the same state.
Both values use the state after the completed step. For performance reasons,
when fixed DT mode is used, no CFL computation is performed except at check/print
intervals given by the user.

#### NOTE:

This is still an inexpensive stability estimate, not a runtime eigensolve of
the complete nonlinear Jacobian (which would introduce significant overhead).
The reference BR1 calculation includes both the auxiliary-gradient equation
and the divergence of its diffusive flux, with central interface traces in each
operator. Full compressible and physical-boundary spectra remain validation work.

The tabulated reference values used by the code can be reproduced offline with
`scripts/reference_stability_spectra.py`.  This utility is not called by the
solver and adds no initialization or per-step cost. Its values are hard-coded
into the table in `include/StabilityEstimate.hpp`.
