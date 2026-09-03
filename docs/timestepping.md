# Timestep and CFL estimation

Theseus keeps timestep selection separate from timestep diagnostics.

For a fixed timestep, the configured `runTime.dt` is immutable.  The step sent
to the ODE solver is shortened only to reach the final time, the next
time-based visualization, or the next checkpoint exactly.  No stability scan
or MPI reduction occurs on ordinary fixed-timestep steps.  An estimated CFL is
computed every `runTime.cfl_check_interval` steps; this interval defaults to
`print_interval`.  The reported nominal value uses the configured fixed
timestep.  If the most recent step was shortened for I/O, its smaller actual
CFL is also reported.

For a variable timestep,

```
dt = target_cfl / stability_rate
```

and growth is limited by `runTime.max_dt_growth`, which defaults to 1.2.

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

### Viscous contribution

For viscous builds, the estimate uses reference spectral radii of the coupled
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
models.  For inviscid builds, $\sigma_{\mathrm{diff}}=0$.

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
Both values use the state after the completed step.  No stability scan or MPI
reduction is performed between configured fixed-DT check intervals.

This is still an inexpensive stability estimate, not a runtime eigensolve of
the complete nonlinear Jacobian.  The reference BR1 calculation includes both
the auxiliary-gradient equation and the divergence of its diffusive flux, with
central interface traces in each operator.  Lifting interface quantities into
the volume representation is a general DG operation and is not specific to
BR1.  Full compressible and physical-boundary spectra remain validation work.

The tabulated reference values can be reproduced offline with
`scripts/reference_stability_spectra.py`.  This utility is not called by the
solver and adds no initialization or per-step cost.
