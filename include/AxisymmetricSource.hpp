// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "AxisymmetricGeometry.hpp"
#include "GasState.hpp"
#include "NavierStokesFlux.hpp"

namespace Theseus
{
  /// Add the swirl-free Euler geometric source at points strictly off axis.
  /// Axis points require parity-based analytic limits and are handled separately.
  template<typename GasT>
  MFEM_HOST_DEVICE inline bool AddAxisymmetricEulerSourceAwayFromAxis(
    const GasT &gas, const mfem::real_t *state, mfem::real_t radius,
    mfem::real_t *state_rate)
  {
    if (radius <= AxisymmetricGeometry::radius_tolerance)
      {
        return false;
      }

    PointStateView U{state};
    const mfem::real_t density = gas.density(U);
    const mfem::real_t axial_momentum =
      gas.momentum(U, AxisymmetricGeometry::axial_coordinate);
    const mfem::real_t radial_momentum =
      gas.momentum(U, AxisymmetricGeometry::radial_coordinate);
    const mfem::real_t radial_velocity = radial_momentum / density;
    const mfem::real_t inverse_radius = 1.0 / radius;

    state_rate[gas.L.eq_mass] -= radial_momentum * inverse_radius;
    state_rate[gas.L.eq_mom[AxisymmetricGeometry::axial_coordinate]] -=
      axial_momentum * radial_velocity * inverse_radius;
    state_rate[gas.L.eq_mom[AxisymmetricGeometry::radial_coordinate]] -=
      radial_momentum * radial_velocity * inverse_radius;
    state_rate[gas.L.eq_energy] -=
      (gas.energy(U) + gas.pressure(U)) * radial_velocity * inverse_radius;
    return true;
  }

  template<typename GasT>
  MFEM_HOST_DEVICE inline void AddAxisymmetricEulerSourceAtAxis(
    const GasT &gas, const mfem::real_t *state,
    mfem::real_t radial_momentum_derivative, mfem::real_t *state_rate)
  {
    PointStateView U{state};
    const mfem::real_t density = gas.density(U);
    const mfem::real_t axial_velocity =
      gas.momentum(U, AxisymmetricGeometry::axial_coordinate) / density;
    const mfem::real_t total_enthalpy =
      (gas.energy(U) + gas.pressure(U)) / density;

    state_rate[gas.L.eq_mass] -= radial_momentum_derivative;
    state_rate[gas.L.eq_mom[AxisymmetricGeometry::axial_coordinate]] -=
      axial_velocity * radial_momentum_derivative;
    // The radial-momentum limit is zero because rho*u_r^2/r is odd.
    state_rate[gas.L.eq_energy] -=
      total_enthalpy * radial_momentum_derivative;
  }

  /// Add the non-divergence viscous terms in the swirl-free cylindrical
  /// equations at points strictly off axis.
  template<typename GasT>
  MFEM_HOST_DEVICE inline bool AddAxisymmetricViscousSourceAwayFromAxis(
    const GasT &gas, const mfem::real_t *state,
    const mfem::real_t *dprim_x, const mfem::real_t *dprim_y,
    const mfem::real_t *dprim_z, mfem::real_t radius,
    mfem::real_t *state_rate)
  {
    if (radius <= AxisymmetricGeometry::radius_tolerance)
      {
        return false;
      }

    mfem::real_t viscous_flux[Theseus::MAXEQ][Theseus::MAXDIM];
    mfem::real_t azimuthal_stress = 0.0;
    NavierStokesFlux::ComputeViscousFluxKernel(
      gas, state, dprim_x, dprim_y, dprim_z, viscous_flux, true, radius,
      &azimuthal_stress);

    const int axial = AxisymmetricGeometry::axial_coordinate;
    const int radial = AxisymmetricGeometry::radial_coordinate;
    const mfem::real_t inverse_radius = 1.0 / radius;
    state_rate[gas.L.eq_mom[axial]] +=
      viscous_flux[gas.L.eq_mom[axial]][radial] * inverse_radius;
    state_rate[gas.L.eq_mom[radial]] +=
      (viscous_flux[gas.L.eq_mom[radial]][radial] - azimuthal_stress) *
      inverse_radius;
    state_rate[gas.L.eq_energy] +=
      viscous_flux[gas.L.eq_energy][radial] * inverse_radius;
    return true;
  }

  template<typename ContextT>
  MFEM_HOST_DEVICE inline mfem::real_t RadialViscousFluxDerivative(
    const ContextT &ctx, const mfem::real_t *element_state,
    const mfem::real_t *element_gradprim_x,
    const mfem::real_t *element_gradprim_y,
    const mfem::real_t *element_gradprim_z,
    const mfem::real_t *element_radius,
    const mfem::real_t *element_jacobian,
    const mfem::real_t *element_metric, int point, int equation)
  {
    const int nx = ctx.Np_x;
    const int ny = ctx.Np_y;
    const int dof = ctx.ndof_scalar_el;
    const int equations = ctx.num_equations;
    const int i = point % nx;
    const int j = (point / nx) % ny;
    mfem::real_t derivative_xi = 0.0;
    mfem::real_t derivative_eta = 0.0;
    mfem::real_t state[Theseus::MAXEQ];
    mfem::real_t dprim_x[Theseus::MAXEQ];
    mfem::real_t dprim_y[Theseus::MAXEQ];
    mfem::real_t dprim_z[Theseus::MAXEQ];
    mfem::real_t flux[Theseus::MAXEQ][Theseus::MAXDIM];
    for (int l = 0; l < nx; ++l)
      {
        const int sample = j*nx + l;
        Kernels::el_gather_state(
          element_state, dof, equations, sample, state);
        Kernels::el_gather_grad_state(
          element_gradprim_x, element_gradprim_y, element_gradprim_z,
          ctx.dim, dof, equations, sample, dprim_x, dprim_y, dprim_z);
        NavierStokesFlux::ComputeViscousFluxKernel(
          ctx.gas, state, dprim_x, dprim_y, dprim_z, flux, true,
          element_radius[sample]);
        derivative_xi +=
          flux[equation][AxisymmetricGeometry::radial_coordinate] *
          ctx.D_d[l + nx*i];
      }
    for (int l = 0; l < ny; ++l)
      {
        const int sample = l*nx + i;
        Kernels::el_gather_state(
          element_state, dof, equations, sample, state);
        Kernels::el_gather_grad_state(
          element_gradprim_x, element_gradprim_y, element_gradprim_z,
          ctx.dim, dof, equations, sample, dprim_x, dprim_y, dprim_z);
        NavierStokesFlux::ComputeViscousFluxKernel(
          ctx.gas, state, dprim_x, dprim_y, dprim_z, flux, true,
          element_radius[sample]);
        derivative_eta +=
          flux[equation][AxisymmetricGeometry::radial_coordinate] *
          ctx.D_d[l + ny*j];
      }

    const mfem::real_t *adjugate = element_metric + point*ctx.dim*ctx.dim;
    return (derivative_xi *
              adjugate[AxisymmetricGeometry::radial_coordinate] +
            derivative_eta *
              adjugate[ctx.dim + AxisymmetricGeometry::radial_coordinate]) /
           element_jacobian[point];
  }

  template<typename ContextT>
  MFEM_HOST_DEVICE inline void AddAxisymmetricViscousSourceAtAxis(
    const ContextT &ctx, const mfem::real_t *element_state,
    const mfem::real_t *element_gradprim_x,
    const mfem::real_t *element_gradprim_y,
    const mfem::real_t *element_gradprim_z,
    const mfem::real_t *element_radius,
    const mfem::real_t *element_jacobian,
    const mfem::real_t *element_metric, int point,
    mfem::real_t *state_rate)
  {
    const int axial_momentum =
      ctx.gas.L.eq_mom[AxisymmetricGeometry::axial_coordinate];
    state_rate[axial_momentum] += RadialViscousFluxDerivative(
      ctx, element_state, element_gradprim_x, element_gradprim_y,
      element_gradprim_z, element_radius, element_jacobian,
      element_metric, point, axial_momentum);
    state_rate[ctx.gas.L.eq_energy] += RadialViscousFluxDerivative(
      ctx, element_state, element_gradprim_x, element_gradprim_y,
      element_gradprim_z, element_radius, element_jacobian,
      element_metric, point, ctx.gas.L.eq_energy);
    // (tau_rr - tau_theta_theta)/r tends to zero by axis parity.
  }

  template<typename ContextT>
  MFEM_HOST_DEVICE inline mfem::real_t RadialMomentumDerivative(
    const ContextT &ctx, const mfem::real_t *element_state,
    const mfem::real_t *element_jacobian,
    const mfem::real_t *element_metric, int point)
  {
    const int nx = ctx.Np_x;
    const int ny = ctx.Np_y;
    const int dof = ctx.ndof_scalar_el;
    const int radial_momentum_equation =
      ctx.gas.L.eq_mom[AxisymmetricGeometry::radial_coordinate];
    const int i = point % nx;
    const int j = (point / nx) % ny;
    mfem::real_t derivative_xi = 0.0;
    mfem::real_t derivative_eta = 0.0;
    for (int l = 0; l < nx; ++l)
      {
        derivative_xi +=
          element_state[j*nx + l + radial_momentum_equation*dof] *
          ctx.D_d[l + nx*i];
      }
    for (int l = 0; l < ny; ++l)
      {
        derivative_eta +=
          element_state[l*nx + i + radial_momentum_equation*dof] *
          ctx.D_d[l + ny*j];
      }

    const mfem::real_t *adjugate = element_metric + point*ctx.dim*ctx.dim;
    return (derivative_xi *
              adjugate[AxisymmetricGeometry::radial_coordinate] +
            derivative_eta *
              adjugate[ctx.dim + AxisymmetricGeometry::radial_coordinate]) /
           element_jacobian[point];
  }

  template<typename ContextT>
  MFEM_HOST_DEVICE inline void AddAxisymmetricEulerElementSource(
    const ContextT &ctx, const mfem::real_t *element_state,
    const mfem::real_t *element_radius,
    const mfem::real_t *element_jacobian,
    const mfem::real_t *element_metric, mfem::real_t *element_rate)
  {
    if (!ctx.axisymmetric)
      {
        return;
      }

    const int dof = ctx.ndof_scalar_el;
    const int equations = ctx.num_equations;
    mfem::real_t state[Theseus::MAXEQ];
    mfem::real_t source[Theseus::MAXEQ];
    for (int point = 0; point < dof; ++point)
      {
        Kernels::el_gather_state(element_state, dof, equations, point, state);
        for (int equation = 0; equation < equations; ++equation)
          {
            source[equation] = 0.0;
          }
        if (!AddAxisymmetricEulerSourceAwayFromAxis(
              ctx.gas, state, element_radius[point], source))
          {
            AddAxisymmetricEulerSourceAtAxis(
              ctx.gas, state,
              RadialMomentumDerivative(ctx, element_state,
                                       element_jacobian, element_metric,
                                       point),
              source);
          }
        Kernels::el_scatter_add(source, dof, equations, point, 1.0,
                                element_rate);
      }
  }

  /// Enforce the even/odd primitive-gradient parity required at r=0.
  /// Primitive-gradient slots contain (rho, velocity components, p, scalars).
  template<typename GasT>
  MFEM_HOST_DEVICE inline bool ProjectAxisPrimitiveGradientDirection(
	      const GasT &gas, mfem::real_t radius, int derivative_direction,
	      mfem::real_t *dprim)
  {
    if (radius > AxisymmetricGeometry::radius_tolerance){ return false; }
    const int axial = AxisymmetricGeometry::axial_coordinate;
    const int radial = AxisymmetricGeometry::radial_coordinate;
    const int radial_velocity = gas.L.eq_mom[radial];
    if (derivative_direction == radial)
      {
	const mfem::real_t radial_velocity_derivative =
	  dprim[radial_velocity];
	for (int equation = 0; equation < gas.L.nequations(); ++equation)
	  {
	    dprim[equation] = 0.0;
	  }
	// u_r is odd, so its radial derivative is even and need not vanish.
	dprim[radial_velocity] = radial_velocity_derivative;
      }
    else if (derivative_direction == axial)
      {
	// The axial derivative of the odd radial velocity vanishes on axis.
	dprim[radial_velocity] = 0.0;
      }
    return true;
  }
}
