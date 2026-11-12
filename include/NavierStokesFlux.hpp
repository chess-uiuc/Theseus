// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "AxisymmetricGeometry.hpp"
#include "GasModel.hpp"

namespace Theseus
{
  namespace NavierStokesFlux
  {
    template<typename GasT>
    MFEM_HOST_DEVICE inline static mfem::real_t
    MaximumNormalWaveSpeed(const GasT &gas,
                           const mfem::real_t *state1,
                           const mfem::real_t *state2,
                           const mfem::real_t *direction)
    {
      PointStateView S1{state1};
      PointStateView S2{state2};
      mfem::real_t velocity_dot_direction1 = 0.0;
      mfem::real_t velocity_dot_direction2 = 0.0;
      mfem::real_t direction_norm_squared = 0.0;
      for (int d = 0; d < gas.dim(); ++d)
        {
          velocity_dot_direction1 += gas.velocity(S1, d) * direction[d];
          velocity_dot_direction2 += gas.velocity(S2, d) * direction[d];
          direction_norm_squared += direction[d] * direction[d];
        }
      const mfem::real_t direction_norm =
        Kernels::rsqrt(direction_norm_squared);
      return Kernels::rmax(
        Kernels::rabs(velocity_dot_direction1)
          + gas.sound_speed(S1) * direction_norm,
        Kernels::rabs(velocity_dot_direction2)
          + gas.sound_speed(S2) * direction_norm);
    }

    // Per-state thermodynamics reused across the normal-flux equations.
    // The state pointer is borrowed for the duration of the face calculation.
    struct InviscidNormalFluxState
    {
      const mfem::real_t *conserved;
      mfem::real_t inverse_density;
      mfem::real_t pressure;
      mfem::real_t total_enthalpy;
    };

    template<typename GasT>
    MFEM_HOST_DEVICE inline static InviscidNormalFluxState
    MakeInviscidNormalFluxState(const GasT &gas, const mfem::real_t *state)
    {
      const PointStateView view{state};
      const mfem::real_t inverse_density = 1.0 / gas.density(view);
      const mfem::real_t pressure = gas.pressure(view);
      return {state, inverse_density, pressure,
              (gas.energy(view) + pressure) * inverse_density};
    }

    // Physical Euler flux contracted with a possibly non-unit normal.
    // No full flux tensor is materialized; the caller supplies dissipation.
    template<typename GasT>
    MFEM_HOST_DEVICE inline static mfem::real_t
    ComputeInviscidNormalFluxEquation(const GasT &gas,
                                     const InviscidNormalFluxState &sample,
                                     const mfem::real_t *normal, int equation)
    {
      const PointStateView view{sample.conserved};
      mfem::real_t result = 0.0;
      for (int direction = 0; direction < gas.dim(); ++direction)
        {
          const mfem::real_t momentum = gas.momentum(view, direction);
          mfem::real_t physical_flux = 0.0;
          if (equation == gas.L.eq_mass)
            {
              physical_flux = momentum;
            }
          else if (equation >= gas.L.eq_mom0 &&
                   equation < gas.L.eq_mom0 + gas.dim())
            {
              const int component = equation - gas.L.eq_mom0;
              physical_flux = gas.momentum(view, component) * momentum *
                sample.inverse_density;
              if (component == direction) { physical_flux += sample.pressure; }
            }
          else if (equation == gas.L.eq_energy)
            {
              physical_flux = momentum * sample.total_enthalpy;
            }
          else if (gas.L.eq_scalar0 >= 0 && equation >= gas.L.eq_scalar0 &&
                   equation < gas.L.eq_scalar0 + gas.L.num_scalars)
            {
              physical_flux = gas.scalar(view, equation - gas.L.eq_scalar0) *
                momentum * sample.inverse_density;
            }
          result += physical_flux * normal[direction];
        }
      return result;
    }

    // Inviscid / Euler Flux
    // Adapted from MFEM's EulerFlux::ComputeFlux in fem/hyperbolic.cpp
    // (https://mfem.org), including the numbered state/flux/speed outline below.
    // MFEM copyright (c) 2010-2026, Lawrence Livermore National Security, LLC;
    // distributed under the BSD-3-Clause license.
    template<typename GasT>
    MFEM_HOST_DEVICE inline static void
    ComputeInviscidFluxKernel(const GasT &gas,
                              const mfem::real_t *state,
                              mfem::real_t inv_flux[Theseus::MAXEQ][Theseus::MAXDIM])
    { 
      PointStateView S{state};
      
      // 1. Get states
      const int dim = gas.dim();
      const mfem::real_t density = gas.density(S);
      const mfem::real_t spec_vol = 1.0/density;
      mfem::real_t momentum[Theseus::MAXDIM] = {0.,0.,0.};
      for(int idim = 0;idim < dim;idim++){
        momentum[idim] = gas.momentum(S, idim);
      }

      const mfem::real_t energy = gas.energy(S);
      const mfem::real_t pressure = gas.pressure(S);
      const mfem::real_t ke = gas.kinetic_energy_density(S);
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_ener = gas.L.eq_energy;
      const int eq_spec = gas.L.eq_scalar0;

      const mfem::real_t H = (energy + pressure)*spec_vol;
      // 2. Compute Flux
      for (int d = 0; d < dim; d++)
        {
          inv_flux[eq_mass][d] = momentum[d];
          for (int i = 0; i < dim; i++)
            {
              // ρuuᵀ
              inv_flux[eq_mom0+i][d] = momentum[i]*momentum[d]*spec_vol;
            }
          // (ρuuᵀ) + p
          inv_flux[eq_mom0+d][d] += pressure;
          inv_flux[eq_ener][d] = momentum[d]*H;
          for(int s = 0;s < gas.L.num_scalars;s++){
            inv_flux[eq_spec+s][d] = gas.scalar(S, s) * momentum[d] * spec_vol;
          }
        }
      // 3. Compute maximum characteristic speed 
      // const mfem::real_t sound = gas.sound_speed(S);
      // fluid speed |u|
      // const mfem::real_t speed = Theseus::Kernels::rsqrt(2.0 * ke / density);
      // max characteristic speed = fluid speed + sound speed
      // return speed + sound;
    }

    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void ComputeViscousFluxKernel(const GasT &gas,
                                         const mfem::real_t *state,
                                         const mfem::real_t *dprim_x,
                                         const mfem::real_t *dprim_y,
                                         const mfem::real_t *dprim_z,
                                         mfem::real_t visc_flux[Theseus::MAXEQ][Theseus::MAXDIM],
                                         bool axisymmetric = false,
                                         mfem::real_t radius = 0.0,
                                         mfem::real_t *azimuthal_stress = nullptr)
    {

      // TODO: Update for scalar transport
      const int dim = gas.dim();
      // Zero the flux to start
      for(int q = 0;q < Theseus::MAXEQ;q++){
        for(int idir = 0;idir < dim;idir++){
          visc_flux[q][idir] = 0.0;
        }
      }

      PointStateView S{state};
 
      // Access some physical constants
      const mfem::real_t mu = gas.viscosity(S);
      const mfem::real_t kappa = gas.thermal_conductivity(S);
      const mfem::real_t mu_bulk = gas.bulk_viscosity(S);

      // State structure constants
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_ener = gas.L.eq_energy;
      const int nscalar = gas.L.num_scalars;
 
      // Make & populate gradient containers
      mfem::real_t grad_rho[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t grad_p[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t grad_t[Theseus::MAXDIM] = {0.0, 0.0, 0.0};

      grad_rho[0] = dprim_x[eq_mass];
      grad_p[0] = dprim_x[eq_ener];
      if(dim > 1){
        grad_rho[1] = dprim_y[eq_mass];
        grad_p[1] = dprim_y[eq_ener];
        if(dim > 2){
          grad_rho[2] = dprim_z[eq_mass];
          grad_p[2] = dprim_z[eq_ener];
        }
      }

      gas.grad_temperature(S, grad_rho, grad_p, grad_t);
      mfem::real_t vel[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t div_vel = 0.0;
      for(int i = 0;i < dim;i++){
        vel[i] = gas.velocity(S, i);
        const mfem::real_t *gradient_row = i == 0 ? dprim_x
          : (i == 1 ? dprim_y : dprim_z);
        div_vel += gradient_row[eq_mom0 + i];
      }
      mfem::real_t radial_rate = 0.0;
      if (axisymmetric)
        {
          const int radial = AxisymmetricGeometry::radial_coordinate;
	  if (radius > AxisymmetricGeometry::radius_tolerance)
	    {
	      radial_rate = vel[radial] / radius;
	    }
          else
	    {
              const mfem::real_t *radial_gradient = radial == 0 ? dprim_x
                : (radial == 1 ? dprim_y : dprim_z);
              radial_rate = radial_gradient[eq_mom0 + radial];
	      // Axis parity requires u_r=0. Use the regular trace in the
	      // viscous energy flux even if the weak axis BC leaves a small
	      // nonzero radial momentum at the boundary node.
	      vel[radial] = 0.0;
	    }
          div_vel += radial_rate;
        }

      if (azimuthal_stress)
        {
          *azimuthal_stress = 0.0;
          if (axisymmetric)
            {
              *azimuthal_stress = mu *
                (2.0 * radial_rate - mu_bulk * div_vel);
            }
        }

      // Build momentum/energy viscous fluxes by physical direction.
      // Output convention:
      //   flux_eq_dir[eq][dir]
      //
      // Mass row eq=0 remains zero.
      for (int dir = 0; dir < dim; ++dir)
        {
          // Borrow the existing velocity derivatives; a second matrix copy
          // would duplicate data already present in the directional inputs.
          const mfem::real_t *direction_gradient = dir == 0 ? dprim_x
            : (dir == 1 ? dprim_y : dprim_z);
          // viscous stress tensor components tau[mom,dir]
          for (int mom = 0; mom < dim; ++mom)
            {
              mfem::real_t tau = 0.0;

              if (mom == dir)
                {
                  // Perserve legacy tau exactly
                  tau = mu * (2.0 * direction_gradient[eq_mom0 + mom]
                              - mu_bulk * div_vel);
                }
              else
                {
                  const mfem::real_t *component_gradient = mom == 0 ? dprim_x
                    : (mom == 1 ? dprim_y : dprim_z);
                  tau = mu * (direction_gradient[eq_mom0 + mom]
                              + component_gradient[eq_mom0 + dir]);
                }
              
              visc_flux[eq_mom0 + mom][dir] = tau;
            }
          mfem::real_t eflux = kappa * grad_t[dir];
          for(int mom = 0;mom < dim;mom++)
            {
              eflux += vel[mom] * visc_flux[eq_mom0 + mom][dir];
            }
          visc_flux[eq_ener][dir] = eflux;
        }
    }

    // Contract stress and heat flux directly with a physical normal. Directional
    // primitive gradients are borrowed; no stress or equation-by-direction tensor
    // is retained across constitutive calls. Output must not alias the inputs.
    template<typename GasT>
    MFEM_HOST_DEVICE inline static void
    ComputeViscousNormalFluxKernel(const GasT &gas, const mfem::real_t *state,
      const mfem::real_t *dprim_x, const mfem::real_t *dprim_y,
      const mfem::real_t *dprim_z, const mfem::real_t *normal,
      mfem::real_t *flux, bool axisymmetric = false, mfem::real_t radius = 0.0)
    {
      const PointStateView view{state};
      const int dim = gas.dim();
      const int momentum = gas.L.eq_mom0;
      const int energy = gas.L.eq_energy;
      const auto mu = gas.viscosity(view);
      const auto kappa = gas.thermal_conductivity(view);
      const auto mu_bulk = gas.bulk_viscosity(view);
      mfem::real_t grad_rho[MAXDIM] = {};
      mfem::real_t grad_thermal[MAXDIM] = {};
      mfem::real_t grad_temperature[MAXDIM] = {};
      mfem::real_t divergence = 0.0;
      for (int d = 0; d < dim; ++d)
        {
          const auto *gradient = d == 0 ? dprim_x : (d == 1 ? dprim_y : dprim_z);
          grad_rho[d] = gradient[gas.L.eq_mass];
          grad_thermal[d] = gradient[energy];
          divergence += gradient[momentum + d];
        }
      gas.grad_temperature(view, grad_rho, grad_thermal, grad_temperature);
      if (axisymmetric)
        {
          const int radial = AxisymmetricGeometry::radial_coordinate;
          const auto *gradient = radial == 0 ? dprim_x : (radial == 1 ? dprim_y : dprim_z);
          divergence += radius > AxisymmetricGeometry::radius_tolerance
            ? gas.velocity(view, radial)/radius : gradient[momentum + radial];
        }
      for (int eq = 0; eq < gas.num_equations(); ++eq)
        {
          flux[eq] = 0.0;
        }
      mfem::real_t energy_flux = 0.0;
      for (int component = 0; component < dim; ++component)
        {
          mfem::real_t traction = 0.0;
          for (int direction = 0; direction < dim; ++direction)
            {
              const auto *direction_gradient = direction == 0 ? dprim_x
                : (direction == 1 ? dprim_y : dprim_z);
              const auto *component_gradient = component == 0 ? dprim_x
                : (component == 1 ? dprim_y : dprim_z);
              const auto stress = component == direction
                ? mu*(2.0*direction_gradient[momentum + component] - mu_bulk*divergence)
                : mu*(direction_gradient[momentum + component] + component_gradient[momentum + direction]);
              traction += normal[direction]*stress;
            }
          flux[momentum + component] = traction;
          const auto velocity = axisymmetric && component == AxisymmetricGeometry::radial_coordinate
            && radius <= AxisymmetricGeometry::radius_tolerance
            ? mfem::real_t(0) : gas.velocity(view, component);
          energy_flux += velocity*traction;
        }
      for (int direction = 0; direction < dim; ++direction)
        {
          energy_flux += normal[direction]*kappa*grad_temperature[direction];
        }
      flux[energy] = energy_flux;
    }

    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void compute_ref_viscous_flux(const GasT &gas,
                                         const int dim,
                                         const int neq,
                                         const mfem::real_t *state,
                                         const mfem::real_t *dqx,
                                         const mfem::real_t *dqy,
                                         const mfem::real_t *dqz,
                                         const mfem::real_t *adj_row,
                                         mfem::real_t *f_ref,
                                         bool axisymmetric = false,
                                         mfem::real_t radius = 0.0)
    {
      mfem::real_t flux_phys[Theseus::MAXEQ][Theseus::MAXDIM] = {{0.}};

      // Grab the physical flux
      ComputeViscousFluxKernel(gas, state, dqx, dqy, dqz, flux_phys,
                               axisymmetric, radius);
      
      for (int q = 0; q < neq; ++q)
        {
          f_ref[q] = 0.0;
          for (int j = 0; j < dim; ++j)
            f_ref[q] += adj_row[j] * flux_phys[q][j];
        }
    }
    
  };
  
}
