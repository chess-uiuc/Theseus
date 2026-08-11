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
    // Inviscid / Euler Flux
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

    // Optimized CPG-specific version written by ChatGPT
    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void ComputeViscousFluxKernel(
					 const GasT &gas,
					 const mfem::real_t *state,
					 const mfem::real_t *dprim_x,
					 const mfem::real_t *dprim_y,
					 const mfem::real_t *dprim_z,
					 mfem::real_t visc_flux[Theseus::MAXEQ][Theseus::MAXDIM],
					 bool axisymmetric = false,
					 mfem::real_t radius = 0.0,
					 mfem::real_t *azimuthal_stress = nullptr)
    {
      using real_t = mfem::real_t;

      const int dim = gas.dim();

      // State layout.
      const int eq_mass = gas.L.eq_mass;
      const int eq_mom0 = gas.L.eq_mom0;
      const int eq_ener = gas.L.eq_energy;

      //--------------------------------------------------------------------------
      // Preserve original output-zeroing behavior exactly for this benchmark.
      //
      // Once correctness is established, this is something worth revisiting:
      // normally only the active equation rows need to be initialized.
      //--------------------------------------------------------------------------
      for (int q = 0; q < Theseus::MAXEQ; ++q)
	{
	  for (int dir = 0; dir < dim; ++dir)
	    {
	      visc_flux[q][dir] = 0.0;
	    }
	}

      Theseus::PointStateView S{state};

      //--------------------------------------------------------------------------
      // CPG / transport constants.
      //
      // These accessors only return constants from phys, so leave them alone.
      // They should inline away essentially for free.
      //--------------------------------------------------------------------------
      const real_t mu      = gas.viscosity(S);
      const real_t kappa   = gas.thermal_conductivity(S);
      const real_t mu_bulk = gas.bulk_viscosity(S);

      const real_t gamma          = gas.phys.gamma;
      const real_t gamma_m1       = gas.phys.gammaM1;
      const real_t gamma_m1_inv   = gas.phys.gammaM1Inverse;

      // cp() is also just a CPG constant.
      const real_t cp = gas.cp(S);

      //--------------------------------------------------------------------------
      // Construct the CPG state ONCE.
      //--------------------------------------------------------------------------

      const real_t rho  = state[eq_mass];
      const real_t irho = 1.0 / rho;

      real_t vel[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      real_t v2 = 0.0;

      for (int d = 0; d < dim; ++d)
	{
	  const real_t vd = state[eq_mom0 + d] * irho;
	  vel[d] = vd;
	  v2 += vd * vd;
	}

      // p = (gamma-1) * rho*e
      //
      // Reuse v^2 computed above rather than calling pressure(), which would
      // perform another momentum-squared reduction and density division.
      const real_t rhoE = state[eq_ener];
      const real_t rhoe = rhoE - 0.5 * rho * v2;
      const real_t p = gamma_m1 * rhoe;
      const real_t p_over_rho = p * irho;

      //--------------------------------------------------------------------------
      // Velocity gradient tensor:
      //
      //     grad_vel[component][physical direction]
      //
      // This is genuinely needed for the viscous stress tensor.
      //--------------------------------------------------------------------------

      real_t grad_vel[Theseus::MAXDIM][Theseus::MAXDIM] = {{0.0}};

      grad_vel[0][0] = dprim_x[eq_mom0];

      if (dim > 1)
	{
	  grad_vel[0][1] = dprim_y[eq_mom0];

	  grad_vel[1][0] = dprim_x[eq_mom0 + 1];
	  grad_vel[1][1] = dprim_y[eq_mom0 + 1];

	  if (dim > 2)
	    {
	      grad_vel[0][2] = dprim_z[eq_mom0];

	      grad_vel[1][2] = dprim_z[eq_mom0 + 1];

	      grad_vel[2][0] = dprim_x[eq_mom0 + 2];
	      grad_vel[2][1] = dprim_y[eq_mom0 + 2];
	      grad_vel[2][2] = dprim_z[eq_mom0 + 2];
	    }
	}

      //--------------------------------------------------------------------------
      // Velocity divergence.
      //--------------------------------------------------------------------------

      real_t div_vel = 0.0;

      for (int d = 0; d < dim; ++d)
	{
	  div_vel += grad_vel[d][d];
	}

      //--------------------------------------------------------------------------
      // Axisymmetric contribution.
      //--------------------------------------------------------------------------

      real_t radial_rate = 0.0;

      if (axisymmetric)
	{
	  const int radial = AxisymmetricGeometry::radial_coordinate;

	  if (radius > AxisymmetricGeometry::radius_tolerance)
	    {
	      radial_rate = vel[radial] / radius;
	    }
	  else
	    {
	      radial_rate = grad_vel[radial][radial];
	      // Axis parity requires u_r = 0.
	      // Preserve original behavior in the energy flux.
	      vel[radial] = 0.0;
	    }
	  div_vel += radial_rate;
	}

      //--------------------------------------------------------------------------
      // Precompute the bulk contribution used by every diagonal stress.
      //
      // Original:
      //
      //   mu * (2 grad_u - mu_bulk * div_u)
      //
      // becomes
      //
      //   2 mu grad_u - bulk_term
      //--------------------------------------------------------------------------

      const real_t bulk_term = mu * mu_bulk * div_vel;

      //--------------------------------------------------------------------------
      // Optional azimuthal normal stress.
      //--------------------------------------------------------------------------

      if (azimuthal_stress)
	{
	  *azimuthal_stress = 0.0;

	  if (axisymmetric)
	    {
	      *azimuthal_stress = 2.0 * mu * radial_rate - bulk_term;
	    }
	}

      //--------------------------------------------------------------------------
      // CPG temperature-gradient coefficient.
      //
      // Existing implementation:
      //
      //   cv  = cp / gamma
      //   fac = gammaM1Inverse / (cv*rho)
      //
      // Therefore:
      //
      //   fac = gamma * gammaM1Inverse / cp * (1/rho)
      //
      // Compute it once here.
      //--------------------------------------------------------------------------
      const real_t grad_t_fac =gamma * gamma_m1_inv * irho / cp;

      //--------------------------------------------------------------------------
      // Build momentum and energy viscous fluxes by physical direction.
      //
      // Calculate grad(T) on demand for each direction, so grad_rho[],
      // grad_p[], and grad_t[] are no longer needed.
      //--------------------------------------------------------------------------

      for (int dir = 0; dir < dim; ++dir)
	{
	  real_t grad_rho_dir;
	  real_t grad_p_dir;

	  if (dir == 0)
	    {
	      grad_rho_dir = dprim_x[eq_mass];
	      grad_p_dir   = dprim_x[eq_ener];
	    }
	  else if (dir == 1)
	    {
	      grad_rho_dir = dprim_y[eq_mass];
	      grad_p_dir   = dprim_y[eq_ener];
	    }
	  else
	    {
	      grad_rho_dir = dprim_z[eq_mass];
	      grad_p_dir   = dprim_z[eq_ener];
	    }

	  // Exact CPG equivalent of gas.grad_temperature().
	  const real_t grad_t = grad_t_fac * (grad_p_dir - p_over_rho * grad_rho_dir);

	  // Start conductive energy flux.
	  real_t eflux = kappa * grad_t;
	  //-----------------------------------------------------------------------
	  // Stress tensor and viscous work.
	  //
	  // Accumulate vel . tau directly while tau is still live instead of
	  // storing tau into visc_flux and then reading it back.
	  //-----------------------------------------------------------------------

	  for (int mom = 0; mom < dim; ++mom)
	    {
	      real_t tau;

	      if (mom == dir)
		{
		  // Preserve legacy diagonal stress exactly.
		  tau = 2.0 * mu * grad_vel[mom][dir] - bulk_term;
		}
	      else
		{
		  tau = mu * (grad_vel[mom][dir] + grad_vel[dir][mom]);
		}
	      visc_flux[eq_mom0 + mom][dir] = tau;
	      eflux += vel[mom] * tau;
	    }
	  visc_flux[eq_ener][dir] = eflux;
	}
    }
    
    template<typename GasT>
    MFEM_HOST_DEVICE inline
    static void ComputeViscousFluxKernel2(const GasT &gas,
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
      mfem::real_t grad_vel[Theseus::MAXDIM][Theseus::MAXDIM] = {{0.0}};

      grad_rho[0] = dprim_x[eq_mass];
      grad_vel[0][0] = dprim_x[eq_mom0];
      grad_p[0] = dprim_x[eq_ener];
      if(dim > 1){
        grad_rho[1] = dprim_y[eq_mass];
        grad_vel[0][1] = dprim_y[eq_mom0];
        grad_vel[1][0] = dprim_x[eq_mom0+1];
        grad_vel[1][1] = dprim_y[eq_mom0+1];
        grad_p[1] = dprim_y[eq_ener];
        if(dim > 2){
          grad_rho[2] = dprim_z[eq_mass];
          grad_vel[0][2] = dprim_z[eq_mom0];
          grad_vel[1][2] = dprim_z[eq_mom0+1];
          grad_vel[2][0] = dprim_x[eq_mom0+2];
          grad_vel[2][1] = dprim_y[eq_mom0+2];
          grad_vel[2][2] = dprim_z[eq_mom0+2];
          grad_p[2] = dprim_z[eq_ener];
        }
      }

      gas.grad_temperature(S, grad_rho, grad_p, grad_t);
      mfem::real_t vel[Theseus::MAXDIM] = {0.0, 0.0, 0.0};
      mfem::real_t div_vel = 0.0;
      for(int i = 0;i < dim;i++){
        vel[i] = gas.velocity(S, i);
        div_vel += grad_vel[i][i];
      }
      mfem::real_t radial_rate = 0.0;
      if (axisymmetric)
        {
          const int radial = AxisymmetricGeometry::radial_coordinate;
          //radial_rate = radius > AxisymmetricGeometry::radius_tolerance ?
          //  vel[radial] / radius : grad_vel[radial][radial];
	  if (radius > AxisymmetricGeometry::radius_tolerance)
	    {
	      radial_rate = vel[radial] / radius;
	    }
          else
	    {
	      radial_rate = grad_vel[radial][radial];
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
          // viscous stress tensor components tau[mom,dir]
          for (int mom = 0; mom < dim; ++mom)
            {
              mfem::real_t tau = 0.0;

              if (mom == dir)
                {
                  // Perserve legacy tau exactly
                  tau = mu * (2.0 * grad_vel[mom][dir] - mu_bulk * div_vel);
                }
              else
                {
                  tau = mu * (grad_vel[mom][dir] + grad_vel[dir][mom]);
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
