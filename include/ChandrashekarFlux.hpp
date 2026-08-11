// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

namespace Theseus
{

  namespace ChandrashekarFlux
  {
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline static mfem::real_t ComputeVolumeFluxKernel(const GasModelT &gasModel,
                                                 const mfem::real_t* q1,
                                                 const mfem::real_t* q2,
                                                 const mfem::real_t* met1,
                                                 const mfem::real_t* met2,
                                                 mfem::real_t* F_tilde)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();
      
      // mean metric row
      mfem::real_t met[3] = {0,0,0};
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      Theseus::PointStateView S1{q1};
      Theseus::PointStateView S2{q2};
      
      const mfem::real_t rho1 = gasModel.density(S1);
      const mfem::real_t rho2 = gasModel.density(S2);
      const mfem::real_t rho_ln = Kernels::ComputeLogMean(rho1, rho2, 1e-4);
      
      mfem::real_t mom_hat[3] = {0,0,0};
      mfem::real_t h_hat = 0;
      mfem::real_t vn = 0;
      mfem::real_t v2_1 = 0;
      mfem::real_t v2_2 = 0;
      
      for (int d=0; d<dim; ++d)
        {
          const mfem::real_t v1 = gasModel.velocity(S1, d);
          const mfem::real_t v2 = gasModel.velocity(S2, d);
          const mfem::real_t vbar = mfem::real_t(0.5)*(v1+v2);
          
          v2_1 += v1*v1;
          v2_2 += v2*v2;
          vn   += vbar * met[d];
          
          mom_hat[d] = rho_ln * vbar;
          
          h_hat += -mfem::real_t(0.25)*(v1*v1 + v2*v2) + vbar*vbar;
        }
      
      
      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);
      
      const mfem::real_t speed1 = Kernels::rsqrt(v2_1);
      const mfem::real_t speed2 = Kernels::rsqrt(v2_2);
      
      const mfem::real_t c1 = gasModel.sound_speed(S1);
      const mfem::real_t c2 = gasModel.sound_speed(S2);
      
      const mfem::real_t lambda_max = Kernels::rmax(speed1 + c1, speed2 + c2);
      
      // Single-component ideal-gas-specific KEPEC bits
      // TODO: Update/Craft KPEC fluxes for mixtures (and passive scalar components)
      const mfem::real_t beta1 = mfem::real_t(0.5) * rho1 / p1;
      const mfem::real_t beta2 = mfem::real_t(0.5) * rho2 / p2;
      const mfem::real_t beta_ln = Kernels::ComputeLogMean(beta1, beta2, 1e-4);
      
      const mfem::real_t p_hat = mfem::real_t(0.5) * (rho1 + rho2) / (beta1 + beta2);
      
      const mfem::real_t gm11 = gasModel.gamma(S1);
      const mfem::real_t gm12 = gasModel.gamma(S2);
      const mfem::real_t gm1_av_inv = mfem::real_t(2.0) / (gm11 + gm12 - mfem::real_t(2.0));
      
      h_hat += mfem::real_t(0.5) / beta_ln * gm1_av_inv + p_hat / rho_ln;
      
      // F_tilde layout: [rho, rhoV, rhoE]
      // NOTE: Caller *must* zero(or own) F_tilde (size: neq)
      // NOTE: HRM!  Why ZERO?  It appears that F_tilde is overwritten below
      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;
      F_tilde[mass_eq] = rho_ln * vn;
      for (int d=0; d<dim; ++d)
        {
          F_tilde[mom0_eq + d] = vn * mom_hat[d] + p_hat * met[d];
        }
      F_tilde[ener_eq] = rho_ln * vn * h_hat;
      
      // TODO: Updte for scalars, sigh
      // for (s=0; s<num_scalars; ++s) F_tilde[XXXX]= XXX
      
      return lambda_max;
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline static mfem::real_t
    ComputeFaceFluxKernel(const GasModelT &gasModel,
			  const mfem::real_t *state1,
			  const mfem::real_t *state2,
			  const mfem::real_t *nor,
			  mfem::real_t *flux)
    {
      using real_t = mfem::real_t;
      
      const int dim = gasModel.dim();
      
      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;
      
      const real_t gamma    = gasModel.phys.gamma;
      const real_t gamma_m1 = gasModel.phys.gammaM1;
      const real_t gamma_m1_inv = 1.0 / gamma_m1;
      
      //--------------------------------------------------------------------------
      // Conservative state
      //--------------------------------------------------------------------------
      
      const real_t rho1 = state1[mass_eq];
      const real_t rho2 = state2[mass_eq];
      
      const real_t irho1 = 1.0 / rho1;
      const real_t irho2 = 1.0 / rho2;
      
      const real_t rhoE1 = state1[ener_eq];
      const real_t rhoE2 = state2[ener_eq];
      
      const real_t rho_sum  = rho1 + rho2;
      const real_t rho_mean = 0.5 * rho_sum;
      const real_t drho     = rho2 - rho1;
      
      //--------------------------------------------------------------------------
      // Log mean of density
      //
      // u = ((rho2-rho1)/(rho2+rho1))^2
      //--------------------------------------------------------------------------
      
      const real_t rho_z = drho / rho_sum;
      const real_t rho_u = rho_z * rho_z;
      
      real_t rho_ln;
      
      if (rho_u < 1.0e-4)
	{
	  const real_t denom =
	    105.0 + rho_u *
	    (35.0 + rho_u *
	     (21.0 + 15.0 * rho_u));
	  
	  rho_ln = rho_sum * 52.5 / denom;
	}
      else
	{
	  rho_ln = drho / Kernels::rlog(rho2 * irho1);
	}
      
      //--------------------------------------------------------------------------
      // Velocity-dependent quantities
      //--------------------------------------------------------------------------
      
      real_t mom1[3] = {0.0, 0.0, 0.0};
      real_t mom2[3] = {0.0, 0.0, 0.0};
      real_t vbar[3] = {0.0, 0.0, 0.0};
      
      real_t v21 = 0.0;
      real_t v22 = 0.0;
      real_t vn = 0.0;
      real_t nor2 = 0.0;
      
      real_t hhat = 0.0;
      real_t diss = 0.0;
      
      for (int d = 0; d < dim; ++d)
	{
	  const real_t m1 = state1[mom0_eq + d];
	  const real_t m2 = state2[mom0_eq + d];
	  
	  mom1[d] = m1;
	  mom2[d] = m2;
	  
	  const real_t v1 = m1 * irho1;
	  const real_t v2 = m2 * irho2;
	  
	  const real_t vb = 0.5 * (v1 + v2);
	  const real_t dv = v2 - v1;
	  
	  vbar[d] = vb;
	  
	  v21 += v1 * v1;
	  v22 += v2 * v2;
	  
	  vn   += vb * nor[d];
	  nor2 += nor[d] * nor[d];
	  
	  // Exact simplification of:
	  //
	  // -0.25*(v1*v1 + v2*v2) + vb*vb
	  //
	  hhat += 0.5 * v1 * v2;
	  
	  diss += 0.5 * drho * v1 * v2
            + rho_mean * dv * vb;
	}
      
      const real_t nor_mag = std::sqrt(nor2);
      
      //--------------------------------------------------------------------------
      // CPG thermodynamics.
      //
      // Avoid gasModel.pressure() and gasModel.sound_speed(), since those
      // independently reconstruct kinetic energy.
      //--------------------------------------------------------------------------
      
      const real_t rhoe1 =
	rhoE1 - 0.5 * rho1 * v21;
      
      const real_t rhoe2 =
	rhoE2 - 0.5 * rho2 * v22;
      
      const real_t p1 = gamma_m1 * rhoe1;
      const real_t p2 = gamma_m1 * rhoe2;
      
      const real_t c1 =
	std::sqrt(gamma * p1 * irho1);
      
      const real_t c2 =
	std::sqrt(gamma * p2 * irho2);
      
      //--------------------------------------------------------------------------
      // Preserve original LLF wave-speed definition for apples-to-apples timing.
      //--------------------------------------------------------------------------
      
      const real_t vmag1 = std::sqrt(v21);
      const real_t vmag2 = std::sqrt(v22);
      
      const real_t lambda_max =
	Kernels::rmax(vmag1 + c1, vmag2 + c2);
      
      //--------------------------------------------------------------------------
      // Beta quantities
      //--------------------------------------------------------------------------
      
      const real_t beta1 = 0.5 * rho1 / p1;
      const real_t beta2 = 0.5 * rho2 / p2;
      
      // Avoid later divisions by beta.
      const real_t ibeta1 = 2.0 * p1 * irho1;
      const real_t ibeta2 = 2.0 * p2 * irho2;
      
      //--------------------------------------------------------------------------
      // Log mean of beta
      //--------------------------------------------------------------------------
      
      const real_t beta_sum = beta1 + beta2;
      const real_t dbeta    = beta2 - beta1;
      
      const real_t beta_z = dbeta / beta_sum;
      const real_t beta_u = beta_z * beta_z;
      
      real_t beta_ln;
      
      if (beta_u < 1.0e-4)
	{
	  const real_t denom =
	    105.0 + beta_u *
	    (35.0 + beta_u *
	     (21.0 + 15.0 * beta_u));
	  
	  beta_ln = beta_sum * 52.5 / denom;
	}
      else
	{
	  beta_ln =
	    dbeta / Kernels::rlog(beta2 / beta1);
	}
      
      const real_t ibeta_ln = 1.0 / beta_ln;
      
      //--------------------------------------------------------------------------
      // KEPEC thermodynamic terms
      //--------------------------------------------------------------------------
      
      const real_t p_hat =
	rho_mean / beta_sum;
      
      hhat +=
	0.5 * ibeta_ln * gamma_m1_inv
	+ p_hat / rho_ln;
      
      diss +=
	0.5 * drho * gamma_m1_inv * ibeta_ln
	+ 0.5 * rho_mean * gamma_m1_inv
	* (ibeta2 - ibeta1);
      
      //--------------------------------------------------------------------------
      // Numerical flux
      //--------------------------------------------------------------------------
      
      const real_t llf =
	0.5 * lambda_max * nor_mag;
      
      flux[mass_eq] =
	rho_ln * vn
	- llf * drho;
      
      for (int d = 0; d < dim; ++d)
	{
	  flux[mom0_eq + d] =
	    rho_ln * vn * vbar[d]
	    + p_hat * nor[d]
	    - llf * (mom2[d] - mom1[d]);
	}
      
      flux[ener_eq] =
	rho_ln * vn * hhat
	- llf * diss;
      
      return lambda_max;
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static mfem::real_t ComputeFaceFluxKernel2(const GasModelT &gasModel,const mfem::real_t *state1,
                                                                const mfem::real_t *state2, const mfem::real_t *nor,
                                                                mfem::real_t *flux)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();
      
      Theseus::PointStateView S1{state1};
      Theseus::PointStateView S2{state2};
    
      const mfem::real_t rho1 = gasModel.density(S1);
      const mfem::real_t rho2 = gasModel.density(S2);
      const mfem::real_t rho_mean = 0.5 * (rho1 + rho2);
      const mfem::real_t rho_ln = Kernels::ComputeLogMean(rho1, rho2, 1e-4);
      const mfem::real_t drho = rho2 - rho1;
      mfem::real_t mom[3] = {0.0, 0.0, 0.0};
      mfem::real_t mom1[3] = {0.0, 0.0, 0.0};
      mfem::real_t mom2[3] = {0.0, 0.0, 0.0};
      mfem::real_t hhat = 0.0;
      mfem::real_t diss = 0.0;
      mfem::real_t v21 = 0.0;
      mfem::real_t v22 = 0.0;
      mfem::real_t vn = 0.0;
      mfem::real_t nor_mag = 0.0;

      for(int idim = 0;idim < dim;idim++){
        nor_mag += nor[idim]*nor[idim];
        mom1[idim] = gasModel.momentum(S1, idim);
        mom2[idim] = gasModel.momentum(S2, idim);
        const mfem::real_t v1 = mom1[idim]/rho1;
        const mfem::real_t v2 = mom2[idim]/rho2;
        const mfem::real_t vbar = 0.5 * (v1 + v2);
        const mfem::real_t dv = v2 - v1;
        v21 += v1*v1;
        v22 += v2*v2;
        vn += vbar * nor[idim];
        mom[idim] = rho_ln * vbar;
        hhat += -0.25 * (v1*v1 + v2*v2) + vbar * vbar;
        diss += 0.5 * drho * v1*v2 + rho_mean * dv * vbar;
      }
      nor_mag = std::sqrt(nor_mag);
      
      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);

      const mfem::real_t vmag1 = std::sqrt(v21);
      const mfem::real_t vmag2 = std::sqrt(v22);

      const mfem::real_t c1 = gasModel.sound_speed(S1);
      const mfem::real_t c2 = gasModel.sound_speed(S2);

      const mfem::real_t lambda_max = Kernels::rmax(vmag1 + c1, vmag2 + c2);

      const mfem::real_t beta1 = 0.5 * rho1 / p1;
      const mfem::real_t beta2 = 0.5 * rho2 / p2;
      const mfem::real_t beta_ln = Kernels::ComputeLogMean(beta1, beta2, 1e-4);
      
      const mfem::real_t p_hat = 0.5 * (rho1 + rho2) / (beta1 + beta2);

      // Use the average gamma for now
      // TODO: Craft KEPEC fluxes for LTE/NLTE
      const mfem::real_t gm11 = gasModel.gamma(S1);
      const mfem::real_t gm12 = gasModel.gamma(S2); 
      const mfem::real_t gm1_av_inv = 2.0/(gm11 + gm12 - 2.0);
      
      hhat += 0.5 / beta_ln * gm1_av_inv + p_hat / rho_ln;
      diss += 0.5 * drho * gm1_av_inv / beta_ln + 0.5 * rho_mean * gm1_av_inv * (1.0 / beta2 - 1.0 / beta1);
      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;
      
      flux[mass_eq] = rho_ln * vn - 0.5 * lambda_max * (rho2 - rho1) * nor_mag;
      for (int d = 0; d < dim; d++)
        {
          flux[mom0_eq + d] = vn * mom[d] + p_hat * nor[d] - 0.5 * lambda_max * (mom2[d]-mom1[d]) * nor_mag;
        }
      flux[ener_eq] = rho_ln * vn * hhat - 0.5 * lambda_max * diss * nor_mag;
      
      return lambda_max;
    }
    struct InviscidFlux {
 
      template<typename GasModelT>
      MFEM_HOST_DEVICE inline mfem::real_t ComputeVolumeFlux(const GasModelT &gasModel,
                                                       const mfem::real_t *q1, const mfem::real_t *q2,
                                                       const mfem::real_t *met1, const mfem::real_t *met2,
                                                       mfem::real_t *F_tilde) const{
        return ComputeVolumeFluxKernel(gasModel, q1, q2, met1, met2, F_tilde); 
      }

      template<typename GasModelT>
      MFEM_HOST_DEVICE inline mfem::real_t ComputeFaceFlux(const GasModelT &gasModel,const mfem::real_t *qminus,
                                                     const mfem::real_t *qplus, const mfem::real_t *nor,
                                                     mfem::real_t *flux) const {
        return ComputeFaceFluxKernel(gasModel, qminus, qplus, nor, flux); 
      }
    };
  };
}
