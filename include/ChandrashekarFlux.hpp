// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "NavierStokesFlux.hpp"
#include "theseus_kernels.hpp"

namespace Theseus
{

  namespace ChandrashekarFlux
  {
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasModelT>
    MFEM_HOST_DEVICE
    inline static void ComputeVolumeFluxKernel(const GasModelT &gasModel,
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
      
      for (int d=0; d<dim; ++d)
        {
          const mfem::real_t v1 = gasModel.velocity(S1, d);
          const mfem::real_t v2 = gasModel.velocity(S2, d);
          const mfem::real_t vbar = mfem::real_t(0.5)*(v1+v2);
          
          vn   += vbar * met[d];
          
          mom_hat[d] = rho_ln * vbar;
          
          h_hat += -mfem::real_t(0.25)*(v1*v1 + v2*v2) + vbar*vbar;
        }
      
      
      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);
      
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
      
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static void ComputeFaceFluxKernel(const GasModelT &gasModel,const mfem::real_t *state1,
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
      mfem::real_t vn = 0.0;
      mfem::real_t vn1 = 0.0;
      mfem::real_t vn2 = 0.0;
      mfem::real_t nor_norm_squared = 0.0;

      for(int idim = 0;idim < dim;idim++){
        mom1[idim] = gasModel.momentum(S1, idim);
        mom2[idim] = gasModel.momentum(S2, idim);
        const mfem::real_t v1 = mom1[idim]/rho1;
        const mfem::real_t v2 = mom2[idim]/rho2;
        const mfem::real_t vbar = 0.5 * (v1 + v2);
        const mfem::real_t dv = v2 - v1;
        vn += vbar * nor[idim];
        vn1 += v1 * nor[idim];
        vn2 += v2 * nor[idim];
        nor_norm_squared += nor[idim] * nor[idim];
        mom[idim] = rho_ln * vbar;
        hhat += -0.25 * (v1*v1 + v2*v2) + vbar * vbar;
        diss += 0.5 * drho * v1*v2 + rho_mean * dv * vbar;
      }
      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);

      const mfem::real_t nor_norm = Kernels::rsqrt(nor_norm_squared);
      const mfem::real_t lambda_max = Kernels::rmax(
        Kernels::rabs(vn1) + gasModel.sound_speed(S1) * nor_norm,
        Kernels::rabs(vn2) + gasModel.sound_speed(S2) * nor_norm);

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
      
      flux[mass_eq] = rho_ln * vn - 0.5 * lambda_max * (rho2 - rho1);
      for (int d = 0; d < dim; d++)
        {
          flux[mom0_eq + d] = vn * mom[d] + p_hat * nor[d]
            - 0.5 * lambda_max * (mom2[d]-mom1[d]);
        }
      flux[ener_eq] = rho_ln * vn * hhat - 0.5 * lambda_max * diss;
      
    }
    struct InviscidFlux {
 
      template<typename GasModelT>
      MFEM_HOST_DEVICE inline void ComputeVolumeFlux(const GasModelT &gasModel,
                                                       const mfem::real_t *q1, const mfem::real_t *q2,
                                                       const mfem::real_t *met1, const mfem::real_t *met2,
                                                       mfem::real_t *F_tilde) const{
        ComputeVolumeFluxKernel(gasModel, q1, q2, met1, met2, F_tilde);
      }

      template<typename GasModelT>
      MFEM_HOST_DEVICE inline void ComputeFaceFlux(const GasModelT &gasModel,const mfem::real_t *qminus,
                                                     const mfem::real_t *qplus, const mfem::real_t *nor,
                                                     mfem::real_t *flux) const {
        ComputeFaceFluxKernel(gasModel, qminus, qplus, nor, flux);
      }
    };
  };
}
