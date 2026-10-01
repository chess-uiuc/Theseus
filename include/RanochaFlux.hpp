// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "NavierStokesFlux.hpp"
#include "theseus_kernels.hpp"
#include "DissipationOperator.hpp"

namespace Theseus
{

  namespace RanochaFlux
  {
    // Two-point entropy-conservative numerical flux for the volume discretization.
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
      
      mfem::real_t vbar[3] = {0,0,0};
      mfem::real_t p_prod_mean[3] = {0,0,0};
      mfem::real_t ke_flux = 0;
      mfem::real_t vn = 0;
      
      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);
      const mfem::real_t pbar = mfem::real_t(0.5) * (p1 + p2);
      
      for (int d=0; d<dim; ++d)
        {
          const mfem::real_t v1 = gasModel.velocity(S1, d);
          const mfem::real_t v2 = gasModel.velocity(S2, d);
          vbar[d] = mfem::real_t(0.5)*(v1+v2);
          vn   += vbar[d] * met[d];

          ke_flux += mfem::real_t(0.5) * (v1*v2);
          p_prod_mean[d] = mfem::real_t(0.5) * (p1*v2 + p2*v1);
        }

      const mfem::real_t inv_beta_ln = (p1 * p2) / Kernels::ComputeLogMean(rho1 * p2, rho2 * p1, 1e-4);

      mfem::real_t inv_gm1 = mfem::real_t(1.0) / (gasModel.gamma(S1) - mfem::real_t(1.0));
      mfem::real_t ie_flux = inv_beta_ln * inv_gm1;

      // F_tilde layout: [rho, rhoV, rhoE]
      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;
      F_tilde[mass_eq] = rho_ln * vn;
      F_tilde[ener_eq] = 0.0;
      for (int d=0; d<dim; ++d)
      {
        F_tilde[mom0_eq + d] = rho_ln * vn * vbar[d] + pbar * met[d];
        F_tilde[ener_eq] += p_prod_mean[d] * met[d];
      }
      F_tilde[ener_eq] += rho_ln * (ie_flux + ke_flux) * vn;
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
      const mfem::real_t rho_ln = Kernels::ComputeLogMean(rho1, rho2, 1e-4);
      mfem::real_t vbar[3] = {0.0, 0.0, 0.0};
      mfem::real_t p_prod_mean[3] = {0.0, 0.0, 0.0};
      mfem::real_t diss[Theseus::MAXEQ] = {0.,0.,0.,0.,0.};
      mfem::real_t ke_flux = 0.0;
      mfem::real_t vn = 0.0;

      const mfem::real_t p1 = gasModel.pressure(S1);
      const mfem::real_t p2 = gasModel.pressure(S2);
      const mfem::real_t pbar = 0.5 * (p1 + p2);

      for(int idim = 0;idim < dim;idim++){
        const mfem::real_t v1 = gasModel.velocity(S1, idim);
        const mfem::real_t v2 = gasModel.velocity(S2, idim);
        vbar[idim] = 0.5 * (v1 + v2);
        vn += vbar[idim] * nor[idim];
        ke_flux += 0.5 * (v1*v2);
        p_prod_mean[idim] = 0.5 * (p1*v2 + p2*v1);
      }

      const mfem::real_t inv_beta_ln = (p1 * p2) / Kernels::ComputeLogMean(rho1 * p2, rho2 * p1, 1e-4);

      mfem::real_t inv_gm1 = mfem::real_t(1.0) / (gasModel.gamma(S1) - mfem::real_t(1.0));
      mfem::real_t ie_flux = inv_beta_ln * inv_gm1;

      const int mass_eq = gasModel.L.eq_mass;
      const int mom0_eq = gasModel.L.eq_mom0;
      const int ener_eq = gasModel.L.eq_energy;

      // Dissipative part of the flux based on Roe's approximate Riemann solver
      Roe_dissipation(gasModel, S1, S2, nor, diss);

      flux[mass_eq] = rho_ln * vn - diss[mass_eq];
      flux[ener_eq] = 0.0;
      for (int d = 0; d < dim; d++)
        {
          flux[mom0_eq + d] = rho_ln * vn * vbar[d] + pbar * nor[d] - diss[mom0_eq + d];
          flux[ener_eq] += p_prod_mean[d] * nor[d];
        }
      flux[ener_eq] += rho_ln * (ie_flux + ke_flux) * vn - diss[ener_eq];
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
