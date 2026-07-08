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

  namespace LaxFriedrichsFlux
  {
  
    // This is Riemann solver that computes the numerical flux for 2 point states
    // Will be ctx.iflux.ComputeVolumeFlux
    template<typename GasT>
    MFEM_HOST_DEVICE
    inline static mfem::real_t ComputeVolumeFluxKernel(const GasT &gas,
                                                 const mfem::real_t* q1,
                                                 const mfem::real_t* q2,
                                                 const mfem::real_t* met1,
                                                 const mfem::real_t* met2,
                                                 mfem::real_t* F_tilde)
    {
      const int dim = gas.dim();
      const int neq = gas.num_equations();

      // mean metric row
      mfem::real_t met[3] = {0,0,0};
      Kernels::ComputeMeanVec(met1, met2, met, dim);
      Theseus::PointStateView S1{q1};
      Theseus::PointStateView S2{q2};
      mfem::real_t inv_flux_1[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t inv_flux_2[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t inv_flux_bar[Theseus::MAXEQ];

      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q1, inv_flux_1);
      NavierStokesFlux::ComputeInviscidFluxKernel(gas, q2, inv_flux_2);
      for(int ieq=0;ieq < neq;ieq++){
        inv_flux_bar[ieq] = 0;
        for(int idim = 0;idim < dim;idim++){
          inv_flux_bar[ieq] += 0.5*(inv_flux_1[ieq][idim] + inv_flux_2[ieq][idim])*met[idim];
        }
      }

      mfem::real_t vn_1 = 0;
      mfem::real_t vn_2 = 0;
      mfem::real_t mnorm = 0;
      for (int d=0; d<dim; ++d)
        {
          const mfem::real_t v1 = gas.velocity(S1, d);
          const mfem::real_t v2 = gas.velocity(S2, d);
          vn_1 += v1*met[d];
          vn_2 += v2*met[d];
          mnorm += met[d]*met[d];
        }
      vn_1 = Theseus::Kernels::rabs(vn_1);
      vn_2 = Theseus::Kernels::rabs(vn_2);
      mnorm = Theseus::Kernels::rsqrt(mnorm);
      const mfem::real_t c1 = gas.sound_speed(S1)*mnorm;
      const mfem::real_t c2 = gas.sound_speed(S2)*mnorm;
      const mfem::real_t lambda_max = Kernels::rmax(vn_1 + c1, vn_2 + c2);

      for(int ieq = 0;ieq < neq;ieq++){
        F_tilde[ieq] = inv_flux_bar[ieq];
      }

      return lambda_max;
    }
  
    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static mfem::real_t
    ComputeFaceFluxKernel(const GasModelT &gasModel,
                          const mfem::real_t *state1,
                          const mfem::real_t *state2,
                          const mfem::real_t *nor,
                          mfem::real_t *flux)
    {
      const int dim = gasModel.dim();
      const int neq = gasModel.num_equations();

      Theseus::PointStateView S1{state1};
      Theseus::PointStateView S2{state2};
    
      mfem::real_t inv_flux_1[Theseus::MAXEQ][Theseus::MAXDIM];
      mfem::real_t inv_flux_2[Theseus::MAXEQ][Theseus::MAXDIM];

      NavierStokesFlux::ComputeInviscidFluxKernel(gasModel, state1, inv_flux_1);
      NavierStokesFlux::ComputeInviscidFluxKernel(gasModel, state2, inv_flux_2);

      mfem::real_t vn1 = 0.0;
      mfem::real_t vn2 = 0.0;
      mfem::real_t nor_mag2 = 0.0;

      for (int d = 0; d < dim; ++d)
        {
          vn1 += gasModel.velocity(S1, d) * nor[d];
          vn2 += gasModel.velocity(S2, d) * nor[d];
          nor_mag2 += nor[d] * nor[d];
        }

      const mfem::real_t nor_mag = Theseus::Kernels::rsqrt(nor_mag2);

      vn1 = Theseus::Kernels::rabs(vn1);
      vn2 = Theseus::Kernels::rabs(vn2);

      const mfem::real_t c1 = gasModel.sound_speed(S1);
      const mfem::real_t c2 = gasModel.sound_speed(S2);
      const mfem::real_t lambda_max =
        Theseus::Kernels::rmax(vn1 + c1 * nor_mag,
                               vn2 + c2 * nor_mag);

      for (int ieq = 0; ieq < neq; ++ieq)
        {
          mfem::real_t fn1 = 0.0;
          mfem::real_t fn2 = 0.0;

          for (int d = 0; d < dim; ++d)
            {
              fn1 += inv_flux_1[ieq][d] * nor[d];
              fn2 += inv_flux_2[ieq][d] * nor[d];
            }

          const mfem::real_t central_flux = 0.5 * (fn1 + fn2);
          const mfem::real_t jump = state2[ieq] - state1[ieq];
 
          flux[ieq] = central_flux - 0.5 * lambda_max * jump;
        }
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
