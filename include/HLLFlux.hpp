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

  namespace HLLFlux
  {
    // This is a central flux method: also used by LLF, needs centralized
    // TODO: Centralize the central flux method
    template<typename GasT>
    MFEM_HOST_DEVICE
    inline static void ComputeVolumeFluxKernel(const GasT &gas,
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

      for(int ieq = 0;ieq < neq;ieq++){
        F_tilde[ieq] = inv_flux_bar[ieq];
      }

    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE inline static void
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

      mfem::real_t un1 = 0.0;
      mfem::real_t un2 = 0.0;
      mfem::real_t nor_mag2 = 0.0;

      for (int d = 0; d < dim; ++d)
        {
          un1 += gasModel.velocity(S1, d) * nor[d];
          un2 += gasModel.velocity(S2, d) * nor[d];
          nor_mag2 += nor[d] * nor[d];
        }

      const mfem::real_t nor_mag = Theseus::Kernels::rsqrt(nor_mag2);

      const mfem::real_t c1 = gasModel.sound_speed(S1) * nor_mag;
      const mfem::real_t c2 = gasModel.sound_speed(S2) * nor_mag;

      const mfem::real_t sL1 = un1 - c1;
      const mfem::real_t sL2 = un2 - c2;
      const mfem::real_t sR1 = un1 + c1;
      const mfem::real_t sR2 = un2 + c2;

      const mfem::real_t sL = (sL1 < sL2) ? sL1 : sL2;
      const mfem::real_t sR = (sR1 > sR2) ? sR1 : sR2;

      for (int ieq = 0; ieq < neq; ++ieq)
        {
          mfem::real_t fn1 = 0.0;
          mfem::real_t fn2 = 0.0;
          for (int d = 0; d < dim; ++d)
            {
              fn1 += inv_flux_1[ieq][d] * nor[d];
              fn2 += inv_flux_2[ieq][d] * nor[d];
            }
          if (sL >= 0.0)
            {
              flux[ieq] = fn1;
            }
          else if (sR <= 0.0)
            {
              flux[ieq] = fn2;
            }
          else
            {
              flux[ieq] =
                (sR * fn1 - sL * fn2 + sL * sR * (state2[ieq] - state1[ieq]))
                / (sR - sL);
            }
        }

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
