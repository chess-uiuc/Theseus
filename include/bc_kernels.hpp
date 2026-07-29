// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "bc_cache_utilities.hpp"
#include "Flow.hpp"
#include "NavierStokesFlux.hpp"

namespace Theseus {

  namespace BC {


    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE inline
    void ComputeBdrFaceGradFlux(const DeviceCacheT &dc,
                                const Theseus::BCDescriptor &bc,
                                const mfem::real_t *state1,
                                mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const int neq = dc.num_equations;
      const int dim = dc.dim;
      
      for (int q = 0; q < neq; ++q)
        {
          fluxN[q] = mfem::real_t(0);
        }

      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::NoSlipAdiab:
          {
            const mfem::real_t *vector_data = dc.bc_vector_d;
            const mfem::real_t *Vwall = vector_data + bc.data_index;
            // unused - but be aware
            // const mfem::real_t *wallHeat = vector_data + bc.data_index + dim;

            // Note this must be entropy state
            Theseus::PointStateView S{state1};
            Theseus::PointStateViewRW F{fluxN};

            const mfem::real_t v = -gas.energy(S);

            // Build the same provisional "boundary" state as legacy, then subtract state1.
            F.set_mass(gas.L, gas.mass(S));
            for(int idim = 0;idim < dim;idim++){
              F.set_momentum(gas.L, idim, Vwall[idim] * v);
            }
            F.set_energy(gas.L, -v);
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] -= state1[q];
              }
            return;
          }

        default:
          {
            // Conservative placeholder for unsupported BCs.
            for (int q = 0; q < neq; ++q)
              {
                fluxN[q] = mfem::real_t(0);
              }
            return;
          }
        }
    }


    template<typename GasModelT>
    MFEM_HOST_DEVICE
    mfem::real_t SlipWallInviscidFluxKernel(const GasModelT &gasModel, const mfem::real_t *state1,
                                      const mfem::real_t *nor, mfem::real_t *fluxN)
    {
      
      mfem::real_t unit_nor[Theseus::MAXDIM];
      mfem::real_t state2[Theseus::MAXEQ];
      const int dim = gasModel.L.dim;
      const int neq = gasModel.L.nequations();
      for(int idim = 0;idim < dim;idim++)
        unit_nor[idim] = nor[idim];
      for(int ieq = 0;ieq < neq;ieq++){
        state2[ieq] = state1[ieq];
        fluxN[ieq] = 0.0;
      }
      Theseus::Kernels::Normalize(dim, unit_nor);
      Theseus::PointStateViewRW S{state2};
      Theseus::Flow::RotateState(gasModel.L, unit_nor, S);
      const mfem::real_t p_star = Theseus::Flow::slipwall_pstar(S, gasModel);
      const mfem::real_t v = gasModel.velocity(S, 0); // the "x" component is v*n
      const mfem::real_t c = gasModel.sound_speed(S);
      const int mom_eq = gasModel.L.eq_mom0;
      for(int idim = 0;idim < dim;idim++)
        fluxN[mom_eq+idim] = p_star * nor[idim];
      return std::abs(v) + c;
    }

    template<typename GasModelT>
    MFEM_HOST_DEVICE
    mfem::real_t NoSlipAdiabWallFluxKernel(const GasModelT &gasModel, const mfem::real_t *state1,
                                     const mfem::real_t *gradPrim_x, const mfem::real_t *gradPrim_y,
                                     const mfem::real_t *gradPrim_z,
                                     const mfem::real_t *nor, const mfem::real_t vWall[Theseus::MAXDIM],
                                     const mfem::real_t qWall, mfem::real_t *fluxN)
    {
      
      mfem::real_t unit_nor[Theseus::MAXDIM];
      mfem::real_t state2[Theseus::MAXEQ];
      mfem::real_t visc_flux[Theseus::MAXEQ][Theseus::MAXDIM];
      const int dim = gasModel.L.dim;
      const int neq = gasModel.L.nequations();
      mfem::real_t normag = 0.0;
      for(int idim = 0;idim < dim;idim++){
        unit_nor[idim] = nor[idim];
        normag += nor[idim]*nor[idim];
      }
      normag = Theseus::Kernels::rsqrt(normag);
      
      for(int ieq = 0;ieq < neq;ieq++){
        state2[ieq] = state1[ieq];
        fluxN[ieq] = 0.0;
      }
      Theseus::Kernels::Normalize(dim, unit_nor);
      Theseus::PointStateViewRW S{state2};
      Theseus::Flow::RotateState(gasModel.L, unit_nor, S);
      const mfem::real_t p_star = Theseus::Flow::slipwall_pstar(S, gasModel);
      const mfem::real_t v = gasModel.velocity(S, 0); // the "x" component is v*n
      const mfem::real_t c = gasModel.sound_speed(S);
      const int mom_eq = gasModel.L.eq_mom0;
      for(int idim = 0;idim < dim;idim++)
        fluxN[mom_eq+idim] = p_star * nor[idim];
      // Inviscid part is done, now for the viscous part
      mfem::real_t qn = qWall * normag;
      NavierStokesFlux::ComputeViscousFluxKernel(gasModel, state1, gradPrim_x, gradPrim_y,
                                                 gradPrim_z, visc_flux);
      mfem::real_t vflux_n[Theseus::MAXEQ];
      for(int j = 0;j < neq;j++){
        vflux_n[j] = 0.0;
        for(int idim = 0;idim < dim;idim++){
          vflux_n[j] += nor[idim]*visc_flux[j][idim];
        }
      }
      const int ener_eq = gasModel.L.eq_energy;
      vflux_n[ener_eq] = qn;
      for(int idim = 0;idim < dim;idim++){
        vflux_n[ener_eq] += vWall[idim]*vflux_n[mom_eq+idim];
      }
      for(int j = 0; j < neq;j++){
        fluxN[j] -= vflux_n[j];
      }
      return std::abs(v) + c;
    }

    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE
    mfem::real_t ApplyBoundaryConditionInviscid(const DeviceCacheT &dc,
                                          const Theseus::BCDescriptor &bc,
                                          const mfem::real_t *state1,
                                          const mfem::real_t *nor,
                                          mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const mfem::real_t *scalar_data = dc.bc_scalar_d;
      const mfem::real_t *vector_data = dc.bc_vector_d;
      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::SlipWall:
          return SlipWallInviscidFluxKernel(gas, state1, nor, fluxN);
          
        case Theseus::BCType::SupersonicOutflow:
          return dc.iflux.ComputeFaceFlux(gas, state1, state1, nor, fluxN);
          
        case Theseus::BCType::SupersonicInflow:
          {
            const mfem::real_t *bc_state = vector_data + bc.data_index;
            return dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
          }
          
        case Theseus::BCType::Symmetry:
          {
            const int neq = dc.num_equations;
            Theseus::PointStateView S{state1};
            mfem::real_t bc_state[Theseus::MAXEQ];
            for(int ieq = 0;ieq < neq;ieq++){
              bc_state[ieq] = state1[ieq];
            }
            Theseus::PointStateViewRW S2{bc_state};
            const int dim = dc.dim;
            mfem::real_t unorm[Theseus::MAXDIM];
            mfem::real_t mom[Theseus::MAXDIM];
            for(int idim = 0;idim < dim;idim++){
              unorm[idim] = nor[idim];
              mom[idim] = S.momentum(gas.L, idim);
            }
            Theseus::Kernels::Normalize(dim, unorm);
            mfem::real_t nv = Theseus::Kernels::Dot(dim, mom, unorm);
            for(int idim = 0;idim < dim;idim++){
              mfem::real_t mm = -2.0*nv*unorm[idim] + mom[idim];
              S2.set_momentum(gas.L, idim, mm);
            }
            return dc.iflux.ComputeFaceFlux(gas, state1, bc_state, nor, fluxN);
          }
        default:
          {
            const int neq = dc.num_equations;
            for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
            return 0.0;
          }
        }
      return 0.0;
    }

    template <typename DeviceCacheT>
    MFEM_HOST_DEVICE
    mfem::real_t ApplyViscousBoundaryCondition(const DeviceCacheT &dc,
                                         const Theseus::BCDescriptor &bc,
                                         const mfem::real_t *state1,
                                         const mfem::real_t *gradPrim_x,
                                         const mfem::real_t *gradPrim_y,
                                         const mfem::real_t *gradPrim_z,
                                         const mfem::real_t *nor,
                                         mfem::real_t *fluxN)
    {
      const auto &gas = dc.gas;
      const int dim = dc.dim;
      const mfem::real_t *scalar_data = dc.bc_scalar_d;
      const mfem::real_t *vector_data = dc.bc_vector_d;
      switch (static_cast<Theseus::BCType>(bc.type))
        {
        case Theseus::BCType::NoSlipAdiab:
          {
            const mfem::real_t *bc_vec_data = vector_data + bc.data_index;
            mfem::real_t vWall[Theseus::MAXDIM];
            for(int idim=0;idim < dim;idim++){
              vWall[idim] = bc_vec_data[idim];
            }
            const mfem::real_t qWall = bc_vec_data[dim];
            return NoSlipAdiabWallFluxKernel(gas, state1, gradPrim_x, gradPrim_y,
                                             gradPrim_z, nor, vWall, qWall, fluxN);
          }
        default:
          {
            const int neq = dc.num_equations;
            for (int eq = 0; eq < neq; ++eq) { fluxN[eq] = 0.0; }
            return 0.0;
          }
        }
      return 0.0;
    }
  }
}
