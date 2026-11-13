// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"

namespace Theseus
{
  namespace Utilities
  {
    template<typename DeviceCacheT, typename OperatorCacheT>
    inline void ComputeVolumeAverages(DeviceCacheT &dc, OperatorCacheT &op_cache, const mfem::real_t *Ue_d, MPI_Comm comm)
    {
        // Device cache parameters
        const int ne = dc.num_elements;
        const int ndof = dc.ndof_scalar_el;
        const int neq = dc.num_equations;
        const int estride = ndof*neq;
        const mfem::real_t *qWts_d = dc.elQWgts_d;
        const int dim = dc.dim;
        const int mass = dc.gas.L.eq_mass, momentum = dc.gas.L.eq_mom0, energy = dc.gas.L.eq_energy;

        mfem::Vector elMass_integral(ne);
        mfem::Vector elMom_x_integral(ne);
        mfem::Vector elEnergy_integral(ne);

        elMass_integral.UseDevice();
        elMom_x_integral.UseDevice();
        elEnergy_integral.UseDevice();

        mfem::real_t *elMass_int_d = elMass_integral.Write();
        mfem::real_t *elMom_x_int_d = elMom_x_integral.Write();
        mfem::real_t *elEnergy_int_d = elEnergy_integral.Write();

        // Kernel to compute Volume averaged mass, rho u and rho e
        mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
        {
          const mfem::real_t *u_el = Ue_d + e * estride;
          const mfem::real_t *qWgt = qWts_d + e * ndof;

          mfem::real_t mass_int = 0.0;
          mfem::real_t mom_x_int = 0.0;
          mfem::real_t en_int = 0.0;

          for(int ep = 0;ep < ndof;ep++){
              const auto rho = u_el[mass*ndof + ep];
              const auto mom_x = u_el[momentum*ndof + ep];
              mfem::real_t momentum_squared = 0.0;
              for (int d = 0; d < dim; ++d)
                {
                  const auto m = u_el[(momentum + d)*ndof + ep];
                  momentum_squared += m*m;
                }
              const auto rhoe = u_el[energy*ndof + ep] - 0.5*momentum_squared/rho;

              mass_int += rho * qWgt[ep];
              mom_x_int += mom_x * qWgt[ep];
              en_int += rhoe * qWgt[ep];
          }

          elMass_int_d[e]   = mass_int;
          elMom_x_int_d[e] = mom_x_int;
          elEnergy_int_d[e] = en_int;
        });

        const mfem::real_t *elMass_int_h = elMass_integral.HostRead();
        const mfem::real_t *elMom_x_int_h = elMom_x_integral.HostRead();
        const mfem::real_t *elEnergy_int_h = elEnergy_integral.HostRead();

        mfem::real_t sendbuf[3] = {0.0, 0.0, 0.0};

        for (int e=0; e < ne; e++)
        {
          sendbuf[0] += elMass_int_h[e];
          sendbuf[1] += elMom_x_int_h[e];
          sendbuf[2] += elEnergy_int_h[e];
        }

        mfem::real_t recvbuf[3] = {0.0, 0.0, 0.0};

        MPI_Allreduce(sendbuf, recvbuf, 3, mfem::MPITypeMap<mfem::real_t>::mpi_type, MPI_SUM, comm);

        auto *volume_avg_h = op_cache.volume_avg_state.HostReadWrite();

        volume_avg_h[0] = recvbuf[0] / dc.total_volume;
        volume_avg_h[1] = recvbuf[1] / dc.total_volume;
        volume_avg_h[dim+1] = recvbuf[2] / dc.total_volume;

        op_cache.volume_avg_state.Read();
    };
  }
}
