// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include <cmath>
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "Utilities.hpp"

namespace Theseus
{
  namespace Source
  {
    template<typename DeviceContext>
    MFEM_HOST_DEVICE
    inline void TargetStateSource(const DeviceContext &dc, const mfem::real_t *Ue, mfem::real_t *dUe)
    {
      const int ndof = dc.ndof_scalar_el;
      const int mass = dc.gas.L.eq_mass;
      const int momentum = dc.gas.L.eq_mom0;
      const int energy = dc.gas.L.eq_energy;
      const auto srho = (dc.target_state_d[mass] - dc.volume_avg_state_d[mass])*dc.tau_inv;
      const auto smom = (dc.target_state_d[momentum] - dc.volume_avg_state_d[momentum])*dc.tau_inv;
      for (int point = 0; point < ndof; ++point)
        {
          const auto rho = Ue[mass*ndof + point];
          const auto ux = Ue[momentum*ndof + point]/rho;
          dUe[mass*ndof + point] += srho;
          dUe[momentum*ndof + point] += smom;
          dUe[energy*ndof + point] += ux*smom;
        }

    };

    template<typename DeviceContext, typename OperatorContext>
    inline void PreProcessSourceTerms(DeviceContext &dc, OperatorContext &op_cache, const mfem::real_t* Ue_d, MPI_Comm comm)
    {
      if(dc.use_target_state_source)
      {
        Utilities::ComputeVolumeAverages(dc, op_cache, Ue_d, comm);
        MFEM_VERIFY(std::isfinite(op_cache.current_dt) && op_cache.current_dt > 0.0,
                    "Channel forcing requires a positive finite step size");
        dc.tau_inv = 1.0/(0.7*op_cache.current_dt);
        dc.volume_avg_state_d = op_cache.volume_avg_state.Read();
      }
    };

    template<typename DeviceContext>
    MFEM_HOST_DEVICE
    inline void AddSourceTerms(const DeviceContext &dc, const mfem::real_t *Ue, mfem::real_t *dUe)
    {
      if(dc.use_target_state_source && dc.tau_inv > 0.0)
      {
        Source::TargetStateSource(dc, Ue, dUe);
      }
    };
  }
}