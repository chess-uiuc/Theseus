// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include "Physics.hpp"
#include "GasState.hpp"

namespace Theseus
{

  // ============================================================================
  // Transport: Simple transport, also support Sutherland model (for now)
  // ============================================================================
  
  struct Transport
  {
    
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t viscosity(const PhysicsConstants &phys, const StateLayout &L,
                            const EOSType &eos, const StateViewType &S) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      const mfem::real_t temptr = eos.temperature(phys, L, S);
      const mfem::real_t Trel = temptr / phys.T0;
      const mfem::real_t T0pTs = phys.T0 + phys.Ts;
      return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (temptr + phys.Ts);
#else
      return phys.mu;
#endif
    }

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t bulk_viscosity(const PhysicsConstants &phys, const StateLayout &L,
                                 const EOSType &eos, const StateViewType &S) const
    {
      return phys.mu_bulk;
    }

    // Thermal cond kappa = mu * cp / Pr
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t thermal_conductivity(const PhysicsConstants &phys, const StateLayout &L,
                                       const EOSType &eos, const StateViewType &S) const
    {
      return viscosity(phys, L, eos, S) * eos.cp(phys, L, S) * phys.PrInverse;
    }
  };
  // TODO: Consider refactoring; would be better (explicit) design
  // struct SutherlandTransport {***} using phys.mu0, phys.T0, phys.Ts, etc.
}
