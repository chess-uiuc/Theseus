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
  // LTETransport: LTE lookup for transport properties
  // ============================================================================

  struct LTETransport
  {

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t viscosity(const Theseus::PhysicsConstants &phys,
                                  const Theseus::StateLayout &L,
                                  const EOSType &eos, const StateViewType &S,
                                  const LTETables &lteTables) const
    {
#ifdef SUTHERLAND
      // mu0 * T0pTs / (T + Ts) * (T / T0) * std::sqrt(T / T0);
      const mfem::real_t temptr = eos.temperature(phys, L, S, thermoTables);
      const mfem::real_t Trel = temptr / phys.T0;
      const mfem::real_t T0pTs = phys.T0 + phys.Ts;
      return phys.mu0 * T0pTs * Trel * std::sqrt(Trel) / (temptr + phys.Ts);
#else
      return eos.property_lookup(lteTables.L.mu_idx, phys, L, S, lteTables);
#endif
    }

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t bulk_viscosity(const Theseus::PhysicsConstants &phys,
                                       const Theseus::StateLayout &L,
                                       const EOSType &eos, const StateViewType &S,
                                       const LTETables &lteTables) const
    {
      return phys.mu_bulk;
    }

    // Thermal cond kappa = mu * cp / Pr
    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t thermal_conductivity(const Theseus::PhysicsConstants &phys,
                                             const Theseus::StateLayout &L,
                                             const EOSType &eos, const StateViewType &S,
                                             const LTETables &lteTables) const
    {
      return eos.property_lookup(lteTables.L.lambda_idx, phys, L, S, lteTables);
    }
  };
  // TODO: Consider refactoring; would be better (explicit) design
  // struct SutherlandTransport {***} using phys.mu0, phys.T0, phys.Ts, etc.
}
