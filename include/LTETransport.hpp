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

    MFEM_HOST_DEVICE
    mfem::real_t viscosity_from_temperature(const PhysicsConstants &phys,
                                           mfem::real_t temperature) const
    {
      const auto relative_temperature = temperature / phys.T0;
      return phys.mu0 * (phys.T0 + phys.Ts) * relative_temperature *
             std::sqrt(relative_temperature) / (temperature + phys.Ts);
    }

    template<typename EOSType, typename StateViewType>
    MFEM_HOST_DEVICE
    inline mfem::real_t viscosity(const Theseus::PhysicsConstants &phys,
                                  const Theseus::StateLayout &L,
                                  const EOSType &eos, const StateViewType &S,
                                  const LTETables &lteTables) const
    {
#ifdef SUTHERLAND
      const auto temperature = eos.temperature(phys, L, S, lteTables);
      return viscosity_from_temperature(phys, temperature);
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
