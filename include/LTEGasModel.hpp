// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "Physics.hpp"
#include "GasState.hpp"
#include "LTETable.hpp"
#include "LTEEOS.hpp"
#include "LTETransport.hpp"

using namespace Theseus::LTETable;

namespace Theseus
{

// ============================================================================
// LTEGasModel: Encapsulate/wrap LTE EOS/Transport in standard interface
// ============================================================================
  template <typename EOSImpl, typename TransportImpl>
  struct LTEGasModel
  {

    Theseus::PhysicsConstants phys;
    Theseus::StateLayout L;
    LTETables T;
    EOSImpl eos;
    TransportImpl transport;

    MFEM_HOST_DEVICE LTEGasModel() = default;

    MFEM_HOST_DEVICE
    LTEGasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
                const LTETables &T_in, const EOSImpl &eos_in, const TransportImpl &tr_in)
      : phys(phys_in), L(L_in), T(T_in), eos(eos_in), transport(tr_in)
    { };

    MFEM_HOST_DEVICE
    LTEGasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
                const LTETables &T_in)
      : phys(phys_in), L(L_in), T(T_in)
    { };

    template<typename HostDataT>
    MFEM_HOST_DEVICE LTEGasModel<EOSImpl, TransportImpl>  to_device(HostDataT &host_data) {
      LTEGasModel<EOSImpl, TransportImpl> retVal(phys, L, T, eos, transport);
      T.tables = {
        host_data.lteTableData->lte_table.Read(),
        host_data.lteTableData->inv_table.Read(),
        host_data.lteTableData->rho_grid.Read(),
        host_data.lteTableData->T_grid.Read(),
        host_data.lteTableData->e_grid.Read()
      };
      return retVal;
    }

    // Utilities and constants etc
    MFEM_HOST_DEVICE
    inline int num_equations() const
    { return L.nequations(); };

    MFEM_HOST_DEVICE
    inline int dim() const
    { return L.dim; };

    // State Access
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t velocity(const StateView &S, int d) const
    { return S.velocity(L,d);};

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t momentum(const StateView &S, int d) const
    { return S.momentum(L,d);};

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t density(const StateView &S) const
    {
      return eos.density(phys, L, S, T);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t mass(const StateView &S) const
    {
      return eos.density(phys, L, S, T);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t scalar(const StateView &S, int s) const
    {
      return S.scalar(L, s);
    };

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t energy(const StateView &S) const
    {
      return S.energy(L);
    };

    // --- Thermodynamics ------------------------------------------------------
    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t pressure(const StateView &S) const
    {
      return eos.pressure(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t gamma(const StateView &S) const
    {
      return eos.gamma(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t cp(const StateView &S) const
    {
      return eos.cp(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t R_gas(const StateView &S) const
    {
      return eos.R_gas(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t temperature(const StateView &S) const
    {
      return eos.temperature(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t sound_speed(const StateView &S) const
    {
      return eos.sound_speed(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t kinetic_energy_density(const StateView &S) const
    {
      return eos.kinetic_energy_density(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t internal_energy_from_pressure(const StateView &S, mfem::real_t pressure) const
    {
        // rho*e = rho*E - 0.5*rho*|u|^2
      return eos.internal_energy_from_pressure(phys, L, S, pressure, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t specific_internal_energy(const StateView &S) const
    {
      return eos.specific_internal_energy(phys, L, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline void grad_temperature(const StateView &S,
                                 const mfem::real_t *grad_r, const mfem::real_t *grad_p,
                                 mfem::real_t *grad_t) const
    {
      return eos.grad_temperature(phys, L, S, grad_r, grad_p, grad_t, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t entropy(const StateView &S) const
    {
      return eos.entropy(phys, L, S, T);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_state(const InStateView &S, OutStateView &E) const
    {
      return eos.entropy_state(phys, L, S, E, T);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void grad_entropy_to_grad_prim(const InStateView &S, const InStateView &dS,
                                          OutStateView &dPrim) const
    {
      return eos.grad_entropy_to_grad_prim(phys, L, S, dS, dPrim, T);
    }

    template<typename InStateView, typename OutStateView>
    MFEM_HOST_DEVICE
    inline void entropy_to_conserved(const InStateView &Se, OutStateView &Sc) const
    {
      return eos.entropy_to_conserved(phys, L, Se, Sc, T);
    }

    template<typename InStateView, typename OutStateView>
    inline void primitive_to_conserved(const InStateView &prim, OutStateView &cons) const
    {
      return eos.primitive_to_conserved(phys, L, prim, cons, T);
    }

    // --- Transport -----------------------------------------------------------

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t viscosity(const StateView &S) const
    {
      return transport.viscosity(phys, L, eos, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t bulk_viscosity(const StateView &S) const
    {
      return transport.bulk_viscosity(phys, L, eos, S, T);
    }

    template<typename StateView>
    MFEM_HOST_DEVICE
    inline mfem::real_t thermal_conductivity(const StateView &S) const
    {
      return transport.thermal_conductivity(phys, L, eos, S, T);
    }

  };

  using LTEGas = LTEGasModel<LTEGasEOS, LTETransport>;

} // namespace Theseus
