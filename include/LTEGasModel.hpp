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
#include "LTEStateConversion.hpp"
#include "GasProperties.hpp"
#include "LTEPropertySample.hpp"

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
    LTEGasModel<EOSImpl, TransportImpl>  to_device(HostDataT &host_data) {
      LTEGasModel<EOSImpl, TransportImpl> retVal(phys, L, T, eos, transport);
      retVal.T.tables = {
        host_data.lteTableData->lte_table.Read(),
        host_data.lteTableData->inv_table.Read(),
        host_data.lteTableData->rho_grid.Read(),
        host_data.lteTableData->T_grid.Read(),
        host_data.lteTableData->e_grid.Read()
      };
      return retVal;
    }

    void ConservativeFromPhysical(const PhysicalStateInput &input, mfem::Vector &out) const
    {
      ValidatePhysicalState(input, L);
      const auto thermo = LTEPhysicalThermodynamics(eos, phys, L, input.thermo, T);
      mfem::Vector result;
      PackPhysicalState(input, thermo, L, result);
      // Check the energy actually representable after adding/subtracting kinetic
      // energy; cancellation at extreme velocities must not bypass table bounds.
      PointStateView state(result.GetData());
      ValidateLTERoundTrip(eos, phys, L, state, thermo.temperature, T);
      out = result;
    }

    template<typename StateView>
    GasProperties EvaluateProperties(const StateView &state,
                                     const GasPropertyRequest &request) const
    {
      GasProperties result;
      if (request.Empty())
        {
          return result;
        }

      const auto recovered_temperature = temperature(state);
      if (request.temperature)
        {
          result.temperature = recovered_temperature;
        }
      if (!request.pressure && !request.sound_speed &&
          !request.viscosity && !request.thermal_conductivity)
        {
          return result;
        }

      const LTEPropertySample sample(T, density(state), recovered_temperature);
      if (request.pressure)
        {
          result.pressure = sample.Value(T.L.P_idx);
        }
      if (request.sound_speed)
        {
          result.sound_speed = sample.Value(T.L.c_idx);
        }
      if (request.viscosity)
        {
#ifdef SUTHERLAND
          result.viscosity = transport.viscosity_from_temperature(phys, recovered_temperature);
#else
          result.viscosity = sample.Value(T.L.mu_idx);
#endif
        }
      if (request.thermal_conductivity)
        {
          result.thermal_conductivity = sample.Value(T.L.lambda_idx);
        }
      return result;
    }

    MFEM_HOST_DEVICE
    mfem::real_t isothermal_wall_beta(mfem::real_t wall_temperature) const
    {
      return eos.isothermal_wall_beta(phys, wall_temperature);
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
