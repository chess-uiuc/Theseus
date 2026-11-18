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
#include "LTEHelmholtzGasEOS.hpp"
#include "LTEHelmholtzStateConversion.hpp"

using namespace Theseus::LTETable;

namespace Theseus
{

// ============================================================================
// LTEGasModel: Encapsulate/wrap LTE EOS/Transport in standard interface
// ============================================================================
  template <typename EOSImpl, typename TransportImpl, typename TableType = LTETables>
  struct LTEGasModel
  {

    // The current local boundary conversions/entropy trace use original LTE tables.
    static constexpr bool supports_local_characteristic = std::is_same_v<TableType, LTETables>;

    Theseus::PhysicsConstants phys;
    Theseus::StateLayout L;
    TableType T;
    EOSImpl eos;
    TransportImpl transport;

    MFEM_HOST_DEVICE LTEGasModel() = default;

    MFEM_HOST_DEVICE
    LTEGasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
                const TableType &T_in, const EOSImpl &eos_in, const TransportImpl &tr_in)
      : phys(phys_in), L(L_in), T(T_in), eos(eos_in), transport(tr_in)
    { };

    MFEM_HOST_DEVICE
    LTEGasModel(const PhysicsConstants &phys_in, const StateLayout &L_in,
                const TableType &T_in)
      : phys(phys_in), L(L_in), T(T_in)
    { };

    template<typename HostDataT>
    LTEGasModel<EOSImpl, TransportImpl, TableType>  to_device(HostDataT &host_data) {
      LTEGasModel<EOSImpl, TransportImpl, TableType> retVal(phys, L, T, eos, transport);
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

    // Restrict the forward bilinear EOS to one axis and invert its monotone
    // piecewise-linear pressure. No secant loop or acoustic-path integration.
    MFEM_HOST_DEVICE static inline int boundary_interval(const mfem::real_t *grid,
                                                         int size, mfem::real_t value)
    {
      int lower = 0;
      int upper = size - 1;
      while (upper - lower > 1)
        {
          const int middle = lower + (upper - lower) / 2;
          if (grid[middle] <= value)
            {
              lower = middle;
            }
          else
            {
              upper = middle;
            }
        }
      return lower;
    }

    MFEM_HOST_DEVICE inline mfem::real_t boundary_property(int property, mfem::real_t rho,
                                                          mfem::real_t temperature) const
    {
      const auto *r = T.tables.rho_grid;
      const auto *t = T.tables.T_grid;
      if (!(rho >= r[0] && rho <= r[T.L.nx - 1] &&
            temperature >= t[0] && temperature <= t[T.L.ny - 1]))
        {
          return std::numeric_limits<mfem::real_t>::quiet_NaN();
        }
      const int i = boundary_interval(r, T.L.nx, rho);
      const int j = boundary_interval(t, T.L.ny, temperature);
      const auto x = (rho - r[i]) / (r[i + 1] - r[i]);
      const auto y = (temperature - t[j]) / (t[j + 1] - t[j]);
      const auto *v = T.tables.lte_table;
      return (1 - y) * ((1 - x) * v[T.L.property_index(property, i, j)]
                            + x * v[T.L.property_index(property, i + 1, j)])
                 + y * ((1 - x) * v[T.L.property_index(property, i, j + 1)]
                            + x * v[T.L.property_index(property, i + 1, j + 1)]);
    }

    MFEM_HOST_DEVICE inline mfem::real_t boundary_pressure_root(mfem::real_t pressure,
        mfem::real_t fixed, bool density_axis) const
    {
      const auto *grid = density_axis ? T.tables.rho_grid : T.tables.T_grid;
      const int size = density_axis ? T.L.nx : T.L.ny;
      int lower = 0;
      int upper = size - 1;
      auto p_lower = boundary_property(T.L.P_idx, density_axis ? grid[lower] : fixed,
                                      density_axis ? fixed : grid[lower]);
      auto p_upper = boundary_property(T.L.P_idx, density_axis ? grid[upper] : fixed,
                                      density_axis ? fixed : grid[upper]);
      // No valid table root: propagate NaN to the solver's normal diagnostics.
      if (!(pressure >= p_lower && pressure <= p_upper))
        {
          return std::numeric_limits<mfem::real_t>::quiet_NaN();
        }
      while (upper - lower > 1)
        {
          const int middle = lower + (upper - lower) / 2;
          const auto value = boundary_property(T.L.P_idx, density_axis ? grid[middle] : fixed,
                                               density_axis ? fixed : grid[middle]);
          if (value <= pressure)
            {
              lower = middle;
              p_lower = value;
            }
          else
            {
              upper = middle;
              p_upper = value;
            }
        }
      return grid[lower] + (grid[upper] - grid[lower]) * (pressure - p_lower) / (p_upper - p_lower);
    }

    MFEM_HOST_DEVICE inline ThermodynamicState boundary_rho_p(mfem::real_t rho, mfem::real_t pressure) const
    {
      const auto temperature = boundary_pressure_root(pressure, rho, false);
      return {rho, temperature, boundary_property(T.L.e_idx, rho, temperature)};
    }

    MFEM_HOST_DEVICE inline ThermodynamicState boundary_p_T(mfem::real_t pressure, mfem::real_t temperature) const
    {
      const auto rho = boundary_pressure_root(pressure, temperature, true);
      return {rho, temperature, boundary_property(T.L.e_idx, rho, temperature)};
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

      if constexpr (std::is_same_v<TableType, LTETables>)
        {
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
        }
      else
        {
          const auto sample = HelmholtzEOS::At(T, density(state), recovered_temperature);
          LTEHelmholtzGasEOS::Require(sample.valid);
          if (request.pressure)
            {
              result.pressure = sample.pressure();
            }
          if (request.sound_speed)
            {
              result.sound_speed = std::sqrt(sample.sound_speed_squared());
            }
          if (request.viscosity)
            {
#ifdef SUTHERLAND
              result.viscosity = transport.viscosity_from_temperature(phys, recovered_temperature);
#else
              result.viscosity = sample.template transport<HelmholtzTable::Layout::mu_idx>(T);
#endif
            }
          if (request.thermal_conductivity)
            {
              result.thermal_conductivity = sample.template transport<HelmholtzTable::Layout::lambda_idx>(T);
            }
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
  using LTEHelmholtzGas = LTEGasModel<LTEHelmholtzGasEOS, LTETransport, HelmholtzTable::Tables>;

} // namespace Theseus
