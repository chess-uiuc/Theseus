// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEHelmholtzEOS.hpp"
#include "GasState.hpp"
#include "Physics.hpp"

namespace Theseus
{
  struct LTEHelmholtzGasEOS
  {
    using Tables = HelmholtzTable::Tables;

    MFEM_HOST_DEVICE static void Require(bool condition)
    {
      if (!condition)
        {
#ifdef __CUDA_ARCH__
          printf("Invalid Helmholtz thermodynamic query\n");
          asm("trap;");
#else
          MFEM_ABORT("Invalid Helmholtz thermodynamic query");
#endif
        }
    }

    MFEM_HOST_DEVICE mfem::real_t isothermal_wall_beta(const PhysicsConstants &, mfem::real_t t) const
    {
      return 1/t;
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t density(const PhysicsConstants &, const StateLayout &layout,
                                         const State &state, const Tables &) const
    {
      return state.mass(layout);
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t kinetic_energy_density(const PhysicsConstants &, const StateLayout &layout,
                                                         const State &state, const Tables &) const
    {
      mfem::real_t momentum_squared = 0;
      for (int d = 0; d < layout.dim; ++d)
        {
          const auto momentum = state.momentum(layout, d);
          momentum_squared += momentum*momentum;
        }
      return 0.5*momentum_squared/state.mass(layout);
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t specific_internal_energy(const PhysicsConstants &phys,
        const StateLayout &layout, const State &state, const Tables &tables) const
    {
      return (state.energy(layout) - kinetic_energy_density(phys, layout, state, tables))/state.mass(layout);
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t temperature(const PhysicsConstants &phys, const StateLayout &layout,
                                              const State &state, const Tables &tables) const
    {
      const auto result = HelmholtzEOS::TemperatureFromEnergy(tables, state.mass(layout),
                            specific_internal_energy(phys, layout, state, tables));
      Require(result.status == HelmholtzEOS::RecoveryStatus::converged);
      return result.temperature;
    }

    template<class State>
    MFEM_HOST_DEVICE HelmholtzEOS::Evaluation evaluate(const PhysicsConstants &phys,
        const StateLayout &layout, const State &state, const Tables &tables) const
    {
      const auto result = HelmholtzEOS::At(tables, state.mass(layout), temperature(phys, layout, state, tables));
      Require(result.valid);
      return result;
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t pressure(const PhysicsConstants &phys, const StateLayout &layout,
                                           const State &state, const Tables &tables) const
    {
      return evaluate(phys, layout, state, tables).pressure();
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t cp(const PhysicsConstants &phys, const StateLayout &layout,
                                     const State &state, const Tables &tables) const
    {
      return evaluate(phys, layout, state, tables).cp();
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t gamma(const PhysicsConstants &phys, const StateLayout &layout,
                                        const State &state, const Tables &tables) const
    {
      return evaluate(phys, layout, state, tables).gamma();
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t R_gas(const PhysicsConstants &phys, const StateLayout &layout,
                                        const State &state, const Tables &tables) const
    {
      return evaluate(phys, layout, state, tables).gas_constant();
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t sound_speed(const PhysicsConstants &phys, const StateLayout &layout,
                                              const State &state, const Tables &tables) const
    {
      return std::sqrt(evaluate(phys, layout, state, tables).sound_speed_squared());
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t entropy(const PhysicsConstants &phys, const StateLayout &layout,
                                          const State &state, const Tables &tables) const
    {
      return evaluate(phys, layout, state, tables).entropy();
    }

    template<class State>
    MFEM_HOST_DEVICE mfem::real_t property_lookup(int property, const PhysicsConstants &phys,
        const StateLayout &layout, const State &state, const Tables &tables) const
    {
      const auto sample = evaluate(phys, layout, state, tables);
      if (property == tables.L.mu_idx)
        {
          return sample.template transport<HelmholtzTable::Layout::mu_idx>(tables);
        }
      Require(property == tables.L.lambda_idx);
      return sample.template transport<HelmholtzTable::Layout::lambda_idx>(tables);
    }

    template<class State>
    MFEM_HOST_DEVICE void grad_temperature(const PhysicsConstants &, const StateLayout &layout,
        const State &, const mfem::real_t *, const mfem::real_t *gradient_temperature,
        mfem::real_t *out, const Tables &) const
    {
      for (int d = 0; d < layout.dim; ++d)
        {
          out[d] = gradient_temperature[d];
        }
    }

    template<class In, class Out>
    MFEM_HOST_DEVICE void entropy_state(const PhysicsConstants &phys, const StateLayout &layout,
        const In &state, Out &entropy, const Tables &tables) const
    {
      const auto sample = evaluate(phys, layout, state, tables);
      const auto beta = 1/sample.temperature;
      const auto kinetic = kinetic_energy_density(phys, layout, state, tables)/state.mass(layout);
      entropy.set_mass(layout, beta*(sample.gibbs() - kinetic));
      for (int d = 0; d < layout.dim; ++d)
        {
          entropy.set_momentum(layout, d, beta*state.velocity(layout, d));
        }
      entropy.set_energy(layout, -beta);
    }

    template<class In, class Out>
    MFEM_HOST_DEVICE void grad_entropy_to_grad_prim(const PhysicsConstants &phys,
        const StateLayout &layout, const In &state, const In &gradient,
        Out &primitive, const Tables &tables) const
    {
      const auto t = temperature(phys, layout, state, tables);
      primitive.set_mass(layout, 0.0);
      for (int d = 0; d < layout.dim; ++d)
        {
          primitive.set_momentum(layout, d, t*gradient.momentum(layout, d)
                                          + t*state.velocity(layout, d)*gradient.energy(layout));
        }
      primitive.set_energy(layout, t*t*gradient.energy(layout));
    }
  };
}
