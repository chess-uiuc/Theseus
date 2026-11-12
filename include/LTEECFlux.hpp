// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEHelmholtzGasEOS.hpp"
#include "DissipationOperator.hpp"

namespace Theseus
{
  namespace LTEECFlux
  {
    template<bool Face, class Gas>
    MFEM_HOST_DEVICE inline void Compute(const Gas &gas, const mfem::real_t *q1,
        const mfem::real_t *q2, const mfem::real_t *direction1,
        const mfem::real_t *direction2, mfem::real_t *flux)
    {
      const PointStateView left(q1);
      const PointStateView right(q2);
      mfem::real_t density_mean;
      mfem::real_t energy_mean;
      mfem::real_t pressure1;
      mfem::real_t pressure2;
      mfem::real_t sound1 = 0;
      mfem::real_t sound2 = 0;
      {
        // Recover once per endpoint and share its selected cell. Consume all
        // thermodynamics before writing output; no coefficient/derivative or
        // dissipation buffers survive into the equation loop.
        const auto state1 = gas.eos.evaluate(gas.phys, gas.L, left, gas.T);
        const auto state2 = gas.eos.evaluate(gas.phys, gas.L, right, gas.T);
        const auto means = HelmholtzEOS::ThermodynamicMeans(gas.T, state1, state2);
        density_mean = means.density;
        energy_mean = means.energy;
        pressure1 = state1.pressure();
        pressure2 = state2.pressure();
        if constexpr (Face)
          {
            sound1 = std::sqrt(state1.sound_speed_squared());
            sound2 = std::sqrt(state2.sound_speed_squared());
          }
      }
      const auto pressure_mean = 0.5*(pressure1 + pressure2);
      mfem::real_t normal_velocity = 0;
      mfem::real_t cross_kinetic = 0;
      mfem::real_t pressure_work = 0;
      mfem::real_t velocity1_normal = 0;
      mfem::real_t velocity2_normal = 0;
      mfem::real_t norm_squared = 0;
      for (int d = 0; d < gas.dim(); ++d)
        {
          const auto direction = Face ? direction1[d] : 0.5*(direction1[d] + direction2[d]);
          const auto u1 = gas.velocity(left, d);
          const auto u2 = gas.velocity(right, d);
          normal_velocity += 0.5*(u1 + u2)*direction;
          cross_kinetic += 0.5*u1*u2;
          pressure_work += 0.5*(pressure1*u2 + pressure2*u1)*direction;
          if constexpr (Face)
            {
              velocity1_normal += u1*direction;
              velocity2_normal += u2*direction;
              norm_squared += direction*direction;
            }
        }
      mfem::real_t dissipation = 0;
      if constexpr (Face)
        {
          // Same convention as MaximumNormalWaveSpeed: nonunit normal
          // magnitude enters the acoustic contribution exactly once.
          const auto norm = std::sqrt(norm_squared);
          dissipation = Lax_dissipation_coefficient(velocity1_normal,
            velocity2_normal, sound1, sound2, norm);
        }
      const auto mass_flux = density_mean*normal_velocity;
      flux[gas.L.eq_mass] = mass_flux - Lax_dissipation(dissipation, q1[gas.L.eq_mass], q2[gas.L.eq_mass]);
      for (int d = 0; d < gas.dim(); ++d)
        {
          const auto direction = Face ? direction1[d] : 0.5*(direction1[d] + direction2[d]);
          const auto velocity = 0.5*(gas.velocity(left, d) + gas.velocity(right, d));
          const int equation = gas.L.eq_mom0 + d;
          flux[equation] = mass_flux*velocity + pressure_mean*direction
                           - Lax_dissipation(dissipation, q1[equation], q2[equation]);
        }
      flux[gas.L.eq_energy] = mass_flux*(energy_mean + cross_kinetic) + pressure_work
                             - Lax_dissipation(dissipation, q1[gas.L.eq_energy], q2[gas.L.eq_energy]);
    }

    template<class Gas>
    MFEM_HOST_DEVICE inline void ComputeFaceFluxKernel(const Gas &gas, const mfem::real_t *left,
        const mfem::real_t *right, const mfem::real_t *normal, mfem::real_t *flux)
    {
      Compute<true>(gas, left, right, normal, normal, flux);
    }

    template<class Gas>
    MFEM_HOST_DEVICE inline void ComputeVolumeFluxKernel(const Gas &gas, const mfem::real_t *left,
        const mfem::real_t *right, const mfem::real_t *metric1, const mfem::real_t *metric2,
        mfem::real_t *flux)
    {
      Compute<false>(gas, left, right, metric1, metric2, flux);
    }

    struct InviscidFlux
    {
      template<class Gas>
      MFEM_HOST_DEVICE void ComputeFaceFlux(const Gas &gas, const mfem::real_t *left,
          const mfem::real_t *right, const mfem::real_t *normal, mfem::real_t *flux) const
      {
        ComputeFaceFluxKernel(gas, left, right, normal, flux);
      }

      template<class Gas>
      MFEM_HOST_DEVICE void ComputeVolumeFlux(const Gas &gas, const mfem::real_t *left,
          const mfem::real_t *right, const mfem::real_t *metric1, const mfem::real_t *metric2,
          mfem::real_t *flux) const
      {
        ComputeVolumeFluxKernel(gas, left, right, metric1, metric2, flux);
      }
    };
  }
}
