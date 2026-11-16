// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "verification/mms/TrigonometricCNSMMS.hpp"

namespace Theseus::Sources
{
MFEM_HOST_DEVICE inline void
TrigonometricSutherlandMMS (int dimension, const mfem::real_t *position, mfem::real_t time,
                            mfem::real_t reference_viscosity, mfem::real_t prandtl,
                            mfem::real_t gas_constant, mfem::real_t *state, mfem::real_t *source,
                            mfem::real_t time_scale = 1.0,
                            mfem::real_t coefficient_gradient_scale = 1.0)
{
  const auto density = TrigonometricPrimitive (0, dimension, position, time, time_scale);
  const auto pressure = TrigonometricPrimitive (4, dimension, position, time, time_scale);
  const auto temperature = pressure.value / (density.value * gas_constant);
  constexpr mfem::real_t reference_temperature = 273.15;
  constexpr mfem::real_t sutherland_temperature = 110.4;
  const auto relative_temperature = temperature / reference_temperature;
  const auto viscosity = reference_viscosity * (reference_temperature + sutherland_temperature)
                         * relative_temperature * std::sqrt (relative_temperature)
                         / (temperature + sutherland_temperature);
  const auto cp_over_prandtl = 3.5 * gas_constant / prandtl;
  const auto conductivity = viscosity * cp_over_prandtl;

  // First differentiate the viscous flux holding its local coefficients fixed.
  TrigonometricCNSMMS (dimension, position, time, viscosity, conductivity, gas_constant, state,
                       source, time_scale);

  ManufacturedPrimitive velocity[3];
  mfem::real_t divergence = 0.0;
  for (int i = 0; i < dimension; ++i)
    {
      velocity[i] = TrigonometricPrimitive (i + 1, dimension, position, time, time_scale);
      divergence += velocity[i].gradient[i];
    }
  const auto viscosity_temperature_derivative
      = viscosity * (1.5 / temperature - 1.0 / (temperature + sutherland_temperature));
  mfem::real_t energy_correction = 0.0;
  for (int j = 0; j < dimension; ++j)
    {
      const auto temperature_gradient
          = (pressure.gradient[j] / density.value
             - pressure.value * density.gradient[j] / (density.value * density.value))
            / gas_constant;
      const auto viscosity_gradient = viscosity_temperature_derivative * temperature_gradient;
      for (int i = 0; i < dimension; ++i)
        {
          const auto strain = velocity[i].gradient[j] + velocity[j].gradient[i]
                              - (i == j ? (2.0 / 3.0) * divergence : 0.0);
          const auto momentum_correction = coefficient_gradient_scale * viscosity_gradient * strain;
          source[i + 1] -= momentum_correction;
          energy_correction += velocity[i].value * momentum_correction;
        }
      energy_correction += coefficient_gradient_scale * cp_over_prandtl * viscosity_gradient
                           * temperature_gradient;
    }
  source[dimension + 1] -= energy_correction;
}
}
