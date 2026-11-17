// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "verification/mms/TrigonometricEulerMMS.hpp"

namespace Theseus::Sources
{
struct ManufacturedSecondDerivative
{
  ManufacturedPrimitive first;
  mfem::real_t hessian[3][3] = {};
};

// Closed-form Hessians of the committed trigonometric primitive family.
// Kept separate so Euler source evaluation does not pay for second derivatives.
MFEM_HOST_DEVICE inline ManufacturedSecondDerivative
TrigonometricSecondDerivative (int field, int dimension, const mfem::real_t *position,
                               mfem::real_t time, mfem::real_t time_scale = 1.0, mfem::real_t mean_velocity_scale = 1.0)
{
  const mfem::real_t coefficients[5][6] = { { .12, .08, -.06, .04, .03, -.02 },
                                            { .07, -.04, .03, .02, -.02, .01 },
                                            { -.05, .06, -.04, .03, .01, -.02 },
                                            { .04, -.03, .05, -.02, .03, .02 },
                                            { .08, -.06, .04, .03, -.02, .02 } };
  constexpr mfem::real_t wave_number = 6.283185307179586476925286766559;
  const auto *a = coefficients[field];
  const auto x = wave_number * (position[0] - .2 * time * time_scale) + .17 * field;
  const auto y = wave_number * (position[1] + .13 * time * time_scale) + .29 * field;
  const auto sx = std::sin (x);
  const auto cx = std::cos (x);
  const auto sy = std::sin (y);
  const auto cy = std::cos (y);
  const auto factor = -wave_number * wave_number;
  ManufacturedSecondDerivative result;
  result.first = TrigonometricPrimitive (field, dimension, position, time, time_scale, mean_velocity_scale);
  result.hessian[0][0] = factor * (a[0] * sx + a[3] * sx * cy);
  result.hessian[1][1] = factor * (a[1] * cy + a[3] * sx * cy);
  result.hessian[0][1] = result.hessian[1][0] = factor * a[3] * cx * sy;
  if (dimension == 3)
    {
      const auto z = wave_number * (position[2] - .11 * time * time_scale) + .31 * field;
      const auto sz = std::sin (z);
      const auto cz = std::cos (z);
      result.hessian[0][0] += factor * a[4] * cx * sz;
      result.hessian[1][1] += factor * a[5] * sy * cz;
      result.hessian[2][2] = factor * (a[2] * sz + a[4] * cx * sz + a[5] * sy * cz);
      result.hessian[0][2] = result.hessian[2][0] = factor * a[4] * sx * cz;
      result.hessian[1][2] = result.hessian[2][1] = factor * a[5] * cy * sz;
    }
  return result;
}

MFEM_HOST_DEVICE inline void
TrigonometricCNSMMS (int dimension, const mfem::real_t *position, mfem::real_t time,
                     mfem::real_t viscosity, mfem::real_t conductivity, mfem::real_t gas_constant,
                     mfem::real_t *state, mfem::real_t *source, mfem::real_t time_scale = 1.0, mfem::real_t mean_velocity_scale = 1.0)
{
  TrigonometricEulerMMS (dimension, position, time, state, source, time_scale, mean_velocity_scale);
  const auto density = TrigonometricSecondDerivative (0, dimension, position, time, time_scale, mean_velocity_scale);
  const auto pressure = TrigonometricSecondDerivative (4, dimension, position, time, time_scale, mean_velocity_scale);
  ManufacturedSecondDerivative velocity[3];
  mfem::real_t divergence = 0.0;
  mfem::real_t divergence_gradient[3] = {};
  for (int i = 0; i < dimension; ++i)
    {
      velocity[i] = TrigonometricSecondDerivative (i + 1, dimension, position, time, time_scale, mean_velocity_scale);
      divergence += velocity[i].first.gradient[i];
      for (int j = 0; j < dimension; ++j)
        {
          divergence_gradient[j] += velocity[i].hessian[i][j];
        }
    }
  const auto rho = density.first.value;
  const auto p = pressure.first.value;
  mfem::real_t viscous_energy_divergence = 0.0;
  for (int i = 0; i < dimension; ++i)
    {
      mfem::real_t stress_divergence = 0.0;
      for (int j = 0; j < dimension; ++j)
        {
          const auto stress = viscosity
                              * (velocity[i].first.gradient[j] + velocity[j].first.gradient[i]
                                 - (i == j ? (2.0 / 3.0) * divergence : 0.0));
          stress_divergence += viscosity
                               * (velocity[i].hessian[j][j] + velocity[j].hessian[i][j]
                                  - (i == j ? (2.0 / 3.0) * divergence_gradient[j] : 0.0));
          viscous_energy_divergence += velocity[i].first.gradient[j] * stress;
        }
      source[i + 1] -= stress_divergence;
      viscous_energy_divergence += velocity[i].first.value * stress_divergence;
    }
  for (int j = 0; j < dimension; ++j)
    {
      const auto rho_j = density.first.gradient[j];
      const auto p_j = pressure.first.gradient[j];
      const auto temperature_jj = (pressure.hessian[j][j] / rho
                                   - (2.0 * p_j * rho_j + p * density.hessian[j][j]) / (rho * rho)
                                   + 2.0 * p * rho_j * rho_j / (rho * rho * rho))
                                  / gas_constant;
      viscous_energy_divergence += conductivity * temperature_jj;
    }
  source[dimension + 1] -= viscous_energy_divergence;
}
}
