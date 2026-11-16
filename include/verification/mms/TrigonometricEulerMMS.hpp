// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include <cmath>

namespace Theseus::Sources
{
  struct ManufacturedPrimitive
  {
    mfem::real_t value;
    mfem::real_t time_derivative;
    mfem::real_t gradient[3];
  };

  // Rows are rho, u, v, w, p; columns are mean, x, y, z, xy, xz, yz.
  // Integer spatial frequencies make values and derivatives periodic on a unit box.
  MFEM_HOST_DEVICE inline ManufacturedPrimitive TrigonometricPrimitive(
    int field, int dimension, const mfem::real_t *position, mfem::real_t time,
    mfem::real_t time_scale = 1.0)
  {
    const mfem::real_t coefficients[5][7] = {
      {2.0,  .12,  .08, -.06,  .04,  .03, -.02},
      { .7,  .07, -.04,  .03,  .02, -.02,  .01},
      { .3, -.05,  .06, -.04,  .03,  .01, -.02},
      { .2,  .04, -.03,  .05, -.02,  .03,  .02},
      {1.0,  .08, -.06,  .04,  .03, -.02,  .02}
    };
    constexpr mfem::real_t wave_number = 6.283185307179586476925286766559;
    const auto *amplitude = coefficients[field];
    const auto x_phase = wave_number*(position[0] - .2*time*time_scale) + .17*field;
    const auto y_phase = wave_number*(position[1] + .13*time*time_scale) + .29*field;
    const auto sx = std::sin(x_phase);
    const auto cx = std::cos(x_phase);
    const auto sy = std::sin(y_phase);
    const auto cy = std::cos(y_phase);
    ManufacturedPrimitive result{};
    result.value = amplitude[0] + amplitude[1]*sx + amplitude[2]*cy
      + amplitude[4]*sx*cy;
    result.gradient[0] = wave_number*(amplitude[1]*cx + amplitude[4]*cx*cy);
    result.gradient[1] = -wave_number*(amplitude[2]*sy + amplitude[4]*sx*sy);
    if (dimension == 3)
      {
        const auto z_phase = wave_number*(position[2] - .11*time*time_scale) + .31*field;
        const auto sz = std::sin(z_phase);
        const auto cz = std::cos(z_phase);
        result.value += amplitude[3]*sz + amplitude[5]*cx*sz + amplitude[6]*sy*cz;
        result.gradient[0] -= wave_number*amplitude[5]*sx*sz;
        result.gradient[1] += wave_number*amplitude[6]*cy*cz;
        result.gradient[2] = wave_number*(amplitude[3]*cz + amplitude[5]*cx*cz
          - amplitude[6]*sy*sz);
      }
    result.time_derivative = time_scale*(-.2*result.gradient[0] + .13*result.gradient[1]
      - .11*result.gradient[2]);
    return result;
  }

  // Independent x/y/z modes and mixed products exercise distinct flux derivatives.
  // This manufactured family uses Cartesian CPG Euler, gamma=1.4 and R=1.
  MFEM_HOST_DEVICE inline void TrigonometricEulerMMS(
    int dimension, const mfem::real_t *position, mfem::real_t time,
    mfem::real_t *state, mfem::real_t *source, mfem::real_t time_scale = 1.0)
  {
    const auto density = TrigonometricPrimitive(0, dimension, position, time, time_scale);
    const auto pressure = TrigonometricPrimitive(4, dimension, position, time, time_scale);
    ManufacturedPrimitive velocity[3];
    mfem::real_t speed_squared = 0.0;
    mfem::real_t velocity_time_work = 0.0;
    mfem::real_t divergence = 0.0;
    for (int d = 0; d < dimension; ++d)
      {
        velocity[d] = TrigonometricPrimitive(d + 1, dimension, position, time, time_scale);
        speed_squared += velocity[d].value*velocity[d].value;
        velocity_time_work += velocity[d].value*velocity[d].time_derivative;
        divergence += velocity[d].gradient[d];
      }
    const auto energy = pressure.value/.4 + .5*density.value*speed_squared;
    state[0] = density.value;
    state[dimension + 1] = energy;
    source[0] = density.time_derivative + density.value*divergence;
    source[dimension + 1] = pressure.time_derivative/.4
      + .5*density.time_derivative*speed_squared + density.value*velocity_time_work
      + (energy + pressure.value)*divergence;
    for (int component = 0; component < dimension; ++component)
      {
        state[component + 1] = density.value*velocity[component].value;
        source[component + 1] = density.time_derivative*velocity[component].value
          + density.value*velocity[component].time_derivative
          + density.value*velocity[component].value*divergence
          + pressure.gradient[component];
      }
    for (int direction = 0; direction < dimension; ++direction)
      {
        mfem::real_t velocity_gradient_work = 0.0;
        for (int component = 0; component < dimension; ++component)
          {
            velocity_gradient_work += velocity[component].value
              *velocity[component].gradient[direction];
            source[component + 1] += velocity[direction].value
              *(density.gradient[direction]*velocity[component].value
                + density.value*velocity[component].gradient[direction]);
          }
        const auto energy_gradient = pressure.gradient[direction]/.4
          + .5*density.gradient[direction]*speed_squared
          + density.value*velocity_gradient_work;
        source[0] += velocity[direction].value*density.gradient[direction];
        source[dimension + 1] += velocity[direction].value
          *(energy_gradient + pressure.gradient[direction]);
      }
  }
  // Canonical direction d maps to physical direction (d + orientation) % dimension.
  // Scalar equations are unchanged; momentum and its forcing rotate together.
  MFEM_HOST_DEVICE inline void OrientedTrigonometricEulerMMS(
    int dimension, int orientation, const mfem::real_t *position, mfem::real_t time,
    mfem::real_t *state, mfem::real_t *source, mfem::real_t time_scale = 1.0)
  {
    mfem::real_t canonical_position[3] = {};
    for (int d = 0; d < dimension; ++d)
      {
        canonical_position[d] = position[(d + orientation) % dimension];
      }
    mfem::real_t canonical_state[5];
    mfem::real_t canonical_source[5];
    TrigonometricEulerMMS(dimension, canonical_position, time, canonical_state, canonical_source, time_scale);
    state[0] = canonical_state[0];
    source[0] = canonical_source[0];
    state[dimension + 1] = canonical_state[dimension + 1];
    source[dimension + 1] = canonical_source[dimension + 1];
    for (int d = 0; d < dimension; ++d)
      {
        const int physical_direction = (d + orientation) % dimension;
        state[physical_direction + 1] = canonical_state[d + 1];
        source[physical_direction + 1] = canonical_source[d + 1];
      }
  }
}
