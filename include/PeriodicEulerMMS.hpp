// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include <cmath>

namespace Theseus::Sources
{
  // Smooth unit-periodic CPG Euler solution; gamma=1.4, phase speed=0.4.
  // Forcing is U_t + dF_x/dx + dF_y/dy, differentiated analytically.
  MFEM_HOST_DEVICE inline void PeriodicEulerMMS(
    mfem::real_t x, mfem::real_t y, mfem::real_t time,
    mfem::real_t *state, mfem::real_t *source)
  {
    constexpr mfem::real_t wave_number = 6.283185307179586476925286766559;
    const auto phase = wave_number*(x + y - 0.4*time);
    const auto sine = std::sin(phase);
    const auto cosine = std::cos(phase);
    const auto rho = 2.0 + 0.2*sine;
    const auto u = 0.7 + 0.1*sine;
    const auto v = 0.3 - 0.05*sine;
    const auto pressure = 1.0 + 0.15*sine;
    const auto drho = 0.2*cosine;
    const auto du = 0.1*cosine;
    const auto dv = -0.05*cosine;
    const auto dp = 0.15*cosine;
    const auto energy = pressure/0.4 + 0.5*rho*(u*u + v*v);
    const auto de = dp/0.4 + 0.5*drho*(u*u + v*v) + rho*(u*du + v*dv);
    const auto speed = u + v;
    const auto dspeed = du + dv;
    const auto dmx = drho*u + rho*du;
    const auto dmy = drho*v + rho*dv;
    state[0] = rho;
    state[1] = rho*u;
    state[2] = rho*v;
    state[3] = energy;
    source[0] = wave_number*(-0.4*drho + drho*speed + rho*dspeed);
    source[1] = wave_number*(-0.4*dmx + dmx*speed + rho*u*dspeed + dp);
    source[2] = wave_number*(-0.4*dmy + dmy*speed + rho*v*dspeed + dp);
    source[3] = wave_number*(-0.4*de + (de + dp)*speed + (energy + pressure)*dspeed);
  }
}
