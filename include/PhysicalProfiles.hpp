// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PhysicalState.hpp"
#include <algorithm>
#include <functional>

namespace Theseus
{
// Host spatial descriptions only. No JSON, gas-model dispatch or MFEM coefficient
// construction: a profile describes physical data at a point, never energy.
using PhysicalStateFunction = std::function<PhysicalStateInput(const mfem::Vector &)>;

struct ThermalBlobProfile
{
  mfem::real_t radius, pressure, ambient_temperature, peak_temperature;
  PhysicalStateInput operator()(const mfem::Vector &x) const
  {
    const auto r2 = (x[0]*x[0] + x[1]*x[1]) / (radius*radius);
    return {PressureTemperature{pressure, ambient_temperature +
      (peak_temperature-ambient_temperature)*std::exp(-r2)}, {0,0}};
  }
};

struct VortexProfile
{
  mfem::real_t radius, speed, strength, density, temperature, gamma, gas_constant;
  PhysicalStateInput operator()(const mfem::Vector &x) const
  {
    // CPG relations prescribe a shape, not the simulated EOS. rho/T are the
    // independent pair; the selected gas determines pressure and energy.
    const auto r2 = (x[0]*x[0]+x[1]*x[1])/(radius*radius);
    const auto cp = gamma*gas_constant/(gamma-1);
    const auto temp = std::max(temperature - 0.5*speed*speed*strength*strength/cp*std::exp(-r2), 0.2*temperature);
    const auto rho = density*std::pow(temp/temperature, 1/(gamma-1));
    const auto factor = strength/radius*std::exp(-0.5*r2);
    return {DensityTemperature{rho,temp}, {speed*(1-factor*x[1]),speed*factor*x[0]}};
  }
};
}
