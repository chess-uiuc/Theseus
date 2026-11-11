// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "ConditionFactory.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace Prandtl
{
  // Deterministic channel profile for [0,4*pi] x [-1,1] x [0,4*pi/3].
  // CPG constants set the initial conservative field, including when evolving LTE.
  inline std::function<void(const mfem::Vector &, mfem::Vector &)>
  ChannelFlowIC(mfem::real_t reference_temperature, mfem::real_t mach)
  {
    if (!std::isfinite(reference_temperature) || reference_temperature <= 0 ||
        !std::isfinite(mach) || mach < 0)
      {
        throw std::invalid_argument("ChannelFlowIC requires positive temperature and nonnegative Mach number");
      }
    return [reference_temperature, mach](const mfem::Vector &x, mfem::Vector &state)
      {
        if (x.Size() != 3 || state.Size() != 5)
          {
            throw std::invalid_argument("ChannelFlowIC requires 3D and five conservative components");
          }
        const mfem::real_t gamma = 1.4;
        const mfem::real_t gas_constant = 287.05;
        const auto speed = mach*std::sqrt(gamma*gas_constant*reference_temperature);
        const auto shape = 1 - x(1)*x(1);
        const auto base_velocity = 1.5*speed*shape;
        const auto kx = mfem::real_t(0.5);
        const auto kz = mfem::real_t(1.5);
        const auto phase = kx*x(0) + M_PI*x(1) + kz*x(2);
        auto u = base_velocity + 0.1*base_velocity*std::sin(phase);
        auto v = 0.1*base_velocity*std::sin(phase + 1);
        auto w = 0.1*base_velocity*std::sin(phase + 2);
        const auto coherent = 0.05*speed*shape*shape*std::cos(kx*x(0));
        u += coherent*std::cos(kz*x(2));
        v += coherent*std::cos(kz*x(2));
        w += 0.05*speed/kz*4*x(1)*shape*std::cos(kx*x(0))*std::sin(kz*x(2));
        state(0) = 1;
        state(1) = u;
        state(2) = v;
        state(3) = w;
        state(4) = gas_constant*reference_temperature/(gamma - 1) + 0.5*(u*u + v*v + w*w);
      };
  }

  struct RegisterChannelFlowIC
  {
    RegisterChannelFlowIC()
    {
      ConditionFactory::Instance().RegisterInitialCondition2("ChannelFlowIC", ChannelFlowIC);
    }
  };
  inline RegisterChannelFlowIC channel_flow_registration;
}
