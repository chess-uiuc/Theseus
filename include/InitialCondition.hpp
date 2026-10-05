// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PhysicalStateConfig.hpp"
#include "GasModel.hpp"
#include "ConditionFactory.hpp"
#include "RadialProfile.hpp"
#include "parse_helpers.hpp"
#include <memory>
#include <sstream>

namespace Theseus
{
namespace detail
{
// Compatibility adapter. Existing functions already emit conservative values;
// their parameter conventions and formulas are deliberately retained here.
inline std::unique_ptr<mfem::VectorFunctionCoefficient> MakeLegacyInitialCondition(
    const nlohmann::json &config, const nlohmann::json &runtime,
    const GasModelInterface &gas)
{
  const auto &layout = gas.layout();
  if (config.contains("cpg_state"))
    {
      const auto model = to_lower(runtime.value("gas_model", std::string("cpg")));
      if (layout.dim != 2 || layout.nequations() != 4 ||
          !(model == "cpg" || model == "ideal" || model == "ideal_gas" || model.empty()))
        throw std::invalid_argument("cpg_state requires 2D CPG");
      const auto &v = config.at("cpg_state");
      const auto state = CPGState(v.at("pressure").get<double>(), v.at("temperature").get<double>(),
        v.value("ux",0.0), v.value("ur",0.0), runtime.value("gamma",1.4), runtime.value("R_gas",287.05));
      return std::make_unique<mfem::VectorFunctionCoefficient>(4,
        [state](const mfem::Vector &, mfem::Vector &out) {
          for (int q=0; q<4; ++q) out[q]=state[q];
        });
    }
  const int signature = config.value("signature",0);
  const auto key = config.value("function",std::string("LidDrivenCavityIC"));
  const auto params = config.value("params",nlohmann::json::object());
  const auto x = [&](int i) { return params.value("x"+std::to_string(i),mfem::real_t(0)); };
  auto &factory = Prandtl::ConditionFactory::Instance();
  std::function<void(const mfem::Vector &, mfem::Vector &)> function;
  switch (signature)
    {
      case 0: function=factory.GetInitialCondition0(key)(); break;
      case 1: function=factory.GetInitialCondition1(key)(x(1)); break;
      case 2: function=factory.GetInitialCondition2(key)(x(1),x(2)); break;
      case 3: function=factory.GetInitialCondition3(key)(x(1),x(2),x(3)); break;
      case 4: function=factory.GetInitialCondition4(key)(x(1),x(2),x(3),x(4)); break;
      case 5: function=factory.GetInitialCondition5(key)(x(1),x(2),x(3),x(4),x(5)); break;
      default: throw std::invalid_argument("Invalid initial condition signature");
    }
  return std::make_unique<mfem::VectorFunctionCoefficient>(layout.nequations(),function);
}
}

// Single host factory for all IC sources: returns a coefficient or reports an
// error, never a null sentinel requiring a second dispatch in Simulation.
inline std::unique_ptr<mfem::VectorFunctionCoefficient> MakeInitialCondition(
    const nlohmann::json &config, const nlohmann::json &runtime,
    const GasModelInterface &gas)
{
  try
    {
      const bool constant = config.contains("state"), profile = config.contains("physical_profile");
      if (!constant && !profile)
        return detail::MakeLegacyInitialCondition(config,runtime,gas);
      if (!config.is_object() || config.size()!=1)
        throw std::invalid_argument("choose exactly one of state, physical_profile, or legacy function/cpg_state");
      if (constant)
        {
          const auto input = ParsePhysicalState(config.at("state"),gas.layout(),"state");
          mfem::Vector state;
          gas.ConservativeFromPhysical(input,state);
          return std::make_unique<mfem::VectorFunctionCoefficient>(state.Size(),
            [state](const mfem::Vector &, mfem::Vector &out) { out=state; });
        }
      const auto function = ParsePhysicalProfile(config.at("physical_profile"),gas.layout());
      return std::make_unique<mfem::VectorFunctionCoefficient>(gas.layout().nequations(),
        [function,&gas](const mfem::Vector &x, mfem::Vector &out) {
          try { gas.ConservativeFromPhysical(function(x),out); }
          catch (const std::exception &error) {
            std::ostringstream message;
            message << "conditions.initial_conditions.physical_profile at (" << x[0] << ", " << x[1] << "): " << error.what();
            throw std::invalid_argument(message.str());
          }
        });
    }
  catch (const std::exception &error)
    { throw std::invalid_argument(std::string("conditions.initial_conditions: ")+error.what()); }
}
}
