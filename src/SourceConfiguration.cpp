// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "SourceConfiguration.hpp"
#include "sources/Affine.hpp"
#include "sources/LinearDrag.hpp"
#include "sources/EulerMMS.hpp"
#include "sources/ChannelFeedback.hpp"
#include "ConditionFactory.hpp"

namespace Theseus::Sources
{
  Registry BuiltinRegistry()
  {
    Registry registry;
    RegisterAffine(registry);
    RegisterDrag(registry);
    RegisterMMS(registry);
    RegisterChannelFeedback(registry);
    return registry;
  }

  std::unique_ptr<System> CreateConfigured(const nlohmann::json &runtime,
    int equations, int dimension, bool axisymmetric)
  {
    const auto configured = ChannelFeedbackConfiguration(runtime);
    if (!configured.contains("sources")) { return nullptr; }
    const auto &configuration = configured.at("sources");
    if (!configuration.is_array()) { throw std::invalid_argument("sources must be an array"); }
    if (configuration.empty()) { return nullptr; }
    const auto registry = BuiltinRegistry();
    const BuildContext context{dimension, equations, axisymmetric, configured};
    auto system = std::make_unique<System>();
    for (const auto &entry : configuration) { system->Add(registry.Create(entry, context)); }
    return system;
  }
}

namespace
{
  // Manufactured initial/exact states register alongside their source consumer.
  struct RegisterMMSStates
  {
    RegisterMMSStates()
    {
      auto &factory = Prandtl::ConditionFactory::Instance();
      factory.RegisterInitialCondition0("PeriodicEulerMMS", []()
      {
        return [](const mfem::Vector &position, mfem::Vector &state)
        {
          MFEM_VERIFY(position.Size() == 2 && state.Size() == 4,
                      "PeriodicEulerMMS requires 2D and four equations");
          mfem::real_t forcing[4];
          Theseus::Sources::PeriodicEulerMMS(position[0], position[1], 0.0, state.HostWrite(), forcing);
        };
      });
      factory.RegisterVectorTDFunctionBoundaryCondition0("PeriodicEulerMMS", []()
      {
        return [](const mfem::Vector &position, mfem::real_t time, mfem::Vector &state)
        {
          MFEM_VERIFY(position.Size() == 2 && state.Size() == 4,
                      "PeriodicEulerMMS requires 2D and four equations");
          mfem::real_t forcing[4];
          Theseus::Sources::PeriodicEulerMMS(position[0], position[1], time, state.HostWrite(), forcing);
        };
      });
    }
  };
  const RegisterMMSStates register_mms_states;
}
