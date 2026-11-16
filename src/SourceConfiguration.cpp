// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#include "SourceConfiguration.hpp"
#include "sources/Affine.hpp"
#include "sources/LinearDrag.hpp"
#include "verification/mms/Registration.hpp"

namespace Theseus::Sources
{
  Registry BuiltinRegistry()
  {
    Registry registry;
    RegisterAffine(registry);
    RegisterDrag(registry);
    RegisterBuiltinManufacturedSources(registry);
    return registry;
  }

  std::unique_ptr<System> CreateConfigured(const nlohmann::json &runtime,
    int equations, int dimension, bool axisymmetric)
  {
    if (!runtime.contains("sources")) { return nullptr; }
    const auto &configuration = runtime.at("sources");
    if (!configuration.is_array()) { throw std::invalid_argument("sources must be an array"); }
    if (configuration.empty()) { return nullptr; }
    const auto registry = BuiltinRegistry();
    const BuildContext context{dimension, equations, axisymmetric, runtime};
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
      Theseus::Sources::RegisterManufacturedStates(factory);
    }
  };
  const RegisterMMSStates register_mms_states;
}
