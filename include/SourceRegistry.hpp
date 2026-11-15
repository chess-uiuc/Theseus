// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceTerms.hpp"
#include "json.hpp"
#include <functional>
#include <map>
#include <string>

namespace Theseus::Sources
{
  struct BuildContext
  {
    int dimension;
    int equations;
    bool axisymmetric;
    const nlohmann::json &runtime;
  };

  // Host configuration dispatch, separate from RHS and device dispatch.
  class Registry
  {
    using Factory = std::function<std::unique_ptr<Source>(const nlohmann::json &, const BuildContext &)>;
    std::map<std::string, Factory> factories;
  public:
    void Register(const std::string &name, Factory factory)
    {
      if (!factories.emplace(name, std::move(factory)).second)
        {
          throw std::invalid_argument("Duplicate source registration: " + name);
        }
    }
    std::unique_ptr<Source> Create(const nlohmann::json &configuration, const BuildContext &context) const
    {
      const auto name = configuration.at("type").get<std::string>();
      const auto found = factories.find(name);
      if (found == factories.end())
        {
          throw std::invalid_argument("Unknown source: " + name);
        }
      return found->second(configuration, context);
    }
  };
}
