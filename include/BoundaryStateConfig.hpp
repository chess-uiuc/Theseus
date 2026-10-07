// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "GasModel.hpp"
#include "PhysicalStateConfig.hpp"
#include "RadialProfile.hpp"
#include "bc_cache_utilities.hpp"

namespace Theseus
{
  struct BoundaryStatePayload
  {
    BCDataKind kind;
    mfem::Vector values;
  };

  inline BoundaryStatePayload ParseBoundaryState(const nlohmann::json &config,
                                                 const GasModelInterface &gas)
  {
    const auto type = config.at("type").get<std::string>();
    const bool radial = type == "radial-profile";
    if (!radial && type != "exterior-state")
      {
        throw std::invalid_argument("Expected exterior-state or radial-profile");
      }
    const std::set<std::string> keys =
        radial ? std::set<std::string>{"type", "file", "flag", "pressure"}
               : std::set<std::string>{"type", "state"};
    for (const auto &entry : config.items())
      {
        if (!keys.count(entry.key()))
          {
            throw std::invalid_argument("Unsupported boundary field '" + entry.key() + "'");
          }
      }

    BoundaryStatePayload result;
    if (!radial)
      {
        const auto input = ParsePhysicalState(config.at("state"), gas.layout(), "state");
        gas.ConservativeFromPhysical(input, result.values);
        result.kind = BCDataKind::VectorConstant;
        return result;
      }

    if (gas.layout().dim != 2 || gas.layout().nequations() != 4)
      {
        throw std::invalid_argument("radial-profile requires 2D without scalars");
      }
    if (!config.at("pressure").is_number())
      {
        throw std::invalid_argument("Profile pressure must be numeric");
      }
    const auto pressure = config.at("pressure").get<mfem::real_t>();
    RequirePositiveFinite(pressure, "Profile pressure");
    if (config.contains("flag") && !config.at("flag").is_number_integer())
      {
        throw std::invalid_argument("Profile flag must be an integer");
      }

    const auto profile =
        RadialProfile::Read(config.at("file").get<std::string>(), config.value("flag", 0));
    result.kind = BCDataKind::RadialPhysical;
    result.values.SetSize(2 + 4 * profile.rows.size());
    result.values[0] = pressure;
    result.values[1] = profile.rows.size();
    for (size_t i = 0; i < profile.rows.size(); ++i)
      {
        for (int q = 0; q < 4; ++q)
          {
            result.values[2 + 4 * i + q] = profile.rows[i][q];
          }
      }

    return result;
  }
} // namespace Theseus
