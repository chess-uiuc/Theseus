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
                                                 const GasModelInterface &gas);

  inline BoundaryStatePayload ParseLocalCharacteristicBoundary(nlohmann::json config,
                                                               const GasModelInterface &gas)
  {
    if (!gas.SupportsLocalCharacteristic())
      {
        throw std::invalid_argument("Local characteristic boundaries support CPG and original LTE; LTE_Helmholtz is not supported");
      }
    const auto &layout = gas.layout();
    if (layout.nequations() != layout.dim + 2)
      {
        throw std::invalid_argument("Characteristic boundaries require a single gas without transported scalars");
      }
    const auto type = config.at("type").get<std::string>();
    if (type != "characteristic-total-inlet" && type != "frozen-total-inlet" &&
        type != "characteristic-inlet" &&
        type != "characteristic-radial-inlet" && type != "characteristic-pressure-outlet")
      {
        throw std::invalid_argument("Unknown local characteristic boundary type");
      }
    if (type != "characteristic-total-inlet" && type != "frozen-total-inlet")
      {
        config["type"] = type == "characteristic-radial-inlet" ? "radial-profile" : "exterior-state";
        return ParseBoundaryState(config, gas);
      }
    const std::set<std::string> keys{"type", "total_pressure", "total_temperature", "direction"};
    for (const auto &entry : config.items())
      {
        if (!keys.count(entry.key()))
          {
            throw std::invalid_argument("Unknown total-inlet field '" + entry.key() + "'");
          }
      }
    if (!config.at("total_pressure").is_number() || !config.at("total_temperature").is_number())
      {
        throw std::invalid_argument("Total pressure and temperature must be numeric");
      }
    const auto pressure = config.at("total_pressure").get<mfem::real_t>();
    const auto temperature = config.at("total_temperature").get<mfem::real_t>();
    RequirePositiveFinite(pressure, "Total pressure");
    RequirePositiveFinite(temperature, "Total temperature");
    const auto direction = config.at("direction").get<std::vector<mfem::real_t>>();
    if (direction.size() != size_t(layout.dim))
      {
        throw std::invalid_argument("Total-inlet direction must have one entry per dimension");
      }
    mfem::real_t norm2 = 0;
    for (const auto component : direction)
      {
        if (!std::isfinite(component))
          {
            throw std::invalid_argument("Total-inlet direction must be finite");
          }
        norm2 += component * component;
      }
    RequirePositiveFinite(norm2, "Flow direction norm");
    BoundaryStatePayload result;
    result.kind = BCDataKind::VectorConstant;
    PhysicalStateInput reservoir{PressureTemperature{pressure, temperature},
                                 std::vector<mfem::real_t>(layout.dim, 0)};
    mfem::Vector reference;
    gas.ConservativeFromPhysical(reservoir, reference);
    result.values.SetSize(layout.nequations() + 2 + layout.dim);
    for (int q = 0; q < layout.nequations(); ++q)
      {
        result.values[q] = reference[q];
      }
    result.values[layout.nequations()] = pressure;
    result.values[layout.nequations() + 1] = temperature;
    for (int d = 0; d < layout.dim; ++d)
      {
        result.values[layout.nequations() + 2 + d] = direction[d] / std::sqrt(norm2);
      }
    return result;
  }

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
