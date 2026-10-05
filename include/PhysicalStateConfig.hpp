// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "PhysicalState.hpp"
#include "PhysicalProfiles.hpp"
#include "json.hpp"
#include <set>

namespace Theseus
{
// Host-only configuration boundary. Unknown keys are errors so species, total
// pressure, or legacy ux/ur cannot accidentally be treated as supported inputs.
inline PhysicalStateInput ParsePhysicalState(const nlohmann::json &value,
                                             const StateLayout &layout,
                                             const std::string &path)
{
  try
    {
      if (!value.is_object()) throw std::invalid_argument("must be an object");
      const std::set<std::string> keys{"density", "pressure", "temperature", "velocity"};
      for (const auto &entry : value.items())
        if (!keys.count(entry.key()))
          throw std::invalid_argument("unsupported field '" + entry.key() + "'");
      const bool rho = value.contains("density"), p = value.contains("pressure"),
                 t = value.contains("temperature");
      if (int(rho) + int(p) + int(t) != 2)
        throw std::invalid_argument("requires exactly two of density, pressure, temperature");
      auto number = [&](const char *name) {
        if (!value.at(name).is_number())
          throw std::invalid_argument(std::string(name) + " must be a number");
        return value.at(name).get<mfem::real_t>();
      };
      ThermodynamicInput thermo;
      if (p && t) thermo = PressureTemperature{number("pressure"), number("temperature")};
      else if (rho && t) thermo = DensityTemperature{number("density"), number("temperature")};
      else thermo = DensityPressure{number("density"), number("pressure")};
      const auto &vel = value.at("velocity");
      if (!vel.is_array()) throw std::invalid_argument("velocity must be an array");
      std::vector<mfem::real_t> velocity;
      for (const auto &u : vel)
        {
          if (!u.is_number()) throw std::invalid_argument("velocity entries must be numbers");
          velocity.push_back(u.get<mfem::real_t>());
        }
      PhysicalStateInput result{thermo, velocity};
      ValidatePhysicalState(result, layout);
      return result;
    }
  catch (const std::exception &error)
    { throw std::invalid_argument(path + ": " + error.what()); }
}
inline PhysicalStateFunction ParsePhysicalProfile(const nlohmann::json &profile,
                                                  const StateLayout &layout)
{
  if (layout.dim != 2) throw std::invalid_argument("physical_profile requires 2D");
  if (!profile.is_object()) throw std::invalid_argument("physical_profile must be an object");
  const auto type = profile.at("type").get<std::string>();
  const std::set<std::string> blob_keys{"type", "radius", "pressure", "ambient_temperature", "peak_temperature"};
  const std::set<std::string> vortex_keys{"type", "radius", "speed", "strength", "density", "temperature", "shape_gamma", "shape_gas_constant"};
  if (type != "thermal-blob" && type != "vortex")
    throw std::invalid_argument("unknown physical_profile type '" + type + "'");
  const auto &keys = type == "thermal-blob" ? blob_keys : vortex_keys;
  for (const auto &entry : profile.items())
    if (!keys.count(entry.key())) throw std::invalid_argument("unsupported profile field '" + entry.key() + "'");
  auto number = [&](const char *key, bool positive = true) {
    const auto &value = profile.at(key);
    if (!value.is_number()) throw std::invalid_argument(std::string(key) + " must be numeric");
    const auto result = value.get<mfem::real_t>();
    if (!std::isfinite(result)) throw std::invalid_argument(std::string(key) + " must be finite");
    if (positive) RequirePositiveFinite(result, key);
    return result;
  };
  const auto radius = number("radius");
  if (type == "thermal-blob")
    {
      const auto pressure = number("pressure"), ambient = number("ambient_temperature"),
                 peak = number("peak_temperature");
      return ThermalBlobProfile{radius, pressure, ambient, peak};
    }
  const auto speed = number("speed", false), strength = number("strength", false),
             density = number("density"), temperature = number("temperature"),
             gamma = number("shape_gamma"), gas_constant = number("shape_gas_constant");
  if (gamma <= 1) throw std::invalid_argument("shape_gamma must be greater than one");
  return VortexProfile{radius, speed, strength, density, temperature, gamma, gas_constant};
}


}
