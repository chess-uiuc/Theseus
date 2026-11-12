// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "json.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace Theseus
{

  enum class VisualizationField
  {
    density,
    velocity,
    pressure,
    blending_coefficient,
    temperature,
    sound_speed,
    mach_number,
    specific_internal_energy,
    internal_energy_density,
    specific_total_energy,
    viscosity,
    thermal_conductivity
  };

  enum class VisualizationMeshMode
  {
    vtk_high_order,
    gll_subcells
  };

  enum class VisualizationProperty
  {
    none,
    pressure,
    temperature,
    sound_speed,
    viscosity,
    thermal_conductivity
  };

  struct VisualizationFieldSpec
  {
    VisualizationField field;
    const char *name;
    const char *output_name;
    int components; // Zero means the spatial dimension.
    VisualizationProperty property;
  };

  class VisualizationConfig
  {
  private:
    std::vector<VisualizationField> fields_;
    VisualizationMeshMode mesh_mode_ = VisualizationMeshMode::gll_subcells;

    static constexpr std::array<VisualizationFieldSpec, 12> field_registry_
    {{
      {VisualizationField::density, "density", "Density", 1, VisualizationProperty::none},
      {VisualizationField::velocity, "velocity", "Velocity", 0, VisualizationProperty::none},
      {VisualizationField::pressure, "pressure", "Pressure", 1, VisualizationProperty::pressure},
      {VisualizationField::blending_coefficient, "blending_coefficient", "Blending Coeff", 1,
       VisualizationProperty::none},
      {VisualizationField::temperature, "temperature", "Temperature", 1,
       VisualizationProperty::temperature},
      {VisualizationField::sound_speed, "sound_speed", "Sound Speed", 1,
       VisualizationProperty::sound_speed},
      {VisualizationField::mach_number, "mach_number", "Mach Number", 1,
       VisualizationProperty::sound_speed},
      {VisualizationField::specific_internal_energy, "specific_internal_energy",
       "Specific Internal Energy", 1, VisualizationProperty::none},
      {VisualizationField::internal_energy_density, "internal_energy_density",
       "Internal Energy Density", 1, VisualizationProperty::none},
      {VisualizationField::specific_total_energy, "specific_total_energy",
       "Specific Total Energy", 1, VisualizationProperty::none},
      {VisualizationField::viscosity, "viscosity", "Viscosity", 1,
       VisualizationProperty::viscosity},
      {VisualizationField::thermal_conductivity, "thermal_conductivity", "Thermal Conductivity", 1,
       VisualizationProperty::thermal_conductivity}
    }};

    static const VisualizationFieldSpec &FindField(const std::string &name)
    {
      const auto it = std::find_if(field_registry_.begin(), field_registry_.end(),
                                   [&name](const auto &spec) { return name == spec.name; });
      if (it == field_registry_.end())
        {
          std::string supported;
          for (const auto &spec : field_registry_)
            {
              if (!supported.empty())
                {
                  supported += ", ";
                }
              supported += spec.name;
            }
          throw std::invalid_argument("Unknown visualization field '" + name +
                                      "'. Supported fields: " + supported);
        }
      return *it;
    }

  public:
    static const VisualizationFieldSpec &Specification(VisualizationField field)
    {
      const auto match = std::find_if(field_registry_.begin(), field_registry_.end(),
                                      [field](const auto &spec) { return spec.field == field; });
      if (match == field_registry_.end())
        {
          throw std::invalid_argument("Invalid visualization field");
        }
      return *match;
    }

    static VisualizationConfig FromRuntime(const nlohmann::json &runtime,
                                           bool blending_available)
    {
      VisualizationConfig result;
      const bool has_blending_default = blending_available;
      result.fields_ = {VisualizationField::density,
                        VisualizationField::velocity,
                        VisualizationField::pressure};
      if (has_blending_default)
        {
          result.fields_.push_back(VisualizationField::blending_coefficient);
        }

      if (!runtime.contains("visualization"))
        {
          return result;
        }

      const auto &visualization = runtime.at("visualization");
      if (!visualization.is_object())
        {
          throw std::invalid_argument("runTime.visualization must be an object");
        }
      if (!visualization.contains("fields"))
        {
          // Continue parsing other visualization options.
        }
      else
        {
          const auto &configured_fields = visualization.at("fields");
          if (!configured_fields.is_array())
            {
              throw std::invalid_argument(
                "runTime.visualization.fields must be an array of strings");
            }
          if (configured_fields.empty())
            {
              throw std::invalid_argument(
                "runTime.visualization.fields must contain at least one field");
            }

          result.fields_.clear();
          for (const auto &entry : configured_fields)
            {
              if (!entry.is_string())
                {
                  throw std::invalid_argument(
                    "runTime.visualization.fields entries must be strings");
                }

              const auto &spec = FindField(entry.get<std::string>());
              if (spec.field == VisualizationField::blending_coefficient &&
                  !blending_available)
                {
                  throw std::invalid_argument(
                    "Visualization field 'blending_coefficient' is unavailable because Theseus "
                    "was built without SUBCELL_FV_BLENDING");
                }
              if (std::find(result.fields_.begin(), result.fields_.end(), spec.field) ==
                  result.fields_.end())
                {
                  result.fields_.push_back(spec.field);
                }
            }
        }

      if (visualization.contains("mesh_mode"))
        {
          const auto &mesh_mode = visualization.at("mesh_mode");
          if (!mesh_mode.is_string())
            {
              throw std::invalid_argument(
                "runTime.visualization.mesh_mode must be a string");
            }
          const std::string mode = mesh_mode.get<std::string>();
          if (mode == "vtk_high_order")
            {
              result.mesh_mode_ = VisualizationMeshMode::vtk_high_order;
            }
          else if (mode == "gll_subcells")
            {
              result.mesh_mode_ = VisualizationMeshMode::gll_subcells;
            }
          else
            {
              throw std::invalid_argument(
                "Unknown visualization mesh mode '" + mode +
                "'. Supported modes: vtk_high_order, gll_subcells");
            }
        }
      return result;
    }

    bool Has(VisualizationField field) const
    {
      return std::find(fields_.begin(), fields_.end(), field) != fields_.end();
    }

    const std::vector<VisualizationField> &Fields() const { return fields_; }

    VisualizationMeshMode MeshMode() const { return mesh_mode_; }
  };

}
