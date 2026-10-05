// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "GasModel.hpp"
#include "VisualizationConfig.hpp"

#include <memory>
#include <vector>

namespace Theseus
{

  class VisualizationFields
  {
  private:
    struct OutputField
    {
      VisualizationField field;
      std::unique_ptr<mfem::ParGridFunction> storage;
      mfem::ParGridFunction *values = nullptr;
      mfem::real_t *host_values = nullptr;
    };

    std::vector<OutputField> fields;
    GasPropertyRequest properties_needed;
    bool has_derived_fields = false;

    void RequestProperty(VisualizationProperty property)
    {
      switch (property)
        {
        case VisualizationProperty::pressure:
          properties_needed.pressure = true;
          break;
        case VisualizationProperty::temperature:
          properties_needed.temperature = true;
          break;
        case VisualizationProperty::sound_speed:
          properties_needed.sound_speed = true;
          break;
        case VisualizationProperty::viscosity:
          properties_needed.viscosity = true;
          break;
        case VisualizationProperty::thermal_conductivity:
          properties_needed.thermal_conductivity = true;
          break;
        case VisualizationProperty::none:
          break;
        }
    }

    static mfem::real_t ScalarValue(VisualizationField field, const GasProperties &properties,
                                    mfem::real_t density, mfem::real_t total_energy_density,
                                    mfem::real_t kinetic_energy_density)
    {
      switch (field)
        {
        case VisualizationField::pressure:
          return properties.pressure;
        case VisualizationField::temperature:
          return properties.temperature;
        case VisualizationField::sound_speed:
          return properties.sound_speed;
        case VisualizationField::mach_number:
          return std::sqrt(2 * kinetic_energy_density / density) / properties.sound_speed;
        case VisualizationField::specific_internal_energy:
          return (total_energy_density - kinetic_energy_density) / density;
        case VisualizationField::internal_energy_density:
          return total_energy_density - kinetic_energy_density;
        case VisualizationField::specific_total_energy:
          return total_energy_density / density;
        case VisualizationField::viscosity:
          return properties.viscosity;
        case VisualizationField::thermal_conductivity:
          return properties.thermal_conductivity;
        default:
          throw std::logic_error("Visualization field is not a derived scalar");
        }
    }

  public:
    VisualizationFields(const VisualizationConfig &config,
                        mfem::ParFiniteElementSpace &scalar_space,
                        mfem::ParFiniteElementSpace &vector_space,
                        mfem::ParGridFunction &density,
                        mfem::ParGridFunction *blending)
    {
      for (const auto field : config.Fields())
        {
          OutputField output;
          output.field = field;
          const auto &spec = VisualizationConfig::Specification(field);

          if (field == VisualizationField::density)
            {
              output.values = &density;
            }
          else if (field == VisualizationField::blending_coefficient)
            {
              if (!blending)
                {
                  throw std::invalid_argument("Blending field is unavailable");
                }
              output.values = blending;
            }
          else
            {
              auto *space = spec.components == 0 ? &vector_space : &scalar_space;
              output.storage = std::make_unique<mfem::ParGridFunction>(space);
              output.values = output.storage.get();
              has_derived_fields = true;
            }

          RequestProperty(spec.property);
          fields.push_back(std::move(output));
        }
    }

    void Register(mfem::DataCollection &collection) const
    {
      for (const auto &output : fields)
        {
          const auto &spec = VisualizationConfig::Specification(output.field);
          collection.RegisterField(spec.output_name, output.values);
        }
    }

    void Update(const mfem::Vector &solution, const GasModelInterface &gas)
    {
      if (!has_derived_fields)
        {
          return;
        }

      const auto &layout = gas.layout();
      const auto *conservative = solution.HostRead();
      for (auto &output : fields)
        {
          if (output.storage)
            {
              output.host_values = output.storage->HostWrite();
            }
        }

      for (int point = 0; point < layout.num_dofs_scalar; ++point)
        {
          const DofStateView state{conservative, point};
          GasProperties properties;
          if (!properties_needed.Empty())
            {
              properties = gas.EvaluateProperties(state, properties_needed);
            }

          const auto density = state.mass(layout);
          const auto total_energy_density = state.energy(layout);
          mfem::real_t momentum_squared = 0;
          for (int component = 0; component < layout.dim; ++component)
            {
              const auto momentum = state.momentum(layout, component);
              momentum_squared += momentum * momentum;
            }
          const auto kinetic_energy_density = 0.5 * momentum_squared / density;

          for (const auto &output : fields)
            {
              if (!output.storage)
                {
                  continue;
                }
              if (output.field == VisualizationField::velocity)
                {
                  for (int component = 0; component < layout.dim; ++component)
                    {
                      const int offset = point + component * layout.num_dofs_scalar;
                      output.host_values[offset] = state.velocity(layout, component);
                    }
                }
              else
                {
                  output.host_values[point] = ScalarValue(output.field, properties, density,
                                                         total_energy_density, kinetic_energy_density);
                }
            }
        }
    }
  };

}
