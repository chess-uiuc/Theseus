// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "LTETable.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Theseus
{

  // Host-only view of one table cell, shared by all requested properties.
  class LTEPropertySample
  {
  private:
    const LTETable::LTETables &tables;
    int density_index;
    int temperature_index;
    mfem::real_t density_weight;
    mfem::real_t temperature_weight;

    static int FindInterval(const mfem::real_t *grid, int size,
                            mfem::real_t value)
    {
      if (!grid || size < 2 || !std::isfinite(value) ||
          value < grid[0] || value > grid[size - 1])
        {
          throw std::invalid_argument("Property sample is outside the LTE table");
        }

      const auto upper = std::upper_bound(grid, grid + size, value);
      return std::min(int(upper - grid) - 1, size - 2);
    }

  public:
    LTEPropertySample(const LTETable::LTETables &source,
                      mfem::real_t density, mfem::real_t temperature)
      : tables(source)
    {
      const auto &view = tables.tables;
      density_index = FindInterval(view.rho_grid, tables.L.nx, density);
      temperature_index = FindInterval(view.T_grid, tables.L.ny, temperature);

      const auto density_low = view.rho_grid[density_index];
      const auto density_high = view.rho_grid[density_index + 1];
      const auto temperature_low = view.T_grid[temperature_index];
      const auto temperature_high = view.T_grid[temperature_index + 1];

      density_weight = (density - density_low) / (density_high - density_low);
      temperature_weight = (temperature - temperature_low) /
                           (temperature_high - temperature_low);
    }

    mfem::real_t Value(int property) const
    {
      const auto &layout = tables.L;
      const auto *values = tables.tables.lte_table;
      if (!values || property < 0 || property >= layout.num_properties)
        {
          throw std::invalid_argument("Requested LTE property is unavailable");
        }

      const int lower_density = density_index;
      const int upper_density = density_index + 1;
      const int lower_temperature = temperature_index;
      const int upper_temperature = temperature_index + 1;

      const auto lower_left = values[layout.property_index(property, lower_density, lower_temperature)];
      const auto upper_left = values[layout.property_index(property, lower_density, upper_temperature)];
      const auto lower_right = values[layout.property_index(property, upper_density, lower_temperature)];
      const auto upper_right = values[layout.property_index(property, upper_density, upper_temperature)];

      const auto wx = density_weight;
      const auto wy = temperature_weight;
      return lower_left * ((1 - wx) * (1 - wy)) + upper_left * ((1 - wx) * wy) +
             lower_right * (wx * (1 - wy)) + upper_right * (wx * wy);
    }
  };

}
