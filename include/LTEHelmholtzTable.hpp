// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "LTETable.hpp"
#include <stdexcept>

namespace Theseus
{
  namespace HelmholtzTable
  {
    // Distinct from legacy LTE: physical derivatives, even on logarithmic grids.
    struct Layout
    {
      int nx = 0;
      int ny = 0;
      static constexpr int num_properties = 6;
      static constexpr int H_idx = 0;
      static constexpr int Hr_idx = 1;
      static constexpr int HT_idx = 2;
      static constexpr int HrT_idx = 3;
      static constexpr int mu_idx = 4;
      static constexpr int lambda_idx = 5;

      MFEM_HOST_DEVICE int property_index(int property, int density, int temperature) const
      {
        return property * nx * ny + temperature * nx + density;
      }
    };

    struct Tables
    {
      Layout L;
      LTETable::View tables{};
    };

    using Data = LTETable::Data;

    inline void Validate(const Data &data, const Layout &layout)
    {
      if (layout.nx < 2 || layout.ny < 2 ||
          data.rho_grid.Size() != layout.nx || data.T_grid.Size() != layout.ny ||
          data.e_grid.Size() != layout.ny ||
          data.lte_table.Size() != Layout::num_properties * layout.nx * layout.ny ||
          data.inv_table.Size() != layout.nx * layout.ny)
        {
          throw std::invalid_argument("Invalid six-plane Helmholtz table dimensions");
        }
      const mfem::Vector *grids[] = {&data.rho_grid, &data.T_grid, &data.e_grid};
      for (const auto *grid : grids)
        {
          const auto *values = grid->HostRead();
          for (int i = 0; i < grid->Size(); ++i)
            {
              if (!std::isfinite(values[i]) || (i > 0 && values[i] <= values[i - 1]))
                {
                  throw std::invalid_argument("Helmholtz grids must be finite and strictly increasing");
                }
            }
        }
      if (data.rho_grid.HostRead()[0] <= 0 || data.T_grid.HostRead()[0] <= 0)
        {
          throw std::invalid_argument("Helmholtz density and temperature must be positive");
        }
    }

    inline Tables HostView(const Data &data, const Layout &layout)
    {
      Validate(data, layout);
      return {layout, {data.lte_table.HostRead(), data.inv_table.HostRead(),
                       data.rho_grid.HostRead(), data.T_grid.HostRead(), data.e_grid.HostRead()}};
    }

    // Host cache construction only. The owner must retain storage throughout
    // kernel execution. Rebind after resizing; never capture a host-only view.
    inline Tables DeviceView(const Data &data, const Layout &layout)
    {
      Validate(data, layout);
      return {layout, {data.lte_table.Read(), data.inv_table.Read(),
                       data.rho_grid.Read(), data.T_grid.Read(), data.e_grid.Read()}};
    }

    MFEM_HOST_DEVICE inline int FindCell(const mfem::real_t *grid, int count,
                                       mfem::real_t value)
    {
      if (count < 2 || !std::isfinite(value) || value < grid[0] || value > grid[count - 1])
        {
          return -1;
        }
      int lower = 0;
      int upper = count - 1;
      while (upper - lower > 1)
        {
          const int middle = lower + (upper - lower) / 2;
          if (value < grid[middle])
            {
              upper = middle;
            }
          else
            {
              lower = middle;
            }
        }
      return lower;
    }
  }
}
