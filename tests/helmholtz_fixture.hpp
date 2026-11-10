// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "LTEHelmholtzTable.hpp"
namespace TheseusTest
{
  namespace HT = Theseus::HelmholtzTable;
  inline HT::Tables AnalyticHelmholtz(HT::Data &data)
  {
    Theseus::LTETable::uniform_grid(3, 0.5, 2, data.rho_grid);
    Theseus::LTETable::uniform_grid(3, 2, 5, data.T_grid);
    Theseus::LTETable::uniform_grid(3, 0, 32, data.e_grid);
    data.inv_table.SetSize(9);
    data.inv_table = 5.0;
    data.lte_table.SetSize(54);
    const HT::Layout layout{3, 3};
    for (int j = 0; j < 3; ++j)
      {
        const auto t = data.T_grid[j];
        for (int i = 0; i < 3; ++i)
          {
            const auto r = data.rho_grid[i];
            data.lte_table[layout.property_index(0, i, j)] = 2*r + r*r/2 + r*t/4 - t*t;
            data.lte_table[layout.property_index(1, i, j)] = 2 + r + t/4;
            data.lte_table[layout.property_index(2, i, j)] = r/4 - 2*t;
            data.lte_table[layout.property_index(3, i, j)] = 0.25;
            data.lte_table[layout.property_index(4, i, j)] = 2 + r + t;
            data.lte_table[layout.property_index(5, i, j)] = 4 + 2*r + t;
          }
      }
    return HT::HostView(data, layout);
  }

}
