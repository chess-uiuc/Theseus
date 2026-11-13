// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"
#include <cmath>

namespace Theseus::Sources
{
  // Independent physical example: linear momentum drag and its local work.
  struct DragModel : AnalyticModel
  {
    mfem::real_t coefficient;
    explicit DragModel(mfem::real_t value) : coefficient(value)
    {
      if (!std::isfinite(value) || value < 0.0)
        {
          throw std::invalid_argument("Drag coefficient must be finite and nonnegative");
        }
    }
    struct Evaluator
    {
      mfem::real_t coefficient;
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &source) const
      {
        const auto &layout = point.mesh.layout;
        const auto rho = point.State(layout.eq_mass);
        mfem::real_t energy_source = 0.0;
        for (int direction = 0; direction < point.mesh.dimension; ++direction)
          {
            const auto momentum = point.State(layout.eq_mom0 + direction);
            const auto momentum_source = -coefficient*momentum;
            source.Add(layout.eq_mom0 + direction, momentum_source);
            energy_source += momentum_source*momentum/rho;
          }
        source.Add(layout.eq_energy, energy_source);
      }
    };
    Evaluator MakeEvaluator(const Stage &) const { return {coefficient}; }
  };

  inline void RegisterDrag(Registry &registry)
  {
    registry.Register("linear-drag", [](const nlohmann::json &configuration, const BuildContext &)
    {
      return std::make_unique<PointSource<DragModel>>(
        DragModel(configuration.at("coefficient").get<mfem::real_t>()));
    });
  }
}
