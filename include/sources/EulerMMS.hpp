// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"
#include "../PeriodicEulerMMS.hpp"

namespace Theseus::Sources
{
  struct MMSModel : AnalyticModel
  {
    Requirements Needs() const { return {true}; }
    void Initialize(const MeshView &mesh)
    {
      if (mesh.dimension != 2 || mesh.equations != 4
          || mesh.layout.eq_mass != 0 || mesh.layout.eq_mom0 != 1 || mesh.layout.eq_energy != 3)
        {
          throw std::invalid_argument("This MMS requires two dimensions and four equations");
        }
    }
    struct Evaluator
    {
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &source) const
      {
        mfem::real_t state[4];
        mfem::real_t forcing[4];
        Theseus::Sources::PeriodicEulerMMS(point.Coordinate(0), point.Coordinate(1),
                                         point.stage.time, state, forcing);
        for (int equation = 0; equation < 4; ++equation)
          {
            source.Add(equation, forcing[equation]);
          }
      }
    };
    Evaluator MakeEvaluator(const Stage &) const { return {}; }
  };

  inline void RegisterMMS(Registry &registry)
  {
    registry.Register("periodic-euler-mms", [](const nlohmann::json &, const BuildContext &context)
    {
      if (context.dimension != 2 || context.equations != 4 || context.axisymmetric
          || context.runtime.value("gas_model", std::string("cpg")) != "cpg"
          || context.runtime.value("flow_model", std::string("euler")) != "euler"
          || context.runtime.value("gamma", 1.4) != 1.4)
        {
          throw std::invalid_argument("periodic-euler-mms requires Cartesian 2D CPG Euler with gamma=1.4");
        }
      return std::make_unique<PointSource<MMSModel>>(MMSModel{});
    });
  }
}
