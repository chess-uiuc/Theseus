// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "SourceRegistry.hpp"
#include "verification/mms/sources/Regime.hpp"
#include "verification/mms/TrigonometricEulerMMS.hpp"

namespace Theseus::Sources
{
  struct TrigonometricEulerModel : AnalyticModel
  {
    int orientation = 0;
    mfem::real_t time_scale = 1.0;
    mfem::real_t mean_velocity_scale = 1.0;
    explicit TrigonometricEulerModel(int value = 0, mfem::real_t scale = 1.0, mfem::real_t velocity_scale = 1.0)
      : orientation(value), time_scale(scale), mean_velocity_scale(velocity_scale) {}
    Requirements Needs() const { return {true}; }

    void Initialize(const MeshView &mesh)
    {
      if ((mesh.dimension != 2 && mesh.dimension != 3)
          || mesh.equations != mesh.dimension + 2 || mesh.layout.eq_mass != 0
          || mesh.layout.eq_mom0 != 1 || mesh.layout.eq_energy != mesh.dimension + 1)
        {
          throw std::invalid_argument("Trigonometric Euler MMS requires 2D/3D conservative layout");
        }
    }

    struct Evaluator
    {
      int orientation;
      mfem::real_t time_scale, mean_velocity_scale;
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &source) const
      {
        mfem::real_t position[3] = {};
        for (int d = 0; d < point.mesh.dimension; ++d)
          {
            position[d] = point.Coordinate(d);
          }
        mfem::real_t state[5];
        mfem::real_t forcing[5];
        OrientedTrigonometricEulerMMS(point.mesh.dimension, orientation, position,
                                     point.stage.time, state, forcing, time_scale, mean_velocity_scale);
        for (int equation = 0; equation < point.mesh.equations; ++equation)
          {
            source.Add(equation, forcing[equation]);
          }
      }
    };

    Evaluator MakeEvaluator(const Stage &) const { return {orientation, time_scale, mean_velocity_scale}; }
  };

  inline void RegisterTrigonometricEuler(Registry &registry)
  {
    registry.Register("trigonometric-euler-mms", [](const nlohmann::json &configuration, const BuildContext &context)
    {
      if ((context.dimension != 2 && context.dimension != 3)
          || context.equations != context.dimension + 2 || context.axisymmetric
          || context.runtime.value("gas_model", std::string("cpg")) != "cpg"
          || context.runtime.value("flow_model", std::string("euler")) != "euler"
          || context.runtime.value("gamma", 1.4) != 1.4
          || context.runtime.value("R_gas", 287.058) != 1.0)
        {
          throw std::invalid_argument(
            "trigonometric-euler-mms requires Cartesian 2D/3D CPG Euler, gamma=1.4 and R_gas=1");
        }
      if (configuration.contains("orientation") && !configuration.at("orientation").is_number_integer())
        {
          throw std::invalid_argument("MMS orientation must be an integer axis permutation");
        }
      const int orientation = configuration.value("orientation", 0);
      if (orientation < 0 || orientation >= context.dimension)
        {
          throw std::invalid_argument("MMS orientation must be between zero and dimension-1");
        }
      const mfem::real_t time_scale = configuration.value("time_scale", 1.0);
      if (!std::isfinite(time_scale) || time_scale <= 0.0)
        {
          throw std::invalid_argument("MMS time_scale must be positive and finite");
        }
      return std::make_unique<PointSource<TrigonometricEulerModel>>(
        TrigonometricEulerModel{orientation, time_scale, ManufacturedMeanVelocityScale(configuration)});
    });
  }
}
