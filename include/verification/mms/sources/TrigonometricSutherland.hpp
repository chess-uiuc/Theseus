// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"
#include "verification/mms/sources/Regime.hpp"
#include "verification/mms/TrigonometricSutherlandMMS.hpp"

namespace Theseus::Sources
{
struct TrigonometricSutherlandModel : AnalyticModel
{
  mfem::real_t reference_viscosity, prandtl, gas_constant, time_scale, coefficient_gradient_scale;
  mfem::real_t mean_velocity_scale = 1.0;
  Requirements
  Needs () const
  {
    return { true };
  }
  void
  Initialize (const MeshView &mesh)
  {
    if ((mesh.dimension != 2 && mesh.dimension != 3) || mesh.equations != mesh.dimension + 2
        || mesh.layout.eq_mass != 0 || mesh.layout.eq_mom0 != 1
        || mesh.layout.eq_energy != mesh.dimension + 1)
      {
        throw std::invalid_argument (
            "Trigonometric Sutherland MMS requires 2D/3D conservative layout");
      }
  }
  struct Evaluator
  {
    mfem::real_t reference_viscosity, prandtl, gas_constant, time_scale, coefficient_gradient_scale;
    mfem::real_t mean_velocity_scale = 1.0;
    MFEM_HOST_DEVICE void
    operator() (const Point &point, const SourceAccumulator &source) const
    {
      mfem::real_t position[3] = {};
      for (int d = 0; d < point.mesh.dimension; ++d)
        {
          position[d] = point.Coordinate (d);
        }
      mfem::real_t state[5], forcing[5];
      TrigonometricSutherlandMMS (point.mesh.dimension, position, point.stage.time,
                                  reference_viscosity, prandtl, gas_constant, state, forcing,
                                  time_scale, coefficient_gradient_scale, mean_velocity_scale);
      for (int q = 0; q < point.mesh.equations; ++q)
        {
          source.Add (q, forcing[q]);
        }
    }
  };
  Evaluator
  MakeEvaluator (const Stage &) const
  {
    return { reference_viscosity, prandtl, gas_constant, time_scale, coefficient_gradient_scale, mean_velocity_scale };
  }
};

inline void
RegisterTrigonometricSutherland (Registry &registry)
{
  registry.Register (
      "trigonometric-cns-sutherland-mms",
      [] (const nlohmann::json &configuration, const BuildContext &context)
        {
#ifndef SUTHERLAND
          throw std::invalid_argument (
              "trigonometric-cns-sutherland-mms requires a Sutherland build");
#endif
          const auto &runtime = context.runtime;
          const auto mu0 = runtime.value ("mu0", 1.716e-5);
          const auto R = runtime.value ("R_gas", 287.05);
          const auto Pr = runtime.value ("Pr", .72);
          const auto scale = configuration.value ("time_scale", 1.0);
          const auto viscous_scale = configuration.value ("viscous_scale", 1.0);
          const auto gradient_scale = configuration.value ("coefficient_gradient_scale", 1.0);
          if ((context.dimension != 2 && context.dimension != 3) || context.axisymmetric
              || context.equations != context.dimension + 2
              || runtime.value ("gas_model", std::string ("cpg")) != "cpg"
              || runtime.value ("flow_model", std::string ()) != "navier-stokes"
              || runtime.value ("gamma", 1.4) != 1.4 || !std::isfinite (R) || R <= 0.
              || !std::isfinite (Pr) || Pr <= 0. || !std::isfinite (mu0) || mu0 < 0.
              || !std::isfinite (scale) || scale <= 0. || !std::isfinite (viscous_scale)
              || viscous_scale < 0. || !std::isfinite (gradient_scale) || gradient_scale < 0.)
            {
              throw std::invalid_argument (
                  "trigonometric-cns-sutherland-mms requires Cartesian CPG "
                  "navier-stokes, gamma=1.4, positive R/Pr/time_scale and nonnegative mu0/scales");
            }
          const auto source_mu0 = mu0 * viscous_scale;
          if (!std::isfinite (source_mu0) || !std::isfinite (source_mu0 * 3.5 * R / Pr))
            {
              throw std::invalid_argument ("trigonometric-cns-sutherland-mms transport overflow");
            }
          auto model = TrigonometricSutherlandModel{};
          model.reference_viscosity = source_mu0;
          model.prandtl = Pr;
          model.gas_constant = R;
          model.time_scale = scale;
          model.mean_velocity_scale = ManufacturedMeanVelocityScale(configuration);
          model.coefficient_gradient_scale = gradient_scale;
          return std::make_unique<PointSource<TrigonometricSutherlandModel>> (model);
        });
}
}
