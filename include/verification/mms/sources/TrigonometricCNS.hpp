// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"
#include "verification/mms/sources/Regime.hpp"
#include "verification/mms/TrigonometricCNSMMS.hpp"

namespace Theseus::Sources
{
struct TrigonometricCNSModel : AnalyticModel
{
  mfem::real_t viscosity, conductivity, gas_constant, time_scale;
  mfem::real_t mean_velocity_scale = 1.0;
  TrigonometricCNSModel (mfem::real_t mu, mfem::real_t kappa, mfem::real_t R, mfem::real_t scale, mfem::real_t velocity_scale = 1.0)
      : viscosity (mu), conductivity (kappa), gas_constant (R), time_scale (scale), mean_velocity_scale (velocity_scale)
  {
  }
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
        throw std::invalid_argument ("Trigonometric CNS MMS requires 2D/3D conservative layout");
      }
  }
  struct Evaluator
  {
    mfem::real_t viscosity, conductivity, gas_constant, time_scale;
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
      TrigonometricCNSMMS (point.mesh.dimension, position, point.stage.time, viscosity,
                           conductivity, gas_constant, state, forcing, time_scale, mean_velocity_scale);
      for (int q = 0; q < point.mesh.equations; ++q)
        {
          source.Add (q, forcing[q]);
        }
    }
  };
  Evaluator
  MakeEvaluator (const Stage &) const
  {
    return { viscosity, conductivity, gas_constant, time_scale, mean_velocity_scale };
  }
};

inline void
RegisterTrigonometricCNS (Registry &registry)
{
  registry.Register ("trigonometric-cns-mms",
                     [] (const nlohmann::json &configuration, const BuildContext &context)
                       {
#ifdef SUTHERLAND
                         throw std::invalid_argument (
                             "trigonometric-cns-mms currently requires a constant-transport build");
#endif
                         const auto &runtime = context.runtime;
                         const auto mu = runtime.value ("mu", .02);
                         const auto R = runtime.value ("R_gas", 287.05);
                         const auto Pr = runtime.value ("Pr", .72);
                         const auto scale = configuration.value ("time_scale", 1.0);
                         const auto viscous_scale = configuration.value ("viscous_scale", 1.0);
                         if ((context.dimension != 2 && context.dimension != 3)
                             || context.axisymmetric || context.equations != context.dimension + 2
                             || runtime.value ("gas_model", std::string ("cpg")) != "cpg"
                             || runtime.value ("flow_model", std::string ()) != "navier-stokes"
                             || runtime.value ("gamma", 1.4) != 1.4 || !std::isfinite (R) || R <= 0.
                             || !std::isfinite (mu) || mu < 0. || !std::isfinite (Pr) || Pr <= 0.
                             || !std::isfinite (scale) || scale <= 0.
                             || !std::isfinite (viscous_scale) || viscous_scale < 0.)
                           {
                             throw std::invalid_argument (
                                 "trigonometric-cns-mms requires Cartesian CPG navier-stokes, "
                                 "gamma=1.4, positive R/Pr/scale and nonnegative mu");
                           }
                         const auto source_mu = mu * viscous_scale;
                         const auto conductivity = source_mu * 3.5 * R / Pr;
                         if (!std::isfinite (source_mu) || !std::isfinite (conductivity))
                           {
                             throw std::invalid_argument (
                                 "trigonometric-cns-mms conductivity overflow");
                           }
                         return std::make_unique<PointSource<TrigonometricCNSModel>> (
                             TrigonometricCNSModel{ source_mu, conductivity, R, scale, ManufacturedMeanVelocityScale(configuration) });
                       });
}
}
