// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "SourceRegistry.hpp"
#include <cmath>

namespace Theseus::Sources
{
  struct AffineModel : AnalyticModel
  {
    struct Coefficients
    {
      mfem::real_t constant[MAXEQ] = {};
      mfem::real_t time[MAXEQ] = {};
      mfem::real_t state[MAXEQ] = {};
      mfem::real_t gradient[MAXDIM][MAXEQ] = {};
    } coefficients;
    bool spatial = false;
    int equations;
    int dimension;

    AffineModel(const nlohmann::json &configuration, const BuildContext &context)
      : equations(context.equations), dimension(context.dimension)
    {
      if (dimension < 1 || dimension > MAXDIM || equations < dimension + 2 || equations > MAXEQ)
        {
          throw std::invalid_argument("Unsupported affine source layout");
        }
      auto read = [this](const nlohmann::json &values, mfem::real_t *output)
      {
        if (!values.is_array() || int(values.size()) != equations)
          {
            throw std::invalid_argument("Source vectors require one value per conservative equation");
          }
        for (int q = 0; q < equations; ++q)
          {
            output[q] = values.at(q).get<mfem::real_t>();
            if (!std::isfinite(output[q]))
              {
                throw std::invalid_argument("Source coefficients must be finite");
              }
          }
      };
      if (configuration.contains("constant")) { read(configuration.at("constant"), coefficients.constant); }
      if (configuration.contains("time")) { read(configuration.at("time"), coefficients.time); }
      if (configuration.contains("state")) { read(configuration.at("state"), coefficients.state); }
      if (configuration.contains("gradient"))
        {
          const auto &gradient = configuration.at("gradient");
          if (!gradient.is_array() || int(gradient.size()) != dimension)
            {
              throw std::invalid_argument("Source gradient requires one vector per dimension");
            }
          for (int d = 0; d < dimension; ++d) { read(gradient.at(d), coefficients.gradient[d]); }
          spatial = true;
        }
    }
    Requirements Needs() const { return {spatial}; }
    void Initialize(const MeshView &mesh)
    {
      if (mesh.dimension != dimension || mesh.equations != equations)
        {
          throw std::invalid_argument("Affine source mesh/configuration mismatch");
        }
    }
    struct Evaluator
    {
      Coefficients coefficients;
      bool spatial;
      MFEM_HOST_DEVICE void operator()(const Point &point, const SourceAccumulator &source) const
      {
        for (int q = 0; q < point.mesh.equations; ++q)
          {
            auto source_term = coefficients.constant[q] + point.stage.time*coefficients.time[q]
              + coefficients.state[q]*point.State(q);
            if (spatial)
              {
                for (int d = 0; d < point.mesh.dimension; ++d)
                  {
                    source_term += coefficients.gradient[d][q]*point.Coordinate(d);
                  }
              }
            source.Add(q, source_term);
          }
      }
    };
    Evaluator MakeEvaluator(const Stage &) const { return {coefficients, spatial}; }
  };

  inline void RegisterAffine(Registry &registry)
  {
    registry.Register("affine", [](const nlohmann::json &configuration, const BuildContext &context)
    {
      return std::make_unique<PointSource<AffineModel>>(AffineModel(configuration, context));
    });
  }
}
