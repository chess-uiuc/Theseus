// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "AxisymmetricGeometry.hpp"
#include "json.hpp"
#include "mfem.hpp"

#include <stdexcept>
#include <string>

namespace Theseus
{
  /// Foundational state and coordinate contract for swirl-free axisymmetry.
  struct AxisymmetryConfig
  {
    static constexpr int axial_coordinate = AxisymmetricGeometry::axial_coordinate;
    static constexpr int radial_coordinate = AxisymmetricGeometry::radial_coordinate;
    static constexpr int spatial_dimension = AxisymmetricGeometry::spatial_dimension;
    static constexpr int conservative_equations = 4;

    static void Validate(const nlohmann::json &config,
                         const nlohmann::json &runtime,
                         bool axisymmetric_build)
    {
      if (config.contains("compileTime") &&
          config["compileTime"].contains("AXISYMMETRIC"))
        {
          const bool requested =
            config["compileTime"]["AXISYMMETRIC"].get<bool>();
          if (requested != axisymmetric_build)
            {
              throw std::invalid_argument(
                std::string("compileTime.AXISYMMETRIC requests ") +
                (requested ? "an axisymmetric" : "a Cartesian") +
                " executable, but this executable was built " +
                (axisymmetric_build ? "with" : "without") +
                " AXISYMMETRIC");
            }
        }

      if (!axisymmetric_build)
        {
          return;
        }

      const int dim = runtime.value("dim", spatial_dimension);
      if (dim != spatial_dimension)
        {
          throw std::invalid_argument(
            "axisymmetric simulations require runTime.dim = 2");
        }

      const int num_equations =
        runtime.value("num_equations", conservative_equations);
      if (num_equations != conservative_equations)
        {
          throw std::invalid_argument(
            "swirl-free axisymmetric simulations require "
            "runTime.num_equations = 4");
        }
    }

    static void ValidateMesh(const mfem::Mesh &mesh)
    {
      if (mesh.Dimension() != spatial_dimension ||
          mesh.SpaceDimension() != spatial_dimension)
        {
          throw std::invalid_argument(
            "axisymmetric meshes must have topological and spatial dimension 2");
        }

      for (int vertex = 0; vertex < mesh.GetNV(); ++vertex)
        {
          const mfem::real_t radius = mesh.GetVertex(vertex)[radial_coordinate];
          if (!std::isfinite(radius) ||
              radius < -AxisymmetricGeometry::radius_tolerance)
            {
              throw std::invalid_argument(
                "axisymmetric mesh has a negative radial coordinate at vertex " +
                std::to_string(vertex) + ": r = " + std::to_string(radius));
            }
        }
    }
  };
}
