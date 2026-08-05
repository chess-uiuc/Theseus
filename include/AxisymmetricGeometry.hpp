// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace Theseus
{
  /// Coordinate and cached-radius contract for swirl-free (z,r) geometry.
  struct AxisymmetricGeometry
  {
    static constexpr int axial_coordinate = 0;
    static constexpr int radial_coordinate = 1;
    static constexpr int spatial_dimension = 2;
    static constexpr mfem::real_t radius_tolerance = 1.0e-12;
    static constexpr mfem::real_t two_pi =
      6.283185307179586476925286766559005768;

#ifdef AXISYMMETRIC
    static constexpr bool enabled = true;
#else
    static constexpr bool enabled = false;
#endif

    MFEM_HOST_DEVICE static mfem::real_t Radius(const mfem::real_t *physical)
    {
      return physical[radial_coordinate];
    }

    MFEM_HOST_DEVICE static mfem::real_t MeasureMultiplier(
      bool axisymmetric, mfem::real_t radius)
    {
      return axisymmetric ? two_pi*radius : 1.0;
    }

    static mfem::real_t ValidateRadius(mfem::real_t radius,
                                      const std::string &location)
    {
      if (!std::isfinite(radius) || radius < -radius_tolerance)
        {
          throw std::invalid_argument(
            "axisymmetric geometry has an invalid radial coordinate at " +
            location + ": r = " + std::to_string(radius));
        }
      return radius < 0.0 ? 0.0 : radius;
    }
  };
}
