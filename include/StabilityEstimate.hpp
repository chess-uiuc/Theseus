// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include <algorithm>

namespace Theseus
{
  struct StabilityEstimate
  {
    mfem::real_t advective_rate = 0.0;
    mfem::real_t surface_rate = 0.0;
    mfem::real_t diffusive_rate = 0.0;

    mfem::real_t TotalRate() const
    {
      return std::max(advective_rate, surface_rate) + diffusive_rate;
    }

    mfem::real_t CFL(const mfem::real_t step) const
    {
      return step * TotalRate();
    }
  };

  // Spectral radii of the unit-speed, unit-cell, periodic 1-D upwind DGSEM
  // advection operator.  A five-percent margin covers the finite periodic
  // calibration grid and small implementation differences.  Higher orders use
  // a conservative continuation of the observed O((p+1)^2) envelope.
  inline mfem::real_t ReferenceAdvectionSpectralScale(const int order)
  {
    constexpr mfem::real_t calibrated[] = {
      0.0, 2.19706891727, 5.41995189335, 9.64849524786,
      14.7297419321, 20.5984751391, 27.2141622680, 34.5459214016,
      42.5691944324, 51.2637780299, 60.6126232210, 70.6010566044,
      81.2162499718
    };
    if (order > 0 && order < int(sizeof(calibrated) / sizeof(calibrated[0])))
      return mfem::real_t(1.05) * calibrated[order];
    const mfem::real_t points = std::max(1, order + 1);
    return mfem::real_t(0.65) * points * points;
  }

  // Spectral radii of G*G for the periodic, unit-cell scalar BR1 operator,
  // where G is the DGSEM auxiliary-gradient operator with central traces.
  // The margin is larger than for advection because the compressible viscous
  // operator couples primitive gradients through state-dependent transport and
  // because physical boundary closures need not have the periodic spectrum.
  inline mfem::real_t ReferenceBR1DiffusionSpectralScale(const int order)
  {
    constexpr mfem::real_t calibrated[] = {
      0.0, 4.0, 25.5969429388, 82.9000427145, 203.470468982,
      426.230394541, 799.733889878, 1383.0975821, 2244.38177282,
      3461.2550528, 5121.29136549, 7321.77632194, 10169.7148876
    };
    if (order > 0 && order < int(sizeof(calibrated) / sizeof(calibrated[0])))
      return mfem::real_t(1.25) * calibrated[order];
    const mfem::real_t points = std::max(1, order + 1);
    return mfem::real_t(0.5) * points * points * points * points;
  }
}
