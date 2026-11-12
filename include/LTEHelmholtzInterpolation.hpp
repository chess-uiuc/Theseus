// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"

namespace Theseus
{
  namespace HelmholtzInterpolation
  {
    // Borrow the four property planes at the lower corner of one table cell.
    // Density is contiguous; temperature_stride is the table's density count.
    // The owner must retain the planes for the complete evaluation. Cell widths
    // are positive and selected coordinates are normalized to this same cell.
    struct Cell
    {
      const mfem::real_t *potential;
      const mfem::real_t *density_derivative;
      const mfem::real_t *temperature_derivative;
      const mfem::real_t *mixed_derivative;
      int temperature_stride;
      mfem::real_t density_width;
      mfem::real_t temperature_width;
    };

    namespace detail
    {
      template<int Derivative>
      MFEM_HOST_DEVICE inline mfem::real_t PositionBasis(int side, mfem::real_t x)
      {
        static_assert(Derivative >= 0 && Derivative <= 2,
                      "Hermite derivative order must be zero, one or two");
        const mfem::real_t x2 = x * x;
        if constexpr (Derivative == 0)
          {
            const mfem::real_t x3 = x2 * x;
            return side == 0 ? 2 * x3 - 3 * x2 + 1 : -2 * x3 + 3 * x2;
          }
        else if constexpr (Derivative == 1)
          {
            return side == 0 ? 6 * x2 - 6 * x : -6 * x2 + 6 * x;
          }
        else
          {
            return side == 0 ? 12 * x - 6 : -12 * x + 6;
          }
      }

      template<int Derivative>
      MFEM_HOST_DEVICE inline mfem::real_t SlopeBasis(int side, mfem::real_t x)
      {
        static_assert(Derivative >= 0 && Derivative <= 2,
                      "Hermite derivative order must be zero, one or two");
        const mfem::real_t x2 = x * x;
        if constexpr (Derivative == 0)
          {
            const mfem::real_t x3 = x2 * x;
            return side == 0 ? x3 - 2 * x2 + x : x3 - x2;
          }
        else if constexpr (Derivative == 1)
          {
            return side == 0 ? 3 * x2 - 4 * x + 1 : 3 * x2 - 2 * x;
          }
        else
          {
            return side == 0 ? 6 * x - 4 : 6 * x - 2;
          }
      }

      // Exact divided differences of the cubic basis, including coincident
      // coordinates. No subtraction of nearly equal basis values is needed.
      template<int Derivative>
      MFEM_HOST_DEVICE inline mfem::real_t DividedPositionBasis(int side,
          mfem::real_t x, mfem::real_t y)
      {
        static_assert(Derivative >= 0 && Derivative <= 2);
        mfem::real_t value;
        if constexpr (Derivative == 0)
          {
            value = 2*(x*x + x*y + y*y) - 3*(x + y);
          }
        else if constexpr (Derivative == 1)
          {
            value = 6*(x + y) - 6;
          }
        else
          {
            value = 12;
          }
        return side == 0 ? value : -value;
      }

      template<int Derivative>
      MFEM_HOST_DEVICE inline mfem::real_t DividedSlopeBasis(int side,
          mfem::real_t x, mfem::real_t y)
      {
        static_assert(Derivative >= 0 && Derivative <= 2);
        if constexpr (Derivative == 0)
          {
            return x*x + x*y + y*y - (side == 0 ? 2 : 1)*(x + y)
                   + (side == 0 ? 1 : 0);
          }
        else if constexpr (Derivative == 1)
          {
            return 3*(x + y) - (side == 0 ? 4 : 2);
          }
        else
          {
            return 6;
          }
      }

      template<int Derivative>
      MFEM_HOST_DEVICE inline mfem::real_t CoordinateScale(mfem::real_t width)
      {
        if constexpr (Derivative == 0)
          {
            return 1;
          }
        else if constexpr (Derivative == 1)
          {
            return width;
          }
        else
          {
            return width * width;
          }
      }
    }

    // Evaluate the donor's bicubic Hermite interpolant, or one of its physical
    // coordinate derivatives. Each corner is read and consumed directly: no
    // automatic 4x4 coefficient block, allocation, or nested callback is needed.
    // Reordering corner contributions changes rounding, not the polynomial.
    template<int DensityDerivative = 0, int TemperatureDerivative = 0>
    MFEM_HOST_DEVICE inline mfem::real_t Evaluate(const Cell &cell,
                                                 mfem::real_t xi,
                                                 mfem::real_t eta)
    {
      static_assert(DensityDerivative >= 0 && DensityDerivative <= 2,
                    "Density derivative order must be zero, one or two");
      static_assert(TemperatureDerivative >= 0 && TemperatureDerivative <= 2,
                    "Temperature derivative order must be zero, one or two");
      mfem::real_t result = 0;
      for (int density_side = 0; density_side < 2; ++density_side)
        {
          const mfem::real_t density_value =
            detail::PositionBasis<DensityDerivative>(density_side, xi);
          const mfem::real_t density_slope =
            detail::SlopeBasis<DensityDerivative>(density_side, xi);
          for (int temperature_side = 0; temperature_side < 2; ++temperature_side)
            {
              const int offset = density_side + temperature_side * cell.temperature_stride;
              const mfem::real_t temperature_value =
                detail::PositionBasis<TemperatureDerivative>(temperature_side, eta);
              const mfem::real_t temperature_slope =
                detail::SlopeBasis<TemperatureDerivative>(temperature_side, eta);

              result += cell.potential[offset] * density_value * temperature_value;
              result += (cell.density_derivative[offset] * cell.density_width)
                        * density_slope * temperature_value;
              result += (cell.temperature_derivative[offset] * cell.temperature_width)
                        * density_value * temperature_slope;
              result += (cell.mixed_derivative[offset] * cell.density_width
                         * cell.temperature_width) * density_slope * temperature_slope;
            }
        }
      const mfem::real_t scale =
        detail::CoordinateScale<DensityDerivative>(cell.density_width)
        * detail::CoordinateScale<TemperatureDerivative>(cell.temperature_width);
      return result / scale;
    }

    // Physical divided difference of a requested derivative within ONE cell.
    // Axis0 varies density (eta2 ignored); Axis1 varies T (xi2 ignored).
    // This is the same polynomial secant, evaluated before large corner sums.
    template<int Axis, int DensityDerivative = 0, int TemperatureDerivative = 0>
    MFEM_HOST_DEVICE inline mfem::real_t DividedDifference(const Cell &cell,
        mfem::real_t xi1, mfem::real_t eta1, mfem::real_t xi2, mfem::real_t eta2)
    {
      static_assert(Axis == 0 || Axis == 1);
      static_assert(DensityDerivative >= 0 && DensityDerivative <= 2);
      static_assert(TemperatureDerivative >= 0 && TemperatureDerivative <= 2);
      mfem::real_t result = 0;
      for (int i = 0; i < 2; ++i)
        {
          const auto rv = Axis == 0
            ? detail::DividedPositionBasis<DensityDerivative>(i, xi1, xi2)
            : detail::PositionBasis<DensityDerivative>(i, xi1);
          const auto rs = Axis == 0
            ? detail::DividedSlopeBasis<DensityDerivative>(i, xi1, xi2)
            : detail::SlopeBasis<DensityDerivative>(i, xi1);
          for (int j = 0; j < 2; ++j)
            {
              const auto tv = Axis == 1
                ? detail::DividedPositionBasis<TemperatureDerivative>(j, eta1, eta2)
                : detail::PositionBasis<TemperatureDerivative>(j, eta1);
              const auto ts = Axis == 1
                ? detail::DividedSlopeBasis<TemperatureDerivative>(j, eta1, eta2)
                : detail::SlopeBasis<TemperatureDerivative>(j, eta1);
              const int offset = i + j*cell.temperature_stride;
              result += cell.potential[offset]*rv*tv;
              result += (cell.density_derivative[offset]*cell.density_width)*rs*tv;
              result += (cell.temperature_derivative[offset]*cell.temperature_width)*rv*ts;
              result += (cell.mixed_derivative[offset]*cell.density_width*cell.temperature_width)*rs*ts;
            }
        }
      const auto scale = detail::CoordinateScale<DensityDerivative>(cell.density_width)
                         * detail::CoordinateScale<TemperatureDerivative>(cell.temperature_width)
                         * (Axis == 0 ? cell.density_width : cell.temperature_width);
      return result/scale;
    }
  }
}
