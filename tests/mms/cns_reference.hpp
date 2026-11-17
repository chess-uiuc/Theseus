// Copyright (c) 2026 Board of Trustees of the University of Illinois
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>

namespace CNSReference
{
// Independent scalar specification, without production derivatives/flux/transport.
inline double
Primitive (int field, int dimension, const double *x, double t, double velocity_scale = 1.)
{
  const double mean[] = { 2., .7, .3, .2, 1. };
  const double ax[] = { .12, .07, -.05, .04, .08 };
  const double ay[] = { .08, -.04, .06, -.03, -.06 };
  const double az[] = { -.06, .03, -.04, .05, .04 };
  const double axy[] = { .04, .02, .03, -.02, .03 };
  const double axz[] = { .03, -.02, .01, .03, -.02 };
  const double ayz[] = { -.02, .01, -.02, .02, .02 };
  const double a = 2. * std::acos (-1.) * (x[0] - .2 * t) + .17 * field;
  const double b = 2. * std::acos (-1.) * (x[1] + .13 * t) + .29 * field;
  double value = mean[field] + ax[field] * std::sin (a) + ay[field] * std::cos (b)
                 + axy[field] * std::sin (a) * std::cos (b);
  if (dimension == 3)
    {
      const double c = 2. * std::acos (-1.) * (x[2] - .11 * t) + .31 * field;
      value += az[field] * std::sin (c) + axz[field] * std::cos (a) * std::sin (c)
               + ayz[field] * std::sin (b) * std::cos (c);
    }
  return value + (field >= 1 && field <= 3 ? (velocity_scale - 1.) * mean[field] : 0.);
}

constexpr int offsets[] = { -2, -1, 1, 2 };
constexpr double weights[] = { 1., -8., 8., -1. };
constexpr double step = 2.e-4;

inline double
Derivative (int field, int dim, const double *x, double t, int direction, double velocity_scale = 1.)
{
  double result = 0.;
  for (int sample = 0; sample < 4; ++sample)
    {
      double point[] = { x[0], x[1], x[2] };
      point[direction] += offsets[sample] * step;
      result += weights[sample] * Primitive (field, dim, point, t, velocity_scale) / (12. * step);
    }
  return result;
}

inline void
StateOrFlux (int dim, const double *x, double t, int direction, double mu, double conductivity,
             double R, double *value, bool sutherland = false, double velocity_scale = 1.)
{
  const double rho = Primitive (0, dim, x, t, velocity_scale);
  const double p = Primitive (4, dim, x, t, velocity_scale);
  if (sutherland)
    {
      const auto temperature = p / (rho * R);
      mu = mu * std::pow (temperature / 273.15, 1.5) * (273.15 + 110.4) / (temperature + 110.4);
      conductivity = mu * 3.5 * R / .72;
    }
  double u[3] = {}, gradient[3][3] = {}, divergence = 0.;
  double energy = p / .4;
  for (int i = 0; i < dim; ++i)
    {
      u[i] = Primitive (i + 1, dim, x, t, velocity_scale);
      energy += .5 * rho * u[i] * u[i];
      for (int j = 0; j < dim; ++j)
        {
          gradient[i][j] = Derivative (i + 1, dim, x, t, j, velocity_scale);
        }
      divergence += gradient[i][i];
    }
  if (direction == dim)
    {
      value[0] = rho;
      value[dim + 1] = energy;
      for (int i = 0; i < dim; ++i)
        {
          value[i + 1] = rho * u[i];
        }
      return;
    }
  const double temperature_gradient = (Derivative (4, dim, x, t, direction, velocity_scale) / rho
                                       - p * Derivative (0, dim, x, t, direction, velocity_scale) / (rho * rho))
                                      / R;
  value[0] = rho * u[direction];
  value[dim + 1] = (energy + p) * u[direction] - conductivity * temperature_gradient;
  for (int i = 0; i < dim; ++i)
    {
      const double stress = mu
                            * (gradient[i][direction] + gradient[direction][i]
                               - (i == direction ? (2. / 3.) * divergence : 0.));
      value[i + 1] = rho * u[i] * u[direction] + (i == direction ? p : 0.) - stress;
      value[dim + 1] -= u[i] * stress;
    }
}

inline void
Residual (int dim, const double *x, double t, double mu, double kappa, double R, double scale,
          double *source, bool sutherland = false, double velocity_scale = 1.)
{
  for (int q = 0; q < dim + 2; ++q)
    {
      source[q] = 0.;
    }
  for (int d = 0; d <= dim; ++d)
    {
      for (int sample = 0; sample < 4; ++sample)
        {
          double point[] = { x[0], x[1], x[2] };
          double time = t;
          if (d == dim)
            {
              time += offsets[sample] * step / scale;
            }
          else
            {
              point[d] += offsets[sample] * step;
            }
          double flux[5];
          StateOrFlux (dim, point, time * scale, d, mu, kappa, R, flux, sutherland, velocity_scale);
          for (int q = 0; q < dim + 2; ++q)
            {
              source[q] += weights[sample] * flux[q] / (12. * step) * (d == dim ? scale : 1.);
            }
        }
    }
}

}
